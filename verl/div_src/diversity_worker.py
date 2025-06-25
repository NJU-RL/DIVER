import logging
import os
import warnings

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.import_utils import import_external_libs
from sentence_transformers import SentenceTransformer
import numpy as np
import heapq

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        raise ValueError(
            'HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1')
        assert world_size % fsdp_size == 0
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class DiversityWorker(Worker):
    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)


    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForCausalLM, AutoConfig
        
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        if self.config.model.input_tokenizer:
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))
        
        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        # model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
        diversity_module = SentenceTransformer(model_name_or_path=local_path).to(torch.cuda.current_device())
        # auto_wrap_policy = get_fsdp_wrap_policy(module=diversity_module, config=self.config.model.fsdp_config)

        # fsdp_mesh = self.device_mesh
        # sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # diversity_module = FSDP(
        #     diversity_module,
        #     param_init_fn=init_fn,
        #     use_orig_params=False,
        #     auto_wrap_policy=auto_wrap_policy,
        #     device_id=torch.cuda.current_device(),
        #     sharding_strategy=sharding_strategy,  # zero3
        #     sync_module_states=True,
        #     cpu_offload=CPUOffload(offload_params=True),
        #     forward_prefetch=False,
        #     device_mesh=self.device_mesh)

        return diversity_module
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.diversity_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()
    