# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
from turtle import position
from typing import Iterable, Tuple
from urllib import response

from cv2 import accumulate
from numpy import indices, positive
from regex import F
import torch
from torch import logit, nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers import rollout
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: 
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        use_div = True
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                if use_div:
                    output = self.actor_module(input_ids=input_ids_rmpad,
                                               attention_mask=None,
                                               position_ids=position_ids_rmpad,
                                               use_cache=False,
                                               output_hidden_states=True)
                    # print(f'len: {len(output.hidden_states)}')
                    # print(f'hidden state shape: {output.hidden_states[-1].shape}')
                    hidden_states_rmpad = output.hidden_states[-1].squeeze(0) # (total_nnz, hidden_dim)
                    

                else:
                    output = self.actor_module(input_ids=input_ids_rmpad,
                                               attention_mask=None,
                                               position_ids=position_ids_rmpad,
                                               use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                    if use_div:
                        full_hidden_states = gather_outpus_and_unpad(hidden_states=hidden_states_rmpad,
                                                                     gather_dim=0,
                                                                     unpad_dim=0,
                                                                     padding_size=pad_size)

                # pad back to (bsz, seqlen)
                if use_div:
                    full_hidden_states = pad_input(hidden_states=hidden_states_rmpad,
                                                indices=indices,
                                                batch=batch_size,
                                                seqlen=seqlen)
                    last_indices = torch.clamp(attention_mask[:,-response_length:].sum(dim=1)-2, min=0) // 5
                    batch_indices = torch.arange(batch_size, device=full_hidden_states.device)
                    last_hidden_states = full_hidden_states[:,-response_length:][batch_indices, last_indices] # (bsz, hidden_dim)
                    
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)
                
                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                # print(f"last_hidden_states:{last_hidden_states}")

            else:  # not using rmpad and no ulysses sp
                if use_div:
                    output = self.actor_module(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               position_ids=position_ids,
                                               output_hidden_states=True,
                                               use_cache=False)
                    hidden_states = output.hidden_states[-1].squeeze(0)
                    last_indices = torch.clamp(attention_mask[:,-response_length:].sum(dim=1)-2, min=0) // 5
                    batch_indices = torch.arange(batch_size, device=hidden_states.device)
                    last_hidden_states = hidden_states[:,-response_length:][batch_indices, last_indices]
                else:
                    output = self.actor_module(input_ids=input_ids,
                                               attention_mask=attention_mask,
                                               position_ids=position_ids,
                                               use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
            if use_div:
                return entropy, log_probs, last_hidden_states
            else:
                return entropy, log_probs, None

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        # print(batch)

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        hidden_states_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs, hidden_states = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            hidden_states_lst.append(hidden_states)
        log_probs = torch.concat(log_probs_lst, dim=0)
        hidden_states = torch.concat(hidden_states_lst, dim=0)

        # print(f'log probs: {log_probs.shape}')

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            hidden_states = hidden_states[revert_indices]
        # print(f'log probs: {log_probs.shape}')
        # print(f'hidden states: {hidden_states.shape}')

        return log_probs, hidden_states
    
    def compute_group_disp_loss(self, hidden_states: torch.Tensor, rollout_n: int, tau: float = 1.0):
        """
        计算分组的多样性损失
        
        Args:
            hidden_states: 形状为(batch_size, hidden_dim)的张量
            rollout_n: 每组的样本数量
            tau: 温度参数
            
        Returns:
            所有组内多样性损失的总和
        """
        batch_size = hidden_states.shape[0]
        
        # 确保batch_size可以被rollout_n整除
        if batch_size % rollout_n != 0:
            raise ValueError(f"Batch size ({batch_size}) must be divisible by rollout_n ({rollout_n})")
        
        num_groups = batch_size // rollout_n
        total_loss = 0.0
        
        for i in range(num_groups):
            # 提取当前组的hidden_states
            start_idx = i * rollout_n
            end_idx = (i + 1) * rollout_n
            group_hidden_states = hidden_states[start_idx:end_idx]
            
            # 对每个样本的hidden_states进行平均池化
            Z = group_hidden_states.mean(dim=1)  # (rollout_n, hidden_dim)
            
            # 归一化
            Z = nn.functional.normalize(Z, p=2, dim=1)
            
            # 计算相似度矩阵
            sim_matrix = torch.mm(Z, Z.t())
            
            # 计算距离矩阵
            D_matrix = 2 * (1 - sim_matrix)
            
            # 只取上三角部分（不包括对角线）
            D_matrix = D_matrix.triu(diagonal=1)
            D = D_matrix[D_matrix > 0]
            
            # 如果组内只有一个样本，则没有可比较的对，跳过损失计算
            if D.numel() == 0:
                continue
            
            # 计算当前组的损失
            group_loss = torch.log(torch.mean(torch.exp(-D/tau) + 1e-8))
            
            # 累加到总损失
            total_loss += group_loss
        
        return total_loss
    
    def disp_loss(self, old_hidden_states: torch.Tensor, hidden_states: torch.Tensor, cl_mask, group_mask, tau: float = 1., epsilon=1e-8):
        '''
        old hidden states: [bsz, rollout_n, hidden_dim]
        hidden stats: [bsz, 1, hidden_dim]
        '''
        bsz, n, _ = hidden_states.shape
        device = hidden_states.device

        valid_indices = cl_mask.bool()
        if not valid_indices.any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        old_hidden_states = old_hidden_states[valid_indices]  # [valid_bsz, rollout_n, hidden_dim]
        hidden_states = hidden_states[valid_indices]  # [valid_bsz, 1, hidden_dim]
        group_mask = group_mask[valid_indices]  # [valid_bsz, rollout_n]

        group_mask = group_mask.unsqueeze(-1).bool()
        old_hidden_states = torch.where(group_mask, old_hidden_states, torch.zeros_like(old_hidden_states))

        # old_hidden_states = old_hidden_states[:, 1:, :]
        # hidden_states = hidden_states[:, 1:, :]

        old_hidden_states = old_hidden_states[:, 1:, :]  # [bsz, rollout_n - 1, hidden_dim]
        Z_old = torch.nn.functional.normalize(old_hidden_states, dim=2)
        Z = torch.nn.functional.normalize(hidden_states, dim=2)
        
        sim_matrix = torch.bmm(Z_old, Z.transpose(1, 2)) # [bsz, rollout_n - 1, 1]
        sim_matrix.squeeze(dim=2)
        normed_sim_matrix = torch.nn.functional.normalize(sim_matrix, dim=1)

        mask = (normed_sim_matrix != 0)
        masked_normed_sim_matrix = normed_sim_matrix.clone()
        masked_normed_sim_matrix[~mask] = float('-inf')

        # print(f'masked: {masked_normed_sim_matrix}')

        D_matrix = 1 - normed_sim_matrix
        # print(f'D matrix: {D_matrix}')
        
        total_loss = 0
        for i in range(D_matrix.shape[0]):
            D = D_matrix[i]
            # print(f'D: {D}')
            total_loss += torch.log(torch.mean(torch.exp(-D/tau) + epsilon))
        
        # print(f'total loss: {total_loss}')
        
        return total_loss, sim_matrix.mean()
    
    def contrastive_loss(self, hidden_states, old_hidden_states, labels):
        """
        Args: 
            hidden_states: [batch_size, hidden_dim]
            old_hidden_states: [batch_size, rollout_n, hidden_dim]
            label_pos: [batch_size]
            
        # """
        # valid_indices = cl_mask.bool()
        # if not valid_indices.any():
        #     return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        # # filter invalid sampling
        # hidden_states = hidden_states[valid_indices]
        # old_hidden_states = old_hidden_states[valid_indices]
        # label_pos = label_pos[valid_indices]

        bsz, _ = hidden_states.shape

        hidden_states = nn.functional.normalize(hidden_states, p=2, dim=1)
        old_hidden_states = nn.functional.normalize(old_hidden_states, p=2, dim=2)
        
        logits = torch.bmm(hidden_states.unsqueeze(dim=1), old_hidden_states.transpose(1,2)) #(bsz, 1, rollout_n)

        print(f'logits: {logits}')
        print(f'labels: {labels}')
        labels = torch.zeros(bsz, dtype=torch.long, device=logits.device)
        loss = nn.functional.cross_entropy(logits, labels)
        # loss = self.cross_entropy_loss(logits.squeeze(1), label_pos.to(logits.device))
        # print(f"cl_loss:{loss}, cl_mask:{cl_mask}")
        
        return loss
    
    def _compute_group_contrastive_loss(self, old_hidden_states: torch.Tensor, hidden_states: torch.Tensor, cl_mask, group_mask, temperature=0.5):
        """
        Compute logits for constrastive loss within a group.

        Args:
            old_hidden_states: #(bsz, rollout_n, hidden_dim)
            hidden_states: #(bsz, 1, hidden_dim)
            cl_mask: #(bsz)
            group_mask: #(bsz, rollout_n)

        Return:
            loss
        """

        valid_indices = cl_mask.bool()
        if not valid_indices.any():
            return torch.tensor(0.0, device=old_hidden_states.device), torch.tensor(0.0, device=old_hidden_states.device)

        old_hidden_states = old_hidden_states[valid_indices]  # [valid_bsz, rollout_n, hidden_dim]
        hidden_states = hidden_states[valid_indices]  # [valid_bsz, 1, hidden_dim]
        group_mask = group_mask[valid_indices]  # [valid_bsz, rollout_n]

        group_mask = group_mask.unsqueeze(-1).bool()  # [valid_bsz, rollout_n, 1]
        # print(f'group mask: {group_mask}')
        old_hidden_states = torch.where(group_mask, old_hidden_states, torch.zeros_like(old_hidden_states))


        old_hidden_states_norm = torch.nn.functional.normalize(old_hidden_states, p=2, dim=2)
        hidden_states_norm = torch.nn.functional.normalize(hidden_states, p=2, dim=2)

        logits = torch.bmm(old_hidden_states_norm, hidden_states_norm.transpose(1, 2))  # [bsz, rollout_n, 1]

        logits = torch.nn.functional.normalize(logits, p=2, dim=1)
 
        logits = logits.squeeze(-1) / temperature  # [bsz, rollout_n]
        labels = torch.zeros(old_hidden_states.shape[0], dtype=torch.long, device=logits.device)

        mask = (logits != 0)
        masked_logits = logits.clone()
        masked_logits[~mask] = float('-inf')

        # print(f'masked logits: {masked_logits}')

        loss = nn.functional.cross_entropy(masked_logits, labels)

        return loss, logits.mean().detach()


    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        # print("****config: ", self.config.ppo_mini_b
        # atch_size, self.config.ppo_micro_batch_size)

        assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size == 0
        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'old_hidden_states', 'token_level_rewards', 'group_rewards']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
            
        batch = data.select(batch_keys=select_keys).batch

        
        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, indices = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    # split batch into micro_batches
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size)

                self.actor_optimizer.zero_grad()
                
                # hidden_states_lst = []
                # # responses_lst = []

                # accumulated_policy_loss = 0

                for data in micro_batches:
                    print("MICROBATCH STEP")
                    data = data.cuda()  # actor device is cpu when using offload
                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']
                    old_hidden_states = data['old_hidden_states']
                    cl_mask = 1 - data['token_level_rewards'].sum(dim=1)
                    group_mask = 1 - data['group_rewards']
                    # labels = data['labels']

                    # from transformers import AutoModelForCausalLM, AutoTokenizer
                    # model_path = "/mnt/petrelfs/share_data/huzican/Qwen2.5-7B-orz-tok"
                    # tokenizer = AutoTokenizer.from_pretrained(model_path)

                    # print('======== micro batch ========')
                    # print(f'len: {len(data["input_ids"])}')
        
                    # for i in range(len(data['input_ids'])):
                    #     # 找到第一个非掩码位置（值为1的位置）
                    #     attention_mask = data['attention_mask'][i]
                    #     non_masked_indices = [j for j, mask in enumerate(attention_mask) if mask == 1]
                        
                    #     if non_masked_indices:
                    #         start_idx = non_masked_indices[0]  # 第一个非掩码的位置
                    #         # 从非掩码位置截取最多200个token
                    #         input_ids_slice = data['input_ids'][i][start_idx+50:start_idx+200]
                    #         input_seq = tokenizer.decode(input_ids_slice, skip_special_tokens=True)
                    #         output_text = f"从非掩码位置开始的前200个token: {input_seq}"
                    #     else:
                    #         # 如果全部被掩码，则输出原始序列的前200个token
                    #         input_seq = tokenizer.decode(data['input_ids'][i][:200], skip_special_tokens=True)
                    #         output_text = f"序列全部被掩码 显示前200个token: {input_seq}"
                        
                    #     responses_lst.append(output_text)
                    

                
                    clip_ratio = self.config.clip_ratio
                    clip_ratio_high = self.config.clip_ratio_high
                    entropy_coeff = self.config.entropy_coeff
                    # contrastive_coeff = self.config.contrastive_coeff
                    loss_mode = self.config.loss_mode

                    entropy, log_prob, hidden_states = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                    # constrastive_loss = self.contrastive_loss(hidden_states, old_hidden_states, labels)
                    hidden_states = hidden_states.unsqueeze(dim=1)

                    if loss_mode == "vanilla":
                        pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                                    log_prob=log_prob,
                                                                                    advantages=advantages,
                                                                                    eos_mask=response_mask,
                                                                                    cliprange=clip_ratio,
                                                                                    cliprangehigh=clip_ratio_high)
                    elif loss_mode == "clip_cov":
                        loss_agg_mode="token-mean"
                        pg_loss, pg_clipfrac, ppo_kl= core_algos.compute_policy_loss_clip_cov(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            cliprange=clip_ratio,
                            cliprange_low=clip_ratio,
                            cliprange_high=clip_ratio_high,
                            loss_agg_mode=loss_agg_mode,
                            clip_ratio=0.0002,
                            clip_cov_lb=1.0,
                            clip_cov_ub=5.0,
                        )

                    elif loss_mode == "kl_cov":
                        loss_agg_mode="token-mean"
                        pg_loss, pg_clipfrac, ppo_kl= core_algos.compute_policy_loss_kl_cov(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            k_percent=0.2,
                            ppo_kl_coef=1,
                        )

                    else:
                        raise ValueError(f"Unsupported loss mode: {self.config.loss_mode}")
                    # print(f'pg loss shape: {pg_loss.shape}')
                    # print(f'pg clip shape: {pg_clipfrac.shape}')
                    # compute entropy loss from entropy
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                    # compute policy loss
                    # contrastive_loss.detach()
                    # print(f'contrastive loss: {contrastive_loss}')
                    policy_loss = pg_loss - entropy_loss * entropy_coeff 
                    if self.config.use_kl_loss:
                        ref_log_prob = data['ref_log_prob']
                        # compute kl loss
                        kld = core_algos.kl_penalty(logprob=log_prob,
                                                    ref_logprob=ref_log_prob,
                                                    kl_penalty=self.config.kl_loss_type)
                        kl_loss = masked_mean(kld, response_mask)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics['actor/kl_loss'] = kl_loss.detach().item()
                        metrics['actor/kl_coef'] = self.config.kl_loss_coef

                    loss = policy_loss / self.gradient_accumulation
                    loss.backward()
                    # accumulated_policy_loss += policy_loss / self.gradient_accumulation

                    data = {
                        'actor/entropy_loss': entropy_loss.detach().item(),
                        'actor/pg_loss': pg_loss.detach().item(),
                        'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                        'actor/ppo_kl': ppo_kl.detach().item(),
                    }
                    append_to_dict(metrics, data)

                # hidden_states = torch.concat(hidden_states_lst, dim=0)
                
                # if self.config.use_dynamic_bsz:
                #     indices = list(itertools.chain.from_iterable(indices))
                #     assert len(indices) == hidden_states.size(0), f"{len(indices)} vs. {hidden_states.size()}"
                #     revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                #     hidden_states = hidden_states[revert_indices]


                # print(f'hidden_states: {hidden_states.shape}')

                # disper_loss = self.compute_group_disp_loss(hidden_states, rollout_n=8)

                # print(f'disper loss: {disper_loss}')
                # print(f'disper loss grad: {disper_loss.requires_grad}')

                # total_loss = accumulated_policy_loss + 0.01 * disper_loss
                # total_loss.backward()

                # metrics['actor/disper_loss'] = disper_loss.detach().item()

                grad_norm = self._optimizer_step()
                # print(f'grad norm: {grad_norm.shape}')
                data = {'actor/grad_norm': grad_norm.detach().item()}
                append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
