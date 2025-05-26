#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS


# export TMP=/mnt/petrelfs/huzican/R1/RLKD_RM/tmp
# export TMPDIR=/mnt/petrelfs/huzican/R1/RLKD_RM/tmp
# export TEMPDIR=/mnt/petrelfs/huzican/R1/RLKD_RM/tmp
# mkdir -p $TMP

ray stop

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --rm_model)
            RM_MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
fi

if [ -z "$RM_MODEL_PATH" ]; then
    RM_MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
fi

# export RAY_REDIS_ADDRESS=127.0.0.1:6389

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.rm_src.rm_main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.gamma=0.999 \
    data.train_files=dataset/train.parquet \
    data.val_files=dataset/test.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    reward_model.enable=True \
    reward_model.use_dynamic_bsz=True\
    reward_model.model.use_remove_padding=True \
    reward_model.micro_batch_size=4\
    reward_model.model.path=$RM_MODEL_PATH \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=4\
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.n_val=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='rskd' \
    trainer.experiment_name='simpleRL8K_qwen7b_base_8k_rskd' \
    +trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=30 "${@:1}" 


# while true; do
#     sleep 5
#     ray debug
# done