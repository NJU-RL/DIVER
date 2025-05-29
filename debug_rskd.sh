export SLURM_JOB_ID=5064670
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-7B-orz-tok"
export RM_MODEL_PATH="/mnt/petrelfs/huzican/R1/RLKD/checkpoints/deepscaler/deepscaler40K_ds7b_rein_pp_gamma0_9999/actor/global_step_480"
srun -p MoE -w SH-IDCA1404-10-140-54-86 \
scripts/train/debug_rskd.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH