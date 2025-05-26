
export SLURM_JOB_ID=4878187
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-Math-7B"
srun -p MoE -w SH-IDCA1404-10-140-54-65 \
scripts/train/simpleRL_7b_reinforce_8k.sh --model $MODEL_PATH