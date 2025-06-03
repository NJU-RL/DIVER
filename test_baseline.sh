
# export SLURM_JOB_ID=5079183
export VLLM_ATTENTION_BACKEND=XFORMERS
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-7B-orz-tok"
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/DeepSeek-R1-Distill-Qwen-7B"
srun --job-name test -c 110 -p MoE -w SH-IDCA1404-10-140-54-79 \
scripts/train/baseline.sh --model $MODEL_PATH