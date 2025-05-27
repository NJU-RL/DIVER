
export SLURM_JOB_ID=5064670
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-7B-orz-tok"
srun -p MoE -w SH-IDCA1404-10-140-54-86 \
scripts/train/baseline.sh --model $MODEL_PATH