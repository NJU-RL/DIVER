export SLURM_JOB_ID=4941453
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-7B-orz-tok"
export RM_MODEL_PATH="/mnt/petrelfs/share_data/huzican/DeepSeek-R1-Distill-Qwen-7B"
srun -p MoE -w SH-IDCA1404-10-140-54-5 \
scripts/train/rskd.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH