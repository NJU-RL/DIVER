# export SLURM_JOB_ID=5445043
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-Math-7B-16k-think"
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-7B-orz-tok"
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Llama-3.1-8B-Instruct"
# export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-Math-1.5B-ds-16k"


srun --job-name test -c 110 -p ai_moe -w SH-IDC1-10-140-37-86 \
scripts/train/test_div.sh --model $MODEL_PATH


