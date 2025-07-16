export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-Math-7B-16k-think"
srun --job-name test -p ai_moe -c 120 -w SH-IDC1-10-140-37-35 \
scripts/train/test_div.sh --model $MODEL_PATH