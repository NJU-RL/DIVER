# export SLURM_JOB_ID=5079183
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-Math-7B-16k-think"
srun --job-name test -p ai_moe -c 100 -w SH-IDC1-10-140-37-49 \
scripts/train/debug_div.sh --model $MODEL_PATH
# --export=ALL,CUDA_VISIBLE_DEVICES=3,4,5,7 scripts/train/test_kl.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH