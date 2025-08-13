# export SLURM_JOB_ID=5222192
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-Math-7B-16k-think"
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/group_only_hidden_1/5/actor/global_step_100"
srun --job-name test -p ai_moe -c 90 -w SH-IDC1-10-140-37-82 \
scripts/train/test_div.sh --model $MODEL_PATH