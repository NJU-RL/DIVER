export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/baseline_/actor/global_step_300"
srun --job-name test -p ai_moe -c 64 -w SH-IDC1-10-140-37-23 \
scripts/eval/eval_model.sh --model $MODEL_PATH 