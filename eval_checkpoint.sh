export VLLM_ATTENTION_BACKEND=XFORMERS
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/baseline_/actor/global_step_300"
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/cl_all/actor/global_step_400"
# export MODEL_PATH="/mnt/petrelfs/share_data/zhangshilin/entropy_clip_cov"
# export MODEL_PATH="/mnt/petrelfs/share_data/zhangshilin/rs_equ_100step"
# export MODEL_PATH="/mnt/petrelfs/share_data/zhangshilin/baseline_clp_02_028_300"
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/baseline_passk_training/actor/global_step_350"
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/belu0.01_correct_clip0.2_0.28/actor/global_step_350"
export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/qwen-math1.5B_belu0.01/actor/global_step_200"
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/qwen_math1.5B_baseline_clip0.2_0.28/actor/global_step_200"
export DATA_PATH="eval.ood.passn"
# export DATA_PATH="eval.passn"
# export DATA_PATH="valid.arc_c"
# export DATA_PATH="valid.gpqa"


srun --job-name test -p ai_moe -c 64 -w SH-IDC1-10-140-37-118 \
scripts/eval/eval_model.sh --model $MODEL_PATH --dataset $DATA_PATH