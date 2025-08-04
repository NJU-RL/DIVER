export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/baseline_/actor/global_step_300"
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/cl_all/actor/global_step_400"
# export MODEL_PATH="/mnt/petrelfs/share_data/zhangshilin/constrastive_clp_0_3_t_1/actor/global_step_300"
export DATA_PATH="valid.mmlu_pro"
srun --job-name test -p ai_moe -c 64 -w SH-IDC1-10-140-37-82 \
scripts/eval/eval_model.sh --model $MODEL_PATH --dataset $DATA_PATH