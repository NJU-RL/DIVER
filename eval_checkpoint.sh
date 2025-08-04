export VLLM_ATTENTION_BACKEND=XFORMERS
# export MODEL_PATH="/mnt/petrelfs/huzican/R1/rlvr_div/checkpoints/div/baseline_/actor/global_step_300"
export MODEL_PATH="/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/constrastive_clp_0_3_t_1/actor/global_step_250"
export DATA_PATH="/mnt/petrelfs/share_data/huzican/dataset/eval.passn.parquet"
srun --job-name test -p ai_moe -c 64 -w SH-IDC1-10-140-37-42 \
sh scripts/eval/eval_model.sh --model $MODEL_PATH --dataset $DATA_PATH