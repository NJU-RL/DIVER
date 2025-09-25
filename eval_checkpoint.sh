export VLLM_ATTENTION_BACKEND=XFORMERS
# export MODEL_PATH="/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/baseline_clp_02_028/actor/global_step_300"
export MODEL_PATH="checkpoints/div/rs_equ_100step/actor/global_step_350"
# export MODEL_PATH="checkpoints/div/entropy_mechanism_clip_cov/actor/global_step_350"
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/passk_training"
# export DATA_PATH="/mnt/petrelfs/share_data/huzican/dataset/eval.ood.passn.parquet"
export DATA_PATH=["dataset/mmlu_shuffle/mmlu_pro_0.parquet"]
srun --job-name test -p ai_moe -c 40 -w SH-IDC1-10-140-37-118 \
 sh scripts/eval/eval_model.sh --model $MODEL_PATH --dataset $DATA_PATH