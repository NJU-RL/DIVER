# export SLURM_JOB_ID=5124674
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/filter_low_equ_div/actor/global_step_400"
srun -p ai_moe -c 64 -w SH-IDC1-10-140-37-25 \
 scripts/train/debug.sh --model $MODEL_PATH
# --export=ALL,CUDA_VISIBLE_DEVICES=3,4,5,7 scripts/train/test_kl.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH