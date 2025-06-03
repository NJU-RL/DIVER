# export SLURM_JOB_ID=5079183
export VLLM_ATTENTION_BACKEND=XFORMERS
export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-7B-orz-tok"
# export RM_MODEL_PATH="/mnt/petrelfs/huzican/R1/RLKD/checkpoints/deepscaler/deepscaler40K_ds7b_rein_pp_gamma0_9999/actor/global_step_480"
export RM_MODEL_PATH="/mnt/petrelfs/huzican/R1/rlkd_rm_v1/checkpoints/reward_shaping/teacher/actor/global_step_160/hf"
srun --job-name test_kl -p MoE -c 120 -w SH-IDCA1404-10-140-54-2 \
scripts/train/test_kl.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH
# --export=ALL,CUDA_VISIBLE_DEVICES=3,4,5,7 scripts/train/test_kl.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH