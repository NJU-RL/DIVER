# export SLURM_JOB_ID=5079183
export VLLM_ATTENTION_BACKEND=XFORMERS
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-Math-7B-16k-think"
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-7B-orz-tok"
# export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Llama-3.1-8B-Instruct-ds"
# export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-Math-1.5B-ds-16k"
# export MODEL_PATH="/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/baseline_clp_02_028/actor/global_step_300"
# export MODEL_PATH="/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/rs_equ_100step/actor/global_step_350"
# export MODEL_PATH="/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/entropy_mechanism_clip_cov/actor/global_step_350"
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen-2.5-Math-7B-SimpleRL-Zoo"
export MODEL_PATH="checkpoints/div/base_equ_correct_100step/actor/global_step_250"
# srun -p ai_moe -c 64 -w SH-IDC1-10-140-37-29 \
srun -p ai_moe -c 64 -w SH-IDC1-10-140-37-118 \
 scripts/train/test_div.sh --model $MODEL_PATH
# --export=ALL,CUDA_VISIBLE_DEVICES=3,4,5,7 scripts/train/test_kl.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH