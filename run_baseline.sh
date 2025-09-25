# export SLURM_JOB_ID=5079183
export VLLM_ATTENTION_BACKEND=XFORMERS
# export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Llama-3.1-8B-Instruct-ds"
# export MODEL_PATH="/mnt/petrelfs/share_data/huzican/Qwen2.5-7B-orz-tok"
export MODEL_PATH="/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-Math-1.5B-ds-16k"
srun -p ai_moe -c 88 -w SH-IDC1-10-140-37-14 \
 scripts/train/baseline_qwen_base.sh --model $MODEL_PATH
# --export=ALL,CUDA_VISIBLE_DEVICES=3,4,5,7 scripts/train/test_kl.sh --model $MODEL_PATH --rm_model $RM_MODEL_PATH