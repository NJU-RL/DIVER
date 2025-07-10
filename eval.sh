# export SLURM_JOB_ID=5079183
export VLLM_ATTENTION_BACKEND=XFORMERS
srun -p ai_moe -c 64 -w SH-IDC1-10-140-37-6 \
 bash evaluate.sh
