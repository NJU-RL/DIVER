set -x
export HF_ENDPOINT=https://hf-mirror.com

eval "$(conda shell.bash hook)"
conda activate r1-kd-v1

export PATH=/mnt/petrelfs/share/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-12.1/lib64:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/lib64:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export PATH=/mnt/petrelfs/share/gcc/gcc-11.2.0/bin:$PATH
# libmpfr.so.6
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpfr-4.1.0/lib:$LD_LIBRARY_PATH
# libmpc.so.2
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpc-0.8.1/lib:$LD_LIBRARY_PATH
# libmpfr.so.1
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/mpfr-2.4.2/lib/:$LD_LIBRARY_PATH
# libgmp.so.3
export LD_LIBRARY_PATH=/mnt/petrelfs/share/gcc/gmp-4.3.2/lib/:$LD_LIBRARY_PATH

WANDB_KEY=55cf0dbab72178987b6a6e17c443a7b0c36cb8cd

ROOT=/mnt/petrelfs/yanjianhao/RL/rl4kd-ds/OpenRLHF

# export MODEL_PATH=/mnt/petrelfs/yanjianhao/hf_models/Qwen2.5-7B-orz-tok

# ray stop 

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
# Run 8K context length training

# export MODEL_PATH=$SFT_MODEL
export MODEL_PATH=/mnt/petrelfs/share_data/yanjianhao/Qwen2.5-Math-7B-ds-16k
# export 
# DATA_DIR=/mnt/petrelfs/yanjianhao/RL/r1_kd/deepscaler/open-r1-data
export EXP_NAME=qwen-7b-math-8k-openr1-v9-sft-3epo

cd $ROOT/OpenRLHF

RESULT_DIR=$ROOT/results/sft/$EXP_NAME
mkdir -p $RESULT_DIR

DEVICES="0,1,2,3,4,5,6,7"
# unset CUDA_VISIBLE_DEVICES

MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))
echo $MASTER_ADDR
echo $MASTER_PORT
DATA_DIR=/mnt/petrelfs/yanjianhao/RL/rl4kd-ds/deepscaler/openr1-v9_sft/train.jsonl
WANDB_KEY=55cf0dbab72178987b6a6e17c443a7b0c36cb8cd

deepspeed --master_port=$MASTER_PORT --master_addr=$MASTER_ADDR --include localhost:$DEVICES --module openrlhf.cli.train_sft \
   --max_len 16384 \
   --dataset $DATA_DIR \
   --input_key prompt \
   --output_key target \
   --train_batch_size 64 \
   --apply_chat_template \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain $MODEL_PATH \
   --save_path $RESULT_DIR \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 2 \
   --max_epochs 3 \
   --adam_offload \
   --packing_samples \
   --bf16 \
   --flash_attn \
   --save_hf_ckpt \
   --learning_rate 5e-5 \
   --lr_warmup_ratio 0.1 \
   --wandb_project r1_kd \
   --wandb_run_name qwen-7b-base-8k-openr1-v9-sft \
   --use_wandb $WANDB_KEY \
   --gradient_checkpointing

   #--lr_warmup_ratio 0.1 \
   # --save_steps 1000 \
#    --input_template $'User: {}\nAssistant: ' \
# origin lr 5e-6
