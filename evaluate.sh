ROOT=.
DATA=$ROOT/dataset/valid.arc_c.parquet

OUTPUT_DIR=./results/
mkdir -p $OUTPUT_DIR

# If you want to evaluate other models, you can change the model path and name.
MODEL_PATH=/mnt/petrelfs/zhangshilin/rlvr_div/checkpoints/div/filter_high_equ_div/actor/global_step_400
MODEL_NAME=high_equ_div

if [ $MODEL_NAME == "eurus-2-7b-prime-zero" ]; then
  TEMPLATE=prime
elif [ $MODEL_NAME == "simple-rl-zero" ]; then
  TEMPLATE=qwen
else
  TEMPLATE=own
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_scripts/generate_vllm.py \
  --model_path $MODEL_PATH \
  --input_file $DATA \
  --remove_system True \
  --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
  --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log 

# CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_scripts/generate_vllm.py \
#   --model_path $MODEL_PATH \
#   --input_file $DATA \
#   --remove_system True \
#   --add_oat_evaluate True \
#   --output_file $OUTPUT_DIR/$MODEL_NAME.jsonl \
#   --template $TEMPLATE > $OUTPUT_DIR/$MODEL_NAME.log 