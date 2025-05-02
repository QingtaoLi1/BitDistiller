## An CLI argument example:
# export MODEL_PATH='/home/superbench/qingtaoli/models/Llama-2-7b-hf/'
# export SAVE_PATH='/mnt/sdb1/qingtaoli/Llama-2-7b-bitdistiller/'
# export DATA_PATH='/mnt/sdb1/qingtaoli/data-llama-2-7b/wikitext-2-generated/mix_wiki_alpaca_8000.json'
# export LOG_PATH='/home/superbench/qingtaoli/models/Llama-2-7b-log/'
# export CLIP_PATH='/home/superbench/qingtaoli/BitDistiller/quantization/clip_cache/Llama-2-7b-hf/int2-g64.pt'

export MODEL_PATH=$1
export SAVE_PATH=$2
export DATA_PATH=$3
export LOG_PATH=$4
export CLIP_PATH=$5
export MAX_LENGTH=$6

export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

# CUDA_VISIBLE_DEVICES=0 python \
deepspeed --hostfile=hostfile --include localhost:0,1,2,3,4,5,6,7 \
train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_max_length $MAX_LENGTH \
    --output_dir $SAVE_PATH \
    --logging_dir $LOG_PATH \
    --num_train_epochs 10 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing True \
    --eval_strategy "steps" \
    --eval_steps 400 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 300 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 64 \
    --train_kd False \
    --kd_loss_type "none" \
    --max_train_samples 999999 \
    --clip $CLIP_PATH
