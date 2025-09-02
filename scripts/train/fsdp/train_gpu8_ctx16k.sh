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
export KD_LOSS_TOP_K=$7

export WANDB_DISABLED=true  

unset NCCL_ASYNC_ERROR_HANDLING
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
cd "$(dirname "$0")/../../../train"

accelerate launch --config_file config/fsdp2_8gpu.yaml \
train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_max_length $MAX_LENGTH \
    --output_dir $SAVE_PATH \
    --logging_dir $LOG_PATH \
    --num_train_epochs 1 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "steps" \
    --eval_steps 50 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 300 \
    --learning_rate 1e-6 \
    --warmup_steps 100 \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 64 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --kd_loss_top_k $KD_LOSS_TOP_K \
    --ranking_type "dcg_pair_logistic" \
    --ranking_R 32 \
    --ranking_beta 10000 \
    --max_train_samples 999999 \
    --clip $CLIP_PATH \
    --use_flash_attn \
    --may_resume

