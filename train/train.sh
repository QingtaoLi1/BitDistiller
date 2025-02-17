export MODEL_PATH='/home/superbench/qingtaoli/models/Phi-3.5-mini-instruct/'
export SAVE_PATH='/mnt/sdb1/qingtaoli/Phi-3.5-mini-instruct-bitdistiller-wiki3kalpaca5k_ctx20482/'
export DATA_PATH='/mnt/sdb1/qingtaoli/data-phi-3.5/wikitext-2-generated/mix_wiki_alpaca_8000.json'
export LOG_PATH='/home/superbench/qingtaoli/models/Phi-3.5-mini-instruct-wiki3kalpaca5k_ctx2048-log2/'
export CLIP_PATH='/mnt/sdb1/qingtaoli/clip_cache/Phi-3.5-mini-instruct/int2-g64.pt'

# export MODEL_PATH='/home/superbench/qingtaoli/models/Phi-3-mini-4k-instruct/'
# export SAVE_PATH='/mnt/sdb1/qingtaoli/Phi-3-mini-4k-instruct-bitdistiller-wiki3kalpaca5k_ctx2048/'
# export DATA_PATH='/mnt/sdb1/qingtaoli/data-phi-3-ctx2048/wikitext-2-generated/mix_wiki_alpaca_8000.json'
# export LOG_PATH='/home/superbench/qingtaoli/models/Phi-3-mini-4k-instruct-wiki3kalpaca5k_ctx2048-log/'
# export CLIP_PATH='/mnt/sdb1/qingtaoli/clip_cache/Phi-3-mini-4k-instruct/int2-g64.pt'

# export MODEL_PATH='/home/superbench/qingtaoli/models/Llama-2-7b-hf/'
# export SAVE_PATH='/mnt/sdb1/qingtaoli/Llama-2-7b-bitdistiller/'
# export DATA_PATH='/mnt/sdb1/qingtaoli/data-llama-2-7b/wikitext-2-generated/mix_wiki_alpaca_8000.json'
# export LOG_PATH='/home/superbench/qingtaoli/models/Llama-2-7b-log/'
# export CLIP_PATH='/home/superbench/qingtaoli/BitDistiller/quantization/clip_cache/Llama-2-7b-hf/int2-g64.pt'

export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

# deepspeed --include localhost:0 
CUDA_VISIBLE_DEVICES=0 python train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path $DATA_PATH \
    --model_max_length 2048 \
    --output_dir $SAVE_PATH \
    --logging_dir $LOG_PATH \
    --num_train_epochs 10 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 800 \
    --save_total_limit 30 \
    --learning_rate 8e-6 \
    --lr_scheduler_type "constant" \
    --weight_decay 0. \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --deepspeed config/zero.json \
    --bits 2 \
    --quant_type int2-asym \
    --q_group_size 64 \
    --train_kd True \
    --kd_loss_type "cakld" \
    --max_train_samples 999999 \
    --clip $CLIP_PATH
