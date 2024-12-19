export MODEL_PATH='/home/superbench/qingtaoli/models/Llama-2-7b-hf/'
export SAVE_PATH='/mnt/sdb1/qingtaoli/Llama-2-7b-hf-bitdistiller-gpu4/'
export MASTER_ADDR="localhost"
export MASTER_PORT="1321"
export GLOO_SOCKET_IFNAME="lo"
export NCCL_SOCKET_IFNAME="lo"
export WANDB_DISABLED=true  

deepspeed --include localhost:4,5,6,7 train.py \
    --model_name_or_path $MODEL_PATH \
    --data_path '/home/superbench/qingtaoli/data-llama-2-7b/wikitext-2-generated/mix_wiki_alpaca_8000.json' \
    --model_max_length 2048 \
    --output_dir $SAVE_PATH \
    --logging_dir '/home/superbench/qingtaoli/models/Llama-2-7b-hf-log/' \
    --num_train_epochs 4 \
    --bf16 True \
    --seed 42 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --evaluation_strategy "steps" \
    --eval_steps 25 \
    --load_best_model_at_end True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 15 \
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
    --clip /home/superbench/qingtaoli/BitDistiller/quantization/clip_cache/Llama-2-7b-hf/int2-g64.pt 
