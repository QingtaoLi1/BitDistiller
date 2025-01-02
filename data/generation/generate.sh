MODEL_DIR=$1
DATASET=$2
OUTPUT=$3
BATCH_SIZE=$4
MAX_SAMPLE=$5

CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node 4 --master_port 7830 generate.py \
                        --base_model $MODEL_DIR \
                        --dataset_name $DATASET \
                        --out_path $OUTPUT \
                        --batch_size $BATCH_SIZE \
                        --max_sample $MAX_SAMPLE \
                        --max_new_tokens 2048

# Single Generate
# CUDA_VISIBLE_DEVICES=0 /root/model/miniconda3/envs/qat/bin/torchrun single_generate.py \
#                         --base_model $MODEL_DIR \
#                         --dataset_name $DATASET \
#                         --out_path $OUTPUT \
#                         --batch_size $BATCH_SIZE \
#                         --max_sample $MAX_SAMPLE   
