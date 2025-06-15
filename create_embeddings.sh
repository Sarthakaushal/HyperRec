#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=18
export NCCL_P2P_DISABLE=1

# Run the Python script with the specified arguments
python gneric_vlm_embedding.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --output_file data/All_Books_qwen_embeddings_ordered.npy \
    --data_name Books\
    --image /home/spring2024/sk4858/HyperRec/data/All_Books \
    --metadata_file /home/spring2024/sk4858/HyperRec/data/metadata_Books.json