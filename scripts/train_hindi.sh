#!/bin/bash

# LLaMA-Omni Hindi Training Script
# Minimal setup for training with Whisper + LLaMA 3.2-3B

# Set Python path
export PYTHONPATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try:$PYTHONPATH"

# Set paths
export LLAMA_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/llama-3.2-3b"
export WHISPER_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/whisper-large-v3"
export DATA_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train.json"
export OUTPUT_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/output/hindi_omni_v1"

# Create output directory
mkdir -p $OUTPUT_DIR

# Number of GPUs to use (adjust based on your setup)
NUM_GPUS=4

echo "=============================================="
echo "LLaMA-Omni Hindi Training"
echo "=============================================="
echo "LLaMA Model: $LLAMA_PATH"
echo "Whisper Model: $WHISPER_PATH"
echo "Training Data: $DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "=============================================="

# Training command with DeepSpeed ZeRO-2
torchrun --nproc_per_node=$NUM_GPUS --master_port=29500 \
    omni_speech/train/train.py \
    --model_name_or_path $LLAMA_PATH \
    --speech_encoder $WHISPER_PATH \
    --speech_encoder_type "whisper" \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --s2s False \
    --unit_vocab_size 5100 \
    --ctc_decoder_config "(2,3072,32,8192)" \
    --ctc_upsample_factor 25 \
    --ctc_loss_weight 1.0 \
    --speech_projector_type "linear" \
    --speech_encoder_ds_rate 5 \
    --speech_encoder_hidden_size 1280 \
    --input_type "mel" \
    --mel_size 128 \
    --has_tgt_units True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 500 \
    --save_total_limit 3 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to "tensorboard"
