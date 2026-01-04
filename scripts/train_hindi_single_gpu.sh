#!/bin/bash

# Single GPU training script (for testing or limited resources)

# Set Python path
export PYTHONPATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try:$PYTHONPATH"

export LLAMA_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/llama-3.2-3b"
export WHISPER_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/whisper-large-v3"
export DATA_PATH="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/sample.json"
export OUTPUT_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/output/hindi_omni_test"

mkdir -p $OUTPUT_DIR

echo "=============================================="
echo "LLaMA-Omni Hindi Training (Single GPU Test)"
echo "=============================================="

# Single GPU with small batch for testing
CUDA_VISIBLE_DEVICES=0 python omni_speech/train/train.py \
    --model_name_or_path $LLAMA_PATH \
    --speech_encoder $WHISPER_PATH \
    --speech_encoder_type "whisper" \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --s2s True \
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
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to "none"
