#!/bin/bash
# Training script for Hindi LLaMA-Omni
# Minimal setup for training with Whisper-large-v3 and LLaMA-3.2-3B

# Set paths
LLAMA_MODEL="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/llama-3.2-3b"
WHISPER_MODEL="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/whisper-large-v3"
TRAIN_DATA="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train.json"
OUTPUT_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/exp/hindi_omni_s2s"

# Create output directory
mkdir -p $OUTPUT_DIR

# Run training
python -m omni_speech.train.train \
    --model_name_or_path $LLAMA_MODEL \
    --version llama_3 \
    --data_path $TRAIN_DATA \
    --speech_encoder $WHISPER_MODEL \
    --speech_encoder_type whisper \
    --speech_projector_type linear \
    --speech_encoder_ds_rate 5 \
    --speech_encoder_hidden_size 1280 \
    --s2s True \
    --speech_generator_type ctc \
    --ctc_decoder_config "(2,4096,32,11008)" \
    --ctc_upsample_factor 25 \
    --ctc_loss_weight 1.0 \
    --unit_vocab_size 5000 \
    --tune_speech_projector False \
    --freeze_backbone False \
    --input_type mel \
    --mel_size 128 \
    --has_tgt_units True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard
