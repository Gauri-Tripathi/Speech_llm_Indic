#!/bin/bash
# Optimized 2-Stage Training for Hindi LLaMA-Omni
# Stage 1: Train LLM + Speech Adapter (lower LR)
# Stage 2: Train CTC Speech Decoder (higher LR)

set -e

# Paths
BASE_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try"
LLAMA_MODEL="${BASE_DIR}/llama-3.2-3b"
WHISPER_MODEL="${BASE_DIR}/whisper-large-v3"
TRAIN_DATA="${BASE_DIR}/prepared_data/train_combined.json"
OUTPUT_DIR_STAGE1="${BASE_DIR}/exp/hindi_omni_stage1"
OUTPUT_DIR_STAGE2="${BASE_DIR}/exp/hindi_omni_stage2_final"

# WandB settings
export WANDB_PROJECT="hindi-llama-omni"

# ==============================================================================
# STAGE 1: Train LLM + Speech Adapter (freeze CTC decoder, lower LR)
# ==============================================================================
echo "=============================================="
echo "STAGE 1: Training LLM + Speech Adapter"
echo "LR: 2e-5, Epochs: 5"
echo "=============================================="

mkdir -p $OUTPUT_DIR_STAGE1

accelerate launch \
    -m omni_speech.train.train \
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
    --ctc_decoder_config "(4,4096,32,11008)" \
    --ctc_upsample_factor 10 \
    --ctc_loss_weight 1.0 \
    --unit_vocab_size 5100 \
    --tune_speech_projector False \
    --freeze_backbone False \
    --input_type mel \
    --mel_size 128 \
    --has_tgt_units True \
    --bf16 True \
    --output_dir $OUTPUT_DIR_STAGE1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name "stage1_llm_adapter_noctc"

echo "Stage 1 complete! Model saved to: $OUTPUT_DIR_STAGE1"

# ==============================================================================
# STAGE 2: Fine-tune CTC Speech Decoder (higher LR, fewer epochs)
# ==============================================================================
echo "=============================================="
echo "STAGE 2: Fine-tuning CTC Speech Decoder"
echo "LR: 2e-4, Epochs: 5"
echo "=============================================="

mkdir -p $OUTPUT_DIR_STAGE2

# Use the best checkpoint from Stage 1
STAGE1_CHECKPOINT="${OUTPUT_DIR_STAGE1}"

accelerate launch \
    -m omni_speech.train.train \
    --model_name_or_path $STAGE1_CHECKPOINT \
    --version llama_3 \
    --data_path $TRAIN_DATA \
    --speech_encoder $WHISPER_MODEL \
    --speech_encoder_type whisper \
    --speech_projector_type linear \
    --speech_encoder_ds_rate 5 \
    --speech_encoder_hidden_size 1280 \
    --s2s True \
    --speech_generator_type ctc \
    --ctc_decoder_config "(4,4096,32,11008)" \
    --ctc_upsample_factor 10 \
    --ctc_loss_weight 1.0 \
    --unit_vocab_size 5100 \
    --tune_speech_generator_only True \
    --input_type mel \
    --mel_size 128 \
    --has_tgt_units True \
    --bf16 True \
    --output_dir $OUTPUT_DIR_STAGE2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name "stage2_ctc_decoder"

echo "=============================================="
echo "TRAINING COMPLETE!"
echo "Final model saved to: $OUTPUT_DIR_STAGE2"
echo "=============================================="
