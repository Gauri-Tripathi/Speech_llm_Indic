#!/bin/bash
# Training script using Accelerate for Hindi LLaMA-Omni
# Run step-by-step manually or execute this script

set -e

# ============================================
# STEP 1: Set up environment variables
# ============================================
export WANDB_PROJECT="hindi-llama-omni"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1

# ============================================
# STEP 2: Define paths
# ============================================
BASE_DIR="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try"
LLAMA_MODEL="${BASE_DIR}/llama-3.2-3b"
WHISPER_MODEL="${BASE_DIR}/whisper-large-v3"
TRAIN_DATA="${BASE_DIR}/prepared_data/train_combined.json"
OUTPUT_DIR="${BASE_DIR}/exp/hindi_omni_s2s_accelerate"
WANDB_RUN_NAME="hindi_omni_s2s_$(date +%Y%m%d_%H%M%S)"

mkdir -p $OUTPUT_DIR
cd $BASE_DIR

# ============================================
# STEP 3: Launch training with accelerate
# ============================================
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
    --ctc_decoder_config "(2,2048,8,4096)" \
    --ctc_upsample_factor 25 \
    --ctc_loss_weight 0.1 \
    --unit_vocab_size 5100 \
    --tune_speech_projector False \
    --freeze_backbone False \
    --input_type mel \
    --mel_size 128 \
    --has_tgt_units True \
    --bf16 True \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 0.5 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name $WANDB_RUN_NAME \
    --ddp_find_unused_parameters False
