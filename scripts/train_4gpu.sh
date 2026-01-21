#!/bin/bash
# Simplified training with 4 GPUs for stability testing

cd /nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try

# Environment
export WANDB_PROJECT="hindi-llama-omni"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Simple accelerate launch with 4 GPUs
accelerate launch \
    --num_processes 8 \
    --mixed_precision bf16 \
    -m omni_speech.train.train \
    --model_name_or_path ./llama-3.2-3b \
    --version llama_3 \
    --data_path ./prepared_data/train_combined.json \
    --speech_encoder ./whisper-large-v3 \
    --speech_encoder_type whisper \
    --speech_projector_type linear \
    --speech_encoder_ds_rate 5 \
    --speech_encoder_hidden_size 1280 \
    --s2s True \
    --speech_generator_type ctc \
    --ctc_decoder_config "(2,2048,8,4096)" \
    --ctc_upsample_factor 25 \
    --ctc_loss_weight 0.1 \
    --unit_vocab_size 5000 \
    --tune_speech_projector False \
    --freeze_backbone False \
    --input_type mel \
    --mel_size 128 \
    --has_tgt_units True \
    --bf16 True \
    --output_dir ./exp/hindi_omni_4gpu \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 5e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.10 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_grad_norm 0.5 \
    --model_max_length 1024 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --report_to wandb \
    --run_name "hindi_omni_4gpu_stable" \
    --ddp_find_unused_parameters False
