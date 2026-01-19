# LLaMA-Omni (try folder)

A speech-enabled LLM for instruction-following from audio input, based on LLaMA 3.2-3B and Whisper encoder.

## Project Structure

```
try/
├── scripts/                    # Training scripts
│   ├── train_hindi.sh        # Hindi training (4 GPU)
│   ├── train_hindi_s2s.sh   # Speech-to-speech training
│   ├── train_hindi_single_gpu.sh
│   ├── train_2stage.sh      # Two-stage training
│   └── ...
│
├── omni_speech/              # Main package
│   ├── train/train.py       # Training entry point
│   ├── model/
│   │   ├── omni_speech_arch.py      # Base model architecture
│   │   └── language_model/
│   │       ├── omni_speech_llama.py       # LLaMA with speech input
│   │       └── omni_speech2s_llama.py     # Speech-to-speech model
│   ├── datasets/
│   │   └── preprocess.py    # Data tokenization
│   ├── infer/infer.py       # Inference code
│   └── serve/               # Gradio web server
│
├── llama-3.2-3b/            # LLaMA 3.2 3B model weights
├── whisper-large-v3/         # Whisper encoder weights
└── deepspeed_config.json    # DeepSpeed config
```

## Model Architecture

```
Input Audio (WAV)
       │
       ▼
┌──────────────────┐
│  Whisper Encoder │  (frozen)
│  (speech_encoder)│
└────────┬─────────┘
         │ features
         ▼
┌──────────────────┐
│ Linear Projector│  (trainable)
└────────┬─────────┘
         │ embeddings
         ▼
┌──────────────────┐
│   LLaMA 3.2-3B   │
│      (LLM)       │
└────────┬─────────┘
         │
         ▼ (optional for S2S)
┌──────────────────┐
│  CTC Decoder     │  (speech generator)
│  (speech output)│
└──────────────────┘
```

## Training Data Format

Create a JSON file with this structure:

```json
[
  {
    "speech": "/path/to/audio.wav",
    "conversations": [
      {"from": "human", "value": "<speech>\nWhat is in this audio?"},
      {"from": "gpt", "value": "The audio contains speech about..."}
    ],
    "tgt_units": "12 34 56 78 ..."  (only for speech-to-speech model)
  }
]
```

- `<speech>` token marks where audio embedding goes
- `tgt_units` are discrete speech units (phoneme-like tokens) for S2S model

## Running Training

### Quick Start

1. Edit `scripts/train_hindi.sh` to set your paths:
```bash
export DATA_PATH="/path/to/your/train.json"
export OUTPUT_DIR="/path/to/output"
```

2. Run:
```bash
cd try
bash scripts/train_hindi.sh
```

### Key Training Scripts

| Script | Description |
|--------|-------------|
| `train_hindi.sh` | Standard 4-GPU training |
| `train_hindi_s2s.sh` | Speech-to-speech training |
| `train_hindi_single_gpu.sh` | Single GPU training |
| `train_2stage.sh` | Two-stage (projector first, then full) |

### Important Arguments

```bash
# Model
--model_name_or_path ./llama-3.2-3b      # LLaMA model path
--speech_encoder ./whisper-large-v3     # Whisper path
--speech_encoder_type "whisper"

# Speech-to-Speech (optional)
--s2s True                               # Enable S2S mode
--unit_vocab_size 5100                   # Vocab for CTC decoder
--ctc_loss_weight 1.0                    # Weight for CTC loss

# Training
--num_train_epochs 3
--per_device_train_batch_size 2
--gradient_accumulation_steps 8
--learning_rate 2e-5
--bf16 True
--gradient_checkpointing True            # Save memory
--output_dir ./output/model

# Data
--data_path ./prepared_data/train.json
--input_type "mel"                       # "mel" or "raw"
--mel_size 128
```

### Training Modes

1. **Full fine-tuning**: Train all components
2. **Projector only**: `--tune_speech_projector True` (freeze LLM)
3. **S2S only**: `--tune_speech_generator_only True` (train only CTC decoder)

### Multi-GPU Training

```bash
# 4 GPUs with torchrun
torchrun --nproc_per_node=4 omni_speech/train/train.py [args...]

# With DeepSpeed
deepspeed --num_gpus=4 train.py [args...]
```

## Components Detail

### train.py
- Entry point for training
- Loads LLaMA tokenizer
- Initializes model with speech modules
- Creates dataset and data collator
- Runs HuggingFace Trainer

### omni_speech_arch.py
- `OmniSpeechMetaModel`: Adds speech encoder + projector to base LLM
- `OmniSpeechMetaForCausalLM`: Handles speech encoding and merging with text

### omni_speech2s_llama.py
- Speech-to-speech model with CTC decoder
- Combined loss: LM loss + CTC loss

### preprocess.py
- Tokenizes conversations with LLaMA 3 template
- Handles `<speech>` token insertion
- Masks human input (only train on assistant response)

## Inference

```bash
cd omni_speech/infer
bash run.sh
```

## Requirements

- PyTorch with CUDA
- Transformers
- Whisper
- DeepSpeed (optional)
- Accelerate (optional)

## Hardware

- Recommended: 4x A100 (40GB) for 3B model
- With gradient checkpointing: 2x A100 may work
- Single GPU: possible but slow

## Notes

- Speech encoder is typically frozen (not trained)
- Use `--bf16` for faster training on Ampere+ GPUs
- Gradient checkpointing reduces memory by ~30%
- `<speech>` token position determines where audio embeddings are inserted
