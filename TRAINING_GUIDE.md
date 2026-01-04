# Hindi LLaMA-Omni Training Setup Guide

## Prerequisites

### 1. Environment Setup

```bash
# Create conda environment
conda create -n llama-omni python=3.10
conda activate llama-omni

# Install the package
cd /nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try
pip install pip==24.0
pip install -e .

# Install fairseq (required for vocoder)
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install -e . --no-build-isolation
cd ..

# Install flash-attention
pip install flash-attn --no-build-isolation

# Install training dependencies
pip install deepspeed==0.12.6 ninja wandb tensorboardX
```

### 2. Download LLaMA 3.2 3B Model

The LLaMA model needs to be downloaded properly. You have two options:

**Option A: Using Hugging Face CLI**
```bash
# Login to Hugging Face
huggingface-cli login

# Download the model
huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir ./llama-3.2-3b-instruct
```

**Option B: Using Python**
```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")

tokenizer.save_pretrained("./llama-3.2-3b-instruct")
model.save_pretrained("./llama-3.2-3b-instruct")
```

### 3. Verify Whisper Model

Your Whisper model is already downloaded at:
`/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/whisper-large-v3`

### 4. Data Format

Your data in `prepared_data/train.json` should have this structure:
```json
[
  {
    "id": "question_005721",
    "speech": "/path/to/audio/question_005721.wav",
    "conversations": [
      {
        "from": "human",
        "value": "<speech>\nAnswer the user's question."
      },
      {
        "from": "gpt",
        "value": "Hindi text response..."
      }
    ],
    "tgt_units": "4140 571 58 2025 ..."
  }
]
```

### 5. Verify Audio Files

Make sure all audio files referenced in train.json exist:
```bash
# Check a sample audio file
python -c "
import json
with open('prepared_data/train.json') as f:
    data = json.load(f)
    
import os
missing = []
for item in data[:10]:  # Check first 10
    if not os.path.exists(item['speech']):
        missing.append(item['speech'])
        
if missing:
    print('Missing files:', missing)
else:
    print('All sample files exist!')
"
```

## Training

### Minimal Training (Single GPU, No DeepSpeed)

```bash
cd /nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try
bash scripts/train_hindi_s2s.sh
```

### Training with DeepSpeed (Recommended for multiple GPUs)

```bash
cd /nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try
bash scripts/train_hindi_s2s_deepspeed.sh
```

## Common Issues & Fixes

### Issue 1: CUDA Out of Memory
- Reduce `per_device_train_batch_size` to 1
- Enable `gradient_checkpointing`
- Use DeepSpeed ZeRO Stage 2/3

### Issue 2: Whisper Model Loading Error
The code expects Whisper in OpenAI format. If using HuggingFace format, the model loading may need adjustment.

### Issue 3: Missing tgt_units
If you don't have target units, set `--has_tgt_units False` and the model will train text-only (no speech output).

### Issue 4: Flash Attention Error
If flash-attention fails, you can disable it by removing the flash_attention_2 config or using:
```bash
pip install flash-attn==2.3.6 --no-build-isolation
```

## Unit Vocabulary Size

Your `tgt_units` appear to have values up to ~5000, so set `--unit_vocab_size 5000` or higher.

## Monitoring Training

```bash
# View tensorboard logs
tensorboard --logdir=exp/hindi_omni_s2s/runs
```

## Inference After Training

After training, you can run inference:
```bash
bash omni_speech/infer/run.sh omni_speech/infer/examples
```
