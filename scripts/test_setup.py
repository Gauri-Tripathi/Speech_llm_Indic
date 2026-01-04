#!/usr/bin/env python3
"""
Test script to verify the training setup is correct.
Run this before starting actual training.
"""

import os
import sys
import json
import torch

def test_paths():
    """Test if all required paths exist."""
    print("=" * 50)
    print("Testing Paths...")
    print("=" * 50)
    
    base_dir = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try"
    
    paths = {
        "Whisper Model": os.path.join(base_dir, "whisper-large-v3"),
        "LLaMA Model": os.path.join(base_dir, "llama-3.2-3b"),
        "Training Data": os.path.join(base_dir, "prepared_data/train.json"),
    }
    
    all_ok = True
    for name, path in paths.items():
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name}: {path}")
        if not exists:
            all_ok = False
    
    # Check LLaMA model has proper files
    llama_path = paths["LLaMA Model"]
    if os.path.exists(llama_path):
        required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        has_model = any(f.endswith('.safetensors') or f.endswith('.bin') 
                       for f in os.listdir(llama_path) if os.path.isfile(os.path.join(llama_path, f)))
        if not has_model:
            print(f"  ⚠ Warning: LLaMA model directory exists but may be missing model weights!")
            print(f"    Please download: huggingface-cli download meta-llama/Llama-3.2-3B-Instruct --local-dir {llama_path}")
            all_ok = False
    
    return all_ok


def test_data_format():
    """Test if training data is in correct format."""
    print("\n" + "=" * 50)
    print("Testing Data Format...")
    print("=" * 50)
    
    data_path = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train.json"
    
    if not os.path.exists(data_path):
        print("  ✗ Training data file not found!")
        return False
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"  Total samples: {len(data)}")
    
    # Check first sample
    sample = data[0]
    required_keys = ["id", "speech", "conversations"]
    
    for key in required_keys:
        if key not in sample:
            print(f"  ✗ Missing required key: {key}")
            return False
    
    print(f"  ✓ Sample ID: {sample['id']}")
    print(f"  ✓ Speech path: {sample['speech']}")
    
    # Check if audio file exists
    if os.path.exists(sample['speech']):
        print(f"  ✓ Audio file exists")
    else:
        print(f"  ✗ Audio file NOT found: {sample['speech']}")
        return False
    
    # Check conversations format
    convs = sample['conversations']
    if len(convs) >= 2:
        print(f"  ✓ Conversations: {len(convs)} turns")
        print(f"    - Human: {convs[0]['value'][:50]}...")
        print(f"    - GPT: {convs[1]['value'][:50]}...")
    else:
        print(f"  ✗ Invalid conversations format")
        return False
    
    # Check tgt_units
    if 'tgt_units' in sample:
        units = sample['tgt_units'].split()
        print(f"  ✓ Target units: {len(units)} tokens")
        max_unit = max(int(u) for u in units)
        print(f"    Max unit value: {max_unit}")
        print(f"    Recommended unit_vocab_size: {max_unit + 100}")
    else:
        print(f"  ⚠ No tgt_units - will train text-only (no speech output)")
    
    return True


def test_whisper_loading():
    """Test if Whisper model can be loaded."""
    print("\n" + "=" * 50)
    print("Testing Whisper Model Loading...")
    print("=" * 50)
    
    whisper_path = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/whisper-large-v3"
    
    try:
        from transformers import WhisperModel
        print("  Loading Whisper model...")
        model = WhisperModel.from_pretrained(whisper_path)
        encoder = model.encoder
        print(f"  ✓ Whisper encoder loaded successfully")
        print(f"    Hidden size: {encoder.config.d_model}")
        print(f"    Num layers: {encoder.config.encoder_layers}")
        del model, encoder
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        return True
    except Exception as e:
        print(f"  ✗ Failed to load Whisper: {e}")
        return False


def test_imports():
    """Test if all required imports work."""
    print("\n" + "=" * 50)
    print("Testing Imports...")
    print("=" * 50)
    
    try:
        from omni_speech.model import OmniSpeechLlamaForCausalLM, OmniSpeech2SLlamaForCausalLM
        print("  ✓ OmniSpeech models imported")
    except ImportError as e:
        print(f"  ✗ Failed to import OmniSpeech: {e}")
        return False
    
    try:
        from omni_speech.arguments import ModelArguments, DataArguments, TrainingArguments
        print("  ✓ Arguments imported")
    except ImportError as e:
        print(f"  ✗ Failed to import arguments: {e}")
        return False
    
    try:
        import transformers
        print(f"  ✓ Transformers version: {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ Transformers not installed: {e}")
        return False
    
    try:
        import whisper
        print(f"  ✓ OpenAI Whisper installed")
    except ImportError:
        print(f"  ⚠ OpenAI Whisper not installed (optional, using HuggingFace)")
    
    return True


def test_cuda():
    """Test CUDA availability."""
    print("\n" + "=" * 50)
    print("Testing CUDA...")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available")
        print(f"    Device count: {torch.cuda.device_count()}")
        print(f"    Current device: {torch.cuda.current_device()}")
        print(f"    Device name: {torch.cuda.get_device_name(0)}")
        
        # Check memory
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    GPU Memory: {total_mem:.1f} GB")
        
        if total_mem < 16:
            print(f"  ⚠ Warning: GPU memory may be insufficient. Consider using gradient checkpointing.")
        return True
    else:
        print(f"  ✗ CUDA not available - training will be very slow!")
        return False


def main():
    print("\n" + "=" * 50)
    print("LLaMA-Omni Hindi Training Setup Test")
    print("=" * 50)
    
    # Change to project directory
    os.chdir("/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try")
    sys.path.insert(0, ".")
    
    results = {
        "Paths": test_paths(),
        "Data Format": test_data_format(),
        "Imports": test_imports(),
        "CUDA": test_cuda(),
        "Whisper": test_whisper_loading(),
    }
    
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    all_passed = True
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! You can start training.")
        print("\nTo train, run:")
        print("  bash scripts/train_hindi_s2s.sh")
    else:
        print("Some tests failed. Please fix the issues above before training.")
    print("=" * 50)


if __name__ == "__main__":
    main()
