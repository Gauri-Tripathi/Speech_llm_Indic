#!/usr/bin/env python3
"""
Debug script to test each component step by step.
Run this to identify any issues before training.
"""

import os
import sys
import json
import torch

# Configuration
LLAMA_PATH = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/llama-3.2-3b"
WHISPER_PATH = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/whisper-large-v3"
DATA_PATH = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/sample.json"

def step(num, name):
    print(f"\n{'='*60}")
    print(f"Step {num}: {name}")
    print('='*60)

def test_step_1_imports():
    step(1, "Testing basic imports")
    try:
        import transformers
        print(f"  ✓ transformers version: {transformers.__version__}")
        
        import torch
        print(f"  ✓ torch version: {torch.__version__}")
        print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
        
        import whisper
        print(f"  ✓ openai-whisper imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_step_2_omni_speech_imports():
    step(2, "Testing OmniSpeech imports")
    try:
        # Add path
        sys.path.insert(0, '/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try')
        
        from omni_speech.constants import IGNORE_INDEX, DEFAULT_SPEECH_TOKEN
        print(f"  ✓ constants imported")
        
        from omni_speech.model import OmniSpeechLlamaForCausalLM, OmniSpeech2SLlamaForCausalLM
        print(f"  ✓ model classes imported")
        
        from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal, tokenizer_speech_token
        print(f"  ✓ preprocess functions imported")
        
        from omni_speech import conversation as conversation_lib
        print(f"  ✓ conversation lib imported")
        
        from omni_speech.model.speech_encoder.builder import build_speech_encoder
        print(f"  ✓ speech encoder builder imported")
        
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_3_load_tokenizer():
    step(3, "Testing tokenizer loading")
    try:
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(
            LLAMA_PATH,
            use_fast=False,
            padding_side="right"
        )
        print(f"  ✓ Tokenizer loaded")
        print(f"    vocab_size: {tokenizer.vocab_size}")
        print(f"    pad_token: {tokenizer.pad_token}")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print(f"    (set pad_token to eos_token)")
        
        return True, tokenizer
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_step_4_load_whisper():
    step(4, "Testing Whisper encoder loading")
    try:
        from transformers import WhisperModel
        
        whisper_model = WhisperModel.from_pretrained(WHISPER_PATH)
        encoder = whisper_model.encoder
        print(f"  ✓ Whisper encoder loaded")
        print(f"    hidden_size: {encoder.config.d_model}")
        
        # Test forward pass
        dummy_input = torch.randn(1, 128, 3000)  # [batch, mel_dim, seq_len]
        with torch.no_grad():
            output = encoder(dummy_input)
        print(f"    output shape: {output.last_hidden_state.shape}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_5_load_llama():
    step(5, "Testing LLaMA model loading")
    try:
        from omni_speech.model import OmniSpeech2SLlamaForCausalLM
        
        print(f"  Loading model from {LLAMA_PATH}...")
        model = OmniSpeech2SLlamaForCausalLM.from_pretrained(
            LLAMA_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print(f"  ✓ Model loaded")
        print(f"    hidden_size: {model.config.hidden_size}")
        print(f"    num_layers: {model.config.num_hidden_layers}")
        
        return True, model
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_step_6_initialize_speech_modules(model):
    step(6, "Testing speech module initialization")
    try:
        from dataclasses import dataclass
        from typing import Optional
        
        @dataclass
        class MockModelArgs:
            speech_encoder: str = WHISPER_PATH
            speech_encoder_type: str = "whisper"
            speech_projector_type: str = "linear"
            speech_encoder_ds_rate: int = 5
            speech_encoder_hidden_size: int = 1280
            pretrain_speech_projector: Optional[str] = None
            speech_generator_type: str = "ctc"
            ctc_decoder_config: str = "(2,3072,32,8192)"
            ctc_upsample_factor: int = 25
            ctc_loss_weight: float = 1.0
            unit_vocab_size: int = 5100
            tune_speech_generator_only: bool = False
        
        model_args = MockModelArgs()
        
        print("  Initializing speech modules...")
        model.get_model().initialize_speech_modules(model_args)
        print(f"  ✓ Speech encoder initialized")
        print(f"  ✓ Speech projector initialized")
        
        print("  Initializing speech generator...")
        model.initialize_speech_generator(model_args)
        print(f"  ✓ Speech generator initialized")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"    Total params: {total_params:,}")
        print(f"    Trainable params: {trainable_params:,}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_7_test_data_loading():
    step(7, "Testing data loading")
    try:
        import whisper
        
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
        
        sample = data[0]
        print(f"  ✓ Data loaded: {len(data)} samples")
        
        # Load audio
        speech_file = sample["speech"]
        speech = whisper.load_audio(speech_file)
        print(f"  ✓ Audio loaded: {speech_file}")
        print(f"    shape: {speech.shape}, duration: {len(speech)/16000:.2f}s")
        
        # Convert to mel
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=128).permute(1, 0)
        print(f"  ✓ Mel spectrogram: {speech.shape}")
        
        # Check tgt_units
        if "tgt_units" in sample:
            units = [int(x) for x in sample["tgt_units"].split()]
            print(f"  ✓ Target units: {len(units)} tokens, max={max(units)}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("LLaMA-Omni Hindi Training Debug Script")
    print("="*60)
    
    results = {}
    
    # Step 1
    results[1] = test_step_1_imports()
    if not results[1]:
        print("\n❌ Basic imports failed. Please install dependencies.")
        return
    
    # Step 2
    results[2] = test_step_2_omni_speech_imports()
    if not results[2]:
        print("\n❌ OmniSpeech imports failed. Check the error above.")
        return
    
    # Step 3
    ok, tokenizer = test_step_3_load_tokenizer()
    results[3] = ok
    if not ok:
        print(f"\n❌ Tokenizer loading failed. Check that LLaMA model exists at: {LLAMA_PATH}")
        return
    
    # Step 4
    results[4] = test_step_4_load_whisper()
    if not results[4]:
        print(f"\n❌ Whisper loading failed. Check that model exists at: {WHISPER_PATH}")
        return
    
    # Step 5
    ok, model = test_step_5_load_llama()
    results[5] = ok
    if not ok:
        print(f"\n❌ LLaMA loading failed. Check the model at: {LLAMA_PATH}")
        return
    
    # Step 6
    results[6] = test_step_6_initialize_speech_modules(model)
    if not results[6]:
        print("\n❌ Speech module initialization failed.")
        return
    
    # Step 7
    results[7] = test_step_7_test_data_loading()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    all_passed = all(results.values())
    for step_num, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  Step {step_num}: {status}")
    
    if all_passed:
        print("\n✅ All tests passed! You can now run training with:")
        print("   bash scripts/train_hindi_single_gpu.sh")
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")

if __name__ == "__main__":
    main()
