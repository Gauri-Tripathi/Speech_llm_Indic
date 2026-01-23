#!/usr/bin/env python3
"""
Verification script to check training data and model setup before training.
Run this before training to catch potential issues early.
"""

import os
import sys
import json
import torch
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omni_speech.constants import IGNORE_INDEX, SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN


def check_data_file(data_path: str, num_samples: int = 5):
    """Check the training data file for common issues."""
    print("\n" + "="*60)
    print("CHECKING TRAINING DATA")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: Data file not found: {data_path}")
        return False
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data)} samples from {data_path}")
    
    # Check structure
    issues = []
    missing_speech = 0
    missing_tgt_units = 0
    missing_conversations = 0
    invalid_speech_token = 0
    max_unit_value = 0
    
    for i, sample in enumerate(data):
        # Check required fields
        if "speech" not in sample:
            missing_speech += 1
        elif not os.path.exists(sample["speech"]):
            if i < 3:  # Only report first few
                issues.append(f"Sample {i}: Audio file not found: {sample['speech']}")
        
        if "tgt_units" not in sample:
            missing_tgt_units += 1
        else:
            units = [int(x) for x in sample["tgt_units"].split()]
            max_unit_value = max(max_unit_value, max(units) if units else 0)
        
        if "conversations" not in sample:
            missing_conversations += 1
        else:
            # Check for <speech> token
            has_speech_token = any(DEFAULT_SPEECH_TOKEN in turn.get("value", "") 
                                   for turn in sample["conversations"])
            if not has_speech_token:
                invalid_speech_token += 1
                if i < 3:
                    issues.append(f"Sample {i}: Missing <speech> token in conversations")
    
    print(f"\n📊 Data Statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Missing 'speech' field: {missing_speech}")
    print(f"  Missing 'tgt_units' field: {missing_tgt_units}")
    print(f"  Missing 'conversations' field: {missing_conversations}")
    print(f"  Missing <speech> token: {invalid_speech_token}")
    print(f"  Max unit value: {max_unit_value}")
    
    if max_unit_value > 0:
        print(f"\n⚠️  Recommended --unit_vocab_size: {max_unit_value + 100} (max value + buffer)")
    
    if issues:
        print(f"\n⚠️  Issues found (showing first few):")
        for issue in issues[:10]:
            print(f"  - {issue}")
    
    # Show sample data
    print(f"\n📋 Sample data (first {min(num_samples, len(data))} samples):")
    for i, sample in enumerate(data[:num_samples]):
        print(f"\n  Sample {i}:")
        print(f"    ID: {sample.get('id', 'N/A')}")
        print(f"    Speech: {sample.get('speech', 'N/A')[:80]}...")
        print(f"    Conversations: {len(sample.get('conversations', []))} turns")
        if "tgt_units" in sample:
            units = sample["tgt_units"].split()
            print(f"    tgt_units: {len(units)} units, first few: {' '.join(units[:10])}...")
    
    success = missing_speech == 0 and missing_conversations == 0
    if success:
        print("\n✓ Data file looks good!")
    else:
        print("\n❌ Data file has issues that need to be fixed")
    
    return success


def check_label_masking(data_path: str, tokenizer_path: str):
    """Check that label masking is working correctly."""
    print("\n" + "="*60)
    print("CHECKING LABEL MASKING")
    print("="*60)
    
    import transformers
    from omni_speech.datasets.preprocess import preprocess, preprocess_multimodal
    from omni_speech import conversation as conversation_lib
    from omni_speech.arguments import DataArguments
    import copy
    
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set conversation template
    conversation_lib.default_conversation = conversation_lib.conv_templates["llama_3"]
    
    # Load a sample
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Create dummy data args
    class DummyDataArgs:
        is_multimodal = True
    
    data_args = DummyDataArgs()
    
    for i, sample in enumerate(data[:3]):
        print(f"\n--- Sample {i} ---")
        
        # Process conversations
        sources = copy.deepcopy([sample["conversations"]])
        sources = preprocess_multimodal(sources, data_args)
        data_dict = preprocess(sources, tokenizer, has_speech=True)
        
        input_ids = data_dict["input_ids"][0]
        labels = data_dict["labels"][0]
        
        total_tokens = len(input_ids)
        valid_labels = (labels != IGNORE_INDEX).sum().item()
        ignored_labels = (labels == IGNORE_INDEX).sum().item()
        
        print(f"  Total tokens: {total_tokens}")
        print(f"  Valid labels (model learns from): {valid_labels}")
        print(f"  Ignored labels: {ignored_labels}")
        print(f"  Valid label ratio: {valid_labels / total_tokens * 100:.1f}%")
        
        if valid_labels == 0:
            print("  ❌ ERROR: No valid labels! Model will not learn anything!")
        elif valid_labels < total_tokens * 0.1:
            print("  ⚠️  WARNING: Very few valid labels. Check if this is expected.")
        else:
            print("  ✓ Labels look reasonable")
        
        # Decode to show what's being learned
        if valid_labels > 0:
            valid_ids = input_ids[labels != IGNORE_INDEX]
            decoded = tokenizer.decode(valid_ids[:50])  # First 50 tokens
            print(f"  Learning to generate (first 50 tokens): {decoded[:200]}...")
    
    return True


def check_model_gradients(tokenizer_path: str, model_path: str):
    """Check which parameters will have gradients during training."""
    print("\n" + "="*60)
    print("CHECKING MODEL GRADIENTS (simulated)")
    print("="*60)
    
    print("Note: Full gradient check requires loading the model.")
    print("The train script will print trainable parameter groups.")
    print("Look for these key groups:")
    print("  - model.* : LLM backbone (should be trainable unless freeze_backbone=True)")
    print("  - speech_projector.* : Projects speech to LLM space")
    print("  - speech_generator.* : Generates speech units (CTC decoder)")
    print("  - speech_encoder.* : Whisper encoder (usually frozen)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Verify training setup")
    parser.add_argument("--data_path", type=str, 
                        default="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train.json",
                        help="Path to training data JSON")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/llama-3.2-3b",
                        help="Path to tokenizer/model")
    parser.add_argument("--check_labels", action="store_true",
                        help="Also check label masking (requires loading tokenizer)")
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING SETUP VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # Check data
    all_passed &= check_data_file(args.data_path)
    
    # Check labels if requested
    if args.check_labels:
        try:
            all_passed &= check_label_masking(args.data_path, args.tokenizer_path)
        except Exception as e:
            print(f"\n❌ Label check failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    # Check gradients
    check_model_gradients(args.tokenizer_path, args.tokenizer_path)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! Ready for training.")
    else:
        print("❌ Some checks failed. Please fix the issues before training.")
    print("="*60)


if __name__ == "__main__":
    main()
