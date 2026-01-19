#!/usr/bin/env python3
"""
Convert HuggingFace datasets (dolly_hindi_units, oasst_units) to LLaMA-Omni training format.

Expected input format:
- instruction_text: User's question/instruction
- instruction_units: List of speech units for instruction
- instruction_wav_path: Path to instruction audio
- response_text: Assistant's response
- response_units: List of speech units for response
- response_wav_path: Path to response audio

Output format (JSON):
- id: Sample ID
- speech: Path to instruction audio
- tgt_units: Space-separated response units
- conversations: List of human/gpt turns
"""

import os
import json
import argparse
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm


def convert_sample(sample, idx):
    """Convert a single sample to LLaMA-Omni format."""
    
    # Get instruction (audio path) and response units
    instruction_wav = sample.get('instruction_wav_path', '')
    response_units = sample.get('response_units', [])
    instruction_text = sample.get('instruction_text', '')
    response_text = sample.get('response_text', '')
    
    # Skip if missing required fields
    if not instruction_wav or not os.path.exists(instruction_wav):
        return None
    if not response_units:
        return None
    if not instruction_text or not response_text:
        return None
    
    # Convert units list to space-separated string
    if isinstance(response_units, list):
        tgt_units_str = ' '.join(map(str, response_units))
    else:
        tgt_units_str = str(response_units)
    
    # Create conversation format
    # The <speech>\n at the start tells the model this is audio input
    conversations = [
        {
            "from": "human",
            "value": f"<speech>\n{instruction_text}"
        },
        {
            "from": "gpt", 
            "value": response_text
        }
    ]
    
    return {
        "id": sample.get('id', f"sample_{idx}"),
        "speech": instruction_wav,
        "tgt_units": tgt_units_str,
        "conversations": conversations
    }


def main():
    parser = argparse.ArgumentParser(description="Convert datasets to LLaMA-Omni format")
    parser.add_argument("--dolly_path", type=str, 
                        default="/nlsasfs/home/dibd/dibd-speech/iitm/triga/speech_llm/data/dolly_hindi_units_dataset",
                        help="Path to dolly_hindi_units_dataset")
    parser.add_argument("--oasst_path", type=str,
                        default="/nlsasfs/home/dibd/dibd-speech/iitm/triga/speech_llm/data/oasst_units_dataset",
                        help="Path to oasst_units_dataset")
    parser.add_argument("--output_path", type=str,
                        default="/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train_combined.json",
                        help="Output JSON file path")
    parser.add_argument("--max_unit_value", action="store_true",
                        help="Print max unit value for setting unit_vocab_size")
    args = parser.parse_args()
    
    all_samples = []
    max_unit = 0
    total_skipped = 0
    
    # Process Dolly dataset
    if os.path.exists(args.dolly_path):
        print(f"Loading Dolly dataset from {args.dolly_path}...")
        dolly_ds = load_from_disk(args.dolly_path)
        
        # Handle DatasetDict (has splits like 'train') vs Dataset
        if hasattr(dolly_ds, 'keys'):
            # It's a DatasetDict, get all splits
            print(f"  Found splits: {list(dolly_ds.keys())}")
            datasets_to_process = []
            for split_name in dolly_ds.keys():
                datasets_to_process.append((split_name, dolly_ds[split_name]))
        else:
            # It's a single Dataset
            datasets_to_process = [('all', dolly_ds)]
        
        for split_name, ds in datasets_to_process:
            print(f"  Processing split '{split_name}' with {len(ds)} samples")
            for idx in tqdm(range(len(ds)), desc=f"Dolly-{split_name}"):
                sample = ds[idx]  # Access by index
                converted = convert_sample(sample, f"dolly_{split_name}_{idx}")
            if converted:
                all_samples.append(converted)
                # Track max unit value
                units = [int(u) for u in converted['tgt_units'].split()]
                if units:
                    max_unit = max(max_unit, max(units))
            else:
                total_skipped += 1
    else:
        print(f"Dolly dataset not found at {args.dolly_path}")
    
    # Process OASST dataset
    if os.path.exists(args.oasst_path):
        print(f"\nLoading OASST dataset from {args.oasst_path}...")
        oasst_ds = load_from_disk(args.oasst_path)
        
        # Handle DatasetDict (has splits like 'train') vs Dataset
        if hasattr(oasst_ds, 'keys'):
            datasets_to_process = []
            for split_name in oasst_ds.keys():
                datasets_to_process.append((split_name, oasst_ds[split_name]))
        else:
            datasets_to_process = [('all', oasst_ds)]
        
        for split_name, ds in datasets_to_process:
            print(f"  Processing split '{split_name}' with {len(ds)} samples")
            for idx in tqdm(range(len(ds)), desc=f"OASST-{split_name}"):
                sample = ds[idx]
                converted = convert_sample(sample, f"oasst_{split_name}_{idx}")
                if converted:
                    all_samples.append(converted)
                    # Track max unit value
                    units = [int(u) for u in converted['tgt_units'].split()]
                    if units:
                        max_unit = max(max_unit, max(units))
                else:
                    total_skipped += 1
    else:
        print(f"OASST dataset not found at {args.oasst_path}")
    
    # Save to JSON
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"Total samples: {len(all_samples)}")
    print(f"Skipped samples: {total_skipped}")
    print(f"Max unit value: {max_unit}")
    print(f"Recommended --unit_vocab_size: {max_unit + 100}")
    print(f"Output saved to: {args.output_path}")
    print(f"{'='*60}")
    
    # Show sample
    if all_samples:
        print(f"\nSample entry:")
        sample = all_samples[0]
        print(f"  ID: {sample['id']}")
        print(f"  Speech: {sample['speech']}")
        print(f"  Units (first 10): {' '.join(sample['tgt_units'].split()[:10])}...")
        print(f"  Conversation:")
        for turn in sample['conversations']:
            role = "Human" if turn['from'] == 'human' else "GPT"
            text = turn['value'][:80] + "..." if len(turn['value']) > 80 else turn['value']
            print(f"    {role}: {text}")


if __name__ == "__main__":
    main()
