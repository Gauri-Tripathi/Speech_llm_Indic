"""
Convert JSON training data to HuggingFace Dataset format.
Run once before training - creates efficient Arrow-format dataset.
"""
import json
import os
from datasets import Dataset

# Input/Output paths
JSON_PATH = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train_combined.json"
OUTPUT_PATH = "/nlsasfs/home/dibd/dibd-speech/iitm/triga/LLaMA-Omni/try/prepared_data/train_dataset"

print(f"Loading JSON from {JSON_PATH}...")
with open(JSON_PATH, "r") as f:
    data = json.load(f)
print(f"Loaded {len(data)} samples")

# Create HuggingFace Dataset
dataset = Dataset.from_list(data)
print("Saving to HuggingFace format...")
dataset.save_to_disk(OUTPUT_PATH)
print(f"Done! Saved to {OUTPUT_PATH}")

# Calculate size correctly
total_size = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(OUTPUT_PATH)
    for f in files
)
print(f"Size on disk: {total_size / 1024 / 1024:.1f} MB")
