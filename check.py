#!/usr/bin/env python3
from datasets import load_from_disk
ds = load_from_disk("prepared_data/train_dataset")
# 1. Check a few samples
for i in [0, 100, 1000]:
    item = ds[i]
    print(f"=== Sample {i} ===")
    print(f"Keys: {list(item.keys())}")
    print(f"Speech path: {item['speech']}")
    print(f"Conversations: {item['conversations']}")
    print(f"tgt_units length: {len(item['tgt_units'].split()) if item['tgt_units'] else 0}")
    print()

import os
# 2. Verify all speech files exist
missing = []
for i, item in enumerate(ds):
    if not os.path.exists(item['speech']):
        missing.append(item['speech'])
        
print(f"Missing audio files: {len(missing)}")
if missing:
    print("First 5 missing:", missing[:5])