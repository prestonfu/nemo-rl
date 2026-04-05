#!/usr/bin/env bash
# bash /docker_workspace/hw3/prepare_blend_dataset.sh

set -euo pipefail

HF_HOME="${HF_HOME:-/scratch/preston/hf_cache}"
RAW_DIR="/scratch/preston/nemotron_rl_data/blend_raw"
OUT_DIR="/scratch/preston/nemotron_rl_data/blend"

mkdir -p "$RAW_DIR" "$OUT_DIR"

# 1. Download raw train.jsonl and the restoration script from HF
echo "Downloading blend dataset..."
HF_HOME="$HF_HOME" python3 - << 'PYEOF'
import os
from huggingface_hub import hf_hub_download
hf_home = os.environ["HF_HOME"]
for filename in ["train.jsonl", "create_nanov3_jsonl.py"]:
    path = hf_hub_download(
        repo_id="nvidia/Nemotron-3-Nano-RL-Training-Blend",
        filename=filename,
        repo_type="dataset",
        local_dir=os.environ["RAW_DIR"],
    )
    print(f"Downloaded: {path}")
PYEOF

# 2. Restore placeholders (pulls DAPO-Math-17k and Skywork-OR1-RL-Data from HF)
echo "Restoring placeholder rows..."
HF_HOME="$HF_HOME" python3 "$RAW_DIR/create_nanov3_jsonl.py" \
    --input "$RAW_DIR/train.jsonl" \
    --output "$RAW_DIR/train_restored.jsonl"

# 3. Filter out workplace_assistant and split 99/1 train/val
echo "Filtering and splitting..."
python3 - << 'PYEOF'
import json, random, os

random.seed(42)

input_path = os.path.join(os.environ["RAW_DIR"], "train_restored.jsonl")
train_path = os.path.join(os.environ["OUT_DIR"], "train.jsonl")
val_path   = os.path.join(os.environ["OUT_DIR"], "validation.jsonl")

rows, skipped = [], 0
with open(input_path) as f:
    for line in f:
        d = json.loads(line)
        agent = d.get("agent_ref", {})
        agent_name = agent.get("name", "") if isinstance(agent, dict) else str(agent)
        if "workplace_assistant" in agent_name:
            skipped += 1
            continue
        rows.append(line.rstrip("\n"))

print(f"Skipped {skipped} workplace_assistant rows, {len(rows)} remaining")

random.shuffle(rows)
split_idx = int(len(rows) * 0.99)
train_rows, val_rows = rows[:split_idx], rows[split_idx:]

print(f"Train: {len(train_rows)}, Val: {len(val_rows)}")

with open(train_path, "w") as f:
    f.write("\n".join(train_rows) + "\n")
with open(val_path, "w") as f:
    f.write("\n".join(val_rows) + "\n")

print(f"Written to {os.environ['OUT_DIR']}")
PYEOF

echo "Done. Output at $OUT_DIR"
