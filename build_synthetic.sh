#!/usr/bin/env bash
# The published dataset, from its seeds: validation and test, and the training plans in nine parts of
# 50,000; each part's records, then its images; then the counts and the card.
set -euo pipefail
OUT="${1:-synthetic}"
uv run create_synthetic_records.py --output-dir "$OUT" --split val --first-seed 1000000 --count 5000
uv run create_synthetic_records.py --output-dir "$OUT" --split test --first-seed 2000000 --count 5000
uv run draw_records.py --dataset-dir "$OUT" --split val
uv run draw_records.py --dataset-dir "$OUT" --split test
for part in 0 1 2 3 4 5 6 7 8; do
    uv run create_synthetic_records.py --output-dir "$OUT" --split "train-0$part" --first-seed $((part * 50000)) --count 50000
    uv run draw_records.py --dataset-dir "$OUT" --split "train-0$part"
done
uv run preview_records.py --dataset-dir "$OUT" --split train-00 --output "$OUT/train_examples_8x16.png"
uv run describe_records.py --dataset-dir "$OUT" --card dataset_cards/synthetic.md
