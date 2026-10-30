#!/usr/bin/env bash
# The published dataset, from its seeds: validation and test, and the training plans in parts of
# 50,000; each part's records, then its images; then the counts and the card.
#
#   ./build_synthetic.sh ./synthetic 256 20     # folder, image size, training parts
#
# Records are placed in a 512 px drawing; drawn at another size, the labels kept beside the images
# are the drawn ones, their points scaled to it, and the 512 px ones are let go. A split once built is
# marked done, so that the build, stopped, takes up again where it was.
set -euo pipefail
OUT="${1:-synthetic}"
SIZE="${2:-512}"
PARTS="${3:-9}"
PART="${PART:-50000}"      # plans in a training part
HELD="${HELD:-5000}"       # plans in validation and in test

build() {
    local split="$1" first="$2" count="$3"
    if [ -f "$OUT/$split.done" ]; then return; fi
    uv run create_synthetic_records.py --output-dir "$OUT" --split "$split" --first-seed "$first" --count "$count" --overwrite
    uv run draw_records.py --dataset-dir "$OUT" --split "$split" --size "$SIZE"
    if [ "$SIZE" != 512 ]; then
        mv "$OUT/${split}_${SIZE}_images.npy" "$OUT/${split}_images.npy"
        mv "$OUT/${split}_${SIZE}_labels.jsonl" "$OUT/${split}_labels.jsonl"
    fi
    touch "$OUT/$split.done"
}

build val 1000000 "$HELD"
build test 2000000 "$HELD"
for ((part = 0; part < PARTS; part++)); do
    build "$(printf 'train-%02d' "$part")" $((part * PART)) "$PART"
done
uv run preview_records.py --dataset-dir "$OUT" --split train-00 --tile "$SIZE" --output "$OUT/train_examples_8x16.png"
uv run describe_records.py --dataset-dir "$OUT" --card dataset_cards/synthetic.md
rm -f "$OUT"/*.done
