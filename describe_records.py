"""A dataset folder's `metadata.json`, counting the nodes and relationships of each split, and its
card copied in as `README.md`.

    uv run describe_records.py --dataset-dir ./synthetic --card dataset_cards/synthetic.md
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe a dataset folder of records.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("synthetic"))
    parser.add_argument("--card", type=Path, default=Path("dataset_cards/synthetic.md"))
    args = parser.parse_args()
    splits = {}
    for labels in sorted(args.dataset_dir.glob("*_labels.jsonl")):
        counts, seeds = Counter(), []
        with open(labels, encoding="utf-8") as lines:
            for line in lines:
                held = json.loads(line)
                seeds.append(held["seed"])
                counts.update(node["class"] for node in held["nodes"])
                counts.update(f"relationship: {r['type']}" for r in held["relationships"])
                counts.update(f"furniture: {node['class']}" for node in held.get("furniture", {}).get("nodes", []))
        name = labels.name[: -len("_labels.jsonl")]
        splits[name] = {"plans": len(seeds), "seeds": [min(seeds), max(seeds)], "counts": dict(sorted(counts.items()))}
        print(name, len(seeds), flush=True)
    size = np.load(next(args.dataset_dir.glob("*_images.npy")), mmap_mode="r").shape[1]
    (args.dataset_dir / "metadata.json").write_text(json.dumps({
        "format": "plan-records-v2",
        "source": "synthetic.py of synthetic-floorplans-for-transcription: every plan generated from its seed",
        "images": f"(N, {size}, {size}) uint8, drawn from the label by draw.py, row index = y, white background",
        "frame": "pixels of the drawing, x from the left, y from the top",
        "splits": splits,
    }, indent=2), encoding="utf-8")
    (args.dataset_dir / "README.md").write_text(args.card.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
