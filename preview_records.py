"""A sheet of plans from a split: each input image next to its record, drawn alone.

Outside walls are drawn dark red, inside walls orange, doors magenta, outside doors green, windows
blue, each segment from its start (dot) to its end. The sheet goes into the dataset folder, for the
card.

    uv run preview_records.py --dataset-dir ./synthetic --split train-00 --output ./synthetic/train_examples_8x16.png
"""

import argparse
import json
from itertools import islice
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

COLOURS = {"wall": (139, 26, 26), "outside wall": (139, 26, 26), "inside wall": (212, 105, 30),
           "door": (194, 24, 122), "outside door": (10, 143, 92), "window": (31, 99, 214)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A sheet of plans and their records.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--split", default="train")
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--cols", type=int, default=16, help="Columns of tiles; each plan takes two, image and record.")
    parser.add_argument("--tile", type=int, default=256, help="Side of a tile in pixels.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = np.load(args.dataset_dir / f"{args.split}_images.npy", mmap_mode="r")
    with open(args.dataset_dir / f"{args.split}_labels.jsonl", encoding="utf-8") as lines:
        labels = list(islice(lines, args.rows * (args.cols // 2)))
    size, tile, gap = images.shape[1], args.tile, 6
    pairs = args.cols // 2
    sheet = Image.new("RGB", (args.cols * tile + (pairs - 1) * gap, args.rows * tile), "white")
    for at in range(args.rows * pairs):
        row, col = divmod(at, pairs)
        x = col * (2 * tile + gap)
        sheet.paste(Image.fromarray(np.asarray(images[at])).convert("RGB").resize((tile, tile), Image.LANCZOS), (x, row * tile))

        drawn = Image.new("RGB", (size, size), "white")
        pen = ImageDraw.Draw(drawn)
        for node in json.loads(labels[at])["nodes"]:
            (x0, y0), (x1, y1) = node["points"]
            colour = COLOURS[node["class"]]
            pen.line([(x0, y0), (x1, y1)], fill=colour, width=3)
            pen.ellipse([x0 - 3, y0 - 3, x0 + 3, y0 + 3], fill=colour)
        sheet.paste(drawn.resize((tile, tile), Image.LANCZOS), (x + tile, row * tile))
        ImageDraw.Draw(sheet).rectangle([x, row * tile, x + 2 * tile - 1, (row + 1) * tile - 1], outline=(221, 211, 197))
    output = args.output or args.dataset_dir / f"{args.split}_examples_{args.rows}x{args.cols}.png"
    sheet.save(output, optimize=True)
    print(output)


if __name__ == "__main__":
    main()
