"""The images of a split drawn from its records, as the drawing corpora hold them: `{split}_images.npy`,
`(N, size, size)` uint8, row index = y, white background, dark ink, in the order of
`{split}_labels.jsonl`.

The records' points are placed in a 512 px drawing; drawn at another size, the labels are written
anew beside the images with their points scaled to it.

    uv run draw_records.py --dataset-dir ./synthetic --split val
    uv run draw_records.py --dataset-dir ./synthetic --split train --size 256
"""

import argparse
import json
from itertools import islice
import multiprocessing
from pathlib import Path

import numpy as np

import draw
from create_synthetic_records import SIZE

#: How many records are drawn at a time.
BLOCK = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw the images of a split from its records.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("synthetic"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--size", type=int, default=SIZE)
    parser.add_argument("--count", type=int, default=None, help="Draw only the first this many records.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    return parser.parse_args()


def scaled(record: dict, factor: float) -> dict:
    return {**record, "nodes": [{**node, "points": [[round(x * factor, 2), round(y * factor, 2)] for x, y in node["points"]]}
                                if "points" in node else node for node in record["nodes"]]}


def drawn(job):
    line, size = job
    return draw.draw(scaled(json.loads(line), size / SIZE), size)


def main() -> None:
    args = parse_args()
    source = args.dataset_dir / f"{args.split}_labels.jsonl"
    with open(source, encoding="utf-8") as lines:
        count = sum(1 for _ in lines) if args.count is None else args.count
    name = args.split if args.size == SIZE else f"{args.split}_{args.size}"
    images = np.lib.format.open_memmap(args.dataset_dir / f"{name}_images.npy", mode="w+", dtype=np.uint8,
                                       shape=(count, args.size, args.size))
    # Read, drawn and written a block at a time, so that a split of any size fits in memory.
    labels = args.dataset_dir / f"{name}_labels.jsonl.part"
    with open(source, encoding="utf-8") as lines, open(labels, "w", encoding="utf-8") as out, multiprocessing.Pool(args.workers) as pool:
        done = 0
        while done < count:
            block = [line.rstrip("\n") for line in islice(lines, min(BLOCK, count - done))]
            for at, image in enumerate(pool.map(drawn, [(line, args.size) for line in block], chunksize=32)):
                images[done + at] = image
            out.writelines((line if args.size == SIZE else json.dumps(scaled(json.loads(line), args.size / SIZE))) + "\n" for line in block)
            done += len(block)
            if done % (20 * BLOCK) < BLOCK or done == count:
                print(f"{args.split}: {done} of {count}", flush=True)
    images.flush()
    # The labels kept beside the images are the drawn ones, so that the two line up.
    labels.rename(args.dataset_dir / f"{name}_labels.jsonl")
    print(f"drew {count} images into {args.dataset_dir / f'{name}_images.npy'}")


if __name__ == "__main__":
    main()
