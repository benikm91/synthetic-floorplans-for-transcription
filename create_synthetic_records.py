"""The records of synthetic floor plans: per plan, its seed and its record (`synthetic.py`), one line
each. Generating is what takes the time; `draw_records.py` draws the images from the records after.

A split is a run of seeds: train from 0, validation from 1,000,000, test from 2,000,000. A large split
is written in parts, each its own run of seeds, and joined by concatenating them in seed order.

    uv run create_synthetic_records.py --output-dir ./synthetic --split val --first-seed 1000000 --count 5000
    uv run create_synthetic_records.py --output-dir ./synthetic --split train-00 --first-seed 0 --count 50000
"""

import argparse
import json
import multiprocessing
from pathlib import Path

import synthetic

#: The size of the drawing the records' points are placed in, in pixels.
SIZE = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The records of synthetic floor plans, by seed.")
    parser.add_argument("--output-dir", type=Path, default=Path("synthetic"))
    parser.add_argument("--split", required=True, help="The name of the file: {split}_labels.jsonl.")
    parser.add_argument("--first-seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def record(seed: int) -> str:
    return json.dumps({"seed": seed, **synthetic.record_of(synthetic.generate(seed), SIZE)})


def main() -> None:
    args = parse_args()
    target = args.output_dir / f"{args.split}_labels.jsonl"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} exists. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.count)
    # Written to a partial file first, so that a file under its own name is always whole.
    partial = target.with_suffix(".jsonl.part")
    with multiprocessing.Pool(args.workers) as pool, open(partial, "w", encoding="utf-8") as out:
        for done, line in enumerate(pool.imap(record, seeds, chunksize=16), 1):
            out.write(line + "\n")
            if done % 5000 == 0 or done == args.count:
                print(f"{args.split}: {done} of {args.count}", flush=True)
    partial.rename(target)


if __name__ == "__main__":
    main()
