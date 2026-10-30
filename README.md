# synthetic-floorplans-for-transcription

Synthetic floor plans for transcription: per plan, the drawing a model reads and the record of its
walls, doors, windows and rooms it should write, and which walls are joined, which wall each opening
stands in and which rooms each wall and opening borders. Every plan is generated from its seed and keeps the rules a plan must -- walls horizontal
or vertical, exactly one wall between two rooms, every room reachable and opening off the living
room, windows only in outside walls -- and the drawing is drawn from the record alone. What a plan
holds and how it is made is in [dataset_cards/synthetic.md](dataset_cards/synthetic.md), which each
build copies into its folder as `README.md`.

- `synthetic.py` generates a plan from its seed, checks it and writes its record;
- `draw.py` draws a record.

## Browse

Sample plans by seed in a web UI, each drawn from its record with the record over it; the rooms it
was laid out from can be shown too.

```
uv run view_synthetic.py
```

## Build

The published dataset -- 450,000 training plans in nine parts, 5,000 each for validation and test,
images at 512 px -- from its seeds:

```
./build_synthetic.sh ./synthetic
```

Generating is what takes the time, about 75,000 plans an hour on ten cores; drawing is a few minutes
for all of them. The steps on their own:

```
uv run create_synthetic_records.py --output-dir ./synthetic --split val --first-seed 1000000 --count 5000
uv run draw_records.py --dataset-dir ./synthetic --split val                # --size 256 for smaller images
uv run preview_records.py --dataset-dir ./synthetic --split train-00 --output ./synthetic/train_examples_8x16.png
uv run describe_records.py --dataset-dir ./synthetic --card dataset_cards/synthetic.md
```

## Compared with ResPlan

How the plans compare with the real ones of [ResPlan](https://github.com/m-agour/ResPlan): outline
corners, and per kind of room how many doors it has.

```
./download_resplan.sh
uv run compare_resplan.py --resplan ./source/ResPlan.pkl
```
