---
pretty_name: Synthetic floor plans for transcription
license: cc-by-4.0
size_categories:
  - 100K<n<1M
tags:
  - floor-plans
  - architecture
  - document-transcription
  - synthetic
---

# synthetic-floorplans-for-transcription

Floor plans as a transcription task: each plan as the image a model reads, and the record of its
walls, doors, windows and rooms that the model should write. The plans are generated, so every record is
exact, and the image is drawn from the record alone, so the record is all the image shows.

![The first 64 training plans, each image next to its record](train_examples_8x16.png)

*The first 64 training plans, each image next to its record drawn alone.*

## How a plan is made

A plan lives on a grid of 10 cm cells; each cell belongs to a room or to the outside, and a wall
stands wherever two neighbouring cells belong to different ones. So walls run horizontally or
vertically, rooms never overlap, and between two rooms stands exactly one wall.

A flat is organised around its living room, as real flats are: every room opens off it, and none is
reached through a bathroom, a kitchen or another bedroom.

1. **Programme.** 1 to 10 rooms, most often 6 to 9: a living room, then a bathroom, a kitchen,
   bedrooms, further bathrooms, sometimes a storage room, and balconies. Each kind has a range of
   areas and a narrowest width.
2. **Footprint.** A rectangle of about the programme's area with wings added, corners notched and
   stretches of facade set back; never narrower than a room may be, and no wall shorter than 0.5 m.
3. **Rooms.** Carved from the footprint one by one: the kitchen and the bedrooms from the facade
   inwards, filling the stretch they are carved from or lining up with an end of it; storage and
   bathrooms wherever they meet the living room, or a bathroom from the corner of a bedroom as its
   ensuite. What remains is the living room, with arms reaching every room like a hallway.
4. **Balconies.** Built onto the facade of the living room or a bedroom.
5. **Doors.** One door into every room from the room it opens off — the living room, or its
   bedroom for an ensuite or a balcony — and the outside door from the living room. The doors
   between rooms form a tree, and only the living room and bedrooms have more than one.
6. **Windows.** Only in outside walls, by the room behind: a living room one per 3 m of facade, a
   bedroom or kitchen one or two, a bathroom often a small one, a balcony none.

A plan is kept only if it keeps every rule: every room one area and wide enough, no holes, every door
between the two rooms it joins, every room reachable from outside, the doors a tree walked through
only by the living room and bedrooms, windows and the outside door only in outside walls, openings
clear of corners and of each other, every opening in exactly one wall, and no wall shorter than
0.5 m. Otherwise its seed lays the plan out again; the programme stays the seed's.

## Image

`(N, 512, 512)` uint8, row index = y, black on white, in the order of the labels beside it. Drawn from the record alone (`draw.py`): every wall a bar 6 px thick reaching
half its thickness past its ends; a door or outside door a gap in it, closed across at each jamb; a
window the same gap outlined, with a line down its middle. Rooms are not drawn. The plan is centred and scaled to fill 90%
of the image, at most 40 px per metre.

## Record

```json
{"seed": 3,
 "nodes": [{"class": "outside wall", "points": [[x, y], [x, y]]},
           {"class": "inside wall", "points": [[x, y], [x, y]]},
           {"class": "door", "points": [[x, y], [x, y]]},
           {"class": "outside door", "points": [[x, y], [x, y]]},
           {"class": "window", "points": [[x, y], [x, y]]},
           {"class": "living"}, {"class": "bathroom"}, {"class": "bedroom"}],
 "relationships": [{"type": "joined", "nodes": [0, 7]}, {"type": "in", "nodes": [14, 1]},
                   {"type": "borders", "nodes": [3, 21]}]}
```

Points are in the pixels of the image, x from the left, y from the top. Every node but a room is a
horizontal or vertical segment read rightwards or downwards; a room has no points. The walls come
first, then the openings, then the rooms.

| class | points |
|---|---|
| outside wall | The ends of its middle line; the outside lies on one side of it. |
| inside wall | The ends of its middle line; rooms lie on both sides. |
| door | Its jambs, in an inside wall. |
| outside door | Its jambs, in an outside wall. |
| window | Its jambs, in an outside wall. |
| living, kitchen, bedroom, bathroom, storage, balcony | None: a room is its kind alone, and where it is follows from the walls that border it. |

A wall runs straight from a corner or junction to the next, through its doors and windows; where
another wall meets it, it ends.

- `joined` [wall, wall]: the two walls share an end. Lower index first.
- `in` [opening, wall]: the door, outside door or window stands in that wall.
- `borders` [wall or opening, room]: the room lies on one side of it. An inside wall or a door
  borders the two rooms on either side; an outside wall, an outside door or a window borders one,
  the outside on its other side. A wall ends wherever another meets it, so the same room lies along
  all of it on either side. Two rooms bordered by the same door are walked between through it.

## Splits and files

Every plan is generated from its seed, and every label holds it; the line of a plan in its split is
its seed less the split's first.

| split | plans | seeds | files |
|---|---|---|---|
| train | 450,000 | 0 – 449,999 | `train-00` … `train-08`: `_images.npy` and `_labels.jsonl`, 50,000 plans each, `train-0k` from seed 50,000 k |
| validation | 5,000 | 1,000,000 – 1,004,999 | `val_images.npy`, `val_labels.jsonl` |
| test | 5,000 | 2,000,000 – 2,004,999 | `test_images.npy`, `test_labels.jsonl` |

The training images come in parts so that no file passes 50 GB. `metadata.json` counts the classes and
relationships of each split.

## Regenerating

The records and images come from
[synthetic-floorplans-for-transcription](https://github.com/benikm91/synthetic-floorplans-for-transcription) (`synthetic.py`,
`draw.py`):

```
uv run create_synthetic_records.py --output-dir ./synthetic --split train-00 --first-seed 0 --count 50000
uv run draw_records.py --dataset-dir ./synthetic --split train-00
```

and the same for every part and split. A seed gives the same plan, record and image every time.
