"""How the synthetic plans compare with ResPlan's (https://github.com/m-agour/ResPlan): how many
corners their outlines have, and per kind of room how many there are and how many doors each has --
a room with one door is a dead end, one with more is walked through.

In ResPlan a door joins the rooms its rectangle touches; its outline is the rooms and walls united,
closed over gaps narrower than a wall.

    ./download_resplan.sh
    uv run compare_resplan.py --resplan ./source/ResPlan.pkl
"""

import argparse
import multiprocessing
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

import synthetic

RESPLAN_ROOMS = ("living", "kitchen", "bedroom", "bathroom", "storage", "balcony", "stair")


def parts(geometry):
    if geometry is None or geometry.is_empty:
        return []
    return list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]


def resplan_plan(plan: dict):
    thickness = plan["wall_depth"]
    rooms = [(kind, polygon) for kind in RESPLAN_ROOMS for polygon in parts(plan.get(kind))]
    outline = unary_union([polygon for _, polygon in rooms] + parts(plan["wall"]))
    outline = outline.buffer(thickness * 0.6).buffer(-thickness * 0.6).simplify(thickness * 0.3)
    corners = sum(len(polygon.exterior.coords) - 1 for polygon in parts(outline))
    doors = Counter()
    for key in ("door", "front_door"):
        for door in parts(plan.get(key)):
            for at, (_, polygon) in enumerate(rooms):
                doors[at] += polygon.distance(door) < thickness * 0.6
    return corners, [(kind, doors[at]) for at, (kind, _) in enumerate(rooms)]


def synthetic_plan(seed: int):
    plan = synthetic.generate(seed)
    # The outline turns at every grid corner where one or three of its four cells are inside, and twice
    # where two inside cells meet only at the corner.
    inside = np.pad(plan.rooms >= 0, 1).astype(int)
    around = inside[:-1, :-1] + inside[1:, :-1] + inside[:-1, 1:] + inside[1:, 1:]
    crossed = (inside[:-1, :-1] == inside[1:, 1:]) & (inside[1:, :-1] == inside[:-1, 1:]) & (inside[:-1, :-1] != inside[1:, :-1])
    corners = int(((around == 1) | (around == 3)).sum() + 2 * crossed.sum())
    doors = Counter(room for opening in plan.openings if opening.kind != "window"
                    for room in synthetic._joins(plan.rooms, opening) if room >= 0)
    return corners, [(kind, doors[room]) for room, kind in enumerate(plan.kinds)]


def report(name: str, plans) -> None:
    corners = np.array([c for c, _ in plans])
    print(f"{name}: {len(plans)} plans; outline corners median {np.median(corners):.0f}, "
          f"10th to 90th percentile {np.percentile(corners, 10):.0f} to {np.percentile(corners, 90):.0f}, "
          f"plain rectangles {np.mean(corners == 4):.0%}")
    doors = defaultdict(list)
    for _, rooms in plans:
        for kind, count in rooms:
            doors[kind].append(count)
    for kind, counts in sorted(doors.items(), key=lambda item: -len(item[1])):
        counts = np.array(counts)
        print(f"   {kind:9s} {len(counts) / len(plans):4.2f} per plan, doors {counts.mean():.2f}: "
              f"none {np.mean(counts == 0):4.0%}, one {np.mean(counts == 1):4.0%}, two or more {np.mean(counts >= 2):4.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the synthetic plans with ResPlan's.")
    parser.add_argument("--resplan", type=Path, default=Path("source/ResPlan.pkl"))
    parser.add_argument("--plans", type=int, default=400, help="How many plans of each to compare.")
    args = parser.parse_args()
    resplan = pickle.load(open(args.resplan, "rb"))
    report("ResPlan", [resplan_plan(plan) for plan in resplan[:: max(1, len(resplan) // args.plans)]])
    with multiprocessing.Pool() as pool:
        report("synthetic", pool.map(synthetic_plan, range(args.plans)))


if __name__ == "__main__":
    main()
