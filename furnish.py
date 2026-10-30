"""Furniture for a plan, so that each room shows what it is: a bed and a wardrobe in a bedroom, a toilet
and a bath or shower in a bathroom, a counter with sink and hob in a kitchen, a sofa in the living
room, shelves in a storage room, a small table or plants on a balcony.

Furniture is not part of the record: it is what the image shows besides it. It is placed as it would
be in a real flat -- backed against walls, clear of the doors and of the way between them, nothing tall
in front of a window -- and drawn in one style per plan (`icons.py`). It is placed with random numbers
of its own, so that a seed's plan and record stay what they are without it.

A room is furnished on its cells, the plan's 10 cm grid. An item is a rectangle of cells with a back
and a front; looking for where an item goes, the room is turned four ways, so that the wall it backs
against is always above it.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

import synthetic
from synthetic import CELL, Plan

#: Which way an item's front points, by the step from its back to its front: down, left, up, right.
#: An item facing 1 is one facing 0 turned a quarter clockwise.
FACING = {(1, 0): 0, (0, -1): 1, (-1, 0): 2, (0, 1): 3}
#: Items that look the same turned half round, and turned any way: their facing is the first of those
#: that look alike, so that it can be read off the drawing.
HALF_TURN = {"dining", "bistro", "coffee table"}
ANY_TURN = {"plant"}
#: How far a drawn item keeps from the middle of a wall it stands against, in pixels of a 512 render:
#: half the wall's thickness and a little.
INSET = 4.5


def n(metres: float) -> int:
    """Metres in cells."""
    return int(round(metres / CELL))


@dataclass
class Item:
    """An item on rows `rect[0]:rect[2]` and columns `rect[1]:rect[3]` of the plan's cells."""
    kind: str
    rect: Tuple[int, int, int, int]
    facing: int
    extra: dict = field(default_factory=dict)


def _table(mask: np.ndarray) -> np.ndarray:
    return np.pad(mask.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def _all(table: np.ndarray, i0, j0, i1, j1) -> np.ndarray:
    """Whether the mask summed in `table` holds over every rectangle `[i0, i1) x [j0, j1)`; an empty one
    holds, one reaching off the mask does not."""
    h, w = table.shape[0] - 1, table.shape[1] - 1
    i0, j0, i1, j1 = (np.asarray(v) for v in (i0, j0, i1, j1))
    area = (i1 - i0) * (j1 - j0)
    inside = (i0 >= 0) & (j0 >= 0) & (i1 <= h) & (j1 <= w)
    a, b, c, d = np.clip(i0, 0, h), np.clip(j0, 0, w), np.clip(i1, 0, h), np.clip(j1, 0, w)
    total = table[c, d] - table[a, d] - table[c, b] + table[a, b]
    return (area <= 0) | (inside & (total == area))


class Frame:
    """A room turned `k` quarters counter-clockwise; `R`, `C` are the room's own cells under it."""

    def __init__(self, inside: np.ndarray, k: int):
        self.k = k
        self.inside = np.rot90(inside, k)
        rows, cols = np.indices(inside.shape)
        self.R, self.C = np.rot90(rows, k), np.rot90(cols, k)
        step = lambda di, dj: (int(self.R[di, dj] - self.R[0, 0]), int(self.C[di, dj] - self.C[0, 0]))
        #: The facing of an item whose front points down the frame, and of one whose front points right.
        self.down, self.right = FACING[step(1, 0)], FACING[step(0, 1)]

    def turn(self, mask: np.ndarray) -> np.ndarray:
        return np.rot90(mask, self.k)

    def rect(self, i0: int, j0: int, i1: int, j1: int) -> Optional[Tuple[int, int, int, int]]:
        """A rectangle of the frame as one of the room's own cells, cut to the room's window."""
        h, w = self.R.shape
        i0, j0, i1, j1 = max(0, i0), max(0, j0), min(h, i1), min(w, j1)
        if i0 >= i1 or j0 >= j1:
            return None
        r, c = self.R[i0:i1, j0:j1], self.C[i0:i1, j0:j1]
        return int(r.min()), int(c.min()), int(r.max()) + 1, int(c.max()) + 1


@dataclass
class Spot:
    """Where an item `w` cells wide and `d` deep may go: rows `i:i + d`, columns `j:j + w` of a frame,
    its back at the top."""
    frame: Frame
    i: int
    j: int
    w: int
    d: int
    #: Whether a wall stands beside the back's left end, and beside its right end.
    corner: Tuple[bool, bool] = (False, False)

    @property
    def rect(self):
        return self.frame.rect(self.i, self.j, self.i + self.d, self.j + self.w)

    @property
    def centre(self) -> Tuple[float, float]:
        r0, c0, r1, c1 = self.rect
        return (r0 + r1) / 2, (c0 + c1) / 2


class Space:
    """One room of a plan being furnished, on a window of the plan's cells one wider than the room."""

    def __init__(self, plan: Plan, room: int, rng: np.random.Generator, kitchen_door: Optional[Tuple[float, float]] = None):
        self.rng, self.kind = rng, plan.kinds[room]
        padded = np.pad(plan.rooms, 1, constant_values=-1)
        rows, cols = np.nonzero(padded == room)
        window = padded[rows.min() - 1:rows.max() + 2, cols.min() - 1:cols.max() + 2]
        self.offset = (int(rows.min()) - 2, int(cols.min()) - 2)
        self.inside = window == room
        shape = self.inside.shape
        self.solid, self.keep, self.tall_keep = np.zeros(shape, bool), np.zeros(shape, bool), np.zeros(shape, bool)
        self.window = np.zeros(shape, bool)
        self.doors: List[np.ndarray] = []
        self.door_centres: List[Tuple[float, float]] = []
        self.items: List[Item] = []
        self.frames = [Frame(self.inside, k) for k in range(4)]
        for opening in plan.openings:
            if room in synthetic._joins(plan.rooms, opening):
                self._opening(opening)
        self.area = self.inside.sum() * CELL ** 2
        #: Where the door to the kitchen is, for a dining table in the living room to stand near.
        self.kitchen_door = None if kitchen_door is None else (kitchen_door[0] - self.offset[0], kitchen_door[1] - self.offset[1])
        self.walk_before = self._walkable(self.solid)

    def _opening(self, opening: synthetic.Opening) -> None:
        """Keep the room before a door clear, on this room's side of it, and tall things from before a window."""
        a0 = int(math.floor(opening.low / CELL + 1e-6))
        a1 = int(math.ceil(opening.high / CELL - 1e-6))
        line = opening.line - (self.offset[0] if opening.horizontal else self.offset[1])
        b0, b1 = a0 - (self.offset[1] if opening.horizontal else self.offset[0]), a1 - (self.offset[1] if opening.horizontal else self.offset[0])
        inside = self.inside if opening.horizontal else self.inside.T
        before = inside[line - 1, b0:b1].all()

        def zone(depth: int, widen: int) -> np.ndarray:
            mask = np.zeros(inside.shape, bool)
            rows = slice(max(0, line - depth), line) if before else slice(line, line + depth)
            mask[rows, max(0, b0 - widen):b1 + widen] = True
            return (mask if opening.horizontal else mask.T) & self.inside

        if opening.kind == "window":
            self.window |= zone(1, 0)
            self.tall_keep |= zone(n(0.7), 1)
        else:
            # As deep as the door is wide, or the room a person needs before it, if less.
            swing = zone(min(b1 - b0, n(0.75)), 1)
            self.keep |= swing
            self.doors.append(swing)
            r, c = np.nonzero(swing)
            self.door_centres.append((r.mean(), c.mean()))

    # --- Where things fit -------------------------------------------------------------------------

    def _walkable(self, solid: np.ndarray) -> bool:
        """Whether one can walk half a metre wide between all the room's doors around `solid`."""
        if len(self.doors) < 2:
            return True
        walk = ndimage.binary_erosion(self.inside & ~solid, np.ones((5, 5), bool))
        labels, _ = ndimage.label(walk)
        common = None
        for door in self.doors:
            here = set(np.unique(labels[door & walk]).tolist()) - {0}
            if here:
                common = here if common is None else common & here
        return common is None or bool(common)

    def spots(self, w: int, d: int, front: int = 0, sides: int = 0, one_side_wall: bool = False, tall: bool = False,
              wall: bool = True, ring: int = 0, strict: bool = False, from_row: int = 0) -> List[Spot]:
        """Every spot for an item `w` x `d` cells: on free cells, backed against a wall if `wall`, with
        `front` cells before it and `sides` beside it (from its row `from_row` on) clear of furniture
        -- and of everything kept clear if `strict` -- or, if `one_side_wall`, a wall on one side
        instead; with `ring` cells clear all around it if it stands free."""
        found = []
        for frame in (self.frames if wall else self.frames[:2]):
            inside = frame.inside
            h, wd = inside.shape
            if h < d or wd < w:
                continue
            blocked = frame.turn(self.solid | self.keep | (self.tall_keep if tall else False))
            free = _table(inside & ~blocked)
            clear = _table(inside & ~frame.turn(self.solid | (self.keep if strict else False)))
            i, j = np.mgrid[0:h - d + 1, 0:wd - w + 1]
            ok = _all(free, i, j, i + d, j + w)
            outside = _table(~inside)
            if wall:
                above = np.zeros_like(inside)
                above[1:] = inside[:-1]
                ok &= _all(_table(inside & ~above), i, j, i + 1, j + w)
            if front:
                ok &= _all(clear, i + d, j, i + d + front, j + w)
            if sides:
                left = _all(clear, i + from_row, j - sides, i + d, j)
                right = _all(clear, i + from_row, j + w, i + d, j + w + sides)
                if one_side_wall:
                    left_wall, right_wall = _all(outside, i, j - 1, i + d, j), _all(outside, i, j + w, i + d, j + w + 1)
                    ok &= (left | left_wall) & (right | right_wall) & (left | right)
                else:
                    ok &= left & right
            if ring:
                ok &= _all(clear, i - ring, j - ring, i + d + ring, j + w + ring)
            at_left, at_right = _all(outside, i, j - 1, i + 1, j), _all(outside, i, j + w, i + 1, j + w + 1)
            for a, b in zip(*np.nonzero(ok)):
                found.append(Spot(frame, int(a), int(b), w, d, (bool(at_left[a, b]), bool(at_right[a, b]))))
        return found

    def pick(self, spots: List[Spot], score: Callable[[Spot], float], noise: float = 0.4) -> Optional[Spot]:
        """The best of `spots` by `score`, a little at random, that leaves the way between the doors."""
        if not spots:
            return None
        if len(spots) > 500:
            spots = [spots[at] for at in self.rng.choice(len(spots), 500, replace=False)]
        scores = np.array([score(s) for s in spots]) + self.rng.normal(0, noise, len(spots))
        for at in np.argsort(-scores)[:16]:
            if scores[at] < -50:
                break
            if self.fits(spots[at].rect):
                return spots[at]
        return None

    def fits(self, rect, solid: bool = True) -> bool:
        if not self.walk_before or not solid:
            return True
        return self._walkable(self.solid | self._mask(rect))

    def _mask(self, rect) -> np.ndarray:
        mask = np.zeros(self.inside.shape, bool)
        if rect:
            mask[rect[0]:rect[2], rect[1]:rect[3]] = True
        return mask & self.inside

    def free(self, frame: Frame, i0, j0, i1, j1, tall: bool = False, against: bool = False, keep: bool = True) -> bool:
        """Whether a rectangle of a frame is free cells of the room -- clear of what is kept clear too, if
        `keep` -- and backed by a wall if `against`."""
        h, w = frame.inside.shape
        if i0 < 0 or j0 < 0 or i1 > h or j1 > w or i0 >= i1 or j0 >= j1:
            return False
        blocked = frame.turn(self.solid | (self.keep if keep else False) | (self.tall_keep if tall else False))
        if not (frame.inside[i0:i1, j0:j1] & ~blocked[i0:i1, j0:j1]).all():
            return False
        return not against or (i0 > 0 and not frame.inside[i0 - 1, j0:j1].any())

    def put(self, kind: str, frame: Frame, i0: int, j0: int, i1: int, j1: int, facing: Optional[int] = None,
            solid: bool = True, keep: Tuple[Tuple[int, int, int, int], ...] = (), **extra) -> Item:
        """An item on a rectangle of a frame; its front down the frame unless `facing` says otherwise."""
        rect = frame.rect(i0, j0, i1, j1)
        if solid:
            self.solid |= self._mask(rect)
        for zone in keep:
            self.keep |= self._mask(frame.rect(*zone))
        item = Item(kind, rect, frame.down if facing is None else facing, extra)
        self.items.append(item)
        return item

    def place(self, kind: str, spot: Spot, front: int = 0, sides: int = 0, from_row: int = 0, **extra) -> Item:
        i, j, w, d = spot.i, spot.j, spot.w, spot.d
        keep = [(i + d, j, i + d + front, j + w)] if front else []
        if sides:
            keep += [(i + from_row, j - sides, i + d, j), (i + from_row, j + w, i + d, j + w + sides)]
        return self.put(kind, spot.frame, i, j, i + d, j + w, keep=tuple(keep), **extra)

    # --- What a spot is like ---------------------------------------------------------------------

    def door_distance(self, spot: Spot) -> float:
        """Metres from the middle of a spot to the nearest door."""
        r, c = spot.centre
        return min((math.hypot(r - dr, c - dc) * CELL for dr, dc in self.door_centres), default=5.0)

    def under_window(self, spot: Spot) -> float:
        """How much of the spot's back is under a window, 0 to 1."""
        window = spot.frame.turn(self.window)
        return float(window[spot.i, spot.j:spot.j + spot.w].mean())

    def depth_before(self, spot: Spot) -> int:
        """Cells from the spot's front to the wall across, at its middle."""
        inside, col = spot.frame.inside, spot.j + spot.w // 2
        row = spot.i + spot.d
        while row < inside.shape[0] and inside[row, col]:
            row += 1
        return row - spot.i - spot.d


# --- Rooms ---------------------------------------------------------------------------------------

def _bedroom(space: Space) -> None:
    rng = space.rng
    double = rng.random() < (0.85 if space.area >= 10 else 0.6)
    bed = None
    if double:
        for width in rng.permutation([1.4, 1.6, 1.6, 1.8])[:3].tolist() + [1.4]:
            spots = space.spots(n(width), n(2.0), front=n(0.6), sides=n(0.55), from_row=n(0.45))
            bed = space.pick(spots, lambda s: 0.6 * space.door_distance(s) - 0.8 * space.under_window(s)
                             + 0.6 * (space.depth_before(s) > n(0.9)))
            if bed:
                space.place("bed", bed, front=n(0.6), sides=n(0.55), from_row=n(0.45), size="double")
                break
    if bed is None:
        double = False
        # A single bed, one long side against a wall if need be; tighter in a small room.
        for clear in (0.6, 0.45):
            spots = space.spots(n(0.9), n(2.0), front=n(clear), sides=n(clear), one_side_wall=True, from_row=n(0.45))
            bed = space.pick(spots, lambda s: 0.5 * space.door_distance(s) + 1.5 * any(s.corner))
            if bed:
                space.place("bed", bed, front=n(clear), sides=n(clear), from_row=n(0.45), size="single")
                break
        if bed is None:
            return
    # Nightstands at the head, on each open side.
    f, i, j, w = bed.frame, bed.i, bed.j, bed.w
    stand = n(rng.choice([0.4, 0.45, 0.5]))
    for j0 in (j - stand, j + w):
        if space.free(f, i, j0, i + n(0.4), j0 + stand, against=True) and (double or rng.random() < 0.7):
            space.put("nightstand", f, i, j0, i + n(0.4), j0 + stand)
    # A wardrobe, as long as the room allows.
    for length in (2.4, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0):
        spots = space.spots(n(length), n(0.6), front=n(0.75), tall=True)
        spot = space.pick(spots, lambda s: 1.2 * any(s.corner) + 0.3 * space.door_distance(s))
        if spot:
            space.place("wardrobe", spot, front=n(0.75))
            break
    # A desk under a window in a room with a single bed, or now and then in a large one.
    if not double or (space.area > 12 and rng.random() < 0.35):
        width = rng.choice([1.0, 1.2, 1.4])
        spots = space.spots(n(width), n(0.6), front=n(0.8))
        spot = space.pick(spots, lambda s: 2.0 * space.under_window(s) + 0.5 * any(s.corner))
        if spot:
            space.place("desk", spot, front=n(0.8))


def _counter(space: Space, longest: float, shortest: float) -> bool:
    """A kitchen counter along a wall -- with a leg round a corner if it is short -- and its sink, hob
    and fridge."""
    rng = space.rng
    depth, front = n(0.6), n(0.9)
    run = None
    for length in np.arange(longest, shortest - 1e-6, -0.2):
        spots = space.spots(n(length), depth, front=front)
        run = space.pick(spots, lambda s: 1.0 * sum(s.corner) + 1.5 * (space.under_window(s) > 0)
                         - 0.4 * (space.door_distance(s) < 1.0))
        if run:
            break
    if run is None:
        return False
    f, i, j, w = run.frame, run.i, run.j, run.w
    space.solid |= space._mask(run.rect)
    # The leg round a corner, if the run is short and a wall meets it there: backed against that wall,
    # its end against the run's front.
    leg = None
    if any(run.corner) and (w < n(3.2) or rng.random() < 0.3):
        for length in np.arange(2.4, 1.0, -0.2):
            for spot in space.spots(n(length), depth, front=front):
                if (spot.frame.k - f.k) % 2 == 0:
                    continue
                r0, c0, r1, c1 = spot.rect
                cr0, cc0, cr1, cc1 = run.rect
                beside = (r0 < cr1 and cr0 < r1 and (c1 == cc0 or c0 == cc1)) or (c0 < cc1 and cc0 < c1 and (r1 == cr0 or r0 == cr1))
                if beside and space.fits(spot.rect):
                    leg = spot
                    break
            if leg:
                break
    space.keep |= space._mask(f.rect(i + depth, j, i + depth + front, j + w))
    # The leg meets the run at one end; the fridge goes at the other, unless before a window.
    leg_end = None
    if leg:
        r, c = leg.centre
        near_left = f.rect(i, j, i + depth, j + 1)
        near_right = f.rect(i, j + w - 1, i + depth, j + w)
        d_left = abs(r - (near_left[0] + near_left[2]) / 2) + abs(c - (near_left[1] + near_left[3]) / 2)
        d_right = abs(r - (near_right[0] + near_right[2]) / 2) + abs(c - (near_right[1] + near_right[3]) / 2)
        leg_end = "left" if d_left < d_right else "right"
        space.place("counter", leg, front=front)
    top_j, top_w = j, w
    if w >= n(1.8):
        ends = [("left", j), ("right", j + w - depth)]
        if run.corner[0] and not run.corner[1]:
            ends.reverse()
        for side, start in ends:
            if side != leg_end and not f.turn(space.tall_keep)[i:i + depth, start:start + depth].any():
                space.put("fridge", f, i, start, i + depth, start + depth)
                top_j, top_w = (j + depth, w - depth) if side == "left" else (j, w - depth)
                break
    space.put("counter", f, i, top_j, i + depth, top_j + top_w)
    # The sink under a window if there is one, the hob apart from it.
    window = f.turn(space.window)[i, top_j:top_j + top_w]
    sink_w = n(rng.choice([0.6, 0.8, 0.8, 1.0]))
    if top_w < sink_w + n(0.2):
        return True
    if window.any():
        centre = int(np.flatnonzero(window).mean())
        start = int(np.clip(top_j + centre - sink_w // 2, top_j, top_j + top_w - sink_w))
    else:
        start = top_j + int(rng.integers(0, top_w - sink_w + 1))
    space.put("sink", f, i, start, i + depth, start + sink_w, solid=False)
    hob_w = n(0.6)
    places = [q for q in range(top_j + n(0.2), top_j + top_w - hob_w - n(0.2) + 1)
              if q + hob_w + n(0.3) <= start or q >= start + sink_w + n(0.3)]
    if places:
        q = places[int(rng.integers(len(places)))]
        space.put("hob", f, i, q, i + depth, q + hob_w, solid=False)
    elif leg is not None and leg.w >= hob_w + n(0.5):
        q = leg.j + (leg.w - hob_w) // 2
        space.put("hob", leg.frame, leg.i, q, leg.i + depth, q + hob_w, solid=False)
    return True


def _dining(space: Space, seats: int, ring: float = 0.35, near: Optional[Tuple[float, float]] = None) -> bool:
    """A table and its chairs, standing free."""
    shapes = {2: (0.7, 0.7, [1, 1, 0, 0]), 4: (1.2, 0.8, [2, 2, 0, 0]), 6: (1.6, 0.9, [3, 3, 0, 0])}
    while seats >= 2:
        width, depth, chairs = shapes.get(seats, shapes[4])
        w = n(width) + n(0.5) * (chairs[2] + chairs[3])
        d = n(depth) + n(0.5) * 2
        spots = space.spots(w, d, wall=False, ring=n(ring), strict=True)

        def score(s: Spot) -> float:
            r, c = s.centre
            close = -0.35 * math.hypot(r - near[0], c - near[1]) * CELL if near else 0.0
            return close + 0.2 * space.door_distance(s)

        spot = space.pick(spots, score)
        if spot:
            space.put("dining", spot.frame, spot.i, spot.j, spot.i + d, spot.j + w,
                      keep=((spot.i - n(ring), spot.j - n(ring), spot.i + d + n(ring), spot.j + w + n(ring)),),
                      chairs=chairs, table=[round(width, 2), round(depth, 2)])
            return True
        seats = {6: 4, 4: 2}.get(seats, 0)
    return False


def _living(space: Space, kitchenette: bool) -> None:
    rng = space.rng
    if kitchenette:
        _counter(space, 3.0, 1.6)
    sofa = None
    # Its front kept clear of the doors' room if it can be, else only of furniture.
    tries = [(length, True) for length in rng.permutation([2.0, 2.2, 2.4]).tolist() + [1.8, 1.6]] + [(1.8, False), (1.6, False)]
    for length, strict in tries:
        spots = space.spots(n(length), n(0.9), front=n(1.1), strict=strict)
        sofa = space.pick(spots, lambda s: 2.0 * (n(2.2) <= space.depth_before(s) <= n(4.5)) + 0.4 * space.door_distance(s)
                          - 0.3 * space.under_window(s))
        if sofa:
            break
    if sofa:
        f, i, j, w, d = sofa.frame, sofa.i, sofa.j, sofa.w, sofa.d
        space.put("sofa", f, i, j, i + d, j + w)
        table_w = max(n(0.8), int(w * rng.uniform(0.45, 0.6)))
        table_d = n(rng.choice([0.5, 0.6]))
        t0 = j + (w - table_w) // 2
        gap = n(0.4)
        table = space.free(f, i + d + gap, t0, i + d + gap + table_d, t0 + table_w)
        if table:
            space.put("coffee table", f, i + d + gap, t0, i + d + gap + table_d, t0 + table_w)
        space.keep |= space._mask(f.rect(i + d, j, i + d + gap + table_d + n(0.4), j + w))
        # The television on the wall across, if it is in reach.
        across = space.depth_before(sofa)
        if n(2.2) <= across <= n(4.5):
            tv_w, tv_d = n(rng.choice([1.2, 1.4, 1.6, 1.8])), n(0.45)
            row = i + d + across
            c0 = j + (w - tv_w) // 2
            window = f.turn(space.window)
            if space.free(f, row - tv_d, c0, row, c0 + tv_w) and row < f.inside.shape[0] \
                    and not f.inside[row, c0:c0 + tv_w].any() and not window[row - 1, c0:c0 + tv_w].any():
                space.put("tv", f, row - tv_d, c0, row, c0 + tv_w, facing=(f.down + 2) % 4,
                          keep=((row - tv_d - n(0.6), c0, row - tv_d, c0 + tv_w),))
        # An armchair beside the coffee table, now and then.
        if table and rng.random() < 0.5:
            size = n(0.8)
            r0 = i + d + gap + table_d // 2 - size // 2
            for c0, facing in ((j - size - n(0.1), f.right), (j + w + n(0.1), (f.right + 2) % 4)):
                if space.free(f, r0, c0, r0 + size, c0 + size, keep=False) and space.fits(f.rect(r0, c0, r0 + size, c0 + size)):
                    space.put("armchair", f, r0, c0, r0 + size, c0 + size, facing=facing)
                    break
    seats = int(rng.choice([4, 4, 6, 6, 2])) if space.area > 20 else int(rng.choice([2, 4]))
    near = None
    if kitchenette:
        counters = [it for it in space.items if it.kind == "counter"]
        if counters:
            r0, c0, r1, c1 = counters[0].rect
            near = ((r0 + r1) / 2, (c0 + c1) / 2)
    elif space.kitchen_door is not None:
        near = space.kitchen_door
    if kitchenette or rng.random() < 0.85:
        _dining(space, seats, near=near)
    # A sideboard or bookcase, and plants in corners.
    if rng.random() < 0.6:
        length = rng.choice([1.2, 1.6, 2.0])
        spot = space.pick(space.spots(n(length), n(0.4), front=n(0.6), tall=True), lambda s: 0.5 * space.door_distance(s))
        if spot:
            space.place("shelves", spot, front=n(0.6))
    _plants(space, int(rng.choice([0, 1, 1, 2, 3])))


def _plants(space: Space, count: int) -> None:
    for _ in range(count):
        size = n(space.rng.choice([0.35, 0.45, 0.55]))
        spot = space.pick(space.spots(size, size, front=n(0.2)), lambda s: 3.0 * any(s.corner) - 100 * (not any(s.corner)))
        if spot:
            space.place("plant", spot)


def _kitchen(space: Space) -> None:
    longest = min(4.2, max(1.8, math.sqrt(space.area) * 1.3))
    _counter(space, longest, 1.6)
    if space.area > 8 and space.rng.random() < 0.7:
        _dining(space, int(space.rng.choice([2, 2, 4])), ring=0.3)


def _bathroom(space: Space) -> None:
    """A bath or a shower, a toilet and a basin -- the toilet first now and then, so that a narrow room
    finds room for all three -- and now and then a washing machine."""
    rng = space.rng
    order = [_wet, _toilet, _basin] if rng.random() < 0.5 else [_toilet, _wet, _basin]
    for step in order:
        step(space)
    if rng.random() < 0.35:
        spot = space.pick(space.spots(n(0.6), n(0.6), front=n(0.5)), lambda s: any(s.corner) * 1.0)
        if spot:
            space.place("washer", spot, front=n(0.5))


def _wet(space: Space) -> None:
    rng = space.rng
    if rng.random() < 0.6:
        for length in (rng.choice([1.6, 1.7, 1.8]), 1.5):
            spots = space.spots(n(length), n(0.75), front=n(0.55))
            spot = space.pick(spots, lambda s: 2.0 * sum(s.corner) - 0.3 * space.under_window(s))
            if spot:
                space.place("bathtub", spot, front=n(0.55))
                return
    for size in (rng.choice([0.8, 0.9, 0.9, 1.0]), 0.8):
        spots = space.spots(n(size), n(size), front=n(0.5))
        spot = space.pick(spots, lambda s: 3.0 * any(s.corner) + 1.0 * all(s.corner))
        if spot:
            space.place("shower", spot, front=n(0.5))
            return


def _toilet(space: Space) -> None:
    for width, front in ((0.7, 0.5), (0.6, 0.45)):
        spot = space.pick(space.spots(n(width), n(0.7), front=n(front)),
                          lambda s: 0.2 * space.door_distance(s) + 0.5 * any(s.corner))
        if spot:
            space.place("toilet", spot, front=n(front))
            return


def _basin(space: Space) -> None:
    for width, depth in ((space.rng.choice([0.6, 0.6, 0.8, 1.0, 1.2]), 0.5), (0.6, 0.5), (0.5, 0.4)):
        spot = space.pick(space.spots(n(width), n(depth), front=n(0.55)), lambda s: -0.2 * space.door_distance(s))
        if spot:
            space.place("basin", spot, front=n(0.55), double=bool(width >= 1.2))
            return


def _storage(space: Space) -> None:
    for _ in range(int(space.rng.choice([1, 2, 2, 3]))):
        for length in (2.4, 2.0, 1.6, 1.2, 0.8):
            spot = space.pick(space.spots(n(length), n(0.4), front=n(0.6), tall=True), lambda s: any(s.corner) * 1.0)
            if spot:
                space.place("shelves", spot, front=n(0.6))
                break


def _balcony(space: Space) -> None:
    rng = space.rng
    if rng.random() < 0.75:
        spots = space.spots(n(0.7), n(1.5), wall=False, ring=n(0.15), strict=True)
        spot = space.pick(spots, lambda s: 0.3 * space.door_distance(s))
        if spot:
            space.put("bistro", spot.frame, spot.i, spot.j, spot.i + spot.d, spot.j + spot.w)
    _plants(space, int(rng.choice([0, 1, 2])))


#: What a room must show to be told for what it is: one item of each group.
NEEDS = {"living": [{"sofa"}], "kitchen": [{"counter"}, {"sink"}, {"hob"}], "bedroom": [{"bed"}],
         "bathroom": [{"toilet"}, {"bathtub", "shower"}, {"basin"}], "storage": [{"shelves"}], "balcony": []}
#: How often a room is furnished anew, at most, until it shows all it must.
ATTEMPTS = 10


def furnish(plan: Plan, rng: np.random.Generator) -> List[Item]:
    """The furniture of every room, in plan cells."""
    items = []
    kitchen = plan.kinds.index("kitchen") if "kitchen" in plan.kinds else None
    recipes = {"living": lambda s: _living(s, kitchenette=kitchen is None), "kitchen": _kitchen, "bedroom": _bedroom,
               "bathroom": _bathroom, "storage": _storage, "balcony": _balcony}
    for room, kind in enumerate(plan.kinds):
        kitchen_door = None
        if kind == "living" and kitchen is not None:
            for opening in plan.openings:
                if opening.kind == "door" and synthetic._joins(plan.rooms, opening) == {room, kitchen}:
                    along = (opening.low + opening.high) / 2 / CELL
                    kitchen_door = (opening.line, along) if opening.horizontal else (along, opening.line)
        best = None
        for _ in range(ATTEMPTS):
            space = Space(plan, room, rng, kitchen_door)
            recipes[kind](space)
            have = {item.kind for item in space.items}
            shown = sum(bool(need & have) for need in NEEDS[kind])
            if best is None or shown > best[0]:
                best = (shown, space)
            if shown == len(NEEDS[kind]):
                break
        space = best[1]
        dr, dc = space.offset
        for item in space.items:
            r0, c0, r1, c1 = item.rect
            items.append(Item(item.kind, (r0 + dr, c0 + dc, r1 + dr, c1 + dc), item.facing, item.extra))
    return items


def furnishing_of(plan: Plan, seed: int, size: int) -> dict:
    """The furniture of the plan of `seed`, in the pixels of a render `size` wide: each item's box, the
    way it faces, and what else its drawing needs; and the look the plan is drawn in."""
    rng = np.random.default_rng([seed, 1])
    items = furnish(plan, rng)
    scale, origin = synthetic.frame(plan, size)
    cell = CELL * scale
    inset = INSET * size / 512
    rooms = np.pad(plan.rooms, 1, constant_values=-1)
    out = []
    for item in items:
        r0, c0, r1, c1 = item.rect
        room = rooms[r0 + 1, c0 + 1]
        # Kept off the walls it stands against, which the render draws over the cells' edges.
        wall = [not (rooms[r0, c0 + 1:c1 + 1] == room).any(), not (rooms[r1 + 1, c0 + 1:c1 + 1] == room).any(),
                not (rooms[r0 + 1:r1 + 1, c0] == room).any(), not (rooms[r0 + 1:r1 + 1, c1 + 1] == room).any()]
        x0, y0 = origin[0] + c0 * cell + inset * wall[2], origin[1] + r0 * cell + inset * wall[0]
        x1, y1 = origin[0] + c1 * cell - inset * wall[3], origin[1] + r1 * cell - inset * wall[1]
        facing = 0 if item.kind in ANY_TURN else item.facing % 2 if item.kind in HALF_TURN else item.facing
        out.append({"kind": item.kind, "box": [round(float(v), 2) for v in (x0, y0, x1, y1)], "facing": facing,
                    **{k: (v.item() if hasattr(v, "item") else v) for k, v in item.extra.items()}})
    return {"look": int(rng.integers(2 ** 31)), "scale": round(float(scale), 3), "items": out}
