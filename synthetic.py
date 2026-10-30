"""Synthetic floor plans: rooms, the walls between them, their doors and their windows, generated so
that every plan holds what a plan must.

A plan lives on a grid of 10 cm cells; every cell belongs to a room or to the outside, and a wall
stands on every grid line between two cells that belong to different ones. So walls are horizontal or
vertical, rooms never overlap, and between two rooms there is exactly one wall.

A flat is organised around its living room, as real flats are: the rooms open off it, and none is
reached through a bathroom or another bedroom. A plan is made in steps, each of which may give up;
the plan is then laid out again from the seed's next attempt, its programme kept:

1. a programme of 1 to 10 rooms of realistic kinds and areas;
2. a footprint of about their area: a rectangle with wings added, corners notched and stretches of
   facade set back;
3. the rooms carved from it: bedrooms and the kitchen from the facade inwards, bathrooms and storage
   wherever they touch the living room, or a bathroom from the corner of a bedroom; what remains is the
   living room, which reaches every room;
4. balconies built onto the facade of the living room or a bedroom;
5. one door into every room from the room it opens off -- the living room, or for an ensuite bathroom
   or a balcony its bedroom -- and the door to the outside, from the living room;
6. windows in the outside walls, by the kind of room behind them.

`problems` checks every rule a plan must keep; a plan that breaks one is never returned.

The record names the walls -- outside or inside -- and the doors, windows and outside doors, each as
a segment, and the rooms, each by its kind alone; which walls are joined, which wall each opening
stands in, and which rooms each wall and opening borders. A wall runs straight
from a corner or junction to the next, through its doors and windows; where another wall meets it,
it ends, so walls are joined exactly where they share an end. The render is drawn from the record
alone (`draw.py`).
"""

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

#: Metres per grid cell.
CELL = 0.1

#: The area of a room of each kind, in square metres, and the narrowest it may be, in metres. The
#: living room's area is what the others leave of the footprint.
AREA = {"living": (16, 30), "kitchen": (7, 13), "bedroom": (9, 16), "bathroom": (3.5, 6.5), "storage": (2, 4), "balcony": (3, 8)}
NARROWEST = {"living": 1.2, "kitchen": 2.0, "bedroom": 2.6, "bathroom": 1.5, "storage": 1.2, "balcony": 1.2}
#: The living room's arms may be as narrow as a hallway, but most of it must be a room.
LIVING_CORE = 2.8

#: The width of a door into a room of each kind, in metres.
DOOR_WIDTH = {"bathroom": (0.7, 0.8), "storage": (0.6, 0.75), "balcony": (0.8, 1.2)}
DOOR_WIDTH_ELSE = (0.8, 0.9)
OUTSIDE_DOOR_WIDTH = (0.9, 1.1)
#: How wide the windows of a room of each kind are, in metres.
WINDOW_WIDTH = {"living": (1.4, 2.4), "kitchen": (0.9, 1.5), "bedroom": (1.0, 1.8), "bathroom": (0.5, 0.9), "storage": (0.5, 0.8)}

#: The least room an opening keeps from a corner or junction of its wall, and from another opening
#: on the same wall, in metres: clear of the walls the render draws, at its thickest.
CLEAR_OF_CORNER = 0.35
CLEAR_OF_OPENING = 0.3
#: The shortest a wall may be, in metres, so that no two junctions run together in the render.
SHORTEST_WALL = 0.5
#: Room left around the footprint for balconies, in metres.
MARGIN = 2.5

#: How often a plan has 1, 2, ... 10 rooms, balconies among them.
ROOM_COUNTS = np.array([1, 2, 3, 5, 8, 11, 13, 14, 12, 10])


class Retry(Exception):
    """A step that could not be taken: the plan is laid out again."""


@dataclass(frozen=True)
class Opening:
    """A door, window or outside door in the wall on grid line `line` -- along x if `horizontal`,
    else along y -- from `low` to `high` metres along it."""
    kind: str
    horizontal: bool
    line: int
    low: float
    high: float


@dataclass(frozen=True)
class Wall:
    """Wall on grid line `line` from cell `low` to cell `high` along it, between two corners or
    junctions; an `outside` wall has the outside on one side."""
    horizontal: bool
    line: int
    low: int
    high: int
    outside: bool


@dataclass
class Plan:
    #: Per cell, the room it belongs to, -1 for outside.
    rooms: np.ndarray
    kinds: List[str]
    openings: List[Opening]


# --- Reading a plan's grid ---------------------------------------------------------------------------

def sides(rooms: np.ndarray, horizontal: bool) -> Tuple[np.ndarray, np.ndarray]:
    """The room on either side of every grid edge: of the horizontal edges, `(H + 1, W)`, above and
    below; of the vertical ones, `(W + 1, H)`, left and right. Outside is -1."""
    padded = np.pad(rooms, 1, constant_values=-1)
    if horizontal:
        return padded[:-1, 1:-1], padded[1:, 1:-1]
    return padded[1:-1, :-1].T, padded[1:-1, 1:].T


def runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """The stretches `[start, end)` where `flags` holds."""
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def walls_of(rooms: np.ndarray) -> List[Wall]:
    """The walls of a plan: every straight run of wall, cut wherever another wall meets or crosses it."""
    stands = {horizontal: np.not_equal(*sides(rooms, horizontal)) for horizontal in (True, False)}
    walls = []
    for horizontal in (True, False):
        one, other = sides(rooms, horizontal)
        across = stands[not horizontal]          # [line across, cell along it]
        for line in range(one.shape[0]):
            outside = (one[line] < 0) | (other[line] < 0)
            for low, high in runs(stands[horizontal][line]):
                # Where a wall across meets this line, on either side of it, this wall ends.
                meets = [at for at in range(low + 1, high)
                         if (line > 0 and across[at, line - 1]) or (line < across.shape[1] and across[at, line])]
                stops = [low, *meets, high]
                walls += [Wall(horizontal, line, a, b, bool(outside[a])) for a, b in zip(stops[:-1], stops[1:])]
    return walls


def around(region: np.ndarray) -> Tuple[slice, slice]:
    """The rows and columns of `region`'s bounding box, one cell wider on every side the grid allows."""
    rows, cols = np.flatnonzero(region.any(1)), np.flatnonzero(region.any(0))
    if not len(rows):
        return slice(0, 0), slice(0, 0)
    return (slice(max(0, rows[0] - 1), rows[-1] + 2), slice(max(0, cols[0] - 1), cols[-1] + 2))


def stretches_between(rooms: np.ndarray, a: int, b: int) -> List[Tuple[bool, int, int, int]]:
    """Where rooms `a` and `b` meet (`b` may be -1, outside): `(horizontal, line, low, high)`.

    Read around room `a` only, every edge of which lies there; where that window stops short of the
    grid's border, the outside `sides` pads it with is next to no cell of `a`. The outside itself is
    read whole.
    """
    rows, cols = around(rooms == a) if a >= 0 else (slice(0, rooms.shape[0]), slice(0, rooms.shape[1]))
    window = rooms[rows, cols]
    found = []
    for horizontal in (True, False):
        one, other = sides(window, horizontal)
        meet = ((one == a) & (other == b)) | ((one == b) & (other == a))
        line_offset, along_offset = (rows.start, cols.start) if horizontal else (cols.start, rows.start)
        for line in np.flatnonzero(meet.any(axis=1)):
            found += [(horizontal, int(line) + line_offset, low + along_offset, high + along_offset) for low, high in runs(meet[line])]
    return found


def wide_enough(region: np.ndarray, metres: float) -> bool:
    """Whether every cell of `region` lies in a square `metres` wide that lies wholly in it."""
    region = region[around(region)]
    side = max(1, int(round(metres / CELL)))

    def window_sums(cells: np.ndarray) -> np.ndarray:
        """Per cell, the sum over the `side` × `side` window whose top left it is."""
        table = np.pad(cells.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]

    if region.shape[0] < side or region.shape[1] < side:
        return not region.any()
    # The top left corners of the squares that lie wholly in the region, and the cells they cover.
    corners = window_sums(region) == side * side
    padded = np.pad(corners, ((side - 1, side - 1), (side - 1, side - 1)))
    covered = window_sums(padded)[: region.shape[0], : region.shape[1]] > 0
    return not (region & ~covered).any()


def connected(region: np.ndarray) -> bool:
    return ndimage.label(region)[1] == 1


def door_room(stretches: List[Tuple[bool, int, int, int]], width: float) -> bool:
    """Whether a door `width` metres wide fits on one of the stretches, clear of their ends."""
    return any((high - low) * CELL >= width + 2 * CLEAR_OF_CORNER for _, _, low, high in stretches)


# --- Making a plan -----------------------------------------------------------------------------------

def programme(rng: np.random.Generator, count: int) -> List[str]:
    """The kinds of the rooms: the living room first, then what is carved from it."""
    if count == 1:
        return ["living"]
    kinds = ["living", "bathroom"]
    if count >= 3:
        kinds.append("kitchen" if count >= 4 or rng.random() < 0.3 else "bedroom")
    rest = count - len(kinds)
    balconies = min(rest - 1, int(rng.choice([0, 1, 2], p=[0.15, 0.45, 0.4]))) if rest >= 2 else 0
    rest -= balconies
    for _ in range(2):
        if rest >= 3 and rng.random() < 0.6:
            kinds.append("bathroom")
            rest -= 1
    if rest >= 4 and rng.random() < 0.25:
        kinds.append("storage")
        rest -= 1
    return kinds + ["bedroom"] * rest + ["balcony"] * balconies


def footprint(rng: np.random.Generator, area: float) -> np.ndarray:
    """A rectangle of about `area` square metres with wings added, corners notched and stretches of
    facade set back; nowhere narrower than a room may be. Placed with `MARGIN` around it."""
    aspect = rng.uniform(0.65, 1.6)
    width = math.sqrt(area * 0.85 * aspect)
    height = area * 0.85 / width
    margin = int(MARGIN / CELL)
    h, w = int(round(height / CELL)), int(round(width / CELL))
    grow = int(0.6 * max(h, w))
    cells = np.zeros((h + 2 * (grow + margin), w + 2 * (grow + margin)), bool)
    top, left = grow + margin, grow + margin
    cells[top:top + h, left:left + w] = True
    arm = int(round(2.8 / CELL))

    def kept(changed: np.ndarray) -> bool:
        return connected(changed) and wide_enough(changed, 2.8) and connected(np.pad(~changed, 1, constant_values=True)) \
            and all((wall.high - wall.low) * CELL >= SHORTEST_WALL for wall in walls_of(np.where(changed, 0, -1)))

    # Wings: a rectangle off a side of the main one, reaching out a third to a half of it.
    for _ in range(int(rng.choice([0, 1, 2], p=[0.35, 0.45, 0.2]))):
        along, out = rng.uniform(0.3, 0.7), rng.uniform(0.25, 0.55)
        side = int(rng.integers(4))
        changed = cells.copy()
        if side in (0, 1):                       # above or below
            length, depth = max(arm, int(w * along)), max(arm, int(h * out))
            start = left + int(rng.integers(0, max(1, w - length + 1)))
            rows = slice(top - depth, top) if side == 0 else slice(top + h, top + h + depth)
            changed[rows, start:start + length] = True
        else:                                    # left or right
            length, depth = max(arm, int(h * along)), max(arm, int(w * out))
            start = top + int(rng.integers(0, max(1, h - length + 1)))
            cols = slice(left - depth, left) if side == 2 else slice(left + w, left + w + depth)
            changed[start:start + length, cols] = True
        if kept(changed):
            cells = changed

    # Notches at corners and set-backs along the facade: rectangles taken from the outline inwards.
    for _ in range(int(rng.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.25, 0.2, 0.15]))):
        rows, cols = np.flatnonzero(cells.any(1)), np.flatnonzero(cells.any(0))
        notch = (rng.uniform(0.8, 3.5), rng.uniform(0.8, 3.5)) if rng.random() < 0.5 else (rng.uniform(0.6, 1.2), rng.uniform(1.5, 5.0))
        depth, length = (int(notch[0] / CELL), int(notch[1] / CELL)) if rng.random() < 0.5 else (int(notch[1] / CELL), int(notch[0] / CELL))
        r = int(rng.integers(rows[0], rows[-1] + 1))
        c = int(rng.integers(cols[0], cols[-1] + 1))
        changed = cells.copy()
        changed[r:r + depth, c:c + length] = False
        # Only a notch that opens to the outside, and leaves the rest whole.
        if changed.sum() < cells.sum() and kept(changed) and not (cells[r:r + depth, c:c + length]).all():
            cells = changed
    return cells


def carve(rng: np.random.Generator, rooms: np.ndarray, host: int, room: int, kind: str, area: float,
          facade: bool, keep_door: Tuple[int, ...]) -> Optional[np.ndarray]:
    """`rooms` with a rectangle of about `area` square metres taken from room `host` for the new room
    `room`: from the facade inwards if `facade`, else from where the host meets anything. The host must
    stay one area, wide enough, and keep room for a door to each room of `keep_door`; the new room must
    have room for a door to the host.

    A room fills the stretch it is carved from, or lines up with one end of it, before it is placed
    anywhere along it; it reaches as deep as its area asks, and no deeper than the host does.
    """
    narrowest = NARROWEST[kind]
    host_kind = "living" if host == 0 else "bedroom"
    door = DOOR_WIDTH.get(kind, DOOR_WIDTH_ELSE)[1]
    neighbours = [-1] if facade else sorted(set(np.unique(rooms).tolist()) - {host})
    anchors = [s for other in neighbours for s in stretches_between(rooms, host, other)]
    rng.shuffle(anchors)
    least = int(math.ceil(narrowest / CELL))
    for horizontal, line, low, high in anchors[:40]:
        if high - low < least:
            continue
        one, _ = sides(rooms, horizontal)
        host_before = one[line, low] == host
        wanted = int(round(rng.uniform(narrowest, max(narrowest, min((high - low) * CELL, math.sqrt(area * 1.8)))) / CELL))
        spans = [(low, high), (low, low + wanted), (high - wanted, high)]
        if high - low > wanted:
            spans.append((int(rng.integers(low, high - wanted + 1)), 0))
        hallway = int(math.ceil(NARROWEST[host_kind] / CELL))
        for start, stop in spans:
            stop = stop or start + wanted
            # Rather than leave the host a sliver beside the room, the room takes it.
            start = low if 0 < start - low < hallway else start
            stop = high if 0 < high - stop < hallway else stop
            length = stop - start
            if length < least or start < low or stop > high:
                continue
            # How deep the host reaches behind the span.
            reach = 0
            limit = rooms.shape[0 if horizontal else 1]
            while reach < int(8 / CELL):
                at = line - reach - 1 if host_before else line + reach
                if not 0 <= at < limit:
                    break
                strip = rooms[at, start:stop] if horizontal else rooms[start:stop, at]
                if not (strip == host).all():
                    break
                reach += 1
            depth = max(least, min(reach, int(round(area / (length * CELL) / CELL))))
            # Nor a sliver behind it: the room reaches back all the way, or stops a hallway short.
            if 0 < reach - depth < hallway:
                depth = reach if reach * length * CELL ** 2 <= 1.8 * area else reach - hallway
            if not least <= depth <= reach or not 0.6 * area <= length * depth * CELL ** 2 <= 1.8 * area \
                    or max(length, depth) > 2.4 * min(length, depth):
                continue
            across = slice(line - depth, line) if host_before else slice(line, line + depth)
            block = (across, slice(start, stop)) if horizontal else (slice(start, stop), across)
            changed = rooms.copy()
            changed[block] = room
            rest = changed == host
            if not (connected(rest) and wide_enough(rest, NARROWEST[host_kind]) and wide_enough(changed == room, narrowest)):
                continue
            if host == 0 and not (_has_core(rest) and door_room(stretches_between(changed, 0, -1), OUTSIDE_DOOR_WIDTH[1] + 0.15)):
                continue
            if not door_room(stretches_between(changed, room, host), door):
                continue
            if not all(door_room(stretches_between(changed, other, host), DOOR_WIDTH_ELSE[1]) for other in keep_door):
                continue
            return changed
    return None


def _has_core(living: np.ndarray) -> bool:
    """Whether most of the living room lies in squares `LIVING_CORE` wide: a room, not only hallway."""
    living = living[around(living)]
    side = int(round(LIVING_CORE / CELL))
    if living.shape[0] < side or living.shape[1] < side:
        return False
    table = np.pad(living.astype(np.int32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = table[side:, side:] - table[:-side, side:] - table[side:, :-side] + table[:-side, :-side]
    return bool((sums == side * side).sum() * 1.0 >= 0.05 * living.sum())


def generate(seed: int) -> Plan:
    """The plan of `seed`: its room programme, laid out by the first attempt that keeps every rule.

    The programme is drawn once, so that a programme hard to lay out is as common as any other.
    """
    rng = np.random.default_rng(seed)
    count = int(rng.choice(np.arange(1, 11), p=ROOM_COUNTS / ROOM_COUNTS.sum()))
    kinds = programme(rng, count)
    for _ in range(1000):
        try:
            plan = _attempt(rng, kinds)
        except Retry:
            continue
        if not problems(plan):
            return plan
    raise RuntimeError(f"no plan for seed {seed}")


def _attempt(rng: np.random.Generator, programme_kinds: List[str]) -> Plan:
    areas = {at: rng.uniform(*AREA[kind]) for at, kind in enumerate(programme_kinds)}
    inside = [at for at, kind in enumerate(programme_kinds) if kind != "balcony"]
    # A little more than the rooms ask for: a carved room may come out larger than its area.
    slack = 1.0 + 0.02 * len(inside)
    rooms = np.where(footprint(rng, slack * sum(areas[at] for at in inside)), 0, -1)
    kinds = ["living"]
    parents = {0: None}

    # Bedrooms and the kitchen from the facade, then storage and bathrooms wherever they meet the living room
    # or, as an ensuite, from a bedroom's corner.
    order = sorted((at for at in inside if at), key=lambda at: ("kitchen", "bedroom", "storage", "bathroom").index(programme_kinds[at]))
    def placed(rooms: np.ndarray, kinds: List[str], parents: Dict[int, Optional[int]], at: int):
        """The plan with the programme's room `at` carved into it, or None where it finds no room.

        A room that finds none gives the attempt up: the footprint seldom suits the programme then,
        and a new one is cheaper than undoing rooms carved before it."""
        kind = programme_kinds[at]
        room = len(kinds)
        bedrooms = [r for r, k in enumerate(kinds) if k == "bedroom"]
        ensuite = kind == "bathroom" and bedrooms and rng.random() < 0.5
        host = int(rng.choice(bedrooms)) if ensuite else 0
        others = tuple(r for r, p in parents.items() if p == host) if host == 0 else ()
        facade = kind in ("bedroom", "kitchen") or (kind == "bathroom" and rng.random() < 0.3)
        changed = carve(rng, rooms, host, room, kind, areas[at], facade, others)
        if changed is None and kind == "kitchen":
            changed = carve(rng, rooms, host, room, kind, areas[at], False, others)
        if changed is None and ensuite:
            host, others = 0, tuple(r for r, p in parents.items() if p == 0)
            changed = carve(rng, rooms, 0, room, kind, areas[at], False, others)
        if changed is None or (host != 0 and not door_room(stretches_between(changed, host, 0), DOOR_WIDTH_ELSE[1])):
            return None
        return changed, kinds + [kind], {**parents, room: host}

    for at in order:
        state = placed(rooms, kinds, parents, at)
        if state is None:
            raise Retry(f"no room for a {programme_kinds[at]}")
        rooms, kinds, parents = state

    living = rooms == 0
    if living.sum() * CELL ** 2 < AREA["living"][0] * 0.8 or not _has_core(living):
        raise Retry("living room too small")

    # Balconies on the facade of the living room or a bedroom.
    for _ in range(programme_kinds.count("balcony")):
        room = len(kinds)
        # As often off a bedroom as off the living room.
        hosts = [r for r, k in enumerate(kinds) if k == "bedroom"]
        rng.shuffle(hosts)
        hosts = hosts + [0] if rng.random() < 0.5 else [0] + hosts
        for host in hosts:
            changed = _balcony(rng, rooms, host, room)
            if changed is not None:
                rooms, kinds, parents[room] = changed, kinds + ["balcony"], host
                break
        else:
            raise Retry("no room for a balcony")

    rows, cols = np.flatnonzero((rooms >= 0).any(1)), np.flatnonzero((rooms >= 0).any(0))
    rooms = rooms[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    openings = _doors(rng, rooms, kinds, parents)
    openings += _windows(rng, rooms, kinds, openings)
    return Plan(rooms, kinds, openings)


def _balcony(rng: np.random.Generator, rooms: np.ndarray, host: int, room: int) -> Optional[np.ndarray]:
    """`rooms` with a balcony built onto a facade of room `host`, touching nothing else of the plan."""
    for horizontal, line, low, high in sorted(stretches_between(rooms, host, -1), key=lambda s: s[2] - s[3])[:4]:
        if (high - low) * CELL < 2.4 + 2 * CLEAR_OF_CORNER:
            continue
        one, _ = sides(rooms, horizontal)
        outward_before = one[line, low] == -1
        length = int(rng.uniform(2.4, min(5.0, (high - low) * CELL - 2 * CLEAR_OF_CORNER)) / CELL)
        depth = int(rng.uniform(1.2, 1.8) / CELL)
        start = int(rng.integers(low + int(CLEAR_OF_CORNER / CELL), high - length - int(CLEAR_OF_CORNER / CELL) + 1))
        across = slice(line - depth, line) if outward_before else slice(line, line + depth)
        block = (across, slice(start, start + length)) if horizontal else (slice(start, start + length), across)
        # The balcony and a ring of outside around it, so that it touches no other room.
        ring = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in block)
        if across.start < 1 or across.stop >= rooms.shape[0 if horizontal else 1]:
            continue
        if not (rooms[block] == -1).all() or set(np.unique(rooms[ring]).tolist()) - {-1, host}:
            continue
        changed = rooms.copy()
        changed[block] = room
        if connected(np.pad(changed < 0, 1, constant_values=True)):
            return changed
    return None


def _place(rng: np.random.Generator, stretch: Tuple[bool, int, int, int], width: float, margin: float,
           taken: List[Opening], kind: str, near_end: bool) -> Optional[Opening]:
    """An opening `width` metres wide on `stretch`, clear of its ends by `margin` and of the openings
    `taken`; near one end of it if `near_end`, else anywhere along it."""
    horizontal, line, low, high = stretch
    first, last = low * CELL + margin, high * CELL - margin - width
    if last < first:
        return None
    for _ in range(8):
        start = (first if rng.random() < 0.5 else last) if near_end else rng.uniform(first, last)
        candidate = Opening(kind, horizontal, line, round(start, 3), round(start + width, 3))
        if all(o.horizontal != horizontal or o.line != line or candidate.high + CLEAR_OF_OPENING <= o.low
               or o.high + CLEAR_OF_OPENING <= candidate.low for o in taken):
            return candidate
        near_end = False
    return None


def _doors(rng: np.random.Generator, rooms: np.ndarray, kinds: List[str], parents: Dict[int, Optional[int]]) -> List[Opening]:
    """One door into every room from the room it opens off, and the door to the outside, from the
    living room."""
    doors: List[Opening] = []
    for room, parent in parents.items():
        if parent is None:
            continue
        width = rng.uniform(*DOOR_WIDTH.get(kinds[room], DOOR_WIDTH_ELSE))
        for stretch in sorted(stretches_between(rooms, room, parent), key=lambda s: s[2] - s[3]):
            door = _place(rng, stretch, width, CLEAR_OF_CORNER, doors, "door", near_end=rng.random() < 0.65)
            if door:
                doors.append(door)
                break
        else:
            raise Retry(f"no door into the {kinds[room]}")
    width = rng.uniform(*OUTSIDE_DOOR_WIDTH)
    for stretch in sorted(stretches_between(rooms, 0, -1), key=lambda s: s[2] - s[3]):
        door = _place(rng, stretch, width, CLEAR_OF_CORNER + 0.15, doors, "outside door", near_end=rng.random() < 0.3)
        if door:
            return doors + [door]
    raise Retry("no way in")


def _windows(rng: np.random.Generator, rooms: np.ndarray, kinds: List[str], doors: List[Opening]) -> List[Opening]:
    """Windows in the outside walls, by the kind of room behind them, spread along its facade."""
    taken = list(doors)
    margin = CLEAR_OF_CORNER + 0.2
    for room, kind in enumerate(kinds):
        if kind == "balcony":
            continue
        facade = sorted(stretches_between(rooms, room, -1), key=lambda s: s[2] - s[3])
        length = sum(high - low for _, _, low, high in facade) * CELL
        wanted = {
            "living": min(4, max(1, round(length / 3.0))), "kitchen": 1 + (length > 6 and rng.random() < 0.3),
            "bedroom": 1 + (length > 5 and rng.random() < 0.5), "bathroom": int(rng.random() < 0.6),
            "storage": int(rng.random() < 0.15),
        }[kind]
        for at in range(wanted):
            if not facade:
                break
            stretch = facade[at % len(facade)]
            room_length = (stretch[3] - stretch[2]) * CELL
            width = min(rng.uniform(*WINDOW_WIDTH[kind]), room_length - 2 * margin)
            if width < WINDOW_WIDTH[kind][0] * 0.8:
                continue
            window = _place(rng, stretch, width, margin, taken, "window", near_end=False)
            if window:
                taken.append(window)
    return taken[len(doors):]


# --- The rules -------------------------------------------------------------------------------------

def _joins(rooms: np.ndarray, opening: Opening) -> set:
    """The rooms on the two sides of an opening (-1 for outside), over all of its length."""
    one, other = sides(rooms, opening.horizontal)
    cells = slice(int(math.floor(opening.low / CELL + 1e-6)), int(math.ceil(opening.high / CELL - 1e-6)))
    return set(one[opening.line, cells].tolist()) | set(other[opening.line, cells].tolist())


def problems(plan: Plan) -> List[str]:
    """Every rule the plan breaks."""
    found = []
    rooms, kinds = plan.rooms, plan.kinds
    if not connected(np.pad(rooms < 0, 1, constant_values=True)):
        found.append("a hole in the plan")
    for room, kind in enumerate(kinds):
        region = rooms == room
        if not region.any() or not connected(region):
            found.append(f"room {room} is not one area")
        elif not wide_enough(region, NARROWEST[kind]):
            found.append(f"room {room} is narrower than a {kind} may be")

    walls = walls_of(rooms)
    for wall in walls:
        if (wall.high - wall.low) * CELL < SHORTEST_WALL - 1e-6:
            found.append(f"{wall} is shorter than {SHORTEST_WALL} m")
    graph: Dict[int, set] = {room: set() for room in range(-1, len(kinds))}
    doors = 0
    for opening in plan.openings:
        joined = _joins(rooms, opening)
        hosts = [w for w in walls if w.horizontal == opening.horizontal and w.line == opening.line
                 and w.low * CELL <= opening.low and opening.high <= w.high * CELL]
        if len(hosts) != 1:
            found.append(f"{opening} stands in {len(hosts)} walls")
        if len(joined) != 2:
            found.append(f"{opening} does not stand between two spaces: {joined}")
            continue
        if opening.kind == "door" and -1 in joined:
            found.append(f"{opening} leads outside")
        if opening.kind in ("outside door", "window") and -1 not in joined:
            found.append(f"{opening} is in an inner wall")
        if opening.kind == "window" and any(kinds[r] == "balcony" for r in joined if r >= 0):
            found.append(f"{opening} is a window of a balcony")
        if opening.kind != "window":
            a, b = joined
            graph[a].add(b)
            graph[b].add(a)
            doors += opening.kind == "door"
    for a, b in combinations(plan.openings, 2):
        if a.horizontal == b.horizontal and a.line == b.line and a.low < b.high + CLEAR_OF_OPENING - 1e-6 and b.low < a.high + CLEAR_OF_OPENING - 1e-6:
            found.append(f"{a} and {b} overlap")
    seen, frontier = {-1}, [-1]
    while frontier:
        for nxt in graph[frontier.pop()] - seen:
            seen.add(nxt)
            frontier.append(nxt)
    if len(seen) != len(kinds) + 1:
        found.append(f"rooms {sorted(set(range(len(kinds))) - seen)} cannot be reached from outside")
    # The doors between rooms form a tree, and only the living room and bedrooms are walked through.
    if doors != len(kinds) - 1:
        found.append(f"{doors} doors between {len(kinds)} rooms: not a tree")
    for room, kind in enumerate(kinds):
        if kind not in ("living", "bedroom") and len(graph[room] - {-1}) > 1:
            found.append(f"the {kind} {room} is walked through")
        if -1 in graph[room] and kind != "living":
            found.append(f"the {kind} {room} has a door to the outside")
    return found


# --- The record ------------------------------------------------------------------------------------

#: The part of the render a plan may fill, and the most pixels a metre may take, so that a small plan
#: is not drawn larger than a large one would be.
FILL = 0.9
MOST_PER_METRE = 40.0


def frame(plan: Plan, size: int) -> Tuple[float, np.ndarray]:
    """Pixels per metre, and where the plan's grid origin lands, in pixels: the plan centred."""
    extent = np.array(plan.rooms.shape[::-1]) * CELL
    scale = min(MOST_PER_METRE * size / 512, FILL * size / float(extent.max()))
    return scale, (size - extent * scale) / 2


def record_of(plan: Plan, size: int) -> dict:
    """The record of a plan, in the pixels of a render `size` wide."""
    scale, origin = frame(plan, size)

    def px(horizontal: bool, line: float, along: float) -> List[float]:
        x, y = (along, line) if horizontal else (line, along)
        return [round(float(origin[0] + x * scale), 2), round(float(origin[1] + y * scale), 2)]

    walls = walls_of(plan.rooms)
    nodes = [{"class": "outside wall" if w.outside else "inside wall",
              "points": [px(w.horizontal, w.line * CELL, w.low * CELL), px(w.horizontal, w.line * CELL, w.high * CELL)]}
             for w in walls]
    relationships = [{"type": "joined", "nodes": [i, j]} for i, j in combinations(range(len(walls)), 2)
                     if {tuple(p) for p in nodes[i]["points"]} & {tuple(p) for p in nodes[j]["points"]}]
    # Every node that borders a room, and the spaces on its two sides: a wall ends wherever another
    # meets it, so the same room lies along all of it on either side.
    bordered = []
    for at, wall in enumerate(walls):
        one, other = sides(plan.rooms, wall.horizontal)
        joined = set(one[wall.line, wall.low:wall.high].tolist()) | set(other[wall.line, wall.low:wall.high].tolist())
        assert len(joined) == 2, f"{wall} borders {joined}"
        bordered.append((at, joined))
    for opening in sorted(plan.openings, key=lambda o: (o.kind, o.horizontal, o.line, o.low)):
        nodes.append({"class": opening.kind, "points": [px(opening.horizontal, opening.line * CELL, opening.low),
                                                         px(opening.horizontal, opening.line * CELL, opening.high)]})
        host = next(at for at, w in enumerate(walls) if w.horizontal == opening.horizontal and w.line == opening.line
                    and w.low * CELL <= opening.low and opening.high <= w.high * CELL)
        relationships.append({"type": "in", "nodes": [len(nodes) - 1, host]})
        bordered.append((len(nodes) - 1, _joins(plan.rooms, opening)))
    # The rooms last, in the order they were laid out, each by its kind alone.
    first_room = len(nodes)
    nodes += [{"class": kind} for kind in plan.kinds]
    for at, joined in bordered:
        relationships += [{"type": "borders", "nodes": [at, first_room + room]} for room in sorted(joined - {-1})]
    return {"nodes": nodes, "relationships": relationships}
