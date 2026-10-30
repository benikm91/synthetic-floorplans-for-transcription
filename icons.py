"""Furniture drawn as the symbols of an architect's plan, all of one plan in one style.

A plan's look picks its style -- how dark and heavy the lines are, whether furniture is tinted,
how round its corners -- and, for every kind of furniture, one of its variants: every bed of a plan
is drawn the same way, as every chair is. Each symbol is drawn in its own frame, in metres: its width
along x, its back at y = 0 and its front at y = depth.
"""

import math

import numpy as np
import skia
from scipy import ndimage

#: Line ink and weight (pixels of a 512 render), the detail lines' ink and weight, the fill, and the
#: radius of corners in metres.
STYLES = [
    dict(ink=35, width=1.0, detail=35, detail_width=0.6, fill=255, round=0.0),
    dict(ink=95, width=0.9, detail=125, detail_width=0.6, fill=255, round=0.03),
    dict(ink=55, width=0.9, detail=95, detail_width=0.6, fill=226, round=0.04),
    dict(ink=20, width=1.4, detail=20, detail_width=0.8, fill=255, round=0.06),
    dict(ink=75, width=0.8, detail=115, detail_width=0.5, fill=238, round=0.0),
    dict(ink=45, width=1.1, detail=70, detail_width=0.7, fill=246, round=0.08),
]
VARIANTS = 3


class Pen:
    """Drawing in an item's frame, in metres."""

    def __init__(self, canvas: skia.Canvas, style: dict, scale: float, factor: float):
        self.canvas, self.style = canvas, style
        unit = factor / scale                      # one pixel of a 512 render, in metres
        gray = lambda v: skia.Color(v, v, v)
        self.line = skia.Paint(Color=gray(style["ink"]), Style=skia.Paint.kStroke_Style, StrokeWidth=style["width"] * unit, AntiAlias=True)
        self.thin = skia.Paint(Color=gray(style["detail"]), Style=skia.Paint.kStroke_Style, StrokeWidth=style["detail_width"] * unit, AntiAlias=True)
        self.fill = skia.Paint(Color=gray(style["fill"]), AntiAlias=True)
        self.paper = skia.Paint(Color=skia.ColorWHITE, AntiAlias=True)
        self.dark = skia.Paint(Color=gray(style["ink"]), AntiAlias=True)
        self.r = style["round"]

    def box(self, x0, y0, x1, y1, r=None, thin=False, fill=True, paper=False):
        r = self.r if r is None else r
        r = min(r, abs(x1 - x0) / 2, abs(y1 - y0) / 2)
        rect = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x0, y0, x1, y1), r, r)
        if fill:
            self.canvas.drawRRect(rect, self.paper if paper else self.fill)
        self.canvas.drawRRect(rect, self.thin if thin else self.line)

    def oval(self, x0, y0, x1, y1, thin=False, fill=True):
        rect = skia.Rect.MakeLTRB(x0, y0, x1, y1)
        if fill:
            self.canvas.drawOval(rect, self.fill)
        self.canvas.drawOval(rect, self.thin if thin else self.line)

    def circle(self, x, y, r, thin=False, fill=False, dot=False):
        if dot:
            self.canvas.drawCircle(x, y, r, self.dark)
            return
        if fill:
            self.canvas.drawCircle(x, y, r, self.fill)
        self.canvas.drawCircle(x, y, r, self.thin if thin else self.line)

    def seg(self, x0, y0, x1, y1, thin=True):
        self.canvas.drawLine(x0, y0, x1, y1, self.thin if thin else self.line)

    def dashed(self, x0, y0, x1, y1, dash):
        paint = skia.Paint(self.thin)
        paint.setPathEffect(skia.DashPathEffect.Make([dash, dash * 0.8], 0))
        self.canvas.drawLine(x0, y0, x1, y1, paint)

    def path(self, points, close=True, thin=False, fill=True):
        path = skia.Path()
        path.moveTo(*points[0])
        for p in points[1:]:
            path.lineTo(*p)
        if close:
            path.close()
            if fill:
                self.canvas.drawPath(path, self.fill)
        self.canvas.drawPath(path, self.thin if thin else self.line)

    def arc(self, x0, y0, x1, y1, start, sweep, thin=True):
        self.canvas.drawArc(skia.Rect.MakeLTRB(x0, y0, x1, y1), start, sweep, False, self.thin if thin else self.line)


# --- The symbols -----------------------------------------------------------------------------------

def bed(p: Pen, w, d, v, it):
    p.box(0, 0, w, d)
    pillows = 2 if w > 1.2 else 1
    gap = 0.08
    pw = (w - gap * (pillows + 1)) / pillows
    head = 0.0
    if v == 2:
        head = 0.07
        p.box(0, 0, w, head, r=0, fill=False)
    for k in range(pillows):
        x = gap + k * (pw + gap)
        if v == 1:
            p.oval(x, head + 0.07, x + pw, head + 0.4, thin=True)
        else:
            p.box(x, head + 0.07, x + pw, head + 0.4, r=0.06 if v == 2 else p.r, thin=True)
    top = head + 0.55
    p.seg(0, top, w, top, thin=False)
    if v == 0:
        p.path([(w * 0.45, top), (w, top + w * 0.55)], close=False, thin=True)
    elif v == 1:
        p.seg(0, top + 0.12, w, top + 0.12)
    else:
        p.path([(w - 0.45, top), (w - 0.45, top + 0.45), (w, top + 0.45)], close=False, thin=True)


def nightstand(p, w, d, v, it):
    p.box(0, 0, w, d)
    if v == 1:
        p.circle(w / 2, d / 2, min(w, d) * 0.28, thin=True)
    elif v == 2:
        p.box(0.04, 0.04, w - 0.04, d - 0.04, thin=True, fill=False)


def wardrobe(p, w, d, v, it):
    p.box(0, 0, w, d)
    if v == 0:
        p.dashed(0.05, d * 0.45, w - 0.05, d * 0.45, 0.06)
    elif v == 1:
        p.seg(0.05, d * 0.45, w - 0.05, d * 0.45)
        x = 0.12
        while x < w - 0.08:
            p.seg(x - 0.05, d * 0.2, x + 0.05, d * 0.7)
            x += 0.14
    else:
        doors = max(1, round(w / 0.5))
        for k in range(1, doors):
            p.seg(w * k / doors, 0, w * k / doors, d)
        p.seg(0, d - 0.05, w, d - 0.05)


def chair(p, x, y, w, d, v):
    """A chair with its back at y = 0 of its own box."""
    if v == 2:
        p.circle(x + w / 2, y + d / 2, min(w, d) * 0.42, fill=True)
        return
    p.box(x, y, x + w, y + d, r=0.08 if v == 1 else p.r)
    if v == 1:
        p.arc(x + 0.03, y + 0.02, x + w - 0.03, y + d * 0.6, 180, 180, thin=False)
    else:
        p.box(x, y, x + w, y + 0.09, r=0, thin=True)


def chair_at(p, cx, cy, size, angle, v):
    """A chair centred at (cx, cy) whose back points `angle` degrees from up."""
    p.canvas.save()
    p.canvas.translate(cx, cy)
    p.canvas.rotate(angle)
    chair(p, -size / 2, -size / 2, size, size, v)
    p.canvas.restore()


def desk(p, w, d, v, it):
    p.box(0, 0, w, d)
    if v == 1:
        p.seg(0, d - 0.05, w, d - 0.05)
    elif v == 2:
        p.box(w - 0.45, 0.04, w - 0.04, d - 0.04, thin=True, fill=False)
    chair_at(p, w / 2, d + 0.3, 0.45, 180, it["chair"])


def sofa(p, w, d, v, it):
    back, arm = 0.2, 0.18
    if v == 1:
        p.box(0, 0, w, d, r=max(p.r, 0.08))
        p.box(arm, back, w - arm, d - 0.04, r=0.06, thin=True, fill=False)
        return
    p.box(0, 0, w, d)
    p.seg(arm, back, w - arm, back, thin=False)
    p.seg(arm, back, arm, d, thin=False)
    p.seg(w - arm, back, w - arm, d, thin=False)
    seats = max(2, round((w - 2 * arm) / 0.65))
    if v == 0:
        for k in range(1, seats):
            x = arm + (w - 2 * arm) * k / seats
            p.seg(x, back, x, d)
    else:
        sw = (w - 2 * arm) / seats
        for k in range(seats):
            x = arm + k * sw
            p.box(x + 0.03, back + 0.03, x + sw - 0.03, d - 0.05, r=0.05, thin=True, fill=False)


def armchair(p, w, d, v, it):
    back, arm = 0.18, 0.15
    if v == 1:
        p.box(0, 0, w, d, r=max(p.r, 0.1))
        p.box(arm, back, w - arm, d - 0.04, r=0.06, thin=True, fill=False)
        return
    p.box(0, 0, w, d)
    p.seg(arm, back, w - arm, back, thin=False)
    p.seg(arm, back, arm, d, thin=False)
    p.seg(w - arm, back, w - arm, d, thin=False)


def coffee_table(p, w, d, v, it):
    if v == 2:
        p.oval(0, 0, w, d)
    elif v == 1:
        p.box(0, 0, w, d, r=0.08)
        p.box(0.05, 0.05, w - 0.05, d - 0.05, r=0.05, thin=True, fill=False)
    else:
        p.box(0, 0, w, d)


def tv(p, w, d, v, it):
    p.box(0, 0, w, d)
    screen = w * 0.7
    x = (w - screen) / 2
    if v == 0:
        p.box(x, 0.06, x + screen, 0.11, r=0, thin=False, fill=True)
    elif v == 1:
        p.path([(x, 0.08), (x + screen, 0.08), (x + screen - 0.05, 0.13), (x + 0.05, 0.13)], thin=True)
        p.seg(w / 2 - 0.1, 0.2, w / 2 + 0.1, 0.2, thin=False)
    else:
        p.seg(0, d - 0.06, w, d - 0.06)
        p.box(x, 0.04, x + screen, 0.09, r=0, thin=True)


def dining(p, w, d, v, it):
    top, bottom, left, right = it["chairs"]
    c = 0.5
    x0, x1, y0, y1 = c * (left > 0), w - c * (right > 0), c, d - c
    size = 0.42
    for count, edge in ((top, "top"), (bottom, "bottom")):
        for k in range(count):
            x = x0 + (x1 - x0) * (k + 0.5) / count
            if edge == "top":
                chair_at(p, x, y0 - 0.26, size, 0, it["chair"])
            else:
                chair_at(p, x, y1 + 0.26, size, 180, it["chair"])
    for count, edge in ((left, "left"), (right, "right")):
        for k in range(count):
            y = y0 + (y1 - y0) * (k + 0.5) / count
            if edge == "left":
                chair_at(p, x0 - 0.26, y, size, -90, it["chair"])
            else:
                chair_at(p, x1 + 0.26, y, size, 90, it["chair"])
    if v == 2 and abs((x1 - x0) - (y1 - y0)) < 0.15:
        p.oval(x0, y0, x1, y1)
    elif v == 2:
        p.box(x0, y0, x1, y1, r=min(x1 - x0, y1 - y0) / 2)
    else:
        p.box(x0, y0, x1, y1, r=0.05 if v == 1 else None)


def bistro(p, w, d, v, it):
    r = min(w, 0.6) / 2
    chair_at(p, w / 2, 0.24, 0.4, 0, it["chair"])
    chair_at(p, w / 2, d - 0.24, 0.4, 180, it["chair"])
    p.circle(w / 2, d / 2, r * 0.95, fill=True)


def counter(p, w, d, v, it):
    p.box(0, 0, w, d, r=0)
    if v == 0:
        p.dashed(0, 0.35, w, 0.35, 0.08)
    elif v == 1:
        x = 0.6
        while x < w - 0.1:
            p.seg(x, 0, x, d)
            x += 0.6
    else:
        p.seg(0, d - 0.04, w, d - 0.04)


def sink(p, w, d, v, it):
    if v == 2:
        p.oval(0.1, 0.1, w - 0.1, d - 0.08, thin=False)
        p.circle(w / 2, 0.06, 0.02, dot=True)
        return
    basin = w if v == 0 else min(w, 0.5)
    p.box(0.07, 0.08, basin - 0.07, d - 0.07, r=0.05)
    p.circle((basin) / 2, d / 2 + 0.03, 0.025, thin=True)
    p.seg(basin / 2, 0.02, basin / 2, 0.12, thin=False)
    if v == 1:
        x = basin + 0.04
        while x < w - 0.04:
            p.seg(x, 0.1, x, d - 0.1)
            x += 0.06


def hob(p, w, d, v, it):
    if v == 2:
        p.box(0.04, 0.04, w - 0.04, d - 0.04, r=0.02, thin=True, fill=False)
    for x, y, r in ((0.17, 0.17, 0.09), (w - 0.17, 0.17, 0.07), (0.17, d - 0.17, 0.07), (w - 0.17, d - 0.17, 0.09)):
        p.circle(x, y, r, thin=False)
        if v == 1:
            p.circle(x, y, r * 0.5, thin=True)


def fridge(p, w, d, v, it):
    p.box(0, 0, w, d, r=0)
    if v == 0:
        p.seg(0, 0, w, d)
        p.seg(0, d, w, 0)
    elif v == 1:
        p.box(0.05, 0.05, w - 0.05, d - 0.05, r=0, thin=True, fill=False)
    else:
        p.seg(0, d - 0.06, w, d - 0.06)
        p.seg(w - 0.12, d - 0.03, w - 0.04, d - 0.03, thin=False)


def toilet(p, w, d, v, it):
    tw, x = 0.4, (w - 0.4) / 2
    if v == 2:
        p.box(x + 0.02, 0, x + tw - 0.02, 0.07, r=0.02)
        p.oval(x + 0.03, 0.07, x + tw - 0.03, min(d, 0.62))
        p.oval(x + 0.08, 0.14, x + tw - 0.08, min(d, 0.62) - 0.06, thin=True, fill=False)
        return
    p.box(x, 0, x + tw, 0.18, r=0.03)
    bottom = min(d, 0.68)
    if v == 0:
        p.oval(x + 0.03, 0.15, x + tw - 0.03, bottom)
        p.oval(x + 0.08, 0.22, x + tw - 0.08, bottom - 0.06, thin=True, fill=False)
    else:
        r = (tw - 0.06) / 2
        pts = [(x + 0.03, 0.18), (x + 0.03, bottom - r)]
        pts += [(x + tw / 2 - r * math.cos(math.pi * t / 12), bottom - r + r * math.sin(math.pi * t / 12)) for t in range(13)]
        pts += [(x + tw - 0.03, 0.18)]
        p.path(pts)
        p.seg(x + 0.03, 0.24, x + tw - 0.03, 0.24)


def basin(p, w, d, v, it):
    count = 2 if it.get("double") else 1
    if v == 1 and count == 1:
        p.path([(0, 0)] + [(w / 2 + w / 2 * math.cos(math.pi * t / 16), d * math.sin(math.pi * t / 16)) for t in range(17)][::-1] + [(w, 0)], thin=False)
        p.oval(w * 0.2, 0.08, w * 0.8, d * 0.75, thin=True, fill=False)
        return
    p.box(0, 0, w, d, r=0)
    for k in range(count):
        cx = w * (k + 0.5) / count
        bw = min(0.45, w / count - 0.1)
        if v == 2:
            p.box(cx - bw / 2, 0.08, cx + bw / 2, d - 0.06, r=0.06, thin=True, fill=False)
        else:
            p.oval(cx - bw / 2, 0.08, cx + bw / 2, d - 0.06, thin=True, fill=False)
        p.circle(cx, 0.05, 0.015, dot=True)


def bathtub(p, w, d, v, it):
    p.box(0, 0, w, d, r=0.02)
    m = 0.07
    if v == 0:
        p.box(m, m, w - m, d - m, r=0.12, thin=True, fill=False)
        p.circle(w - 0.2, d / 2, 0.025, thin=True)
    elif v == 1:
        r = (d - 2 * m) / 2
        path = [(w - m - 0.05, m), (m + r, m)]
        path += [(m + r - r * math.sin(math.pi * t / 12), m + r - r * math.cos(math.pi * t / 12)) for t in range(13)]
        path += [(w - m - 0.05, d - m)]
        p.path(path, thin=True, fill=False)
        p.circle(w - 0.2, d / 2, 0.025, thin=True)
    else:
        p.oval(m, m, w - m, d - m, thin=True, fill=False)
        p.circle(w - 0.22, d / 2, 0.025, dot=True)


def shower(p, w, d, v, it):
    p.box(0, 0, w, d, r=0)
    if v == 0:
        p.seg(0, 0, w, d)
        p.seg(0, d, w, 0)
    elif v == 1:
        p.box(0.06, 0.06, w - 0.06, d - 0.06, r=0, thin=True, fill=False)
        p.circle(w / 2, d / 2, 0.04, thin=True)
    else:
        p.arc(-w, 0, w, 2 * d, 270, 90)
        p.circle(w * 0.3, d * 0.3, 0.04, thin=True)


def washer(p, w, d, v, it):
    p.box(0, 0, w, d, r=0.03 if v == 2 else None)
    p.circle(w / 2, d / 2 + 0.03, min(w, d) * 0.32, thin=v != 1)
    if v == 1:
        p.circle(w / 2, d / 2 + 0.03, min(w, d) * 0.2, thin=True)


def shelves(p, w, d, v, it):
    p.box(0, 0, w, d, r=0)
    if v == 0:
        p.seg(0, d / 2, w, d / 2)
    elif v == 1:
        x = 0.3
        while x < w - 0.05:
            p.seg(x, 0, x, d)
            x += 0.3
    else:
        p.seg(0, d * 0.35, w, d * 0.35)
        p.seg(0, d * 0.7, w, d * 0.7)


def plant(p, w, d, v, it):
    cx, cy, r = w / 2, d / 2, min(w, d) / 2 * 0.9
    if v == 0:
        p.circle(cx, cy, r, fill=True)
        p.circle(cx, cy, r * 0.45, thin=True)
    elif v == 1:
        p.circle(cx, cy, r * 0.5, fill=True)
        for t in range(7):
            a = 2 * math.pi * t / 7
            p.seg(cx + 0.5 * r * math.cos(a), cy + 0.5 * r * math.sin(a), cx + r * math.cos(a), cy + r * math.sin(a), thin=False)
    else:
        for t in range(6):
            a = 2 * math.pi * t / 6
            p.circle(cx + 0.55 * r * math.cos(a), cy + 0.55 * r * math.sin(a), r * 0.4, thin=True, fill=True)
        p.circle(cx, cy, r * 0.2, dot=True)


SYMBOLS = {"bed": bed, "nightstand": nightstand, "wardrobe": wardrobe, "desk": desk, "sofa": sofa, "armchair": armchair,
           "coffee table": coffee_table, "tv": tv, "dining": dining, "bistro": bistro, "counter": counter, "sink": sink,
           "hob": hob, "fridge": fridge, "toilet": toilet, "basin": basin, "bathtub": bathtub, "shower": shower,
           "washer": washer, "shelves": shelves, "plant": plant}


#: The way an item faces, by its facing: where its front points in the image.
FACINGS = ("down", "left", "up", "right")


def _look(furnishing: dict, factor: float, canvas: skia.Canvas) -> tuple:
    """A plan's pen and its variant of every symbol."""
    rng = np.random.default_rng(furnishing["look"])
    style = dict(STYLES[int(rng.integers(len(STYLES)))])
    style["width"] *= rng.uniform(0.85, 1.15)
    variants = {kind: int(rng.integers(VARIANTS)) for kind in sorted(SYMBOLS) + ["chair"]}
    return Pen(canvas, style, furnishing["scale"], factor), variants


def _draw_item(pen: Pen, item: dict, variants: dict, scale: float) -> None:
    canvas = pen.canvas
    x0, y0, x1, y1 = item["box"]
    k = item["facing"]
    w, d = (x1 - x0, y1 - y0) if k % 2 == 0 else (y1 - y0, x1 - x0)
    canvas.save()
    canvas.translate((x0 + x1) / 2, (y0 + y1) / 2)
    canvas.rotate(90 * k)
    canvas.scale(scale, scale)
    canvas.translate(-w / scale / 2, -d / scale / 2)
    SYMBOLS[item["kind"]](pen, w / scale, d / scale, variants[item["kind"]], {**item, "chair": variants["chair"]})
    canvas.restore()


def draw_furnishing(canvas: skia.Canvas, furnishing: dict, factor: float) -> None:
    """Every item of a plan's furnishing; `factor` is the render's size over 512."""
    pen, variants = _look(furnishing, factor, canvas)
    for item in furnishing["items"]:
        _draw_item(pen, item, variants, furnishing["scale"])


#: How finely an item's ink is measured: samples per pixel.
FINE = 4


def ink(furnishing: dict, item: dict) -> tuple:
    """The box of the ink an item leaves, drawn alone in its plan's look, in the furnishing's pixels; and
    its lines within it, `FINE` samples a pixel."""
    x0, y0, x1, y1 = item["box"]
    margin = 0.8 * furnishing["scale"] + 4          # room for what is drawn past the box, a desk's chair
    left, top = x0 - margin, y0 - margin
    width, height = int((x1 - x0 + 2 * margin) * FINE) + 1, int((y1 - y0 + 2 * margin) * FINE) + 1
    surface = skia.Surface(width, height)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorTRANSPARENT)
    canvas.scale(FINE, FINE)
    canvas.translate(-left, -top)
    pen, variants = _look(furnishing, 1.0, canvas)
    _draw_item(pen, item, variants, furnishing["scale"])
    rgba = surface.makeImageSnapshot().toarray()
    drawn = rgba[:, :, 3] >= 96
    # The lines, as they show on white: what is darker than the lightest tint a symbol is filled with.
    alpha = rgba[:, :, 3] / 255.0
    lines = (1 - alpha) * 255 + alpha * rgba[:, :, :3].mean(2) * (255 / np.maximum(rgba[:, :, 3], 1)) < 215
    rows, cols = np.flatnonzero(drawn.any(1)), np.flatnonzero(drawn.any(0))
    box = (left + cols[0] / FINE, top + rows[0] / FINE, left + (cols[-1] + 1) / FINE, top + (rows[-1] + 1) / FINE)
    return box, lines[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def turns(mask: np.ndarray) -> int:
    """How an item's lines look the same turned: 4 if turned a quarter, 2 if only half round, else 1.
    Alike is every sample of the one within half a pixel of the other's lines."""
    def alike(a: np.ndarray, b: np.ndarray) -> bool:
        near = lambda m: ndimage.binary_dilation(m, np.ones((3, 3), bool), iterations=FINE // 2)
        return a.shape == b.shape and not (a & ~near(b)).any() and not (b & ~near(a)).any()

    if not alike(mask, mask[::-1, ::-1]):
        return 1
    h, w = mask.shape
    side = min(h, w)
    if abs(h - w) <= FINE // 2 and alike(mask[:side, :side], np.rot90(mask[:side, :side])):
        return 4
    return 2


def furniture_record(furnishing: dict) -> dict:
    """The record of a plan's furniture, in the format of the plan's record: a node per item, its class
    its kind, its points the corners of the box of its ink -- back-left, then front-right, as the item
    itself has them -- and the way it faces. An item whose lines look the same turned a quarter faces
    down; one that looks the same turned half round faces down or left. Relationships it has none: an
    item's room is the one whose walls enclose it."""
    nodes = []
    for item in furnishing["items"]:
        box, lines = ink(furnishing, item)
        x0, y0, x1, y1 = (round(float(v), 2) for v in box)
        # An item that looks the same turned faces the first way it looks so: down, or down or left.
        k = {4: 0, 2: item["facing"] % 2, 1: item["facing"]}[turns(lines)]
        corners = {0: ([x0, y0], [x1, y1]), 1: ([x1, y0], [x0, y1]), 2: ([x1, y1], [x0, y0]), 3: ([x0, y1], [x1, y0])}[k]
        nodes.append({"class": item["kind"], "points": list(corners), "facing": FACINGS[k]})
    nodes.sort(key=lambda node: (node["class"], min(p[1] for p in node["points"]), min(p[0] for p in node["points"])))
    return {"nodes": nodes, "relationships": []}
