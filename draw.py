"""A record drawn as the image a model reads: the image follows from the record and nothing else.

Every wall, outside or inside, is a black bar of one thickness along its segment, reaching half a
thickness past its ends so that walls meeting in a corner close it. A door or outside door is a gap in
its wall, closed across at each jamb; a window the same gap outlined, with a line down its middle. A room is not drawn: it is what its
walls enclose. A record's furnishing, if it has one, is drawn beneath the walls (`icons.py`).
"""

import numpy as np
import skia

import icons

#: The thickness of a wall, and of the lines of doors and windows, in pixels of a 512 render.
WALL = 6.0
LINE = 1.0
#: The nodes drawn as gaps in their walls; rooms are not drawn at all.
OPENINGS = ("door", "outside door", "window")


def draw(record: dict, size: int) -> np.ndarray:
    """`(size, size)` uint8, white background, row index = y."""
    wall, line = WALL * size / 512, LINE * size / 512
    surface = skia.Surface(size, size)
    canvas = surface.getCanvas()
    canvas.clear(skia.ColorWHITE)
    black = skia.Paint(Color=skia.ColorBLACK, AntiAlias=True)
    white = skia.Paint(Color=skia.ColorWHITE, AntiAlias=False)
    stroke = skia.Paint(Color=skia.ColorBLACK, Style=skia.Paint.kStroke_Style, StrokeWidth=line, AntiAlias=True)
    # The furniture first, for the walls to be drawn over it.
    if record.get("furnishing"):
        icons.draw_furnishing(canvas, record["furnishing"], size / 512)

    def bar(start, end, width: float, reach: float) -> skia.Rect:
        """The rectangle `width` across along a horizontal or vertical segment, `reach` past its ends."""
        (x0, y0), (x1, y1) = sorted([tuple(start), tuple(end)])
        if y0 == y1:
            return skia.Rect.MakeLTRB(x0 - reach, y0 - width / 2, x1 + reach, y0 + width / 2)
        return skia.Rect.MakeLTRB(x0 - width / 2, y0 - reach, x0 + width / 2, y1 + reach)

    for node in record["nodes"]:
        if node["class"].endswith("wall"):
            canvas.drawRect(bar(*node["points"], wall, wall / 2), black)
    for node in record["nodes"]:
        if node["class"] not in OPENINGS:
            continue
        gap = bar(*node["points"], wall, 0)
        canvas.drawRect(bar(*node["points"], wall + 2, 0), white)
        (x0, y0), (x1, y1) = node["points"]
        if node["class"] == "window":
            canvas.drawRect(gap, stroke)
            canvas.drawLine(x0, y0, x1, y1, stroke)
        else:
            for x, y in ((x0, y0), (x1, y1)):
                across = (0, wall / 2) if y0 == y1 else (wall / 2, 0)
                canvas.drawLine(x - across[0], y - across[1], x + across[0], y + across[1], stroke)
    image = surface.makeImageSnapshot().toarray(colorType=skia.ColorType.kGray_8_ColorType)
    return np.asarray(image)[:, :, 0] if image.ndim == 3 else np.asarray(image)
