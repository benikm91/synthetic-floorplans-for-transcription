"""Sample synthetic floor plans in a local web UI: each plan generated from its seed, drawn from its
record, with the record laid over it.

Hovering a node or a relationship lights it up on the drawing and in the lists; a room, which has no
points, is shown at the middle of the walls that border it. The cells of the rooms the plan was laid
out from can be laid over it too; they are not part of the record.

    uv run view_synthetic.py
"""

import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template_string, request, send_file

import draw
import furnish
import icons
import synthetic

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Synthetic Floor Plans</title>
    <style>
        :root {
            --bg: #f5f1e8; --panel: #fffdf9; --ink: #161514; --muted: #6d665c; --line: #ddd3c5;
            --accent: #186a73;
            --wall: #8b1a1a; --inside: #d4691e; --door: #c2187a; --outside: #0a8f5c; --window: #1f63d6;
            --lit: #e3b505; --partner: #7b3fc4;
        }
        * { box-sizing: border-box; }
        [hidden] { display: none !important; }
        body { margin: 0; font-family: Georgia, "Iowan Old Style", serif; background: var(--bg); color: var(--ink); }
        .shell { max-width: 1600px; margin: 0 auto; padding: 24px; }
        .title { margin: 0; font-size: 34px; }
        .subtitle { margin: 8px 0 18px; color: var(--muted); }
        .layout { display: grid; grid-template-columns: minmax(420px, 1.4fr) minmax(360px, 1fr); gap: 20px; align-items: start; }
        .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 18px; overflow: hidden; }
        .controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; padding: 14px; border-bottom: 1px solid var(--line); }
        .field { display: grid; gap: 4px; }
        label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
        select, input, button { border-radius: 10px; border: 1px solid var(--line); padding: 8px 10px; font: inherit; background: white; color: var(--ink); }
        button { cursor: pointer; background: var(--accent); color: white; border-color: var(--accent); }
        button.secondary { background: white; color: var(--ink); }
        .nav-row { display: flex; gap: 8px; flex-wrap: wrap; padding: 12px 14px 0; }
        .viewer { padding: 14px; }
        .meta { display: flex; justify-content: space-between; color: var(--muted); font-size: 14px; margin-bottom: 8px; }
        .frame { position: relative; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: white; aspect-ratio: 1 / 1; }
        .frame img { width: 100%; height: 100%; display: block; }
        .frame svg { position: absolute; inset: 0; width: 100%; height: 100%; }
        .toggles { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 10px; font-size: 14px; color: var(--muted); }
        .toggles label { text-transform: none; letter-spacing: 0; font-size: 14px; }
        .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 13px; color: var(--muted); }
        .legend span::before { content: ""; display: inline-block; width: 18px; height: 3px; margin-right: 5px; vertical-align: middle; background: var(--c); }
        .sidebar { padding: 14px; }
        .info-title { margin: 0 0 6px; font-size: 17px; }
        .info-block + .info-block { margin-top: 16px; }
        .info-copy { margin: 0 0 6px; color: var(--muted); font-size: 13px; line-height: 1.4; }
        .rows { border: 1px solid var(--line); border-radius: 12px; background: #201b17; color: #f7efe2; font: 12px/1.55 Menlo, Consolas, monospace; max-height: 380px; overflow: auto; padding: 6px; }
        .row { display: block; padding: 1px 6px; border-radius: 4px; cursor: pointer; white-space: nowrap; }
        .row .kind { color: #9fd3d9; }
        .row .muted { color: #8f8577; }
        .row.lit { background: #8a6d00; color: white; }
        .row.partner { background: #4b2a78; }
        .node { fill: none; stroke-width: 2.4; opacity: 0.85; cursor: pointer; }
        .hit { fill: none; stroke: transparent; stroke-width: 12; cursor: pointer; }
        .hinge { stroke: none; }
        .relation { stroke: var(--partner); stroke-width: 1.5; stroke-dasharray: 3 3; fill: none; }
        .joint { fill: var(--partner); }
        .lit .node, .node.lit { stroke: var(--lit) !important; stroke-width: 4; opacity: 1; }
        .lit .hinge { fill: var(--lit) !important; }
        .partner .node { stroke: var(--partner) !important; stroke-width: 3.5; opacity: 1; }
        .source { fill: none; stroke-width: 1.4; opacity: 0.9; }
        .hide-furniture #furniture, .hide-record #record, .hide-relations #relations, .hide-source #source, .hide-image #render { visibility: hidden; }
        @media (max-width: 1100px) { .layout { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="shell">
    <h1 class="title">Synthetic Floor Plans</h1>
    <p class="subtitle">Each plan generated from its seed and drawn from its record, the record drawn over it. Hover a node or a
        relationship to see it on the drawing; arrow keys step through the seeds.</p>
    <div class="layout">
        <section class="panel">
            <div class="controls">
                <div class="field"><label for="indexInput">Seed</label><input id="indexInput" type="number" min="0" value="0"></div>
            </div>
            <div class="nav-row">
                <button id="prevButton" class="secondary">Previous</button>
                <button id="nextButton">Next</button>
                <button id="randomButton" class="secondary">Random</button>
            </div>
            <div class="viewer">
                <div class="meta"><span id="sampleMeta">Loading…</span><span id="countMeta"></span></div>
                <div class="frame" id="frame">
                    <img id="render" alt="">
                    <svg id="overlay"><g id="source"></g><g id="furniture"></g><g id="relations"></g><g id="record"></g></svg>
                </div>
                <div class="toggles">
                    <label><input type="checkbox" id="showImage" checked> image</label>
                    <label><input type="checkbox" id="showRecord" checked> record</label>
                    <label><input type="checkbox" id="showRelations"> relationships</label>
                    <label><input type="checkbox" id="showSource"> rooms</label>
                    <label><input type="checkbox" id="showFurniture" checked> furniture</label>
                </div>
                <div class="legend">
                    <span style="--c: var(--wall)">outside wall</span><span style="--c: var(--inside)">inside wall</span>
                    <span style="--c: var(--door)">door</span><span style="--c: var(--outside)">outside door</span>
                    <span style="--c: var(--window)">window</span><span style="--c: var(--partner)">relationship</span>
                </div>
                <p class="info-copy" style="margin-top:8px">The colours are the viewer's: the input image is black on white.
                    Every node is a segment from its start (dot) to its end (square): a wall along its middle, a door or
                    window between its jambs.</p>
            </div>
        </section>
        <aside class="panel sidebar">
            <div class="info-block">
                <h2 class="info-title">Nodes</h2>
                <div class="rows" id="nodes"></div>
            </div>
            <div class="info-block">
                <h2 class="info-title">Relationships</h2>
                <div class="rows" id="relationships"></div>
            </div>
            <div class="info-block">
                <h2 class="info-title">Dataset</h2>
                <p class="info-copy" id="datasetInfo"></p>
                <div class="rows" id="counts"></div>
            </div>
        </aside>
    </div>
</div>
<script>
    const $ = id => document.getElementById(id);
    let info = null, sample = null;
    const colour = c => ({'outside wall': 'var(--wall)', 'inside wall': 'var(--inside)', 'door': 'var(--door)',
        'outside door': 'var(--outside)', 'window': 'var(--window)'})[c] || '#999';
    const ROOM = {living: '#f2c14e', kitchen: '#f78154', bedroom: '#4d9de0', bathroom: '#15b097', storage: '#a37b73',
        balcony: '#9ccf6b'};
    const P = ([x, y]) => `${x} ${y}`;
    const f1 = v => v.toFixed(1);
    const pt = ([x, y]) => `(${f1(x)},${f1(y)})`;

    function nodeShape(n, i) {
        if (!n.points) {
            const [x, y] = centre(n, i);
            return `<g data-node="${i}"><circle class="node" style="fill:${ROOM[n.class] || '#999'}" cx="${x}" cy="${y}" r="5"/>`
                + `<text x="${x}" y="${y - 8}" font-size="11" text-anchor="middle" fill="#333" stroke="white" stroke-width="3" paint-order="stroke">${n.class}</text></g>`;
        }
        const c = colour(n.class), [a, b] = n.points;
        return `<g data-node="${i}"><path class="node" style="stroke:${c}" d="M ${P(a)} L ${P(b)}"/>`
            + `<circle class="hinge" style="fill:${c}" cx="${a[0]}" cy="${a[1]}" r="2.5"/>`
            + `<rect class="hinge" style="fill:${c}" x="${b[0] - 3}" y="${b[1] - 3}" width="6" height="6"/>`
            + `<path class="hit" d="M ${P(a)} L ${P(b)}"/></g>`;
    }

    const middle = ps => ps.reduce((s, q) => [s[0] + q[0] / ps.length, s[1] + q[1] / ps.length], [0, 0]);
    // A room has no points: it is shown at the middle of the walls that border it.
    const centre = (n, i) => n.points ? middle(n.points) : middle(sample.relationships
        .filter(r => r.type === 'borders' && r.nodes[1] === i && sample.nodes[r.nodes[0]].class.endsWith('wall'))
        .map(r => middle(sample.nodes[r.nodes[0]].points)));

    function draw() {
        $('overlay').setAttribute('viewBox', `0 0 ${info.size} ${info.size}`);
        $('record').innerHTML = sample.nodes.map(nodeShape).join('');
        $('relations').innerHTML = sample.relationships.map((r, k) => {
            const [a, b] = r.nodes.map(i => centre(sample.nodes[i], i));
            return `<g data-rel="${k}"><path class="relation" d="M ${P(a)} L ${P(b)}"/>`
                 + (r.type === 'joined' ? `<circle class="joint" cx="${(a[0] + b[0]) / 2}" cy="${(a[1] + b[1]) / 2}" r="3"/>` : '') + '</g>';
        }).join('');
        // The furniture record: each box, a dot at its first point (back-left) and a tick to its front.
        $('furniture').innerHTML = ((sample.furniture || {}).nodes || []).map(f => {
            const [[ax, ay], [bx, by]] = f.points, cx = (ax + bx) / 2, cy = (ay + by) / 2;
            const [dx, dy] = {down: [0, 1], left: [-1, 0], up: [0, -1], right: [1, 0]}[f.facing];
            const reach = (dx ? Math.abs(bx - ax) : Math.abs(by - ay)) / 2;
            return `<g><title>${f.class} · ${f.facing}</title><rect x="${Math.min(ax, bx)}" y="${Math.min(ay, by)}" width="${Math.abs(bx - ax)}" height="${Math.abs(by - ay)}" `
                + `style="fill:none;stroke:#0a8f5c;stroke-width:0.8"/><circle cx="${ax}" cy="${ay}" r="1.8" style="fill:#0a8f5c"/>`
                + `<path d="M ${cx} ${cy} L ${cx + dx * reach} ${cy + dy * reach}" style="stroke:#c2187a;stroke-width:1"/></g>`;
        }).join('');
        $('source').innerHTML = (sample.rooms || []).map(r =>
            `<rect x="${r.x}" y="${r.y}" width="${r.size}" height="${r.size}" style="fill:${ROOM[r.kind]};opacity:0.35"/>`).join('')
            + (sample.labels || []).map(l => `<text x="${l.x}" y="${l.y}" font-size="11" text-anchor="middle" fill="#333">${l.kind}</text>`).join('');
        document.querySelectorAll('[data-node]').forEach(g => {
            g.addEventListener('mouseenter', () => litNode(Number(g.dataset.node)));
            g.addEventListener('mouseleave', clear);
        });
    }

    function clear() { document.querySelectorAll('.lit, .partner').forEach(n => n.classList.remove('lit', 'partner')); }
    const mark = (selector, cls) => document.querySelectorAll(selector).forEach(n => n.classList.add(cls));

    function litNode(i) {
        clear();
        mark(`#record [data-node="${i}"], .row[data-node="${i}"]`, 'lit');
        sample.relationships.forEach((r, k) => {
            if (!r.nodes.includes(i)) return;
            mark(`.row[data-rel="${k}"]`, 'partner');
            r.nodes.filter(j => j !== i).forEach(j => mark(`#record [data-node="${j}"], .row[data-node="${j}"]`, 'partner'));
        });
    }

    function litRelation(k) {
        clear();
        mark(`.row[data-rel="${k}"]`, 'lit');
        sample.relationships[k].nodes.forEach(j => mark(`#record [data-node="${j}"], .row[data-node="${j}"]`, 'partner'));
    }

    function lists() {
        $('nodes').innerHTML = sample.nodes.map((n, i) =>
            `<span class="row" data-node="${i}">#${String(i).padStart(3)} <span class="kind">${n.class.padEnd(14)}</span> `
            + (n.points || []).map(pt).join(' ') + '</span>'
        ).join('') || '<span class="row muted">none</span>';
        $('relationships').innerHTML = sample.relationships.map((r, k) =>
            `<span class="row" data-rel="${k}"><span class="kind">${r.type.padEnd(7)}</span> `
            + r.nodes.map(i => `#${i} ${sample.nodes[i].class}`).join('  ·  ') + '</span>'
        ).join('') || '<span class="row muted">none</span>';
        document.querySelectorAll('.row[data-node]').forEach(r => {
            r.addEventListener('mouseenter', () => litNode(Number(r.dataset.node)));
            r.addEventListener('mouseleave', clear);
        });
        document.querySelectorAll('.row[data-rel]').forEach(r => {
            r.addEventListener('mouseenter', () => litRelation(Number(r.dataset.rel)));
            r.addEventListener('mouseleave', clear);
        });
    }

    async function load(params) {
        const response = await fetch('/api/sample?' + new URLSearchParams(params));
        const data = await response.json();
        if (!response.ok) { $('sampleMeta').textContent = data.error; return; }
        sample = data;
        history.replaceState(null, '', `?seed=${data.seed}`);
        $('indexInput').value = data.seed;
        showBackground();
        $('sampleMeta').textContent = `seed ${data.seed} · ${data.room_count} rooms`;
        const walls = data.nodes.filter(n => n.class.endsWith('wall')).length;
        $('countMeta').textContent = `${data.nodes.length} nodes (${walls} walls) · ${data.relationships.length} relationships`;
        draw();
        lists();
    }

    function showBackground() { $('render').src = `/api/image?seed=${sample.seed}`; }

    const step = d => load({ seed: Math.max(0, Number($('indexInput').value) + d) });
    $('prevButton').onclick = () => step(-1);
    $('nextButton').onclick = () => step(1);
    $('randomButton').onclick = () => load({ seed: Math.floor(Math.random() * 1e6) });
    $('indexInput').onkeydown = e => { if (e.key === 'Enter') step(0); };
    const toggle = (box, cls) => $(box).onchange = e => $('frame').classList.toggle(cls, !e.target.checked);
    toggle('showImage', 'hide-image'); toggle('showRecord', 'hide-record'); toggle('showRelations', 'hide-relations');
    toggle('showSource', 'hide-source'); toggle('showFurniture', 'hide-furniture');
    $('frame').classList.add('hide-source', 'hide-relations');
    window.addEventListener('keydown', e => {
        if (['INPUT', 'SELECT'].includes(e.target.tagName)) return;
        if (e.key === 'ArrowLeft') { e.preventDefault(); step(-1); }
        if (e.key === 'ArrowRight') { e.preventDefault(); step(1); }
    });

    window.addEventListener('DOMContentLoaded', async () => {
        info = await (await fetch('/api/dataset')).json();
        $('datasetInfo').textContent = `generated live · render ${info.size} px`;
        $('counts').innerHTML = '<span class="row muted">counts are per plan, in the lists above</span>';
        const asked = new URLSearchParams(location.search);
        await load({ seed: asked.get('seed') || 0 });
    });
</script>
</body>
</html>
"""


class Generator:
    def __init__(self, size: int, furnished: bool = True):
        self.size, self.furnished = size, furnished

    def summary(self) -> dict:
        return {"size": self.size}

    def sample(self, seed: int) -> dict:
        plan = synthetic.generate(seed)
        record = synthetic.record_of(plan, self.size)
        if self.furnished:
            record["furnishing"] = furnish.furnishing_of(plan, seed, self.size)
            record["furniture"] = icons.furniture_record(record["furnishing"])
        scale, origin = synthetic.frame(plan, self.size)
        cell = synthetic.CELL * scale
        rows, cols = np.nonzero(plan.rooms >= 0)
        rooms = [{"x": round(float(origin[0] + c * cell), 2), "y": round(float(origin[1] + r * cell), 2),
                  "size": round(float(cell) + 0.3, 2), "kind": plan.kinds[plan.rooms[r, c]]} for r, c in zip(rows[::1], cols[::1])]
        labels = []
        for room, kind in enumerate(plan.kinds):
            r, c = np.nonzero(plan.rooms == room)
            labels.append({"x": round(float(origin[0] + (c.mean() + 0.5) * cell), 1), "y": round(float(origin[1] + (r.mean() + 0.5) * cell), 1), "kind": kind})
        return {**record, "seed": seed, "room_count": len(plan.kinds), "rooms": rooms, "labels": labels}

    def record(self, seed: int) -> dict:
        plan = synthetic.generate(seed)
        record = synthetic.record_of(plan, self.size)
        if self.furnished:
            record["furnishing"] = furnish.furnishing_of(plan, seed, self.size)
        return record

    def image(self, seed: int) -> io.BytesIO:
        png = io.BytesIO()
        Image.fromarray(draw.draw(self.record(seed), self.size)).save(png, format="PNG")
        png.seek(0)
        return png


def create_app(size: int) -> Flask:
    generator = Generator(size)
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template_string(HTML_TEMPLATE)

    @app.get("/api/dataset")
    def dataset_info():
        return jsonify(generator.summary())

    @app.get("/api/sample")
    def sample():
        try:
            return jsonify(generator.sample(int(request.args.get("seed", 0))))
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

    @app.get("/api/image")
    def image():
        return send_file(generator.image(int(request.args.get("seed", 0))), mimetype="image/png")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample synthetic floor plans in a local web UI.")
    parser.add_argument("--size", type=int, default=512, help="Side of the render in pixels.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.size)
    print(f"Serving synthetic plans at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
