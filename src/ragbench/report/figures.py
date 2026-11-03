"""Static Pareto figure for the README (Task 6.1), written as dependency-free SVG.

Quality (nDCG@10) against p95 latency on a log axis, one point per dev run, coloured by
retrieval mode with the reference categorical palette (slots 1-3), the Pareto frontier
drawn as a dotted step line, and a legend. Output is deterministic for the same runs.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape
from pathlib import Path

from ragbench.eval.results import RunResult
from ragbench.sweep.select import describe

MODE_COLORS = {"dense": "#2a78d6", "hybrid": "#eb6834", "sparse": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
W, H = 760, 420
LEFT, RIGHT, TOP, BOTTOM = 64, 20, 44, 52


def _mode(run: RunResult) -> str:
    return "sparse" if run.config.backend == "control_bm25" else run.config.retrieval.mode


def frontier(points: Sequence[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Points not beaten on both lower latency and higher quality, by latency."""
    out: list[tuple[float, float, str]] = []
    best = -math.inf
    for x, y, key in sorted(points, key=lambda p: (p[0], -p[1])):
        if y > best:
            out.append((x, y, key))
            best = y
    return out


def pareto_svg(runs: Sequence[RunResult], metric: str = "ndcg@10") -> str:
    pts = [(r.latency_ms["total"].p95, r.metrics[metric].mean, r.config_id) for r in runs]
    if not pts:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="80">'
            f'<text x="10" y="40" fill="{INK}" font-family="sans-serif">not yet run</text></svg>'
        )
    by_id = {r.config_id: r for r in runs}
    lx = [math.log10(max(p[0], 1e-3)) for p in pts]
    x0, x1 = math.floor(min(lx)), math.ceil(max(lx))
    if x0 == x1:
        x1 = x0 + 1

    def sx(v: float) -> float:
        return LEFT + (math.log10(max(v, 1e-3)) - x0) / (x1 - x0) * (W - LEFT - RIGHT)

    def sy(v: float) -> float:
        return TOP + (1.0 - v) * (H - TOP - BOTTOM)

    font = 'font-family="system-ui, sans-serif"'
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" role="img" aria-label="{metric} versus p95 latency">',
        f'<rect width="{W}" height="{H}" fill="{SURFACE}"/>',
    ]
    for i in range(6):
        v = i / 5
        out.append(
            f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
            f'<text x="{LEFT - 8}" y="{sy(v) + 4:.1f}" text-anchor="end" font-size="11" '
            f'fill="{INK_2}" {font}>{v:.1f}</text>'
        )
    for e in range(x0, x1 + 1):
        x = LEFT + (e - x0) / (x1 - x0) * (W - LEFT - RIGHT)
        label = f"{10**e:g}"
        out.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{TOP}" y2="{H - BOTTOM}" stroke="{GRID}"/>'
            f'<text x="{x:.1f}" y="{H - BOTTOM + 16}" text-anchor="middle" font-size="11" '
            f'fill="{INK_2}" {font}>{label}</text>'
        )
    out.append(
        f'<text x="{(LEFT + W - RIGHT) / 2}" y="{H - 12}" text-anchor="middle" font-size="12" '
        f'fill="{INK}" {font}>p95 total latency (ms, log scale)</text>'
        f'<text transform="translate(16 {(TOP + H - BOTTOM) / 2}) rotate(-90)" '
        f'text-anchor="middle" font-size="12" fill="{INK}" {font}>{escape(metric)}</text>'
    )
    front = frontier(pts)
    path = []
    for i, (x, y, _) in enumerate(front):
        if i == 0:
            path.append(f"M{sx(x):.1f},{sy(y):.1f}")
        else:
            path.append(f"H{sx(x):.1f}V{sy(y):.1f}")
    out.append(
        f'<path d="{" ".join(path)}" fill="none" stroke="{INK_2}" stroke-width="2" '
        f'stroke-dasharray="3 3"/>'
    )
    front_ids = {k for _, _, k in front}
    for x, y, key in sorted(pts, key=lambda p: p[2] in front_ids):
        run = by_id[key]
        r = 6.5 if key in front_ids else 4.5
        out.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r}" fill="{MODE_COLORS[_mode(run)]}" '
            f'stroke="{SURFACE}" stroke-width="2"><title>{escape(describe(run.config))}: '
            f"{metric} {y:.3f}, p95 {x:.1f} ms</title></circle>"
        )
    lx_ = LEFT
    for mode, color in [*MODE_COLORS.items()]:
        out.append(
            f'<circle cx="{lx_ + 5}" cy="20" r="5" fill="{color}"/>'
            f'<text x="{lx_ + 14}" y="24" font-size="12" fill="{INK_2}" {font}>{mode}</text>'
        )
        lx_ += 90
    out.append(
        f'<line x1="{lx_}" x2="{lx_ + 22}" y1="20" y2="20" stroke="{INK_2}" stroke-width="2" '
        f'stroke-dasharray="3 3"/><text x="{lx_ + 28}" y="24" font-size="12" fill="{INK_2}" '
        f"{font}>Pareto frontier (larger points)</text>"
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def write_pareto(runs: Sequence[RunResult], path: Path, metric: str = "ndcg@10") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pareto_svg(runs, metric), encoding="utf-8")
    return path
