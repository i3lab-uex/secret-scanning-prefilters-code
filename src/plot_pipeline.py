"""Generate the secret-scanning pipeline schematic (Figure 1).

Shows where prefilters sit relative to scanner output and triage: scanners
already discard candidates before emitting output, so the filters this paper
measures consume recall on top of that.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FIGURES_DIR = _PROJECT_ROOT / "figures"

_COLORS = {
    "blue": "#4c72b0",
    "orange": "#dd8452",
    "green": "#55a868",
    "gray": "#8c8c8c",
}

# (title, body lines, colour, is_scope)
_STAGES = [
    ("Repository",
     ["every credential", "present in the code"],
     _COLORS["gray"], False),
    ("Detection and\nbuilt-in filtering",
     ["regex patterns", "per-rule entropy gates", "path allowlists"],
     _COLORS["blue"], False),
    ("Pre-triage filters",
     ["entropy thresholds", "path and filename rules", "value-shape heuristics"],
     _COLORS["orange"], True),
    ("Triage",
     ["human review", "automated verification", "LLM classification"],
     _COLORS["green"], False),
]

_X0, _X1 = 0.02, 0.985
_GAP = 0.038
_BOX_BOTTOM, _BOX_HEIGHT = 0.46, 0.42


def _box_spans() -> list[tuple[float, float]]:
    n_gaps = len(_STAGES) - 1
    width = (_X1 - _X0 - n_gaps * _GAP) / len(_STAGES)
    return [(_X0 + i * (width + _GAP), _X0 + i * (width + _GAP) + width)
            for i in range(len(_STAGES))]


def _draw_stage(ax, x0, x1, title, lines, colour, is_scope) -> None:
    ax.add_patch(FancyBboxPatch(
        (x0, _BOX_BOTTOM), x1 - x0, _BOX_HEIGHT,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        linewidth=2.2 if is_scope else 1.2,
        edgecolor=colour,
        facecolor=colour, alpha=0.11 if is_scope else 0.055,
        zorder=2,
    ))
    cx = (x0 + x1) / 2
    top = _BOX_BOTTOM + _BOX_HEIGHT
    # Title centred in a fixed header zone so body text aligns across boxes,
    # regardless of whether the title wraps to two lines.
    ax.text(cx, top - 0.085, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=colour, zorder=3,
            linespacing=1.35)
    y = top - 0.205
    for line in lines:
        ax.text(cx, y, line, ha="center", va="center",
                fontsize=8.8, color="#333333", zorder=3)
        y -= 0.072
    if is_scope:
        ax.text(cx, _BOX_BOTTOM - 0.085, "the filters measured in this paper",
                ha="center", va="center", fontsize=9.3, style="italic",
                fontweight="bold", color=colour, zorder=3)


def plot_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 2.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    spans = _box_spans()
    for (x0, x1), stage in zip(spans, _STAGES):
        _draw_stage(ax, x0, x1, *stage)

    mid_y = _BOX_BOTTOM + _BOX_HEIGHT / 2
    for i in range(len(spans) - 1):
        ax.add_patch(FancyArrowPatch(
            (spans[i][1] + 0.004, mid_y), (spans[i + 1][0] - 0.004, mid_y),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.3, color="#666666", zorder=1,
        ))

    # recall-budget annotation
    ax.add_patch(FancyArrowPatch(
        (_X0, 0.235), (_X1, 0.235), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.3, color=_COLORS["gray"], zorder=1,
    ))
    ax.text((_X0 + _X1) / 2, 0.135, "recall can only decrease",
            ha="center", va="center", fontsize=9.8, color="#666666")

    _FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIGURES_DIR / "pipeline_stages.pdf", bbox_inches="tight")
    plt.close(fig)
