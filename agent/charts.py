"""Shared chart chrome and palette, plus the base64-PNG encode helper.

Centralized so every chart in the app — tool plots, the column profiler —
reads as one visual system instead of each matplotlib call carrying its own
raw defaults (a full black box border, the stock matplotlib blue). Colors
are the validated reference palette from the project's dataviz skill: fixed
categorical order, one hue for magnitude, a blue/red diverging pair for
correlation.
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── Palette (validated: light-mode worst adjacent categorical CVD ΔE 24.2,
# well clear of the ≥12 target — see the dataviz skill's reference palette) ────

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Fixed order — never cycle or reassign per filter; an 9th+ series folds into "Other".
CATEGORICAL = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
OTHER_COLOR = INK_MUTED
MAX_CATEGORICAL_SERIES = len(CATEGORICAL)

SEQUENTIAL = CATEGORICAL[0]  # single-hue magnitude (histograms, single-series bars)

# Correlation heatmaps: low (-1) cool/blue, high (+1) warm/red, neutral gray at 0.
DIVERGING_CMAP = LinearSegmentedColormap.from_list("diverging_blue_red", ["#2a78d6", "#f0efec", "#e34948"])

# Magnitude-only heatmaps (e.g. retention %): one hue, light (near zero, recedes
# toward the surface) to dark (high) — never a rainbow, never diverging.
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("sequential_blue", [SURFACE, "#2a78d6"])


def style_axes(ax, *, grid: str = "y") -> None:
    """Shared chart chrome: recessive spines/gridlines, muted ticks, consistent
    ink. grid is "y", "x", "both", or "" (heatmaps/no grid)."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.title.set_color(INK_PRIMARY)
    ax.set_axisbelow(True)
    if grid in ("y", "both"):
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8)
    if grid in ("x", "both"):
        ax.grid(axis="x", color=GRIDLINE, linewidth=0.8)


def new_figure(figsize=(7, 4), grid: str = "y"):
    """A figure/axes pair pre-styled with the shared chart chrome."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax, grid=grid)
    return fig, ax


def categorical_colors(labels: list[str]) -> dict[str, str]:
    """Map each label to a fixed-order categorical color, sorted for a
    deterministic assignment. Labels beyond the palette's 8 slots share
    OTHER_COLOR rather than generating new hues (never cycle the palette)."""
    ordered = sorted(labels, key=str)
    colors = {}
    for i, label in enumerate(ordered):
        colors[label] = CATEGORICAL[i] if i < MAX_CATEGORICAL_SERIES else OTHER_COLOR
    return colors


def fig_to_base64_png(fig, dpi: int = 120, facecolor=None) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=facecolor if facecolor is not None else fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
