"""Unit tests for the shared matplotlib-figure-to-base64-PNG helper."""

import base64

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.charts import fig_to_base64_png


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


def test_fig_to_base64_png_produces_valid_png():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    assert _is_valid_png(fig_to_base64_png(fig))


def test_fig_to_base64_png_respects_explicit_facecolor():
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    assert _is_valid_png(fig_to_base64_png(fig, dpi=100, facecolor="#0f172a"))


def test_fig_to_base64_png_defaults_facecolor_to_figure_own():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#123456")
    ax.plot([1, 2], [1, 2])
    assert _is_valid_png(fig_to_base64_png(fig))
