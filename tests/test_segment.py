"""Unit tests for segmentation/clustering (no API key required)."""

import base64

import numpy as np
import pandas as pd
from agent.segment import find_segments, plot_segments


def _three_group_df(n=150):
    rng = np.random.RandomState(0)
    per_group = n // 3
    a = pd.DataFrame({"income": rng.normal(30000, 2000, per_group), "spending": rng.normal(500, 50, per_group)})
    b = pd.DataFrame({"income": rng.normal(80000, 2000, per_group), "spending": rng.normal(5000, 200, per_group)})
    c = pd.DataFrame({"income": rng.normal(50000, 2000, per_group), "spending": rng.normal(1500, 100, per_group)})
    return pd.concat([a, b, c], ignore_index=True)


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


def test_find_segments_applicable_with_enough_data():
    result = find_segments(_three_group_df(), n_clusters=3)
    assert result["applicable"] is True
    assert result["n_clusters"] == 3
    assert len(result["clusters"]) == 3


def test_find_segments_cluster_sizes_sum_to_rows_used():
    result = find_segments(_three_group_df(), n_clusters=3)
    assert sum(c["size"] for c in result["clusters"]) == result["n_rows_used"]


def test_find_segments_recovers_well_separated_groups():
    result = find_segments(_three_group_df(), n_clusters=3)
    sizes = sorted(c["size"] for c in result["clusters"])
    # Each synthetic group has 50 rows -- clusters should roughly match, not
    # collapse into one giant cluster and two tiny ones.
    assert all(size > 30 for size in sizes)


def test_find_segments_top_features_include_z_diff():
    result = find_segments(_three_group_df(), n_clusters=3)
    for cluster in result["clusters"]:
        assert "z_diff" in cluster["top_features"][0]
        assert cluster["top_features"][0]["column"] in ("income", "spending")


def test_find_segments_not_applicable_with_one_numeric_column():
    df = pd.DataFrame({"a": range(50)})
    result = find_segments(df, n_clusters=3)
    assert result["applicable"] is False
    assert "reason" in result


def test_find_segments_not_applicable_with_too_few_rows():
    df = pd.DataFrame({"a": range(5), "b": range(5)})
    result = find_segments(df, n_clusters=3)
    assert result["applicable"] is False


def test_find_segments_caps_n_clusters_to_max():
    df = _three_group_df()
    result = find_segments(df, n_clusters=100)
    assert result["n_clusters"] <= 8


def test_find_segments_respects_column_selection():
    df = _three_group_df()
    df["irrelevant"] = np.random.RandomState(1).normal(size=len(df))
    result = find_segments(df, columns=["income", "spending"], n_clusters=3)
    assert result["columns_used"] == ["income", "spending"]


def test_find_segments_drops_rows_with_missing_values():
    df = _three_group_df()
    df.loc[0:5, "income"] = None
    result = find_segments(df, n_clusters=3)
    assert result["n_rows_used"] == len(df) - 6


def test_plot_segments_produces_valid_png():
    result = find_segments(_three_group_df(), n_clusters=3)
    assert _is_valid_png(plot_segments(result))
