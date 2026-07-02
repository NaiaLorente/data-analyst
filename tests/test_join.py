"""Unit tests for multi-file join/merge (no API key required)."""

import pandas as pd
from agent.join import join_dataframes, join_stats, suggest_join_keys


def _customers():
    return pd.DataFrame({"customer_id": [1, 2, 3, 4], "name": ["A", "B", "C", "D"]})


def _orders():
    return pd.DataFrame({"order_id": [100, 101, 102], "customer_id": [1, 2, 2], "amount": [10, 20, 30]})


def test_suggest_join_keys_prefers_id_like_names():
    df_a = pd.DataFrame({"customer_id": [1], "region": ["West"]})
    df_b = pd.DataFrame({"customer_id": [1], "region": ["West"]})
    keys = suggest_join_keys(df_a, df_b)
    assert keys[0] == "customer_id"


def test_suggest_join_keys_only_common_columns():
    df_a = pd.DataFrame({"a": [1], "shared": [1]})
    df_b = pd.DataFrame({"b": [1], "shared": [1]})
    assert suggest_join_keys(df_a, df_b) == ["shared"]


def test_suggest_join_keys_empty_when_no_overlap():
    df_a = pd.DataFrame({"a": [1]})
    df_b = pd.DataFrame({"b": [1]})
    assert suggest_join_keys(df_a, df_b) == []


def test_join_dataframes_left_join_keeps_unmatched_rows():
    result = join_dataframes(_customers(), _orders(), "customer_id", "customer_id", how="left")
    assert len(result) == 5  # customer 1 (1 order) + customer 2 (2 orders) + 3, 4 unmatched
    assert result["amount"].isna().sum() == 2


def test_join_dataframes_inner_join_drops_unmatched_rows():
    result = join_dataframes(_customers(), _orders(), "customer_id", "customer_id", how="inner")
    assert len(result) == 3
    assert result["customer_id"].tolist() == [1, 2, 2]


def test_join_dataframes_outer_join():
    result = join_dataframes(_customers(), _orders(), "customer_id", "customer_id", how="outer")
    assert len(result) == 5


def test_join_stats_reports_row_counts_and_match_rate():
    df_a, df_b = _customers(), _orders()
    result = join_dataframes(df_a, df_b, "customer_id", "customer_id", how="left")
    stats = join_stats(df_a, df_b, result, "customer_id", "customer_id")
    assert stats["rows_a"] == 4
    assert stats["rows_b"] == 3
    assert stats["rows_result"] == 5
    assert stats["match_rate"] == 0.5  # 2 of 4 customer_ids in df_a matched


def test_join_stats_full_match_rate_is_one():
    df_a = pd.DataFrame({"id": [1, 2, 3]})
    df_b = pd.DataFrame({"id": [1, 2, 3], "val": ["x", "y", "z"]})
    result = join_dataframes(df_a, df_b, "id", "id", how="inner")
    stats = join_stats(df_a, df_b, result, "id", "id")
    assert stats["match_rate"] == 1.0


def test_join_stats_handles_unhashable_key_values_without_crashing():
    df_a = pd.DataFrame({"id": [{"a": 1}, {"a": 2}]})
    df_b = pd.DataFrame({"id": [{"a": 1}], "val": ["x"]})
    # pandas can't merge on unhashable keys either -- this documents that limit
    # rather than papering over it, but join_stats itself must not crash.
    try:
        result = join_dataframes(df_a, df_b, "id", "id", how="left")
    except TypeError:
        result = pd.DataFrame()
    stats = join_stats(df_a, df_b, result, "id", "id")
    assert stats["match_rate"] is None


def test_join_dataframes_uses_a_b_suffixes_for_overlapping_columns():
    df_a = pd.DataFrame({"id": [1], "value": ["from_a"]})
    df_b = pd.DataFrame({"id": [1], "value": ["from_b"]})
    result = join_dataframes(df_a, df_b, "id", "id", how="inner")
    assert "value_a" in result.columns
    assert "value_b" in result.columns
