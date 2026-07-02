"""Unit tests for per-column deep-dive profiling (no API key required)."""

import base64

import pandas as pd

from agent.profiling import plot_column_distribution, profile_column


def _df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 25, None],
            "department": ["Eng", "HR", "Eng", "Sales", "HR", "Eng"],
            "active": [True, False, True, True, False, True],
            "signup": pd.to_datetime(["2024-01-01", "2024-02-15", None, "2024-03-10", "2024-01-20", "2024-04-01"]),
        }
    )


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


def test_profile_numeric_column():
    profile = profile_column(_df(), "age")
    assert profile["kind"] == "numeric"
    assert profile["missing"] == 1
    assert profile["mean"] == 31.0
    assert profile["min"] == 25.0
    assert profile["max"] == 40.0


def test_profile_categorical_column():
    profile = profile_column(_df(), "department")
    assert profile["kind"] == "categorical"
    assert profile["unique"] == 3
    top = {v["value"]: v["count"] for v in profile["top_values"]}
    assert top["Eng"] == 3


def test_profile_boolean_column():
    profile = profile_column(_df(), "active")
    assert profile["kind"] == "boolean"
    assert profile["value_counts"]["True"] == 4
    assert profile["value_counts"]["False"] == 2


def test_profile_datetime_column():
    profile = profile_column(_df(), "signup")
    assert profile["kind"] == "datetime"
    assert profile["missing"] == 1
    assert profile["min"] == "2024-01-01T00:00:00"
    assert profile["max"] == "2024-04-01T00:00:00"


def test_profile_column_handles_unhashable_values_without_crashing():
    df = pd.DataFrame({"meta": [{"a": 1}, {"a": 2}, {"a": 1}]})
    profile = profile_column(df, "meta")
    assert profile["kind"] == "categorical"
    assert profile["unique"] is None


def test_profile_all_missing_numeric_column():
    df = pd.DataFrame({"a": pd.Series([None, None, None], dtype="float64")})
    profile = profile_column(df, "a")
    assert profile["kind"] == "numeric"
    assert profile["missing"] == 3
    assert "mean" not in profile


def test_plot_column_distribution_numeric():
    assert _is_valid_png(plot_column_distribution(_df(), "age"))


def test_plot_column_distribution_categorical():
    assert _is_valid_png(plot_column_distribution(_df(), "department"))


def test_plot_column_distribution_none_when_all_missing():
    df = pd.DataFrame({"a": [None, None]})
    assert plot_column_distribution(df, "a") is None


def test_plot_column_distribution_handles_unhashable_values_without_crashing():
    # pandas' value_counts() tolerates unhashable cell values (dicts/lists) even
    # though set()/nunique() don't — this must not crash either way.
    df = pd.DataFrame({"meta": [{"a": 1}, {"a": 2}]})
    result = plot_column_distribution(df, "meta")
    assert result is None or _is_valid_png(result)
