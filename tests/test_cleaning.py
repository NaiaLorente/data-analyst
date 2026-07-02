"""Unit tests for the one-click data-cleaning module (no API key required)."""

import numpy as np
import pandas as pd

from agent.cleaning import (
    cap_outliers,
    convert_to_datetime,
    convert_to_numeric,
    count_open_issues,
    detect_all_issues,
    detect_constant_columns,
    detect_duplicates,
    detect_missing,
    detect_outliers,
    detect_type_issues,
    detect_whitespace,
    drop_column,
    drop_missing_rows,
    fill_missing,
    remove_duplicate_rows,
    remove_outlier_rows,
    trim_whitespace,
)


def _messy_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 3, 4, 5, 6, 7, 8, 9],
            "age": [25, 30, 35, 35, np.nan, 28, 32, 31, 29, 500],
            "city": [" NYC", "LA", "LA", "LA", "SF ", "NYC", "LA", "SF", "NYC", "LA"],
        }
    )


def test_detect_duplicates_finds_dupes():
    issue = detect_duplicates(_messy_df())
    assert issue is not None
    assert issue["count"] == 1


def test_detect_duplicates_none_when_clean():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert detect_duplicates(df) is None


def test_detect_missing_flags_column():
    issues = detect_missing(_messy_df())
    cols = [i["column"] for i in issues]
    assert "age" in cols
    assert issues[0]["numeric"] is True


def test_detect_outliers_flags_extreme_value():
    issues = detect_outliers(_messy_df())
    cols = [i["column"] for i in issues]
    assert "age" in cols


def test_detect_outliers_skips_zero_variance_column():
    df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
    assert detect_outliers(df) == []


def test_detect_whitespace_flags_stray_spaces():
    issues = detect_whitespace(_messy_df())
    cols = [i["column"] for i in issues]
    assert "city" in cols
    assert issues[0]["count"] == 2  # " NYC" and "SF "


def test_detect_all_issues_and_count_open_issues():
    issues = detect_all_issues(_messy_df())
    assert count_open_issues(issues) > 0

    clean = pd.DataFrame({"a": [1, 2, 3]})
    clean_issues = detect_all_issues(clean)
    assert count_open_issues(clean_issues) == 0


def test_remove_duplicate_rows():
    df = _messy_df()
    cleaned = remove_duplicate_rows(df)
    assert len(cleaned) == len(df) - 1
    assert detect_duplicates(cleaned) is None


def test_remove_duplicate_rows_does_not_mutate_input():
    df = _messy_df()
    original_len = len(df)
    remove_duplicate_rows(df)
    assert len(df) == original_len


def test_fill_missing_median():
    df = _messy_df()
    cleaned = fill_missing(df, "age", "median")
    assert cleaned["age"].isna().sum() == 0
    assert df["age"].isna().sum() == 1  # original untouched


def test_fill_missing_mode_on_categorical():
    df = pd.DataFrame({"c": ["x", "x", None, "y"]})
    cleaned = fill_missing(df, "c", "mode")
    assert cleaned["c"].isna().sum() == 0
    assert cleaned["c"].iloc[2] == "x"


def test_fill_missing_unknown_strategy_raises():
    df = pd.DataFrame({"a": [1.0, None]})
    try:
        fill_missing(df, "a", "bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_drop_missing_rows():
    df = _messy_df()
    cleaned = drop_missing_rows(df, "age")
    assert cleaned["age"].isna().sum() == 0
    assert len(cleaned) == len(df) - 1


def test_cap_outliers_reduces_extreme_value():
    df = _messy_df()
    cleaned = cap_outliers(df, "age")
    assert cleaned["age"].max() < 500
    assert df["age"].max() == 500  # original untouched


def test_cap_outliers_noop_on_zero_variance_column():
    df = pd.DataFrame({"a": [5, 5, 5, 5]})
    cleaned = cap_outliers(df, "a")
    assert cleaned["a"].tolist() == df["a"].tolist()


def test_remove_outlier_rows():
    df = _messy_df()
    cleaned = remove_outlier_rows(df, "age")
    assert 500 not in cleaned["age"].tolist()
    assert len(cleaned) < len(df)


def test_trim_whitespace():
    df = _messy_df()
    cleaned = trim_whitespace(df, "city")
    assert cleaned["city"].iloc[0] == "NYC"
    assert cleaned["city"].iloc[4] == "SF"
    assert df["city"].iloc[0] == " NYC"  # original untouched


def test_trim_whitespace_leaves_non_string_values_alone():
    df = pd.DataFrame({"c": ["  a  ", None, 5]})
    cleaned = trim_whitespace(df, "c")
    assert cleaned["c"].iloc[0] == "a"
    assert cleaned["c"].iloc[2] == 5


def test_detect_type_issues_finds_numeric_stored_as_text():
    df = pd.DataFrame({"price": ["10", "20", "30.5", "40"]})
    issues = detect_type_issues(df)
    assert issues == [{"kind": "type_numeric", "column": "price"}]


def test_detect_type_issues_finds_dates_stored_as_text():
    df = pd.DataFrame({"signup": ["2024-01-01", "2024-02-15", "2024-03-10", "2024-04-01"]})
    issues = detect_type_issues(df)
    assert issues == [{"kind": "type_datetime", "column": "signup"}]


def test_detect_type_issues_ignores_short_digit_ids():
    df = pd.DataFrame({"zip": ["10001", "94107", "60614", "73301"]})
    issues = detect_type_issues(df)
    # Should be flagged numeric (they're digits), not misdetected as dates.
    assert issues == [{"kind": "type_numeric", "column": "zip"}]


def test_detect_type_issues_ignores_genuine_text():
    df = pd.DataFrame({"name": ["Alice", "Bob", "Carol", "Dave"]})
    assert detect_type_issues(df) == []


def test_convert_to_numeric():
    df = pd.DataFrame({"price": ["10", "20", "not-a-number"]})
    cleaned = convert_to_numeric(df, "price")
    assert cleaned["price"].tolist()[:2] == [10.0, 20.0]
    assert pd.isna(cleaned["price"].iloc[2])
    assert df["price"].iloc[0] == "10"  # original untouched


def test_convert_to_datetime():
    df = pd.DataFrame({"signup": ["2024-01-01", "2024-02-15"]})
    cleaned = convert_to_datetime(df, "signup")
    assert pd.api.types.is_datetime64_any_dtype(cleaned["signup"])
    assert df["signup"].iloc[0] == "2024-01-01"  # original untouched


def test_detect_constant_columns_finds_constant_and_empty():
    df = pd.DataFrame({"country": ["US", "US", "US"], "notes": [None, None, None], "id": [1, 2, 3]})
    issues = detect_constant_columns(df)
    kinds = {i["column"]: i["kind"] for i in issues}
    assert kinds["country"] == "constant_column"
    assert kinds["notes"] == "empty_column"
    assert "id" not in kinds


def test_drop_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    cleaned = drop_column(df, "b")
    assert list(cleaned.columns) == ["a"]
    assert "b" in df.columns  # original untouched


def test_detect_all_issues_includes_types_and_constant_columns():
    df = pd.DataFrame({"price": ["10", "20", "30"], "country": ["US", "US", "US"]})
    issues = detect_all_issues(df)
    assert issues["types"] == [{"kind": "type_numeric", "column": "price"}]
    assert issues["constant_columns"][0]["column"] == "country"
    assert count_open_issues(issues) == 2
