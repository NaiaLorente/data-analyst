"""Unit tests for the 'What Changed' drift/root-cause engine (no API key required)."""

import pandas as pd
from agent.drift import (
    common_categorical_columns,
    common_numeric_columns,
    compare_categories,
    compare_metrics,
    compare_schemas,
    decompose_aggregates,
    drift_report_to_html,
    drift_report_to_markdown,
    explain_driver,
    generate_drift_report,
    split_by_column,
    split_by_range,
    suggest_metric_column,
    suggest_segment_column,
    suggest_split_column,
)


def _before():
    return pd.DataFrame(
        {
            "region": ["West", "West", "East", "East", "North"],
            "revenue": [1000, 1000, 500, 500, 300],
            "status": ["active", "active", "active", "churned", "active"],
        }
    )


def _after_mix_shift():
    # Overall revenue drops, driven almost entirely by West collapsing,
    # while East and North both grow — a classic mix-shift.
    return pd.DataFrame(
        {
            "region": ["West", "West", "East", "East", "North", "North"],
            "revenue": [100, 100, 600, 600, 400, 400],
            "status": ["active", "churned", "active", "active", "active", "active"],
        }
    )


def test_compare_schemas_detects_added_and_removed_columns():
    df_a = pd.DataFrame({"a": [1], "b": [2]})
    df_b = pd.DataFrame({"a": [1], "c": [3]})
    schema = compare_schemas(df_a, df_b)
    assert schema["added_columns"] == ["c"]
    assert schema["removed_columns"] == ["b"]
    assert schema["common_columns"] == ["a"]


def test_compare_schemas_detects_dtype_change():
    df_a = pd.DataFrame({"a": [1, 2]})
    df_b = pd.DataFrame({"a": ["1", "2"]})
    schema = compare_schemas(df_a, df_b)
    assert schema["dtype_changes"][0]["column"] == "a"


def test_compare_schemas_row_counts():
    df_a = pd.DataFrame({"a": [1, 2]})
    df_b = pd.DataFrame({"a": [1, 2, 3, 4]})
    schema = compare_schemas(df_a, df_b)
    assert schema["row_count_before"] == 2
    assert schema["row_count_after"] == 4
    assert schema["row_count_pct_change"] == 100.0


def test_common_numeric_and_categorical_columns():
    before, after = _before(), _after_mix_shift()
    assert common_numeric_columns(before, after) == ["revenue"]
    assert set(common_categorical_columns(before, after)) == {"region", "status"}


def test_compare_metrics_ranks_by_magnitude_of_change():
    before, after = _before(), _after_mix_shift()
    metrics = compare_metrics(before, after)
    assert metrics[0]["column"] == "revenue"
    assert metrics[0]["sum_before"] == 3300
    assert metrics[0]["sum_after"] == 2200


def test_compare_categories_detects_share_shift_and_new_dropped_values():
    df_a = pd.DataFrame({"tier": ["gold", "gold", "silver"]})
    df_b = pd.DataFrame({"tier": ["gold", "silver", "silver", "platinum"]})
    result = compare_categories(df_a, df_b)[0]
    assert result["column"] == "tier"
    assert "platinum" in result["new_categories"]
    assert result["dropped_categories"] == []


def test_explain_driver_identifies_biggest_contributor():
    before, after = _before(), _after_mix_shift()
    driver = explain_driver(before, after, metric="revenue", segment="region")
    assert driver["total_before"] == 3300
    assert driver["total_after"] == 2200
    assert driver["total_delta"] == -1100
    top = driver["by_segment"][0]
    assert top["segment"] == "West"
    assert top["delta"] == -1800


def test_explain_driver_flags_mix_shift():
    before, after = _before(), _after_mix_shift()
    driver = explain_driver(before, after, metric="revenue", segment="region")
    assert driver["mix_shift_warning"] is not None


def test_explain_driver_no_mix_shift_when_uniform():
    df_a = pd.DataFrame({"region": ["A", "B"], "revenue": [100, 100]})
    df_b = pd.DataFrame({"region": ["A", "B"], "revenue": [200, 200]})
    driver = explain_driver(df_a, df_b, metric="revenue", segment="region")
    assert driver["mix_shift_warning"] is None


def test_explain_driver_mean_aggregation():
    df_a = pd.DataFrame({"region": ["A", "A"], "revenue": [100, 300]})
    df_b = pd.DataFrame({"region": ["A", "A"], "revenue": [200, 200]})
    driver = explain_driver(df_a, df_b, metric="revenue", segment="region", agg="mean")
    assert driver["total_before"] == 200  # mean of [100, 300]
    assert driver["total_after"] == 200  # mean of [200, 200]


def test_generate_drift_report_without_driver():
    before, after = _before(), _after_mix_shift()
    report = generate_drift_report(before, after)
    assert "driver" not in report
    assert "schema" in report and "metrics" in report and "categories" in report


def test_generate_drift_report_with_driver():
    before, after = _before(), _after_mix_shift()
    report = generate_drift_report(before, after, metric="revenue", segment="region")
    assert "driver" in report


def test_drift_report_to_markdown_and_html_render():
    before, after = _before(), _after_mix_shift()
    report = generate_drift_report(before, after, metric="revenue", segment="region")
    md = drift_report_to_markdown(report)
    html = drift_report_to_html(report)
    assert md.startswith("- ")
    assert "revenue" in md
    assert "<ul>" in html and "</ul>" in html


def test_drift_report_to_markdown_no_differences():
    df = pd.DataFrame({"a": [1, 2, 3]})
    report = generate_drift_report(df, df.copy())
    md = drift_report_to_markdown(report)
    assert "Row count" in md  # row count line always present


def test_split_by_column_matches_exact_values():
    df = pd.DataFrame({"month": ["Jan", "Jan", "Feb", "Feb", "Mar"], "revenue": [1, 2, 3, 4, 5]})
    jan, feb = split_by_column(df, "month", "Jan", "Feb")
    assert len(jan) == 2
    assert len(feb) == 2
    assert jan["revenue"].sum() == 3
    assert feb["revenue"].sum() == 7


def test_split_by_column_no_match_returns_empty():
    df = pd.DataFrame({"month": ["Jan", "Feb"], "revenue": [1, 2]})
    jan, missing = split_by_column(df, "month", "Jan", "Nonexistent")
    assert len(jan) == 1
    assert missing.empty


def test_split_by_range_inclusive_bounds():
    df = pd.DataFrame({"day": [1, 2, 3, 4, 5, 6], "revenue": [10, 20, 30, 40, 50, 60]})
    first_half, second_half = split_by_range(df, "day", (1, 3), (4, 6))
    assert first_half["revenue"].sum() == 60
    assert second_half["revenue"].sum() == 150


def test_split_by_column_feeds_generate_drift_report():
    df = pd.DataFrame(
        {
            "month": ["Jan"] * 3 + ["Feb"] * 3,
            "region": ["West", "East", "East", "West", "East", "East"],
            "revenue": [100, 50, 50, 20, 60, 60],
        }
    )
    jan, feb = split_by_column(df, "month", "Jan", "Feb")
    report = generate_drift_report(jan, feb, metric="revenue", segment="region")
    assert report["driver"]["total_before"] == 200
    assert report["driver"]["total_after"] == 140


def _titanic_like(n=50):
    return pd.DataFrame(
        {
            "PassengerId": range(1, n + 1),
            "Name": [f"Person {i}" for i in range(n)],
            "Fare": [10.0 + i for i in range(n)],
            "Pclass": ([1, 2, 3] * ((n // 3) + 1))[:n],
        }
    )


def test_suggest_split_column_avoids_id_columns():
    df = _titanic_like()
    assert suggest_split_column(df) == "Pclass"


def test_suggest_split_column_falls_back_to_first_when_all_high_cardinality():
    df = pd.DataFrame({"id": range(30), "name": [f"n{i}" for i in range(30)]})
    assert suggest_split_column(df) == "id"


def test_suggest_metric_column_avoids_id_like_names():
    df_a = _titanic_like()
    df_b = _titanic_like()
    assert suggest_metric_column(df_a, df_b) == "Fare"


def test_suggest_metric_column_respects_exclude():
    # Excluding the split column itself avoids a tautological "Survived went
    # from 0 to 1" comparison when splitting a file by Survived.
    df_a = pd.DataFrame({"Survived": [0, 0], "Fare": [10.0, 20.0]})
    df_b = pd.DataFrame({"Survived": [1, 1], "Fare": [30.0, 40.0]})
    assert suggest_metric_column(df_a, df_b, exclude=("Survived",)) == "Fare"


def test_suggest_segment_column_prefers_low_cardinality():
    df_a = _titanic_like()
    df_a["Sex"] = (["male", "female"] * 25)[: len(df_a)]
    df_b = df_a.copy()
    assert suggest_segment_column(df_a, df_b) == "Sex"


def test_suggest_segment_column_falls_back_when_only_high_cardinality_available():
    df_a = _titanic_like()
    df_b = _titanic_like()
    # "Name" is the only categorical column and it's high-cardinality — still
    # better to suggest it than nothing.
    assert suggest_segment_column(df_a, df_b) == "Name"


def test_suggest_segment_column_respects_exclude():
    df = pd.DataFrame({"tier": ["a", "b", "a", "b"], "region": ["x", "y", "x", "y"]})
    assert suggest_segment_column(df, df, exclude=("tier",)) == "region"


def test_decompose_aggregates_matches_explain_driver():
    before, after = _before(), _after_mix_shift()
    from_dataframes = explain_driver(before, after, metric="revenue", segment="region")

    agg_a = before.groupby("region")["revenue"].sum().to_dict()
    agg_b = after.groupby("region")["revenue"].sum().to_dict()
    from_aggregates = decompose_aggregates(agg_a, agg_b, metric="revenue", segment="region")

    assert from_dataframes == from_aggregates


def test_decompose_aggregates_no_raw_data_needed():
    # The whole point: reconstruct a driver breakdown from just two small
    # {segment: value} dicts, as a saved project would store, with zero
    # access to the original rows.
    agg_a = {"West": 20000.0, "East": 10000.0}
    agg_b = {"West": 2000.0, "East": 12000.0}
    result = decompose_aggregates(agg_a, agg_b, metric="revenue", segment="region")
    assert result["total_before"] == 30000.0
    assert result["total_after"] == 14000.0
    assert result["by_segment"][0]["segment"] == "West"
    assert result["by_segment"][0]["delta"] == -18000.0


def test_compare_categories_handles_unhashable_values_without_crashing():
    # A column of dicts (e.g. from a JSON upload with nested objects) breaks
    # Python's set() even though pandas' value_counts() tolerates it — this
    # must not crash.
    df_a = pd.DataFrame({"meta": [{"a": 1}, {"a": 2}]})
    df_b = pd.DataFrame({"meta": [{"a": 1}, {"a": 3}]})
    result = compare_categories(df_a, df_b)
    assert result[0]["column"] == "meta"
    assert "{'a': 3}" in result[0]["new_categories"]
    assert "{'a': 2}" in result[0]["dropped_categories"]


def test_suggest_split_column_handles_unhashable_values_without_crashing():
    df = pd.DataFrame({"meta": [{"a": 1}] * 10, "region": ["x", "y"] * 5})
    assert suggest_split_column(df) == "region"


def test_suggest_segment_column_handles_unhashable_values_without_crashing():
    df = pd.DataFrame({"meta": [{"a": 1}] * 10, "region": ["x", "y"] * 5})
    assert suggest_segment_column(df, df) == "region"
