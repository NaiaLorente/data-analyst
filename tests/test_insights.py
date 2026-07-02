"""Unit tests for the zero-cost, LLM-free auto-insights module."""

import pandas as pd
from agent.insights import (
    escape_markdown_math,
    generate_insights,
    insights_to_html,
    insights_to_markdown,
    safe_nunique,
)


def _df():
    return pd.DataFrame(
        {
            "age": [25, 30, 35, 40, 25, None, 200],
            "salary": [50000, 60000, 70000, 80000, 55000, 62000, 58000],
            "department": ["Eng", "HR", "Eng", "Sales", "HR", "Eng", "Eng"],
            "user_id": [f"id-{i}" for i in range(7)],
        }
    )


def test_generate_insights_shape():
    insights = generate_insights(_df())
    assert insights["shape"] == {"rows": 7, "columns": 4}


def test_generate_insights_missing():
    insights = generate_insights(_df())
    cols = [m["column"] for m in insights["missing"]]
    assert "age" in cols


def test_generate_insights_correlation():
    insights = generate_insights(_df())
    assert isinstance(insights["top_correlations"], list)
    for pair in insights["top_correlations"]:
        assert -1 <= pair["correlation"] <= 1


def test_generate_insights_outliers():
    insights = generate_insights(_df())
    cols = [o["column"] for o in insights["outlier_columns"]]
    assert "age" in cols  # 200 is a clear outlier


def test_generate_insights_high_cardinality():
    insights = generate_insights(_df())
    assert "user_id" in insights["high_cardinality_columns"]


def test_generate_insights_no_duplicates_by_default():
    insights = generate_insights(_df())
    assert insights["duplicate_rows"] == 0


def test_insights_to_markdown_and_html():
    insights = generate_insights(_df())
    md = insights_to_markdown(insights)
    html = insights_to_html(insights)
    assert "rows" in md
    assert md.startswith("- ")
    assert "<ul>" in html and "</ul>" in html


def test_generate_insights_single_column_no_correlation():
    df = pd.DataFrame({"a": [1, 2, 3]})
    insights = generate_insights(df)
    assert insights["top_correlations"] == []


def test_generate_insights_handles_unhashable_values_without_crashing():
    # A column of dicts (e.g. from a JSON upload with nested objects) can't be
    # hashed, which breaks pandas' nunique()/duplicated() — this must not crash.
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "meta": [{"a": 1}, {"a": 2}, {"a": 3}],
            "val": [10.0, 20.0, 30.0],
        }
    )
    insights = generate_insights(df)
    assert insights["shape"] == {"rows": 3, "columns": 3}
    assert "meta" in insights["high_cardinality_columns"]
    assert insights["duplicate_rows"] == 0


def test_generate_insights_detects_duplicates_with_unhashable_column_present():
    df = pd.DataFrame(
        {
            "id": [1, 1],
            "meta": [{"a": 1}, {"a": 1}],
        }
    )
    insights = generate_insights(df)
    assert insights["duplicate_rows"] == 1


def test_safe_nunique_returns_none_on_unhashable_values():
    series = pd.Series([{"a": 1}, {"a": 2}, {"a": 1}])
    assert safe_nunique(series) is None


def test_safe_nunique_normal_case_unaffected():
    series = pd.Series(["a", "b", "a"])
    assert safe_nunique(series) == 2


def test_escape_markdown_math_escapes_dollar_signs():
    text = "Annual Income (k$) ↔ Age (+0.38), Spending Score ↔ Annual Income (k$) (+0.09)"
    escaped = escape_markdown_math(text)
    assert "\\$" in escaped
    assert "$" not in escaped.replace("\\$", "")


def test_escape_markdown_math_noop_without_dollar_signs():
    text = "Age ↔ Income (+0.38)"
    assert escape_markdown_math(text) == text


def test_insights_to_markdown_with_dollar_column_name_is_escapable():
    df = pd.DataFrame({"Age": [20, 30, 40], "Annual Income (k$)": [15, 20, 25]})
    md = insights_to_markdown(generate_insights(df))
    escaped = escape_markdown_math(md)
    # every literal '$' must be preceded by a backslash after escaping
    for i, ch in enumerate(escaped):
        if ch == "$":
            assert escaped[i - 1] == "\\"
