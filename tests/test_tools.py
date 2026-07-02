"""Unit tests for the data analysis tools (no API key required)."""

import pandas as pd
import pytest
import base64

from agent.tools import (
    compare_groups,
    filter_rows,
    get_correlation_matrix,
    get_summary,
    get_value_counts,
    plot_scatter,
    set_dataframe,
    tool_call_to_code,
)


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def sample_df():
    df = pd.DataFrame({
        "age": [25, 30, 35, 40, 25, None],
        "salary": [50000, 60000, 70000, 80000, 55000, 62000],
        "department": ["Eng", "HR", "Eng", "Sales", "HR", "Eng"],
    })
    set_dataframe(df)
    return df


def test_get_summary_all():
    result = get_summary()
    assert result["shape"] == [6, 3]
    assert "age" in result["dtypes"]


def test_get_summary_subset():
    result = get_summary(columns=["salary"])
    assert list(result["dtypes"].keys()) == ["salary"]


def test_get_value_counts():
    result = get_value_counts("department")
    assert result["Eng"] == 3


def test_get_correlation_matrix():
    result = get_correlation_matrix()
    assert "salary" in result
    assert result["salary"]["salary"] == pytest.approx(1.0)


def test_filter_rows_eq():
    result = filter_rows({"department": {"op": "eq", "value": "Eng"}})
    assert result["shape"][0] == 3


def test_filter_rows_gt():
    result = filter_rows({"salary": {"op": "gt", "value": 60000}})
    assert result["shape"][0] == 3


def test_filter_rows_contains():
    result = filter_rows({"department": {"op": "contains", "value": "ng"}})
    assert result["shape"][0] == 3


def test_filter_rows_gte_lte():
    assert filter_rows({"salary": {"op": "gte", "value": 60000}})["shape"][0] == 4
    assert filter_rows({"salary": {"op": "lte", "value": 55000}})["shape"][0] == 2


def test_filter_rows_multiple_conditions_are_anded():
    result = filter_rows(
        {
            "department": {"op": "eq", "value": "Eng"},
            "salary": {"op": "gt", "value": 55000},
        }
    )
    assert result["shape"][0] == 2


def test_filter_rows_unknown_column_raises():
    with pytest.raises(KeyError):
        filter_rows({"nonexistent": {"op": "eq", "value": 1}})


def test_filter_rows_unknown_operator_raises():
    with pytest.raises(ValueError):
        filter_rows({"salary": {"op": "DROP TABLE", "value": 1}})


def test_filter_rows_rejects_query_injection_via_column_name():
    # A malicious/prompt-injected column name must never be evaluated as code —
    # it should just fail to match a real column, not execute anything.
    malicious_column = "salary`).__class__.__mro__[1].__subclasses__()#"
    with pytest.raises(KeyError):
        filter_rows({malicious_column: {"op": "eq", "value": 1}})


def test_compare_groups_basic():
    result = compare_groups(split_column="department", value_a="Eng", value_b="HR")
    assert result["schema"]["row_count_before"] == 3
    assert result["schema"]["row_count_after"] == 2


def test_compare_groups_with_metric_and_segment():
    df = pd.DataFrame(
        {
            "month": ["Jan"] * 4 + ["Feb"] * 4,
            "region": ["West", "West", "East", "East"] * 2,
            "revenue": [100, 100, 50, 50, 10, 10, 60, 60],
        }
    )
    set_dataframe(df)
    result = compare_groups(
        split_column="month", value_a="Jan", value_b="Feb", metric="revenue", segment="region"
    )
    assert "driver" in result
    assert result["driver"]["by_segment"][0]["segment"] == "West"


def test_compare_groups_unknown_column():
    result = compare_groups(split_column="nonexistent", value_a="a", value_b="b")
    assert "error" in result


def test_compare_groups_no_matching_rows():
    result = compare_groups(split_column="department", value_a="Eng", value_b="Nonexistent")
    assert "error" in result


def test_tool_call_to_code_get_summary():
    assert tool_call_to_code("get_summary", {}) == "df.describe(include='all')"
    assert tool_call_to_code("get_summary", {"columns": ["age"]}) == "df[['age']].describe(include='all')"


def test_tool_call_to_code_get_value_counts():
    code = tool_call_to_code("get_value_counts", {"column": "department", "top_n": 5})
    assert code == "df['department'].value_counts().head(5)"


def test_tool_call_to_code_filter_rows():
    code = tool_call_to_code(
        "filter_rows", {"conditions": {"age": {"op": "gt", "value": 30}}}
    )
    assert code == "df[(df['age'] > 30)]"


def test_tool_call_to_code_filter_rows_contains():
    code = tool_call_to_code(
        "filter_rows", {"conditions": {"department": {"op": "contains", "value": "eng"}}}
    )
    assert "str.contains('eng'" in code


def test_tool_call_to_code_plot_scatter_with_hue():
    code = tool_call_to_code("plot_scatter", {"x": "age", "y": "salary", "hue": "department"})
    assert code == "df.plot.scatter(x='age', y='salary', c='department')"


def test_tool_call_to_code_plot_scatter_with_fit_line():
    code = tool_call_to_code("plot_scatter", {"x": "age", "y": "salary", "fit_line": True})
    assert "df.plot.scatter(x='age', y='salary')" in code
    assert "linregress" in code


def test_tool_call_to_code_plot_scatter_fit_line_ignored_with_hue():
    code = tool_call_to_code("plot_scatter", {"x": "age", "y": "salary", "hue": "department", "fit_line": True})
    assert "linregress" not in code


def test_plot_scatter_with_fit_line_produces_valid_png():
    assert _is_valid_png(plot_scatter("age", "salary", fit_line=True))


def test_plot_scatter_fit_line_skipped_when_x_has_no_variance():
    df = pd.DataFrame({"x": [5, 5, 5, 5], "y": [1, 2, 3, 4]})
    set_dataframe(df)
    # Must not crash even though a fit line isn't meaningful here.
    assert _is_valid_png(plot_scatter("x", "y", fit_line=True))


def test_plot_scatter_without_fit_line_still_works():
    assert _is_valid_png(plot_scatter("age", "salary"))


def test_tool_call_to_code_compare_groups():
    code = tool_call_to_code(
        "compare_groups",
        {"split_column": "month", "value_a": "Jan", "value_b": "Feb", "metric": "revenue", "segment": "region"},
    )
    assert "group_a = df[df['month'] == 'Jan']" in code
    assert "groupby('region')['revenue']" in code


def test_tool_call_to_code_unknown_tool_falls_back_to_comment():
    code = tool_call_to_code("some_future_tool", {"x": 1})
    assert code.startswith("#")
