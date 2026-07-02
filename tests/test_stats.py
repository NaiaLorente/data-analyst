"""Unit tests for statistical significance testing (no API key required)."""

import pandas as pd

from agent.stats import compare_categorical_significance, compare_conversion_rates, compare_numeric_significance


def test_numeric_significance_detects_clear_difference():
    before = pd.Series([9.0, 10.0, 11.0] * 10)
    after = pd.Series([99.0, 100.0, 101.0] * 10)
    result = compare_numeric_significance(before, after)
    assert result["applicable"] is True
    assert result["test"] == "welch_t_test"
    assert result["significant"] is True
    assert result["p_value"] < 0.05


def test_numeric_significance_no_difference_not_significant():
    before = pd.Series([10, 11, 9, 10, 12, 8, 10, 11, 9, 10])
    after = pd.Series([10, 9, 11, 10, 8, 12, 10, 9, 11, 10])
    result = compare_numeric_significance(before, after)
    assert result["applicable"] is True
    assert result["significant"] is False


def test_numeric_significance_not_applicable_with_too_few_values():
    result = compare_numeric_significance(pd.Series([1.0]), pd.Series([1.0, 2.0]))
    assert result["applicable"] is False
    assert "reason" in result


def test_numeric_significance_drops_missing_values():
    before = pd.Series([10.0, None, 12.0, None])
    result = compare_numeric_significance(before, pd.Series([10.0, 12.0]))
    assert result["n_before"] == 2


def test_categorical_significance_detects_shift():
    before = pd.Series(["A"] * 90 + ["B"] * 10)
    after = pd.Series(["A"] * 10 + ["B"] * 90)
    result = compare_categorical_significance(before, after)
    assert result["applicable"] is True
    assert result["test"] == "chi_square"
    assert result["significant"] is True


def test_categorical_significance_no_shift_not_significant():
    before = pd.Series(["A"] * 50 + ["B"] * 50)
    after = pd.Series(["A"] * 50 + ["B"] * 50)
    result = compare_categorical_significance(before, after)
    assert result["applicable"] is True
    assert result["significant"] is False


def test_categorical_significance_not_applicable_single_category():
    result = compare_categorical_significance(pd.Series(["A", "A"]), pd.Series(["A", "A"]))
    assert result["applicable"] is False


def test_categorical_significance_handles_unhashable_values_without_crashing():
    before = pd.Series([{"a": 1}, {"a": 1}, {"a": 2}])
    after = pd.Series([{"a": 2}, {"a": 2}, {"a": 1}])
    result = compare_categorical_significance(before, after)
    assert result["applicable"] is True


def test_conversion_rates_detects_significant_difference():
    result = compare_conversion_rates(200, 2000, 260, 2000)
    assert result["applicable"] is True
    assert result["test"] == "two_proportion_z_test"
    assert result["rate_a"] == 0.1
    assert result["rate_b"] == 0.13
    assert result["significant"] is True
    assert result["p_value"] < 0.05
    assert result["ci_low"] > 0  # CI on the +3pt difference should exclude 0


def test_conversion_rates_no_difference_not_significant():
    result = compare_conversion_rates(100, 1000, 102, 1000)
    assert result["applicable"] is True
    assert result["significant"] is False
    assert result["ci_low"] < 0 < result["ci_high"]  # CI should straddle 0


def test_conversion_rates_identical_rates_zero_diff():
    result = compare_conversion_rates(100, 1000, 100, 1000)
    assert result["diff"] == 0.0
    assert result["significant"] is False


def test_conversion_rates_not_applicable_with_zero_total():
    result = compare_conversion_rates(0, 0, 10, 100)
    assert result["applicable"] is False
    assert "reason" in result


def test_conversion_rates_not_applicable_with_count_exceeding_total():
    result = compare_conversion_rates(150, 100, 10, 100)
    assert result["applicable"] is False


def test_conversion_rates_not_applicable_with_negative_count():
    result = compare_conversion_rates(-5, 100, 10, 100)
    assert result["applicable"] is False


def test_conversion_rates_not_applicable_when_both_groups_zero_percent():
    result = compare_conversion_rates(0, 100, 0, 100)
    assert result["applicable"] is False
    assert "reason" in result


def test_conversion_rates_not_applicable_when_both_groups_hundred_percent():
    result = compare_conversion_rates(100, 100, 100, 100)
    assert result["applicable"] is False
