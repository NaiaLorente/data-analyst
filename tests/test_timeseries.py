"""Unit tests for time-series trend/seasonality/forecast (no API key required)."""

import base64

import numpy as np
import pandas as pd
from agent.timeseries import analyze_trend, plot_trend_forecast


def _rising_series_with_weekend_dip(n=120):
    rng = np.random.RandomState(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    trend = np.linspace(100, 300, n)
    weekday_bump = np.where(pd.Series(dates).dt.dayofweek.isin([5, 6]), -30, 0)
    noise = rng.normal(scale=5, size=n)
    values = trend + weekday_bump + noise
    return pd.DataFrame({"date": dates, "revenue": values})


def _is_valid_png(b64_str: str) -> bool:
    return base64.b64decode(b64_str)[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_trend_detects_rising_direction():
    df = _rising_series_with_weekend_dip()
    result = analyze_trend(df, "date", "revenue")
    assert result["applicable"] is True
    assert result["trend_direction"] == "rising"
    assert result["trend_pct_over_period"] > 0


def test_analyze_trend_detects_weekly_seasonality():
    df = _rising_series_with_weekend_dip()
    result = analyze_trend(df, "date", "revenue")
    assert result["has_weekly_seasonality"] is True
    assert result["weekday_effect"]["Saturday"] < result["weekday_effect"]["Monday"]


def test_analyze_trend_falling_direction():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    values = np.linspace(500, 100, 30)
    df = pd.DataFrame({"date": dates, "metric": values})
    result = analyze_trend(df, "date", "metric")
    assert result["trend_direction"] == "falling"
    assert result["trend_pct_over_period"] < 0


def test_analyze_trend_forecast_length_matches_request():
    df = _rising_series_with_weekend_dip()
    result = analyze_trend(df, "date", "revenue", forecast_days=7)
    assert len(result["forecast"]) == 7


def test_analyze_trend_not_applicable_with_too_few_days():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates, "metric": [1, 2, 3, 4, 5]})
    result = analyze_trend(df, "date", "metric")
    assert result["applicable"] is False
    assert "reason" in result


def test_analyze_trend_handles_unparseable_dates_gracefully():
    df = pd.DataFrame({"date": ["not-a-date"] * 20, "metric": range(20)})
    result = analyze_trend(df, "date", "metric")
    assert result["applicable"] is False


def test_analyze_trend_aggregates_multiple_rows_per_day():
    dates = list(pd.date_range("2024-01-01", periods=20, freq="D")) * 2
    df = pd.DataFrame({"date": dates, "metric": [10] * 40})
    result = analyze_trend(df, "date", "metric", agg="sum")
    # two rows of 10 per day, summed -> 20 per day
    assert result["history"][0]["value"] == 20.0


def test_analyze_trend_mean_aggregation():
    dates = list(pd.date_range("2024-01-01", periods=20, freq="D")) * 2
    df = pd.DataFrame({"date": dates, "metric": [10] * 40})
    result = analyze_trend(df, "date", "metric", agg="mean")
    assert result["history"][0]["value"] == 10.0


def test_plot_trend_forecast_produces_valid_png():
    df = _rising_series_with_weekend_dip()
    result = analyze_trend(df, "date", "revenue")
    assert _is_valid_png(plot_trend_forecast(result))
