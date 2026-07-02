"""Time-series trend, weekly seasonality, and a short forecast for a date column
+ metric — closes the gap between Auto-Insights (single-snapshot description)
and What Changed (two-snapshot comparison): what's happening *continuously*
over time, and where is it likely headed. A simple linear-trend + day-of-week
decomposition, not full ARIMA/Prophet, to stay fast and dependency-light on any
dataset size. Zero AI cost — pure pandas/numpy.
"""

import calendar

import numpy as np
import pandas as pd

from agent.charts import CATEGORICAL, GRIDLINE, INK_SECONDARY, SURFACE, fig_to_base64_png, new_figure

MIN_DAYS = 10


def _to_daily_series(df: pd.DataFrame, date_column: str, metric_column: str, agg: str) -> pd.Series:
    working = df[[date_column, metric_column]].dropna()
    dates = pd.to_datetime(working[date_column], errors="coerce", format="mixed")
    working = working.assign(**{date_column: dates}).dropna(subset=[date_column])

    grouped = working.groupby(working[date_column].dt.floor("D"))[metric_column]
    daily = grouped.mean() if agg == "mean" else grouped.sum()
    daily = daily.sort_index()
    if daily.empty:
        return daily
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range)


def analyze_trend(
    df: pd.DataFrame, date_column: str, metric_column: str, agg: str = "sum", forecast_days: int = 14
) -> dict:
    """Fit a linear trend + weekly-seasonality adjustment to a daily-aggregated
    metric, then extrapolate `forecast_days` forward. Returns
    {"applicable": False, "reason": ...} if there isn't enough usable data."""
    daily = _to_daily_series(df, date_column, metric_column, agg)
    non_null_days = int(daily.notna().sum())
    if non_null_days < MIN_DAYS:
        return {"applicable": False, "reason": f"Need at least {MIN_DAYS} distinct days of data to detect a trend."}

    y = daily.to_numpy(dtype=float)
    x = np.arange(len(y))
    mask = ~np.isnan(y)

    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    fitted = intercept + slope * x
    residuals = y - fitted

    weekday_effect = {}
    for weekday in range(7):
        day_mask = mask & (daily.index.dayofweek.to_numpy() == weekday)
        if day_mask.any():
            weekday_effect[weekday] = float(np.mean(residuals[day_mask]))

    resid_std = float(np.nanstd(residuals[mask])) if mask.sum() > 1 else 0.0
    y_std = float(np.nanstd(y[mask])) if mask.sum() > 1 else 0.0
    has_weekly_seasonality = bool(weekday_effect) and (max(weekday_effect.values()) - min(weekday_effect.values())) > (
        0.15 * y_std if y_std else 0
    )

    last_x = int(x[-1])
    forecast_index = pd.date_range(daily.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="D")
    forecast_x = np.arange(last_x + 1, last_x + 1 + forecast_days)
    forecast_trend = intercept + slope * forecast_x
    forecast_values = [
        forecast_trend[i] + weekday_effect.get(int(forecast_index[i].dayofweek), 0.0) for i in range(forecast_days)
    ]

    first_fitted = float(fitted[mask][0])
    total_trend_change = float(slope * (int(mask.sum()) - 1))
    trend_pct_total = round(100 * total_trend_change / first_fitted, 2) if first_fitted else None

    return {
        "applicable": True,
        "date_column": date_column,
        "metric_column": metric_column,
        "agg": agg,
        "n_days": non_null_days,
        "slope_per_day": round(float(slope), 6),
        "trend_direction": "rising" if slope > 0 else ("falling" if slope < 0 else "flat"),
        "trend_pct_over_period": trend_pct_total,
        "has_weekly_seasonality": has_weekly_seasonality,
        "weekday_effect": {calendar.day_name[d]: round(v, 2) for d, v in sorted(weekday_effect.items())},
        "forecast_std": round(resid_std, 4),
        "history": [
            {"date": daily.index[i].date().isoformat(), "value": None if np.isnan(y[i]) else round(float(y[i]), 4)}
            for i in range(len(daily))
        ],
        "forecast": [
            {"date": forecast_index[i].date().isoformat(), "value": round(float(forecast_values[i]), 4)}
            for i in range(forecast_days)
        ],
    }


def plot_trend_forecast(result: dict) -> str:
    """Line chart of the historical daily series plus the forecast with a
    residual-based confidence band. Returns base64 PNG."""
    history = result["history"]
    forecast = result["forecast"]
    std = result["forecast_std"]

    hist_dates = pd.to_datetime([h["date"] for h in history])
    hist_values = [h["value"] for h in history]
    fc_dates = pd.to_datetime([f["date"] for f in forecast])
    fc_values = [f["value"] for f in forecast]

    fig, ax = new_figure(figsize=(9, 4.5), grid="y")
    ax.plot(hist_dates, hist_values, color=CATEGORICAL[0], linewidth=1.6, label="Actual")

    bridge_dates = [hist_dates[-1], *fc_dates]
    bridge_values = [hist_values[-1], *fc_values]
    ax.plot(bridge_dates, bridge_values, color=CATEGORICAL[0], linewidth=1.6, linestyle="--", label="Forecast")
    lower = [v - 1.28 * std for v in bridge_values]
    upper = [v + 1.28 * std for v in bridge_values]
    ax.fill_between(bridge_dates, lower, upper, color=CATEGORICAL[0], alpha=0.15, linewidth=0, label="80% interval")

    ax.set_ylabel(f"{result['metric_column']} ({result['agg']})")
    ax.set_title(f"{result['metric_column']} over time")
    ax.legend(fontsize=8, frameon=False, facecolor=SURFACE, labelcolor=INK_SECONDARY)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.5)
    fig.tight_layout()
    return fig_to_base64_png(fig)
