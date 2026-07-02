"""Per-column deep-dive profiling: full descriptive stats and a distribution chart
for any single column, computed instantly with pandas — the same zero-AI-cost
philosophy as Auto-Insights, but on demand and scoped to one column at a time,
closer to a lightweight pandas-profiling report than a whole-dataset summary.
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.charts import SEQUENTIAL, SURFACE, fig_to_base64_png, new_figure
from agent.insights import safe_nunique


def profile_column(df: pd.DataFrame, column: str) -> dict:
    series = df[column]
    n = len(series)
    missing = int(series.isna().sum())
    profile = {
        "column": column,
        "dtype": str(series.dtype),
        "count": n,
        "missing": missing,
        "missing_pct": round(100 * missing / n, 2) if n else 0.0,
    }

    if pd.api.types.is_bool_dtype(series):
        profile["kind"] = "boolean"
        counts = series.value_counts(dropna=True)
        profile["value_counts"] = {str(k): int(v) for k, v in counts.items()}
    elif pd.api.types.is_numeric_dtype(series):
        profile["kind"] = "numeric"
        non_null = series.dropna()
        if not non_null.empty:
            profile.update(
                {
                    "mean": float(non_null.mean()),
                    "median": float(non_null.median()),
                    "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
                    "min": float(non_null.min()),
                    "p25": float(non_null.quantile(0.25)),
                    "p75": float(non_null.quantile(0.75)),
                    "max": float(non_null.max()),
                }
            )
    elif pd.api.types.is_datetime64_any_dtype(series):
        profile["kind"] = "datetime"
        non_null = series.dropna()
        if not non_null.empty:
            profile["min"] = non_null.min().isoformat()
            profile["max"] = non_null.max().isoformat()
    else:
        profile["kind"] = "categorical"
        profile["unique"] = safe_nunique(series)
        counts = series.value_counts(dropna=True).head(10)
        profile["top_values"] = [{"value": str(k), "count": int(v)} for k, v in counts.items()]

    return profile


def plot_column_distribution(df: pd.DataFrame, column: str) -> str | None:
    """Returns a base64 PNG, or None if there's nothing meaningful to plot (e.g. every
    value in the column is missing)."""
    series = df[column].dropna()
    if series.empty:
        return None

    fig, ax = new_figure(figsize=(7, 3.5))
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        bins = min(30, max(5, series.nunique()))
        ax.hist(series, bins=bins, color=SEQUENTIAL, edgecolor=SURFACE, linewidth=0.6)
    else:
        try:
            counts = series.value_counts().head(15)
        except TypeError:
            plt.close(fig)
            return None
        counts.plot(kind="bar", ax=ax, color=SEQUENTIAL, edgecolor=SURFACE, linewidth=0.6)
        ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {column}")
    fig.tight_layout()
    return fig_to_base64_png(fig)
