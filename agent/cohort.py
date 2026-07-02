"""Cohort/retention analysis: group users by the period of their first
activity, then track what fraction of each cohort is still active in each
subsequent period — the classic "retention triangle" used constantly in
growth/product analytics, with no free generic tool that does it well. Needs
an event log shaped like (user_id, event_date, ...) — one row per user per
activity, not an aggregated snapshot. Pure pandas/numpy, zero AI cost.
"""

import numpy as np
import pandas as pd

from agent.charts import INK_PRIMARY, INK_SECONDARY, SEQUENTIAL_CMAP, SURFACE, fig_to_base64_png

MIN_USERS = 2
MIN_COHORTS = 2


def compute_retention(df: pd.DataFrame, user_column: str, date_column: str, period: str = "M") -> dict:
    """period is a pandas period alias: "D", "W", or "M". Returns
    {"applicable": False, "reason": ...} if there isn't enough usable data."""
    working = df[[user_column, date_column]].dropna()
    dates = pd.to_datetime(working[date_column], errors="coerce", format="mixed")
    working = working.assign(**{date_column: dates}).dropna(subset=[date_column])

    if working[user_column].nunique() < MIN_USERS:
        return {"applicable": False, "reason": f"Need at least {MIN_USERS} distinct users with valid dates."}

    working = working.copy()
    working["_period"] = working[date_column].dt.to_period(period)
    first_period = working.groupby(user_column)["_period"].min().rename("_cohort")
    working = working.join(first_period, on=user_column)
    working["_period_index"] = (working["_period"] - working["_cohort"]).apply(lambda offset: offset.n)

    cohort_sizes = working.groupby("_cohort")[user_column].nunique()
    if len(cohort_sizes) < MIN_COHORTS:
        return {"applicable": False, "reason": f"Need at least {MIN_COHORTS} distinct {period}-periods of activity."}

    active = working.groupby(["_cohort", "_period_index"])[user_column].nunique().reset_index(name="_active")
    active["_cohort_size"] = active["_cohort"].map(cohort_sizes)
    active["_retention"] = active["_active"] / active["_cohort_size"]

    pivot = active.pivot(index="_cohort", columns="_period_index", values="_retention").sort_index()
    pivot = pivot[sorted(pivot.columns)]

    matrix = [[None if pd.isna(v) else round(float(v), 4) for v in row] for row in pivot.itertuples(index=False)]

    return {
        "applicable": True,
        "period": period,
        "cohort_labels": [str(c) for c in pivot.index],
        "period_offsets": [int(c) for c in pivot.columns],
        "cohort_sizes": {str(k): int(v) for k, v in cohort_sizes.items()},
        "matrix": matrix,
    }


def plot_retention_heatmap(result: dict) -> str:
    """Cohort (row) x periods-since-first-activity (column) retention-rate
    heatmap. Returns base64 PNG."""
    matrix = result["matrix"]
    cohort_labels = result["cohort_labels"]
    period_offsets = result["period_offsets"]
    data = np.array([[np.nan if v is None else v for v in row] for row in matrix])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(6, len(period_offsets) * 0.8), max(3, len(cohort_labels) * 0.5 + 1)))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(data, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1, aspect="auto")

    for i in range(len(cohort_labels)):
        for j in range(len(period_offsets)):
            value = matrix[i][j]
            if value is not None:
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color=INK_PRIMARY)

    ax.set_xticks(range(len(period_offsets)))
    ax.set_xticklabels([f"+{p}" for p in period_offsets])
    ax.set_yticks(range(len(cohort_labels)))
    ax.set_yticklabels(cohort_labels)
    ax.set_xlabel(f"{result['period']}-periods since first activity")
    ax.set_ylabel("Cohort")
    ax.set_title("Retention by cohort")
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, label="Retention rate")
    fig.tight_layout()
    return fig_to_base64_png(fig)
