"""'What Changed' drift analysis between two dataset snapshots.

Every figure here is computed directly with pandas — never guessed or
estimated by an LLM. The AI narration layer (see agent.analyst.narrate_drift)
is only ever allowed to explain these pre-computed, verified numbers, never
to produce numbers of its own.

Typical use: compare this week's export against last week's, or this
month's against last month's, and get a plain-English answer to "what
changed, and which segment drove it?" — including a check for mix-shift
(Simpson's paradox), where the total moves one way while most individual
segments move the other way.
"""

import pandas as pd

from agent.insights import escape_html, safe_nunique
from agent.stats import compare_categorical_significance, compare_numeric_significance


def pct_change(before: float, after: float) -> float | None:
    if before == 0:
        return None if after == 0 else float("inf")
    return round(100 * (after - before) / before, 2)


def fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    if pct == float("inf"):
        return "+∞% (from zero)"
    return f"{pct:+.2f}%"


def split_by_column(df: pd.DataFrame, column: str, value_a, value_b) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a single dataframe into two comparable subsets by matching `column`
    against two exact values — e.g. splitting one sales log into "January"
    rows vs "February" rows, so users don't need two separately exported files
    just to compare two periods or two categories.
    """
    return df[df[column] == value_a], df[df[column] == value_b]


def split_by_range(
    df: pd.DataFrame, column: str, range_a: tuple, range_b: tuple
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split using two inclusive (start, end) ranges — handy for date/numeric columns."""
    start_a, end_a = range_a
    start_b, end_b = range_b
    df_a = df[(df[column] >= start_a) & (df[column] <= end_a)]
    df_b = df[(df[column] >= start_b) & (df[column] <= end_b)]
    return df_a, df_b


def compare_schemas(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    cols_a, cols_b = set(df_a.columns), set(df_b.columns)
    common = cols_a & cols_b
    dtype_changes = [
        {"column": c, "before": str(df_a[c].dtype), "after": str(df_b[c].dtype)}
        for c in sorted(common)
        if str(df_a[c].dtype) != str(df_b[c].dtype)
    ]
    return {
        "added_columns": sorted(cols_b - cols_a),
        "removed_columns": sorted(cols_a - cols_b),
        "common_columns": sorted(common),
        "dtype_changes": dtype_changes,
        "row_count_before": len(df_a),
        "row_count_after": len(df_b),
        "row_count_pct_change": pct_change(len(df_a), len(df_b)),
    }


def common_numeric_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df_a.columns
        if c in df_b.columns
        and pd.api.types.is_numeric_dtype(df_a[c])
        and pd.api.types.is_numeric_dtype(df_b[c])
    ]


def common_categorical_columns(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df_a.columns
        if c in df_b.columns
        and not pd.api.types.is_numeric_dtype(df_a[c])
        and not pd.api.types.is_numeric_dtype(df_b[c])
    ]


def suggest_split_column(df: pd.DataFrame) -> str | None:
    """
    Pick a reasonable default column to split/compare a single file by —
    prefers a low-cardinality column (a period, cohort, or category, the kind
    of thing you'd actually want to compare two values of) over an ID-like
    column such as a row index or primary key. Columns with unhashable values
    (e.g. nested JSON objects) are never suggested, since they can't be used
    for value-based splitting or grouping.
    """
    if df.shape[1] == 0:
        return None
    candidates = []
    hashable_fallback = None
    for c in df.columns:
        nunique = safe_nunique(df[c])
        if nunique is None:
            continue
        if hashable_fallback is None:
            hashable_fallback = c
        if 2 <= nunique <= 20:
            candidates.append(c)
    if candidates:
        return candidates[0]
    return hashable_fallback if hashable_fallback is not None else df.columns[0]


def suggest_metric_column(df_a: pd.DataFrame, df_b: pd.DataFrame, exclude: tuple = ()) -> str | None:
    """Pick a reasonable default numeric column to explain — avoids obvious ID columns
    and, if this comparison came from splitting one file by a column, that split column
    itself (comparing it against itself would just restate the split)."""
    cols = [c for c in common_numeric_columns(df_a, df_b) if c not in exclude]
    non_id = [c for c in cols if "id" not in c.lower()]
    return non_id[0] if non_id else (cols[0] if cols else None)


def suggest_segment_column(df_a: pd.DataFrame, df_b: pd.DataFrame, exclude: tuple = ()) -> str | None:
    """Pick a reasonable default categorical column to break a change down by.
    Columns with unhashable values (e.g. nested JSON objects) are never suggested."""
    cols = [c for c in common_categorical_columns(df_a, df_b) if c not in exclude]
    hashable_cols = []
    low_cardinality = []
    for c in cols:
        nunique = safe_nunique(df_a[c])
        if nunique is None:
            continue
        hashable_cols.append(c)
        if 2 <= nunique <= 20:
            low_cardinality.append(c)
    if low_cardinality:
        return low_cardinality[0]
    if hashable_cols:
        return hashable_cols[0]
    return cols[0] if cols else None


def compare_metrics(df_a: pd.DataFrame, df_b: pd.DataFrame, columns: list[str] | None = None) -> list[dict]:
    """Sum/mean before vs after for every shared numeric column, ranked by size of change."""
    target_cols = columns if columns is not None else common_numeric_columns(df_a, df_b)
    results = []
    for c in target_cols:
        sum_a, sum_b = float(df_a[c].sum()), float(df_b[c].sum())
        mean_a = float(df_a[c].mean()) if len(df_a) else 0.0
        mean_b = float(df_b[c].mean()) if len(df_b) else 0.0
        results.append(
            {
                "column": c,
                "sum_before": round(sum_a, 4),
                "sum_after": round(sum_b, 4),
                "sum_pct_change": pct_change(sum_a, sum_b),
                "mean_before": round(mean_a, 4),
                "mean_after": round(mean_b, 4),
                "mean_pct_change": pct_change(mean_a, mean_b),
            }
        )

    def _sort_key(r):
        pct = r["sum_pct_change"]
        return abs(pct) if pct is not None else 0

    return sorted(results, key=_sort_key, reverse=True)


def compare_categories(
    df_a: pd.DataFrame, df_b: pd.DataFrame, columns: list[str] | None = None, top_n: int = 5
) -> list[dict]:
    """Share-of-total shifts, plus new/dropped categories, for every shared categorical column."""
    target_cols = columns if columns is not None else common_categorical_columns(df_a, df_b)
    results = []
    for c in target_cols:
        # Convert to string-keyed dicts immediately: value_counts() tolerates
        # unhashable cell values (e.g. dicts/lists from a nested-JSON column),
        # but Python's set() below does not — stringifying the keys up front
        # keeps everything hashable without changing behavior for ordinary
        # (already-hashable) categorical values.
        counts_a = {str(k): v for k, v in df_a[c].value_counts(normalize=True).items()}
        counts_b = {str(k): v for k, v in df_b[c].value_counts(normalize=True).items()}
        all_values = set(counts_a) | set(counts_b)

        shifts = []
        for value in all_values:
            share_before = float(counts_a.get(value, 0.0)) * 100
            share_after = float(counts_b.get(value, 0.0)) * 100
            shifts.append(
                {
                    "value": value,
                    "share_before": round(share_before, 2),
                    "share_after": round(share_after, 2),
                    "shift_pts": round(share_after - share_before, 2),
                }
            )
        shifts.sort(key=lambda s: abs(s["shift_pts"]), reverse=True)

        results.append(
            {
                "column": c,
                "top_shifts": shifts[:top_n],
                "new_categories": sorted(set(counts_b) - set(counts_a))[:top_n],
                "dropped_categories": sorted(set(counts_a) - set(counts_b))[:top_n],
            }
        )
    return results


def decompose_aggregates(agg_a: dict, agg_b: dict, metric: str, segment: str, agg: str = "sum") -> dict:
    """
    Core driver-decomposition math, operating on two {segment_value: aggregated_value}
    dicts rather than two dataframes. explain_driver() below is a thin wrapper that
    computes these aggregates from live dataframes.
    """
    segments = sorted(set(agg_a) | set(agg_b), key=str)
    total_before = float(sum(agg_a.values()))
    total_after = float(sum(agg_b.values()))
    total_delta = total_after - total_before

    rows = []
    for seg in segments:
        before = float(agg_a.get(seg, 0.0))
        after = float(agg_b.get(seg, 0.0))
        delta = after - before
        contribution_pct = round(100 * delta / total_delta, 2) if total_delta != 0 else None
        rows.append(
            {
                "segment": str(seg),
                "before": round(before, 4),
                "after": round(after, 4),
                "delta": round(delta, 4),
                "contribution_pct_of_total_change": contribution_pct,
            }
        )
    rows.sort(key=lambda r: abs(r["delta"]), reverse=True)

    mix_shift_warning = None
    if total_delta != 0 and len(rows) > 1:
        same_direction = sum(1 for r in rows if (r["delta"] > 0) == (total_delta > 0) and r["delta"] != 0)
        if same_direction < len(rows) / 2:
            mix_shift_warning = (
                "Most individual segments moved the opposite direction of the overall "
                "total — this looks like a mix-shift (Simpson's paradox), not a uniform trend."
            )

    return {
        "metric": metric,
        "segment": segment,
        "aggregation": agg,
        "total_before": round(total_before, 4),
        "total_after": round(total_after, 4),
        "total_delta": round(total_delta, 4),
        "total_pct_change": pct_change(total_before, total_after),
        "by_segment": rows,
        "mix_shift_warning": mix_shift_warning,
    }


def explain_driver(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str, segment: str, agg: str = "sum") -> dict:
    """
    Decompose the total change in `metric` between df_a and df_b by `segment`,
    so you can see which segment actually caused the overall move — e.g. a
    total that looks like a broad 12% decline might really be one region
    dropping 40% while every other region grew.
    """
    if agg == "mean":
        agg_a = df_a.groupby(segment)[metric].mean().to_dict()
        agg_b = df_b.groupby(segment)[metric].mean().to_dict()
    else:
        agg = "sum"
        agg_a = df_a.groupby(segment)[metric].sum().to_dict()
        agg_b = df_b.groupby(segment)[metric].sum().to_dict()

    return decompose_aggregates(agg_a, agg_b, metric, segment, agg)


def generate_drift_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    metric: str | None = None,
    segment: str | None = None,
    agg: str = "sum",
    exclude_columns: list[str] | None = None,
) -> dict:
    """
    exclude_columns is useful when df_a/df_b were produced by splitting one
    dataframe on a column's value (e.g. month == "Jan" vs month == "Feb") —
    excluding that split column avoids a trivially-true "month changed from
    Jan to Feb" line cluttering the report.
    """
    exclude = set(exclude_columns or [])
    numeric_cols = [c for c in common_numeric_columns(df_a, df_b) if c not in exclude]
    categorical_cols = [c for c in common_categorical_columns(df_a, df_b) if c not in exclude]
    report = {
        "schema": compare_schemas(df_a, df_b),
        "metrics": compare_metrics(df_a, df_b, columns=numeric_cols),
        "categories": compare_categories(df_a, df_b, columns=categorical_cols),
    }
    if metric and segment:
        report["driver"] = explain_driver(df_a, df_b, metric, segment, agg)
    if metric and metric in df_a.columns and metric in df_b.columns:
        significance = compare_numeric_significance(df_a[metric], df_b[metric])
        significance["metric"] = metric
        report["metric_significance"] = significance
    if segment and segment in df_a.columns and segment in df_b.columns:
        significance = compare_categorical_significance(df_a[segment], df_b[segment])
        significance["column"] = segment
        report["segment_significance"] = significance
    return report


def _report_lines(report: dict) -> list[str]:
    lines = []
    schema = report["schema"]
    lines.append(
        f"Row count: {schema['row_count_before']:,} → {schema['row_count_after']:,} "
        f"({fmt_pct(schema['row_count_pct_change'])})"
    )
    if schema["added_columns"]:
        lines.append(f"New columns: {', '.join(schema['added_columns'])}")
    if schema["removed_columns"]:
        lines.append(f"Removed columns: {', '.join(schema['removed_columns'])}")
    if schema["dtype_changes"]:
        changes = ", ".join(f"{d['column']} ({d['before']} → {d['after']})" for d in schema["dtype_changes"])
        lines.append(f"Dtype changes: {changes}")

    top_metrics = [m for m in report["metrics"] if m["sum_pct_change"]][:5]
    if top_metrics:
        for m in top_metrics:
            lines.append(
                f"{m['column']}: {m['sum_before']:,.2f} → {m['sum_after']:,.2f} ({fmt_pct(m['sum_pct_change'])})"
            )

    for cat in report["categories"]:
        if cat["top_shifts"]:
            shifts = ", ".join(f"{s['value']} ({s['shift_pts']:+.1f}pts)" for s in cat["top_shifts"][:3])
            lines.append(f"{cat['column']} distribution shift: {shifts}")
        if cat["new_categories"]:
            lines.append(f"{cat['column']} new values: {', '.join(cat['new_categories'])}")
        if cat["dropped_categories"]:
            lines.append(f"{cat['column']} dropped values: {', '.join(cat['dropped_categories'])}")

    if "driver" in report:
        driver = report["driver"]
        lines.append(
            f"{driver['metric']} ({driver['aggregation']}) by {driver['segment']}: "
            f"{driver['total_before']:,.2f} → {driver['total_after']:,.2f} "
            f"({fmt_pct(driver['total_pct_change'])})"
        )
        top_drivers = driver["by_segment"][:5]
        if top_drivers:
            driver_str = ", ".join(
                f"{d['segment']} ({d['delta']:+,.2f}"
                + (
                    f", {d['contribution_pct_of_total_change']:+.1f}% of total change"
                    if d["contribution_pct_of_total_change"] is not None
                    else ""
                )
                + ")"
                for d in top_drivers
            )
            lines.append(f"Top contributors: {driver_str}")
        if driver["mix_shift_warning"]:
            lines.append(f"⚠️ {driver['mix_shift_warning']}")

    if "metric_significance" in report and report["metric_significance"]["applicable"]:
        sig = report["metric_significance"]
        verdict = "statistically significant" if sig["significant"] else "not statistically significant (could be random variation)"
        lines.append(f"Significance: the change in {sig['metric']} is {verdict} (p={sig['p_value']:.3f}, Welch's t-test).")

    if "segment_significance" in report and report["segment_significance"]["applicable"]:
        sig = report["segment_significance"]
        verdict = "statistically significant" if sig["significant"] else "not statistically significant (could be random variation)"
        lines.append(
            f"Significance: the shift in {sig['column']} distribution is {verdict} (p={sig['p_value']:.3f}, chi-square test)."
        )

    return lines


def drift_report_to_markdown(report: dict) -> str:
    lines = _report_lines(report)
    if not lines:
        return "- No meaningful differences detected."
    return "\n".join(f"- {line}" for line in lines)


def drift_report_to_html(report: dict) -> str:
    lines = _report_lines(report) or ["No meaningful differences detected."]
    items = "".join(f"<li>{escape_html(line)}</li>" for line in lines)
    return f"<ul>{items}</ul>"
