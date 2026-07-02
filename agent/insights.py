"""Zero-cost, LLM-free auto-insights computed directly from the dataframe.

Runs the instant a file is uploaded, so the app is useful and demo-able even
before anyone enters an API key.
"""

import numpy as np
import pandas as pd


def safe_nunique(series: pd.Series) -> int | None:
    """nunique() that returns None instead of crashing on unhashable cell values
    (dicts/lists — e.g. a column of nested objects from a JSON upload), since such a
    column can't be meaningfully deduplicated or counted. Callers decide how to treat
    None for their use case (e.g. "definitely not a good candidate column")."""
    try:
        return series.nunique(dropna=True)
    except TypeError:
        return None


def safe_duplicate_count(df: pd.DataFrame) -> int:
    """duplicated().sum() that falls back to a string-cast comparison when the
    dataframe contains unhashable cell values (dicts/lists) pandas can't hash directly."""
    try:
        return int(df.duplicated().sum())
    except TypeError:
        return int(df.astype(str).duplicated().sum())


def generate_insights(df: pd.DataFrame, max_items: int = 5) -> dict:
    numeric = df.select_dtypes(include="number")
    n_rows, n_cols = df.shape

    missing = df.isnull().mean().sort_values(ascending=False)
    missing = missing[missing > 0].head(max_items)

    dtype_breakdown = {
        "numeric": len(df.select_dtypes(include="number").columns),
        "categorical": len(df.select_dtypes(include=["object", "category", "str"]).columns),
        "datetime": len(df.select_dtypes(include="datetime").columns),
        "boolean": len(df.select_dtypes(include="bool").columns),
    }

    top_correlations = []
    if numeric.shape[1] >= 2:
        corr = numeric.corr()
        abs_corr = corr.abs().mask(np.triu(np.ones(corr.shape, dtype=bool)))
        pairs = abs_corr.stack().sort_values(ascending=False).head(max_items)
        top_correlations = [
            {"columns": [a, b], "correlation": round(float(corr.loc[a, b]), 3)}
            for a, b in pairs.index
        ]

    outlier_columns = []
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((series < lower) | (series > upper)).sum())
        if n_outliers:
            outlier_columns.append(
                {"column": col, "outliers": n_outliers, "pct": round(100 * n_outliers / len(series), 2)}
            )
    outlier_columns = sorted(outlier_columns, key=lambda x: -x["outliers"])[:max_items]

    categorical = df.select_dtypes(include=["object", "category", "str"])
    high_cardinality = []
    for col in categorical.columns:
        nunique = safe_nunique(categorical[col])
        if nunique is None:
            # Unhashable values (e.g. nested JSON objects) can't be deduplicated or
            # grouped, so — like an ID column — they're not useful to analyze further.
            high_cardinality.append(col)
        elif n_rows and nunique / n_rows > 0.9 and nunique > 1:
            high_cardinality.append(col)

    return {
        "shape": {"rows": n_rows, "columns": n_cols},
        "dtype_breakdown": dtype_breakdown,
        "missing": [{"column": c, "pct": round(float(v) * 100, 2)} for c, v in missing.items()],
        "top_correlations": top_correlations,
        "outlier_columns": outlier_columns,
        "high_cardinality_columns": high_cardinality,
        "duplicate_rows": safe_duplicate_count(df),
    }


def _insight_lines(insights: dict) -> list[str]:
    lines = [
        f"{insights['shape']['rows']:,} rows × {insights['shape']['columns']} columns",
        (
            f"{insights['dtype_breakdown']['numeric']} numeric, "
            f"{insights['dtype_breakdown']['categorical']} categorical, "
            f"{insights['dtype_breakdown']['datetime']} datetime, "
            f"{insights['dtype_breakdown']['boolean']} boolean columns"
        ),
    ]
    if insights["duplicate_rows"]:
        lines.append(f"{insights['duplicate_rows']:,} duplicate rows detected")
    if insights["missing"]:
        top = ", ".join(f"{m['column']} ({m['pct']}%)" for m in insights["missing"])
        lines.append(f"Missing data: {top}")
    if insights["top_correlations"]:
        top = ", ".join(
            f"{c['columns'][0]} ↔ {c['columns'][1]} ({c['correlation']:+.2f})"
            for c in insights["top_correlations"]
        )
        lines.append(f"Strongest correlations: {top}")
    if insights["outlier_columns"]:
        top = ", ".join(f"{o['column']} ({o['outliers']} rows, {o['pct']}%)" for o in insights["outlier_columns"])
        lines.append(f"Possible outliers: {top}")
    if insights["high_cardinality_columns"]:
        cols = ", ".join(insights["high_cardinality_columns"])
        lines.append(f"Likely ID columns (high cardinality): {cols}")
    return lines


def insights_to_markdown(insights: dict) -> str:
    return "\n".join(f"- {line}" for line in _insight_lines(insights))


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def escape_markdown_math(text: str) -> str:
    """Streamlit's st.markdown auto-renders $...$ as LaTeX. A column name or value
    containing a literal '$' (e.g. "Annual Income (k$)", a common real-world column
    name) can pair up with another '$' later in the same rendered block and get
    silently mangled into a math span. Escaping keeps dollar signs literal."""
    return text.replace("$", "\\$")


def insights_to_html(insights: dict) -> str:
    items = "".join(f"<li>{escape_html(line)}</li>" for line in _insight_lines(insights))
    return f"<ul>{items}</ul>"
