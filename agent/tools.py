"""Data analysis tools used by the AI agent via Claude's tool use."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from agent.charts import (
    CATEGORICAL,
    DIVERGING_CMAP,
    INK_PRIMARY,
    INK_SECONDARY,
    MAX_CATEGORICAL_SERIES,
    OTHER_COLOR,
    SEQUENTIAL,
    SURFACE,
    categorical_colors,
    fig_to_base64_png,
    new_figure,
)
from agent.drift import generate_drift_report, split_by_column


_df: pd.DataFrame | None = None


def set_dataframe(df: pd.DataFrame) -> None:
    global _df
    _df = df


def _require_df() -> pd.DataFrame:
    if _df is None:
        raise ValueError("No dataset loaded.")
    return _df


def get_summary(columns: list[str] | None = None) -> dict:
    """Return descriptive statistics for the dataset or a subset of columns."""
    df = _require_df()
    target = df[columns] if columns else df
    return {
        "shape": list(target.shape),
        "dtypes": target.dtypes.astype(str).to_dict(),
        "describe": target.describe(include="all").fillna("").to_dict(),
        "null_counts": target.isnull().sum().to_dict(),
    }


def get_value_counts(column: str, top_n: int = 10) -> dict:
    """Return the top N most frequent values for a categorical column."""
    df = _require_df()
    counts = df[column].value_counts().head(top_n)
    return counts.to_dict()


def get_correlation_matrix(columns: list[str] | None = None) -> dict:
    """Return the Pearson correlation matrix for numeric columns."""
    df = _require_df()
    target = df[columns] if columns else df.select_dtypes(include="number")
    return target.corr().round(4).to_dict()


_FILTER_OPS = {
    "eq": lambda s, v: s == v,
    "gt": lambda s, v: s > v,
    "lt": lambda s, v: s < v,
    "gte": lambda s, v: s >= v,
    "lte": lambda s, v: s <= v,
    "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
}


def filter_rows(conditions: dict) -> dict:
    """
    Filter rows by column value conditions.
    conditions: {column: {"op": "eq"|"gt"|"lt"|"gte"|"lte"|"contains", "value": ...}}
    Returns shape and first 20 rows as records.
    """
    df = _require_df()
    mask = pd.Series(True, index=df.index)
    for col, cond in conditions.items():
        if col not in df.columns:
            raise KeyError(f"Unknown column: {col}")
        op, val = cond["op"], cond["value"]
        if op not in _FILTER_OPS:
            raise ValueError(f"Unknown operator: {op}")
        mask &= _FILTER_OPS[op](df[col], val)
    filtered = df[mask]
    return {"shape": list(filtered.shape), "rows": filtered.head(20).to_dict(orient="records")}


def compare_groups(
    split_column: str,
    value_a,
    value_b,
    metric: str | None = None,
    segment: str | None = None,
) -> dict:
    """
    Compare two subsets of the dataset split by matching `split_column` against
    two exact values (e.g. two months, two regions, two cohorts) — explains what
    changed between them and, if metric/segment are given, which segment drove it.
    """
    df = _require_df()
    if split_column not in df.columns:
        return {"error": f"Unknown column: {split_column}"}
    df_a, df_b = split_by_column(df, split_column, value_a, value_b)
    if df_a.empty or df_b.empty:
        return {
            "error": (
                f"No rows found for {split_column} == {value_a!r} "
                f"or {split_column} == {value_b!r}."
            )
        }
    return generate_drift_report(df_a, df_b, metric=metric, segment=segment, exclude_columns=[split_column])


def plot_histogram(column: str, bins: int = 20) -> str:
    """Plot a histogram for a numeric column. Returns base64 PNG."""
    df = _require_df()
    fig, ax = new_figure(figsize=(7, 4))
    ax.hist(df[column].dropna(), bins=bins, color=SEQUENTIAL, edgecolor=SURFACE, linewidth=0.6)
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig_to_base64_png(fig)


def plot_bar(column: str, top_n: int = 10) -> str:
    """Plot a bar chart of top_n value counts for a column. Returns base64 PNG."""
    df = _require_df()
    counts = df[column].value_counts().head(top_n)
    fig, ax = new_figure(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax, color=SEQUENTIAL, edgecolor=SURFACE, linewidth=0.6)
    ax.set_title(f"Top {top_n} values in {column}")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig_to_base64_png(fig)


def plot_scatter(x: str, y: str, hue: str | None = None, fit_line: bool = False) -> str:
    """Plot a scatter plot of two numeric columns, optionally with a linear
    regression fit line (R²/slope/p-value annotated). Returns base64 PNG."""
    df = _require_df()
    fig, ax = new_figure(figsize=(7, 5), grid="both")
    if hue and hue in df.columns:
        # Fixed-order categorical color, capped at 8 series — beyond that, every
        # extra category folds into "Other" instead of generating new hues.
        top_categories = df[hue].value_counts().head(MAX_CATEGORICAL_SERIES).index.tolist()
        color_by_category = categorical_colors(top_categories)
        for label, group in df.groupby(hue):
            in_top = label in color_by_category
            ax.scatter(
                group[x],
                group[y],
                label=str(label) if in_top else "Other",
                alpha=0.7,
                s=24,
                color=color_by_category.get(label, OTHER_COLOR),
                edgecolor=SURFACE,
                linewidth=0.3,
            )
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # de-dupes repeated "Other" entries
        ax.legend(by_label.values(), by_label.keys(), title=hue, fontsize=8, frameon=False)
    else:
        ax.scatter(df[x], df[y], alpha=0.6, s=24, color=SEQUENTIAL, edgecolor=SURFACE, linewidth=0.3)
        # A single fit line across multiple hue groups would be misleading (Simpson's
        # paradox risk), so it's only offered when there's one undifferentiated series.
        if fit_line:
            paired = df[[x, y]].dropna()
            if len(paired) >= 2 and paired[x].nunique() > 1:
                result = stats.linregress(paired[x], paired[y])
                xs = np.linspace(paired[x].min(), paired[x].max(), 100)
                ys = result.intercept + result.slope * xs
                ax.plot(xs, ys, color=CATEGORICAL[5], linewidth=2, linestyle="--")
                r_squared = result.rvalue**2
                ax.text(
                    0.02,
                    0.98,
                    f"R² = {r_squared:.3f}   slope = {result.slope:+.3g}   p = {result.pvalue:.3g}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=9,
                    color=INK_SECONDARY,
                )
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs {y}")
    fig.tight_layout()
    return fig_to_base64_png(fig)


def plot_heatmap(columns: list[str] | None = None) -> str:
    """Plot a correlation heatmap. Returns base64 PNG."""
    df = _require_df()
    target = df[columns] if columns else df.select_dtypes(include="number")
    corr = target.corr()
    fig, ax = plt.subplots(figsize=(max(6, len(corr)), max(5, len(corr) - 1)))
    fig.patch.set_facecolor(SURFACE)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        ax=ax,
        linewidths=1,
        linecolor=SURFACE,
        cbar_kws={"label": ""},
    )
    ax.set_title("Correlation Heatmap", color=INK_PRIMARY)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    fig.tight_layout()
    return fig_to_base64_png(fig)


# ── Tool definitions for the Claude API ────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "get_summary",
        "description": "Get descriptive statistics, dtypes, and null counts for the dataset.",
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of columns to summarize. Omit for all columns.",
                }
            },
        },
    },
    {
        "name": "get_value_counts",
        "description": "Get the most frequent values for a categorical column.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "top_n": {"type": "integer", "default": 10},
            },
            "required": ["column"],
        },
    },
    {
        "name": "get_correlation_matrix",
        "description": "Compute the Pearson correlation matrix for numeric columns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to these columns. Omit for all numeric columns.",
                }
            },
        },
    },
    {
        "name": "filter_rows",
        "description": "Filter rows by column conditions and return a preview.",
        "input_schema": {
            "type": "object",
            "properties": {
                "conditions": {
                    "type": "object",
                    "description": (
                        'Map of column -> {op, value}. op is one of: '
                        '"eq", "gt", "lt", "gte", "lte", "contains".'
                    ),
                }
            },
            "required": ["conditions"],
        },
    },
    {
        "name": "compare_groups",
        "description": (
            "Compare two subsets of the dataset split by a column value (e.g. two months, "
            "two regions, two cohorts) to explain what changed between them and, if metric "
            "and segment are given, which segment drove the change. Use this whenever the "
            "user asks a 'why did X change' or 'compare A vs B' style question — never "
            "estimate the answer yourself, always call this tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "split_column": {"type": "string", "description": "Column used to split the data into two groups."},
                "value_a": {"description": "Value identifying the first group (the 'before')."},
                "value_b": {"description": "Value identifying the second group (the 'after')."},
                "metric": {"type": "string", "description": "Optional numeric column to explain the change in."},
                "segment": {
                    "type": "string",
                    "description": "Optional categorical column to break the change down by, to find the driver.",
                },
            },
            "required": ["split_column", "value_a", "value_b"],
        },
    },
    {
        "name": "plot_histogram",
        "description": "Plot a histogram for a numeric column.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "bins": {"type": "integer", "default": 20},
            },
            "required": ["column"],
        },
    },
    {
        "name": "plot_bar",
        "description": "Plot a bar chart of the most frequent values in a column.",
        "input_schema": {
            "type": "object",
            "properties": {
                "column": {"type": "string"},
                "top_n": {"type": "integer", "default": 10},
            },
            "required": ["column"],
        },
    },
    {
        "name": "plot_scatter",
        "description": (
            "Plot a scatter chart comparing two numeric columns. Use fit_line=true whenever the "
            "user asks whether two variables are related, correlated, or predictive of each other — "
            "it adds a linear regression line with R-squared and a significance p-value, rather "
            "than leaving the answer to eyeballing the scatter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "string"},
                "hue": {
                    "type": "string",
                    "description": "Optional categorical column to colour points by. Not combined with fit_line.",
                },
                "fit_line": {
                    "type": "boolean",
                    "default": False,
                    "description": "Add a linear regression fit line with R-squared/slope/p-value. Ignored if hue is set.",
                },
            },
            "required": ["x", "y"],
        },
    },
    {
        "name": "plot_heatmap",
        "description": "Plot a correlation heatmap for numeric columns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Limit to these columns. Omit for all numeric columns.",
                }
            },
        },
    },
]

_FILTER_OP_SYMBOLS = {"eq": "==", "gt": ">", "lt": "<", "gte": ">=", "lte": "<="}


def tool_call_to_code(name: str, args: dict) -> str:
    """Best-effort pandas/matplotlib snippet equivalent to a tool call, shown to the
    user as "View the code" so a chat answer is reproducible outside the app — not
    just a black-box result. Falls back to a Python-call-shaped comment for any tool
    without a dedicated template, so a new tool never breaks this rather than crash."""
    if name == "get_summary":
        columns = args.get("columns")
        target = f"df[{columns!r}]" if columns else "df"
        return f"{target}.describe(include='all')"

    if name == "get_value_counts":
        return f"df[{args['column']!r}].value_counts().head({args.get('top_n', 10)})"

    if name == "get_correlation_matrix":
        columns = args.get("columns")
        target = f"df[{columns!r}]" if columns else "df.select_dtypes(include='number')"
        return f"{target}.corr()"

    if name == "filter_rows":
        parts = []
        for col, cond in args.get("conditions", {}).items():
            op, val = cond.get("op"), cond.get("value")
            if op == "contains":
                parts.append(f"df[{col!r}].astype(str).str.contains({val!r}, case=False, na=False)")
            else:
                parts.append(f"(df[{col!r}] {_FILTER_OP_SYMBOLS.get(op, '==')} {val!r})")
        mask = " & ".join(parts) if parts else "True"
        return f"df[{mask}]"

    if name == "compare_groups":
        split_column, value_a, value_b = args.get("split_column"), args.get("value_a"), args.get("value_b")
        lines = [
            f"group_a = df[df[{split_column!r}] == {value_a!r}]",
            f"group_b = df[df[{split_column!r}] == {value_b!r}]",
        ]
        metric, segment = args.get("metric"), args.get("segment")
        if metric and segment:
            lines.append(f"group_a.groupby({segment!r})[{metric!r}].sum()  # compare to group_b's, same groupby")
        elif metric:
            lines.append(f"group_a[{metric!r}].sum()  # compare to group_b[{metric!r}].sum()")
        return "\n".join(lines)

    if name == "plot_histogram":
        return f"df[{args['column']!r}].plot.hist(bins={args.get('bins', 20)})"

    if name == "plot_bar":
        return f"df[{args['column']!r}].value_counts().head({args.get('top_n', 10)}).plot.bar()"

    if name == "plot_scatter":
        x, y, hue = args["x"], args["y"], args.get("hue")
        base = f"df.plot.scatter(x={x!r}, y={y!r}, c={hue!r})" if hue else f"df.plot.scatter(x={x!r}, y={y!r})"
        if args.get("fit_line") and not hue:
            base += f"\nfrom scipy import stats\nstats.linregress(df[{x!r}], df[{y!r}])  # slope, R², p-value"
        return base

    if name == "plot_heatmap":
        columns = args.get("columns")
        target = f"df[{columns!r}]" if columns else "df.select_dtypes(include='number')"
        return f"sns.heatmap({target}.corr(), annot=True)"

    call_args = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"# {name}({call_args})"


TOOL_FUNCTIONS = {
    "get_summary": get_summary,
    "get_value_counts": get_value_counts,
    "get_correlation_matrix": get_correlation_matrix,
    "filter_rows": filter_rows,
    "compare_groups": compare_groups,
    "plot_histogram": plot_histogram,
    "plot_bar": plot_bar,
    "plot_scatter": plot_scatter,
    "plot_heatmap": plot_heatmap,
}
