"""One-click data cleaning: detect common data-quality issues (duplicate rows,
missing values, outliers, stray whitespace) and fix them with a single click —
no pandas required. Detection reuses the same pure-pandas logic as Auto-Insights,
so what's flagged there is exactly what can be acted on here. Every fix function
returns a new dataframe rather than mutating its input.
"""

import pandas as pd

from agent.insights import safe_duplicate_count, safe_nunique


def _iqr_bounds(series: pd.Series) -> tuple[float, float] | None:
    """Tukey's-fence bounds for outlier detection/capping. None if the column has
    no spread (IQR == 0), since bounds would collapse to a single value and flag
    almost everything as an outlier."""
    series = series.dropna()
    if series.empty:
        return None
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return None
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def detect_duplicates(df: pd.DataFrame) -> dict | None:
    count = safe_duplicate_count(df)
    if not count:
        return None
    return {"kind": "duplicates", "count": count, "pct": round(100 * count / len(df), 2) if len(df) else 0.0}


def detect_missing(df: pd.DataFrame, max_items: int = 8) -> list[dict]:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(max_items)
    return [
        {
            "kind": "missing",
            "column": col,
            "count": int(count),
            "pct": round(100 * count / len(df), 2) if len(df) else 0.0,
            "numeric": pd.api.types.is_numeric_dtype(df[col]),
        }
        for col, count in missing.items()
    ]


def detect_outliers(df: pd.DataFrame, max_items: int = 8) -> list[dict]:
    issues = []
    for col in df.select_dtypes(include="number").columns:
        bounds = _iqr_bounds(df[col])
        if bounds is None:
            continue
        lower, upper = bounds
        mask = (df[col] < lower) | (df[col] > upper)
        count = int(mask.sum())
        if count:
            issues.append(
                {
                    "kind": "outliers",
                    "column": col,
                    "count": count,
                    "pct": round(100 * count / len(df), 2) if len(df) else 0.0,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return sorted(issues, key=lambda i: -i["count"])[:max_items]


def detect_whitespace(df: pd.DataFrame, max_items: int = 8) -> list[dict]:
    issues = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        series = df[col].dropna()
        if series.empty:
            continue
        try:
            as_str = series.astype(str)
        except (TypeError, ValueError):
            continue
        count = int((as_str != as_str.str.strip()).sum())
        if count:
            issues.append(
                {
                    "kind": "whitespace",
                    "column": col,
                    "count": count,
                    "pct": round(100 * count / len(series), 2) if len(series) else 0.0,
                }
            )
    return sorted(issues, key=lambda i: -i["count"])[:max_items]


def _looks_numeric(series: pd.Series) -> bool:
    """True if a text column is really numbers stored as strings (e.g. a "price"
    column exported with currency-free digits) — coercing and checking the failure
    rate rather than requiring every value to parse, since a stray typo shouldn't
    disqualify an otherwise-numeric column."""
    converted = pd.to_numeric(series, errors="coerce")
    return bool(converted.notna().mean() >= 0.95)


def looks_datetime(series: pd.Series) -> bool:
    """True if a text column is really dates stored as strings. Short pure-digit
    strings (IDs, years, small counts) are excluded first, since pd.to_datetime
    would happily "parse" e.g. "12345" into a date — almost never what's meant."""
    as_str = series.astype(str).str.strip()
    short_digit_like = (as_str.str.fullmatch(r"\d+") & (as_str.str.len() <= 6)).fillna(False)
    if short_digit_like.mean() >= 0.95:
        return False
    converted = pd.to_datetime(series, errors="coerce", format="mixed")
    return bool(converted.notna().mean() >= 0.95)


def detect_type_issues(df: pd.DataFrame, max_items: int = 8) -> list[dict]:
    issues = []
    for col in df.select_dtypes(include=["object", "string"]).columns:
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        if looks_datetime(non_null):
            issues.append({"kind": "type_datetime", "column": col})
        elif _looks_numeric(non_null):
            issues.append({"kind": "type_numeric", "column": col})
    return issues[:max_items]


def detect_constant_columns(df: pd.DataFrame, max_items: int = 8) -> list[dict]:
    issues = []
    for col in df.columns:
        non_null = df[col].dropna()
        if non_null.empty:
            issues.append({"kind": "empty_column", "column": col})
            continue
        if safe_nunique(non_null) == 1:
            issues.append({"kind": "constant_column", "column": col, "value": str(non_null.iloc[0])})
    return issues[:max_items]


def detect_all_issues(df: pd.DataFrame) -> dict:
    return {
        "duplicates": detect_duplicates(df),
        "missing": detect_missing(df),
        "outliers": detect_outliers(df),
        "whitespace": detect_whitespace(df),
        "types": detect_type_issues(df),
        "constant_columns": detect_constant_columns(df),
    }


def count_open_issues(issues: dict) -> int:
    return (
        (1 if issues["duplicates"] else 0)
        + len(issues["missing"])
        + len(issues["outliers"])
        + len(issues["whitespace"])
        + len(issues.get("types", []))
        + len(issues.get("constant_columns", []))
    )


# ── Fixes — each returns a new dataframe, never mutates the input ──────────────


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def fill_missing(df: pd.DataFrame, column: str, strategy: str) -> pd.DataFrame:
    df = df.copy()
    if strategy == "median":
        df[column] = df[column].fillna(df[column].median())
    elif strategy == "mean":
        df[column] = df[column].fillna(df[column].mean())
    elif strategy == "mode":
        mode = df[column].mode(dropna=True)
        if not mode.empty:
            df[column] = df[column].fillna(mode.iloc[0])
    elif strategy == "zero":
        df[column] = df[column].fillna(0)
    else:
        raise ValueError(f"Unknown fill strategy: '{strategy}'.")
    return df


def drop_missing_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.dropna(subset=[column]).reset_index(drop=True)


def cap_outliers(df: pd.DataFrame, column: str) -> pd.DataFrame:
    bounds = _iqr_bounds(df[column])
    if bounds is None:
        return df
    lower, upper = bounds
    df = df.copy()
    df[column] = df[column].clip(lower=lower, upper=upper)
    return df


def remove_outlier_rows(df: pd.DataFrame, column: str) -> pd.DataFrame:
    bounds = _iqr_bounds(df[column])
    if bounds is None:
        return df
    lower, upper = bounds
    mask = df[column].isna() | df[column].between(lower, upper)
    return df[mask].reset_index(drop=True)


def trim_whitespace(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = df[column].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def convert_to_numeric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def convert_to_datetime(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce", format="mixed")
    return df


def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return df.drop(columns=[column])
