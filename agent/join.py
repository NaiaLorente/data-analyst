"""Multi-file join/merge: combine two dataframes on a shared key. Everything
else in this app works on one flat file (or two same-shaped snapshots for What
Changed) — but real analysis almost always needs to enrich one table with
another (orders + customers, events + users). Pure pandas, zero AI cost.
"""

import pandas as pd

JOIN_TYPES = ["inner", "left", "right", "outer"]

JOIN_TYPE_DESCRIPTIONS = {
    "inner": "Only rows with a match in both files",
    "left": "Every row from the first file, matched where possible",
    "right": "Every row from the second file, matched where possible",
    "outer": "Every row from both files, matched where possible",
}


def suggest_join_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    """Column names present in both frames, ranked with likely-key names
    (ending in "id", or exactly "id"/"key") first."""
    common = [c for c in df_a.columns if c in df_b.columns]

    def _rank(col: str) -> tuple:
        lowered = str(col).lower()
        looks_like_key = lowered in ("id", "key") or lowered.endswith("_id") or lowered.endswith("id")
        return (0 if looks_like_key else 1, str(col))

    return sorted(common, key=_rank)


def join_dataframes(
    df_a: pd.DataFrame, df_b: pd.DataFrame, left_on: str, right_on: str, how: str = "inner"
) -> pd.DataFrame:
    return df_a.merge(df_b, left_on=left_on, right_on=right_on, how=how, suffixes=("_a", "_b"))


def join_stats(df_a: pd.DataFrame, df_b: pd.DataFrame, result: pd.DataFrame, left_on: str, right_on: str) -> dict:
    """Row counts before/after and the key match rate, so a join that silently
    exploded into a cartesian product or dropped almost everything is obvious
    before committing to it."""
    try:
        left_keys = set(df_a[left_on].dropna())
        right_keys = set(df_b[right_on].dropna())
        match_rate = round(len(left_keys & right_keys) / len(left_keys), 4) if left_keys else 0.0
    except TypeError:
        match_rate = None  # unhashable key values (e.g. nested JSON) — can't compute set overlap

    return {
        "rows_a": len(df_a),
        "rows_b": len(df_b),
        "rows_result": len(result),
        "columns_result": len(result.columns),
        "match_rate": match_rate,
    }
