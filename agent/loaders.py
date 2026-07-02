"""Load tabular data from CSV, Excel, JSON, or Parquet uploads."""

import pandas as pd

SUPPORTED_EXTENSIONS = ("csv", "xlsx", "xls", "json", "parquet")


def load_dataframe(uploaded_file, filename: str) -> pd.DataFrame:
    """Read an uploaded file into a DataFrame based on its extension."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        return pd.read_csv(uploaded_file)
    if ext in ("xlsx", "xls"):
        return pd.read_excel(uploaded_file)
    if ext == "json":
        return pd.read_json(uploaded_file)
    if ext == "parquet":
        return pd.read_parquet(uploaded_file)
    raise ValueError(f"Unsupported file type: .{ext}")
