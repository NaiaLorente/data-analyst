"""Unit tests for multi-format dataset loading (no API key required)."""

import io

import pandas as pd
import pytest
from agent.loaders import load_dataframe


def _sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_load_csv():
    df = _sample_df()
    buf = io.BytesIO(df.to_csv(index=False).encode())
    result = load_dataframe(buf, "data.csv")
    pd.testing.assert_frame_equal(result, df)


def test_load_json():
    df = _sample_df()
    buf = io.BytesIO(df.to_json(orient="records").encode())
    result = load_dataframe(buf, "data.json")
    assert list(result.columns) == ["a", "b"]
    assert len(result) == 3


def test_load_excel():
    df = _sample_df()
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    result = load_dataframe(buf, "data.xlsx")
    pd.testing.assert_frame_equal(result, df)


def test_load_parquet():
    df = _sample_df()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)
    result = load_dataframe(buf, "data.parquet")
    pd.testing.assert_frame_equal(result, df)


def test_unsupported_extension():
    with pytest.raises(ValueError):
        load_dataframe(io.BytesIO(b""), "data.txt")
