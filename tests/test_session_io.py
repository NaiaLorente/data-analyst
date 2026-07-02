"""Unit tests for save/load session persistence (no API key required)."""

import pandas as pd
import pytest
from agent.session_io import (
    SessionLoadError,
    build_session,
    parse_session,
    restore_dataframe,
    session_to_json,
)


def _df():
    return pd.DataFrame({"a": range(5), "b": list("abcde")})


def _messages():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there", "charts": [], "code": ["df.head()"]},
    ]


def _history():
    return [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi there"}]


def test_build_session_round_trips_dataset():
    df = _df()
    session = build_session(df, "data.csv", _messages(), _history(), [])
    restored = restore_dataframe(session)
    pd.testing.assert_frame_equal(restored, df)


def test_session_to_json_and_parse_round_trip():
    df = _df()
    session = build_session(df, "data.csv", _messages(), _history(), ["email"])
    raw = session_to_json(session)
    parsed = parse_session(raw)
    assert parsed["filename"] == "data.csv"
    assert parsed["messages"] == _messages()
    assert parsed["history"] == _history()
    assert parsed["excluded_pii_columns"] == ["email"]


def test_build_session_omits_cleaned_dataset_when_unchanged():
    df = _df()
    session = build_session(df, "data.csv", [], [], [], cleaned_df=df.copy())
    assert "cleaned_dataset_csv" not in session


def test_build_session_includes_cleaned_dataset_when_changed():
    df = _df()
    cleaned = df.drop_duplicates()
    cleaned.loc[0, "a"] = 999
    session = build_session(df, "data.csv", [], [], [], cleaned_df=cleaned)
    assert "cleaned_dataset_csv" in session
    restored_cleaned = restore_dataframe(session, "cleaned_dataset_csv")
    pd.testing.assert_frame_equal(restored_cleaned, cleaned)


def test_build_session_truncates_large_datasets():
    big_df = pd.DataFrame({"a": range(25000)})
    session = build_session(big_df, "big.csv", [], [], [])
    assert session["dataset_truncated"] is True
    restored = restore_dataframe(session)
    assert len(restored) == 20000


def test_parse_session_rejects_malformed_json():
    with pytest.raises(SessionLoadError):
        parse_session("not json at all {{{")


def test_parse_session_rejects_non_session_json():
    with pytest.raises(SessionLoadError):
        parse_session('{"foo": "bar"}')


def test_parse_session_rejects_incompatible_format_version():
    df = _df()
    session = build_session(df, "data.csv", [], [], [])
    session["format_version"] = 999
    raw = session_to_json(session)
    with pytest.raises(SessionLoadError):
        parse_session(raw)
