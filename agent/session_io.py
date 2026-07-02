"""Save/restore a whole working session as one downloadable JSON file — the
zero-cost, no-backend alternative to server-side persistence. Everything else
in this app lives only in Streamlit's in-memory session_state, so closing the
tab throws away the dataset, the conversation, and any cleaning progress; this
lets a user pick up exactly where they left off, on any machine, with no
account and no hosting cost. Predictive results (feature importance,
segments, trend) are cheap to regenerate from the restored dataset and are
deliberately left out to keep the file small, human-inspectable, and free of
any need to (de)serialize model objects.
"""

import io
import json

import pandas as pd

SESSION_FORMAT_VERSION = 1
MAX_EMBEDDED_ROWS = 20000


class SessionLoadError(ValueError):
    pass


def _embed_csv(df: pd.DataFrame) -> tuple[str, bool]:
    truncated = len(df) > MAX_EMBEDDED_ROWS
    data = df.head(MAX_EMBEDDED_ROWS) if truncated else df
    return data.to_csv(index=False), truncated


def build_session(
    df: pd.DataFrame,
    filename: str,
    messages: list[dict],
    history: list[dict],
    excluded_pii_columns: list[str],
    cleaned_df: pd.DataFrame | None = None,
) -> dict:
    """Bundles the dataset, chat history, and PII exclusions into one dict.
    `cleaned_df` is only included if it differs from `df` (i.e. cleaning fixes
    were actually applied), to avoid embedding the same data twice."""
    dataset_csv, dataset_truncated = _embed_csv(df)
    session = {
        "format_version": SESSION_FORMAT_VERSION,
        "filename": filename,
        "dataset_csv": dataset_csv,
        "dataset_truncated": dataset_truncated,
        "messages": messages,
        "history": history,
        "excluded_pii_columns": excluded_pii_columns,
    }
    if cleaned_df is not None and not cleaned_df.equals(df):
        cleaned_csv, cleaned_truncated = _embed_csv(cleaned_df)
        session["cleaned_dataset_csv"] = cleaned_csv
        session["cleaned_dataset_truncated"] = cleaned_truncated
    return session


def session_to_json(session: dict) -> str:
    return json.dumps(session)


def parse_session(raw: str) -> dict:
    """Returns the session dict, or raises SessionLoadError with a message
    safe to show directly in the UI."""
    try:
        session = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SessionLoadError(f"That doesn't look like a valid session file: {exc}") from exc
    if not isinstance(session, dict) or "dataset_csv" not in session:
        raise SessionLoadError("Not a valid AI Data Analyst session file.")
    if session.get("format_version") != SESSION_FORMAT_VERSION:
        raise SessionLoadError("This session file was saved by an incompatible version of the app.")
    return session


def restore_dataframe(session: dict, key: str = "dataset_csv") -> pd.DataFrame:
    return pd.read_csv(io.StringIO(session[key]))
