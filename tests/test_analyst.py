"""Unit tests for the provider-agnostic agent loop helpers (no API key required)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from agent.analyst import (
    _complete_text,
    _execute_tool,
    _json_safe,
    _schema_note,
    _to_gemini_tools,
    _to_openai_tools,
    narrate_drift,
    run_query,
)
from agent.tools import TOOL_DEFINITIONS, set_dataframe


def test_schema_note():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    note = _schema_note(df)
    assert "1 rows" in note
    assert "a (" in note


def test_json_safe_converts_numpy_scalars():
    payload = {"count": np.int64(3), "avg": np.float64(1.5), "flag": np.bool_(True)}
    safe = _json_safe(payload)
    assert safe == {"count": 3, "avg": 1.5, "flag": True}
    assert isinstance(safe["count"], int)
    assert isinstance(safe["avg"], float)


def test_json_safe_converts_timestamp():
    safe = _json_safe(pd.Timestamp("2024-01-01"))
    assert safe == "2024-01-01T00:00:00"


def test_execute_tool_unknown():
    result, chart, code = _execute_tool("nonexistent", {})
    assert "error" in result
    assert chart is None
    assert code is None


def test_execute_tool_get_summary_is_json_safe():
    set_dataframe(pd.DataFrame({"a": [1, 2, None]}))
    result, chart, code = _execute_tool("get_summary", {})
    assert chart is None
    assert result["shape"] == [3, 1]
    # null_counts previously carried numpy.int64, which json.dumps can't handle
    assert isinstance(result["null_counts"]["a"], int)
    assert code == "df.describe(include='all')"


def test_execute_tool_omits_code_on_error():
    set_dataframe(pd.DataFrame({"a": [1, 2, 3]}))
    result, chart, code = _execute_tool("get_value_counts", {"column": "nonexistent"})
    assert "error" in result
    assert code is None


def test_to_openai_tools_shape():
    tools = _to_openai_tools(TOOL_DEFINITIONS)
    assert tools[0]["type"] == "function"
    assert "name" in tools[0]["function"]
    assert len(tools) == len(TOOL_DEFINITIONS)


def test_to_gemini_tools_shape():
    tools = _to_gemini_tools(TOOL_DEFINITIONS)
    assert "function_declarations" in tools[0]
    names = [d["name"] for d in tools[0]["function_declarations"]]
    assert names == [d["name"] for d in TOOL_DEFINITIONS]


def test_run_query_unknown_provider():
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError):
        run_query(df, "hi", [], provider_id="not-a-provider", model="x", api_key="k")


def test_complete_text_unknown_provider():
    with pytest.raises(KeyError):
        _complete_text("not-a-provider", "x", "k", None, "system", "user")


def test_narrate_drift_unknown_provider():
    with pytest.raises(KeyError):
        narrate_drift({"schema": {}}, provider_id="not-a-provider", model="x", api_key="k")


def test_narrate_drift_serializes_numpy_types_without_error():
    # Regression guard: a report containing numpy scalars must not blow up
    # json.dumps inside narrate_drift before the (unreachable, in this test)
    # network call — proves the payload building step is JSON-safe.
    report = {"total_delta": np.float64(-1100.0), "count": np.int64(3)}
    with pytest.raises(KeyError):
        narrate_drift(report, provider_id="not-a-provider", model="x", api_key="k")


def test_run_query_openai_compatible_survives_malformed_tool_call_json():
    # Regression guard: a malformed tool-call arguments payload (a known
    # real-world quirk of smaller/local models served via Groq/Ollama) must
    # degrade to a per-tool-call error, not crash the whole turn like an
    # unguarded json.loads would.
    df = pd.DataFrame({"a": [1, 2, 3]})

    bad_call = MagicMock()
    bad_call.id = "call_1"
    bad_call.function.name = "get_summary"
    bad_call.function.arguments = "{not valid json"

    first_message = MagicMock()
    first_message.tool_calls = [bad_call]
    first_message.model_dump.return_value = {"role": "assistant", "tool_calls": [{"id": "call_1"}]}
    first_choice = MagicMock(message=first_message, finish_reason="tool_calls")
    first_response = MagicMock(choices=[first_choice])

    final_message = MagicMock()
    final_message.tool_calls = None
    final_message.content = "done"
    final_message.model_dump.return_value = {"role": "assistant", "content": "done"}
    final_choice = MagicMock(message=final_message, finish_reason="stop")
    final_response = MagicMock(choices=[final_choice])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [first_response, final_response]

    with patch("openai.OpenAI", return_value=mock_client):
        answer, charts, code = run_query(df, "hi", [], provider_id="groq", model="x", api_key="k")

    assert answer == "done"
    assert charts == []
    assert code == []
    second_call_messages = mock_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    tool_result_msg = next(m for m in second_call_messages if m.get("role") == "tool")
    assert "error" in tool_result_msg["content"]
