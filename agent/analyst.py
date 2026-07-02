"""AI analyst agent: drives a multi-turn tool-use loop against a user-chosen LLM provider.

Three SDK families are supported, one per provider.sdk value:
  - "anthropic"        -> native Anthropic Messages API tool use
  - "openai_compatible" -> OpenAI's chat.completions tool calling (also used
                            for Groq and Ollama, which both expose an
                            OpenAI-compatible endpoint)
  - "gemini"            -> Google Generative AI function calling

Conversation history is kept provider-agnostic (plain user/assistant text
pairs) so switching providers mid-session doesn't require any migration —
each call re-wraps history into that provider's native message format.
"""

import json
import numpy as np
import pandas as pd
from agent.providers import PROVIDERS
from agent.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS, set_dataframe, tool_call_to_code

MAX_TOOL_ITERATIONS = 15

SYSTEM_PROMPT = """You are an expert data analyst. The user has uploaded a dataset.
Your job is to answer their questions by calling the available analysis tools.

Guidelines:
- Always call at least one tool before giving a final answer.
- When producing charts, tell the user what they are looking at.
- Be concise but insightful — highlight interesting patterns, outliers, or correlations.
- If a question is ambiguous, make a reasonable assumption and state it.
- Never invent numbers; only report what the tools return.
- If the user asks "why did X change" or to compare two periods/groups/cohorts,
  use the compare_groups tool instead of manually filtering and eyeballing the difference."""

DRIFT_NARRATION_SYSTEM_PROMPT = """You are a data analyst explaining a "what changed" report to a
non-technical reader. You will be given pre-computed, verified statistics as JSON — every number in
it was already calculated correctly with pandas, not by you.

Write a short (3-6 sentence) plain-English narrative explaining what changed and, if a driver
breakdown is present, which segment most likely caused it. If metric_significance or
segment_significance is present, mention in one clause whether the change is statistically
significant or could plausibly be random variation — this matters more than the raw percentage.
Use ONLY the numbers provided — do not invent, estimate, round differently, or restate them with
different precision. If nothing notable changed, say so plainly instead of manufacturing a story."""


def _schema_note(df: pd.DataFrame) -> str:
    return (
        f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns: "
        + ", ".join(f"{c} ({df[c].dtype})" for c in df.columns)
    )


def _json_safe(obj):
    """Recursively convert numpy/pandas scalar types into native JSON-serializable ones."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, pd.Timedelta):
        return str(obj)
    return obj


def _execute_tool(name: str, args: dict) -> tuple[dict, str | None, str | None]:
    """Run a tool by name. Returns (result_for_model, chart_b64_or_none, code_or_none) —
    code is the pandas snippet equivalent of the call, omitted (None) if the call
    errored, since showing "the code" for a failed call would be misleading."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}, None, None
    try:
        raw = fn(**args)
    except Exception as exc:
        return {"error": str(exc)}, None, None
    code = tool_call_to_code(name, args)
    if name.startswith("plot_"):
        return {"status": "chart generated"}, raw, code
    return _json_safe(raw), None, code


def _to_openai_tools(defs: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": d["name"],
                "description": d["description"],
                "parameters": d["input_schema"],
            },
        }
        for d in defs
    ]


def _to_gemini_tools(defs: list[dict]) -> list[dict]:
    return [
        {
            "function_declarations": [
                {"name": d["name"], "description": d["description"], "parameters": d["input_schema"]}
                for d in defs
            ]
        }
    ]


def run_query(
    df: pd.DataFrame,
    question: str,
    history: list[dict],
    provider_id: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
) -> tuple[str, list[str], list[str]]:
    """
    Run one user question through the agent loop, using whichever provider the
    user selected.

    Returns:
        answer   – final text answer from the model
        charts   – list of base64 PNG strings produced during this turn
        code     – list of pandas code snippets equivalent to the tool calls made
    """
    set_dataframe(df)
    provider = PROVIDERS[provider_id]
    resolved_base_url = base_url or provider.base_url

    if provider.sdk == "anthropic":
        return _run_anthropic(df, question, history, model, api_key)
    if provider.sdk == "gemini":
        return _run_gemini(df, question, history, model, api_key)
    if provider.sdk == "openai_compatible":
        return _run_openai_compatible(df, question, history, model, api_key, resolved_base_url)
    raise ValueError(f"Unknown provider sdk: {provider.sdk}")


def _run_anthropic(df, question, history, model, api_key):
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    messages = [{"role": h["role"], "content": h["content"]} for h in history]
    messages.append({"role": "user", "content": f"{_schema_note(df)}\n\n{question}"})

    charts: list[str] = []
    code: list[str] = []
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return text, charts, code

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result, chart, snippet = _execute_tool(block.name, block.input)
            if chart:
                charts.append(chart)
            if snippet:
                code.append(snippet)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )
        messages.append({"role": "user", "content": tool_results})

    return "I hit the tool-call limit for this turn — try a more specific question.", charts, code


def _run_openai_compatible(df, question, history, model, api_key, base_url):
    from openai import OpenAI

    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(history)
        + [{"role": "user", "content": f"{_schema_note(df)}\n\n{question}"}]
    )

    charts: list[str] = []
    code: list[str] = []
    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.chat.completions.create(
            model=model,
            tools=_to_openai_tools(TOOL_DEFINITIONS),
            messages=messages,
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))

        if choice.finish_reason == "stop" or not msg.tool_calls:
            return msg.content or "", charts, code

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
                result, chart, snippet = _execute_tool(tc.function.name, args)
            except json.JSONDecodeError as exc:
                result, chart, snippet = {"error": f"Invalid tool call arguments: {exc}"}, None, None
            if chart:
                charts.append(chart)
            if snippet:
                code.append(snippet)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return "I hit the tool-call limit for this turn — try a more specific question.", charts, code


def _run_gemini(df, question, history, model, api_key):
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gmodel = genai.GenerativeModel(
        model, system_instruction=SYSTEM_PROMPT, tools=_to_gemini_tools(TOOL_DEFINITIONS)
    )
    gemini_history = [
        {"role": "user" if h["role"] == "user" else "model", "parts": [h["content"]]} for h in history
    ]
    chat = gmodel.start_chat(history=gemini_history)

    charts: list[str] = []
    code: list[str] = []
    response = chat.send_message(f"{_schema_note(df)}\n\n{question}")

    for _ in range(MAX_TOOL_ITERATIONS):
        function_calls = [p.function_call for p in response.parts if p.function_call]
        if not function_calls:
            return response.text, charts, code

        function_responses = []
        for fc in function_calls:
            result, chart, snippet = _execute_tool(fc.name, dict(fc.args))
            if chart:
                charts.append(chart)
            if snippet:
                code.append(snippet)
            function_responses.append(
                {"function_response": {"name": fc.name, "response": {"result": result}}}
            )
        response = chat.send_message(function_responses)

    return "I hit the tool-call limit for this turn — try a more specific question.", charts, code


def _complete_text(
    provider_id: str, model: str, api_key: str, base_url: str | None, system_prompt: str, user_prompt: str
) -> str:
    """Single-shot text completion (no tools, no multi-turn loop) against the chosen provider."""
    provider = PROVIDERS[provider_id]
    resolved_base_url = base_url or provider.base_url

    if provider.sdk == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text")

    if provider.sdk == "gemini":
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        gmodel = genai.GenerativeModel(model, system_instruction=system_prompt)
        response = gmodel.generate_content(user_prompt)
        return response.text

    if provider.sdk == "openai_compatible":
        from openai import OpenAI

        client = OpenAI(api_key=api_key or "not-needed", base_url=resolved_base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content or ""

    raise ValueError(f"Unknown provider sdk: {provider.sdk}")


def narrate_drift(
    report: dict, provider_id: str, model: str, api_key: str, base_url: str | None = None
) -> str:
    """
    Narrate a pre-computed drift report (see agent.drift) in plain English.

    The report's numbers are already verified — this call is only ever allowed
    to explain them, never to compute or restate its own figures.
    """
    user_prompt = "Verified change report (JSON):\n\n" + json.dumps(_json_safe(report), indent=2)
    return _complete_text(provider_id, model, api_key, base_url, DRIFT_NARRATION_SYSTEM_PROMPT, user_prompt)
