# Contributing

Thanks for considering a contribution — this project is small on purpose, but that
makes it a good place for a first PR.

## Setup

```bash
git clone https://github.com/NaiaLorente/ai-data-analyst.git
cd ai-data-analyst
pip install -r requirements.txt
pip install pytest ruff
```

## Before opening a PR

```bash
pytest tests/ -v   # all tests run without any API key or network access
ruff check .
```

Both must pass — the CI workflow runs the same checks on every push.

## Ground rules

- **No provider gets special treatment.** Anthropic, OpenAI, Gemini, Groq, and Ollama
  should stay equally supported; if you add a capability, wire it into all three SDK
  code paths in `agent/analyst.py` (or explain in the PR why it can't be).
- **The AI never invents numbers.** Every statistic shown to the user — in chat,
  Auto-Insights, or What Changed — must come from an actual pandas/numpy computation.
  If you add a feature that calls an LLM, make sure it's either calling a tool for the
  numbers or explicitly restricted to narrating numbers it was already given (see
  `narrate_drift` in `agent/analyst.py` for the pattern).
- **No API keys, ever, anywhere in the repo.** This project is bring-your-own-key by
  design; don't add anything that requires a maintainer-provided key to run or test.
- **Zero-cost paths stay zero-cost.** Auto-Insights and What Changed must keep working
  with no API key at all (Ollama or nothing) — don't make them depend on a live LLM call.

## Good first contributions

- A new analysis tool in `agent/tools.py` (follow the existing `TOOL_DEFINITIONS` /
  `TOOL_FUNCTIONS` pattern).
- A new file format in `agent/loaders.py`.
- A new LLM provider in `agent/providers.py` (if it exposes an OpenAI-compatible
  endpoint, it's a two-line addition).

## Reporting bugs / requesting features

Use the issue templates — they're short on purpose. If something's unclear, open an
issue anyway; a vague bug report is more useful than no bug report.
