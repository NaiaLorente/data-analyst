## Summary

<!-- What does this PR change, and why? -->

## Checklist

- [ ] `pytest tests/ -v` passes locally (no API key required)
- [ ] `ruff check .` passes locally
- [ ] No API key, secret, or credential is committed anywhere in the diff
- [ ] If this touches the agent loop, it works across all three SDK code paths (Anthropic, OpenAI-compatible, Gemini) — or the PR explains why not
- [ ] If this adds a number shown to the user, it comes from a real computation (pandas/numpy), not from the LLM
