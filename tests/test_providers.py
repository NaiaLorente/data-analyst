"""Unit tests for the LLM provider registry (no API key required)."""

from agent.providers import DEFAULT_PROVIDER, PROVIDERS


def test_default_provider_exists():
    assert DEFAULT_PROVIDER in PROVIDERS


def test_all_providers_have_required_fields():
    for pid, provider in PROVIDERS.items():
        assert provider.id == pid
        assert provider.label
        assert provider.sdk in {"anthropic", "openai_compatible", "gemini"}
        assert provider.default_model
        if provider.sdk == "openai_compatible":
            assert provider.base_url


def test_local_provider_requires_no_key():
    assert PROVIDERS["ollama"].requires_key is False


def test_hosted_providers_require_a_key():
    for pid in ("anthropic", "openai", "gemini", "groq", "xai"):
        assert PROVIDERS[pid].requires_key is True


def test_xai_is_not_confused_with_groq():
    # xAI (Grok) and Groq are unrelated companies with near-identical names —
    # a very common source of user error (pasting an xai- key into Groq).
    # They must be distinct providers with distinct endpoints.
    assert PROVIDERS["xai"].base_url != PROVIDERS["groq"].base_url
    assert PROVIDERS["xai"].base_url == "https://api.x.ai/v1"
