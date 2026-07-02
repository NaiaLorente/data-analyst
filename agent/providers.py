"""Registry of user-selectable LLM providers.

The app never ships an API key. Every provider here is "bring your own key" —
the user picks a provider, pastes their own key (or runs Ollama locally for a
fully free, fully private option), and the key is used only for the duration
of their browser session.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    sdk: str  # "anthropic" | "openai_compatible" | "gemini"
    default_model: str
    base_url: str | None  # only meaningful for sdk == "openai_compatible"
    requires_key: bool
    key_help: str
    get_key_url: str


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        id="anthropic",
        label="Anthropic (Claude)",
        sdk="anthropic",
        default_model="claude-sonnet-5",
        base_url=None,
        requires_key=True,
        key_help="Paid API, usage-based billing.",
        get_key_url="https://console.anthropic.com/",
    ),
    "openai": Provider(
        id="openai",
        label="OpenAI (GPT)",
        sdk="openai_compatible",
        default_model="gpt-4.1",
        base_url="https://api.openai.com/v1",
        requires_key=True,
        key_help="Paid API, usage-based billing.",
        get_key_url="https://platform.openai.com/api-keys",
    ),
    "gemini": Provider(
        id="gemini",
        label="Google (Gemini)",
        sdk="gemini",
        default_model="gemini-2.5-flash",
        base_url=None,
        requires_key=True,
        key_help="Free tier available.",
        get_key_url="https://aistudio.google.com/apikey",
    ),
    "groq": Provider(
        id="groq",
        label="Groq (Llama — fast, free tier)",
        sdk="openai_compatible",
        default_model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        requires_key=True,
        key_help="Generous free tier, very fast inference. Not the same company as xAI/Grok — Groq keys start with 'gsk_'.",
        get_key_url="https://console.groq.com/keys",
    ),
    "xai": Provider(
        id="xai",
        label="xAI (Grok)",
        sdk="openai_compatible",
        default_model="grok-4",
        base_url="https://api.x.ai/v1",
        requires_key=True,
        key_help="Paid API, usage-based billing. Not the same company as Groq — xAI keys start with 'xai-'.",
        get_key_url="https://console.x.ai/",
    ),
    "ollama": Provider(
        id="ollama",
        label="Ollama (local — free & private)",
        sdk="openai_compatible",
        default_model="llama3.1",
        base_url="http://localhost:11434/v1",
        requires_key=False,
        key_help="Runs on your machine. No key, no cost, no data ever leaves your computer.",
        get_key_url="https://ollama.com/download",
    ),
}

DEFAULT_PROVIDER = "groq"
