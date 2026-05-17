"""
config.py
---------
Per-model configuration. Add an entry here to support a new model;
or rely on the prefix-based fallback in get_provider().
"""

from typing import TypedDict


class ModelConfig(TypedDict, total=False):
    provider:              str    # "anthropic" | "openai" | "google"  (required)
    max_tokens:            int    # (required)
    delay:                 float  # seconds between successive calls     (required)
    max_completion_tokens: bool   # OpenAI models requiring max_completion_tokens
    api_model_id:          str    # actual API model ID when it differs from the key


MODELS: dict[str, ModelConfig] = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-sonnet-4-6": {
        "provider":              "anthropic",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
    },
    "claude-opus-4-6": {
        "provider":              "anthropic",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
    },
    "claude-haiku-4-5": {
        "provider":              "anthropic",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
    },
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4.1": {
        "provider":              "openai",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
    },
    "gpt-4o": {
        "provider":              "openai",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
    },
    "gpt-4o-mini": {
        "provider":              "openai",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
    },
    "gpt-5.5": {
        "provider":              "openai",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": True,   # requires max_completion_tokens, not max_tokens
    },
    # ── Google ───────────────────────────────────────────────────────────────
    "gemini-3-flash": {
        "provider":              "google",
        "max_tokens":            64,
        "delay":                 0.5,
        "max_completion_tokens": False,
        "api_model_id":          "models/gemini-3-flash-preview",
    },
}

# Prefix rules used when a model name is not in the MODELS table.
_PREFIX_MAP: list[tuple[str, str]] = [
    ("claude-",  "anthropic"),
    ("gpt-",     "openai"),
    ("o1",       "openai"),
    ("o3",       "openai"),
    ("o4",       "openai"),
    ("gemini-",  "google"),
]


def get_provider(model: str) -> str:
    """Return the provider for *model*, via explicit table then prefix inference."""
    if model in MODELS:
        return MODELS[model]["provider"]
    for prefix, provider in _PREFIX_MAP:
        if model.startswith(prefix):
            return provider
    raise ValueError(
        f"Unknown model '{model}'. Add it to config.MODELS or use a known prefix "
        f"(claude-, gpt-, o1, o3, o4)."
    )


def get_config(model: str) -> ModelConfig:
    """Return the ModelConfig for *model*, falling back to sensible defaults."""
    if model in MODELS:
        return MODELS[model]
    return ModelConfig(
        provider=get_provider(model),
        max_tokens=64,
        delay=0.5,
        max_completion_tokens=False,
    )
