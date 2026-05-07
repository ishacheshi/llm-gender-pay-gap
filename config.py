"""
config.py
---------
Per-model configuration. Add an entry here to support a new model;
or rely on the prefix-based fallback in get_provider().
"""

from typing import TypedDict


class ModelConfig(TypedDict):
    provider:   str    # "anthropic" | "openai"
    max_tokens: int
    delay:      float  # seconds between successive calls


MODELS: dict[str, ModelConfig] = {
    # ── Anthropic ────────────────────────────────────────────────────────────
    "claude-sonnet-4-6": {
        "provider":   "anthropic",
        "max_tokens": 64,
        "delay":      0.5,
    },
    "claude-opus-4-6": {
        "provider":   "anthropic",
        "max_tokens": 64,
        "delay":      0.5,
    },
    "claude-haiku-4-5": {
        "provider":   "anthropic",
        "max_tokens": 64,
        "delay":      0.5,
    },
    # ── OpenAI ───────────────────────────────────────────────────────────────
    "gpt-4.1": {
        "provider":   "openai",
        "max_tokens": 64,
        "delay":      0.5,
    },
    "gpt-4o": {
        "provider":   "openai",
        "max_tokens": 64,
        "delay":      0.5,
    },
    "gpt-4o-mini": {
        "provider":   "openai",
        "max_tokens": 64,
        "delay":      0.5,
    },
}

# Prefix rules used when a model name is not in the MODELS table.
_PREFIX_MAP: list[tuple[str, str]] = [
    ("claude-",  "anthropic"),
    ("gpt-",     "openai"),
    ("o1",       "openai"),
    ("o3",       "openai"),
    ("o4",       "openai"),
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
    )
