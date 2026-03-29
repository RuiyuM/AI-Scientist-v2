from __future__ import annotations

from typing import Any


def is_gpt5_model(model: str | None) -> bool:
    return isinstance(model, str) and model.startswith("gpt-5")


def normalize_chat_completion_kwargs(
    model: str | None, kwargs: dict[str, Any]
) -> dict[str, Any]:
    normalized = dict(kwargs)
    if (
        is_gpt5_model(model)
        and "max_tokens" in normalized
        and "max_completion_tokens" not in normalized
    ):
        normalized["max_completion_tokens"] = normalized.pop("max_tokens")
    return normalized


def chat_completion_create(client: Any, model: str, **kwargs: Any) -> Any:
    return client.chat.completions.create(
        model=model,
        **normalize_chat_completion_kwargs(model, kwargs),
    )
