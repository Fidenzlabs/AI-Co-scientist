"""Provider-agnostic LLM access via langchain's ``init_chat_model``.

The same code works with OpenAI, Anthropic, Ollama, etc. The provider/model are
read from settings; structured output is requested with pydantic schemas so the
rest of the pipeline gets typed objects back.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Type, TypeVar

from pydantic import BaseModel

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


def _export_provider_keys() -> None:
    """Make configured API keys visible to langchain provider SDKs."""
    settings = get_settings()
    if settings.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key


@lru_cache(maxsize=1)
def get_llm():
    """Return a configured chat model instance (cached)."""
    from langchain.chat_models import init_chat_model

    _export_provider_keys()
    settings = get_settings()
    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
    )


def structured_call(schema: Type[T], system: str, user: str) -> T:
    """Invoke the LLM and parse the response into ``schema``.

    Falls back to manual JSON parsing if the provider lacks native structured
    output support, so this works across providers.
    """
    llm = get_llm()
    messages = [("system", system), ("human", user)]

    try:
        structured = llm.with_structured_output(schema)
        return structured.invoke(messages)
    except Exception:
        # Fallback: ask for raw JSON and validate it ourselves.
        json_system = (
            f"{system}\n\nRespond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        raw = llm.invoke([("system", json_system), ("human", user)])
        text = raw.content if hasattr(raw, "content") else str(raw)
        return schema.model_validate_json(_extract_json(text))


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object/array from model text."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = min(
        (i for i in (text.find("{"), text.find("[")) if i != -1),
        default=0,
    )
    return text[start:].strip()
