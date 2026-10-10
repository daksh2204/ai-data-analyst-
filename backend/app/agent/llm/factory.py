"""Build the configured LLM provider."""

from __future__ import annotations

from app.agent.llm.anthropic_provider import AnthropicProvider
from app.agent.llm.base import LLMProvider
from app.agent.llm.offline import OfflineProvider
from app.agent.llm.openai_compat import OpenAICompatibleProvider
from app.config import Settings


def create_provider(settings: Settings) -> LLMProvider:
    name = settings.resolved_provider()
    model = settings.resolved_model()
    if name == "offline":
        return OfflineProvider()
    if name == "anthropic":
        return AnthropicProvider(
            model=model,
            api_key=settings.anthropic_api_key,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return OpenAICompatibleProvider(
        name=name,
        model=model,
        api_key=settings.api_key_for(name),
        base_url=settings.base_url_for(name),
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
