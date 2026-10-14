"""LLM interaction (OpenAI Chat Completions, as in notebook 04) with failure mapping."""

from __future__ import annotations

import logging
from typing import Protocol

from .errors import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)

log = logging.getLogger(__name__)


class ChatModel(Protocol):
    model: str

    def complete(self, messages: list[dict[str, str]]) -> str: ...


class OpenAIChatModel:
    def __init__(
        self,
        api_key: str | None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        timeout: float = 30.0,
        max_retries: int = 2,
        client=None,
    ):
        if client is None:
            if not api_key:
                raise LLMConfigurationError(detail="OPENAI_API_KEY not set")
            from openai import OpenAI

            # The SDK retries connection errors, 429s and 5xx with exponential backoff.
            client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)
        self._client = client
        self.model = model
        self.temperature = temperature

    def complete(self, messages: list[dict[str, str]]) -> str:
        import openai

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
        except openai.AuthenticationError as exc:
            raise LLMConfigurationError(
                "The OpenAI API key was rejected. Check OPENAI_API_KEY. "
                "Retrieved patent passages are still shown below.",
                detail="authentication error",
            ) from exc
        except openai.RateLimitError as exc:
            # Also raised for exhausted quota / billing problems.
            raise LLMRateLimitError(detail="rate limited or quota exceeded") from exc
        except openai.NotFoundError as exc:
            raise LLMConfigurationError(
                f"The configured model '{self.model}' is not available to this API key. "
                "Set OPENAI_MODEL to a model you can access.",
                detail="model not found",
            ) from exc
        except (openai.APITimeoutError, openai.APIConnectionError) as exc:
            raise LLMUnavailableError(detail=f"network error: {type(exc).__name__}") from exc
        except openai.APIStatusError as exc:
            raise LLMUnavailableError(detail=f"API status {exc.status_code}") from exc
        except openai.OpenAIError as exc:
            raise LLMUnavailableError(detail=type(exc).__name__) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError(detail="response had no choices/message") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError(detail="empty completion")
        return content.strip()
