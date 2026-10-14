"""OpenAI wrapper: every provider failure becomes a user-safe error (no network calls)."""

from types import SimpleNamespace

import openai
import pytest

from patent_rag.errors import (
    LLMConfigurationError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)
from patent_rag.llm import OpenAIChatModel


def make_exc(cls, **attrs):
    """Instantiate an SDK exception without its version-specific constructor."""
    exc = Exception.__new__(cls)
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


class StubClient:
    def __init__(self, result=None, error=None):
        self.kwargs = None
        self._result, self._error = result, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self._error is not None:
            raise self._error
        return self._result


def reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_missing_key_raises_configuration_error():
    with pytest.raises(LLMConfigurationError):
        OpenAIChatModel(api_key=None)


def test_successful_completion_passes_model_and_temperature():
    client = StubClient(result=reply("  answer [S1]  "))
    model = OpenAIChatModel(api_key=None, model="m", temperature=0.2, client=client)
    assert model.complete([{"role": "user", "content": "q"}]) == "answer [S1]"
    assert client.kwargs["model"] == "m" and client.kwargs["temperature"] == 0.2


@pytest.mark.parametrize(
    "sdk_error, expected",
    [
        (make_exc(openai.AuthenticationError, status_code=401), LLMConfigurationError),
        (make_exc(openai.RateLimitError, status_code=429), LLMRateLimitError),
        (make_exc(openai.NotFoundError, status_code=404), LLMConfigurationError),
        (make_exc(openai.APITimeoutError), LLMUnavailableError),
        (make_exc(openai.APIConnectionError), LLMUnavailableError),
        (make_exc(openai.InternalServerError, status_code=500), LLMUnavailableError),
    ],
)
def test_sdk_errors_are_mapped_to_user_safe_errors(sdk_error, expected):
    model = OpenAIChatModel(api_key=None, client=StubClient(error=sdk_error))
    with pytest.raises(expected) as exc:
        model.complete([])
    assert "sk-" not in exc.value.user_message


@pytest.mark.parametrize("bad", [reply(""), reply(None), SimpleNamespace(choices=[]), SimpleNamespace()])
def test_malformed_or_empty_responses_raise_response_error(bad):
    model = OpenAIChatModel(api_key=None, client=StubClient(result=bad))
    with pytest.raises(LLMResponseError):
        model.complete([])
