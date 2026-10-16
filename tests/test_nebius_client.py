"""Tests for the Nebius client. All API calls are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
)

from src.config import Settings
from src.nebius_client import NebiusClient, NebiusError

FAKE_SETTINGS = Settings(
    api_key="test-key-not-real",
    base_url="https://api.tokenfactory.nebius.com/v1/",
    model_id="nvidia/Nemotron-3_5-Lightning",
    timeout_seconds=30.0,
    max_retries=2,
)


def _status_error(exc_cls: type[APIStatusError], status_code: int) -> APIStatusError:
    response = MagicMock()
    response.status_code = status_code
    response.request = MagicMock()
    response.headers = {}
    return exc_cls(message="redacted", response=response, body=None)


def test_client_passes_configuration_to_openai_sdk() -> None:
    with patch("src.nebius_client.OpenAI") as openai_cls:
        openai_cls.return_value = MagicMock()
        NebiusClient(FAKE_SETTINGS)

    openai_cls.assert_called_once_with(
        api_key="test-key-not-real",
        base_url="https://api.tokenfactory.nebius.com/v1/",
        timeout=30.0,
        max_retries=2,
    )


def test_send_test_prompt_returns_model_text() -> None:
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="  pong  "))]
    )
    client = NebiusClient(FAKE_SETTINGS, client=mock_sdk)

    reply = client.send_test_prompt()

    assert reply == "pong"
    mock_sdk.chat.completions.create.assert_called_once()
    kwargs = mock_sdk.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == FAKE_SETTINGS.model_id
    assert kwargs["messages"] == [{"role": "user", "content": "Reply with the single word pong."}]


def test_empty_model_response_is_not_treated_as_success() -> None:
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="   "))]
    )
    client = NebiusClient(FAKE_SETTINGS, client=mock_sdk)

    with pytest.raises(NebiusError, match="empty response"):
        client.send_test_prompt()


@pytest.mark.parametrize(
    ("exc", "match"),
    [
        (_status_error(AuthenticationError, 401), "Authentication failed"),
        (_status_error(NotFoundError, 404), "was not found"),
        (_status_error(RateLimitError, 429), "rate limit"),
        (APITimeoutError(request=MagicMock()), "timed out"),
        (APIConnectionError(message="redacted", request=MagicMock()), "Could not reach"),
    ],
)
def test_api_errors_are_mapped_without_raw_exception_text(exc: Exception, match: str) -> None:
    mock_sdk = MagicMock()
    mock_sdk.chat.completions.create.side_effect = exc
    client = NebiusClient(FAKE_SETTINGS, client=mock_sdk)

    with pytest.raises(NebiusError, match=match) as raised:
        client.send_test_prompt()

    assert "test-key-not-real" not in str(raised.value)
    assert "redacted" not in str(raised.value)
