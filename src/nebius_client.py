"""Nebius Token Factory client using the official OpenAI-compatible SDK."""

from __future__ import annotations

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from src.config import Settings

TEST_PROMPT = "Reply with the single word pong."


class NebiusError(Exception):
    """User-facing Nebius API error. Messages must never include secrets."""


class NebiusClient:
    """Small wrapper around one configured Nebius chat model."""

    def __init__(self, settings: Settings, *, client: OpenAI | None = None) -> None:
        self.settings = settings
        self._client = client or OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout_seconds,
            max_retries=settings.max_retries,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Send chat messages to the configured model and return the text reply."""
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except AuthenticationError as exc:
            raise NebiusError(
                "Authentication failed. Check that NEBIUS_API_KEY is set to a valid Token Factory key."
            ) from exc
        except PermissionDeniedError as exc:
            raise NebiusError(
                "Access was denied for this key or model. Confirm the model is enabled for your account."
            ) from exc
        except NotFoundError as exc:
            raise NebiusError(
                f"The model '{self.settings.model_id}' was not found. Open the Token Factory "
                "console, copy the exact model ID, and set NEBIUS_MODEL_ID in .env."
            ) from exc
        except RateLimitError as exc:
            raise NebiusError(
                "The Token Factory rate limit was reached. Wait a moment and try again."
            ) from exc
        except APITimeoutError as exc:
            raise NebiusError(
                "The request timed out before the model replied. Check your network and try again."
            ) from exc
        except APIConnectionError as exc:
            raise NebiusError(
                "Could not reach the Nebius Token Factory API. Check your internet connection."
            ) from exc
        except APIStatusError as exc:
            status = getattr(exc, "status_code", None)
            if status == 404:
                raise NebiusError(
                    f"The model '{self.settings.model_id}' was not found. Verify the ID in the "
                    "Token Factory console and set NEBIUS_MODEL_ID if it differs."
                ) from exc
            if status in {401, 403}:
                raise NebiusError(
                    "The API rejected this request. Check your API key and model access in Token Factory."
                ) from exc
            if status == 429:
                raise NebiusError(
                    "The Token Factory rate limit was reached. Wait a moment and try again."
                ) from exc
            raise NebiusError(
                f"The Nebius API returned an error (HTTP {status}). Try again, or verify the model in the console."
            ) from exc
        except Exception as exc:
            raise NebiusError(
                "The Nebius request failed for an unexpected reason. Check your configuration and try again."
            ) from exc

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise NebiusError("The model returned no choices, so the connection cannot be confirmed.")

        content = getattr(choices[0].message, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise NebiusError("The model returned an empty response, so the connection cannot be confirmed.")

        return content.strip()

    def send_test_prompt(self, prompt: str = TEST_PROMPT) -> str:
        """Send a tiny prompt and return the model text if a response arrives."""
        return self.generate(
            [{"role": "user", "content": prompt}],
            max_tokens=32,
            temperature=0.0,
        )
