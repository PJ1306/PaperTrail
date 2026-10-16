"""Load PaperTrail settings from the environment without exposing secrets."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
# Documented Token Factory ID for NVIDIA Nemotron 3.5 Lightning.
# Confirm this exact value in your Token Factory console; accounts can differ.
DEFAULT_MODEL_ID = "nvidia/Nemotron-3_5-Lightning"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_FILE_SIZE_MB = 20


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model_id: str
    timeout_seconds: float
    max_retries: int
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB


def load_settings(*, load_env_file: bool = True, environ: dict[str, str] | None = None) -> Settings:
    """Read Nebius settings from the environment.

    `load_env_file` loads a local `.env` file when present. Tests should pass
    `load_env_file=False` so a developer's real key is never pulled in.
    """
    if load_env_file:
        load_dotenv()

    source = os.environ if environ is None else environ

    api_key = (source.get("NEBIUS_API_KEY") or "").strip()
    if not api_key:
        raise ConfigError(
            "NEBIUS_API_KEY is missing. Copy .env.example to .env, add your "
            "Nebius Token Factory key, and restart the app. Do not paste the "
            "key into chat or commit it to git."
        )

    base_url = (source.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL).strip()
    if not base_url:
        raise ConfigError("NEBIUS_BASE_URL is empty. Remove it to use the default Token Factory URL.")

    model_id = (source.get("NEBIUS_MODEL_ID") or DEFAULT_MODEL_ID).strip()
    if not model_id:
        raise ConfigError("NEBIUS_MODEL_ID is empty. Set it to a model ID from your Token Factory console.")

    timeout_raw = (source.get("NEBIUS_TIMEOUT_SECONDS") or str(DEFAULT_TIMEOUT_SECONDS)).strip()
    retries_raw = (source.get("NEBIUS_MAX_RETRIES") or str(DEFAULT_MAX_RETRIES)).strip()
    max_size_raw = (source.get("MAX_PDF_SIZE_MB") or str(DEFAULT_MAX_FILE_SIZE_MB)).strip()

    try:
        timeout_seconds = float(timeout_raw)
        max_retries = int(retries_raw)
        max_file_size_mb = int(max_size_raw)
    except ValueError as exc:
        raise ConfigError(
            "NEBIUS_TIMEOUT_SECONDS must be a number, NEBIUS_MAX_RETRIES must be an integer, "
            "and MAX_PDF_SIZE_MB must be an integer."
        ) from exc

    if timeout_seconds <= 0:
        raise ConfigError("NEBIUS_TIMEOUT_SECONDS must be greater than 0.")
    if max_retries < 0:
        raise ConfigError("NEBIUS_MAX_RETRIES cannot be negative.")
    if max_file_size_mb <= 0:
        raise ConfigError("MAX_PDF_SIZE_MB must be greater than 0.")

    return Settings(
        api_key=api_key,
        base_url=base_url,
        model_id=model_id,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        max_file_size_mb=max_file_size_mb,
    )


def get_max_pdf_size_mb(*, environ: dict[str, str] | None = None) -> int:
    """Return the configured maximum PDF file size in MB without requiring an API key."""
    source = os.environ if environ is None else environ
    raw = (source.get("MAX_PDF_SIZE_MB") or "").strip()
    if not raw:
        return DEFAULT_MAX_FILE_SIZE_MB
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_MAX_FILE_SIZE_MB
    except ValueError:
        return DEFAULT_MAX_FILE_SIZE_MB
