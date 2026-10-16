"""Tests for environment-based configuration. These never call the network."""

from __future__ import annotations

import pytest

from src.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL_ID,
    DEFAULT_TIMEOUT_SECONDS,
    ConfigError,
    load_settings,
)


def test_missing_api_key_raises_clear_error() -> None:
    with pytest.raises(ConfigError, match="NEBIUS_API_KEY is missing"):
        load_settings(load_env_file=False, environ={})


def test_blank_api_key_raises_clear_error() -> None:
    with pytest.raises(ConfigError, match="NEBIUS_API_KEY is missing"):
        load_settings(load_env_file=False, environ={"NEBIUS_API_KEY": "   "})


def test_load_settings_uses_defaults() -> None:
    settings = load_settings(
        load_env_file=False,
        environ={"NEBIUS_API_KEY": "test-key-not-real"},
    )

    assert settings.api_key == "test-key-not-real"
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.model_id == DEFAULT_MODEL_ID
    assert settings.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert settings.max_retries == DEFAULT_MAX_RETRIES


def test_load_settings_reads_optional_overrides() -> None:
    settings = load_settings(
        load_env_file=False,
        environ={
            "NEBIUS_API_KEY": "test-key-not-real",
            "NEBIUS_BASE_URL": "https://example.test/v1/",
            "NEBIUS_MODEL_ID": "nvidia/example-model",
            "NEBIUS_TIMEOUT_SECONDS": "15",
            "NEBIUS_MAX_RETRIES": "0",
        },
    )

    assert settings.base_url == "https://example.test/v1/"
    assert settings.model_id == "nvidia/example-model"
    assert settings.timeout_seconds == 15.0
    assert settings.max_retries == 0


def test_invalid_timeout_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="NEBIUS_TIMEOUT_SECONDS"):
        load_settings(
            load_env_file=False,
            environ={
                "NEBIUS_API_KEY": "test-key-not-real",
                "NEBIUS_TIMEOUT_SECONDS": "fast",
            },
        )


def test_pdf_max_size_settings_and_helpers() -> None:
    from src.config import DEFAULT_MAX_FILE_SIZE_MB, get_max_pdf_size_mb

    # Default value
    assert get_max_pdf_size_mb(environ={}) == DEFAULT_MAX_FILE_SIZE_MB

    # Custom override
    assert get_max_pdf_size_mb(environ={"MAX_PDF_SIZE_MB": "50"}) == 50

    # Fallback on invalid values
    assert get_max_pdf_size_mb(environ={"MAX_PDF_SIZE_MB": "invalid"}) == DEFAULT_MAX_FILE_SIZE_MB
    assert get_max_pdf_size_mb(environ={"MAX_PDF_SIZE_MB": "-5"}) == DEFAULT_MAX_FILE_SIZE_MB

    # Config error in load_settings for non-integer
    with pytest.raises(ConfigError, match="MAX_PDF_SIZE_MB"):
        load_settings(
            load_env_file=False,
            environ={"NEBIUS_API_KEY": "key", "MAX_PDF_SIZE_MB": "abc"},
        )

