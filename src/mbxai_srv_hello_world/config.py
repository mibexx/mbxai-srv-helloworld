import importlib.metadata
import logging
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).parent.parent.parent
SERVICE_NAME = "MBXAI_SRV_HELLO_WORLD_"


def _get_version() -> str:
    """Get the package version."""
    try:
        return importlib.metadata.version("mbxai_srv_hello_world")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0"  # Default during development


class ApplicationConfig(BaseSettings):
    """Application configuration."""

    name: str = "MBXAI SRV Hello World"
    version: str = Field(default_factory=_get_version)
    log_level: int = logging.INFO

    model_config = SettingsConfigDict(
        env_prefix=SERVICE_NAME,
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class OpenRouterAPIConfig(BaseSettings):
    """OpenRouter API configuration."""

    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"

    model_config = SettingsConfigDict(
        env_prefix=SERVICE_NAME + "OPENROUTER_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    server_url: str | None = None

    model_config = SettingsConfigDict(
        env_prefix=SERVICE_NAME + "MCP_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_config() -> ApplicationConfig:
    """Get the application configuration singleton."""
    return ApplicationConfig()


@lru_cache
def get_openrouter_api_config() -> OpenRouterAPIConfig:
    """Get the OpenRouter API configuration singleton."""
    return OpenRouterAPIConfig()


@lru_cache
def get_mcp_config() -> MCPConfig:
    """Get the MCP configuration singleton."""
    return MCPConfig()
