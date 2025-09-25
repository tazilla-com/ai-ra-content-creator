# File: workshops_ai/settings.py
# Purpose: Centralized application settings (Pydantic v2) with safe secret redaction,
#          nested ENV support, list parsing (JSON/CSV), and test-friendly caching.

from __future__ import annotations

import json
import secrets
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _parse_csv_or_json_list(v: Any) -> list[str]:
    """
    Accept JSON array (e.g. '["http://a","http://b"]') or CSV ('http://a,http://b').
    """
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        if s.startswith("["):
            return list(json.loads(s))
        return [p.strip() for p in s.split(",") if p.strip()]
    return list(v) if isinstance(v, (list, tuple)) else []


# ------------------------------------------------------------------------------
# Sub-settings
# ------------------------------------------------------------------------------
class SecuritySettings(BaseModel):
    """Authentication & authorization settings."""

    # --- API Key auth ---
    enable_api_key: bool = Field(
        default=True,
        description="Enable X-API-Key / ?api_key authentication.",
    )
    # Allow multiple keys; any match passes. In prod, keep keys short-lived.
    api_keys: list[str] = Field(
        default_factory=list,
        description="Authorized API keys.",
    )

    # --- JWT auth (HS/RS/ES by algorithm; HS* uses 'jwt_secret') ---
    enable_jwt: bool = Field(
        default=False,
        description="Enable Authorization: Bearer <JWT> authentication.",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT algorithm, e.g. HS256/RS256/ES256.",
    )
    # For HS* algorithms this is the shared secret; for RS*/ES* supply public key (or use jwks_url).
    jwt_secret: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="JWT secret (HS*) or public key (RS*/ES*).",
    )
    jwt_issuer: Optional[str] = Field(default=None, description="Expected iss claim.")
    jwt_audience: Optional[str] = Field(default=None, description="Expected aud claim.")
    jwks_url: Optional[str] = Field(default=None, description="JWKS endpoint for RS*/ES* verification.")

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        return _parse_csv_or_json_list(v)


class PathsSettings(BaseModel):
    """Filesystem roots for data/state/catalogs."""
    data_dir: str = "data"
    state_dir: str = "data/state"
    catalogs_dir: str = "data/catalogs"


class TazilaSettings(BaseModel):
    """Outbound ChangeSet export settings (file/http/disabled)."""
    mode: Literal["file", "http", "disabled"] = "file"
    outbox_path: str = Field(default="data/outbox")
    endpoint: Optional[str] = None     # used for http mode
    auth_token: Optional[str] = None   # used for http mode
    timeout_sec: int = 30


class RedisSettings(BaseModel):
    """Redis connection for sessions/rate-limiting."""
    url: Optional[str] = None  # e.g. redis://localhost:6379/0
    # future: tls, username/password, pool size, etc.


class RateLimitSettings(BaseModel):
    """Token-bucket caps per session/IP/route (simple minute window)."""
    enabled: bool = False
    per_minute: int = 60


class MemorySettings(BaseModel):
    """In-process memory/kv guardrails."""
    max_org_keys: int = 2000
    max_value_bytes: int = 64 * 1024  # 64KB per value
    soft_total_bytes_per_org: int = 8 * 1024 * 1024  # 8MB soft cap per org


class TelemetrySettings(BaseModel):
    """App-internal telemetry/log shipping knobs (adapter around logging)."""
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = False
    log_file_path: str = "logs/workshops_ai.log"
    max_payload_len: int = 16_000  # truncate oversized blobs


class LLMSettings(BaseModel):
    """LLM client knobs consumed by agents."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    http_timeout_sec: int = 60
    openai_max_retries: int = 3


class ChatSettings(BaseModel):
    """HTTP /chat endpoint knobs."""
    timeout_sec: int = 20
    max_message_len: int = 4000


class WSSettings(BaseModel):
    """WebSocket knobs."""
    timeout_sec: int = 75
    cookie_name: str = "wsai.sid"


class LangchainSettings(BaseModel):
    """LangChain/LangSmith tracing config (settings = jediný zdroj pravdy)."""
    tracing_v2: bool = False
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    project: Optional[str] = None


# ------------------------------------------------------------------------------
# Root settings
# ------------------------------------------------------------------------------
class AppSettings(BaseSettings):
    """
    Application root settings. Loads from environment with nested delimiter '__':
      WORKSHOPS_AI_SECURITY__API_KEYS=["dev-key"]
      WORKSHOPS_AI_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
      WORKSHOPS_AI_PATHS__DATA_DIR=/mnt/data
      WORKSHOPS_AI_LLM__MODEL=gpt-4o-mini 
    """
    model_config = SettingsConfigDict(
        env_prefix="WORKSHOPS_AI_",
        env_file=".env",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # App meta / server
    env: str = "dev"
    service_name: str = "workshops_ai"
    version: str = "0.1.0"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    enable_docs: bool = True
    reload: bool = True

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Feature groups
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    tazila: TazilaSettings = Field(default_factory=TazilaSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    ws: WSSettings = Field(default_factory=WSSettings)
    langchain: LangchainSettings = Field(default_factory=LangchainSettings)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        return _parse_csv_or_json_list(v)

    # ---- Compatibility shims (older code expects these top-level attrs) ----
    @property
    def chat_timeout_sec(self) -> int:
        return self.chat.timeout_sec

    @property
    def max_message_len(self) -> int:
        return self.chat.max_message_len

    @property
    def ws_timeout_sec(self) -> int:
        return self.ws.timeout_sec

    def redacted_dict(self) -> dict:
        """
        Return a dict safe for logging (secrets redacted).
        """
        data = self.model_dump()
        # Security
        sec = data.get("security", {})
        if sec.get("api_keys"):
            sec["api_keys"] = ["***", f"({len(self.security.api_keys)} keys)"]
        if sec.get("jwt_secret"):
            sec["jwt_secret"] = "***"
        # Tazila
        tz = data.get("tazila", {})
        if tz.get("auth_token"):
            tz["auth_token"] = "***"
        # LangChain
        lc = data.get("langchain", {})
        if lc.get("api_key"):
            lc["api_key"] = "***"
        return data


# ------------------------------------------------------------------------------
# Cached accessor (test-friendly)
# ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Load settings from environment (cached)."""
    return AppSettings()


def reset_settings_cache() -> None:
    """Clear settings cache (useful in tests)."""
    get_settings.cache_clear()
