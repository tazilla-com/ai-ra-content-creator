# File: workshops_ai/settings.py
# Purpose:
#   Centralized, Pydantic v2-based application settings with:
#   - Nested ENV loading via delimiter "__" (e.g., WORKSHOPS_AI_SECURITY__API_KEYS=…)
#   - Robust CSV/JSON list parsing (for CORS origins, API keys, etc.)
#   - Safe redaction for secrets when logging
#   - Test-friendly caching and explicit cache reset
#
# How to use:
#   from workshops_ai.settings import get_settings
#   settings = get_settings()  # cached singleton


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
    Parse a value as either JSON array or CSV string.
    - Accepts: '["http://a","http://b"]'  OR  'http://a,http://b'
    - Returns: list[str]
    - Empty/None -> [].
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
# Sub-settings groups (namespaced under WORKSHOPS_AI_* via env_nested_delimiter)
# ------------------------------------------------------------------------------
class SecuritySettings(BaseModel):
    """
    Authentication & authorization settings.

    ENV mapping (examples):
      WORKSHOPS_AI_SECURITY__ENABLE_API_KEY=true
      WORKSHOPS_AI_SECURITY__API_KEYS=["K1","K2"]
      WORKSHOPS_AI_SECURITY__ENABLE_JWT=true
      WORKSHOPS_AI_SECURITY__JWT_ALGORITHM=HS256
      WORKSHOPS_AI_SECURITY__JWT_SECRET=...
      WORKSHOPS_AI_SECURITY__JWT_ISSUER=https://auth.example.com
      WORKSHOPS_AI_SECURITY__JWT_AUDIENCE=workshops-ai
      WORKSHOPS_AI_SECURITY__JWKS_URL=https://auth.example.com/.well-known/jwks.json
    """

    # --- API Key auth ---
    enable_api_key: bool = Field(
        default=True,
        description="Enable X-API-Key / ?api_key authentication.",
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description="Authorized API keys (any match passes).",
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
    # For HS* algorithms this is the shared secret; for RS*/ES* this is typically a public key
    # or you provide a JWKS URL (see jwks_url).
    jwt_secret: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        description="JWT secret (HS*) or public key (RS*/ES*).",
    )
    jwt_issuer: Optional[str] = Field(default=None, description="Expected 'iss' claim.")
    jwt_audience: Optional[str] = Field(default=None, description="Expected 'aud' claim.")
    jwks_url: Optional[str] = Field(default=None, description="JWKS endpoint for RS*/ES* verification.")

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        return _parse_csv_or_json_list(v)


class PathsSettings(BaseModel):
    """
    Filesystem roots for data/state/catalogs.

    ENV mapping:
      WORKSHOPS_AI_PATHS__DATA_DIR=./workshops_ai/data
      WORKSHOPS_AI_PATHS__STATE_DIR=./workshops_ai/data/state
      WORKSHOPS_AI_PATHS__CATALOGS_DIR=./workshops_ai/data/catalogs
    """
    data_dir: str = "data"
    state_dir: str = "data/state"
    catalogs_dir: str = "data/catalogs"


class TazilaSettings(BaseModel):
    """
    Outbound ChangeSet export settings:
      - 'file': write JSON payloads to an outbox directory (non-blocking submit path)
      - 'http': POST JSON to remote API (httpx/AsyncClient; retry/backoff elsewhere)
      - 'disabled': no-op (returns a structured reason)

    ENV mapping:
      WORKSHOPS_AI_TAZILA__MODE=file|http|disabled
      WORKSHOPS_AI_TAZILA__OUTBOX_PATH=/app/data/outbox
      WORKSHOPS_AI_TAZILA__ENDPOINT=https://backend.example.com/changesets
      WORKSHOPS_AI_TAZILA__AUTH_TOKEN=Bearer ...
      WORKSHOPS_AI_TAZILA__TIMEOUT_SEC=30
    """
    mode: Literal["file", "http", "disabled"] = "file"
    outbox_path: str = Field(default="data/outbox")
    endpoint: Optional[str] = None     # used when mode = http
    auth_token: Optional[str] = None   # used when mode = http
    timeout_sec: int = 60


class RedisSettings(BaseModel):
    """
    Redis connection settings for sessions / rate-limiting.

    ENV mapping:
      WORKSHOPS_AI_REDIS__URL=redis://redis:6379/0
    """
    url: Optional[str] = None  # e.g. redis://localhost:6379/0
    # Future: tls, username/password, pool size, etc.


class RateLimitSettings(BaseModel):
    """
    Simple per-minute token-bucket caps per session/IP/route.

    ENV mapping:
      WORKSHOPS_AI_RATELIMIT__ENABLED=true
      WORKSHOPS_AI_RATELIMIT__PER_MINUTE=120
    """
    enabled: bool = False
    per_minute: int = 60


class MemorySettings(BaseModel):
    """
    In-process memory / KV guardrails to avoid unbounded growth (per org).

    ENV mapping:
      WORKSHOPS_AI_MEMORY__MAX_ORG_KEYS=2000
      WORKSHOPS_AI_MEMORY__MAX_VALUE_BYTES=65536
      WORKSHOPS_AI_MEMORY__SOFT_TOTAL_BYTES_PER_ORG=8388608
    """
    max_org_keys: int = 2000
    max_value_bytes: int = 64 * 1024  # 64KB per value
    soft_total_bytes_per_org: int = 8 * 1024 * 1024  # 8MB soft cap per org


class TelemetrySettings(BaseModel):
    """
    App-internal telemetry/log shipping knobs (adapter around standard logging).

    ENV mapping:
      WORKSHOPS_AI_TELEMETRY__ENABLED=true
      WORKSHOPS_AI_TELEMETRY__LEVEL=INFO
      WORKSHOPS_AI_TELEMETRY__LOG_TO_FILE=false
      WORKSHOPS_AI_TELEMETRY__LOG_FILE_PATH=logs/workshops_ai.log
      WORKSHOPS_AI_TELEMETRY__MAX_PAYLOAD_LEN=16000
    """
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = False
    log_file_path: str = "logs/workshops_ai.log"
    max_payload_len: int = 16_000  # truncate oversized blobs


class LLMSettings(BaseModel):
    """
    LLM client configuration used by agents.

    ENV mapping:
      WORKSHOPS_AI_LLM__MODEL=gpt-4o-mini
      WORKSHOPS_AI_LLM__TEMPERATURE=0.0
      WORKSHOPS_AI_LLM__HTTP_TIMEOUT_SEC=60
      WORKSHOPS_AI_LLM__OPENAI_MAX_RETRIES=3
    """
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    http_timeout_sec: int = 60
    openai_max_retries: int = 3


class ChatSettings(BaseModel):
    """
    HTTP /chat endpoint knobs.

    ENV mapping:
      WORKSHOPS_AI_CHAT__TIMEOUT_SEC=60
      WORKSHOPS_AI_CHAT__MAX_MESSAGE_LEN=4000
    """
    timeout_sec: int = 60
    max_message_len: int = 4000


class WSSettings(BaseModel):
    """
    WebSocket knobs.

    ENV mapping:
      WORKSHOPS_AI_WS__TIMEOUT_SEC=75
      WORKSHOPS_AI_WS__COOKIE_NAME=wsai.sid
    """
    timeout_sec: int = 75
    cookie_name: str = "wsai.sid"


class LangchainSettings(BaseModel):
    """
    LangChain/LangSmith tracing config. We keep these under our settings so the app
    has a single source of truth and can propagate to environment as needed.

    ENV mapping:
      WORKSHOPS_AI_LANGCHAIN__TRACING_V2=false
      WORKSHOPS_AI_LANGCHAIN__ENDPOINT=...
      WORKSHOPS_AI_LANGCHAIN__API_KEY=...
      WORKSHOPS_AI_LANGCHAIN__PROJECT=workshops-ai
    """
    tracing_v2: bool = False
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    project: Optional[str] = None


class ManagerSettings(BaseModel):
    """
    Manager / router timeouts for agents (production-grade knobs).

    Semantics:
      - agent_timeout_sec: default per-agent budget if a per-tab override is not set.
      - threats_timeout_sec: optional override specifically for the Threats tab.

    ENV mapping:
      WORKSHOPS_AI_MANAGER__AGENT_TIMEOUT_SEC=45
      WORKSHOPS_AI_MANAGER__THREATS_TIMEOUT_SEC=90
    """
    agent_timeout_sec: int = 45
    threats_timeout_sec: Optional[int] = None  # None => use agent_timeout_sec


# ------------------------------------------------------------------------------
# Root settings
# ------------------------------------------------------------------------------
class AppSettings(BaseSettings):
    """
    Application root settings.

    Loads from environment with:
      - prefix: WORKSHOPS_AI_
      - nested delimiter: "__"
      - .env file support (for local dev)
      - strict but ignore unknown keys (extra="ignore")

    Examples:
      WORKSHOPS_AI_ENV=dev
      WORKSHOPS_AI_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
      WORKSHOPS_AI_SECURITY__API_KEYS=["dev-key"]
      WORKSHOPS_AI_PATHS__DATA_DIR=/mnt/data
      WORKSHOPS_AI_LLM__MODEL=gpt-4o-mini
      ENABLE_DOCS=false
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
    version: str = "1.0.0"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    enable_docs: bool = False  # may be also toggled by ENABLE_DOCS in your process env
    reload: bool = True        # dev default; uvicorn --reload typically controls this

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Feature groups (namespaced, map to WORKSHOPS_AI_<SECTION>__*)
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
    manager: ManagerSettings = Field(default_factory=ManagerSettings)  # <-- NEW (agent timeouts)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        return _parse_csv_or_json_list(v)

    # ---- Compatibility shims (older code paths expect top-level attributes) ----
    @property
    def chat_timeout_sec(self) -> int:
        """Legacy: return chat timeout as a top-level property."""
        return self.chat.timeout_sec

    @property
    def max_message_len(self) -> int:
        """Legacy: return max message length as a top-level property."""
        return self.chat.max_message_len

    @property
    def ws_timeout_sec(self) -> int:
        """Legacy: return WebSocket timeout as a top-level property."""
        return self.ws.timeout_sec

    def redacted_dict(self) -> dict:
        """
        Return a dict safe for logging (secrets redacted).
        - API keys -> "***"
        - JWT secret -> "***"
        - Tazila auth token -> "***"
        - LangChain API key -> "***"
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
# Cached accessor (singleton) + explicit cache reset for tests
# ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Load and cache application settings.
    Use reset_settings_cache() in tests to force a reload after mutating ENV.
    """
    return AppSettings()


def reset_settings_cache() -> None:
    """Clear the settings cache (useful in tests)."""
    get_settings.cache_clear()
