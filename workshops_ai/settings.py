# File: workshops_ai/settings.py
# Purpose:
#   Centralized, Pydantic v2-based application settings with:
#   - Nested ENV loading via delimiter "__" (e.g., WORKSHOPS_AI_SECURITY__API_KEYS=…)
#   - Backward-compatible aliases for legacy flat env vars used in the project
#   - Robust CSV/JSON list parsing (for CORS origins, API keys, etc.)
#   - Safe redaction for secrets when logging
#   - Test-friendly caching and explicit cache reset
#
# Usage:
#   from workshops_ai.settings import get_settings
#   settings = get_settings()  # cached singleton

from __future__ import annotations

import json
import secrets
from functools import lru_cache
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, Field, field_validator
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
      WORKSHOPS_AI_SECURITY__DISABLE_ALL_AUTH=true
      WORKSHOPS_AI_SECURITY__ENABLE_API_KEY=true
      WORKSHOPS_AI_SECURITY__API_KEYS=["K1","K2"]
      WORKSHOPS_AI_SECURITY__ENABLE_JWT=true
      WORKSHOPS_AI_SECURITY__JWT_ALGORITHM=HS256
      WORKSHOPS_AI_SECURITY__JWT_SECRET=...
      WORKSHOPS_AI_SECURITY__JWT_ISSUER=https://auth.example.com
      WORKSHOPS_AI_SECURITY__JWT_AUDIENCE=workshops-ai
      WORKSHOPS_AI_SECURITY__JWKS_URL=https://auth.example.com/.well-known/jwks.json
    """

    disable_all_auth: bool = Field(
        default=False,
        description="Disable all authentication",
    )

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

    Canonical ENV (nested):
      WORKSHOPS_AI_PATHS__DATA_DIR=./workshops_ai/data
      WORKSHOPS_AI_PATHS__STATE_DIR=./workshops_ai/data/state
      WORKSHOPS_AI_PATHS__CATALOGS_DIR=./workshops_ai/data/catalogs

    Legacy (flat) ENV supported via AppSettings.model_post_init():
      WORKSHOPS_AI_DATA_DIR, WORKSHOPS_AI_STATE_DIR, WORKSHOPS_AI_CAT_DIR
    """
    data_dir: str = "data"
    state_dir: str = "data/state"
    catalogs_dir: str = "data/catalogs"


class CatalogSettings(BaseModel):
    """
    Catalog provider settings:
      - 'file': use local CSV files (dev/mockup mode)
      - 'http': fetch from Tazilla API (production mode)
    """
    mode: Literal["file", "http"] = "file"
    endpoint: Optional[str] = None
    auth_token: Optional[str] = None
    timeout_sec: int = 30


class RedisSettings(BaseModel):
    """
    Redis connection settings for sessions / rate-limiting.

    ENV mapping:
      WORKSHOPS_AI_REDIS__URL=redis://redis:6379/0
    """
    url: Optional[str] = None  # e.g. redis://localhost:6379/0


class RateLimitSettings(BaseModel):
    """
    Simple per-minute token-bucket caps per session/IP/route.

    ENV mapping:
      WORKSHOPS_AI_RATELIMIT__ENABLED=true
      WORKSHOPS_AI_RATELIMIT__PER_MINUTE=120
    """
    enabled: bool = False
    per_minute: int = 60


class OpenAIThrottleSettings(BaseModel):
    """
    OpenAI API throttling (RPM + TPM limits).

    ENV mapping:
      WORKSHOPS_AI_OPENAI_THROTTLE__ENABLED=true
      WORKSHOPS_AI_OPENAI_THROTTLE__RPM=450
      WORKSHOPS_AI_OPENAI_THROTTLE__TPM=2000000
      WORKSHOPS_AI_OPENAI_THROTTLE__MAX_CONCURRENT=50
    """
    enabled: bool = True
    rpm: int = 450
    tpm: int = 2_000_000
    max_concurrent: int = 50


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

    Canonical (nested) ENV:
      WORKSHOPS_AI_LLM__MODEL=gpt-5-mini
      WORKSHOPS_AI_LLM__TEMPERATURE=0.2
      WORKSHOPS_AI_LLM__CALL_TIMEOUT_SEC=240  # timeout for a single LLM/agent call
      WORKSHOPS_AI_LLM__OPENAI_MAX_RETRIES=3
      WORKSHOPS_AI_LLM__API_KEY=sk-...

    Legacy (flat) ENV handled in AppSettings.model_post_init():
      LLM_MODEL, LLM_TEMPERATURE, OPENAI_API_KEY

    Legacy nested alias still accepted: WORKSHOPS_AI_LLM__HTTP_TIMEOUT_SEC (use CALL_TIMEOUT_SEC instead).
    """
    model: str = "gpt-5.1-chat"
    temperature: float = 0.2
    call_timeout_sec: int = Field(
        default=240,
        validation_alias=AliasChoices("call_timeout_sec", "http_timeout_sec"),
        description=(
            "Timeout (s) for a single LLM/agent call. "
            "Distinct from chat.timeout_sec (whole chat request, may contain multiple LLM calls + tool calls)."
        ),
    )
    openai_max_retries: int = 3
    api_key: Optional[str] = None

    # Azure OpenAI settings (used when provider=azure)
    provider: str = "openai"  # "openai" or "azure"
    azure_endpoint: Optional[str] = None  # e.g. https://myresource.openai.azure.com/
    azure_api_version: str = "2025-01-01-preview"
    azure_deployment: Optional[str] = None  # if None, uses model name as deployment name


class ChatSettings(BaseModel):
    """
    HTTP /chat endpoint knobs.

    ENV mapping:
      WORKSHOPS_AI_CHAT__TIMEOUT_SEC=60
      WORKSHOPS_AI_CHAT__MAX_MESSAGE_LEN=4000

    Legacy (flat) ENV:
      WORKSHOPS_AI_MAX_MESSAGE_LEN (handled in AppSettings.model_post_init()).
    """
    timeout_sec: int = 60
    max_message_len: int = 4000


class WSSettings(BaseModel):
    """
    WebSocket knobs.

    Canonical ENV:
      WORKSHOPS_AI_WS__TIMEOUT_SEC=75
      WORKSHOPS_AI_WS__COOKIE_NAME=wsai.sid

    Legacy (flat) ENV:
      WORKSHOPS_AI_WS_TIMEOUT_SEC (handled in AppSettings.model_post_init()).
    """
    timeout_sec: int = 75
    cookie_name: str = "wsai.sid"


class LangchainSettings(BaseModel):
    """
    LangChain/LangSmith tracing config.

    Canonical (nested) ENV:
      WORKSHOPS_AI_LANGCHAIN__TRACING_V2=false
      WORKSHOPS_AI_LANGCHAIN__ENDPOINT=...
      WORKSHOPS_AI_LANGCHAIN__API_KEY=...
      WORKSHOPS_AI_LANGCHAIN__PROJECT=workshops-ai

    Legacy (flat) ENV handled in AppSettings.model_post_init():
      LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
    """
    tracing_v2: bool = False
    endpoint: Optional[str] = None
    api_key: Optional[str] = None
    project: Optional[str] = None


class LangfuseSettings(BaseModel):
    """
    Langfuse tracing and prompt management config.

    Canonical (nested) ENV:
      WORKSHOPS_AI_LANGFUSE__ENABLED=true
      WORKSHOPS_AI_LANGFUSE__PUBLIC_KEY=pk-lf-...
      WORKSHOPS_AI_LANGFUSE__SECRET_KEY=sk-lf-...
      WORKSHOPS_AI_LANGFUSE__BASE_URL=https://cloud.langfuse.com
      WORKSHOPS_AI_LANGFUSE__PROMPT_LABEL=production
      WORKSHOPS_AI_LANGFUSE__ENVIRONMENT=production

    Legacy (flat) ENV handled in AppSettings.model_post_init():
      LANGFUSE_ENABLED, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL, LANGFUSE_PROMPT_LABEL, LANGFUSE_ENVIRONMENT
    """
    enabled: bool = False
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: str = "https://cloud.langfuse.com"
    prompt_label: Optional[str] = Field(
        default=None,
        description="Label to use when fetching prompts from Langfuse (e.g., 'production', 'development'). If not set, uses latest production version."
    )
    environment: Optional[str] = Field(
        default=None,
        description="Environment to organize traces in Langfuse (e.g., 'production', 'staging', 'development'). Maps to LANGFUSE_TRACING_ENVIRONMENT."
    )


class ManagerSettings(BaseModel):
    """
    Manager / router timeouts for agents (production-grade knobs).

    Semantics:
      - agent_timeout_sec: default per-agent budget if a per-tab override is not set.
      - threats_timeout_sec: optional override specifically for the Threats tab.
      - max_history_tokens: max tokens for conversation history sent to LLM (prevents context overflow).

    ENV mapping:
      WORKSHOPS_AI_MANAGER__AGENT_TIMEOUT_SEC=45
      WORKSHOPS_AI_MANAGER__THREATS_TIMEOUT_SEC=90
      WORKSHOPS_AI_MANAGER__MAX_HISTORY_TOKENS=4000
    """
    agent_timeout_sec: int = 45
    threats_timeout_sec: Optional[int] = None  # None => use agent_timeout_sec
    max_history_tokens: int = Field(
        default=4000,
        description="Max tokens for conversation history sent to LLM (trimmed by pre_model_hook)"
    )


class WizardSettings(BaseModel):
    """
    AI Wizard configuration for onboarding flow.

    ENV mapping:
      WORKSHOPS_AI_WIZARD__LLM_TIMEOUT_SEC=60
      WORKSHOPS_AI_WIZARD__LLM_MAX_RETRIES=3
      WORKSHOPS_AI_WIZARD__DEFAULT_MODEL=gpt-5-mini
      WORKSHOPS_AI_WIZARD__TEMPERATURE=0.2
      WORKSHOPS_AI_WIZARD__ALLOWED_MODELS=["gpt-5","gpt-5-mini","gpt-4.1","gpt-4o","gpt-4o-mini"]
      WORKSHOPS_AI_WIZARD__EXPOSE_DEBUG_INFO=true
      WORKSHOPS_AI_WIZARD__MAX_CONTENT_LENGTH=20000
      WORKSHOPS_AI_WIZARD__OPENAI_PRICING='{"gpt-4o":{"prompt":0.0025,"completion":0.010}}'

      # Entity mapping
      WORKSHOPS_AI_WIZARD__USE_THREAT_MAPPING=true

      # Content retrieval method
      WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD=scrape

      # Batch processing
      WORKSHOPS_AI_WIZARD__MAX_BATCH_SIZE=20

      # Web search (AI-powered)
      WORKSHOPS_AI_WIZARD__WEB_SEARCH_ENABLED=true
      WORKSHOPS_AI_WIZARD__WEB_SEARCH_MODEL=gpt-4o
      WORKSHOPS_AI_WIZARD__WEB_SEARCH_TIMEOUT_SEC=60
      WORKSHOPS_AI_WIZARD__WEB_SEARCH_MIN_CONTENT_CHARS=50

      # Traditional scraping (BeautifulSoup)
      WORKSHOPS_AI_WIZARD__SCRAPE_ENABLED=true
      WORKSHOPS_AI_WIZARD__SCRAPE_TIMEOUT_SEC=10
      WORKSHOPS_AI_WIZARD__SCRAPE_MAX_RETRIES=2
      WORKSHOPS_AI_WIZARD__SCRAPE_CONCURRENT_LIMIT=8
      WORKSHOPS_AI_WIZARD__SCRAPE_MAX_PAGE_CHARS=6144
      WORKSHOPS_AI_WIZARD__SCRAPE_MAX_COMBINED_CHARS=200000
      WORKSHOPS_AI_WIZARD__SCRAPE_MIN_PAGE_CHARS=100
      WORKSHOPS_AI_WIZARD__SCRAPE_STRICT_SANITIZATION=true
      WORKSHOPS_AI_WIZARD__SCRAPE_MAX_SECURITY_WARNINGS=5
    """
    llm_timeout_sec: int = Field(
        default=240,
        description="Timeout per wizard step LLM call (seconds) - increased for complex operations"
    )
    llm_max_retries: int = Field(
        default=3,
        description="Max retries for LLM validation errors"
    )
    default_model: str = Field(
        default="gpt-5-mini",
        description="Default LLM model for wizard steps (used when request body has no ai_model)"
    )
    temperature: float = Field(
        default=0.2,
        description="Temperature for non-reasoning models in wizard (reasoning models ignore this)"
    )
    reasoning_effort: Optional[str] = Field(
        default="medium",
        description=(
            "Reasoning effort for reasoning models. Accepts 'minimal' (gpt-5/-mini/-nano only), "
            "'low', 'medium', 'high' (universal). Set to None to omit parameter (model default)."
        )
    )
    allowed_models: list[str] = Field(
        default_factory=lambda: [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5.1-chat",
            "gpt-5.4-mini",
            "gpt-5.4-nano",
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini"
        ],
        description="List of allowed LLM models for wizard"
    )
    expose_debug_info: bool = Field(
        default=True,
        description="Include debug info in response body and headers (X-GPT-*, X-Wizard-*)"
    )
    max_content_length: int = Field(
        default=20000,
        description="Max characters of website content to send to LLM"
    )
    openai_pricing: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "gpt-5": {
                "prompt": 0.00125,     # $1.25 per 1M tokens (input)
                "completion": 0.010,   # $10.00 per 1M tokens (output)
            },
            "gpt-5-mini": {
                "prompt": 0.00025,     # $0.25 per 1M tokens (input)
                "completion": 0.002,   # $2.00 per 1M tokens (output)
            },
            "gpt-4.1": {
                "prompt": 0.002,       # $2.00 per 1M tokens (input)
                "completion": 0.008,   # $8.00 per 1M tokens (output)
            },
            "gpt-4o": {
                "prompt": 0.0025,      # $2.50 per 1M tokens (input)
                "completion": 0.010,   # $10.00 per 1M tokens (output)
            },
            "gpt-4o-mini": {
                "prompt": 0.00015,     # $0.15 per 1M tokens (input)
                "completion": 0.0006,  # $0.60 per 1M tokens (output)
            },
        },
        description="OpenAI pricing per 1M tokens (January 2025). Can be overridden via ENV JSON."
    )

    openai_per_request_limits: dict[str, int] = Field(
        default_factory=lambda: {
            # Max tokens accepted in a single LLM call. Triggers RequestTooLargeError
            # (with suggested smaller batch size) when exceeded — independent of capacity TPM.
            "gpt-5-mini": 300000,
            "gpt-5-nano": 500000,
            "gpt-5.1-chat": 300000,
            "gpt-5.4-mini": 300000,
            "gpt-5.4-nano": 500000,
            "gpt-4.1": 300000,
            "gpt-4o": 30000,
            "gpt-4o-mini": 200000,
        },
        description="Per-request token limits per model (max tokens per single LLM call)"
    )

    openai_capacity_limits: dict[str, int] = Field(
        default_factory=lambda: {
            # Azure deployment TPM (tokens-per-minute) capacity per model.
            # Enforced by OpenAIThrottle via per-model sliding window — calls wait
            # when the rolling 60-second sum would exceed the limit.
            "gpt-5-mini": 1_000_000,
            "gpt-5-nano": 5_000_000,
            "gpt-5.1-chat": 1_000_000,
            "gpt-5.4-mini": 1_000_000,
            "gpt-5.4-nano": 5_000_000,
            "gpt-4.1": 1_000_000,
            "gpt-4o": 450_000,
            "gpt-4o-mini": 2_000_000,
        },
        description="Per-model TPM capacity (Azure deployment quota). Overrides global openai_throttle.tpm for matching models."
    )

    # Content retrieval method
    default_search_method: Literal["scrape", "web_search"] = Field(
        default="scrape",
        description="Default content retrieval method for wizard endpoints"
    )

    # Batch processing (internal chunking)
    max_batch_size: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Max assets per LLM call for impacts/risks/controls generation. Assets are automatically split into batches of this size. Lower = safer/slower, higher = faster but may hit LLM context limits."
    )

    # Web search settings
    web_search_enabled: bool = Field(
        default=True,
        description="Enable AI web search as alternative to scraping"
    )
    web_search_model: str = Field(
        default="gpt-5-nano",
        description="OpenAI model for web search (requires gpt-4o or newer)"
    )
    web_search_reasoning_effort: Optional[str] = Field(
        default=None,
        description="Reasoning effort for web search with reasoning models (gpt-5, o1, etc): 'low', 'medium', 'high', or None for non-reasoning models"
    )
    web_search_timeout_sec: int = Field(
        default=120,
        description="Timeout for web search API calls (seconds) - AI web search can take longer"
    )
    web_search_min_content_chars: int = Field(
        default=50,
        description="Minimum content length to consider search successful"
    )

    # Traditional scraping settings
    scrape_enabled: bool = Field(
        default=True,
        description="Enable traditional BeautifulSoup scraping"
    )
    scrape_timeout_sec: int = Field(
        default=10,
        description="HTTP request timeout for scraping (seconds)"
    )
    scrape_max_retries: int = Field(
        default=2,
        description="Max retries for failed HTTP requests"
    )
    scrape_concurrent_limit: int = Field(
        default=8,
        description="Max concurrent HTTP requests when fetching multiple pages"
    )
    scrape_max_page_chars: int = Field(
        default=6_144,
        description="Max characters per scraped page"
    )
    scrape_max_combined_chars: int = Field(
        default=200_000,
        description="Max characters across all scraped pages combined"
    )
    scrape_min_page_chars: int = Field(
        default=100,
        description="Minimum characters for page content to be valid"
    )
    scrape_user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        description="User-Agent header for HTTP requests"
    )

    # Content security settings
    scrape_strict_sanitization: bool = Field(
        default=True,
        description="Enable strict content sanitization (removes suspicious patterns)"
    )
    scrape_max_security_warnings: int = Field(
        default=5,
        description="Max security warnings before rejecting content"
    )

    # Session settings
    session_ttl_hours: int = Field(
        default=8,
        description="Session TTL in hours (wizard state expiration)"
    )
    session_creation_rate_limit: int = Field(
        default=10,
        description="Max new sessions per IP per hour (0 = unlimited)"
    )

    @field_validator("allowed_models", mode="before")
    @classmethod
    def parse_allowed_models(cls, v):
        return _parse_csv_or_json_list(v)


class RagSettings(BaseModel):
    """
    Retrieval-Augmented Generation (RAG) configuration.

    Canonical (nested) ENV:
      WORKSHOPS_AI_RAG__ENABLED=true
      WORKSHOPS_AI_RAG__STORE=chroma
      WORKSHOPS_AI_RAG__DATA_DIR=./data/chroma
      WORKSHOPS_AI_RAG__NAMESPACE=rag
      WORKSHOPS_AI_RAG__EMBED_PROVIDER=fastembed
      WORKSHOPS_AI_RAG__EMBED_MODEL=BAAI/bge-small-en-v1.5
      WORKSHOPS_AI_RAG__TOP_K=6
      WORKSHOPS_AI_RAG__MMR_ENABLED=true|false
      WORKSHOPS_AI_RAG__MMR_LAMBDA=0.5
      WORKSHOPS_AI_RAG__MMR_POOL=12
      WORKSHOPS_AI_RAG__CHUNK_SIZE=1200
      WORKSHOPS_AI_RAG__CHUNK_OVERLAP=200
      WORKSHOPS_AI_RAG__MAX_PAGES=0
      WORKSHOPS_AI_RAG__UUID5_NAMESPACE=00000000-0000-5000-a000-000000000000

    Back-compat:
      - 'chroma_dir' kept for older code paths; prefer 'data_dir'.
    """
    enabled: bool = True
    store: Literal["chroma"] = "chroma"

    # index paths / namespacing
    data_dir: str = "data/chroma"
    chroma_dir: Optional[str] = None  # old name; if set, prefer this over data_dir
    namespace: str = "rag"

    # embeddings
    embed_provider: Literal["fastembed", "openai"] = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"  # or text-embedding-3-small if openai

    # retriever params
    top_k: int = 6
    use_mmr: bool = True          # old name
    mmr_enabled: Optional[bool] = None  # new explicit flag
    mmr_lambda: float = 0.5
    mmr_pool: int = 12

    # ingest params
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_pages: int = 0
    uuid5_namespace: str = "00000000-0000-5000-a000-000000000000"

    # convenience
    def effective_index_dir(self) -> str:
        return self.chroma_dir or self.data_dir

    def effective_use_mmr(self) -> bool:
        return self.mmr_enabled if self.mmr_enabled is not None else self.use_mmr


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
      WORKSHOPS_AI_LLM__MODEL=gpt-5-mini
      ENABLE_DOCS=false  (legacy flat toggle; also supports WORKSHOPS_AI_ENABLE_DOCS)
    """
    model_config = SettingsConfigDict(
        env_prefix="WORKSHOPS_AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
    )

    # App meta / server
    env: str = "dev"
    service_name: str = "AI Risk Analysis API"
    version: str = "1.1.0"
    log_level: str = "INFO"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    enable_docs: bool = False
    reload: bool = True

    # CORS
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Feature groups (namespaced)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    openai_throttle: OpenAIThrottleSettings = Field(default_factory=OpenAIThrottleSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    telemetry: TelemetrySettings = Field(default_factory=TelemetrySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    ws: WSSettings = Field(default_factory=WSSettings)
    langchain: LangchainSettings = Field(default_factory=LangchainSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    manager: ManagerSettings = Field(default_factory=ManagerSettings)
    wizard: WizardSettings = Field(default_factory=WizardSettings)
    rag: RagSettings = Field(default_factory=RagSettings)

    # AI manager/router spec (used by HTTP/WS chat routers)
    # ENV: WORKSHOPS_AI_AI_MANAGER_SPEC=workshops_ai.graph.manager:route_chat
    ai_manager_spec: str = "workshops_ai.graph.manager:route_chat"

    # LangGraph checkpointer knobs (flat keys to match .env and checkpointer factory)
    # ENV (flat, not nested):
    #   WORKSHOPS_AI_CHECKPOINTER=redis|sqlite|memory
    #   WORKSHOPS_AI_REDIS_URL=redis://localhost:6379/0
    #   WORKSHOPS_AI_CHECKPOINT_NS=lg:ckpt
    #   WORKSHOPS_AI_CHECKPOINT_TTL_SEC=86400
    #   WORKSHOPS_AI_SQLITE_CKPT_PATH=/mnt/data/checkpoints.db
    checkpointer: Optional[str] = None
    redis_url: Optional[str] = None
    checkpoint_ns: str = "lg:ckpt"
    checkpoint_ttl_sec: Optional[int] = None
    sqlite_ckpt_path: str = "/mnt/data/checkpoints.db"

    # ---- Per-tab agent timeouts (flat env) ----
    agent_timeout_sec_assets: Optional[int] = Field(
        default=None, validation_alias="AGENT_TIMEOUT_SEC_ASSETS"
    )
    agent_timeout_sec_impacts: Optional[int] = Field(
        default=None, validation_alias="AGENT_TIMEOUT_SEC_IMPACTS"
    )
    agent_timeout_sec_threats: Optional[int] = Field(
        default=None, validation_alias="AGENT_TIMEOUT_SEC_THREATS"
    )

    # Paths
    data_dir_legacy: Optional[str] = Field(default=None, validation_alias="DATA_DIR")
    state_dir_legacy: Optional[str] = Field(default=None, validation_alias="STATE_DIR")
    cat_dir_legacy: Optional[str] = Field(default=None, validation_alias="CAT_DIR")

    # WS/chat legacy flat
    ws_timeout_sec_legacy: Optional[int] = Field(default=None, validation_alias="WS_TIMEOUT_SEC")
    max_message_len_legacy: Optional[int] = Field(default=None, validation_alias="MAX_MESSAGE_LEN")

    # Host/port legacy (no WORKSHOPS_AI_ prefix in .env)
    app_host_env: Optional[str] = Field(default=None, validation_alias="APP_HOST")
    app_port_env: Optional[int] = Field(default=None, validation_alias="APP_PORT")

    # Docs legacy flat
    enable_docs_flat: Optional[bool] = Field(default=None, validation_alias="ENABLE_DOCS")

    # LangChain legacy flat
    langchain_tracing_v2_flat: Optional[bool] = Field(default=None, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_endpoint_flat: Optional[str] = Field(default=None, validation_alias="LANGCHAIN_ENDPOINT")
    langchain_api_key_flat: Optional[str] = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project_flat: Optional[str] = Field(default=None, validation_alias="LANGCHAIN_PROJECT")

    # Langfuse flat (standard Langfuse ENV names without WORKSHOPS_AI_ prefix)
    langfuse_enabled_flat: Optional[bool] = Field(default=None, validation_alias="LANGFUSE_ENABLED")
    langfuse_public_key_flat: Optional[str] = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key_flat: Optional[str] = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url_flat: Optional[str] = Field(default=None, validation_alias="LANGFUSE_BASE_URL")
    langfuse_prompt_label_flat: Optional[str] = Field(default=None, validation_alias="LANGFUSE_PROMPT_LABEL")
    langfuse_environment_flat: Optional[str] = Field(default=None, validation_alias="LANGFUSE_ENVIRONMENT")

    # LLM legacy flat
    llm_model_flat: Optional[str] = Field(default=None, validation_alias="LLM_MODEL")
    llm_temperature_flat: Optional[float] = Field(default=None, validation_alias="LLM_TEMPERATURE")
    openai_api_key_flat: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors(cls, v):
        return _parse_csv_or_json_list(v)

    # ---- Convenience / compatibility helpers ----
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

    # Effective timeout per tab with overrides from .env
    def agent_timeout_for_tab(self, tab: str) -> int:
        t = tab.lower()
        if t == "assets" and self.agent_timeout_sec_assets:
            return self.agent_timeout_sec_assets
        if t == "impacts" and self.agent_timeout_sec_impacts:
            return self.agent_timeout_sec_impacts
        if t == "threats":
            return (
                self.agent_timeout_sec_threats
                or self.manager.threats_timeout_sec
                or self.manager.agent_timeout_sec
            )
        return self.manager.agent_timeout_sec

    def redacted_dict(self) -> dict:
        """
        Return a dict safe for logging (secrets redacted).
        - API keys -> "***"
        - JWT secret -> "***"
        - Catalog auth token -> "***"
        - LangChain API key -> "***"
        """
        data = self.model_dump()

        # Security
        sec = data.get("security", {})
        if sec.get("api_keys"):
            sec["api_keys"] = ["***", f"({len(self.security.api_keys)} keys)"]
        if sec.get("jwt_secret"):
            sec["jwt_secret"] = "***"

        # Catalog
        cat = data.get("catalog", {})
        if cat.get("auth_token"):
            cat["auth_token"] = "***"

        # LangChain
        lc = data.get("langchain", {})
        if lc.get("api_key"):
            lc["api_key"] = "***"

        # Langfuse
        lf = data.get("langfuse", {})
        if lf.get("public_key"):
            lf["public_key"] = "***"
        if lf.get("secret_key"):
            lf["secret_key"] = "***"

        return data

    # Post-processing to apply legacy/flat env knobs
    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        # ENABLE_DOCS (flat) wins if present
        if self.enable_docs_flat is not None:
            self.enable_docs = bool(self.enable_docs_flat)

        # Host/port (APP_HOST / APP_PORT) if provided
        if self.app_host_env:
            self.app_host = self.app_host_env
        if self.app_port_env:
            self.app_port = self.app_port_env

        # Paths legacy mapping
        if self.data_dir_legacy:
            self.paths.data_dir = self.data_dir_legacy
        if self.state_dir_legacy:
            self.paths.state_dir = self.state_dir_legacy
        if self.cat_dir_legacy:
            self.paths.catalogs_dir = self.cat_dir_legacy

        # WS timeout legacy (flat)
        if self.ws_timeout_sec_legacy:
            self.ws.timeout_sec = self.ws_timeout_sec_legacy

        # Chat max message length legacy (flat)
        if self.max_message_len_legacy:
            self.chat.max_message_len = self.max_message_len_legacy

        # LangChain flat env overrides (common upstream names)
        if self.langchain_tracing_v2_flat is not None:
            self.langchain.tracing_v2 = bool(self.langchain_tracing_v2_flat)
        if self.langchain_endpoint_flat:
            self.langchain.endpoint = self.langchain_endpoint_flat
        if self.langchain_api_key_flat:
            self.langchain.api_key = self.langchain_api_key_flat
        if self.langchain_project_flat:
            self.langchain.project = self.langchain_project_flat

        # Langfuse flat env overrides (standard Langfuse ENV names)
        if self.langfuse_enabled_flat is not None:
            self.langfuse.enabled = bool(self.langfuse_enabled_flat)
        if self.langfuse_public_key_flat:
            self.langfuse.public_key = self.langfuse_public_key_flat
        if self.langfuse_secret_key_flat:
            self.langfuse.secret_key = self.langfuse_secret_key_flat
        if self.langfuse_base_url_flat:
            self.langfuse.base_url = self.langfuse_base_url_flat
        if self.langfuse_prompt_label_flat:
            self.langfuse.prompt_label = self.langfuse_prompt_label_flat
        if self.langfuse_environment_flat:
            self.langfuse.environment = self.langfuse_environment_flat

        # LLM flat env overrides
        if self.llm_model_flat:
            self.llm.model = self.llm_model_flat
        if self.llm_temperature_flat is not None:
            self.llm.temperature = float(self.llm_temperature_flat)
        if self.openai_api_key_flat and not self.llm.api_key:
            self.llm.api_key = self.openai_api_key_flat

        # RAG back-compat: prefer chroma_dir if provided
        if self.rag.chroma_dir:
            pass


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
