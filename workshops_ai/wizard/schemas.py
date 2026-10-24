# workshops_ai/wizard/schemas.py
"""
Wizard request/response schemas.

For LLM output schemas, we reuse existing schemas from workshops_ai.api.schema
where possible. Wizard-specific request models are defined here.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import Optional, List, Literal, Any
from workshops_ai.api.schema.asset import Asset
from workshops_ai.api.schema.impact import AssetImpact, ImpactEvaluation
from workshops_ai.api.schema.threat import AssetRisk
from workshops_ai.api.schema.control import AssetControls
from workshops_ai.api.schema.organization import Organization

# ===================================================================================
# Domain Models
# ===================================================================================

class Service(BaseModel):
    """Represents a core business service within the organization."""
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Unique localized name of the service (e.g., Broadband Internet)."
    )
    description: Optional[str] = Field(
        None,
        description="Description of the service, its function, or business relevance."
    )
    sector: Optional[str] = Field(
        None,
        description="Optional sector classification. Must match catalog; must not be translated."
    )


# ===================================================================================
# Structured Output Response Models (for OpenAI Structured Outputs)
# ===================================================================================

class ServicesResponse(BaseModel):
    """
    Wrapper for services generation response with OpenAI Structured Outputs.

    This model is used as response_format in LLM calls to enforce JSON schema
    validation at the API level, ensuring the LLM returns valid structured data.
    """
    model_config = ConfigDict(extra="forbid")

    services: List[Service] = Field(
        ...,
        description="List of core business services identified for the organization"
    )


class AssetsResponse(BaseModel):
    """
    Wrapper for assets generation response with OpenAI Structured Outputs.

    This model is used as response_format in LLM calls to enforce JSON schema
    validation at the API level, ensuring the LLM returns valid structured data.
    """
    model_config = ConfigDict(extra="forbid")

    assets: List[Asset] = Field(
        ...,
        description="List of IT and business assets identified for the organization"
    )


class ImpactsResponse(BaseModel):
    """
    Wrapper for impacts evaluation response with OpenAI Structured Outputs.

    This model is used as response_format in LLM calls to enforce JSON schema
    validation at the API level, ensuring the LLM returns valid structured data.

    Reuses ImpactEvaluation from api.schema.impact for consistency.
    """
    model_config = ConfigDict(extra="forbid")

    impacts: List[ImpactEvaluation] = Field(
        ...,
        description="List of impact evaluations with impactName and impactLevel (0-4) for a single asset"
    )


class BatchImpactsResponse(BaseModel):
    """
    Wrapper for batch impacts evaluation response with OpenAI Structured Outputs.

    Used when processing multiple assets in a single LLM call for better performance.
    Each asset_impacts entry contains the asset identification and its impact evaluations.
    """
    model_config = ConfigDict(extra="forbid")

    asset_impacts: List[AssetImpact] = Field(
        ...,
        description="List of impact evaluations for multiple assets. Each entry contains assetName, assetId, and impacts list."
    )


class RiskSelection(BaseModel):
    """
    Minimal LLM output for one evaluated threat.

    Only the two genuine model judgments are kept. The computed risk values
    (inherentRisk / residualRiskCalculated / estimatedResidualRisk) are not
    produced by the model — they are left unset and calculated downstream,
    exactly as before this change.
    """
    model_config = ConfigDict(extra="forbid")

    threatName: str = Field(
        ...,
        description="Exact threat name copied from the provided candidate threat catalog"
    )
    probability: Literal[0, 1, 2, 3, 4] = Field(
        ...,
        description="Probability level on the 0-4 scale (0 = not applicable)"
    )


class AssetRiskSelection(BaseModel):
    """
    Minimal LLM output for the threats evaluated for one asset.

    Asset IDs and full Risk metadata are restored/enriched from input after
    model generation.
    """
    model_config = ConfigDict(extra="forbid")

    assetName: str = Field(
        ...,
        description="Exact asset name copied from the input asset"
    )
    risks: List[RiskSelection] = Field(
        ...,
        description="Evaluated threats for this asset, threatName + probability only"
    )


class RisksResponse(BaseModel):
    """
    Wrapper for risks evaluation response with OpenAI Structured Outputs.

    This model is used as response_format in LLM calls to enforce JSON schema
    validation at the API level, ensuring the LLM returns valid structured data.

    Uses a minimal DTO because the model only decides threatName + probability.
    Computed risk values are calculated by the system after generation.
    """
    model_config = ConfigDict(extra="forbid")

    risks: List[RiskSelection] = Field(
        ...,
        description="List of evaluated threats (threatName + probability) for a single asset"
    )


class BatchRisksResponse(BaseModel):
    """
    Wrapper for batch risks evaluation response with OpenAI Structured Outputs.

    Used when processing multiple assets in a single LLM call for better performance.
    Each asset_risks entry contains the asset name and its evaluated threats.
    Asset IDs and full Risk metadata are restored/enriched after generation.
    """
    model_config = ConfigDict(extra="forbid")

    asset_risks: List[AssetRiskSelection] = Field(
        ...,
        description="List of assets with evaluated threats (threatName + probability)"
    )


class ControlSelection(BaseModel):
    """
    Minimal LLM output for one selected control.

    Full PlannedControl metadata is enriched from the authoritative catalog after
    model generation. The LLM should only decide which catalog control names apply.
    """
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Exact selected control name from the provided relevant controls catalog"
    )


class AssetControlSelection(BaseModel):
    """
    Minimal LLM output for controls selected for one asset.

    Asset IDs and full control metadata are restored/enriched from input and catalog
    data after model generation.
    """
    model_config = ConfigDict(extra="forbid")

    assetName: str = Field(
        ...,
        description="Exact asset name copied from the input asset"
    )
    controls: List[ControlSelection] = Field(
        ...,
        description="Selected controls for this asset, represented by exact catalog names only"
    )


class ControlsResponse(BaseModel):
    """
    Wrapper for control recommendations response with OpenAI Structured Outputs.

    This model is used as response_format in LLM calls to enforce JSON schema
    validation at the API level, ensuring the LLM returns valid structured data.

    Uses a minimal DTO because the endpoint only needs selected control names
    from the model. Full PlannedControl metadata is enriched from the catalog.
    """
    model_config = ConfigDict(extra="forbid")

    controls: List[ControlSelection] = Field(
        ...,
        description="List of selected control names for a single asset"
    )


class BatchControlsResponse(BaseModel):
    """
    Wrapper for batch control recommendations response with OpenAI Structured Outputs.

    Used when processing multiple assets in a single LLM call for better performance.
    Each asset_controls entry contains the asset name and selected control names.
    Asset IDs and full control metadata are restored/enriched after generation.
    """
    model_config = ConfigDict(extra="forbid")

    asset_controls: List[AssetControlSelection] = Field(
        ...,
        description="List of assets with selected control names"
    )


# ===================================================================================
# Common/Shared Request Models
# ===================================================================================

class WizardStepInput(BaseModel):
    """Base model for wizard step inputs."""
    model_config = ConfigDict(extra="forbid")

    ai_model: Optional[str] = Field(
        None,
        description="LLM model override (e.g., gpt-5, gpt-4o, gpt-4o-mini)"
    )
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = Field(
        None,
        examples=["low"],
        description=(
            "Reasoning effort override for reasoning models (gpt-5*, gpt-6*, o1, o3, o4). "
            "Cascade: body > env per model (WORKSHOPS_AI_WIZARD__REASONING_EFFORT_BY_MODEL) "
            "> env (WORKSHOPS_AI_WIZARD__REASONING_EFFORT) > Pydantic default. "
            "Per-model support: 'minimal' works only for gpt-5/gpt-5-mini/gpt-5-nano; "
            "'low'/'medium'/'high' work for all reasoning models. "
            "Ignored for non-reasoning models (gpt-4*, gpt-5.1-chat in chat-only mode)."
        )
    )


# ===================================================================================
# Scrape/Website
# ===================================================================================

class ScrapeWebsiteInput(BaseModel):
    """Input for /scrape/website endpoint (standalone, not session-based)."""
    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., description="URL to scrape")
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )


class ScrapeWebsiteOutput(BaseModel):
    """Output from website scraping - compatible with both scrape and web_search methods."""
    url: str = Field(..., description="Website URL (final URL after redirects)")
    content: str = Field(..., description="Combined scraped content from all pages")
    sources: Optional[List[str]] = Field(None, description="List of source URLs consulted during web search (only available for web_search method)")


# ===================================================================================
# Services
# ===================================================================================

class WebsiteContent(BaseModel):
    """Website content data when not using session.

    Can be empty object {} to disable web scraping entirely.
    """
    model_config = ConfigDict(extra="forbid")

    url: Optional[str] = Field(default=None, description="Website URL (optional)")
    content: Optional[str] = Field(default=None, description="Scraped website content (optional)")

    @model_validator(mode='before')
    @classmethod
    def handle_empty_object(cls, data: Any) -> Any:
        """Allow empty object {} to pass validation with None defaults."""
        if isinstance(data, dict) and not data:
            # Empty dict - return with None defaults
            return {"url": None, "content": None}
        return data


class ServicesInput(WizardStepInput):
    """Input for /wizard/services endpoint."""

    # Website content sources
    website_url: Optional[str] = Field(
        None,
        description="Website URL to scrape (will be cached in session)"
    )
    website_content: Optional[WebsiteContent] = Field(
        None,
        description="Provide website content directly when not using session"
    )

    # Search method
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )

    # Generation parameters
    language: str = Field(
        default="en",
        description="Output language (en/sk/cs)"
    )
    min_services: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Minimum number of services to generate (optional, 1-10)"
    )
    max_services: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Maximum number of services to generate (optional, 1-10)"
    )

    # Optional context for better generation
    organization_profile: Optional[Organization] = Field(
        None,
        description="Optional organization data (name, address, employees, etc.)"
    )

    # Refinement (for iterating on previous results)
    previous_services: Optional[List[Service]] = Field(
        None,
        description="Optional previous services for refinement; if omitted and user_instructions provided, loaded from session"
    )
    user_instructions: Optional[str] = Field(
        None,
        description="User instructions for refining previous services"
    )


class ServicesOutput(BaseModel):
    """Output for business services generation."""
    model_config = ConfigDict(extra="forbid")

    services: List[Service] = Field(
        ...,
        description="List of generated business services"
    )
    debug: Optional[dict] = Field(
        None,
        description="Debug information (only included when expose_debug_info=true)"
    )


# ===================================================================================
# Assets
# ===================================================================================

class AssetsInput(WizardStepInput):
    """Input for /wizard/assets endpoint."""

    # Website content sources
    website_url: Optional[str] = Field(
        None,
        description="Website URL to scrape (will be cached in session)"
    )
    website_content: Optional[WebsiteContent] = Field(
        None,
        description="Provide website content directly when not using session"
    )

    # Search method
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )

    # Services context (from previous step or direct)
    services: Optional[List[Service]] = Field(
        None,
        description="Services from previous step or provided directly"
    )

    # Generation parameters
    language: str = Field(
        default="en",
        description="Output language (en/sk/cs)"
    )
    min_assets: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Minimum number of assets to generate (optional, 1-100)"
    )
    max_assets: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Maximum number of assets to generate (optional, 1-100)"
    )

    # Optional context for better generation
    organization_profile: Optional[Organization] = Field(
        None,
        description="Optional organization data (name, address, employees, etc.)"
    )

    # Refinement (for iterating on previous results)
    previous_assets: Optional[List[Asset]] = Field(
        None,
        description="Optional previous assets for refinement; if omitted and user_instructions provided, loaded from session"
    )
    user_instructions: Optional[str] = Field(
        None,
        description="User instructions for refining previous assets"
    )


class AssetsOutput(BaseModel):
    """Output for assets generation."""
    model_config = ConfigDict(extra="forbid")

    assets: List[Asset] = Field(
        ...,
        description="List of generated IT and business assets"
    )
    debug: Optional[dict] = Field(
        None,
        description="Debug information (only included when expose_debug_info=true)"
    )


# ===================================================================================
# Impacts
# ===================================================================================

class ImpactsInput(WizardStepInput):
    """Input for /wizard/impacts endpoint."""

    # Website content sources
    website_url: Optional[str] = Field(
        None,
        description="Website URL to scrape (will be cached in session)"
    )
    website_content: Optional[WebsiteContent] = Field(
        None,
        description="Provide website content directly when not using session"
    )

    # Search method
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )

    # Assets context (from previous step or direct)
    assets: Optional[List[Asset]] = Field(
        None,
        description="Assets from previous step or provided directly"
    )

    # Services context (optional, for better impact evaluation)
    services: Optional[List[Service]] = Field(
        None,
        description="Optional services for context"
    )

    # Generation parameters
    language: str = Field(
        default="en",
        description="Output language (en/sk/cs)"
    )
    min_impacts_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Minimum impacts to evaluate per asset (optional, 1-20)"
    )
    max_impacts_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum impacts to evaluate per asset (optional, 1-20)"
    )

    # Optional context for better generation
    organization_profile: Optional[Organization] = Field(
        None,
        description="Optional organization data (name, address, employees, etc.)"
    )

    # Refinement (for iterating on previous results)
    user_instructions: Optional[str] = Field(
        None,
        description="User instructions for refining impact evaluations"
    )


class ImpactsOutput(BaseModel):
    """Output for impact evaluations."""
    model_config = ConfigDict(extra="forbid")

    asset_impacts: List[AssetImpact] = Field(
        ...,
        description="List of assets with their impact evaluations"
    )
    debug: Optional[dict] = Field(
        None,
        description="Debug information (only included when expose_debug_info=true)"
    )


# ===================================================================================
# Risks
# ===================================================================================

class RisksInput(WizardStepInput):
    """Input for /wizard/risks endpoint."""

    # Website content sources
    website_url: Optional[str] = Field(
        None,
        description="Website URL to scrape (will be cached in session)"
    )
    website_content: Optional[WebsiteContent] = Field(
        None,
        description="Provide website content directly when not using session"
    )

    # Search method
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )

    # Assets context (from previous step or direct)
    assets: Optional[List[Asset]] = Field(
        None,
        description="Assets from previous step or provided directly"
    )

    # Asset impacts context (optional, for better risk evaluation)
    asset_impacts: Optional[List[AssetImpact]] = Field(
        None,
        description="Optional asset impacts from previous step for context"
    )

    # Services context (optional, for better risk evaluation)
    services: Optional[List[Service]] = Field(
        None,
        description="Optional services for context"
    )

    # Generation parameters
    language: str = Field(
        default="en",
        description="Output language (en/sk/cs)"
    )
    min_threats_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=30,
        description="Minimum threats to evaluate per asset (optional, 1-30)"
    )
    max_threats_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=30,
        description="Maximum threats to evaluate per asset (optional, 1-30)"
    )

    # Optional context for better generation
    organization_profile: Optional[Organization] = Field(
        None,
        description="Optional organization data (name, address, employees, etc.)"
    )

    # Refinement (for iterating on previous results)
    user_instructions: Optional[str] = Field(
        None,
        description="User instructions for refining risk evaluations"
    )


class RisksOutput(BaseModel):
    """Output for risk evaluations."""
    model_config = ConfigDict(extra="forbid")

    asset_risks: List[AssetRisk] = Field(
        ...,
        description="List of assets with their risk evaluations (threat + probability)"
    )
    debug: Optional[dict] = Field(
        None,
        description="Debug information (only included when expose_debug_info=true)"
    )


# ===================================================================================
# Controls
# ===================================================================================

class ControlsInput(WizardStepInput):
    """Input for /wizard/controls endpoint."""

    # Website content sources
    website_url: Optional[str] = Field(
        None,
        description="Website URL to scrape (will be cached in session)"
    )
    website_content: Optional[WebsiteContent] = Field(
        None,
        description="Provide website content directly when not using session"
    )

    # Search method
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )

    # Assets context (from previous step or direct)
    assets: Optional[List[Asset]] = Field(
        None,
        description="Assets from previous step or provided directly"
    )

    # Asset risks context (REQUIRED - from previous step or direct)
    asset_risks: Optional[List[AssetRisk]] = Field(
        None,
        description="Asset risks from previous step (REQUIRED)"
    )

    # Generation parameters
    language: str = Field(
        default="en",
        description="Output language (en/sk/cs)"
    )
    min_controls_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Minimum controls per asset (optional, 1-20)"
    )
    max_controls_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum controls per asset (optional, 1-20)"
    )

    # Optional context for better generation
    organization_profile: Optional[Organization] = Field(
        None,
        description="Optional organization data (name, address, employees, etc.)"
    )

    # Refinement (for iterating on previous results)
    user_instructions: Optional[str] = Field(
        None,
        description="User instructions for control selection (e.g., 'Focus on low-cost controls')"
    )


class ControlsOutput(BaseModel):
    """Output for control recommendations."""
    model_config = ConfigDict(extra="forbid")

    asset_controls: List[AssetControls] = Field(
        ...,
        description="List of assets with their selected controls (with costs and threat coverage)"
    )
    debug: Optional[dict] = Field(
        None,
        description="Debug information (only included when expose_debug_info=true)"
    )


# ===================================================================================
# Evaluation (Meta-Endpoint)
# ===================================================================================

class StepTiming(BaseModel):
    """Timing information for a single wizard step."""
    model_config = ConfigDict(extra="forbid")

    step_name: str = Field(..., description="Step name (services/assets/impacts/risks/controls)")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    success: bool = Field(..., description="Whether step completed successfully")
    error: Optional[str] = Field(None, description="Error message if step failed")


class EvaluationData(BaseModel):
    """Combined results from full wizard evaluation."""
    model_config = ConfigDict(extra="forbid")

    services: Optional[List[Service]] = None
    assets: Optional[List[Asset]] = None
    asset_impacts: Optional[List[AssetImpact]] = None
    asset_risks: Optional[List[AssetRisk]] = None
    asset_controls: Optional[List[AssetControls]] = None


class EvaluationInput(WizardStepInput):
    """Input for /wizard/evaluation endpoint - runs all 5 wizard steps in sequence."""

    # Website input (required - either url or website_content)
    website_url: Optional[str] = Field(
        None,
        description="Website URL to scrape (will be cached in session)"
    )
    website_content: Optional[WebsiteContent] = Field(
        None,
        description="Pre-scraped website content (alternative to URL)"
    )

    # Search method
    search_method: Optional[Literal["scrape", "web_search"]] = Field(
        default=None,
        description="Content retrieval method: 'scrape' (BeautifulSoup) or 'web_search' (AI web search). If not provided, uses WORKSHOPS_AI_WIZARD__DEFAULT_SEARCH_METHOD setting."
    )

    # Common parameters
    organization_profile: Optional[Organization] = Field(
        None,
        description="Optional organization metadata for all steps"
    )
    language: str = Field(
        default="en",
        description="Output language for all steps (en/sk/cs)"
    )

    # Step-specific parameters (optional ranges)
    min_services: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Minimum number of services to generate (optional, 1-10)"
    )
    max_services: Optional[int] = Field(
        None,
        ge=1,
        le=10,
        description="Maximum number of services to generate (optional, 1-10)"
    )
    min_assets: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Minimum number of assets to generate (optional, 1-100)"
    )
    max_assets: Optional[int] = Field(
        None,
        ge=1,
        le=100,
        description="Maximum number of assets to generate (optional, 1-100)"
    )
    min_controls_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Minimum controls per asset (optional, 1-20)"
    )
    max_controls_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum controls per asset (optional, 1-20)"
    )
    min_impacts_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Minimum impacts to evaluate per asset (optional, 1-20)"
    )
    max_impacts_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Maximum impacts to evaluate per asset (optional, 1-20)"
    )
    min_threats_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=30,
        description="Minimum threats to evaluate per asset (optional, 1-30)"
    )
    max_threats_per_asset: Optional[int] = Field(
        None,
        ge=1,
        le=30,
        description="Maximum threats to evaluate per asset (optional, 1-30)"
    )

    # Control flags
    skip_existing: bool = Field(
        default=False,
        description="Skip steps if data already exists in session"
    )
    continue_on_error: bool = Field(
        default=True,
        description="Continue even if a step fails (returns partial results)"
    )


class EvaluationOutput(BaseModel):
    """Output for full wizard evaluation."""
    model_config = ConfigDict(extra="forbid")

    total_processing_time_ms: int = Field(
        ...,
        description="Total processing time for all steps in milliseconds"
    )
    data: EvaluationData = Field(
        ...,
        description="Results from all wizard steps (fields may be None if steps failed)"
    )
    step_timings: List[StepTiming] = Field(
        ...,
        description="Per-step timing and success information"
    )
    errors: Optional[List[str]] = Field(
        None,
        description="List of error messages if any step failed"
    )
    debug: Optional[dict] = Field(
        None,
        description="Debug information from all steps (only included when expose_debug_info=true)"
    )


# ===================================================================================
# State Management
# ===================================================================================

class WizardStateResponse(BaseModel):
    """Response for GET /wizard/state."""
    session_id: str
    steps_completed: List[str]
    data: dict
