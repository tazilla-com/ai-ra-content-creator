# workshops_ai/wizard/router.py
"""
FastAPI router for AI Wizard endpoints.

Provides endpoints for cybersecurity onboarding workflow:
- /wizard/services - Generate critical business services
- /wizard/assets - Generate IT assets for services
- /wizard/impacts - Evaluate CIA impacts of asset compromise
- /wizard/risks - Identify and evaluate risks
- /wizard/controls - Recommend security controls
- /wizard/evaluation - Overall security evaluation
- /wizard/state - Get/clear wizard state

Each endpoint:
- Secured by API key/JWT (reuses existing auth)
- Accepts input data directly or loads from previous step in session
- Stores step data in Redis (org_id + user_id keys)
- Returns structured JSON
- Supports model override per request
"""
from __future__ import annotations

import logging
import secrets
import time
from fastapi import APIRouter, Depends, HTTPException, status, Response, Header, Request
from typing import Dict, Any, Optional

from workshops_ai.app.security import require_auth
from workshops_ai.app.services.redis_pool import get_redis
from workshops_ai.settings import get_settings
from workshops_ai.wizard.schemas import (
    # Inputs
    ServicesInput,
    AssetsInput,
    ImpactsInput,
    RisksInput,
    ControlsInput,
    EvaluationInput,
    # Outputs
    ServicesOutput,
    AssetsOutput,
    ImpactsOutput,
    RisksOutput,
    ControlsOutput,
    EvaluationOutput,
    # Evaluation specific
    EvaluationData,
    StepTiming,
    # State
    WizardStateResponse,
)
from workshops_ai.wizard.session import (
    get_wizard_state,
    save_wizard_step,
    get_wizard_step,
    clear_wizard_state,
    validate_session_id,
)
from workshops_ai.wizard.llm_helper import call_llm_with_structured_output
# Note: Old placeholder prompts (SERVICES_PROMPT, etc.) are not used in new implementation
# New implementation uses ServiceGenerator with prompts from prompts.py
from workshops_ai.wizard.service_generator import ServiceGenerator
from workshops_ai.wizard.asset_generator import AssetGenerator
from workshops_ai.wizard.impact_generator import ImpactGenerator
from workshops_ai.wizard.risk_generator import RiskGenerator
from workshops_ai.wizard.control_generator import ControlGenerator
from workshops_ai.wizard.content_resolver import ContentResolver
from workshops_ai.wizard.exceptions import ContentNotFoundError

logger = logging.getLogger(__name__)

# Router (auth applied in main.py when mounting)
router = APIRouter(
    prefix="/wizard",
    tags=["wizard"],
)


# =============================================================================
# Dependencies: Session ID management
# =============================================================================

async def get_session_id(
    request: Request,
    response: Response,
    x_session_id: Optional[str] = Header(None, alias="X-Session-Id"),
) -> str:
    """
    Dependency: Extract or generate session ID for wizard state.

    Flow:
    1. If client provides X-Session-Id header:
       - Validate format (22 base64url chars)
       - Return session ID (session will be created/loaded by endpoint)
    2. If client doesn't provide session ID:
       - Check rate limit for new session creation (per IP)
       - Generate new session ID
       - Set X-Session-Id in response header
       - Return session ID

    Returns:
        Session ID (format: 1-40 alphanumeric characters)

    Raises:
        HTTPException:
            - 400 if session ID format invalid
            - 429 if rate limit exceeded for new sessions
    """
    settings = get_settings()

    # Case 1: Client provided session ID
    if x_session_id and x_session_id.strip():
        # Validate format
        if not validate_session_id(x_session_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid session ID format. Expected: 1-40 alphanumeric characters (a-z, A-Z, 0-9, _, -)"
            )

        response.headers["X-Session-Id"] = x_session_id

        logger.debug(
            "Using client-provided session ID",
            extra={"session_id": x_session_id}
        )
        return x_session_id

    # Case 2: Generate new session ID
    # Apply rate limit for new session creation
    rate_limit = settings.wizard.session_creation_rate_limit
    if rate_limit > 0:
        redis = await get_redis()
        if redis:
            try:
                # Get client IP
                client_ip = request.client.host if request.client else "unknown"
                rl_key = f"session_create_rl:{client_ip}"

                # Increment counter
                count = await redis.incr(rl_key)

                # Set expiry on first increment (1 hour window)
                if count == 1:
                    await redis.expire(rl_key, 3600)

                # Check limit
                if count > rate_limit:
                    logger.warning(
                        "Session creation rate limit exceeded",
                        extra={
                            "client_ip": client_ip,
                            "count": count,
                            "limit": rate_limit
                        }
                    )
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail=f"Too many new sessions created. Limit: {rate_limit} per hour per IP."
                    )
            except HTTPException:
                raise
            except Exception as e:
                # Log error but don't block session creation
                logger.error(
                    "Failed to check session creation rate limit",
                    extra={"error": str(e)}
                )

    # Generate new session ID (22 characters base64url)
    new_id = secrets.token_urlsafe(16)

    # Set in response header
    response.headers["X-Session-Id"] = new_id

    logger.info(
        "Generated new session ID",
        extra={"session_id": new_id}
    )

    return new_id


# =============================================================================
# Endpoints
# =============================================================================

@router.post(
    "/services",
    response_model=ServicesOutput,
    summary="Generate business services",
    description="Identify critical business services for the organization"
)
async def generate_services(
    request: ServicesInput,
    response: Response,
    session_id: str = Depends(get_session_id),
) -> ServicesOutput:
    """
    Generate critical business services from website content.

    Resolution priority for website content:
    1. Direct text from request.website_text
    2. URL scraping from request.website_url (cached in session)
    3. Cached content from previous request in session

    Args:
        request: ServicesInput with website content and generation params
        response: FastAPI Response for injecting headers
        session_id: Session ID (22 random base64url chars)

    Returns:
        ServicesOutput with generated services and optional debug info

    Raises:
        ContentNotFoundError: If no website content available
        AITimeoutError: If LLM request times out
        AIResponseError: If LLM response invalid
        ValidationRetriesExhausted: If validation fails after retries
    """
    settings = get_settings()

    logger.info(
        "wizard_services_started",
        extra={
            "session_id": session_id,
            "model": request.ai_model,
            "language": request.language,
            "min_services": request.min_services,
            "max_services": request.max_services,
        }
    )

    # Initialize components
    redis = await get_redis()
    content_resolver = ContentResolver(redis)
    service_generator = ServiceGenerator(model=request.ai_model)

    # Step 1: Resolve website content
    # Check if web scraping is disabled (empty website_content object)
    scraping_disabled = (
        request.website_content is not None and
        not request.website_content.url and
        not request.website_content.content
    )

    if scraping_disabled:
        # Web scraping explicitly disabled - use empty content
        website_content = ""
        content_metadata = {
            "source": "disabled",
            "content_length": 0,
        }
        logger.info(
            "web_scraping_disabled",
            extra={
                "session_id": session_id,
                "reason": "empty_website_content_object",
            }
        )
    else:
        # Normal flow - resolve content from cache or scraping
        try:
            website_content, content_metadata = await content_resolver.resolve_content(
                session_id=session_id,
                website_text=request.website_content.content if request.website_content else None,
                website_url=request.website_url,
                search_method=request.search_method,
                ai_model=request.ai_model,
            )

            logger.info(
                "content_resolved",
                extra={
                    "session_id": session_id,
                    "source": content_metadata.get("source"),
                    "content_length": content_metadata.get("content_length"),
                }
            )

        except ContentNotFoundError as e:
            logger.error(
                "content_resolution_failed",
                extra={
                    "session_id": session_id,
                    "error": str(e),
                }
            )
            raise

    # Step 2: Build organization profile from request
    organization_profile = None
    if request.organization_profile:
        organization_profile = {
            "name": request.organization_profile.name,
            "industry": request.organization_profile.industry,
            "address": request.organization_profile.address,
            "employees": request.organization_profile.employees,
        }

    # Step 3: Load previous services from session (if refining)
    previous_services = None
    if request.user_instructions:
        # User is refining - load previous services
        previous_step = await get_wizard_step(session_id, "services")
        if previous_step and previous_step.get("services"):
            from workshops_ai.wizard.schemas import Service
            previous_services = [
                Service(**s) for s in previous_step["services"]
            ]
            logger.info(
                "previous_services_loaded",
                extra={
                    "session_id": session_id,
                    "count": len(previous_services),
                }
            )

    # Step 4: Generate services with LLM
    try:
        services, usage_metrics, debug_info = await service_generator.generate_services(
            website_content=website_content,
            language=request.language,
            min_services=request.min_services,
            max_services=request.max_services,
            organization_profile=organization_profile,
            user_instructions=request.user_instructions,
            previous_services=previous_services,
            session_id=session_id,
        )

        logger.info(
            "services_generated",
            extra={
                "session_id": session_id,
                "services_count": len(services),
                "tokens": usage_metrics.total_tokens,
                "cost_usd": usage_metrics.cost_usd,
            }
        )

    except Exception as e:
        logger.error(
            "service_generation_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise

    # Step 5: Save to session
    step_data = {
        "services": [s.model_dump() for s in services],
        "language": request.language,
        "model": service_generator.model_name,
        "usage": usage_metrics.model_dump(),
    }

    await save_wizard_step(session_id, "services", step_data)

    logger.info(
        "services_saved_to_session",
        extra={"session_id": session_id}
    )

    # Step 6: Add response headers (if enabled)
    if settings.wizard.expose_debug_info:
        # GPT headers (per-request metrics)
        response.headers["X-GPT-Model"] = service_generator.model_name
        response.headers["X-GPT-Tokens-Prompt"] = str(usage_metrics.prompt_tokens)
        response.headers["X-GPT-Tokens-Completion"] = str(usage_metrics.completion_tokens)
        response.headers["X-GPT-Tokens-Total"] = str(usage_metrics.total_tokens)
        response.headers["X-GPT-Cost-USD"] = f"{usage_metrics.cost_usd:.6f}"

        # Wizard headers (session-level metrics)
        response.headers["X-Wizard-Content-Source"] = content_metadata.get("source", "unknown")

    # Step 7: Build response
    output = ServicesOutput(services=services)

    # Include debug info if enabled
    if settings.wizard.expose_debug_info:
        output.debug = {
            "model": service_generator.model_name,
            "usage": usage_metrics.model_dump(),
            "content_source": content_metadata.get("source"),
            "content_length": content_metadata.get("content_length"),
            "prompt_preview": debug_info.get("prompt_text", "")[:500],
            "response_preview": debug_info.get("response_text", "")[:500],
        }

    logger.info(
        "wizard_services_completed",
        extra={
            "session_id": session_id,
            "services_count": len(services),
        }
    )

    return output


@router.post(
    "/assets",
    response_model=AssetsOutput,
    summary="Generate IT assets",
    description="Identify IT assets that support business services"
)
async def generate_assets(
    request: AssetsInput,
    response: Response,
    session_id: str = Depends(get_session_id),
) -> AssetsOutput:
    """
    Generate IT and business assets from services and website content.

    Resolution priority for content:
    1. Services from request.services (direct)
    2. Services from session (previous step)
    3. Website content for additional context

    Args:
        request: AssetsInput with services, website content, and generation params
        response: FastAPI Response for injecting headers
        session_id: Session ID (22 random base64url chars)

    Returns:
        AssetsOutput with generated assets and optional debug info

    Raises:
        ContentNotFoundError: If no website content available
        AITimeoutError: If LLM request times out
        AIResponseError: If LLM response invalid
        ValidationRetriesExhausted: If validation fails after retries
    """
    settings = get_settings()

    logger.info(
        "wizard_assets_started",
        extra={
            "session_id": session_id,
            "model": request.ai_model,
            "language": request.language,
            "min_assets": request.min_assets,
            "max_assets": request.max_assets,
        }
    )

    # Initialize components
    redis = await get_redis()
    content_resolver = ContentResolver(redis)
    asset_generator = AssetGenerator(model=request.ai_model)

    # Step 1: Resolve services (from request or previous step)
    services = request.services
    if not services:
        services_step = await get_wizard_step(session_id, "services")
        if services_step and services_step.get("services"):
            from workshops_ai.wizard.schemas import Service
            services = [
                Service(**s) for s in services_step["services"]
            ]
            logger.info(
                "services_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(services),
                }
            )

    # Step 2: Resolve website content (optional additional context)
    # Check if web scraping is disabled (empty website_content object)
    scraping_disabled = (
        request.website_content is not None and
        not request.website_content.url and
        not request.website_content.content
    )

    website_content = ""
    content_metadata = {"source": "none"}

    if scraping_disabled:
        # Web scraping explicitly disabled - use empty content
        content_metadata = {"source": "disabled", "content_length": 0}
        logger.info(
            "web_scraping_disabled",
            extra={
                "session_id": session_id,
                "reason": "empty_website_content_object",
            }
        )
    else:
        # Normal flow - try to resolve content
        try:
            website_content, content_metadata = await content_resolver.resolve_content(
                session_id=session_id,
                website_text=request.website_content.content if request.website_content else None,
                website_url=request.website_url,
                search_method=request.search_method,
                ai_model=request.ai_model,
            )

            logger.info(
                "content_resolved",
                extra={
                    "session_id": session_id,
                    "source": content_metadata.get("source"),
                    "content_length": content_metadata.get("content_length"),
                }
            )

        except ContentNotFoundError:
            # Website content is optional for assets (services are primary input)
            logger.info(
                "no_website_content_available",
                extra={"session_id": session_id}
            )

    # Step 3: Build organization profile from request
    organization_profile = None
    if request.organization_profile:
        organization_profile = {
            "name": request.organization_profile.name,
            "industry": request.organization_profile.industry,
            "address": request.organization_profile.address,
            "employees": request.organization_profile.employees,
        }

    # Step 4: Load previous assets from session (if refining)
    previous_assets = request.previous_assets
    if not previous_assets and request.user_instructions:
        # User is refining - load previous assets
        previous_step = await get_wizard_step(session_id, "assets")
        if previous_step and previous_step.get("assets"):
            from workshops_ai.api.schema.asset import Asset
            previous_assets = [
                Asset(**a) for a in previous_step["assets"]
            ]
            logger.info(
                "previous_assets_loaded",
                extra={
                    "session_id": session_id,
                    "count": len(previous_assets),
                }
            )

    # Step 5: Generate assets with LLM
    try:
        assets, usage_metrics, debug_info = await asset_generator.generate_assets(
            website_content=website_content,
            language=request.language,
            min_assets=request.min_assets,
            max_assets=request.max_assets,
            services=services,
            organization_profile=organization_profile,
            user_instructions=request.user_instructions,
            previous_assets=previous_assets,
        )

        logger.info(
            "assets_generated",
            extra={
                "session_id": session_id,
                "assets_count": len(assets),
                "tokens": usage_metrics.total_tokens,
                "cost_usd": usage_metrics.cost_usd,
            }
        )

    except Exception as e:
        logger.error(
            "asset_generation_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise

    # Step 6: Save to session
    step_data = {
        "assets": [a.model_dump() for a in assets],
        "language": request.language,
        "model": asset_generator.model_name,
        "usage": usage_metrics.model_dump(),
    }

    await save_wizard_step(session_id, "assets", step_data)

    logger.info(
        "assets_saved_to_session",
        extra={"session_id": session_id}
    )

    # Step 7: Add response headers (if enabled)
    if settings.wizard.expose_debug_info:
        # GPT headers (per-request metrics)
        response.headers["X-GPT-Model"] = asset_generator.model_name
        response.headers["X-GPT-Tokens-Prompt"] = str(usage_metrics.prompt_tokens)
        response.headers["X-GPT-Tokens-Completion"] = str(usage_metrics.completion_tokens)
        response.headers["X-GPT-Tokens-Total"] = str(usage_metrics.total_tokens)
        response.headers["X-GPT-Cost-USD"] = f"{usage_metrics.cost_usd:.6f}"

        # Wizard headers (session-level metrics)
        # Session ID already set in response header by get_session_id dependency
        response.headers["X-Wizard-Content-Source"] = content_metadata.get("source", "unknown")

    # Step 8: Build response
    output = AssetsOutput(assets=assets)

    # Include debug info if enabled
    if settings.wizard.expose_debug_info:
        output.debug = {
            "model": asset_generator.model_name,
            "usage": usage_metrics.model_dump(),
            "content_source": content_metadata.get("source"),
            "content_length": content_metadata.get("content_length"),
            "services_count": len(services) if services else 0,
            "prompt_preview": debug_info.get("prompt_text", "")[:500],
            "response_preview": debug_info.get("response_text", "")[:500],
            "allowed_asset_types": debug_info.get("allowed_asset_types", []),
        }

    logger.info(
        "wizard_assets_completed",
        extra={
            "session_id": session_id,
            "assets_count": len(assets),
        }
    )

    return output


@router.post(
    "/impacts",
    response_model=ImpactsOutput,
    summary="Evaluate asset impacts",
    description="Evaluate impact severity levels for each asset based on official Impact catalog"
)
async def generate_impacts(
    request: ImpactsInput,
    response: Response,
    session_id: str = Depends(get_session_id),
) -> ImpactsOutput:
    """
    Generate impact evaluations for assets using catalog-based approach.

    Resolution priority for content:
    1. Assets from request.assets (direct)
    2. Assets from session (previous step)
    3. Services and organization profile for context

    Args:
        request: ImpactsInput with assets and generation params
        response: FastAPI Response for injecting headers
        session_id: Session ID (22 random base64url chars)

    Returns:
        ImpactsOutput with asset impact evaluations and optional debug info

    Raises:
        HTTPException: If no assets available
        AITimeoutError: If LLM request times out
        AIResponseError: If LLM response invalid
        ValidationRetriesExhausted: If validation fails after retries
    """
    settings = get_settings()

    logger.info(
        "wizard_impacts_started",
        extra={
            "session_id": session_id,
            "model": request.ai_model,
            "language": request.language,
        }
    )

    # Initialize components
    impact_generator = ImpactGenerator(model=request.ai_model)

    # Step 1: Resolve assets (from request or previous step)
    assets = request.assets
    if not assets:
        assets_step = await get_wizard_step(session_id, "assets")
        if assets_step and assets_step.get("assets"):
            from workshops_ai.api.schema.asset import Asset
            assets = [
                Asset(**a) for a in assets_step["assets"]
            ]
            logger.info(
                "assets_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(assets),
                }
            )

    # Validate we have assets
    if not assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No assets provided and none found in session"
        )

    # Step 2: Resolve services (optional, for context)
    services = request.services
    if not services:
        services_step = await get_wizard_step(session_id, "services")
        if services_step and services_step.get("services"):
            from workshops_ai.wizard.schemas import Service
            services = [
                Service(**s) for s in services_step["services"]
            ]
            logger.info(
                "services_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(services),
                }
            )

    # Step 2.5: Resolve website content (scraping or AI search)
    redis = await get_redis()
    content_resolver = ContentResolver(redis)

    # Check if web scraping is disabled (empty website_content object)
    scraping_disabled = (
        request.website_content is not None and
        not request.website_content.url and
        not request.website_content.content
    )

    if scraping_disabled:
        website_content = ""
        content_metadata = {"source": "disabled", "content_length": 0}
        logger.info(
            "web_scraping_disabled",
            extra={
                "session_id": session_id,
                "reason": "empty_website_content_object",
                "endpoint": "impacts"
            }
        )
    else:
        website_content, content_metadata = await content_resolver.resolve_content(
            session_id=session_id,
            website_text=request.website_content.content if request.website_content else None,
            website_url=request.website_url,
            search_method=request.search_method,
            ai_model=request.ai_model,
        )

    # Step 3: Build organization profile from request
    organization_profile = None
    if request.organization_profile:
        organization_profile = {
            "name": request.organization_profile.name,
            "industry": request.organization_profile.industry,
            "address": request.organization_profile.address,
            "employees": request.organization_profile.employees,
        }

    # Step 4: Generate impact evaluations with LLM
    try:
        asset_impacts, usage_metrics, debug_info = await impact_generator.generate_asset_impacts(
            assets=assets,
            language=request.language,
            min_impacts_per_asset=request.min_impacts_per_asset,
            max_impacts_per_asset=request.max_impacts_per_asset,
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            user_instructions=request.user_instructions,
        )

        logger.info(
            "impacts_generated",
            extra={
                "session_id": session_id,
                "assets_count": len(assets),
                "asset_impacts_count": len(asset_impacts),
                "tokens": usage_metrics.total_tokens,
                "cost_usd": usage_metrics.cost_usd,
            }
        )

    except Exception as e:
        logger.error(
            "impact_generation_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise

    # Step 5: Save to session
    step_data = {
        "asset_impacts": [ai.model_dump() for ai in asset_impacts],
        "language": request.language,
        "model": impact_generator.model_name,
        "usage": usage_metrics.model_dump(),
    }

    await save_wizard_step(session_id, "impacts", step_data)

    logger.info(
        "impacts_saved_to_session",
        extra={"session_id": session_id}
    )

    # Step 6: Add response headers (if enabled)
    if settings.wizard.expose_debug_info:
        # GPT headers (per-request metrics)
        response.headers["X-GPT-Model"] = impact_generator.model_name
        response.headers["X-GPT-Tokens-Prompt"] = str(usage_metrics.prompt_tokens)
        response.headers["X-GPT-Tokens-Completion"] = str(usage_metrics.completion_tokens)
        response.headers["X-GPT-Tokens-Total"] = str(usage_metrics.total_tokens)
        response.headers["X-GPT-Cost-USD"] = f"{usage_metrics.cost_usd:.6f}"

        # Wizard headers (session-level metrics)
        # Session ID already set in response header by get_session_id dependency

    # Step 7: Build response
    output = ImpactsOutput(asset_impacts=asset_impacts)

    # Include debug info if enabled
    if settings.wizard.expose_debug_info:
        output.debug = {
            "model": impact_generator.model_name,
            "usage": usage_metrics.model_dump(),
            "assets_processed": debug_info.get("assets_processed"),
            "catalog_impacts_count": debug_info.get("catalog_impacts_count"),
            "services_count": len(services) if services else 0,
            "last_prompt_preview": debug_info.get("last_prompt_preview", "")[:500],
            "last_response_preview": debug_info.get("last_response_preview", "")[:500],
        }

    logger.info(
        "wizard_impacts_completed",
        extra={
            "session_id": session_id,
            "asset_impacts_count": len(asset_impacts),
        }
    )

    return output


@router.post(
    "/risks",
    response_model=RisksOutput,
    summary="Evaluate asset risks",
    description="Evaluate risk probability levels for each asset based on official Threat catalog"
)
async def generate_risks(
    request: RisksInput,
    response: Response,
    session_id: str = Depends(get_session_id),
) -> RisksOutput:
    """
    Generate risk evaluations for assets using catalog-based approach.

    Resolution priority for content:
    1. Assets from request.assets (direct)
    2. Assets from session (previous step)
    3. Asset impacts, services, and organization profile for context

    Args:
        request: RisksInput with assets and generation params
        response: FastAPI Response for injecting headers
        session_id: Session ID (22 random base64url chars)

    Returns:
        RisksOutput with asset risk evaluations and optional debug info

    Raises:
        HTTPException: If no assets available
        AITimeoutError: If LLM request times out
        AIResponseError: If LLM response invalid
        ValidationRetriesExhausted: If validation fails after retries
    """
    settings = get_settings()

    logger.info(
        "wizard_risks_started",
        extra={
            "session_id": session_id,
            "model": request.ai_model,
            "language": request.language,
        }
    )

    # Initialize components
    risk_generator = RiskGenerator(model=request.ai_model)

    # Step 1: Resolve assets (from request or previous step)
    assets = request.assets
    if not assets:
        assets_step = await get_wizard_step(session_id, "assets")
        if assets_step and assets_step.get("assets"):
            from workshops_ai.api.schema.asset import Asset
            assets = [
                Asset(**a) for a in assets_step["assets"]
            ]
            logger.info(
                "assets_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(assets),
                }
            )

    # Validate we have assets
    if not assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No assets provided and none found in session"
        )

    # Step 2: Resolve asset impacts (optional, for context)
    asset_impacts = request.asset_impacts
    if not asset_impacts:
        impacts_step = await get_wizard_step(session_id, "impacts")
        if impacts_step and impacts_step.get("asset_impacts"):
            from workshops_ai.api.schema.impact import AssetImpact
            asset_impacts = [
                AssetImpact(**ai) for ai in impacts_step["asset_impacts"]
            ]
            logger.info(
                "asset_impacts_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(asset_impacts),
                }
            )

    # Step 3: Resolve services (optional, for context)
    services = request.services
    if not services:
        services_step = await get_wizard_step(session_id, "services")
        if services_step and services_step.get("services"):
            from workshops_ai.wizard.schemas import Service
            services = [
                Service(**s) for s in services_step["services"]
            ]
            logger.info(
                "services_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(services),
                }
            )

    # Step 3.5: Resolve website content (scraping or AI search)
    redis = await get_redis()
    content_resolver = ContentResolver(redis)

    # Check if web scraping is disabled (empty website_content object)
    scraping_disabled = (
        request.website_content is not None and
        not request.website_content.url and
        not request.website_content.content
    )

    if scraping_disabled:
        website_content = ""
        content_metadata = {"source": "disabled", "content_length": 0}
        logger.info(
            "web_scraping_disabled",
            extra={
                "session_id": session_id,
                "reason": "empty_website_content_object",
                "endpoint": "risks"
            }
        )
    else:
        website_content, content_metadata = await content_resolver.resolve_content(
            session_id=session_id,
            website_text=request.website_content.content if request.website_content else None,
            website_url=request.website_url,
            search_method=request.search_method,
            ai_model=request.ai_model,
        )

    # Step 4: Build organization profile from request
    organization_profile = None
    if request.organization_profile:
        organization_profile = {
            "name": request.organization_profile.name,
            "industry": request.organization_profile.industry,
            "address": request.organization_profile.address,
            "employees": request.organization_profile.employees,
        }

    # Step 5: Generate risk evaluations with LLM
    try:
        asset_risks, usage_metrics, debug_info = await risk_generator.generate_asset_risks(
            assets=assets,
            language=request.language,
            min_threats_per_asset=request.min_threats_per_asset,
            max_threats_per_asset=request.max_threats_per_asset,
            website_content=website_content,
            organization_profile=organization_profile,
            services=services,
            asset_impacts=asset_impacts,
            user_instructions=request.user_instructions,
        )

        logger.info(
            "risks_generated",
            extra={
                "session_id": session_id,
                "assets_count": len(assets),
                "asset_risks_count": len(asset_risks),
                "tokens": usage_metrics.total_tokens,
                "cost_usd": usage_metrics.cost_usd,
            }
        )

    except Exception as e:
        logger.error(
            "risk_generation_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise

    # Step 6: Save to session
    step_data = {
        "asset_risks": [ar.model_dump() for ar in asset_risks],
        "language": request.language,
        "model": risk_generator.model_name,
        "usage": usage_metrics.model_dump(),
    }

    await save_wizard_step(session_id, "risks", step_data)

    logger.info(
        "risks_saved_to_session",
        extra={"session_id": session_id}
    )

    # Step 7: Add response headers (if enabled)
    if settings.wizard.expose_debug_info:
        # GPT headers (per-request metrics)
        response.headers["X-GPT-Model"] = risk_generator.model_name
        response.headers["X-GPT-Tokens-Prompt"] = str(usage_metrics.prompt_tokens)
        response.headers["X-GPT-Tokens-Completion"] = str(usage_metrics.completion_tokens)
        response.headers["X-GPT-Tokens-Total"] = str(usage_metrics.total_tokens)
        response.headers["X-GPT-Cost-USD"] = f"{usage_metrics.cost_usd:.6f}"

        # Wizard headers (session-level metrics)
        # Session ID already set in response header by get_session_id dependency

    # Step 8: Build response
    output = RisksOutput(asset_risks=asset_risks)

    # Include debug info if enabled
    if settings.wizard.expose_debug_info:
        output.debug = {
            "model": risk_generator.model_name,
            "usage": usage_metrics.model_dump(),
            "assets_processed": debug_info.get("assets_processed"),
            "catalog_threats_total": debug_info.get("catalog_threats_total"),
            "threats_per_asset": debug_info.get("threats_per_asset"),
            "priors_available": debug_info.get("priors_available"),
            "priors_count": debug_info.get("priors_count"),
            "services_count": len(services) if services else 0,
            "asset_impacts_count": len(asset_impacts) if asset_impacts else 0,
            "last_prompt_preview": debug_info.get("last_prompt_preview", "")[:500],
            "last_response_preview": debug_info.get("last_response_preview", "")[:500],
        }

    logger.info(
        "wizard_risks_completed",
        extra={
            "session_id": session_id,
            "asset_risks_count": len(asset_risks),
        }
    )

    return output


@router.post(
    "/controls",
    response_model=ControlsOutput,
    summary="Recommend security controls",
    description="Recommend security controls to mitigate identified risks using hybrid catalog + AI approach"
)
async def generate_controls(
    request: ControlsInput,
    response: Response,
    session_id: str = Depends(get_session_id),
) -> ControlsOutput:
    """
    Generate control recommendations for assets using hybrid catalog + AI approach.

    Resolution priority for content:
    1. Assets from request.assets (direct)
    2. Assets from session (previous step)
    3. Asset risks from request.asset_risks (REQUIRED - direct)
    4. Asset risks from session (previous step)

    Args:
        request: ControlsInput with assets, asset risks, and generation params
        response: FastAPI Response for injecting headers
        session_id: Session ID (22 random base64url chars)

    Returns:
        ControlsOutput with asset control recommendations and optional debug info

    Raises:
        HTTPException: If no assets or asset risks available
        AITimeoutError: If LLM request times out
        AIResponseError: If LLM response invalid
        ValidationRetriesExhausted: If validation fails after retries
    """
    settings = get_settings()

    logger.info(
        "wizard_controls_started",
        extra={
            "session_id": session_id,
            "model": request.ai_model,
            "language": request.language,
            "min_controls_per_asset": request.min_controls_per_asset,
            "max_controls_per_asset": request.max_controls_per_asset,
        }
    )

    # Initialize components
    control_generator = ControlGenerator(model=request.ai_model)

    # Step 1: Resolve assets (from request or previous step)
    assets = request.assets
    if not assets:
        assets_step = await get_wizard_step(session_id, "assets")
        if assets_step and assets_step.get("assets"):
            from workshops_ai.api.schema.asset import Asset
            assets = [
                Asset(**a) for a in assets_step["assets"]
            ]
            logger.info(
                "assets_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(assets),
                }
            )

    # Validate we have assets
    if not assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No assets provided and none found in session"
        )

    # Step 2: Resolve asset risks (REQUIRED - from request or previous step)
    asset_risks = request.asset_risks
    if not asset_risks:
        risks_step = await get_wizard_step(session_id, "risks")
        if risks_step and risks_step.get("asset_risks"):
            from workshops_ai.api.schema.threat import AssetRisk
            asset_risks = [
                AssetRisk(**ar) for ar in risks_step["asset_risks"]
            ]
            logger.info(
                "asset_risks_loaded_from_session",
                extra={
                    "session_id": session_id,
                    "count": len(asset_risks),
                }
            )

    # Validate we have asset risks
    if not asset_risks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No asset risks provided and none found in session. Run /wizard/risks first."
        )

    # Step 3: Build organization profile from request (optional)
    organization_profile = None
    if request.organization_profile:
        organization_profile = {
            "name": request.organization_profile.name,
            "industry": request.organization_profile.industry,
            "address": request.organization_profile.address,
            "employees": request.organization_profile.employees,
        }

    # Step 3.5: Resolve website content (scraping or AI search)
    redis = await get_redis()
    content_resolver = ContentResolver(redis)

    # Check if web scraping is disabled (empty website_content object)
    scraping_disabled = (
        request.website_content is not None and
        not request.website_content.url and
        not request.website_content.content
    )

    if scraping_disabled:
        website_content = ""
        content_metadata = {"source": "disabled", "content_length": 0}
        logger.info(
            "web_scraping_disabled",
            extra={
                "session_id": session_id,
                "reason": "empty_website_content_object",
                "endpoint": "controls"
            }
        )
    else:
        website_content, content_metadata = await content_resolver.resolve_content(
            session_id=session_id,
            website_text=request.website_content.content if request.website_content else None,
            website_url=request.website_url,
            search_method=request.search_method,
            ai_model=request.ai_model,
        )

    # Step 4: Generate control recommendations with LLM
    try:
        asset_controls, usage_metrics, debug_info = await control_generator.generate_asset_controls(
            assets=assets,
            asset_risks=asset_risks,
            language=request.language,
            min_controls_per_asset=request.min_controls_per_asset,
            max_controls_per_asset=request.max_controls_per_asset,
            website_content=website_content,
            user_instructions=request.user_instructions,
        )

        logger.info(
            "controls_generated",
            extra={
                "session_id": session_id,
                "assets_count": len(assets),
                "asset_controls_count": len(asset_controls),
                "tokens": usage_metrics.total_tokens,
                "cost_usd": usage_metrics.cost_usd,
            }
        )

    except Exception as e:
        logger.error(
            "control_generation_failed",
            extra={
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
        )
        raise

    # Step 5: Save to session
    step_data = {
        "asset_controls": [ac.model_dump() for ac in asset_controls],
        "language": request.language,
        "model": control_generator.model_name,
        "usage": usage_metrics.model_dump(),
    }

    await save_wizard_step(session_id, "controls", step_data)

    logger.info(
        "controls_saved_to_session",
        extra={"session_id": session_id}
    )

    # Step 6: Add response headers (if enabled)
    if settings.wizard.expose_debug_info:
        # GPT headers (per-request metrics)
        response.headers["X-GPT-Model"] = control_generator.model_name
        response.headers["X-GPT-Tokens-Prompt"] = str(usage_metrics.prompt_tokens)
        response.headers["X-GPT-Tokens-Completion"] = str(usage_metrics.completion_tokens)
        response.headers["X-GPT-Tokens-Total"] = str(usage_metrics.total_tokens)
        response.headers["X-GPT-Cost-USD"] = f"{usage_metrics.cost_usd:.6f}"

        # Wizard headers (session-level metrics)
        # Session ID already set in response header by get_session_id dependency

    # Step 7: Build response
    output = ControlsOutput(asset_controls=asset_controls)

    # Include debug info if enabled
    if settings.wizard.expose_debug_info:
        output.debug = {
            "model": control_generator.model_name,
            "usage": usage_metrics.model_dump(),
            "assets_processed": debug_info.get("assets_processed"),
            "assets_skipped": debug_info.get("assets_skipped"),
            "catalog_controls_total": debug_info.get("catalog_controls_total"),
            "min_controls_per_asset": debug_info.get("min_controls_per_asset"),
            "max_controls_per_asset": debug_info.get("max_controls_per_asset"),
            "controls_constraint": debug_info.get("controls_constraint"),
            "last_prompt_preview": debug_info.get("last_prompt_preview", "")[:500],
            "last_response_preview": debug_info.get("last_response_preview", "")[:500],
        }

    logger.info(
        "wizard_controls_completed",
        extra={
            "session_id": session_id,
            "asset_controls_count": len(asset_controls),
        }
    )

    return output


@router.post(
    "/evaluation",
    response_model=EvaluationOutput,
    summary="Full wizard evaluation",
    description="Run complete wizard flow: services → assets → impacts → risks → controls (returns partial results if any step fails)"
)
async def generate_evaluation(
    request: EvaluationInput,
    response: Response,
    session_id: str = Depends(get_session_id),
) -> EvaluationOutput:
    """
    Run complete wizard evaluation: services → assets → impacts → risks → controls.

    This meta-endpoint orchestrates all 5 wizard steps in a single API call.
    Returns partial results if any step fails and continue_on_error=True.

    Args:
        request: EvaluationInput with website content and generation params
        response: FastAPI Response for injecting headers
        session_id: Session ID (22 random base64url chars)

    Returns:
        EvaluationOutput with all step results, timings, and optional errors

    Raises:
        HTTPException: If no website content provided or step fails with continue_on_error=False
    """
    settings = get_settings()
    start_total = time.time()

    logger.info(
        "wizard_evaluation_started",
        extra={
            "session_id": session_id,
            "model": request.ai_model,
            "language": request.language,
            "skip_existing": request.skip_existing,
            "continue_on_error": request.continue_on_error,
        }
    )

    # Track results and timings
    results = EvaluationData()
    timings = []
    errors = []
    debug_prompts = {}

    # Initialize components
    redis = await get_redis()
    content_resolver = ContentResolver(redis)

    # Build organization profile once
    organization_profile = None
    if request.organization_profile:
        organization_profile = {
            "name": request.organization_profile.name,
            "industry": request.organization_profile.industry,
            "address": request.organization_profile.address,
            "employees": request.organization_profile.employees,
        }

    # ==========================================================================
    # Step 1: Services
    # ==========================================================================
    try:
        start = time.time()

        # Check skip_existing
        if request.skip_existing:
            existing = await get_wizard_step(session_id, "services")
            if existing and existing.get("services"):
                from workshops_ai.wizard.schemas import Service
                results.services = [Service(**s) for s in existing["services"]]
                timings.append(StepTiming(
                    step_name="services",
                    processing_time_ms=0,
                    success=True,
                    error="Skipped - using existing data"
                ))
                logger.info(
                    "evaluate_step_skipped",
                    extra={"session_id": session_id, "step": "services"}
                )

        if results.services is None:
            # Check if web scraping is disabled (empty website_content object)
            scraping_disabled = (
                request.website_content is not None and
                not request.website_content.url and
                not request.website_content.content
            )

            # Resolve website content
            if scraping_disabled:
                # Web scraping explicitly disabled
                website_content = ""
                content_metadata = {"source": "disabled", "content_length": 0}
                logger.info(
                    "web_scraping_disabled",
                    extra={
                        "session_id": session_id,
                        "reason": "empty_website_content_object",
                        "step": "services",
                    }
                )
            else:
                website_content, content_metadata = await content_resolver.resolve_content(
                    session_id=session_id,
                    website_text=request.website_content.content if request.website_content else None,
                    website_url=request.website_url,
                    search_method=request.search_method,
                    ai_model=request.ai_model,
                )

            if not website_content and not scraping_disabled:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No website content or URL provided"
                )

            # Generate services
            service_generator = ServiceGenerator(model=request.ai_model)
            services, usage_metrics, debug_info = await service_generator.generate_services(
                website_content=website_content,
                language=request.language,
                min_services=request.min_services,
                max_services=request.max_services,
                organization_profile=organization_profile,
                user_instructions=None,  # Not exposed in evaluation mode
                previous_services=None,
                session_id=session_id,
            )

            results.services = services
            await save_wizard_step(session_id, "services", {
                "services": [s.model_dump() for s in services],
                "language": request.language,
                "model": service_generator.model_name,
                "usage": usage_metrics.model_dump(),
            })

            debug_prompts["services"] = {
                "prompt_preview": debug_info.get("prompt_text", "")[:500],
                "response_preview": debug_info.get("response_text", "")[:500],
            }

            elapsed = int((time.time() - start) * 1000)
            timings.append(StepTiming(
                step_name="services",
                processing_time_ms=elapsed,
                success=True
            ))
            logger.info(
                "evaluate_step_completed",
                extra={"session_id": session_id, "step": "services", "elapsed_ms": elapsed}
            )

    except Exception as e:
        elapsed = int((time.time() - start) * 1000)
        error_msg = f"Services generation failed: {str(e)}"
        errors.append(error_msg)
        timings.append(StepTiming(
            step_name="services",
            processing_time_ms=elapsed,
            success=False,
            error=str(e)
        ))
        logger.error(
            "evaluate_step_failed",
            extra={"session_id": session_id, "step": "services", "error": str(e)}
        )

        if not request.continue_on_error:
            raise HTTPException(status_code=500, detail=error_msg)

    # ==========================================================================
    # Step 2: Assets
    # ==========================================================================
    if results.services or request.continue_on_error:
        try:
            start = time.time()

            # Check skip_existing
            if request.skip_existing:
                existing = await get_wizard_step(session_id, "assets")
                if existing and existing.get("assets"):
                    from workshops_ai.api.schema.asset import Asset
                    results.assets = [Asset(**a) for a in existing["assets"]]
                    timings.append(StepTiming(
                        step_name="assets",
                        processing_time_ms=0,
                        success=True,
                        error="Skipped - using existing data"
                    ))
                    logger.info(
                        "evaluate_step_skipped",
                        extra={"session_id": session_id, "step": "assets"}
                    )

            if results.assets is None and results.services:
                # Check if web scraping is disabled (empty website_content object)
                scraping_disabled = (
                    request.website_content is not None and
                    not request.website_content.url and
                    not request.website_content.content
                )

                # Get website content from session (or re-scrape if cache fails)
                if scraping_disabled:
                    website_content = ""
                    logger.info(
                        "web_scraping_disabled",
                        extra={
                            "session_id": session_id,
                            "reason": "empty_website_content_object",
                            "step": "assets",
                        }
                    )
                else:
                    website_content, _ = await content_resolver.resolve_content(
                        session_id=session_id,
                        website_text=request.website_content.content if request.website_content else None,
                        website_url=request.website_url,
                        search_method=request.search_method,
                        ai_model=request.ai_model,
                    )

                # Generate assets
                asset_generator = AssetGenerator(model=request.ai_model)
                assets, usage_metrics, debug_info = await asset_generator.generate_assets(
                    website_content=website_content or "",
                    language=request.language,
                    min_assets=request.min_assets,
                    max_assets=request.max_assets,
                    services=results.services,
                    organization_profile=organization_profile,
                    user_instructions=None,
                    previous_assets=None,
                )

                results.assets = assets
                await save_wizard_step(session_id, "assets", {
                    "assets": [a.model_dump() for a in assets],
                    "language": request.language,
                    "model": asset_generator.model_name,
                    "usage": usage_metrics.model_dump(),
                })

                debug_prompts["assets"] = {
                    "prompt_preview": debug_info.get("prompt_text", "")[:500],
                    "response_preview": debug_info.get("response_text", "")[:500],
                }

                elapsed = int((time.time() - start) * 1000)
                timings.append(StepTiming(
                    step_name="assets",
                    processing_time_ms=elapsed,
                    success=True
                ))
                logger.info(
                    "evaluate_step_completed",
                    extra={"session_id": session_id, "step": "assets", "elapsed_ms": elapsed}
                )

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            error_msg = f"Assets generation failed: {str(e)}"
            errors.append(error_msg)
            timings.append(StepTiming(
                step_name="assets",
                processing_time_ms=elapsed,
                success=False,
                error=str(e)
            ))
            logger.error(
                "evaluate_step_failed",
                extra={"session_id": session_id, "step": "assets", "error": str(e)}
            )

            if not request.continue_on_error:
                raise HTTPException(status_code=500, detail=error_msg)

    # ==========================================================================
    # Step 3: Impacts
    # ==========================================================================
    if results.assets or request.continue_on_error:
        try:
            start = time.time()

            # Check skip_existing
            if request.skip_existing:
                existing = await get_wizard_step(session_id, "impacts")
                if existing and existing.get("asset_impacts"):
                    from workshops_ai.api.schema.impact import AssetImpact
                    results.asset_impacts = [AssetImpact(**ai) for ai in existing["asset_impacts"]]
                    timings.append(StepTiming(
                        step_name="impacts",
                        processing_time_ms=0,
                        success=True,
                        error="Skipped - using existing data"
                    ))
                    logger.info(
                        "evaluate_step_skipped",
                        extra={"session_id": session_id, "step": "impacts"}
                    )

            if results.asset_impacts is None and results.assets:
                # Generate impacts
                impact_generator = ImpactGenerator(model=request.ai_model)
                asset_impacts, usage_metrics, debug_info = await impact_generator.generate_asset_impacts(
                    assets=results.assets,
                    language=request.language,
                    min_impacts_per_asset=request.min_impacts_per_asset,
                    max_impacts_per_asset=request.max_impacts_per_asset,
                    organization_profile=organization_profile,
                    services=results.services,
                    user_instructions=None,
                )

                results.asset_impacts = asset_impacts
                await save_wizard_step(session_id, "impacts", {
                    "asset_impacts": [ai.model_dump() for ai in asset_impacts],
                    "language": request.language,
                    "model": impact_generator.model_name,
                    "usage": usage_metrics.model_dump(),
                })

                debug_prompts["impacts"] = {
                    "last_prompt_preview": debug_info.get("last_prompt_preview", "")[:500],
                    "last_response_preview": debug_info.get("last_response_preview", "")[:500],
                }

                elapsed = int((time.time() - start) * 1000)
                timings.append(StepTiming(
                    step_name="impacts",
                    processing_time_ms=elapsed,
                    success=True
                ))
                logger.info(
                    "evaluate_step_completed",
                    extra={"session_id": session_id, "step": "impacts", "elapsed_ms": elapsed}
                )

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            error_msg = f"Impacts generation failed: {str(e)}"
            errors.append(error_msg)
            timings.append(StepTiming(
                step_name="impacts",
                processing_time_ms=elapsed,
                success=False,
                error=str(e)
            ))
            logger.error(
                "evaluate_step_failed",
                extra={"session_id": session_id, "step": "impacts", "error": str(e)}
            )

            if not request.continue_on_error:
                raise HTTPException(status_code=500, detail=error_msg)

    # ==========================================================================
    # Step 4: Risks
    # ==========================================================================
    if results.assets or request.continue_on_error:
        try:
            start = time.time()

            # Check skip_existing
            if request.skip_existing:
                existing = await get_wizard_step(session_id, "risks")
                if existing and existing.get("asset_risks"):
                    from workshops_ai.api.schema.threat import AssetRisk
                    results.asset_risks = [AssetRisk(**ar) for ar in existing["asset_risks"]]
                    timings.append(StepTiming(
                        step_name="risks",
                        processing_time_ms=0,
                        success=True,
                        error="Skipped - using existing data"
                    ))
                    logger.info(
                        "evaluate_step_skipped",
                        extra={"session_id": session_id, "step": "risks"}
                    )

            if results.asset_risks is None and results.assets:
                # Generate risks
                risk_generator = RiskGenerator(model=request.ai_model)
                asset_risks, usage_metrics, debug_info = await risk_generator.generate_asset_risks(
                    assets=results.assets,
                    language=request.language,
                    min_threats_per_asset=request.min_threats_per_asset,
                    max_threats_per_asset=request.max_threats_per_asset,
                    organization_profile=organization_profile,
                    services=results.services,
                    asset_impacts=results.asset_impacts,
                    user_instructions=None,
                )

                results.asset_risks = asset_risks
                await save_wizard_step(session_id, "risks", {
                    "asset_risks": [ar.model_dump() for ar in asset_risks],
                    "language": request.language,
                    "model": risk_generator.model_name,
                    "usage": usage_metrics.model_dump(),
                })

                debug_prompts["risks"] = {
                    "last_prompt_preview": debug_info.get("last_prompt_preview", "")[:500],
                    "last_response_preview": debug_info.get("last_response_preview", "")[:500],
                }

                elapsed = int((time.time() - start) * 1000)
                timings.append(StepTiming(
                    step_name="risks",
                    processing_time_ms=elapsed,
                    success=True
                ))
                logger.info(
                    "evaluate_step_completed",
                    extra={"session_id": session_id, "step": "risks", "elapsed_ms": elapsed}
                )

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            error_msg = f"Risks generation failed: {str(e)}"
            errors.append(error_msg)
            timings.append(StepTiming(
                step_name="risks",
                processing_time_ms=elapsed,
                success=False,
                error=str(e)
            ))
            logger.error(
                "evaluate_step_failed",
                extra={"session_id": session_id, "step": "risks", "error": str(e)}
            )

            if not request.continue_on_error:
                raise HTTPException(status_code=500, detail=error_msg)

    # ==========================================================================
    # Step 5: Controls
    # ==========================================================================
    if (results.assets and results.asset_risks) or request.continue_on_error:
        try:
            start = time.time()

            # Check skip_existing
            if request.skip_existing:
                existing = await get_wizard_step(session_id, "controls")
                if existing and existing.get("asset_controls"):
                    from workshops_ai.api.schema.control import AssetControls
                    results.asset_controls = [AssetControls(**ac) for ac in existing["asset_controls"]]
                    timings.append(StepTiming(
                        step_name="controls",
                        processing_time_ms=0,
                        success=True,
                        error="Skipped - using existing data"
                    ))
                    logger.info(
                        "evaluate_step_skipped",
                        extra={"session_id": session_id, "step": "controls"}
                    )

            if results.asset_controls is None and results.assets and results.asset_risks:
                # Generate controls
                control_generator = ControlGenerator(model=request.ai_model)
                asset_controls, usage_metrics, debug_info = await control_generator.generate_asset_controls(
                    assets=results.assets,
                    asset_risks=results.asset_risks,
                    language=request.language,
                    min_controls_per_asset=request.min_controls_per_asset,
                    max_controls_per_asset=request.max_controls_per_asset,
                    user_instructions=None,
                )

                results.asset_controls = asset_controls
                await save_wizard_step(session_id, "controls", {
                    "asset_controls": [ac.model_dump() for ac in asset_controls],
                    "language": request.language,
                    "model": control_generator.model_name,
                    "usage": usage_metrics.model_dump(),
                })

                debug_prompts["controls"] = {
                    "last_prompt_preview": debug_info.get("last_prompt_preview", "")[:500],
                    "last_response_preview": debug_info.get("last_response_preview", "")[:500],
                }

                elapsed = int((time.time() - start) * 1000)
                timings.append(StepTiming(
                    step_name="controls",
                    processing_time_ms=elapsed,
                    success=True
                ))
                logger.info(
                    "evaluate_step_completed",
                    extra={"session_id": session_id, "step": "controls", "elapsed_ms": elapsed}
                )

        except Exception as e:
            elapsed = int((time.time() - start) * 1000)
            error_msg = f"Controls generation failed: {str(e)}"
            errors.append(error_msg)
            timings.append(StepTiming(
                step_name="controls",
                processing_time_ms=elapsed,
                success=False,
                error=str(e)
            ))
            logger.error(
                "evaluate_step_failed",
                extra={"session_id": session_id, "step": "controls", "error": str(e)}
            )

            if not request.continue_on_error:
                raise HTTPException(status_code=500, detail=error_msg)

    # ==========================================================================
    # Build final response
    # ==========================================================================
    total_elapsed = int((time.time() - start_total) * 1000)

    # Build output
    output = EvaluationOutput(
        total_processing_time_ms=total_elapsed,
        data=results,
        step_timings=timings,
        errors=errors if errors else None,
        debug=debug_prompts if settings.wizard.expose_debug_info and debug_prompts else None
    )

    # Add response headers if enabled
    if settings.wizard.expose_debug_info:
        # Session ID already set in response header by get_session_id dependency
        response.headers["X-Wizard-Total-Time-MS"] = str(total_elapsed)
        response.headers["X-Wizard-Steps-Completed"] = str(len([t for t in timings if t.success]))
        response.headers["X-Wizard-Steps-Failed"] = str(len([t for t in timings if not t.success]))

    logger.info(
        "wizard_evaluation_completed",
        extra={
            "session_id": session_id,
            "total_time_ms": total_elapsed,
            "steps_completed": len([t for t in timings if t.success]),
            "steps_failed": len([t for t in timings if not t.success]),
        }
    )

    return output


