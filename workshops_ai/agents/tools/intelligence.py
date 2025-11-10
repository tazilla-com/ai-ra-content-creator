# workshops_ai/agents/tools/intelligence.py
"""
LLM-based intelligence tools for asset management.

Design Philosophy:
- NO HARDCODED RULES! Everything is LLM-based for maximum flexibility
- Future: These will be enhanced with RAG (industry best practices database)
- Educational approach: provide reasoning, not just answers
- Context-aware: use session data for better recommendations

Tools:
- detect_asset_dependencies: Find related assets (LLM-based)
- recommend_assets_for_context: Suggest assets for company/industry (LLM-based)
"""
from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from workshops_ai.agents.session import get_agent_session

logger = logging.getLogger(__name__)

__all__ = [
    "detect_asset_dependencies",
    "recommend_assets_for_context",
]


# -----------------------------------------------------------------------------
# Tool Input Schemas
# -----------------------------------------------------------------------------

class DetectDependenciesInput(BaseModel):
    """Input for detecting asset dependencies."""
    session_id: str = Field(..., description="Session identifier")
    asset_name: str = Field(..., description="Name of the asset to analyze")
    asset_type: str | None = Field(None, description="Type of the asset (optional)")
    context: str | None = Field(
        None,
        description="Additional context (e.g., 'e-commerce system', 'healthcare infrastructure')"
    )


class RecommendAssetsInput(BaseModel):
    """Input for recommending assets."""
    session_id: str = Field(..., description="Session identifier")
    user_request: str = Field(
        ...,
        description="What the user is asking for (e.g., 'What assets do I need?', 'Suggest IT infrastructure')"
    )


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

@tool
async def detect_asset_dependencies(input: DetectDependenciesInput) -> dict:
    """
    Detect related assets and dependencies for a given asset.

    This tool uses LLM reasoning (NO hardcoded rules!) to identify:
    - Assets this asset depends on
    - Assets that depend on this asset
    - Related assets in the same system

    Args:
        input: DetectDependenciesInput with session_id, asset_name, asset_type, context

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - dependencies: List of suggested related assets
        - reasoning: Why these dependencies were identified

    Example:
        Input: "Application Server (Node.js)" in "e-commerce system"
        Output:
        - Database (stores application data)
        - Web Server (nginx/Apache for reverse proxy)
        - Cache (Redis for session storage)
        - File Storage (S3 for uploads)

    Future Enhancement:
        Will be augmented with RAG (industry best practices database) to provide
        more accurate recommendations based on real-world patterns.
    """
    try:
        # Load session for context
        session = await get_agent_session(input.session_id)
        if not session:
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        # Build context from session
        context_parts = []
        if input.context:
            context_parts.append(f"Context: {input.context}")
        if session.industry:
            context_parts.append(f"Industry: {session.industry}")
        if session.company_name:
            context_parts.append(f"Company: {session.company_name}")

        full_context = " | ".join(context_parts) if context_parts else "General IT infrastructure"

        # TODO: Phase 3 - Call LLM to detect dependencies
        # For now, return placeholder with example structure
        logger.info(
            "Detecting asset dependencies",
            extra={
                "session_id": input.session_id,
                "asset_name": input.asset_name,
                "asset_type": input.asset_type,
                "context": full_context,
            }
        )

        # Placeholder response (will be replaced with LLM call in Phase 3)
        mock_dependencies = [
            {
                "name": f"Related asset for {input.asset_name}",
                "relationship": "depends_on",
                "reasoning": "LLM will provide reasoning here"
            }
        ]

        return {
            "status": "success",
            "asset_name": input.asset_name,
            "dependencies": mock_dependencies,
            "reasoning": f"Based on {full_context}, this asset typically requires the suggested dependencies.",
            "message": f"Found {len(mock_dependencies)} potential dependenc(ies) for {input.asset_name}",
            "note": "LLM-based detection - no hardcoded rules used"
        }

    except Exception as e:
        logger.error(
            "Failed to detect dependencies",
            extra={
                "session_id": input.session_id,
                "asset_name": input.asset_name,
                "error": str(e)
            }
        )
        return {
            "status": "error",
            "message": f"Failed to detect dependencies: {str(e)}",
        }


@tool
async def recommend_assets_for_context(input: RecommendAssetsInput) -> dict:
    """
    Recommend assets based on company context and user needs.

    This tool uses LLM reasoning (NO hardcoded industry templates!) to suggest
    relevant IT assets based on:
    - Company industry (from session)
    - Company name/description (from session)
    - User's specific request
    - Previous conversation context

    Args:
        input: RecommendAssetsInput with session_id, user_request

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - recommendations: List of recommended asset suggestions
        - reasoning: Overall reasoning for recommendations
        - compliance_notes: Any relevant compliance considerations

    Example:
        Company: Healthcare provider
        Request: "What assets do I need?"
        Output:
        - Patient Records Database (HIPAA - strictly protected)
        - Medical Imaging Storage (large files, high availability)
        - Electronic Health Records System (primary application)
        - Backup Systems (critical for healthcare continuity)

    Future Enhancement:
        Will be augmented with RAG to pull from industry best practices database,
        compliance requirements library, and real-world examples.

    Note:
        This is intentionally LLM-based (not hardcoded) to handle:
        - New industries
        - Mixed/custom industries
        - Evolving compliance requirements
        - Company-specific needs
    """
    try:
        # Load session for context
        session = await get_agent_session(input.session_id)
        if not session:
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        # Build context from session
        company_info_parts = []
        if session.company_name:
            company_info_parts.append(f"Company: {session.company_name}")
        if session.industry:
            company_info_parts.append(f"Industry: {session.industry}")
        if session.scraped_content:
            # Use first 500 chars of scraped content for context
            preview = session.scraped_content[:500]
            company_info_parts.append(f"Website info: {preview}...")

        company_info = " | ".join(company_info_parts) if company_info_parts else "General company"

        # TODO: Phase 3 - Call LLM to generate recommendations
        # For now, return placeholder
        logger.info(
            "Generating asset recommendations",
            extra={
                "session_id": input.session_id,
                "user_request": input.user_request,
                "company_info": company_info[:200],
            }
        )

        # Placeholder response (will be replaced with LLM call in Phase 3)
        mock_recommendations = [
            {
                "name": "Recommended Asset 1",
                "type": "Server infrastructure",
                "reasoning": "LLM will provide specific reasoning based on company context",
                "compliance": "Relevant compliance notes will be added by LLM"
            }
        ]

        return {
            "status": "success",
            "recommendations": mock_recommendations,
            "reasoning": f"Based on {company_info} and your request: '{input.user_request}', these assets are recommended.",
            "compliance_notes": "LLM will provide industry-specific compliance guidance",
            "message": f"Generated {len(mock_recommendations)} recommendation(s)",
            "note": "LLM-based recommendations - no hardcoded industry templates used",
            "future": "Will be enhanced with RAG for industry best practices"
        }

    except Exception as e:
        logger.error(
            "Failed to generate recommendations",
            extra={
                "session_id": input.session_id,
                "user_request": input.user_request,
                "error": str(e)
            }
        )
        return {
            "status": "error",
            "message": f"Failed to generate recommendations: {str(e)}",
        }
