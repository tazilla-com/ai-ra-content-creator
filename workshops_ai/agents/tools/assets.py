# workshops_ai/agents/tools/assets.py
"""
Asset management tools for LangGraph agents.

Design:
- Tools work with session state (drafts stored in Redis)
- Multi-asset batch support (create multiple assets at once)
- Visual diff preview (before/after comparison)
- Educational approach (reasoning, dependencies, compliance)
- Confirmation required before writing to Tazilla
"""
from __future__ import annotations

import logging
from typing import Annotated

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from workshops_ai.agents.session import (
    AgentSessionState,
    AssetDraft,
    get_agent_session,
    save_agent_session,
)

logger = logging.getLogger(__name__)

__all__ = [
    "asset_create_batch",
    "asset_update_with_diff",
    "confirm_and_submit",
    "show_draft_preview",
    "clear_drafts",
]


# -----------------------------------------------------------------------------
# Tool Input Schemas
# -----------------------------------------------------------------------------

class AssetCreateBatchInput(BaseModel):
    """Input for creating multiple assets at once."""
    session_id: str = Field(..., description="Session identifier")
    assets: list[AssetDraft] = Field(
        ...,
        min_length=1,
        description="List of asset drafts to create (1 or more)"
    )
    reasoning: str | None = Field(
        None,
        description="Overall reasoning for creating these assets together"
    )


class AssetUpdateInput(BaseModel):
    """Input for updating an existing asset."""
    session_id: str = Field(..., description="Session identifier")
    asset_id: str = Field(..., description="Asset ID to update")
    changes: dict = Field(..., description="Fields to change (e.g., {'confidentiality': 'PROTECTED'})")
    reasoning: str | None = Field(None, description="Why this change?")


class ConfirmSubmitInput(BaseModel):
    """Input for confirming and submitting drafts to Tazilla."""
    session_id: str = Field(..., description="Session identifier")


class PreviewInput(BaseModel):
    """Input for showing draft preview."""
    session_id: str = Field(..., description="Session identifier")


class ClearDraftsInput(BaseModel):
    """Input for clearing all drafts."""
    session_id: str = Field(..., description="Session identifier")


# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

@tool
async def asset_create_batch(input: AssetCreateBatchInput) -> dict:
    """
    Create one or multiple asset drafts.

    This tool DOES NOT write to Tazilla immediately!
    - Assets are stored as drafts in the session
    - User must confirm via confirm_and_submit() tool
    - Use this for batch creation (multiple assets at once)

    Args:
        input: AssetCreateBatchInput with session_id, assets, reasoning

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - drafts_count: Number of drafts created
        - message: Human-readable message
        - preview: Summary of created drafts

    Example:
        Create 3 assets for an e-commerce system:
        - Web Server (nginx)
        - Application Server (Node.js)
        - Database (PostgreSQL)
    """
    try:
        logger.info(
            f"asset_create_batch called with {len(input.assets)} assets",
            extra={"session_id": input.session_id, "asset_count": len(input.assets)}
        )

        # Load session
        session = await get_agent_session(input.session_id)
        if not session:
            logger.error(f"Session not found: {input.session_id}")
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        logger.info(
            f"Session loaded with {len(session.drafts)} existing drafts",
            extra={"session_id": input.session_id, "existing_drafts": len(session.drafts)}
        )

        # Add drafts to session
        for asset in input.assets:
            # Apply overall reasoning if no individual reasoning
            if not asset.reasoning and input.reasoning:
                asset.reasoning = input.reasoning

            # Mark as draft
            asset.status = "draft"

            # Add to session
            session.drafts.append(asset)

        # Save session
        await save_agent_session(session)

        # Build preview
        preview = _format_drafts_preview(session.drafts)

        logger.info(
            "Asset drafts created and saved",
            extra={
                "session_id": input.session_id,
                "count": len(input.assets),
                "total_drafts": len(session.drafts),
                "draft_names": [d.name for d in session.drafts],
            }
        )

        # Build structured data for UI
        drafts_data = [
            {
                "name": d.name,
                "assetType": d.assetType,
                "isPrimary": d.isPrimary,
                "confidentiality": d.confidentiality,
                "integrity": d.integrity,
                "availability": d.availability,
                "reasoning": d.reasoning,
                "dependencies": d.dependencies,
                "compliance_notes": d.compliance_notes,
            }
            for d in session.drafts
        ]

        return {
            "status": "success",
            "drafts_count": len(input.assets),
            "total_drafts": len(session.drafts),
            "message": f"Created {len(input.assets)} asset draft(s). Total drafts in session: {len(session.drafts)}",
            "preview": preview,  # Markdown for agent
            "drafts": drafts_data,  # Structured data for UI
            "action_required": "User must confirm via 'yes' or modify drafts before submitting to Tazilla"
        }

    except Exception as e:
        logger.error(
            "Failed to create asset drafts",
            extra={"session_id": input.session_id, "error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Failed to create drafts: {str(e)}",
        }


@tool
async def asset_update_with_diff(input: AssetUpdateInput) -> dict:
    """
    Update an existing asset with visual diff preview.

    Shows before/after comparison before applying changes.
    User must confirm the update.

    Args:
        input: AssetUpdateInput with session_id, asset_id, changes, reasoning

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - diff: Before/after comparison
        - draft: Updated draft for preview
        - message: Human-readable message

    Note:
        This creates a draft with asset_id (indicates update).
        Actual update is sent to client after confirm_and_submit.
    """
    try:
        logger.info(
            f"asset_update_with_diff called for asset {input.asset_id}",
            extra={"session_id": input.session_id, "asset_id": input.asset_id, "changes": input.changes}
        )

        # Load session
        session = await get_agent_session(input.session_id)
        if not session:
            logger.error(f"Session not found: {input.session_id}")
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        # Find existing asset in session
        existing_asset = None
        for asset in session.existing_assets:
            if str(asset.get("id")) == input.asset_id:
                existing_asset = asset
                break

        if not existing_asset:
            logger.error(f"Asset {input.asset_id} not found in session")
            return {
                "status": "error",
                "message": f"Asset with ID {input.asset_id} not found. Please select an asset first.",
            }

        # Create updated draft by merging existing asset with changes
        updated_draft = AssetDraft(
            asset_id=input.asset_id,  # Mark as update (not create)
            name=input.changes.get("name", existing_asset.get("name")),
            assetType=input.changes.get("assetType", existing_asset.get("assetType")),
            isPrimary=input.changes.get("isPrimary", existing_asset.get("isPrimary")),
            confidentiality=input.changes.get("confidentiality", existing_asset.get("confidentiality")),
            integrity=input.changes.get("integrity", existing_asset.get("integrity")),
            availability=input.changes.get("availability", existing_asset.get("availability")),
            description=input.changes.get("description", existing_asset.get("description")),
            reasoning=input.reasoning or f"Updated asset: {', '.join(input.changes.keys())}",
            created_from="edit",
        )

        # Clear any existing drafts (update only one asset at a time)
        session.drafts = [updated_draft]

        # Save session
        await save_agent_session(session)

        # Build diff
        diff_changes = {}
        for key, new_value in input.changes.items():
            old_value = existing_asset.get(key)
            if old_value != new_value:
                diff_changes[key] = {"before": old_value, "after": new_value}

        # Build preview
        preview = _format_drafts_preview(session.drafts)

        logger.info(
            "Asset update draft created",
            extra={
                "session_id": input.session_id,
                "asset_id": input.asset_id,
                "changes_count": len(diff_changes),
            }
        )

        return {
            "status": "success",
            "message": f"Update preview ready for asset '{existing_asset.get('name')}'. Confirm to apply changes.",
            "diff": diff_changes,
            "preview": preview,
            "draft": {
                "asset_id": input.asset_id,
                "name": updated_draft.name,
                "changes": diff_changes,
                "reasoning": updated_draft.reasoning,
            },
            "action_required": "User must confirm via 'yes' to apply update"
        }

    except Exception as e:
        logger.error(
            "Failed to create update draft",
            extra={"session_id": input.session_id, "error": str(e)},
            exc_info=True
        )
        return {
            "status": "error",
            "message": f"Failed to create update preview: {str(e)}",
        }


@tool
async def confirm_and_submit(input: ConfirmSubmitInput) -> dict:
    """
    Submit asset drafts to Tazilla after user confirmation.

    This tool finalizes asset creation and sends them to the client.
    After calling this tool successfully:
    - Assets are prepared and sent to Tazilla client
    - Drafts are cleared from session
    - Task is COMPLETE - no further action needed
    - Agent should respond conversationally and wait for user's next request

    Use this tool ONLY when user explicitly confirms (says "yes", "ok", "confirm", etc.)

    IMPORTANT: After this tool returns success, DO NOT call asset_create_batch again
    unless the user explicitly asks for NEW or DIFFERENT assets.

    Args:
        input: ConfirmSubmitInput with session_id

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - assets: List of created asset objects (sent to client)
        - asset_names: List of asset names
        - count: Number of assets created
        - message: Detailed success message
        - final: True (indicates task completion)
    """
    from workshops_ai.api.schema.asset import Asset
    from pydantic import ValidationError

    try:
        # Load session
        session = await get_agent_session(input.session_id)
        if not session:
            logger.error(f"Session not found: {input.session_id}")
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        logger.info(
            f"confirm_and_submit: loaded session with {len(session.drafts)} drafts",
            extra={
                "session_id": input.session_id,
                "drafts_count": len(session.drafts),
                "drafts": [d.name for d in session.drafts] if session.drafts else []
            }
        )

        if not session.drafts:
            logger.warning(f"No drafts in session {input.session_id}")
            return {
                "status": "error",
                "message": "No drafts to submit. Create assets first using asset_create_batch.",
            }

        # Convert drafts → validated Asset models
        assets = []
        for i, draft in enumerate(session.drafts):
            try:
                # Apply defaults for None values (fallback for agent mistakes)
                is_primary = draft.isPrimary if draft.isPrimary is not None else True

                # Determine if this is an update (has asset_id) or create (no asset_id)
                asset_id = None
                if draft.asset_id:
                    # Update existing asset - parse UUID
                    from uuid import UUID
                    try:
                        asset_id = UUID(draft.asset_id)
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Invalid asset_id format: {draft.asset_id}",
                            extra={"session_id": input.session_id, "draft_index": i}
                        )

                asset = Asset(
                    id=asset_id,  # None for create, UUID for update
                    name=draft.name,
                    assetType=draft.assetType,
                    isPrimary=is_primary,
                    confidentiality=draft.confidentiality,
                    integrity=draft.integrity,
                    availability=draft.availability,
                    description=draft.description,
                )
                assets.append(asset)
            except ValidationError as e:
                logger.error(
                    f"Draft validation failed at index {i}",
                    extra={"error": str(e), "draft": draft.model_dump(), "validation_errors": e.errors()}
                )
                return {
                    "status": "error",
                    "message": f"Invalid draft at position {i+1}: {e.errors()[0]['msg']}. Error details: {str(e)}",
                }

        # Track created asset names for conversation context
        asset_names = [a.name for a in assets]
        session.created_asset_names.extend(asset_names)

        # Clear drafts (committed)
        session.drafts = []

        # Save session
        await save_agent_session(session)

        logger.info(
            "Assets prepared for client",
            extra={
                "session_id": input.session_id,
                "count": len(assets),
                "names": asset_names,
            }
        )

        # Build detailed success message
        # Determine if this is create or update
        is_update = any(a.id is not None for a in assets)
        operation = "Updated" if is_update else "Created"

        asset_list = "\n".join([f"  - {name}" for name in asset_names])
        success_msg = f"""SUCCESS! {operation} {len(assets)} asset(s) and sent them to client:
{asset_list}

The asset(s) are now in the client's Tazilla dashboard. Drafts have been cleared from the session.
Task is COMPLETE. Wait for user's next request."""

        # Return assets - agent_handler will put this in UIResponse.proposals
        return {
            "status": "success",
            "assets": [a.model_dump() for a in assets],
            "asset_names": asset_names,
            "count": len(assets),
            "message": success_msg,
            "final": True,
        }

    except Exception as e:
        logger.error(
            "Failed to prepare assets",
            extra={"session_id": input.session_id, "error": str(e)},
            exc_info=True
        )
        return {
            "status": "error",
            "message": f"Failed to prepare assets: {str(e)}",
        }


@tool
async def show_draft_preview(input: PreviewInput) -> dict:
    """
    Show preview of current drafts without submitting.

    Use this to:
    - Review current drafts
    - Show user what will be created
    - Check draft count

    Args:
        input: PreviewInput with session_id

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - drafts_count: Number of current drafts
        - preview: Formatted preview of drafts
    """
    try:
        session = await get_agent_session(input.session_id)
        if not session:
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        if not session.drafts:
            return {
                "status": "success",
                "drafts_count": 0,
                "message": "No drafts in session. Create some using asset_create_batch.",
            }

        preview = _format_drafts_preview(session.drafts)

        return {
            "status": "success",
            "drafts_count": len(session.drafts),
            "preview": preview,
            "message": f"Current session has {len(session.drafts)} draft(s)",
        }

    except Exception as e:
        logger.error(
            "Failed to show draft preview",
            extra={"session_id": input.session_id, "error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Failed to show preview: {str(e)}",
        }


@tool
async def clear_drafts(input: ClearDraftsInput) -> dict:
    """
    Clear all drafts from session.

    Use this when:
    - User wants to start over
    - User cancels current operation
    - User says "clear", "reset", "start again"

    Args:
        input: ClearDraftsInput with session_id

    Returns:
        Dictionary with:
        - status: "success" or "error"
        - cleared_count: Number of drafts cleared
        - message: Human-readable message
    """
    try:
        session = await get_agent_session(input.session_id)
        if not session:
            return {
                "status": "error",
                "message": f"Session not found: {input.session_id}",
            }

        cleared_count = len(session.drafts)
        session.drafts = []

        await save_agent_session(session)

        logger.info(
            "Drafts cleared",
            extra={"session_id": input.session_id, "count": cleared_count}
        )

        return {
            "status": "success",
            "cleared_count": cleared_count,
            "message": f"Cleared {cleared_count} draft(s). Ready to start fresh.",
        }

    except Exception as e:
        logger.error(
            "Failed to clear drafts",
            extra={"session_id": input.session_id, "error": str(e)}
        )
        return {
            "status": "error",
            "message": f"Failed to clear drafts: {str(e)}",
        }


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def _format_drafts_preview(drafts: list[AssetDraft]) -> str:
    """
    Format drafts into human-readable Markdown table.

    Returns:
        Markdown-formatted table preview string
    """
    if not drafts:
        return "No drafts"

    lines = [f"## Draft Assets ({len(drafts)})"]
    lines.append("")

    # Markdown table header
    lines.append("| # | Asset Name | Type | Primary | CIA | Reasoning |")
    lines.append("|---|------------|------|---------|-----|-----------|")

    for i, draft in enumerate(drafts, 1):
        # Format CIA levels
        cia_parts = []
        if draft.confidentiality:
            cia_parts.append(f"C:{draft.confidentiality}")
        if draft.integrity:
            cia_parts.append(f"I:{draft.integrity}")
        if draft.availability:
            cia_parts.append(f"A:{draft.availability}")
        cia = "<br>".join(cia_parts) if cia_parts else "-"

        # Format reasoning (truncate if too long)
        reasoning = draft.reasoning or "-"
        if len(reasoning) > 100:
            reasoning = reasoning[:97] + "..."

        # Primary indicator
        primary = "✓" if draft.isPrimary else "-"

        # Asset type
        asset_type = draft.assetType or "-"

        # Build table row
        lines.append(
            f"| {i} | {draft.name} | {asset_type} | {primary} | {cia} | {reasoning} |"
        )

    lines.append("")

    # Add additional details below table if needed
    has_details = any(
        d.dependencies or d.compliance_notes or (d.reasoning and len(d.reasoning) > 100)
        for d in drafts
    )

    if has_details:
        lines.append("### Additional Details")
        lines.append("")
        for i, draft in enumerate(drafts, 1):
            if draft.dependencies or draft.compliance_notes or (draft.reasoning and len(draft.reasoning) > 100):
                lines.append(f"**{i}. {draft.name}:**")

                if draft.reasoning and len(draft.reasoning) > 100:
                    lines.append(f"- **Full Reasoning**: {draft.reasoning}")

                if draft.dependencies:
                    deps = ", ".join(draft.dependencies)
                    lines.append(f"- **Dependencies**: {deps}")

                if draft.compliance_notes:
                    lines.append(f"- **Compliance**: {draft.compliance_notes}")

                lines.append("")

    return "\n".join(lines)
