"""
JSON Patch computation for fire-and-forget workflow.

Generates RFC 6902 patches for client to apply locally.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_patches(
    original_assets: list[dict],
    generated_assets: list[dict],
    operation_type: str = "create"
) -> list[dict]:
    """
    Compute RFC 6902 JSON patches for asset changes.

    Args:
        original_assets: Current assets from client
        generated_assets: Assets created/updated by agent
        operation_type: "create" or "update"

    Returns:
        List of JSON Patch operations

    Examples:
        CREATE:
        >>> compute_patches([], [{"name": "Server", ...}], "create")
        [{"op": "add", "path": "/assets/-", "value": {...}}]

        UPDATE:
        >>> compute_patches(
        ...     [{"id": "1", "name": "Server", "confidentiality": "LOW"}],
        ...     [{"id": "1", "confidentiality": "HIGH"}],
        ...     "update"
        ... )
        [{"op": "replace", "path": "/assets/0/confidentiality", "value": "HIGH"}]
    """
    patches = []

    if operation_type == "create":
        # Add new assets to the end of the list
        for asset in generated_assets:
            patches.append({
                "op": "add",
                "path": "/assets/-",  # Append to end
                "value": asset
            })

    elif operation_type == "update":
        # Find and update existing assets
        for updated_asset in generated_assets:
            asset_id = updated_asset.get("id")
            if not asset_id:
                logger.warning("Updated asset missing 'id', skipping patch generation")
                continue

            # Find index of original asset
            asset_index = None
            for i, orig in enumerate(original_assets):
                if orig.get("id") == asset_id:
                    asset_index = i
                    break

            if asset_index is None:
                logger.warning(f"Asset with id={asset_id} not found in original assets, skipping")
                continue

            # Generate replace patches for each changed field
            for field, new_value in updated_asset.items():
                if field == "id":
                    continue  # Don't patch the ID itself

                original_value = original_assets[asset_index].get(field)
                if original_value != new_value:
                    patches.append({
                        "op": "replace",
                        "path": f"/assets/{asset_index}/{field}",
                        "value": new_value
                    })

    else:
        logger.error(f"Unknown operation_type: {operation_type}")

    return patches


def compute_patches_auto(
    original_assets: list[dict],
    generated_assets: list[dict]
) -> list[dict]:
    """
    Auto-detect operation type and compute patches.

    Args:
        original_assets: Current assets from client
        generated_assets: Assets from agent

    Returns:
        List of JSON Patch operations
    """
    if not generated_assets:
        return []

    # Check if generated assets have IDs
    has_ids = all(asset.get("id") for asset in generated_assets)

    if has_ids:
        # Assets have IDs -> UPDATE
        return compute_patches(original_assets, generated_assets, "update")
    else:
        # No IDs -> CREATE
        return compute_patches(original_assets, generated_assets, "create")
