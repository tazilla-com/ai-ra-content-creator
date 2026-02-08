"""
JSON Patch computation for fire-and-forget workflow.

Generates patches using asset UUIDs as stable identifiers (not array indices).
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
    Compute JSON patches for asset changes using UUID-based paths.

    Args:
        original_assets: Current assets from client
        generated_assets: Assets created/updated by agent (or names for delete)
        operation_type: "create", "update", or "delete"

    Returns:
        List of patch operations with UUID-based paths

    Examples:
        CREATE:
        >>> compute_patches([], [{"name": "Server", ...}], "create")
        [{"op": "add", "path": "/assets/-", "value": {...}}]

        UPDATE:
        >>> compute_patches(
        ...     [{"id": "abc-123", "name": "Server", "confidentiality": "LOW"}],
        ...     [{"id": "abc-123", "confidentiality": "HIGH"}],
        ...     "update"
        ... )
        [{"op": "replace", "path": "/assets/abc-123/confidentiality", "value": "HIGH"}]

        DELETE:
        >>> compute_patches(
        ...     [{"id": "abc-123", "name": "Server"}],
        ...     ["Server"],
        ...     "delete"
        ... )
        [{"op": "remove", "path": "/assets/abc-123"}]
    """
    patches = []

    if operation_type == "create":
        for asset in generated_assets:
            patches.append({
                "op": "add",
                "path": "/assets/-",
                "value": asset
            })

    elif operation_type == "update":
        for updated_asset in generated_assets:
            asset_id = updated_asset.get("id")
            if not asset_id:
                logger.warning("Updated asset missing 'id', skipping patch generation")
                continue

            # Find original asset by ID
            original = next(
                (a for a in original_assets if a.get("id") == asset_id),
                None
            )
            if original is None:
                logger.warning(f"Asset with id={asset_id} not found in original assets, skipping")
                continue

            # Generate replace patches for each changed field
            for field, new_value in updated_asset.items():
                if field == "id":
                    continue
                if original.get(field) != new_value:
                    patches.append({
                        "op": "replace",
                        "path": f"/assets/{asset_id}/{field}",
                        "value": new_value
                    })

    elif operation_type == "delete":
        # generated_assets is list of names for delete
        for name in generated_assets:
            original = next(
                (a for a in original_assets if a.get("name") == name),
                None
            )
            if original is None:
                logger.warning(f"Asset '{name}' not found in original assets, skipping remove patch")
                continue

            asset_id = original.get("id", name)
            patches.append({
                "op": "remove",
                "path": f"/assets/{asset_id}",
            })

    else:
        logger.error(f"Unknown operation_type: {operation_type}")

    return patches


def compute_patches_auto(
    original_assets: list[dict],
    generated_assets: list[dict],
    deleted_asset_names: list[str] | None = None,
) -> list[dict]:
    """
    Auto-detect operation type and compute patches.

    Args:
        original_assets: Current assets from client
        generated_assets: Assets from agent (created or updated)
        deleted_asset_names: Asset names to remove

    Returns:
        List of patch operations (remove first, then add/replace)
    """
    patches = []

    # 1. Remove patches first
    if deleted_asset_names:
        patches.extend(
            compute_patches(original_assets, deleted_asset_names, "delete")
        )

    # 2. Add/replace patches
    if generated_assets:
        # Check if generated assets match existing ones by ID → update
        original_ids = {a.get("id") for a in original_assets if a.get("id")}
        has_ids = all(asset.get("id") for asset in generated_assets)

        if has_ids and any(a.get("id") in original_ids for a in generated_assets):
            patches.extend(
                compute_patches(original_assets, generated_assets, "update")
            )
        else:
            patches.extend(
                compute_patches(original_assets, generated_assets, "create")
            )

    return patches
