"""
Session utility functions.

Helpers for working with SessionState, including patch analysis for recent_operations.
"""

from __future__ import annotations

from typing import Literal


def get_action_details_from_patches(
    patches: list[dict],
) -> tuple[Literal["created", "updated", "deleted"], str]:
    """
    Determine action type and human-readable details from JSON patches.

    Analyzes JSON Patch operations (RFC 6902) to understand what changed:
    - "add" operations on /assets/N → created
    - "remove" operations on /assets/N → deleted
    - "replace" operations on /assets/N/field → updated (with field names)

    Args:
        patches: List of JSON Patch operations

    Returns:
        Tuple of (action_type, details_string)

    Examples:
        >>> patches = [
        ...     {"op": "add", "path": "/assets/0", "value": {...}},
        ...     {"op": "add", "path": "/assets/1", "value": {...}}
        ... ]
        >>> get_action_details_from_patches(patches)
        ("created", "2 asset(s) created")

        >>> patches = [
        ...     {"op": "replace", "path": "/assets/0/confidentiality", "value": "HIGH"},
        ...     {"op": "replace", "path": "/assets/0/integrity", "value": "MEDIUM"}
        ... ]
        >>> get_action_details_from_patches(patches)
        ("updated", "Updated: confidentiality, integrity")

        >>> patches = [{"op": "remove", "path": "/assets/2"}]
        >>> get_action_details_from_patches(patches)
        ("deleted", "1 asset(s) deleted")
    """

    if not patches:
        return ("updated", "no changes")

    # Group patches by operation type
    ops_by_type: dict[str, list[str]] = {"add": [], "remove": [], "replace": []}

    for patch in patches:
        op = patch.get("op")
        path = patch.get("path", "")

        if op in ops_by_type:
            ops_by_type[op].append(path)

    # Determine action based on operations
    if ops_by_type["add"]:
        # Count top-level asset additions (path like /assets/0, not /assets/0/name)
        asset_adds = [
            p for p in ops_by_type["add"]
            if p.startswith("/assets/") and p.count("/") == 2
        ]
        count = len(asset_adds)
        return ("created", f"{count} asset(s) created")

    elif ops_by_type["remove"]:
        # Count top-level asset removals
        asset_removes = [
            p for p in ops_by_type["remove"]
            if p.startswith("/assets/") and p.count("/") == 2
        ]
        count = len(asset_removes)
        return ("deleted", f"{count} asset(s) deleted")

    elif ops_by_type["replace"]:
        # Parse which fields were updated
        fields_updated: set[str] = set()

        for path in ops_by_type["replace"]:
            # Path format: /assets/0/name or /assets/1/confidentiality
            parts = path.split("/")
            if len(parts) >= 3 and parts[1] == "assets":
                field = parts[-1]  # Last part is the field name
                fields_updated.add(field)

        # Build human-readable details
        if fields_updated:
            field_list = ", ".join(sorted(fields_updated))
            return ("updated", f"Updated: {field_list}")
        else:
            return ("updated", "Updated")

    # Fallback
    return ("updated", "modified")


def add_recent_operation(
    session,
    action: Literal["created", "updated", "deleted"],
    assets: list[str],
    details: str | None = None,
) -> None:
    """
    Add operation to session.recent_operations (FIFO, max 5).

    Args:
        session: SessionState instance
        action: Type of operation performed
        assets: Names of assets affected
        details: Human-readable details (optional)

    Note:
        Turn number is auto-calculated from conversation history.
        The field_validator on SessionState ensures max 5 operations.
    """
    from workshops_ai.session.models import RecentOperation

    # Calculate turn number (count assistant messages + 1)
    turn = len([m for m in session.conversation if m.role == "assistant"]) + 1

    operation = RecentOperation(
        turn=turn,
        action=action,
        assets=assets,
        details=details,
    )

    session.recent_operations.append(operation)
    # No need to manually trim - field_validator does it automatically
