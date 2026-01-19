# workshops_ai/agents/tools
"""
Reusable tools for LangGraph agents.

Tools:
- Asset operations (create batch, update with diff, confirm/submit)
- Website scraping (wrapper around existing scraper)
- Dependency detection (LLM-based)
- Asset recommendations (LLM-based, NO hardcoded rules)
"""
from __future__ import annotations

from workshops_ai.graph.assets.tools.assets import (
    asset_create_batch,
    asset_update_with_diff,
    confirm_and_submit,
    show_draft_preview,
    clear_drafts,
)

from workshops_ai.agents.tools.scraping import (
    scrape_company_website,
)

from workshops_ai.agents.tools.intelligence import (
    detect_asset_dependencies,
    recommend_assets_for_context,
)

__all__ = [
    # Asset operations
    "asset_create_batch",
    "asset_update_with_diff",
    "confirm_and_submit",
    "show_draft_preview",
    "clear_drafts",
    # Scraping
    "scrape_company_website",
    # Intelligence (LLM-based)
    "detect_asset_dependencies",
    "recommend_assets_for_context",
]
