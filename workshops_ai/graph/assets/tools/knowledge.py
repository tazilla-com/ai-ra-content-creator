"""
Knowledge lookup tool for RAG-based methodology and best practices.
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from pydantic import BaseModel, Field


class LookupKnowledgeInput(BaseModel):
    """Input for knowledge lookup."""
    query: str = Field(
        ...,
        description="Search query for knowledge base. Be specific about what you need.",
        examples=[
            "healthcare hospital IT assets best practices",
            "CIA classification methodology integrity levels",
            "typical assets for manufacturing company"
        ]
    )


@tool(args_schema=LookupKnowledgeInput)
def lookup_knowledge(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState]
) -> Command:
    """
    Search internal knowledge base for methodology, industry templates, and best practices.

    IMPORTANT: RAG ONLY - does NOT search the web (cost control).
    For company-specific information, use scrape_company_info instead.

    USE THIS TOOL WHEN:
    - User asks for industry-specific assets ("assety pre nemocnicu", "e-shop infrastructure")
    - You need methodology guidance for CIA classification
    - You're unsure about best practices for a specific industry

    SKIP THIS TOOL WHEN:
    - User gives very specific asset description with clear context
    - You already have enough context from organization info in state
    - User is updating existing assets (not creating new ones)
    - User asks simple questions you can answer from general knowledge

    USAGE PATTERN:
    1. Call this BEFORE create_assets when you need industry context
    2. Read the returned text carefully
    3. Use the knowledge to inform your create_assets description

    Returns plain text with relevant knowledge, or message if nothing found.
    """
    from workshops_ai.graph.assets.tools.rag import search_rag

    results = search_rag(query)

    if not results:
        content = f"No knowledge found for: {query}. Proceeding with general best practices."
        return Command(
            update={
                "tools_used": state.get("tools_used", []) + ["lookup_knowledge"],
                "messages": [ToolMessage(content=content, tool_call_id=tool_call_id)]
            }
        )

    return Command(
        update={
            "tools_used": state.get("tools_used", []) + ["lookup_knowledge"],
            "messages": [ToolMessage(content=results, tool_call_id=tool_call_id)]
        }
    )
