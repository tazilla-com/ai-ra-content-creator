from typing import Literal, Optional
from pydantic import BaseModel, Field

class Service(BaseModel):
    """
    Represents a core business service within the organization.
    Provides business context to support AI-assisted risk assessments, including asset mapping, impacts, threats, and control considerations.
    """
    name: str = Field(..., description="Unique localized name of the service (e.g., Broadband Internet).")
    description: Optional[str] = Field(
        None,
        description="Description of the service, its function, or business relevance."
    )
    # Code list; always in english; get via API from backend
    sector: Optional[
        Literal[
            "Digital infrastructure",
            "Transport",
            "Energy",
            "Finance",
            "Financial markets infrastructure",
            "Post",
            "Industry",
            "Public administration",
            "Water and atmosphere",
            "Health"
        ]
    ] = Field(
        None,
        description=(
            "Optional sector classification of the service for regulatory and risk context. Fixed code list; must not be translated or localized."
        )
    )
