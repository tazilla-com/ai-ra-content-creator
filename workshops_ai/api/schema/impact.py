from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from uuid import UUID

class Impact(BaseModel):
    """
    Represents an impact defined by the organization, based on ISO/IEC 27001, 27005, and 27002.
    Used to describe standardized types of negative effects (e.g., data loss, legal violation) considered during risk and asset assessment.
    """
    name: str = Field(
        ..., description="Unique name of the impact (e.g., 'Finance', 'Reputation'). Used as a reference key; already localized."
    )
    description: str = Field(
        ..., description="Explanation of the impact's relevance in the security or business context; already localized."
    )
    # Code list; always in english; get via API from backend
    impactCategory: Literal[
        "Availability",
        "Clients",
        "Confidentiality",
        "Finance",
        "Integrity",
        "Legislation",
        "Reputation",
        "People",
        "Other"
    ] = Field(
        ..., description="Higher-level category grouping the impact (e.g., 'Confidentiality', 'Finance'). Must not be translated or localized."
    )

class ImpactEvaluation(BaseModel):
    """
    Represents the assigned impact level (0-4) for a specific impact in the context of a single asset.
    Used to quantify the severity of a given impact on an asset.
    """
    impactName: str = Field(
        ..., description="Referencial name of the impact. Must match exactly."
    )
    impactLevel: Literal[0, 1, 2, 3, 4] = Field(
        ..., description=(
            "Impact level assigned to the concrete asset:\n"
            "0 = Not applicable,\n"
            "1 = Minimal,\n"
            "2 = Moderate,\n"
            "3 = Severe,\n"
            "4 = Catastrophic"
        )
    )

class AssetImpact(BaseModel):
    """
    Represents the collection of assigned impacts and their levels for a single asset.
    List of impacts, including their levels, can be generated or modified by AI when evaluating or updating the impacts of an asset.
    """
    assetId: Optional[UUID] = Field(
        None, description="UUID of the asset, if available; used to match against existing asset records."
    )
    assetName: str = Field(
        ..., description="Unique name of the asset. Used as a reference for new assets if ID has not yet been assigned."
    )
    impacts: List[ImpactEvaluation] = Field(
        ..., description="List of impacts assigned to the asset, each with its corresponding impact level."
    )
