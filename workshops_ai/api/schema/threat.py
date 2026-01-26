from typing import List, Literal, Optional
from decimal import Decimal
from pydantic import BaseModel, Field
from uuid import UUID

class Threat(BaseModel):
    """
    Represents a known threat type in the risk management system.
    Threats are predefined as standardized lists and are used as references for risk evaluation.
    """
    name: str = Field(
        ...,
        description=(
            "Unique name of the threat, e.g., 'Fire', 'Malware infiltration', 'Unauthorized access'. Used as a reference key; already localized."
        )
    )
    description: str = Field(
        ...,
        description=(
            "Detailed description of the threat, including how it may affect organizational assets; already localized."
        )
    )
    # Code list; always in english; get via API from backend
    category: Literal[
        "Physical threats",
        "Natural threats",
        "Compromise of functions or services",
        "Human actions",
        "Technical failures and infrastructure disruptions",
        "Organizational threats",
        "Legal and regulatory threats",
        "Violation of privacy",
        "Other"
    ] = Field(
        ...,
        description=(
            "Category of the threat for classification and filtering in risk analysis. " \
            "Must match one of the predefined categories."
        )
    )

    # Boolean flags for threat nature; already established
    unintentional: bool = Field(
        False,
        description="Indicates if the threat is unintentional (e.g., human error, accidental misconfiguration)."
    )
    intentional: bool = Field(
        False,
        description="Indicates if the threat is intentional (e.g., cyberattack, sabotage)."
    )
    environmental: bool = Field(
        False,
        description="Indicates if the threat is environmental or natural (e.g., flood, fire, earthquake)."
    )

    # Asset type associations
    assetTypes: List[str] = Field(
        default_factory=list,
        description="List of asset type names that this threat is relevant for. Empty list means threat applies to all asset types."
    )

class Risk(BaseModel):
    """
    Represents the evaluation of a single threat against an asset.
    The probability is a simplified 0-4 scale for AI-assisted risk assessment.
    """
    threatName: str = Field(
        ...,
        description="Reference name of the threat. Must match exactly."
    )
    probability: Literal[0, 1, 2, 3, 4] = Field(
        ...,
        description=(
            "Simplified probability of the threat materializing against the asset:\n"
            "0 = Not applicable (0%),\n"
            "1 = Very low (~10%),\n"
            "2 = Low (~50%),\n"
            "3 = Medium (~80%),\n"
            "4 = High (~100%)."
        )
    )
    inherentRisk: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description=(
            "Calculated inherent risk value in the range 0-100.\n"
            "Defined as: Inherent risk = (Maximum impact value x Probability).\n"
            "Represents the baseline risk level before considering any controls."
        ),
        json_schema_extra={"example": 75.5}
    )
    residualRiskCalculated: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description=(
            "Calculated residual risk after applying implemented controls, in the range 0-100.\n"
            "Derived from inherent risk and combined effectiveness of implemented controls.\n"
            "Combined effectiveness is computed as product: 1 - Π(1 - e_i), where e_i are coverage values of implemented controls expressed as fractions.\n"
            "Residual risk = Inherent risk × (1 - combined effectiveness)."
        ),
        json_schema_extra={"example": 45.2}
    )
    estimatedResidualRisk: Optional[Decimal] = Field(
        None,
        ge=0,
        le=100,
        description=(
            "Estimated residual risk after applying both implemented and planned controls, in the range 0-100.\n"
            "Uses the same combined effectiveness formula: 1 - Π(1 - e_i), where e_i are coverage values of all implemented and planned controls expressed as fractions.\n"
            "Estimated residual risk = Inherent risk × (1 - combined effectiveness)."
        ),
        json_schema_extra={"example": 32.8}
    )

class AssetRisk(BaseModel):
    """
    Represents the collection of evaluated risks for a single asset.
    List of risks, including their probability, can be generated or modified by AI when evaluating or updating the risks of an asset.
    """
    assetId: Optional[UUID] = Field(
        None, description="UUID of the asset, if available; used to match against existing asset records."
    )
    assetName: str = Field(
        ..., description=("Unique name of the asset. Used as a reference for new assets if ID has not yet been assigned.")
    )
    risks: List[Risk] = Field(
        ..., description=("List of evaluated risks (threats with their probability levels) for this asset. Generated or updated by AI-assisted risk analysis.")
    )
