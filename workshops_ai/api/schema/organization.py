from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

class OrganizationalUnit(BaseModel):
    """
    Represents a department or unit within the organization.
    Provides business context for asset classification and impact assessment.
    Supports AI-driven risk analysis.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(..., description="Localized name of the department or unit (e.g., IT, Finance, Production).")
    description: Optional[str] = Field(None, description="Optional description of the unit's responsibilities or function.")

class JobPosition(BaseModel):
    """
    Represents a role or job position in the organization.
    Relevant for human-related risks, access management, and responsibility mapping.
    Supports AI-driven risk analysis.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(..., description="Localized job title (e.g., Network Administrator, HR Specialist).")
    description: Optional[str] = Field(None, description="Optional description of responsibilities and risk relevance.")

class Organization(BaseModel):
    """
    Represents the core profile of an organization for AI-assisted risk analysis.
    Includes structure, locations, and workforce context to support AI.
    """
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    organizationName: str            = Field(...,  description="Official name of the organization.")
    legalForm: Optional[str]         = Field(None, description="Localized legal form of the organization (e.g., s.r.o., Inc.).")
    registeredOffice: Optional[str]  = Field(None, description="Official registered address of the organization.")
    country: Optional[str]           = Field(None, description="International country code of the registered office for regulatory context.")
    numberOfEmployees: Optional[str] = Field(None, description="Number of employees in the whole organization (can be text like '50-100' or '~200').")
    organizationalUnits: List[OrganizationalUnit] = Field(
        default_factory=list,
        description="Optional list of organizational units or departments."
    )
    jobPositions: List[JobPosition] = Field(
        default_factory=list,
        description="Optional list of job positions relevant for access control and human-factor risk considerations."
    )
