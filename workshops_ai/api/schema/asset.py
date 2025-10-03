from pydantic import BaseModel, Field
from typing import Literal, Optional
from uuid import UUID

class Asset(BaseModel):
    org_id: str = Field(..., description="Organization ID of the asset.")
    
    id: Optional[UUID] = Field(
        None, description="UUID v7 of an existing asset. Omit or set to null when creating a new asset."
    )
    name:        str            = Field(...,  description="Localized asset name, using the language specified by the user.")
    description: Optional[str]  = Field(None, description="Localized asset description, using the language specified by the user.")

    isPrimary: bool = Field(True,  description=
        "Indicates if the asset is a primary (business-critical) or supporting asset. " \
        "This property cannot be changed for existing assets that have UUID. " \
        "For supporting assets, confidentiality, integrity, and availability must be set to null."
    )
    
    confidentiality: Optional[
        Literal['PUBLIC', 'INTERNAL', 'PROTECTED', 'STRICTLY_PROTECTED']
    ] = Field(None, description="Confidentiality level based on ISO 27001. Must not be translated or localized.")

    integrity: Optional[
        Literal['LOW', 'MEDIUM', 'HIGH']
    ] = Field(None, description="Integrity level based on ISO 27001. Must not be translated or localized.")

    availability: Optional[
        Literal['LOW', 'MEDIUM', 'HIGH']
    ] = Field(None, description="Availability level based on ISO 27001. Must not be translated or localized.")

    # Code list; always in english; get via API from backend
    assetType: Literal[
        "Application software", "Cloud services", "Data storage",
        "Electronic data/Business process", "Other", "Information system",
        "IoT", "Location", "Mobile devices", "Laptop", "Operating system",
        "Paper data", "PC", "Staff", "Portable data media",
        "Server infrastructure", "Network infrastructure",
        "Virtualization platform", "Equipment, machines"
    ] = Field(..., description="Asset type from a fixed code list. Must not be translated or localized.")
