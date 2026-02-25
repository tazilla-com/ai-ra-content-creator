# workshops_ai/wizard/catalog_loader.py
"""
Impact and Threat catalog loader for wizard endpoints.

Loads Impact and Threat catalogs from CSV files and caches in memory.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

from workshops_ai.api.schema.impact import Impact
from workshops_ai.api.schema.threat import Threat
from workshops_ai.api.schema.control import PlannedControl

logger = logging.getLogger(__name__)

# Cache
_impact_cache: Optional[List[Impact]] = None
_map_cache: Optional[Dict[str, str]] = None
_threat_cache: Optional[List[Threat]] = None
_threat_map_cache: Optional[Dict[str, str]] = None
_control_cache: Optional[List[PlannedControl]] = None
_control_map_cache: Optional[Dict[str, str]] = None
_control_threat_coverage_cache: Optional[Dict[str, Dict[str, int]]] = None

# Catalog base path
CATALOG_BASE = Path(__file__).parent.parent / "data" / "catalogs"


def _read_csv(file_path: Path) -> List[Dict[str, str]]:
    """
    Read CSV file and return list of dicts.

    Args:
        file_path: Path to CSV file

    Returns:
        List of row dicts
    """
    if not file_path.exists():
        logger.error(f"CSV file not found: {file_path}")
        return []

    rows = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        logger.error(f"Failed to read CSV {file_path}: {e}")
        return []

    return rows


def load_impact_category_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load impact category catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of dicts with name and description
    """
    # CSV provider: ignores session_id, uses static catalog files
    rows = _read_csv(CATALOG_BASE / "Impact category.csv")
    categories: List[Dict[str, str]] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and notes
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            notes = (r.get("SK Notes") or r.get("Notes") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()
        else:  # en
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()

        categories.append({
            "name": name,
            "description": notes
        })

    logger.info(f"Loaded {len(categories)} impact categories for language '{language}'")
    return categories


def load_threat_category_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load threat category catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of dicts with name and description
    """
    # CSV provider: ignores session_id, uses static catalog files
    rows = _read_csv(CATALOG_BASE / "Threat category.csv")
    categories: List[Dict[str, str]] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and notes
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            notes = (r.get("SK Notes") or r.get("Notes") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()
        else:  # en
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()

        categories.append({
            "name": name,
            "description": notes
        })

    logger.info(f"Loaded {len(categories)} threat categories for language '{language}'")
    return categories


def load_control_category_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load control category catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of dicts with name and description
    """
    # CSV provider: ignores session_id, uses static catalog files
    rows = _read_csv(CATALOG_BASE / "Control category.csv")
    categories: List[Dict[str, str]] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and notes
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            notes = (r.get("SK Notes") or r.get("Notes") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()
        else:  # en
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()

        categories.append({
            "name": name,
            "description": notes
        })

    logger.info(f"Loaded {len(categories)} control categories for language '{language}'")
    return categories


def load_impact_to_category() -> Dict[str, str]:
    """
    Load impact-to-category mapping from CSV.

    Returns:
        Dict mapping impact name (lowercase) to category
    """
    global _map_cache
    if _map_cache is not None:
        return _map_cache

    rows = _read_csv(CATALOG_BASE / "Impact x Impact category.csv")
    mapping: Dict[str, str] = {}

    for r in rows:
        impact = (r.get("Impact") or "").strip()
        category = (r.get("Impact category") or "Other").strip()
        if not impact:
            continue
        # Store keys case-insensitively to improve matching
        mapping[impact.lower()] = category

    _map_cache = mapping
    logger.info(f"Loaded {len(mapping)} impact-to-category mappings")
    return mapping


def load_impact_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Impact]:
    """
    Load impact catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of Impact objects with name, description, weight, and category
    """
    global _impact_cache

    # CSV provider: ignores session_id, uses static catalog files
    # Use cache only if language is EN (base language)
    # For other languages, always reload to get localized names
    if language == "en" and _impact_cache is not None:
        return _impact_cache

    rows = _read_csv(CATALOG_BASE / "Impact.csv")
    mapping = load_impact_to_category()

    # Allowed categories (from Impact model)
    allowed_categories = {
        "Availability",
        "Clients",
        "Confidentiality",
        "Finance",
        "Integrity",
        "Legislation",
        "Reputation",
        "People",
        "Other"
    }

    impacts: List[Impact] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("Name") or r.get("EN Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and description
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            desc = (r.get("SK Description") or r.get("Description") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            desc = (r.get("EN Description") or r.get("Description") or "").strip()
        else:  # en
            name = base_name
            desc = (r.get("EN Description") or r.get("Description") or "").strip()

        # Get category from mapping (use base name for lookup)
        category = mapping.get(base_name.lower(), "Other")
        if category not in allowed_categories:
            logger.warning(f"Invalid category '{category}' for impact '{name}', using 'Other'")
            category = "Other"

        impacts.append(Impact(
            name=name,
            description=desc,
            impactCategory=category
        ))

    # Cache only EN version
    if language == "en":
        _impact_cache = impacts

    logger.info(f"Loaded {len(impacts)} impacts for language '{language}'")
    return impacts


def get_impact_names(language: str = "en", session_id: Optional[str] = None) -> List[str]:
    """
    Get ordered list of impact names from catalog.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of impact names in catalog order
    """
    impacts = load_impact_catalog(language, session_id=session_id)
    return [imp.name for imp in impacts]


def clear_cache():
    """Clear cached catalog data (for testing)."""
    global _impact_cache, _map_cache, _threat_cache, _threat_map_cache, _threat_asset_types_cache
    global _control_cache, _control_map_cache, _control_threat_coverage_cache
    _impact_cache = None
    _map_cache = None
    _threat_cache = None
    _threat_map_cache = None
    _threat_asset_types_cache = None
    _control_cache = None
    _control_map_cache = None
    _control_threat_coverage_cache = None


# ===================================================================================
# Threat Catalog Functions
# ===================================================================================

def load_threat_to_category() -> Dict[str, str]:
    """
    Load threat-to-category mapping from CSV.

    Returns:
        Dict mapping threat name (lowercase) to category
    """
    global _threat_map_cache
    if _threat_map_cache is not None:
        return _threat_map_cache

    rows = _read_csv(CATALOG_BASE / "Threat x Threat category.csv")
    mapping: Dict[str, str] = {}

    for r in rows:
        threat = (r.get("Threat") or "").strip()
        category = (r.get("Threat category") or "Other").strip()
        if not threat:
            continue
        # Store keys case-insensitively to improve matching
        mapping[threat.lower()] = category

    _threat_map_cache = mapping
    logger.info(f"Loaded {len(mapping)} threat-to-category mappings")
    return mapping


# Cache for threat-to-asset-types mapping
_threat_asset_types_cache: Optional[Dict[str, List[str]]] = None


def load_threat_to_asset_types() -> Dict[str, List[str]]:
    """
    Load threat-to-asset-types mapping from CSV.

    The presence of a row in the CSV indicates the mapping relationship.
    The Value column (TRUE/FALSE) is ignored - it represents other information.

    Returns:
        Dict mapping threat name (lowercase) to list of asset type names
    """
    global _threat_asset_types_cache
    if _threat_asset_types_cache is not None:
        return _threat_asset_types_cache

    rows = _read_csv(CATALOG_BASE / "Asset type x Threat.csv")
    mapping: Dict[str, List[str]] = {}

    for r in rows:
        threat = (r.get("Threat") or "").strip()
        asset_type = (r.get("Asset type") or "").strip()

        if not threat or not asset_type:
            continue

        # Row presence = mapping relationship (ignore Value column)
        threat_key = threat.lower()
        if threat_key not in mapping:
            mapping[threat_key] = []
        # Avoid duplicates (some threats have multiple rows per asset type)
        if asset_type not in mapping[threat_key]:
            mapping[threat_key].append(asset_type)

    _threat_asset_types_cache = mapping
    logger.info(f"Loaded threat-to-asset-types mappings for {len(mapping)} threats")
    return mapping


def load_threat_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Threat]:
    """
    Load threat catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of Threat objects with name, description, category, and threat type flags
    """
    global _threat_cache

    # CSV provider: ignores session_id, uses static catalog files
    # Use cache only if language is EN (base language)
    # For other languages, always reload to get localized names
    if language == "en" and _threat_cache is not None:
        return _threat_cache

    rows = _read_csv(CATALOG_BASE / "Threat.csv")
    category_mapping = load_threat_to_category()
    asset_types_mapping = load_threat_to_asset_types()

    # Allowed categories (from Threat model)
    allowed_categories = {
        "Physical threats",
        "Natural threats",
        "Compromise of functions or services",
        "Human actions",
        "Technical failures and infrastructure disruptions",
        "Organizational threats",
        "Legal and regulatory threats",
        "Violation of privacy",
        "Other"
    }

    threats: List[Threat] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("Name") or r.get("EN Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and description
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            desc = (r.get("SK Description") or r.get("Description") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            desc = (r.get("EN Description") or r.get("Description") or "").strip()
        else:  # en
            name = base_name
            desc = (r.get("EN Description") or r.get("Description") or "").strip()

        # Parse threat type flags (Unintentional, Intentional, Environmental)
        unintentional = r.get("Unintentional", "").strip().lower() in ["true", "1", "yes"]
        intentional = r.get("Intentional", "").strip().lower() in ["true", "1", "yes"]
        environmental = r.get("Environmental", "").strip().lower() in ["true", "1", "yes"]

        # Get asset types from mapping (use base name for lookup)
        asset_types = asset_types_mapping.get(base_name.lower(), [])

        # Get category from mapping (use base name for lookup)
        category = category_mapping.get(base_name.lower(), "Other")
        if category not in allowed_categories:
            logger.warning(f"Invalid category '{category}' for threat '{name}', using 'Other'")
            category = "Other"

        threats.append(Threat(
            name=name,
            description=desc,
            category=category,
            unintentional=unintentional,
            intentional=intentional,
            environmental=environmental,
            assetTypes=asset_types
        ))

    # Cache only EN version
    if language == "en":
        _threat_cache = threats

    logger.info(f"Loaded {len(threats)} threats for language '{language}'")
    return threats


def get_threat_names(language: str = "en", session_id: Optional[str] = None) -> List[str]:
    """
    Get ordered list of threat names from catalog.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of threat names in catalog order
    """
    threats = load_threat_catalog(language, session_id=session_id)
    return [threat.name for threat in threats]


def load_asset_type_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Dict[str, any]]:
    """
    Load asset type catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of dicts with name, description/notes, and isDependent flag
    """
    # CSV provider: ignores session_id, uses static catalog files
    rows = _read_csv(CATALOG_BASE / "Asset type.csv")
    asset_types: List[Dict[str, any]] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and notes
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            notes = (r.get("SK Notes") or r.get("Notes") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()
        else:  # en
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()

        # Parse isDependent flag
        is_dependent_str = (r.get("Is dependent") or "FALSE").strip().upper()
        is_dependent = is_dependent_str in ("TRUE", "1", "YES")

        asset_types.append({
            "name": name,
            "description": notes,
            "isDependent": is_dependent
        })

    logger.info(f"Loaded {len(asset_types)} asset types for language '{language}'")
    return asset_types


def get_asset_type_names(language: str = "en", session_id: Optional[str] = None) -> List[str]:
    """
    Get ordered list of asset type names from catalog.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of asset type names in catalog order
    """
    asset_types = load_asset_type_catalog(language, session_id=session_id)
    return [at["name"] for at in asset_types]


def load_sector_catalog(language: str = "en", session_id: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load sector catalog from CSV file with localization.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of dicts with name and description
    """
    rows = _read_csv(CATALOG_BASE / "Sector.csv")
    sectors: List[Dict[str, str]] = []

    for r in rows:
        base_name = (r.get("EN Name") or r.get("Name") or "").strip()
        if not base_name:
            continue

        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            notes = (r.get("SK Notes") or r.get("Notes") or "").strip()
        elif language == "cs":
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()
        else:
            name = base_name
            notes = (r.get("EN Notes") or r.get("Notes") or "").strip()

        sectors.append({
            "name": name,
            "description": notes
        })

    logger.info(f"Loaded {len(sectors)} sectors for language '{language}'")
    return sectors


def get_sector_names(language: str = "en", session_id: Optional[str] = None) -> List[str]:
    """
    Get ordered list of sector names from catalog.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of sector names in catalog order
    """
    sectors = load_sector_catalog(language, session_id=session_id)
    return [s["name"] for s in sectors]


def load_asset_type_threat_mapping() -> Dict[str, List[str]]:
    """
    Load AssetType -> Threat mapping from CSV (optional for smart threat selection).

    The CSV maps which threats are relevant for each asset type.
    The 'Value' column is metadata (not a relevance filter) - all rows
    represent valid asset-type-to-threat associations.

    Returns:
        Dict mapping asset type to list of relevant threat names
    """
    rows = _read_csv(CATALOG_BASE / "Asset type x Threat.csv")
    threat_sets: Dict[str, set] = {}

    for r in rows:
        asset_type = (r.get("Asset type") or "").strip()
        threat = (r.get("Threat") or "").strip()

        if not asset_type or not threat:
            continue

        threat_sets.setdefault(asset_type, set()).add(threat)

    threat_mapping = {k: sorted(v) for k, v in threat_sets.items()}

    logger.info(f"Loaded threat-to-asset-types mappings for {len(threat_mapping)} asset types")
    return threat_mapping


# ===================================================================================
# Control Catalog Functions
# ===================================================================================

def load_control_to_category() -> Dict[str, str]:
    """
    Load control-to-category mapping from CSV.

    Returns:
        Dict mapping control name (case-sensitive) to category
    """
    global _control_map_cache
    if _control_map_cache is not None:
        return _control_map_cache

    rows = _read_csv(CATALOG_BASE / "Control x Control category.csv")
    mapping: Dict[str, str] = {}

    for r in rows:
        control = (r.get("Control") or "").strip()
        category = (r.get("Control category") or "Other").strip()
        if not control:
            continue
        # Store keys case-sensitively (control names must match exactly)
        mapping[control] = category

    _control_map_cache = mapping
    logger.info(f"Loaded {len(mapping)} control-to-category mappings")
    return mapping


def load_control_catalog(language: str = "en", session_id: Optional[str] = None) -> List[PlannedControl]:
    """
    Load control catalog from CSV file with localization.

    Note: Returns controls WITHOUT threat coverage or costs.
    Those are populated by the AI generator based on context.

    Args:
        language: Output language (en/sk/cs)
        session_id: Session ID for client-specific catalogs (CSV provider ignores this)

    Returns:
        List of PlannedControl objects with empty threats list
    """
    global _control_cache

    # CSV provider: ignores session_id, uses static catalog files
    # Use cache only if language is EN (base language)
    # For other languages, always reload to get localized names
    if language == "en" and _control_cache is not None:
        return _control_cache

    rows = _read_csv(CATALOG_BASE / "Control.csv")
    mapping = load_control_to_category()

    # Allowed categories (from PlannedControl model)
    allowed_categories = {
        "Information security policies",
        "Organization of information security",
        "Human resource security",
        "Threat intelligence",
        "Asset management",
        "Access control",
        "Cryptography",
        "Physical and environmental security",
        "Business continuity management",
        "Operations security",
        "Vulnerability Management",
        "Communications security",
        "System acquisition, development and maintenance",
        "Supplier relationships",
        "Compliance",
        "Logging and monitoring",
        "Other"
    }

    controls: List[PlannedControl] = []

    for r in rows:
        # Get base name (always EN)
        base_name = (r.get("Name") or r.get("EN Name") or "").strip()
        if not base_name:
            continue

        # Get localized name and description
        if language == "sk":
            name = (r.get("SK Name") or base_name).strip()
            desc = (r.get("SK Notes") or r.get("Notes") or "").strip()
        elif language == "cs":
            # CS not in this CSV, fallback to EN
            name = base_name
            desc = (r.get("EN Notes") or r.get("Notes") or "").strip()
        else:  # en
            name = base_name
            desc = (r.get("EN Notes") or r.get("Notes") or "").strip()

        # Get category from mapping (use base name for lookup)
        category = mapping.get(base_name, "Other")

        # Try case-insensitive match if exact match not found
        if category not in allowed_categories:
            # Try to find case-insensitive match
            category_lower = category.lower()
            for allowed_cat in allowed_categories:
                if allowed_cat.lower() == category_lower:
                    category = allowed_cat
                    break
            else:
                logger.warning(f"Invalid category '{category}' for control '{name}', using 'Other'")
                category = "Other"

        controls.append(PlannedControl(
            id=None,  # No ID from catalog
            name=name,
            description=desc if desc else None,
            category=category,
            implementationCost=None,  # AI estimates these
            estimatedAnnualCost=None,
            threats=[]  # AI populates based on context
        ))

    # Cache only EN version
    if language == "en":
        _control_cache = controls

    logger.info(f"Loaded {len(controls)} controls for language '{language}'")
    return controls


def load_control_threat_coverage() -> Dict[str, Dict[str, int]]:
    """
    Load control-to-threat coverage mapping from CSV.

    Returns:
        Dict mapping control name -> {threat name -> coverage percentage}

    Example:
        {
            "Firewall": {
                "Unauthorized manipulation of software": 50,
                "Malware infiltration": 40,
                ...
            }
        }
    """
    global _control_threat_coverage_cache
    if _control_threat_coverage_cache is not None:
        return _control_threat_coverage_cache

    rows = _read_csv(CATALOG_BASE / "Control x Threat.csv")
    coverage: Dict[str, Dict[str, int]] = {}

    for r in rows:
        control = (r.get("Control") or "").strip()
        threat = (r.get("Threat") or "").strip()
        value_str = (r.get("Value") or "0").strip()

        if not control or not threat:
            continue

        # Parse coverage value
        try:
            value = int(value_str)
        except ValueError:
            logger.warning(f"Invalid coverage value '{value_str}' for control '{control}' -> threat '{threat}'")
            continue

        # Build nested dict
        if control not in coverage:
            coverage[control] = {}

        coverage[control][threat] = value

    _control_threat_coverage_cache = coverage
    logger.info(f"Loaded threat coverage for {len(coverage)} controls")
    return coverage


def get_control_names(language: str = "en") -> List[str]:
    """
    Get ordered list of control names from catalog.

    Args:
        language: Output language (en/sk/cs)

    Returns:
        List of control names in catalog order
    """
    controls = load_control_catalog(language)
    return [ctrl.name for ctrl in controls]
