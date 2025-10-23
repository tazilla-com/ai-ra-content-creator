# File: workshops_ai/app/security.py
# Purpose: Unified authentication logic for HTTP and WebSocket.
# - Reads exclusively from central settings (get_settings()).
# - Supports API Key (X-API-Key / ?api_key) and JWT (Authorization: Bearer ...).
# - Shared core → thin wrappers for FastAPI HTTP dependency and WS handshake.
#
# Notes:
#   • All header lookups are case-insensitive (keys normalized to lowercase).
#   • API key is accepted from the "x-api-key" header OR the "api_key" query param.
#   • Bearer token is extracted from the lowercase "authorization" header.
#
# HTTP usage:
#   from fastapi import APIRouter, Depends
#   from workshops_ai.app.security import require_auth
#   router = APIRouter(dependencies=[Depends(require_auth)])
#
# WebSocket usage:
#   from fastapi import WebSocket
#   from workshops_ai.app.security import verify_ws_auth
#   @router.websocket("/ws")
#   async def ws_endpoint(ws: WebSocket):
#       identity = await verify_ws_auth(ws)   # call BEFORE accept to reject early
#       if identity is None:
#           return  # closed with 4401
#       await ws.accept()
#       ...
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import jwt  # PyJWT
from fastapi import HTTPException, Request, WebSocket, status

from workshops_ai.settings import get_settings

__all__ = [
    "require_auth",
    "verify_ws_auth",
]

logger = logging.getLogger("workshops_ai.security")


# ----------------------------- Core helpers ---------------------------------


def _normalize_headers(headers_src) -> Dict[str, str]:
    """
    Normalize various header sources (Request.headers, WebSocket.headers, .raw, list of pairs)
    into a plain dict[str, str] with lowercase keys.
    - Accepts mapping-like objects (have .items())
    - Accepts iterables of (key, value) where key/value may be bytes or str
    """
    # Mapping-like (e.g., Starlette Headers object, dict)
    if hasattr(headers_src, "items"):
        return {str(k).lower(): str(v) for k, v in headers_src.items()}

    # Iterable of pairs
    result: Dict[str, str] = {}
    try:
        iterator = iter(headers_src)  # may raise TypeError if not iterable
    except TypeError:
        return result

    for item in iterator:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        k, v = item
        if isinstance(k, bytes):
            k = k.decode()
        if isinstance(v, bytes):
            v = v.decode()
        result[str(k).lower()] = str(v)
    return result


def _extract_api_key(headers: Dict[str, str], query: Dict[str, str]) -> Optional[str]:
    """
    API key is accepted either via:
      - HTTP header: X-API-Key (any capitalization) → normalized to 'x-api-key'
      - Query param: ?api_key=...
    """
    return (headers.get("x-api-key") or query.get("api_key") or "").strip() or None


def _validate_api_key(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate the provided API key against settings.
    Returns an identity dict on success, or None when API-key auth is disabled
    or when the key doesn't match.
    Raises 401 when API-key auth is enabled but no keys are configured.
    """
    sec = get_settings().security
    if not sec.enable_api_key:
        return None
    if not sec.api_keys:
        # API-key mode is enabled but no keys configured → deny by design.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key authentication is enabled, but no keys are configured.",
        )
    if api_key and api_key in sec.api_keys:
        # Do not return the raw key; return a surrogate identity.
        return {"sub": "apikey", "method": "api_key"}
    return None


def _extract_bearer(headers: Dict[str, str]) -> Optional[str]:
    """
    Extract "Bearer <token>" from the Authorization header (case-insensitive).
    """
    auth = headers.get("authorization")
    if not auth:
        return None
    parts = auth.strip().split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def _validate_jwt(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate JWT based on settings (algorithm/secret/issuer/audience).
    Returns an identity dict on success, or None when JWT auth is disabled or token missing.
    """
    sec = get_settings().security
    if not sec.enable_jwt:
        return None
    if not token:
        return None

    options = {
        "verify_signature": True,
        "verify_exp": True,
        "verify_aud": bool(sec.jwt_audience),
        "verify_iss": bool(sec.jwt_issuer),
    }

    try:
        claims = jwt.decode(
            token,
            sec.jwt_secret,              # HS* secret or RS*/ES* public key (as configured)
            algorithms=[sec.jwt_algorithm],
            audience=sec.jwt_audience,
            issuer=sec.jwt_issuer,
            options=options,
        )
        sub = claims.get("sub") or claims.get("uid") or "jwt"
        return {"sub": sub, "method": "jwt", "claims": claims}
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT") from e


def _choose_identity(headers: Dict[str, str], query: Dict[str, str]) -> Dict[str, Any]:
    """
    Try API-key first (if enabled), then JWT (if enabled).
    If neither method is enabled → 401.
    If methods are enabled but none succeed → 401 Unauthorized.
    """
    sec = get_settings().security

    if sec.disable_all_auth:
        return {"sub": "system", "method": "disabled"}

    if sec.enable_api_key:
        ident = _validate_api_key(_extract_api_key(headers, query))
        if ident:
            return ident

    if sec.enable_jwt:
        ident = _validate_jwt(_extract_bearer(headers))
        if ident:
            return ident

    if not sec.enable_api_key and not sec.enable_jwt:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication method is enabled.",
        )

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


# ------------------------------ Public API ----------------------------------


async def require_auth(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency for HTTP endpoints.
    Returns an identity dict (e.g., {'sub': '...', 'method': 'api_key'|'jwt', ...}).
    Also attaches identity to request.state.identity for downstream usage.
    """
    headers = _normalize_headers(request.headers)
    query = {str(k): str(v) for k, v in request.query_params.items()}
    identity = _choose_identity(headers, query)
    request.state.identity = identity
    return identity


async def verify_ws_auth(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    """
    Guard for WebSocket handlers.
    Extracts auth from headers/query, validates via central settings.
    On failure: closes with code 4401 and returns None (DO NOT raise HTTPException here).
    On success: attaches identity to websocket.state.identity and returns it.
    """
    # Normalize WS headers and query
    headers = _normalize_headers(websocket.headers)
    query = dict(websocket.query_params)

    try:
        identity = _choose_identity(headers, query)
        websocket.state.identity = identity
        return identity
    except HTTPException:
        # Close gracefully with an "Unauthorized" close code for WS clients
        try:
            await websocket.close(code=4401)
        except Exception:
            pass
        return None
