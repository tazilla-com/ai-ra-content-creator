# File: workshops_ai/app/security.py
# Purpose: Unified authentication logic for HTTP and WebSocket.
# - Reads exclusively from central settings (get_settings()).
# - Supports API Key (X-API-Key / ?api_key) and JWT (Authorization: Bearer ...).
# - Shared core → thin wrappers for FastAPI HTTP dependency and WS handshake.
#
# Notes in this revision:
#   • All header lookups are case-insensitive (keys normalized to lowercase).
#   • API key is accepted from the "x-api-key" header OR the "api_key" query param.
#   • Bearer token is extracted from the lowercase "authorization" header.
#
# Requirements:
#   pip install "PyJWT>=2"
#
# Usage:
#   HTTP (router-level):
#     from fastapi import APIRouter, Depends
#     from workshops_ai.app.security import require_auth
#     router = APIRouter(dependencies=[Depends(require_auth)])
#
#   WebSocket:
#     from workshops_ai.app.security import verify_ws_auth
#     await verify_ws_auth(websocket)   # call before accepting messages

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import jwt  # PyJWT
from fastapi import HTTPException, Request, status
from fastapi.websockets import WebSocket

from workshops_ai.settings import get_settings

logger = logging.getLogger("workshops_ai.security")


# ----------------------------- Core helpers ---------------------------------

def _extract_api_key(headers: Dict[str, str], query: Dict[str, str]) -> Optional[str]:
    """
    API key is accepted either via:
      - HTTP header: X-API-Key (any capitalization) → normalized to 'x-api-key'
      - Query param: ?api_key=...
    """
    return headers.get("x-api-key") or query.get("api_key")


def _validate_api_key(api_key: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate the provided API key against settings.
    Returns an identity dict on success, or None when API-key auth is disabled.
    Raises 401 when API-key auth is enabled but no keys are configured.
    """
    s = get_settings().security
    if not s.enable_api_key:
        return None
    if not s.api_keys:
        # API-key mode is enabled but no keys configured → deny by design
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key authentication is enabled, but no keys are configured.",
        )
    if api_key and api_key in s.api_keys:
        # Do not return the raw key; return a simple hash surrogate
        return {"sub": "apikey", "method": "api_key", "key_hash": hash(api_key)}
    return None


def _extract_bearer(headers: Dict[str, str]) -> Optional[str]:
    """
    Extract "Bearer <token>" from the Authorization header.
    Header keys are expected to be lowercase already.
    """
    auth = headers.get("authorization")
    if not auth or not auth.startswith("Bearer "):
        return None
    return auth.split(" ", 1)[1].strip() or None


def _validate_jwt(token: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Validate JWT based on settings (algorithm/secret/issuer/audience).
    Returns an identity dict on success, or None when JWT auth is disabled.
    """
    s = get_settings().security
    if not s.enable_jwt:
        return None
    if not token:
        return None

    options = {
        "verify_signature": True,
        "verify_exp": True,
        "verify_aud": s.jwt_audience is not None,
        "verify_iss": s.jwt_issuer is not None,
    }

    try:
        claims = jwt.decode(
            token,
            s.jwt_secret,                 # HS* secret or RS*/ES* public key
            algorithms=[s.jwt_algorithm],
            audience=s.jwt_audience,
            issuer=s.jwt_issuer,
            options=options,
        )
        sub = claims.get("sub") or claims.get("uid") or "jwt"
        return {"sub": sub, "method": "jwt", "claims": claims}
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="JWT expired") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid JWT") from e


def _headers_from_request(request: Request) -> Dict[str, str]:
    """
    Build a case-insensitive header dict by normalizing keys to lowercase.
    Starlette exposes a case-insensitive mapping, but we normalize for consistency.
    """
    return {str(k).lower(): str(v) for k, v in request.headers.items()}


def _headers_from_websocket(ws: WebSocket) -> Dict[str, str]:
    """
    For WebSocket, Starlette stores raw header tuples (bytes).
    Normalize keys to lowercase and decode to str.
    """
    return {k.decode().lower(): v.decode() for k, v in ws.headers.raw}


def _query_from_websocket(ws: WebSocket) -> Dict[str, str]:
    """Extract query params as a plain dict."""
    return dict(ws.query_params)


def _choose_identity(headers: Dict[str, str], query: Dict[str, str]) -> Dict[str, Any]:
    """
    Try API-key first (if enabled), then JWT (if enabled).
    If neither method is enabled → 401.
    If methods are enabled but none succeed → 401 Unauthorized.
    """
    s = get_settings().security

    if s.enable_api_key:
        ident = _validate_api_key(_extract_api_key(headers, query))
        if ident:
            return ident

    if s.enable_jwt:
        ident = _validate_jwt(_extract_bearer(headers))
        if ident:
            return ident

    if not s.enable_api_key and not s.enable_jwt:
        # Fail-closed: at least one auth mechanism must be enabled
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No authentication method is enabled.",
        )

    # Methods are enabled but nothing validated
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")


# ------------------------------ Public API ----------------------------------

async def require_auth(request: Request) -> Dict[str, Any]:
    """
    FastAPI dependency for HTTP endpoints.
    Returns an identity dict (e.g., {'sub': '...', 'method': 'api_key'|'jwt', ...}).
    """
    headers = _headers_from_request(request)  # lowercase keys
    query = {str(k): str(v) for k, v in request.query_params.items()}
    identity = _choose_identity(headers, query)
    # Optionally attach identity to request.state for downstream usage
    request.state.identity = identity
    return identity


async def verify_ws_auth(websocket: WebSocket) -> Dict[str, Any]:
    """
    Call inside WS handlers *before* accepting the connection.
    Extracts auth from headers/query, validates via central settings.
    On failure: closes with 4401 Unauthorized and re-raises HTTPException.
    """
    headers = _headers_from_websocket(websocket)  # lowercase keys
    query = _query_from_websocket(websocket)

    try:
        identity = _choose_identity(headers, query)
        websocket.state.identity = identity
        return identity
    except HTTPException as exc:
        # Close without sending payload; dedicated code 4401 for clarity
        await websocket.close(code=4401)
        raise exc
