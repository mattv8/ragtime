"""
RAG API with Tool Calling - OpenAI-Compatible FastAPI Server
=============================================================

Main application entry point.

Usage:
    uvicorn ragtime.main:app --host 0.0.0.0 --port 8000 --reload

    Or run directly:
    python -m ragtime.main
"""

import asyncio
import html
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import quote, urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ragtime import __version__
from ragtime.api import router
from ragtime.api.auth import (
    AUTH_CODE_EXPIRY,
    _auth_codes,
    _cleanup_expired_auth_codes,
    authenticate,
)
from ragtime.api.auth import oauth2_token as _oauth2_token_handler
from ragtime.api.auth import router as auth_router
from ragtime.api.auth import validate_redirect_uri
from ragtime.config import settings
from ragtime.core.app_settings import get_app_settings
from ragtime.core.auth import (
    create_access_token,
    create_session,
    get_browser_matched_origin,
    get_external_origin,
    validate_session,
)
from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.core.logging import setup_logging
from ragtime.core.rate_limit import LOGIN_RATE_LIMIT, SHARE_AUTH_RATE_LIMIT, limiter
from ragtime.core.ssl import setup_ssl
from ragtime.indexer.background_tasks import background_task_service
from ragtime.indexer.chat_attachments import cleanup_expired_chat_attachments
from ragtime.indexer.chunking import shutdown_process_pool
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.pdm_service import pdm_indexer
from ragtime.indexer.repository import repository
from ragtime.indexer.routes import ASSETS_DIR as INDEXER_ASSETS_DIR
from ragtime.indexer.routes import DIST_DIR
from ragtime.indexer.routes import router as indexer_router
from ragtime.indexer.schema_service import schema_indexer
from ragtime.indexer.service import indexer
from ragtime.mcp.config_routes import default_filter_router as mcp_default_filter_router
from ragtime.mcp.config_routes import router as mcp_config_router
from ragtime.mcp.oauth import (
    build_authorization_server_metadata,
    build_protected_resource_metadata,
)
from ragtime.mcp.routes import get_mcp_routes, mcp_lifespan_manager
from ragtime.mcp.routes import router as mcp_router
from ragtime.rag import rag
from ragtime.userspace.preview_host import PreviewHostDispatchMiddleware
from ragtime.userspace.routes import router as userspace_router
from ragtime.userspace.runtime_routes import (
    _proxy_websocket_request,
    _sanitize_preview_query,
    _to_websocket_url,
)
from ragtime.userspace.runtime_routes import router as userspace_runtime_router
from ragtime.userspace.runtime_service import userspace_runtime_service
from ragtime.userspace.service import userspace_service
from ragtime.userspace.share_auth import (
    clear_share_auth_cookie,
    set_share_auth_cookie,
    share_auth_token_from_request,
)

# Import indexer routes (always available now that it's part of ragtime)
# Import MCP routes and transport for HTTP API access
# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging("rag_api")

_SHARE_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


def _oauth_request_context(request: Request) -> str:
    """Return sanitized request context for OAuth diagnostics."""
    headers = request.headers
    fields = {
        "client": request.client.host if request.client else "unknown",
        "user_agent": headers.get("user-agent") or "-",
        "origin": headers.get("origin") or "-",
        "referer": headers.get("referer") or "-",
        "sec_fetch_mode": headers.get("sec-fetch-mode") or "-",
        "sec_fetch_site": headers.get("sec-fetch-site") or "-",
        "query_keys": ",".join(sorted(request.query_params.keys())) or "-",
    }
    return ", ".join(f"{key}={value}" for key, value in fields.items())


def _validate_ssl_certificates() -> None:
    """Validate SSL certificates if HTTPS is enabled."""
    if not settings.enable_https:
        return

    # Run SSL validation - this logs any errors/warnings
    result = setup_ssl(
        cert_file=settings.ssl_cert_file,
        key_file=settings.ssl_key_file,
        auto_generate=False,  # Don't generate - entrypoint.sh handles that
    )

    if result is None:
        logger.warning(
            "SSL validation failed - uvicorn may fail to start with HTTPS. "
            "Check the SSL errors above."
        )


# Global ThreadPoolExecutor for I/O-bound operations
# Used by asyncio.to_thread() for file operations, encoding detection, etc.
_io_thread_pool: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan handler for startup/shutdown."""
    global _io_thread_pool

    logger.info(f"Starting RAG API v{__version__}")

    # Configure a shared ThreadPoolExecutor for I/O-bound operations
    # This improves thread reuse and allows concurrent file operations
    cpu_count = os.cpu_count() or 4
    max_io_workers = min(64, cpu_count * 4)  # 4 threads per core, max 64
    _io_thread_pool = ThreadPoolExecutor(
        max_workers=max_io_workers, thread_name_prefix="ragtime-io-"
    )
    # Set as the default executor for asyncio.to_thread() calls
    loop = asyncio.get_event_loop()
    loop.set_default_executor(_io_thread_pool)
    logger.info(f"Initialized I/O thread pool with {max_io_workers} workers")

    # Validate SSL certificates if HTTPS is enabled
    _validate_ssl_certificates()

    # Connect to database
    await connect_db()

    # Initialize RAG components
    await rag.initialize()

    # Recover any interrupted indexing jobs (survives hot-reloads)
    recovered = await indexer.recover_interrupted_jobs()
    if recovered > 0:
        logger.info(f"Recovered {recovered} interrupted indexing job(s)")

    # Clean up any orphaned filesystem indexing jobs from previous runs
    await filesystem_indexer._cleanup_stale_jobs()

    # Clean up any orphaned schema indexing jobs from previous runs
    await schema_indexer._cleanup_stale_jobs()

    # Clean up any orphaned PDM indexing jobs from previous runs
    await pdm_indexer._cleanup_stale_jobs()

    # Discover orphan indexes (FAISS files without database metadata)
    discovered = await indexer.discover_orphan_indexes()
    if discovered > 0:
        logger.info(f"Discovered {discovered} orphan index(es) on disk")

    # Garbage collect orphaned pgvector embeddings
    gc_results = await repository.cleanup_orphaned_embeddings()
    gc_total = sum(gc_results.values())
    if gc_total > 0:
        logger.info(f"Garbage collected {gc_total} orphaned embedding(s)")

    # Remove expired chat attachment uploads left from previous sessions
    await cleanup_expired_chat_attachments()

    # Start background task service for chat
    await background_task_service.start()

    # Backfill userspace Git ignore policy for existing workspaces in the background.
    userspace_service.schedule_startup_git_drift_reconciliation()
    userspace_service.schedule_workspace_mount_watch()
    userspace_service.schedule_workspace_scm_watch()

    # Start MCP session manager (enable/disable checked at request time)
    async with mcp_lifespan_manager():
        yield

    # Cleanup - cancel the startup Git policy reconciliation task
    await userspace_service.shutdown_git_drift_reconciliation()
    await userspace_service.shutdown_workspace_mount_watch()
    await userspace_service.shutdown_workspace_scm_watch()

    # Cleanup - stop background services before disconnecting DB
    await background_task_service.stop()

    # Shutdown chunking process pool
    shutdown_process_pool()

    # Shutdown I/O thread pool
    if _io_thread_pool is not None:
        _io_thread_pool.shutdown(wait=False)
        logger.info("I/O thread pool shut down")

    # Stop PDM indexer tasks
    await pdm_indexer.shutdown()

    # Stop schema indexer (schema_indexer imported above)
    await schema_indexer.shutdown()

    # Stop filesystem indexer tasks (filesystem_indexer imported above)
    await filesystem_indexer.shutdown()

    await disconnect_db()
    logger.info("Shutting down RAG API")


# Create FastAPI app
# Security: the interactive docs and OpenAPI schema enumerate admin-only and
# maintenance routes (tool testing, filesystem browsing, MCP admin, etc.) that
# unauthenticated callers should not be able to inventory. In production the
# docs/schema endpoints are disabled; enable them only when DEBUG_MODE=true.
_docs_enabled = bool(getattr(settings, "debug_mode", False))
app = FastAPI(
    title="Ragtime RAG API",
    description="RAG + Tool Calling API for business intelligence queries",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs" if _docs_enabled else None,
    redoc_url="/redoc" if _docs_enabled else None,
    openapi_url="/openapi.json" if _docs_enabled else None,
)


# Exception handler for validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Log validation errors to help debug malformed requests.

    Security: headers and body are only logged in debug mode to avoid leaking
    Authorization tokens, cookies, or other secrets to container logs.
    The response never echoes back the request body.
    """
    logger.error(
        f"Validation error on {request.method} {request.url.path}: {exc.errors()}"
    )

    if settings.debug_mode:
        # Only log body/headers in debug mode; redact sensitive headers
        try:
            body = await request.body()
            body_str = body.decode("utf-8")[:500]
        except Exception:
            body_str = "<unable to read body>"
        _sensitive_header_keys = {
            "authorization",
            "cookie",
            "x-api-key",
            "proxy-authorization",
        }
        safe_headers = {
            k: ("***" if k.lower() in _sensitive_header_keys else v)
            for k, v in request.headers.items()
        }
        logger.debug(f"Request body (debug): {body_str}")
        logger.debug(f"Request headers (debug): {safe_headers}")

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# CORS middleware
# Note: allow_origins=["*"] with allow_credentials=True is problematic per CORS spec.
# When ALLOWED_ORIGINS is unset, default to loopback-only origins; dev clients
# on arbitrary localhost ports are still matched via allow_origin_regex below.
# For production, set ALLOWED_ORIGINS to specific origins like
# "https://ragtime.example.com".
if settings.allowed_origins == "*":
    origins = ["*"]
    logger.warning(
        "CORS: ALLOWED_ORIGINS='*' with credentials is permissive. "
        "For production, set ALLOWED_ORIGINS to specific trusted origins."
    )
else:
    origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
    # Always allow loopback addresses for OAuth callbacks (RFC 8252)
    # These are safe because they only allow requests from the local machine
    loopback_origins = ["http://127.0.0.1", "http://localhost", "http://[::1]"]
    for lb in loopback_origins:
        if lb not in origins:
            origins.append(lb)
    if not settings.allowed_origins:
        logger.info(
            "CORS: ALLOWED_ORIGINS unset; defaulting to loopback-only origins. "
            "Set ALLOWED_ORIGINS to an explicit list for non-loopback deployments."
        )
    else:
        logger.info(f"CORS: Allowing origins: {origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Allow any port on loopback for OAuth callbacks
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
)
app.add_middleware(PreviewHostDispatchMiddleware)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include API routes
app.include_router(router)

# Include auth routes
app.include_router(auth_router)

# Include indexer routes
app.include_router(indexer_router)
app.include_router(userspace_router)
app.include_router(userspace_runtime_router)
# Mount static files for indexer UI assets at root
# Ensure assets directory exists so static file mount always works
INDEXER_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/assets", StaticFiles(directory=INDEXER_ASSETS_DIR), name="indexer_ui_assets"
)
logger.info("Indexer API enabled at /indexes, UI served at root (/)")

# Include MCP routes (enabled/disabled controlled via database setting)
app.include_router(mcp_router)  # Debug endpoints at /mcp-debug/*
app.include_router(
    mcp_default_filter_router
)  # Default route filters at /mcp-routes/default-filters/*
app.include_router(mcp_config_router)  # MCP route configuration at /mcp-routes/*
# Add all MCP routes (default /mcp and custom /mcp/{route_path})
for route in get_mcp_routes():
    app.routes.append(route)
logger.info("MCP routes registered (enable/disable via Settings UI)")


# =============================================================================
# OAuth2 Authorization Code Flow with PKCE (for VS Code MCP)
# =============================================================================

# Simple in-memory storage for dynamically registered clients
# Limited to prevent memory exhaustion attacks
_registered_clients: dict[str, dict] = {}
MAX_REGISTERED_CLIENTS = 1000  # Prevent memory exhaustion


@app.post("/register", include_in_schema=False)
@limiter.limit(LOGIN_RATE_LIMIT)
async def register_client(request: Request):
    """
    OAuth2 Dynamic Client Registration (RFC 7591).

    Allows MCP clients to register themselves and obtain a client_id.
    Rate limited and capped to prevent abuse.
    """

    # Prevent memory exhaustion
    if len(_registered_clients) >= MAX_REGISTERED_CLIENTS:
        # Evict oldest 10% of clients
        to_remove = list(_registered_clients.keys())[: MAX_REGISTERED_CLIENTS // 10]
        for key in to_remove:
            del _registered_clients[key]
        logger.warning(
            f"OAuth client registry full, evicted {len(to_remove)} oldest clients"
        )

    try:
        body = await request.json()
    except Exception:
        body = {}

    # Generate a unique client_id
    client_id = str(uuid.uuid4())

    # Store client metadata
    client_data = {
        "client_id": client_id,
        "client_name": body.get("client_name", "Unknown Client"),
        "redirect_uris": body.get("redirect_uris", []),
        "grant_types": body.get("grant_types", ["authorization_code"]),
        "response_types": body.get("response_types", ["code"]),
        "token_endpoint_auth_method": "none",  # Public client
    }
    _registered_clients[client_id] = client_data

    logger.info(
        f"Dynamically registered client: {client_data['client_name']} with id {client_id}"
    )

    # Return registration response per RFC 7591
    return {
        "client_id": client_id,
        "client_id_issued_at": int(__import__("time").time()),
        "client_name": client_data["client_name"],
        "redirect_uris": client_data["redirect_uris"],
        "grant_types": client_data["grant_types"],
        "response_types": client_data["response_types"],
        "token_endpoint_auth_method": "none",
    }


@app.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_discovery(request: Request):
    """
    OAuth2 Authorization Server Metadata (RFC 8414).

    Provides OAuth2 discovery for MCP clients (VS Code, JetBrains, CLI tools, etc.)
    to automatically configure authorization endpoints.
    """
    # Use the browser-visible origin so OAuth metadata stays HTTPS-correct
    # when Ragtime runs behind a TLS-terminating reverse proxy.
    base_url = get_browser_matched_origin(request)

    # Security warning: OAuth should use HTTPS in production
    if base_url.startswith("http://") and not settings.debug_mode:
        logger.warning(
            "OAuth metadata served over HTTP. Use HTTPS in production for security."
        )

    logger.info(
        f"OAuth authorization server metadata requested from {request.client.host if request.client else 'unknown'}"
    )

    app_settings = await get_app_settings()
    if (
        app_settings.get("mcp_default_route_auth")
        and app_settings.get("mcp_default_route_auth_method") == "client_credentials"
    ):
        return build_authorization_server_metadata(base_url, None)

    return {
        "issuer": base_url,
        "authorization_endpoint": f"{base_url}/authorize",
        "token_endpoint": f"{base_url}/token",
        # Dynamic Client Registration (RFC 7591) - fallback for clients without pre-registration
        "registration_endpoint": f"{base_url}/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "password"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        # Client ID Metadata Documents support (draft-ietf-oauth-client-id-metadata-document-00)
        # This tells MCP clients they can use HTTPS URLs as client_ids without pre-registration
        "client_id_metadata_document_supported": True,
        # Additional standard fields for broader client compatibility
        "scopes_supported": [],
        "response_modes_supported": ["query"],
        "service_documentation": f"{base_url}/docs",
    }


@app.get("/.well-known/oauth-protected-resource", include_in_schema=False)
@app.get("/.well-known/oauth-protected-resource/{path:path}", include_in_schema=False)
async def oauth_protected_resource(request: Request, path: str = ""):
    """
    OAuth2 Protected Resource Metadata (RFC 9728).

    Tells MCP clients which authorization server protects MCP resources.
    Supports both default /mcp route and custom routes like /mcp/my_toolset.

    Examples:
        /.well-known/oauth-protected-resource       -> resource: /mcp
        /.well-known/oauth-protected-resource/mcp   -> resource: /mcp
        /.well-known/oauth-protected-resource/mcp/custom -> resource: /mcp/custom
    """
    # Use the browser-visible origin so protected-resource metadata points
    # clients at the same public HTTPS origin they are already using.
    base_url = get_browser_matched_origin(request)

    normalized_path = (path or "").strip("/")
    route_path: str | None = None
    if normalized_path and normalized_path != "mcp":
        route_path = (
            normalized_path[len("mcp/") :]
            if normalized_path.startswith("mcp/")
            else normalized_path
        )

    app_settings = await get_app_settings()
    if route_path is None:
        if (
            app_settings.get("mcp_default_route_auth")
            and app_settings.get("mcp_default_route_auth_method")
            == "client_credentials"
        ):
            logger.debug(
                "OAuth protected-resource discovery resolved default /mcp route to client_credentials metadata"
            )
            return build_protected_resource_metadata(base_url, None)
    else:
        db = await get_db()
        route = await db.mcprouteconfig.find_unique(where={"routePath": route_path})
        if (
            route
            and route.enabled
            and route.requireAuth
            and (route.authMethod or "password") == "client_credentials"
        ):
            logger.debug(
                "OAuth protected-resource discovery resolved custom route /mcp/%s to client_credentials metadata",
                route_path,
            )
            return build_protected_resource_metadata(base_url, route_path)

    # Determine the resource path from the suffix
    # If path is empty or just "mcp", use default /mcp
    # Otherwise, use the full path for custom routes
    if not path or path == "mcp":
        resource_path = "/mcp"
    elif path.startswith("mcp/"):
        resource_path = f"/{path}"
    else:
        resource_path = f"/mcp/{path}"

    return {
        "resource": f"{base_url}{resource_path}",
        "authorization_servers": [base_url],
        "scopes_supported": [],
        "bearer_methods_supported": ["header"],
    }


@app.get("/authorize", include_in_schema=False)
async def authorize_get(
    request: Request,
    client_id: str = Query(..., description="Client identifier"),
    redirect_uri: str = Query(..., description="Redirect URI"),
    response_type: str = Query(..., description="Must be 'code'"),
    code_challenge: str = Query(..., description="PKCE code challenge"),
    code_challenge_method: str = Query(
        "S256", description="PKCE method (must be S256)"
    ),
    state: Optional[str] = Query(None, description="CSRF state parameter"),
):
    """
    OAuth2 Authorization Endpoint (RFC 6749) - GET request.

    Initiates the Authorization Code flow with PKCE (RFC 7636).
    Redirects to the web UI for user authentication.

    Supports MCP clients from any IDE (VS Code, JetBrains, Cursor, etc.)
    that implement OAuth2 Authorization Code flow with PKCE.
    """
    logger.info(
        f"OAuth authorize request: client_id={client_id}, redirect_uri={redirect_uri}"
    )

    # Validate redirect_uri (security: prevent open redirect attacks)
    # Only loopback addresses allowed per RFC 8252 for native apps
    if not validate_redirect_uri(redirect_uri):
        return HTMLResponse(
            content="<h1>Error</h1><p>Invalid redirect_uri. Only loopback addresses (127.0.0.1, localhost) are allowed for native OAuth2 clients per RFC 8252.</p>",
            status_code=400,
        )

    if response_type != "code":
        return HTMLResponse(
            content=f"<h1>Error</h1><p>Unsupported response_type: {response_type}</p>",
            status_code=400,
        )

    if code_challenge_method != "S256":
        return HTMLResponse(
            content=f"<h1>Error</h1><p>Unsupported code_challenge_method: {code_challenge_method}. Only S256 is supported.</p>",
            status_code=400,
        )

    # Redirect to frontend with OAuth params preserved
    # The frontend will detect these params and show the OAuth login form
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": response_type,
        "code_challenge": code_challenge,
        "code_challenge_method": code_challenge_method,
    }
    if state:
        params["state"] = state

    return RedirectResponse(url=f"/?{urlencode(params)}", status_code=302)


@app.post("/authorize", include_in_schema=False)
@limiter.limit(LOGIN_RATE_LIMIT)
async def authorize_post(
    request: Request,
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    response_type: str = Form(...),
    code_challenge: str = Form(...),
    code_challenge_method: str = Form("S256"),
    state: str = Form(""),
    username: str = Form(...),
    password: str = Form(...),
):
    """
    OAuth2 Authorization Endpoint - POST request handles login.
    Returns JSON with redirect_url on success (frontend navigates).
    Returns JSON error on failure.

    Rate limited to prevent brute-force attacks on OAuth login.
    """
    # Validate redirect_uri (security: prevent open redirect attacks)
    if not validate_redirect_uri(redirect_uri):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid redirect_uri. Only loopback addresses are allowed."
            },
        )

    # Authenticate user
    result = await authenticate(username, password)

    if not result.success or not result.user_id:
        # Return JSON error for frontend to display
        return JSONResponse(
            status_code=401, content={"error": result.error or "Authentication failed"}
        )

    # Cleanup expired codes
    _cleanup_expired_auth_codes()

    # Generate authorization code
    code = secrets.token_urlsafe(32)

    # Store authorization code with PKCE challenge
    _auth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "user_id": result.user_id,
        "username": result.username,
        "role": result.role,
        "expires": time.time() + AUTH_CODE_EXPIRY,
    }

    logger.info(f"OAuth2 authorization code issued for '{result.username}'")

    # Create session for the user (so they stay logged in to Ragtime)
    token = create_access_token(result.user_id, result.username, result.role)

    await create_session(
        user_id=result.user_id,
        token=token,
        user_agent=request.headers.get("User-Agent"),
        ip_address=request.client.host if request.client else None,
    )

    # Build redirect URL with authorization code
    params = {"code": code}
    if state:
        params["state"] = state

    redirect_url = f"{redirect_uri}?{urlencode(params)}"

    # Return JSON response - frontend will navigate to redirect_url
    # Cannot use RedirectResponse because fetch would try to follow it
    # and the OAuth callback server doesn't have CORS headers
    response = JSONResponse(content={"redirect_url": redirect_url})

    # Set session cookie so user stays logged in
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        httponly=settings.session_cookie_httponly,
        secure=settings.session_cookie_secure,
        samesite=settings.session_cookie_samesite,
        max_age=settings.jwt_expire_hours * 3600,
    )

    return response


@app.post("/authorize/session", include_in_schema=False)
async def authorize_with_session(
    request: Request,
    client_id: str = Form(...),
    redirect_uri: str = Form(...),
    response_type: str = Form(...),
    code_challenge: str = Form(...),
    code_challenge_method: str = Form("S256"),
    state: str = Form(""),
):
    """
    OAuth2 Authorization using existing session.

    For already-authenticated users, this endpoint issues an auth code
    using their session cookie instead of requiring username/password.
    """
    # Validate redirect_uri (security: prevent open redirect attacks)
    if not validate_redirect_uri(redirect_uri):
        return JSONResponse(
            status_code=400,
            content={
                "error": "Invalid redirect_uri. Only loopback addresses are allowed."
            },
        )

    # Extract session token from cookie
    session_cookie = request.cookies.get("ragtime_session")
    if not session_cookie:
        return JSONResponse(
            status_code=401,
            content={"error": "Not authenticated", "require_login": True},
        )

    # Validate session
    token_data = await validate_session(session_cookie)
    if not token_data:
        return JSONResponse(
            status_code=401, content={"error": "Session expired", "require_login": True}
        )

    # Get user from database
    db = await get_db()
    user = await db.user.find_unique(where={"id": token_data.user_id})
    if not user:
        return JSONResponse(
            status_code=401, content={"error": "User not found", "require_login": True}
        )

    # Cleanup expired codes
    _cleanup_expired_auth_codes()

    # Generate authorization code
    code = secrets.token_urlsafe(32)

    # Store authorization code with PKCE challenge
    _auth_codes[code] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "code_challenge": code_challenge,
        "user_id": user.id,
        "username": user.username,
        "role": user.role,
        "expires": time.time() + AUTH_CODE_EXPIRY,
    }

    logger.info(
        "OAuth2 authorization code issued for '%s' via session "
        "(client_id=%s, redirect_uri=%s, %s)"
        % (
            user.username,
            client_id,
            redirect_uri,
            _oauth_request_context(request),
        )
    )

    # Build redirect URL with authorization code
    params = {"code": code}
    if state:
        params["state"] = state

    redirect_url = f"{redirect_uri}?{urlencode(params)}"

    # Return JSON with redirect URL (frontend will navigate)
    return JSONResponse(status_code=200, content={"redirect_url": redirect_url})


# =============================================================================
# OAuth2 Token Endpoint (RFC 6749 Section 3.2)
# =============================================================================


@app.get("/token", include_in_schema=False)
async def token_endpoint_get(request: Request):
    """Log unexpected browser-style token requests without changing behavior."""
    logger.warning(
        "Unexpected OAuth token GET request received (%s)"
        % _oauth_request_context(request)
    )
    response = JSONResponse(
        status_code=405,
        content={
            "detail": "Method Not Allowed. OAuth token exchange requires POST.",
        },
    )
    response.headers["Allow"] = "POST"
    return response


@app.post("/token", include_in_schema=False)
async def token_endpoint(
    request: Request,
    grant_type: str = Form(...),
    username: Optional[str] = Form(default=None),
    password: Optional[str] = Form(default=None),
    code: Optional[str] = Form(default=None),
    code_verifier: Optional[str] = Form(default=None),
    redirect_uri: Optional[str] = Form(default=None),
    client_id: Optional[str] = Form(default=None),
    scope: Optional[str] = Form(default=None),
):
    """
    OAuth2 Token Endpoint (RFC 6749 Section 3.2).

    Exchanges authorization codes for access tokens (authorization_code grant)
    or authenticates directly with credentials (password grant).

    Standard endpoint location at /token for OAuth2 client compatibility.
    Also available at /auth/oauth2/token for API consistency.
    """
    if grant_type == "authorization_code":
        logger.info(
            "OAuth2 token exchange requested "
            "(client_id=%s, redirect_uri=%s, %s)"
            % (client_id or "-", redirect_uri or "-", _oauth_request_context(request))
        )

    return await _oauth2_token_handler(
        request=request,
        grant_type=grant_type,
        username=username,
        password=password,
        code=code,
        code_verifier=code_verifier,
        redirect_uri=redirect_uri,
        client_id=client_id,
        scope=scope,
    )


def _share_reserved_roots() -> set[str]:
    return {
        "auth",
        "assets",
        "docs",
        "indexes",
        "mcp",
        "mcp-debug",
        "mcp-routes",
        "openapi.json",
        "redoc",
        "shared",
        "v1",
    }


def _render_share_unlock_prompt(
    title: str,
    form_action: str,
    subtitle: str | None = None,
    owner_label: str | None = None,
    error: str | None = None,
    next_target: str | None = None,
) -> str:
    safe_error = html.escape(error) if error else ""
    safe_form_action = html.escape(form_action, quote=True)
    safe_next_target = html.escape(next_target, quote=True) if next_target else ""
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle) if subtitle else ""
    safe_owner_label = html.escape(owner_label) if owner_label else ""
    subtitle_block = (
        f"<p style='margin:0 0 6px 0;color:#e2e8f0;font-size:15px;font-weight:600'>{safe_subtitle}</p>"
        if safe_subtitle
        else ""
    )
    owner_block = (
        f"<p style='margin:0 0 14px 0;color:#94a3b8;font-size:13px'>{safe_owner_label}</p>"
        if safe_owner_label
        else ""
    )
    error_block = (
        f"<p style='color:#fca5a5;margin:0 0 12px 0;font-size:14px'>{safe_error}</p>"
        if safe_error
        else ""
    )
    next_block = (
        f"<input type='hidden' name='next' value='{safe_next_target}'>"
        if safe_next_target
        else ""
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Shared Workspace</title></head><body style='margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0f172a;color:#e2e8f0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif'>"
        f"<form method='post' action='{safe_form_action}' style='width:min(92vw,360px);padding:20px;border:1px solid #334155;border-radius:12px;background:#111827'>"
        f"<h1 style='font-size:18px;margin:0 0 10px 0'>{safe_title}</h1>{subtitle_block}{owner_block}"
        f"{error_block}"
        f"{next_block}"
        "<label for='share_password' style='display:block;margin-bottom:8px;font-size:13px'>Password</label>"
        "<input id='share_password' name='share_password' type='password' required autofocus autocomplete='current-password' style='width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid #334155;background:#0b1220;color:#e2e8f0'>"
        "<button type='submit' style='margin-top:12px;width:100%;padding:10px 12px;border-radius:8px;border:1px solid #334155;background:#1d4ed8;color:#fff;cursor:pointer'>Continue</button>"
        "</form></body></html>"
    )


def _render_share_password_prompt(
    workspace_name: str | None,
    form_action: str,
    owner_label: str | None = None,
    error: str | None = None,
    next_target: str | None = None,
) -> str:
    return _render_share_unlock_prompt(
        "Unlock shared workspace",
        form_action,
        workspace_name or "Shared workspace",
        owner_label,
        error,
        next_target,
    )


def _render_share_token_password_prompt(
    workspace_name: str | None,
    form_action: str,
    error: str | None = None,
    next_target: str | None = None,
) -> str:
    return _render_share_unlock_prompt(
        "Unlock shared workspace",
        form_action,
        workspace_name or "Shared workspace",
        error=error,
        next_target=next_target,
    )


def _share_slug_authorize_path(owner_username: str, share_slug: str) -> str:
    return f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}/authorize"


def _share_token_authorize_path(share_token: str) -> str:
    return f"/shared/{quote(share_token, safe='')}/authorize"


def _share_request_target(request: Request) -> str:
    return request.url.path + (f"?{request.url.query}" if request.url.query else "")


def _normalize_share_next_target(next_target: str | None, fallback: str) -> str:
    candidate = str(next_target or "").strip()
    if not candidate or not candidate.startswith("/") or candidate.startswith("//"):
        return fallback
    return candidate


def _share_cookie_max_age(expires_at: datetime | None) -> int:
    if expires_at is None:
        return 60 * 30
    return max(60, int((expires_at - datetime.now(timezone.utc)).total_seconds()))


def _share_form_value(form: Any, key: str) -> str:
    if form is None:
        return ""
    try:
        value = form.get(key, "")
    except Exception:
        return ""
    return str(value or "").strip()


async def _share_current_user_from_request(request: Request):
    session_cookie = request.cookies.get("ragtime_session")
    if not session_cookie:
        return None
    token_data = await validate_session(session_cookie)
    if not token_data:
        return None
    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


async def _share_current_user_from_websocket(websocket: WebSocket):
    session_cookie = websocket.cookies.get("ragtime_session")
    if not session_cookie:
        return None
    token_data = await validate_session(session_cookie)
    if not token_data:
        return None
    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


async def _proxy_shared_workspace_websocket(
    websocket: WebSocket,
    *,
    workspace_id: str,
    path: str,
) -> None:
    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path or "/",
        query=_sanitize_preview_query(websocket.url.query),
    )
    await _proxy_websocket_request(websocket, _to_websocket_url(upstream_url))


def _shared_chat_shell_response() -> Response:
    dist_index = DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(dist_index, media_type="text/html")
    return JSONResponse(
        status_code=503,
        content={"detail": "Frontend build not available for shared conversation"},
    )


async def _shared_launch_redirect_by_slug(
    owner_username: str,
    share_slug: str,
    request: Request,
    path: str,
):
    if owner_username in _share_reserved_roots():
        raise HTTPException(status_code=404, detail="Not found")

    current_user = await _share_current_user_from_request(request)
    share_auth_token = share_auth_token_from_request(
        request.headers,
        request.cookies,
        owner_username=owner_username,
        share_slug=share_slug,
    )
    workspace_name, owner_display_name = (
        await userspace_service.get_share_prompt_metadata_by_slug(
            owner_username,
            share_slug,
        )
    )
    target_type = await userspace_service.resolve_public_share_target_by_slug(
        owner_username,
        share_slug,
    )
    if target_type == "unknown":
        raise HTTPException(status_code=404, detail="Not found")

    if target_type == "conversation":
        try:
            await userspace_service.authorize_shared_conversation_access_by_slug(
                owner_username,
                share_slug,
                current_user=current_user,
                share_auth_token=share_auth_token,
            )
        except HTTPException as exc:
            detail = str(exc.detail).lower() if isinstance(exc.detail, str) else ""
            is_password_error = exc.status_code == 401 and (
                "password required" in detail or "invalid password" in detail
            )
            if is_password_error and request.method == "GET":
                response = HTMLResponse(
                    _render_share_password_prompt(
                        workspace_name,
                        _share_slug_authorize_path(owner_username, share_slug),
                        owner_display_name or owner_username,
                        "Invalid password" if "invalid password" in detail else None,
                        _share_request_target(request),
                    )
                )
                clear_share_auth_cookie(
                    response,
                    owner_username=owner_username,
                    share_slug=share_slug,
                )
                return response
            if exc.status_code in {401, 403} and request.method == "GET":
                return _shared_chat_shell_response()
            raise
        return _shared_chat_shell_response()

    try:
        workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
            owner_username,
            share_slug,
            current_user=current_user,
            share_auth_token=share_auth_token,
        )
    except HTTPException as exc:
        detail = str(exc.detail).lower() if isinstance(exc.detail, str) else ""
        is_password_error = exc.status_code == 401 and (
            "password required" in detail or "invalid password" in detail
        )
        if is_password_error and request.method == "GET":
            response = HTMLResponse(
                _render_share_password_prompt(
                    workspace_name,
                    _share_slug_authorize_path(owner_username, share_slug),
                    owner_display_name or owner_username,
                    "Invalid password" if "invalid password" in detail else None,
                    _share_request_target(request),
                )
            )
            clear_share_auth_cookie(
                response,
                owner_username=owner_username,
                share_slug=share_slug,
            )
            return response
        raise

    # Redirect to preview origin subdomain.
    # Auth (password, group, etc.) was validated above by resolve_shared_workspace_id_by_slug.
    target_path = f"/{path}" if path else "/"
    query = _sanitize_preview_query(request.url.query)
    current_access_mode = await userspace_service.get_share_access_mode(workspace_id)
    launch = await userspace_runtime_service.issue_shared_preview_launch(
        workspace_id,
        control_plane_origin=get_external_origin(request),
        path=f"{target_path}?{query}" if query else target_path,
        owner_username=owner_username,
        share_slug=share_slug,
        subject_user_id=str(getattr(current_user, "id", "") or "") or None,
        share_access_mode=current_access_mode,
    )
    return RedirectResponse(url=launch.preview_url, status_code=302)


async def _shared_launch_redirect_by_token(
    share_token: str,
    request: Request,
    path: str,
):
    current_user = await _share_current_user_from_request(request)
    share_auth_token = share_auth_token_from_request(
        request.headers,
        request.cookies,
        share_token=share_token,
    )
    workspace_name, _ = await userspace_service.get_share_prompt_metadata_by_token(
        share_token,
    )
    target_type = await userspace_service.resolve_public_share_target_by_token(
        share_token
    )
    if target_type == "unknown":
        raise HTTPException(status_code=404, detail="Not found")

    if target_type == "conversation":
        try:
            await userspace_service.authorize_shared_conversation_access(
                share_token,
                current_user=current_user,
                share_auth_token=share_auth_token,
            )
        except HTTPException as exc:
            detail = str(exc.detail).lower() if isinstance(exc.detail, str) else ""
            is_password_error = exc.status_code == 401 and (
                "password required" in detail or "invalid password" in detail
            )
            if is_password_error and request.method == "GET":
                response = HTMLResponse(
                    _render_share_token_password_prompt(
                        workspace_name,
                        _share_token_authorize_path(share_token),
                        "Invalid password" if "invalid password" in detail else None,
                        _share_request_target(request),
                    )
                )
                clear_share_auth_cookie(
                    response,
                    share_token=share_token,
                )
                return response
            if exc.status_code in {401, 403} and request.method == "GET":
                return _shared_chat_shell_response()
            raise
        return _shared_chat_shell_response()

    try:
        workspace_id = await userspace_service.resolve_shared_workspace_id(
            share_token,
            current_user=current_user,
            share_auth_token=share_auth_token,
        )
    except HTTPException as exc:
        detail = str(exc.detail).lower() if isinstance(exc.detail, str) else ""
        is_password_error = exc.status_code == 401 and (
            "password required" in detail or "invalid password" in detail
        )
        if is_password_error and request.method == "GET":
            response = HTMLResponse(
                _render_share_token_password_prompt(
                    workspace_name,
                    _share_token_authorize_path(share_token),
                    "Invalid password" if "invalid password" in detail else None,
                    _share_request_target(request),
                )
            )
            clear_share_auth_cookie(
                response,
                share_token=share_token,
            )
            return response
        raise

    # Redirect to preview origin subdomain.
    # Auth (password, group, etc.) was validated above by resolve_shared_workspace_id.
    target_path = f"/{path}" if path else "/"
    query = _sanitize_preview_query(request.url.query)
    current_access_mode = await userspace_service.get_share_access_mode(workspace_id)
    launch = await userspace_runtime_service.issue_shared_preview_launch(
        workspace_id,
        control_plane_origin=get_external_origin(request),
        path=f"{target_path}?{query}" if query else target_path,
        share_token=share_token,
        subject_user_id=str(getattr(current_user, "id", "") or "") or None,
        share_access_mode=current_access_mode,
    )
    return RedirectResponse(url=launch.preview_url, status_code=302)


@app.post(
    "/shared/{share_token}/authorize",
    include_in_schema=False,
)
@limiter.limit(SHARE_AUTH_RATE_LIMIT)
async def userspace_share_token_authorize(share_token: str, request: Request):
    current_user = await _share_current_user_from_request(request)
    workspace_name, _ = await userspace_service.get_share_prompt_metadata_by_token(
        share_token,
    )
    target_type = await userspace_service.resolve_public_share_target_by_token(
        share_token
    )
    if target_type == "unknown":
        raise HTTPException(status_code=404, detail="Not found")
    fallback_target = f"/shared/{quote(share_token, safe='')}"
    try:
        form = await request.form()
    except Exception:
        form = None

    share_password = _share_form_value(form, "share_password")
    next_target = _normalize_share_next_target(
        _share_form_value(form, "next"),
        fallback_target,
    )
    if not share_password:
        return HTMLResponse(
            _render_share_token_password_prompt(
                workspace_name,
                _share_token_authorize_path(share_token),
                "Password is required",
                next_target,
            ),
            status_code=400,
        )

    try:
        if target_type == "conversation":
            authorization = (
                await userspace_service.authorize_shared_conversation_access(
                    share_token,
                    current_user=current_user,
                    password=share_password,
                )
            )
        else:
            authorization = await userspace_service.authorize_shared_workspace_access(
                share_token,
                current_user=current_user,
                password=share_password,
            )
    except HTTPException as exc:
        detail = str(exc.detail) if isinstance(exc.detail, str) else "Invalid password"
        error_response = HTMLResponse(
            _render_share_token_password_prompt(
                workspace_name,
                _share_token_authorize_path(share_token),
                detail,
                next_target,
            ),
            status_code=exc.status_code,
        )
        clear_share_auth_cookie(error_response, share_token=share_token)
        return error_response

    redirect_response = RedirectResponse(url=next_target, status_code=303)
    if authorization["share_auth_token"]:
        set_share_auth_cookie(
            redirect_response,
            authorization["share_auth_token"],
            max_age=_share_cookie_max_age(authorization["expires_at"]),
            secure=settings.session_cookie_secure,
            share_token=share_token,
        )
    return redirect_response


@app.api_route(
    "/shared/{share_token}",
    methods=["GET"],
    include_in_schema=False,
)
async def userspace_share_token_path(share_token: str, request: Request):
    return await _shared_launch_redirect_by_token(share_token, request, "")


@app.api_route(
    "/shared/{share_token}/{path:path}",
    methods=_SHARE_PROXY_METHODS,
    include_in_schema=False,
)
async def userspace_share_token_path_with_suffix(
    share_token: str,
    path: str,
    request: Request,
):
    return await _shared_launch_redirect_by_token(share_token, request, path)


@app.post(
    "/{owner_username}/{share_slug}/authorize",
    include_in_schema=False,
)
@limiter.limit(SHARE_AUTH_RATE_LIMIT)
async def userspace_share_slug_authorize(
    owner_username: str,
    share_slug: str,
    request: Request,
):
    if owner_username in _share_reserved_roots():
        raise HTTPException(status_code=404, detail="Not found")

    current_user = await _share_current_user_from_request(request)
    workspace_name, owner_display_name = (
        await userspace_service.get_share_prompt_metadata_by_slug(
            owner_username,
            share_slug,
        )
    )
    target_type = await userspace_service.resolve_public_share_target_by_slug(
        owner_username,
        share_slug,
    )
    if target_type == "unknown":
        raise HTTPException(status_code=404, detail="Not found")
    fallback_target = f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}"
    try:
        form = await request.form()
    except Exception:
        form = None

    share_password = _share_form_value(form, "share_password")
    next_target = _normalize_share_next_target(
        _share_form_value(form, "next"),
        fallback_target,
    )
    if not share_password:
        return HTMLResponse(
            _render_share_password_prompt(
                workspace_name,
                _share_slug_authorize_path(owner_username, share_slug),
                owner_display_name or owner_username,
                "Password is required",
                next_target,
            ),
            status_code=400,
        )

    try:
        if target_type == "conversation":
            authorization = (
                await userspace_service.authorize_shared_conversation_access_by_slug(
                    owner_username,
                    share_slug,
                    current_user=current_user,
                    password=share_password,
                )
            )
        else:
            authorization = (
                await userspace_service.authorize_shared_workspace_access_by_slug(
                    owner_username,
                    share_slug,
                    current_user=current_user,
                    password=share_password,
                )
            )
    except HTTPException as exc:
        detail = str(exc.detail) if isinstance(exc.detail, str) else "Invalid password"
        error_response = HTMLResponse(
            _render_share_password_prompt(
                workspace_name,
                _share_slug_authorize_path(owner_username, share_slug),
                owner_display_name or owner_username,
                detail,
                next_target,
            ),
            status_code=exc.status_code,
        )
        clear_share_auth_cookie(
            error_response,
            owner_username=owner_username,
            share_slug=share_slug,
        )
        return error_response

    redirect_response = RedirectResponse(url=next_target, status_code=303)
    if authorization["share_auth_token"]:
        set_share_auth_cookie(
            redirect_response,
            authorization["share_auth_token"],
            max_age=_share_cookie_max_age(authorization["expires_at"]),
            secure=settings.session_cookie_secure,
            owner_username=owner_username,
            share_slug=share_slug,
        )
    return redirect_response


@app.api_route(
    "/{owner_username}/{share_slug}",
    methods=["GET"],
    include_in_schema=False,
)
async def userspace_share_root_proxy(
    owner_username: str, share_slug: str, request: Request
):
    return await _shared_launch_redirect_by_slug(
        owner_username, share_slug, request, ""
    )


@app.api_route(
    "/{owner_username}/{share_slug}/{path:path}",
    methods=_SHARE_PROXY_METHODS,
    include_in_schema=False,
)
async def userspace_share_path_proxy(
    owner_username: str,
    share_slug: str,
    path: str,
    request: Request,
):
    return await _shared_launch_redirect_by_slug(
        owner_username, share_slug, request, path
    )


@app.websocket(
    "/shared/{share_token}",
)
async def userspace_share_token_websocket_root(share_token: str, websocket: WebSocket):
    current_user = await _share_current_user_from_websocket(websocket)
    share_auth_token = share_auth_token_from_request(
        websocket.headers,
        websocket.cookies,
        share_token=share_token,
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id(
        share_token,
        current_user=current_user,
        share_auth_token=share_auth_token,
    )
    await _proxy_shared_workspace_websocket(
        websocket,
        workspace_id=workspace_id,
        path="",
    )


@app.websocket(
    "/shared/{share_token}/{path:path}",
)
async def userspace_share_token_websocket_path(
    share_token: str,
    path: str,
    websocket: WebSocket,
):
    current_user = await _share_current_user_from_websocket(websocket)
    share_auth_token = share_auth_token_from_request(
        websocket.headers,
        websocket.cookies,
        share_token=share_token,
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id(
        share_token,
        current_user=current_user,
        share_auth_token=share_auth_token,
    )
    await _proxy_shared_workspace_websocket(
        websocket,
        workspace_id=workspace_id,
        path=path,
    )


@app.websocket(
    "/{owner_username}/{share_slug}",
)
async def userspace_share_slug_websocket_root(
    owner_username: str,
    share_slug: str,
    websocket: WebSocket,
):
    if owner_username in _share_reserved_roots():
        await websocket.close(code=4404)
        return
    current_user = await _share_current_user_from_websocket(websocket)
    share_auth_token = share_auth_token_from_request(
        websocket.headers,
        websocket.cookies,
        owner_username=owner_username,
        share_slug=share_slug,
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
        owner_username,
        share_slug,
        current_user=current_user,
        share_auth_token=share_auth_token,
    )
    await _proxy_shared_workspace_websocket(
        websocket,
        workspace_id=workspace_id,
        path="",
    )


@app.websocket(
    "/{owner_username}/{share_slug}/{path:path}",
)
async def userspace_share_slug_websocket_path(
    owner_username: str,
    share_slug: str,
    path: str,
    websocket: WebSocket,
):
    if owner_username in _share_reserved_roots():
        await websocket.close(code=4404)
        return
    current_user = await _share_current_user_from_websocket(websocket)
    share_auth_token = share_auth_token_from_request(
        websocket.headers,
        websocket.cookies,
        owner_username=owner_username,
        share_slug=share_slug,
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
        owner_username,
        share_slug,
        current_user=current_user,
        share_auth_token=share_auth_token,
    )
    await _proxy_shared_workspace_websocket(
        websocket,
        workspace_id=workspace_id,
        path=path,
    )


# Root endpoint - serve Indexer UI or API info
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the Indexer UI at root, or return API info if UI not available."""
    dist_index = DIST_DIR / "index.html"

    # In production, serve the built React app
    if dist_index.exists():
        return FileResponse(dist_index, media_type="text/html")

    # Fallback to API info
    return JSONResponse(
        {
            "name": "Ragtime RAG API",
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode,
    )
