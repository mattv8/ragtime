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
import hashlib
import html
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional
from urllib.parse import quote, urlencode

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ragtime import __version__
from ragtime.api import router
from ragtime.api.auth import (AUTH_CODE_EXPIRY, _auth_codes,
                              _cleanup_expired_auth_codes, authenticate)
from ragtime.api.auth import oauth2_token as _oauth2_token_handler
from ragtime.api.auth import router as auth_router
from ragtime.api.auth import validate_redirect_uri
from ragtime.config import settings
from ragtime.core.auth import (create_access_token, create_session,
                               get_external_origin, validate_session)
from ragtime.core.database import connect_db, disconnect_db, get_db
from ragtime.core.logging import setup_logging
from ragtime.core.rate_limit import LOGIN_RATE_LIMIT, limiter
from ragtime.core.ssl import setup_ssl
from ragtime.indexer.background_tasks import background_task_service
from ragtime.indexer.chunking import shutdown_process_pool
from ragtime.indexer.filesystem_service import filesystem_indexer
from ragtime.indexer.pdm_service import pdm_indexer
from ragtime.indexer.repository import repository
from ragtime.indexer.routes import ASSETS_DIR as INDEXER_ASSETS_DIR
from ragtime.indexer.routes import DIST_DIR
from ragtime.indexer.routes import router as indexer_router
from ragtime.indexer.schema_service import schema_indexer
from ragtime.indexer.service import indexer
from ragtime.mcp.config_routes import \
    default_filter_router as mcp_default_filter_router
from ragtime.mcp.config_routes import router as mcp_config_router
from ragtime.mcp.routes import get_mcp_routes, mcp_lifespan_manager
from ragtime.mcp.routes import router as mcp_router
from ragtime.rag import rag
from ragtime.userspace.models import ExecuteComponentRequest
from ragtime.userspace.preview_host import PreviewHostDispatchMiddleware
from ragtime.userspace.routes import router as userspace_router
from ragtime.userspace.runtime_routes import (_proxy_http_request,
                                              _proxy_websocket_request,
                                              _sanitize_preview_query,
                                              _to_websocket_url)
from ragtime.userspace.runtime_routes import router as userspace_runtime_router
from ragtime.userspace.runtime_routes import userspace_runtime_service
from ragtime.userspace.service import userspace_service

# Import indexer routes (always available now that it's part of ragtime)
# Import MCP routes and transport for HTTP API access
# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging("rag_api")

_SHARE_PROXY_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]


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

    # Start background task service for chat
    await background_task_service.start()

    # Backfill userspace Git ignore policy for existing workspaces in the background.
    userspace_service.schedule_startup_git_drift_reconciliation()
    userspace_service.schedule_workspace_mount_watch()

    # Start MCP session manager (enable/disable checked at request time)
    async with mcp_lifespan_manager():
        yield

    # Cleanup - cancel the startup Git policy reconciliation task
    await userspace_service.shutdown_git_drift_reconciliation()
    await userspace_service.shutdown_workspace_mount_watch()

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
app = FastAPI(
    title="Ragtime RAG API",
    description="RAG + Tool Calling API for business intelligence queries",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
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
# When origins is "*", we allow credentials but log a security warning.
# For production, set ALLOWED_ORIGINS to specific origins like "http://localhost:8001"
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
    # Determine base URL from request, considering reverse proxy headers
    base_url = get_external_origin(request)

    # Security warning: OAuth should use HTTPS in production
    if base_url.startswith("http://") and not settings.debug_mode:
        logger.warning(
            "OAuth metadata served over HTTP. Use HTTPS in production for security."
        )

    logger.info(
        f"OAuth authorization server metadata requested from {request.client.host if request.client else 'unknown'}"
    )

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
    # Determine base URL considering reverse proxy headers
    base_url = get_external_origin(request)

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

    logger.info(f"OAuth2 authorization code issued for '{user.username}' via session")

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


def _share_password_cookie_name(owner_username: str, share_slug: str) -> str:
    digest = hashlib.sha256(
        f"{owner_username}:{share_slug}".encode("utf-8")
    ).hexdigest()[:16]
    return f"userspace_share_pw_{digest}"


def _share_token_password_cookie_name(share_token: str) -> str:
    digest = hashlib.sha256((share_token or "").encode("utf-8")).hexdigest()[:16]
    return f"userspace_share_pw_tok_{digest}"


def _render_share_unlock_prompt(
    title: str,
    subtitle: str | None = None,
    error: str | None = None,
) -> str:
    safe_error = html.escape(error) if error else ""
    safe_title = html.escape(title)
    safe_subtitle = html.escape(subtitle) if subtitle else ""
    subtitle_block = (
        f"<p style='margin:0 0 14px 0;color:#94a3b8;font-size:13px'>{safe_subtitle}</p>"
        if safe_subtitle
        else ""
    )
    error_block = (
        f"<p style='color:#fca5a5;margin:0 0 12px 0;font-size:14px'>{safe_error}</p>"
        if safe_error
        else ""
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Shared Workspace</title></head><body style='margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#0f172a;color:#e2e8f0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif'>"
        "<form method='post' style='width:min(92vw,360px);padding:20px;border:1px solid #334155;border-radius:12px;background:#111827'>"
        f"<h1 style='font-size:18px;margin:0 0 6px 0'>{safe_title}</h1>{subtitle_block}"
        f"{error_block}"
        "<label for='share_password' style='display:block;margin-bottom:8px;font-size:13px'>Password</label>"
        "<input id='share_password' name='share_password' type='password' required autofocus autocomplete='current-password' style='width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid #334155;background:#0b1220;color:#e2e8f0'>"
        "<button type='submit' style='margin-top:12px;width:100%;padding:10px 12px;border-radius:8px;border:1px solid #334155;background:#1d4ed8;color:#fff;cursor:pointer'>Continue</button>"
        "</form></body></html>"
    )


def _render_share_password_prompt(
    owner_username: str, share_slug: str, error: str | None = None
) -> str:
    return _render_share_unlock_prompt(
        "Unlock shared workspace",
        f"{owner_username}/{share_slug}",
        error,
    )


def _render_share_token_password_prompt(error: str | None = None) -> str:
    return _render_share_unlock_prompt(
        "Unlock shared workspace",
        "Anonymous shared link",
        error,
    )


async def _share_current_user_from_request(request: Request):
    session_cookie = request.cookies.get("ragtime_session")
    if not session_cookie:
        return None
    token_data = await validate_session(session_cookie)
    if not token_data:
        return None
    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


def _share_target_path(path: str, query: str | None) -> str:
    normalized = "/" + (path or "").lstrip("/")
    if query:
        return f"{normalized}?{query}"
    return normalized


def _render_shared_preview_page(preview_url: str, preview_origin: str) -> str:
    """Render a full-page HTML wrapper that embeds the preview in an iframe.

    This keeps the browser address bar on the server's root origin
    (e.g. http://localhost:8000/shared/TOKEN) while the actual preview
    content loads on its dedicated per-workspace subdomain.
    """
    safe_url = html.escape(preview_url, quote=True)
    safe_origin = html.escape(preview_origin, quote=True)
    return (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Shared Preview</title>"
        "<style>"
        "html,body{margin:0;padding:0;height:100%;overflow:hidden;background:#0f172a}"
        "iframe{width:100%;height:100%;border:none}"
        "</style></head><body>"
        f'<iframe src="{safe_url}" '
        f'sandbox="allow-scripts allow-same-origin allow-forms allow-popups allow-modals allow-downloads" '
        f'allow="clipboard-read;clipboard-write" '
        f'data-preview-origin="{safe_origin}"></iframe>'
        "</body></html>"
    )


def _share_bridge_context(base_path: str) -> dict[str, str | None]:
    return {
        "parent_origin": None,
        "execute_url": f"{base_path.rstrip('/')}/__ragtime/execute-component",
    }


async def _share_current_user_from_websocket(websocket: WebSocket):
    session_cookie = websocket.cookies.get("ragtime_session")
    if not session_cookie:
        return None
    token_data = await validate_session(session_cookie)
    if not token_data:
        return None
    db = await get_db()
    return await db.user.find_unique(where={"id": token_data.user_id})


async def _handle_shared_bridge_request(
    request: Request,
    *,
    workspace_id: str,
) -> Response | None:
    normalized = request.url.path.rstrip("/")
    if normalized.endswith("/__ragtime/bridge.js"):
        return Response(
            content=await userspace_service.build_runtime_bridge_content(workspace_id),
            media_type="application/javascript",
            headers={"Cache-Control": "no-store"},
        )
    if normalized.endswith("/__ragtime/execute-component"):
        if request.method != "POST":
            raise HTTPException(status_code=405, detail="Method not allowed")
        payload = ExecuteComponentRequest.model_validate(await request.json())
        result = await userspace_service.execute_component_from_authorized_shared_preview(
            workspace_id,
            payload,
        )
        return JSONResponse(result.model_dump())
    return None


async def _proxy_shared_workspace_http(
    request: Request,
    *,
    workspace_id: str,
    base_path: str,
    path: str,
) -> Response:
    bridge_response = await _handle_shared_bridge_request(
        request,
        workspace_id=workspace_id,
    )
    if bridge_response is not None:
        return bridge_response

    upstream_url = await userspace_runtime_service.build_shared_preview_upstream_url(
        workspace_id,
        path or "/",
        query=_sanitize_preview_query(request.url.query),
    )
    return await _proxy_http_request(
        request,
        upstream_url,
        proxy_base_path=base_path,
        bridge_workspace_id=workspace_id,
        bridge_context=_share_bridge_context(base_path),
        bridge_script_src=f"{base_path.rstrip('/')}/__ragtime/bridge.js",
    )


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


async def _shared_launch_redirect_by_slug(
    owner_username: str,
    share_slug: str,
    request: Request,
    path: str,
):
    if owner_username in _share_reserved_roots():
        raise HTTPException(status_code=404, detail="Not found")

    current_user = await _share_current_user_from_request(request)
    cookie_name = _share_password_cookie_name(owner_username, share_slug)
    share_password = request.headers.get(
        "x-userspace-share-password"
    ) or request.cookies.get(cookie_name)

    try:
        workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
            owner_username,
            share_slug,
            current_user=current_user,
            password=share_password,
        )
    except HTTPException as exc:
        detail = str(exc.detail).lower() if isinstance(exc.detail, str) else ""
        is_password_error = exc.status_code == 401 and (
            "password required" in detail or "invalid password" in detail
        )
        if is_password_error and request.method == "GET":
            response = HTMLResponse(
                _render_share_password_prompt(
                    owner_username,
                    share_slug,
                    "Invalid password" if "invalid password" in detail else None,
                )
            )
            response.delete_cookie(
                cookie_name,
                path=f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}",
            )
            return response
        raise

    return await _proxy_shared_workspace_http(
        request,
        workspace_id=workspace_id,
        base_path=f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}",
        path=path,
    )


async def _shared_launch_redirect_by_token(
    share_token: str,
    request: Request,
    path: str,
):
    current_user = await _share_current_user_from_request(request)
    cookie_name = _share_token_password_cookie_name(share_token)
    share_password = request.headers.get(
        "x-userspace-share-password"
    ) or request.cookies.get(cookie_name)

    try:
        workspace_id = await userspace_service.resolve_shared_workspace_id(
            share_token,
            current_user=current_user,
            password=share_password,
        )
    except HTTPException as exc:
        detail = str(exc.detail).lower() if isinstance(exc.detail, str) else ""
        is_password_error = exc.status_code == 401 and (
            "password required" in detail or "invalid password" in detail
        )
        if is_password_error and request.method == "GET":
            response = HTMLResponse(
                _render_share_token_password_prompt(
                    "Invalid password" if "invalid password" in detail else None,
                )
            )
            response.delete_cookie(
                cookie_name,
                path=f"/shared/{quote(share_token, safe='')}",
            )
            return response
        raise

    return await _proxy_shared_workspace_http(
        request,
        workspace_id=workspace_id,
        base_path=f"/shared/{quote(share_token, safe='')}",
        path=path,
    )


@app.api_route(
    "/shared/{share_token}",
    methods=["GET", "POST"],
    include_in_schema=False,
)
async def userspace_share_token_path(share_token: str, request: Request):
    if request.method == "POST":
        try:
            form = await request.form()
        except Exception:
            form = None
        if form is not None and "share_password" in form:
            share_password = str(form.get("share_password", "") or "").strip()
            if not share_password:
                return HTMLResponse(
                    _render_share_token_password_prompt("Password is required"),
                    status_code=400,
                )
            response = RedirectResponse(
                url=request.url.path
                + (f"?{request.url.query}" if request.url.query else ""),
                status_code=303,
            )
            response.set_cookie(
                key=_share_token_password_cookie_name(share_token),
                value=share_password,
                max_age=60 * 30,
                httponly=True,
                secure=settings.session_cookie_secure,
                samesite="lax",
                path=f"/shared/{quote(share_token, safe='')}",
            )
            return response

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


@app.api_route(
    "/{owner_username}/{share_slug}",
    methods=["GET", "POST"],
    include_in_schema=False,
)
async def userspace_share_root_proxy(
    owner_username: str, share_slug: str, request: Request
):
    normalized_share_root = (
        f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}/"
    )

    if request.method == "POST":
        try:
            form = await request.form()
        except Exception:
            form = None
        if form is not None and "share_password" in form:
            share_password = str(form.get("share_password", "") or "").strip()
            if not share_password:
                return HTMLResponse(
                    _render_share_password_prompt(
                        owner_username, share_slug, "Password is required"
                    ),
                    status_code=400,
                )
            response = RedirectResponse(
                url=normalized_share_root,
                status_code=303,
            )
            response.set_cookie(
                key=_share_password_cookie_name(owner_username, share_slug),
                value=share_password,
                max_age=60 * 30,
                httponly=True,
                secure=settings.session_cookie_secure,
                samesite="lax",
                path=f"/{quote(owner_username, safe='')}/{quote(share_slug, safe='')}",
            )
            return response

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
    share_password = websocket.cookies.get(
        _share_token_password_cookie_name(share_token),
        "",
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id(
        share_token,
        current_user=current_user,
        password=share_password or None,
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
    share_password = websocket.cookies.get(
        _share_token_password_cookie_name(share_token),
        "",
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id(
        share_token,
        current_user=current_user,
        password=share_password or None,
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
    share_password = websocket.cookies.get(
        _share_password_cookie_name(owner_username, share_slug),
        "",
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
        owner_username,
        share_slug,
        current_user=current_user,
        password=share_password or None,
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
    share_password = websocket.cookies.get(
        _share_password_cookie_name(owner_username, share_slug),
        "",
    )
    workspace_id = await userspace_service.resolve_shared_workspace_id_by_slug(
        owner_username,
        share_slug,
        current_user=current_user,
        password=share_password or None,
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
