"""
RAG API with Tool Calling - OpenAI-Compatible FastAPI Server
=============================================================

Main application entry point.

Usage:
    uvicorn ragtime.main:app --host 0.0.0.0 --port 8000 --reload

    Or run directly:
    python -m ragtime.main
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from ragtime import __version__
from ragtime.api import router
from ragtime.api.auth import router as auth_router
from ragtime.config import settings
from ragtime.core.database import connect_db, disconnect_db
from ragtime.core.logging import get_logger, setup_logging
from ragtime.core.rate_limit import limiter

# Import indexer routes (always available now that it's part of ragtime)
from ragtime.indexer.routes import ASSETS_DIR as INDEXER_ASSETS_DIR
from ragtime.indexer.routes import router as indexer_router
from ragtime.mcp.config_routes import router as mcp_config_router

# Import MCP routes and transport for HTTP API access
from ragtime.mcp.routes import get_mcp_routes, mcp_lifespan_manager, mcp_transport_route
from ragtime.mcp.routes import router as mcp_router
from ragtime.rag import rag

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging("rag_api")


def _log_security_warnings() -> None:
    """Log security warnings for potential plaintext credential transmission."""
    warnings = []

    # Check if API key is not set
    if not settings.api_key:
        warnings.append(
            "API_KEY is not set. The OpenAI-compatible API endpoint is unprotected. "
            "Anyone with network access can use your LLM, which may incur costs."
        )

    # Check for wildcard CORS origins
    if settings.allowed_origins == "*":
        warnings.append(
            "ALLOWED_ORIGINS=* allows requests from any origin. "
            "Consider restricting to specific domains in production."
        )

    # Check for HTTP without HTTPS or reverse proxy
    if not settings.session_cookie_secure and not settings.enable_https:
        warnings.append(
            "SESSION_COOKIE_SECURE=false and ENABLE_HTTPS=false. "
            "Credentials and API keys may be transmitted in plaintext. "
            "Set ENABLE_HTTPS=true or use an HTTPS reverse proxy."
        )

    if warnings:
        logger.warning("=" * 70)
        logger.warning("SECURITY WARNINGS")
        logger.warning("=" * 70)
        for warning in warnings:
            logger.warning(f"  - {warning}")
        logger.warning("-" * 70)
        logger.warning(
            "If you have network-level protection (VPN, firewall, HTTPS proxy), "
            "you can ignore these warnings."
        )
        logger.warning("=" * 70)


def _validate_ssl_certificates() -> None:
    """Validate SSL certificates if HTTPS is enabled."""
    if not settings.enable_https:
        return

    from ragtime.core.ssl import setup_ssl

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan handler for startup/shutdown."""
    logger.info(f"Starting RAG API v{__version__}")

    # Security warnings for plaintext credential transmission
    _log_security_warnings()

    # Validate SSL certificates if HTTPS is enabled
    _validate_ssl_certificates()

    # Connect to database
    await connect_db()

    # Initialize RAG components
    await rag.initialize()

    # Recover any interrupted indexing jobs (survives hot-reloads)
    from ragtime.indexer.service import indexer

    recovered = await indexer.recover_interrupted_jobs()
    if recovered > 0:
        logger.info(f"Recovered {recovered} interrupted indexing job(s)")

    # Clean up any orphaned filesystem indexing jobs from previous runs
    from ragtime.indexer.filesystem_service import filesystem_indexer

    await filesystem_indexer._cleanup_stale_jobs()

    # Discover orphan indexes (FAISS files without database metadata)
    discovered = await indexer.discover_orphan_indexes()
    if discovered > 0:
        logger.info(f"Discovered {discovered} orphan index(es) on disk")

    # Garbage collect orphaned pgvector embeddings
    from ragtime.indexer.repository import repository

    gc_results = await repository.cleanup_orphaned_embeddings()
    gc_total = sum(gc_results.values())
    if gc_total > 0:
        logger.info(f"Garbage collected {gc_total} orphaned embedding(s)")

    # Start background task service for chat
    from ragtime.indexer.background_tasks import background_task_service

    await background_task_service.start()

    # Start MCP session manager (enable/disable checked at request time)
    async with mcp_lifespan_manager():
        yield

    # Cleanup - stop background services before disconnecting DB
    await background_task_service.stop()

    # Stop PDM indexer tasks
    from ragtime.indexer.pdm_service import pdm_indexer

    await pdm_indexer.shutdown()

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

# CORS middleware
origins = (
    settings.allowed_origins.split(",") if settings.allowed_origins != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Include API routes
app.include_router(router)

# Include auth routes
app.include_router(auth_router)

# Include indexer routes
app.include_router(indexer_router)
# Mount static files for indexer UI assets at root
if INDEXER_ASSETS_DIR.exists():
    app.mount(
        "/assets", StaticFiles(directory=INDEXER_ASSETS_DIR), name="indexer_ui_assets"
    )
logger.info("Indexer API enabled at /indexes, UI served at root (/)")

# Include MCP routes (enabled/disabled controlled via database setting)
app.include_router(mcp_router)  # Debug endpoints at /mcp-debug/*
app.include_router(mcp_config_router)  # MCP route configuration at /mcp-routes/*
# Add all MCP routes (default /mcp and custom /mcp/{route_path})
for route in get_mcp_routes():
    app.routes.append(route)
logger.info("MCP routes registered (enable/disable via Settings UI)")


# OAuth discovery endpoint - explicitly reject to prevent client prompts
@app.get("/.well-known/oauth-authorization-server", include_in_schema=False)
async def oauth_discovery():
    """
    OAuth Authorization Server Metadata endpoint.

    Returns 404 to indicate OAuth is not supported.
    MCP clients should use Bearer token in Authorization header instead.
    """
    from fastapi import HTTPException

    raise HTTPException(
        status_code=404, detail="OAuth not supported. Use Bearer token authentication."
    )


# Root endpoint - serve Indexer UI or API info
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the Indexer UI at root, or return API info if UI not available."""
    from fastapi.responses import FileResponse, JSONResponse

    from ragtime.indexer.routes import DIST_DIR

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
