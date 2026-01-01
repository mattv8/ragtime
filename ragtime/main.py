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

from ragtime import __version__
from ragtime.api import router
from ragtime.api.auth import router as auth_router
from ragtime.config import settings
from ragtime.core.database import connect_db, disconnect_db
from ragtime.core.logging import get_logger, setup_logging

# Import indexer routes (always available now that it's part of ragtime)
from ragtime.indexer.routes import ASSETS_DIR as INDEXER_ASSETS_DIR
from ragtime.indexer.routes import router as indexer_router

# Import MCP routes for HTTP API access
from ragtime.mcp.routes import router as mcp_router
from ragtime.rag import rag

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logging("rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async lifespan handler for startup/shutdown."""
    logger.info(f"Starting RAG API v{__version__}")

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

    await filesystem_indexer.cleanup_orphaned_jobs()

    # Discover orphan indexes (FAISS files without database metadata)
    discovered = await indexer.discover_orphan_indexes()
    if discovered > 0:
        logger.info(f"Discovered {discovered} orphan index(es) on disk")

    # Start background task service for chat
    from ragtime.indexer.background_tasks import background_task_service

    await background_task_service.start()

    yield

    # Cleanup
    await background_task_service.stop()

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

# Include MCP routes (HTTP convenience API)
if settings.mcp_enabled:
    app.include_router(mcp_router)
    logger.info(
        "MCP HTTP API enabled at /mcp (for stdio transport: python -m ragtime.mcp)"
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
