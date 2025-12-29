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

from ragtime import __version__
from ragtime.config import settings
from ragtime.core.logging import setup_logging, get_logger
from ragtime.core.database import connect_db, disconnect_db
from ragtime.api import router
from ragtime.rag import rag

# Import indexer routes (always available now that it's part of ragtime)
from ragtime.indexer.routes import router as indexer_router, ASSETS_DIR as INDEXER_ASSETS_DIR
from fastapi.staticfiles import StaticFiles

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
    if settings.enable_indexer:
        from ragtime.indexer.service import indexer
        recovered = await indexer.recover_interrupted_jobs()
        if recovered > 0:
            logger.info(f"Recovered {recovered} interrupted indexing job(s)")

        # Start background task service for chat
        from ragtime.indexer.background_tasks import background_task_service
        await background_task_service.start()

    yield

    # Cleanup
    if settings.enable_indexer:
        from ragtime.indexer.background_tasks import background_task_service
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
    openapi_url="/openapi.json"
)

# CORS middleware
origins = settings.allowed_origins.split(",") if settings.allowed_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Include indexer routes if enabled
if settings.enable_indexer:
    app.include_router(indexer_router)
    # Mount static files for indexer UI assets at root
    if INDEXER_ASSETS_DIR.exists():
        app.mount("/assets", StaticFiles(directory=INDEXER_ASSETS_DIR), name="indexer_ui_assets")
    logger.info("Indexer API enabled at /indexes, UI served at root (/)")


# Root endpoint - serve Indexer UI or API info
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def root():
    """Serve the Indexer UI at root, or return API info if UI not available."""
    from ragtime.indexer.routes import DIST_DIR
    from fastapi.responses import FileResponse, JSONResponse

    dist_index = DIST_DIR / "index.html"

    # In production, serve the built React app
    if dist_index.exists() and settings.enable_indexer:
        return FileResponse(dist_index, media_type="text/html")

    # Fallback to API info
    return JSONResponse({
        "name": "Ragtime RAG API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug_mode
    )
