"""
Indexer API routes.
"""

import os
from pathlib import Path
from typing import List, Optional

import httpx
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from ragtime.core.logging import get_logger
from ragtime.indexer.models import (
    IndexConfig,
    IndexInfo,
    IndexJobResponse,
    IndexStatus,
    CreateIndexRequest,
    AppSettings,
    UpdateSettingsRequest,
)
from ragtime.indexer.service import indexer
from ragtime.indexer.repository import repository

logger = get_logger(__name__)

router = APIRouter(prefix="/indexes", tags=["Indexer"])

# Path to static files - React build lives under dist
STATIC_DIR = Path(__file__).parent / "static"
DIST_DIR = STATIC_DIR / "dist"
ASSETS_DIR = DIST_DIR / "assets"

# Check if running in development mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"


@router.get("", response_model=List[IndexInfo])
async def list_indexes():
    """List all available FAISS indexes."""
    return await indexer.list_indexes()


@router.get("/jobs", response_model=List[IndexJobResponse])
async def list_jobs():
    """List all indexing jobs."""
    jobs = await indexer.list_jobs()
    return [
        IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            processed_chunks=job.processed_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
        for job in jobs
    ]


@router.get("/jobs/{job_id}", response_model=IndexJobResponse)
async def get_job(job_id: str):
    """Get status of an indexing job."""
    job = await indexer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return IndexJobResponse(
        id=job.id,
        name=job.name,
        status=job.status,
        progress_percent=job.progress_percent,
        total_files=job.total_files,
        processed_files=job.processed_files,
        total_chunks=job.total_chunks,
        processed_chunks=job.processed_chunks,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a pending or processing job."""
    job = await indexer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status not in [IndexStatus.PENDING, IndexStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status '{job.status.value}'"
        )

    await indexer.cancel_job(job_id)
    return {"message": f"Job '{job_id}' cancelled successfully"}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job record (must be completed, failed, or cancelled)."""
    job = await indexer.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in [IndexStatus.PENDING, IndexStatus.PROCESSING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete active job. Cancel it first."
        )

    await repository.delete_job(job_id)
    return {"message": f"Job '{job_id}' deleted successfully"}


# Supported archive extensions
SUPPORTED_ARCHIVE_EXTENSIONS = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2")


@router.post("/upload", response_model=IndexJobResponse)
async def upload_and_index(
    file: UploadFile = File(..., description="Archive file containing source code (.zip, .tar, .tar.gz, .tar.bz2)"),
    name: str = Form(..., description="Name for the index"),
    description: str = Form(
        default="",
        description="Description for AI context - helps the model understand what this index contains"
    ),
    file_patterns: str = Form(
        default="**/*.py,**/*.md,**/*.rst,**/*.txt,**/*.xml",
        description="Comma-separated glob patterns for files to include"
    ),
    exclude_patterns: str = Form(
        default="**/node_modules/**,**/__pycache__/**,**/venv/**,**/.git/**",
        description="Comma-separated glob patterns to exclude"
    ),
    chunk_size: int = Form(default=1000, ge=100, le=4000),
    chunk_overlap: int = Form(default=200, ge=0, le=1000),
):
    """
    Upload an archive file and create a FAISS index from it.

    Supported formats: .zip, .tar, .tar.gz, .tar.bz2
    The archive should contain source code files. Large codebases are supported.
    Processing happens in the background - check /jobs/{id} for status.
    """
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in SUPPORTED_ARCHIVE_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_ARCHIVE_EXTENSIONS)}"
        )

    config = IndexConfig(
        name=name,
        description=description,
        file_patterns=[p.strip() for p in file_patterns.split(",")],
        exclude_patterns=[p.strip() for p in exclude_patterns.split(",")],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    try:
        job = await indexer.create_index_from_upload(file.file, file.filename, config)

        return IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
    except Exception as e:
        logger.exception("Failed to start indexing job")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/git", response_model=IndexJobResponse)
async def index_from_git(request: CreateIndexRequest):
    """
    Create a FAISS index from a git repository.

    Clones the repository and indexes the source files.
    Processing happens in the background - check /jobs/{id} for status.
    """
    if not request.git_url:
        raise HTTPException(status_code=400, detail="git_url is required")

    config = request.config or IndexConfig(name=request.name)
    config.name = request.name  # Ensure name is set

    try:
        job = await indexer.create_index_from_git(
            git_url=request.git_url,
            branch=request.git_branch,
            config=config
        )

        return IndexJobResponse(
            id=job.id,
            name=job.name,
            status=job.status,
            progress_percent=job.progress_percent,
            total_files=job.total_files,
            processed_files=job.processed_files,
            total_chunks=job.total_chunks,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
        )
    except Exception as e:
        logger.exception("Failed to start git indexing job")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{name}")
async def delete_index(name: str):
    """Delete an index by name."""
    if await indexer.delete_index(name):
        return {"message": f"Index '{name}' deleted successfully"}
    raise HTTPException(status_code=404, detail="Index not found")


class ToggleIndexRequest(BaseModel):
    """Request to toggle index enabled status."""
    enabled: bool = Field(description="Whether the index is enabled for RAG context")


class UpdateIndexDescriptionRequest(BaseModel):
    """Request to update index description."""
    description: str = Field(description="Description for AI context")


@router.patch("/{name}/toggle")
async def toggle_index(name: str, request: ToggleIndexRequest):
    """Toggle an index's enabled status for RAG context."""
    success = await repository.set_index_enabled(name, request.enabled)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
    return {"message": f"Index '{name}' {'enabled' if request.enabled else 'disabled'}", "enabled": request.enabled}


@router.patch("/{name}/description")
async def update_index_description(name: str, request: UpdateIndexDescriptionRequest):
    """Update an index's description for AI context."""
    success = await repository.update_index_description(name, request.description)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
    return {"message": f"Description updated for '{name}'", "description": request.description}


# -----------------------------------------------------------------------------
# Settings Endpoints
# -----------------------------------------------------------------------------

@router.get("/settings", response_model=AppSettings, tags=["Settings"])
async def get_settings():
    """Get current application settings."""
    return await repository.get_settings()


@router.put("/settings", response_model=AppSettings, tags=["Settings"])
async def update_settings(request: UpdateSettingsRequest):
    """Update application settings."""
    updates = request.model_dump(exclude_unset=True)
    return await repository.update_settings(updates)


# -----------------------------------------------------------------------------
# Tool Configuration Endpoints
# -----------------------------------------------------------------------------

from ragtime.indexer.models import (
    ToolConfig,
    ToolType,
    CreateToolConfigRequest,
    UpdateToolConfigRequest,
    ToolTestRequest,
)


class SSHKeyPairResponse(BaseModel):
    """Response containing generated SSH keypair."""
    private_key: str
    public_key: str
    fingerprint: str


class ToolTestResponse(BaseModel):
    """Response from a tool connection test."""
    success: bool
    message: str
    details: Optional[dict] = None


@router.get("/tools", response_model=List[ToolConfig], tags=["Tools"])
async def list_tool_configs(enabled_only: bool = False):
    """List all tool configurations."""
    return await repository.list_tool_configs(enabled_only=enabled_only)


@router.post("/tools", response_model=ToolConfig, tags=["Tools"])
async def create_tool_config(request: CreateToolConfigRequest):
    """Create a new tool configuration."""
    config = ToolConfig(
        name=request.name,
        tool_type=request.tool_type,
        description=request.description,
        connection_config=request.connection_config,
        max_results=request.max_results,
        timeout=request.timeout,
        allow_write=request.allow_write,
    )
    return await repository.create_tool_config(config)


# Heartbeat models - must be defined before the route
class HeartbeatStatus(BaseModel):
    """Heartbeat status for a single tool."""
    tool_id: str
    alive: bool
    latency_ms: float | None = None
    error: str | None = None
    checked_at: str


class HeartbeatResponse(BaseModel):
    """Response from batch heartbeat check."""
    statuses: dict[str, HeartbeatStatus]


@router.get("/tools/heartbeat", response_model=HeartbeatResponse, tags=["Tools"])
async def check_tool_heartbeats():
    """
    Check connection heartbeat for all enabled tools.
    Returns quick connectivity status without updating database test results.
    Designed for frequent polling (every 10-30 seconds).
    """
    import asyncio
    from datetime import datetime, timezone

    tools = await repository.list_tool_configs(enabled_only=True)
    statuses: dict[str, HeartbeatStatus] = {}

    async def check_single_tool(tool) -> HeartbeatStatus:
        """Check heartbeat for a single tool with timeout."""
        start_time = asyncio.get_event_loop().time()
        checked_at = datetime.now(timezone.utc).isoformat()

        try:
            # Quick ping-style check with short timeout
            result = await asyncio.wait_for(
                _heartbeat_check(tool.tool_type, tool.connection_config),
                timeout=5.0  # Short timeout for heartbeat
            )
            latency = (asyncio.get_event_loop().time() - start_time) * 1000

            return HeartbeatStatus(
                tool_id=tool.id,
                alive=result.success,
                latency_ms=round(latency, 1) if result.success else None,
                error=result.message if not result.success else None,
                checked_at=checked_at
            )
        except asyncio.TimeoutError:
            return HeartbeatStatus(
                tool_id=tool.id,
                alive=False,
                latency_ms=None,
                error="Heartbeat timeout (5s)",
                checked_at=checked_at
            )
        except Exception as e:
            return HeartbeatStatus(
                tool_id=tool.id,
                alive=False,
                latency_ms=None,
                error=str(e),
                checked_at=checked_at
            )

    # Check all tools concurrently
    results = await asyncio.gather(
        *[check_single_tool(tool) for tool in tools],
        return_exceptions=True
    )

    for result in results:
        if isinstance(result, HeartbeatStatus):
            statuses[result.tool_id] = result

    return HeartbeatResponse(statuses=statuses)


@router.get("/tools/{tool_id}", response_model=ToolConfig, tags=["Tools"])
async def get_tool_config(tool_id: str):
    """Get a specific tool configuration."""
    config = await repository.get_tool_config(tool_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")
    return config


@router.put("/tools/{tool_id}", response_model=ToolConfig, tags=["Tools"])
async def update_tool_config(tool_id: str, request: UpdateToolConfigRequest):
    """Update an existing tool configuration."""
    updates = request.model_dump(exclude_unset=True)
    config = await repository.update_tool_config(tool_id, updates)
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")
    return config


@router.delete("/tools/{tool_id}", tags=["Tools"])
async def delete_tool_config(tool_id: str):
    """
    Delete a tool configuration.

    For Odoo tools, also disconnects from the Docker network if no other
    tools are using it.
    """
    # Get the tool config before deleting to check for network cleanup
    tool = await repository.get_tool_config(tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Check if this is an Odoo tool with a docker network
    docker_network = None
    if tool.tool_type == "odoo_shell" and tool.connection_config:
        docker_network = tool.connection_config.get("docker_network")

    # Delete the tool
    success = await repository.delete_tool_config(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Cleanup: disconnect from network if no other tools need it
    if docker_network:
        # Check if any other odoo_shell tools use this network
        all_tools = await repository.list_tool_configs()
        network_still_needed = any(
            t.tool_type == "odoo_shell" and
            t.connection_config.get("docker_network") == docker_network and
            t.id != tool_id
            for t in all_tools
        )

        if not network_still_needed:
            # Disconnect from the network
            try:
                await disconnect_from_network(docker_network)
                logger.info(f"Disconnected from network '{docker_network}' after tool deletion")
            except Exception as e:
                logger.warning(f"Failed to disconnect from network '{docker_network}': {e}")

    return {"message": "Tool configuration deleted"}


@router.post("/tools/{tool_id}/toggle", tags=["Tools"])
async def toggle_tool_config(tool_id: str, enabled: bool):
    """Toggle a tool's enabled status."""
    from ragtime.core.app_settings import invalidate_settings_cache
    from ragtime.rag import rag

    config = await repository.update_tool_config(tool_id, {"enabled": enabled})
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Invalidate cache and reinitialize RAG agent to pick up the change
    invalidate_settings_cache()
    await rag.initialize()
@router.post("/tools/test", response_model=ToolTestResponse, tags=["Tools"])
async def test_tool_connection(request: ToolTestRequest):
    """
    Test a tool connection without saving.
    Used during the wizard to validate connection settings.
    """
    import asyncio
    import subprocess

    tool_type = request.tool_type
    config = request.connection_config

    if tool_type == ToolType.POSTGRES:
        return await _test_postgres_connection(config)
    elif tool_type == ToolType.ODOO_SHELL:
        return await _test_odoo_connection(config)
    elif tool_type == ToolType.SSH_SHELL:
        return await _test_ssh_connection(config)
    else:
        return ToolTestResponse(
            success=False,
            message=f"Unknown tool type: {tool_type}"
        )


@router.post("/tools/ssh/generate-keypair", response_model=SSHKeyPairResponse, tags=["Tools"])
async def generate_ssh_keypair(comment: str = "ragtime", passphrase: str = ""):
    """
    Generate a new SSH keypair for use with remote connections.
    Returns private key, public key, and fingerprint.
    The private key should be stored in the tool's connection config.
    The public key should be added to the remote server's authorized_keys.

    Args:
        comment: Comment for the key (appears in public key)
        passphrase: Optional passphrase to encrypt the private key
    """
    import tempfile
    import subprocess
    import os

    try:
        # Create temp directory for key generation
        with tempfile.TemporaryDirectory() as tmpdir:
            key_path = os.path.join(tmpdir, "id_rsa")

            # Generate RSA keypair using ssh-keygen
            process = subprocess.run(
                [
                    "ssh-keygen",
                    "-t", "rsa",
                    "-b", "4096",
                    "-f", key_path,
                    "-N", passphrase,  # Passphrase (empty string = no passphrase)
                    "-C", comment,
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            if process.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to generate keypair: {process.stderr}"
                )

            # Read the generated keys
            with open(key_path, "r") as f:
                private_key = f.read()
            with open(f"{key_path}.pub", "r") as f:
                public_key = f.read().strip()

            # Get fingerprint
            fp_process = subprocess.run(
                ["ssh-keygen", "-lf", key_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            fingerprint = fp_process.stdout.strip() if fp_process.returncode == 0 else "unknown"

            return SSHKeyPairResponse(
                private_key=private_key,
                public_key=public_key,
                fingerprint=fingerprint
            )

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Key generation timed out")
    except Exception as e:
        logger.exception("Failed to generate SSH keypair")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tools/{tool_id}/test", response_model=ToolTestResponse, tags=["Tools"])
async def test_saved_tool_connection(tool_id: str):
    """Test the connection for a saved tool configuration."""
    config = await repository.get_tool_config(tool_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")

    # Test the connection
    result = await test_tool_connection(
        ToolTestRequest(
            tool_type=config.tool_type,
            connection_config=config.connection_config
        )
    )

    # Update test results in database
    await repository.update_tool_test_result(
        tool_id,
        success=result.success,
        error=result.message if not result.success else None
    )

    return result


async def _heartbeat_check(tool_type: str, config: dict) -> ToolTestResponse:
    """
    Quick heartbeat check for a tool connection.
    Uses minimal queries/commands to verify connectivity.
    """
    import asyncio
    import subprocess

    if tool_type == "postgres":
        return await _heartbeat_postgres(config)
    elif tool_type == "odoo_shell":
        return await _heartbeat_odoo(config)
    elif tool_type == "ssh_shell":
        return await _heartbeat_ssh(config)
    else:
        return ToolTestResponse(success=False, message=f"Unknown tool type: {tool_type}")


async def _heartbeat_postgres(config: dict) -> ToolTestResponse:
    """Quick PostgreSQL heartbeat check."""
    import asyncio
    import subprocess

    host = config.get("host", "")
    port = config.get("port", 5432)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")

    try:
        if host:
            cmd = ["psql", "-h", host, "-p", str(port), "-U", user, "-d", database, "-c", "SELECT 1;"]
            env = {"PGPASSWORD": password}
        elif container:
            cmd = [
                "docker", "exec", "-i", container,
                "bash", "-c",
                'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1;"'
            ]
            env = None
        else:
            return ToolTestResponse(success=False, message="No connection configured")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        stdout, stderr = await process.communicate()

        return ToolTestResponse(
            success=process.returncode == 0,
            message="OK" if process.returncode == 0 else stderr.decode("utf-8", errors="replace").strip()[:100]
        )
    except Exception as e:
        return ToolTestResponse(success=False, message=str(e)[:100])


async def _heartbeat_odoo(config: dict) -> ToolTestResponse:
    """Quick Odoo container/SSH heartbeat check."""
    import asyncio
    import subprocess

    mode = config.get("mode", "docker")

    if mode == "ssh":
        # SSH mode - use Paramiko for heartbeat
        from ragtime.core.ssh import SSHConfig, test_ssh_connection

        ssh_host = config.get("ssh_host", "")
        ssh_port = config.get("ssh_port", 22)
        ssh_user = config.get("ssh_user", "")
        ssh_key_path = config.get("ssh_key_path", "")
        ssh_key_content = config.get("ssh_key_content", "")
        ssh_key_passphrase = config.get("ssh_key_passphrase", "")
        ssh_password = config.get("ssh_password", "")

        if not ssh_host or not ssh_user:
            return ToolTestResponse(success=False, message="SSH not configured")

        ssh_config = SSHConfig(
            host=ssh_host,
            port=ssh_port,
            user=ssh_user,
            password=ssh_password if ssh_password else None,
            key_path=ssh_key_path if ssh_key_path else None,
            key_content=ssh_key_content if ssh_key_content else None,
            key_passphrase=ssh_key_passphrase if ssh_key_passphrase else None,
            timeout=5,
        )

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: test_ssh_connection(ssh_config))

            return ToolTestResponse(
                success=result.success,
                message="OK" if result.success else (result.stderr or result.stdout)[:100]
            )
        except Exception as e:
            return ToolTestResponse(success=False, message=str(e)[:100])
    else:
        # Docker mode - check container is running
        container = config.get("container", "")
        if not container:
            return ToolTestResponse(success=False, message="No container configured")

        cmd = ["docker", "exec", "-i", container, "echo", "OK"]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            return ToolTestResponse(
                success=process.returncode == 0,
                message="OK" if process.returncode == 0 else stderr.decode("utf-8", errors="replace").strip()[:100]
            )
        except Exception as e:
            return ToolTestResponse(success=False, message=str(e)[:100])


async def _heartbeat_ssh(config: dict) -> ToolTestResponse:
    """Quick SSH heartbeat check using Paramiko."""
    import asyncio
    from ragtime.core.ssh import SSHConfig, test_ssh_connection

    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")
    key_path = config.get("key_path", "")
    key_content = config.get("key_content", "")
    key_passphrase = config.get("key_passphrase", "")
    password = config.get("password", "")

    if not host or not user:
        return ToolTestResponse(success=False, message="SSH not configured")

    ssh_config = SSHConfig(
        host=host,
        port=port,
        user=user,
        password=password if password else None,
        key_path=key_path if key_path else None,
        key_content=key_content if key_content else None,
        key_passphrase=key_passphrase if key_passphrase else None,
        timeout=5,
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: test_ssh_connection(ssh_config))

        return ToolTestResponse(
            success=result.success,
            message="OK" if result.success else (result.stderr or result.stdout)[:100]
        )
    except Exception as e:
        return ToolTestResponse(success=False, message=str(e)[:100])


async def _test_postgres_connection(config: dict) -> ToolTestResponse:
    """Test PostgreSQL connection."""
    import asyncio
    import subprocess

    host = config.get("host", "")
    port = config.get("port", 5432)
    user = config.get("user", "")
    password = config.get("password", "")
    database = config.get("database", "")
    container = config.get("container", "")

    try:
        if host:
            # Direct connection test
            cmd = [
                "psql",
                "-h", host,
                "-p", str(port),
                "-U", user,
                "-d", database,
                "-c", "SELECT 1;"
            ]
            env = {"PGPASSWORD": password}
        elif container:
            # Docker container test
            cmd = [
                "docker", "exec", "-i", container,
                "bash", "-c",
                'PGPASSWORD="${POSTGRES_PASSWORD}" psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -c "SELECT 1;"'
            ]
            env = None
        else:
            return ToolTestResponse(
                success=False,
                message="Either host or container must be specified"
            )

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            ),
            timeout=10.0
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return ToolTestResponse(
                success=True,
                message="PostgreSQL connection successful",
                details={"host": host or container, "database": database}
            )
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return ToolTestResponse(
                success=False,
                message=f"Connection failed: {error}"
            )

    except asyncio.TimeoutError:
        return ToolTestResponse(
            success=False,
            message="Connection timed out after 10 seconds"
        )
    except FileNotFoundError:
        return ToolTestResponse(
            success=False,
            message="Required command (psql or docker) not found"
        )
    except Exception as e:
        return ToolTestResponse(
            success=False,
            message=f"Connection test failed: {str(e)}"
        )


async def _test_odoo_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection (Docker or SSH mode)."""
    import asyncio
    import subprocess

    mode = config.get("mode", "docker")

    if mode == "ssh":
        return await _test_odoo_ssh_connection(config)
    else:
        return await _test_odoo_docker_connection(config)


async def _test_odoo_ssh_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection via SSH using Paramiko."""
    import asyncio
    from ragtime.core.ssh import SSHConfig, execute_ssh_command, test_ssh_connection

    ssh_host = config.get("ssh_host", "")
    ssh_port = config.get("ssh_port", 22)
    ssh_user = config.get("ssh_user", "")
    ssh_key_path = config.get("ssh_key_path", "")
    ssh_key_content = config.get("ssh_key_content", "")
    ssh_key_passphrase = config.get("ssh_key_passphrase", "")
    ssh_password = config.get("ssh_password", "")
    database = config.get("database", "odoo")
    config_path = config.get("config_path", "")
    odoo_bin_path = config.get("odoo_bin_path", "odoo-bin")
    working_directory = config.get("working_directory", "")
    run_as_user = config.get("run_as_user", "")

    if not ssh_host or not ssh_user:
        return ToolTestResponse(
            success=False,
            message="SSH host and user are required"
        )

    # Build SSH config
    ssh_config = SSHConfig(
        host=ssh_host,
        port=ssh_port,
        user=ssh_user,
        password=ssh_password if ssh_password else None,
        key_path=ssh_key_path if ssh_key_path else None,
        key_content=ssh_key_content if ssh_key_content else None,
        key_passphrase=ssh_key_passphrase if ssh_key_passphrase else None,
        timeout=30,
    )

    try:
        # First test basic SSH connectivity
        loop = asyncio.get_event_loop()
        ssh_result = await loop.run_in_executor(None, lambda: test_ssh_connection(ssh_config))

        if not ssh_result.success:
            return ToolTestResponse(
                success=False,
                message=f"SSH connection failed: {ssh_result.stderr or ssh_result.stdout}"
            )

        # Build Odoo shell command
        odoo_cmd = f"{odoo_bin_path} shell --no-http -d {database}"
        if config_path:
            odoo_cmd = f"{odoo_cmd} -c {config_path}"
        if run_as_user:
            odoo_cmd = f"sudo -u {run_as_user} {odoo_cmd}"
        if working_directory:
            odoo_cmd = f"cd {working_directory} && {odoo_cmd}"

        # Test Odoo shell with heredoc
        test_input = "print('ODOO_TEST_SUCCESS')\nexit()\n"
        full_command = f"{odoo_cmd} <<'ODOO_EOF'\n{test_input}ODOO_EOF"

        odoo_result = await loop.run_in_executor(
            None,
            lambda: execute_ssh_command(ssh_config, full_command)
        )

        if "ODOO_TEST_SUCCESS" in odoo_result.output:
            return ToolTestResponse(
                success=True,
                message=f"Odoo shell accessible via SSH",
                details={
                    "host": ssh_host,
                    "database": database,
                    "mode": "ssh"
                }
            )
        else:
            output_snippet = odoo_result.output[-500:] if len(odoo_result.output) > 500 else odoo_result.output
            return ToolTestResponse(
                success=False,
                message="Odoo shell test failed via SSH",
                details={"output_tail": output_snippet}
            )

    except Exception as e:
        return ToolTestResponse(
            success=False,
            message=f"SSH test failed: {str(e)}"
        )


async def _test_odoo_docker_connection(config: dict) -> ToolTestResponse:
    """Test Odoo shell connection via Docker."""
    import asyncio
    import subprocess

    container = config.get("container", "")
    database = config.get("database", "odoo")
    config_path = config.get("config_path", "")
    docker_network = config.get("docker_network", "")

    if not container:
        return ToolTestResponse(
            success=False,
            message="Container name is required"
        )

    try:
        # Test if container is running
        cmd = ["docker", "inspect", "-f", "{{.State.Running}}", container]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ),
            timeout=10.0
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            return ToolTestResponse(
                success=False,
                message=f"Container '{container}' not found or not accessible"
            )

        is_running = stdout.decode().strip().lower() == "true"
        if not is_running:
            return ToolTestResponse(
                success=False,
                message=f"Container '{container}' is not running"
            )

        # Get Odoo version - try multiple approaches
        version = "unknown"

        # Try direct odoo --version (works for standard Odoo)
        cmd = ["docker", "exec", "-i", container, "odoo", "--version"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            timeout=10.0
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            version = stdout.decode().strip()
        else:
            # Try alternative: check if odoo shell command exists
            cmd = ["docker", "exec", "-i", container, "which", "odoo"]
            process = await asyncio.wait_for(
                asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
                timeout=5.0
            )
            stdout, _ = await process.communicate()

            if process.returncode != 0:
                return ToolTestResponse(
                    success=False,
                    message="Odoo command not found in container"
                )
            # odoo exists, but --version not supported (custom wrapper)
            version = "detected (custom wrapper)"

        # Test shell execution with stdin pipe (standard approach)
        cmd = [
            "docker", "exec", "-i", container,
            "odoo", "shell", "--no-http", "-d", database
        ]
        if config_path:
            cmd.extend(["-c", config_path])

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            ),
            timeout=120.0  # Shell initialization can take 10-60+ seconds
        )

        test_input = "print('ODOO_TEST_SUCCESS')\nexit()\n"
        stdout, _ = await process.communicate(input=test_input.encode())
        output = stdout.decode()

        if "ODOO_TEST_SUCCESS" in output:
            return ToolTestResponse(
                success=True,
                message=f"Odoo shell accessible: {version}",
                details={
                    "container": container,
                    "database": database,
                    "version": version,
                    "docker_network": docker_network,
                    "mode": "docker"
                }
            )
        else:
            # Check for common errors in output
            if "database" in output.lower() and "not exist" in output.lower():
                return ToolTestResponse(
                    success=False,
                    message=f"Database '{database}' does not exist in container"
                )
            # Include some output context for debugging
            output_snippet = output[-500:] if len(output) > 500 else output
            return ToolTestResponse(
                success=False,
                message="Odoo shell test failed - could not verify shell access",
                details={"output_tail": output_snippet}
            )

    except asyncio.TimeoutError:
        return ToolTestResponse(
            success=False,
            message="Connection timed out (shell initialization may need >2 minutes)"
        )
    except FileNotFoundError:
        return ToolTestResponse(
            success=False,
            message="Docker command not found"
        )
    except Exception as e:
        return ToolTestResponse(
            success=False,
            message=f"Connection test failed: {str(e)}"
        )


async def _test_ssh_connection(config: dict) -> ToolTestResponse:
    """Test SSH shell connection using Paramiko."""
    import asyncio
    from ragtime.core.ssh import SSHConfig, test_ssh_connection

    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")
    key_path = config.get("key_path")
    key_content = config.get("key_content")
    key_passphrase = config.get("key_passphrase")
    password = config.get("password")

    if not host or not user:
        return ToolTestResponse(
            success=False,
            message="Host and user are required"
        )

    ssh_config = SSHConfig(
        host=host,
        port=port,
        user=user,
        password=password if password else None,
        key_path=key_path if key_path else None,
        key_content=key_content if key_content else None,
        key_passphrase=key_passphrase if key_passphrase else None,
        timeout=15,
    )

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: test_ssh_connection(ssh_config))

        if result.success:
            return ToolTestResponse(
                success=True,
                message="SSH connection successful",
                details={"host": host, "user": user, "port": port}
            )
        else:
            return ToolTestResponse(
                success=False,
                message=f"SSH connection failed: {result.stderr or result.stdout}"
            )

    except Exception as e:
        return ToolTestResponse(
            success=False,
            message=f"SSH test failed: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Docker Discovery (Networks and Containers)
# -----------------------------------------------------------------------------

class DockerNetwork(BaseModel):
    """Information about a Docker network."""
    name: str
    driver: str
    scope: str
    containers: List[str] = []


class DockerContainer(BaseModel):
    """Information about a Docker container."""
    name: str
    image: str
    status: str
    networks: List[str] = []
    has_odoo: bool = False


class DockerDiscoveryResponse(BaseModel):
    """Response from Docker discovery."""
    success: bool
    message: str
    networks: List[DockerNetwork] = []
    containers: List[DockerContainer] = []
    current_network: Optional[str] = None
    current_container: Optional[str] = None


@router.get("/docker/discover", response_model=DockerDiscoveryResponse, tags=["Tools"])
async def discover_docker_resources():
    """
    Discover Docker networks and containers for tool configuration.

    Returns available networks, running containers, and which containers have Odoo.
    Also detects the current ragtime container's network.
    """
    import asyncio
    import subprocess
    import json

    networks = []
    containers = []
    current_network = None
    current_container = None

    try:
        # Get our own container name by querying Docker
        import os
        hostname = os.environ.get("HOSTNAME", "")
        if hostname:
            # The hostname in Docker is typically the container ID
            # Get the actual container name from the ID
            cmd = ["docker", "inspect", "-f", "{{.Name}}", hostname]
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, _ = await process.communicate()
            if process.returncode == 0:
                # Remove leading slash from container name
                current_container = stdout.decode().strip().lstrip("/")

        # Get networks
        cmd = ["docker", "network", "ls", "--format", "{{json .}}"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            timeout=10.0
        )
        stdout, _ = await process.communicate()

        if process.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    try:
                        net = json.loads(line)
                        # Skip default networks
                        if net.get("Name") not in ["bridge", "host", "none"]:
                            networks.append(DockerNetwork(
                                name=net.get("Name", ""),
                                driver=net.get("Driver", ""),
                                scope=net.get("Scope", "")
                            ))
                    except json.JSONDecodeError:
                        continue

        # Get running containers
        cmd = ["docker", "ps", "--format", "{{json .}}"]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            timeout=10.0
        )
        stdout, _ = await process.communicate()

        if process.returncode == 0:
            for line in stdout.decode().strip().split("\n"):
                if line:
                    try:
                        cont = json.loads(line)
                        container_name = cont.get("Names", "")

                        # Get networks for this container
                        net_cmd = ["docker", "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}", container_name]
                        net_process = await asyncio.create_subprocess_exec(
                            *net_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                        )
                        net_stdout, _ = await net_process.communicate()
                        container_networks = net_stdout.decode().strip().split() if net_process.returncode == 0 else []

                        # Check if this is the ragtime container
                        if current_container and container_name == current_container:
                            current_network = container_networks[0] if container_networks else None

                        # Check if container has Odoo (simple version check)
                        has_odoo = False
                        if "odoo" in cont.get("Image", "").lower() or "odoo" in container_name.lower():
                            try:
                                ver_cmd = ["docker", "exec", "-i", container_name, "odoo", "--version"]
                                ver_process = await asyncio.wait_for(
                                    asyncio.create_subprocess_exec(
                                        *ver_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                                    ),
                                    timeout=5.0
                                )
                                await ver_process.communicate()
                                has_odoo = ver_process.returncode == 0
                            except Exception:
                                has_odoo = True  # Assume it's Odoo based on name

                        containers.append(DockerContainer(
                            name=container_name,
                            image=cont.get("Image", ""),
                            status=cont.get("Status", ""),
                            networks=container_networks,
                            has_odoo=has_odoo
                        ))
                    except json.JSONDecodeError:
                        continue

        # Add container names to networks
        for network in networks:
            network.containers = [c.name for c in containers if network.name in c.networks]

        return DockerDiscoveryResponse(
            success=True,
            message=f"Found {len(networks)} networks and {len(containers)} containers",
            networks=networks,
            containers=containers,
            current_network=current_network,
            current_container=current_container
        )

    except asyncio.TimeoutError:
        return DockerDiscoveryResponse(
            success=False,
            message="Docker discovery timed out"
        )
    except FileNotFoundError:
        return DockerDiscoveryResponse(
            success=False,
            message="Docker command not found"
        )
    except Exception as e:
        return DockerDiscoveryResponse(
            success=False,
            message=f"Docker discovery failed: {str(e)}"
        )


@router.post("/docker/connect-network", tags=["Tools"])
async def connect_to_network(network_name: str):
    """
    Connect the ragtime container to a Docker network.

    This enables container-to-container communication with services on that network.
    """
    import asyncio
    import subprocess
    import os

    # Get current container name
    container_name = os.environ.get("HOSTNAME", "ragtime-dev")

    try:
        # Check if already connected
        cmd = ["docker", "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}", container_name]
        process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = await process.communicate()

        current_networks = stdout.decode().strip().split() if process.returncode == 0 else []

        if network_name in current_networks:
            return {"success": True, "message": f"Already connected to network '{network_name}'"}

        # Connect to the network
        cmd = ["docker", "network", "connect", network_name, container_name]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            timeout=10.0
        )
        _, stderr = await process.communicate()

        if process.returncode == 0:
            return {"success": True, "message": f"Connected to network '{network_name}'"}
        else:
            error = stderr.decode().strip()
            return {"success": False, "message": f"Failed to connect: {error}"}

    except asyncio.TimeoutError:
        return {"success": False, "message": "Network connection timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.post("/docker/disconnect-network", tags=["Tools"])
async def disconnect_from_network(network_name: str):
    """
    Disconnect the ragtime container from a Docker network.

    Used for cleanup when removing tools that required network access.
    """
    import asyncio
    import subprocess
    import os

    # Get current container name
    container_name = os.environ.get("HOSTNAME", "ragtime-dev")

    # Don't disconnect from default networks
    protected_networks = ["ragtime_default", "bridge", "host", "none"]
    if network_name in protected_networks:
        return {"success": True, "message": f"Network '{network_name}' is protected, skipping disconnect"}

    try:
        # Check if connected
        cmd = ["docker", "inspect", "-f", "{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}", container_name]
        process = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, _ = await process.communicate()

        current_networks = stdout.decode().strip().split() if process.returncode == 0 else []

        if network_name not in current_networks:
            return {"success": True, "message": f"Not connected to network '{network_name}'"}

        # Disconnect from the network
        cmd = ["docker", "network", "disconnect", network_name, container_name]
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE),
            timeout=10.0
        )
        _, stderr = await process.communicate()

        if process.returncode == 0:
            return {"success": True, "message": f"Disconnected from network '{network_name}'"}
        else:
            error = stderr.decode().strip()
            return {"success": False, "message": f"Failed to disconnect: {error}"}

    except asyncio.TimeoutError:
        return {"success": False, "message": "Network disconnection timed out"}
    except Exception as e:
        return {"success": False, "message": str(e)}


# -----------------------------------------------------------------------------
# Ollama Connection Testing
# -----------------------------------------------------------------------------

class OllamaTestRequest(BaseModel):
    """Request to test Ollama connection."""
    protocol: str = Field(default="http", description="Protocol: 'http' or 'https'")
    host: str = Field(default="localhost", description="Ollama server hostname or IP")
    port: int = Field(default=11434, ge=1, le=65535, description="Ollama server port")


class OllamaModel(BaseModel):
    """Information about an available Ollama model."""
    name: str
    modified_at: Optional[str] = None
    size: Optional[int] = None


class OllamaTestResponse(BaseModel):
    """Response from Ollama connection test."""
    success: bool
    message: str
    models: List[OllamaModel] = []
    base_url: str = ""


@router.post("/ollama/test", response_model=OllamaTestResponse, tags=["Settings"])
async def test_ollama_connection(request: OllamaTestRequest):
    """
    Test connection to an Ollama server and retrieve available models.

    Returns a list of available embedding models if the connection is successful.
    """
    base_url = f"{request.protocol}://{request.host}:{request.port}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # First check if the server is reachable
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()

            data = response.json()
            models = []

            # Parse the models list from Ollama API response
            for model in data.get("models", []):
                models.append(OllamaModel(
                    name=model.get("name", ""),
                    modified_at=model.get("modified_at"),
                    size=model.get("size")
                ))

            return OllamaTestResponse(
                success=True,
                message=f"Connected successfully. Found {len(models)} model(s).",
                models=models,
                base_url=base_url
            )

    except httpx.ConnectError:
        return OllamaTestResponse(
            success=False,
            message=f"Cannot connect to Ollama server at {base_url}. Is Ollama running?",
            base_url=base_url
        )
    except httpx.TimeoutException:
        return OllamaTestResponse(
            success=False,
            message=f"Connection to {base_url} timed out.",
            base_url=base_url
        )
    except httpx.HTTPStatusError as e:
        return OllamaTestResponse(
            success=False,
            message=f"HTTP error: {e.response.status_code}",
            base_url=base_url
        )
    except Exception as e:
        return OllamaTestResponse(
            success=False,
            message=f"Connection failed: {str(e)}",
            base_url=base_url
        )


# -----------------------------------------------------------------------------
# LLM Provider Model Fetching
# -----------------------------------------------------------------------------

class LLMModelsRequest(BaseModel):
    """Request to fetch available models from an LLM provider."""
    provider: str = Field(..., description="LLM provider: 'openai' or 'anthropic'")
    api_key: str = Field(..., description="API key for the provider")


class LLMModel(BaseModel):
    """Information about an available LLM model."""
    id: str
    name: str
    created: Optional[int] = None


class LLMModelsResponse(BaseModel):
    """Response from LLM models fetch."""
    success: bool
    message: str
    models: List[LLMModel] = []
    default_model: Optional[str] = None


class AvailableModel(BaseModel):
    """A model available for chat."""
    id: str
    name: str
    provider: str  # 'openai' or 'anthropic'


class AvailableModelsResponse(BaseModel):
    """Response with all available models from configured providers."""
    models: List[AvailableModel] = []
    default_model: Optional[str] = None
    current_model: Optional[str] = None  # Currently selected model in settings
    allowed_models: List[str] = []  # List of allowed model IDs (for settings UI)


# Sensible default models for each provider
OPENAI_DEFAULT_MODEL = "gpt-4o"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Models to prioritize in the list (shown first)
OPENAI_PRIORITY_MODELS = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]
ANTHROPIC_PRIORITY_MODELS = ["claude-sonnet-4-20250514", "claude-opus-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]


@router.post("/llm/models", response_model=LLMModelsResponse, tags=["Settings"])
async def fetch_llm_models(request: LLMModelsRequest):
    """
    Fetch available models from an LLM provider given a valid API key.

    Queries the provider's API and returns a list of available chat/completion models.
    """
    if request.provider == "openai":
        return await _fetch_openai_models(request.api_key)
    elif request.provider == "anthropic":
        return await _fetch_anthropic_models(request.api_key)
    else:
        return LLMModelsResponse(
            success=False,
            message=f"Unknown provider: {request.provider}. Supported: 'openai', 'anthropic'"
        )


async def _fetch_openai_models(api_key: str) -> LLMModelsResponse:
    """Fetch available models from OpenAI API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.openai.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()

            data = response.json()
            models = []

            # Filter for chat-capable models (gpt-* models, excluding embedding/audio/etc)
            for model in data.get("data", []):
                model_id = model.get("id", "")
                # Include GPT models suitable for chat (exclude embeddings, whisper, tts, dall-e, etc.)
                if model_id.startswith("gpt-") and not any(x in model_id for x in ["instruct", "vision", "realtime", "audio"]):
                    models.append(LLMModel(
                        id=model_id,
                        name=model_id,
                        created=model.get("created")
                    ))

            # Sort: priority models first, then alphabetically
            def sort_key(m: LLMModel) -> tuple:
                try:
                    priority_idx = OPENAI_PRIORITY_MODELS.index(m.id)
                except ValueError:
                    priority_idx = 999
                return (priority_idx, m.id)

            models.sort(key=sort_key)

            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} chat model(s).",
                models=models,
                default_model=OPENAI_DEFAULT_MODEL if any(m.id == OPENAI_DEFAULT_MODEL for m in models) else (models[0].id if models else None)
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return LLMModelsResponse(
                success=False,
                message="Invalid API key. Please check your OpenAI API key."
            )
        return LLMModelsResponse(
            success=False,
            message=f"OpenAI API error: {e.response.status_code}"
        )
    except httpx.TimeoutException:
        return LLMModelsResponse(
            success=False,
            message="Request to OpenAI timed out."
        )
    except Exception as e:
        return LLMModelsResponse(
            success=False,
            message=f"Failed to fetch OpenAI models: {str(e)}"
        )


async def _fetch_anthropic_models(api_key: str) -> LLMModelsResponse:
    """Fetch available models from Anthropic API."""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                "https://api.anthropic.com/v1/models",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01"
                }
            )
            response.raise_for_status()

            data = response.json()
            models = []

            for model in data.get("data", []):
                model_id = model.get("id", "")
                display_name = model.get("display_name", model_id)
                models.append(LLMModel(
                    id=model_id,
                    name=display_name,
                    created=None
                ))

            # Sort: priority models first, then alphabetically
            def sort_key(m: LLMModel) -> tuple:
                try:
                    priority_idx = ANTHROPIC_PRIORITY_MODELS.index(m.id)
                except ValueError:
                    priority_idx = 999
                return (priority_idx, m.id)

            models.sort(key=sort_key)

            return LLMModelsResponse(
                success=True,
                message=f"Found {len(models)} model(s).",
                models=models,
                default_model=ANTHROPIC_DEFAULT_MODEL if any(m.id == ANTHROPIC_DEFAULT_MODEL for m in models) else (models[0].id if models else None)
            )

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return LLMModelsResponse(
                success=False,
                message="Invalid API key. Please check your Anthropic API key."
            )
        return LLMModelsResponse(
            success=False,
            message=f"Anthropic API error: {e.response.status_code}"
        )
    except httpx.TimeoutException:
        return LLMModelsResponse(
            success=False,
            message="Request to Anthropic timed out."
        )
    except Exception as e:
        return LLMModelsResponse(
            success=False,
            message=f"Failed to fetch Anthropic models: {str(e)}"
        )


# =============================================================================
# Conversation/Chat Endpoints
# =============================================================================

from ragtime.indexer.models import (
    Conversation,
    ConversationResponse,
    CreateConversationRequest,
    SendMessageRequest,
    ChatMessage,
)


@router.get("/chat/available-models", response_model=AvailableModelsResponse, tags=["Chat"])
async def get_available_chat_models():
    """
    Get all available models from configured LLM providers.

    Returns models from OpenAI and/or Anthropic based on which API keys are configured.
    """
    app_settings = await repository.get_settings()
    if not app_settings:
        return AvailableModelsResponse()

    all_models: List[AvailableModel] = []
    default_model = None

    # Fetch OpenAI models if API key is configured
    if app_settings.openai_api_key and len(app_settings.openai_api_key) > 10:
        try:
            result = await _fetch_openai_models(app_settings.openai_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(AvailableModel(
                        id=m.id,
                        name=m.name,
                        provider="openai"
                    ))
                if not default_model and result.default_model:
                    default_model = result.default_model
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")

    # Fetch Anthropic models if API key is configured
    if app_settings.anthropic_api_key and len(app_settings.anthropic_api_key) > 10:
        try:
            result = await _fetch_anthropic_models(app_settings.anthropic_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(AvailableModel(
                        id=m.id,
                        name=m.name,
                        provider="anthropic"
                    ))
                if not default_model and result.default_model:
                    default_model = result.default_model
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")

    # Use current settings model as default if available
    current_model = app_settings.llm_model
    if current_model and any(m.id == current_model for m in all_models):
        default_model = current_model

    # Filter by allowed models if specified
    allowed_models = app_settings.allowed_chat_models or []
    if allowed_models:
        all_models = [m for m in all_models if m.id in allowed_models]
        # Ensure default model is in allowed list
        if default_model and default_model not in allowed_models:
            default_model = all_models[0].id if all_models else None

    return AvailableModelsResponse(
        models=all_models,
        default_model=default_model,
        current_model=current_model
    )


@router.get("/chat/all-models", response_model=AvailableModelsResponse, tags=["Chat"])
async def get_all_chat_models():
    """
    Get ALL available models from configured LLM providers (unfiltered).

    Used by the settings UI to show all models for selection.
    """
    app_settings = await repository.get_settings()
    if not app_settings:
        return AvailableModelsResponse()

    all_models: List[AvailableModel] = []

    # Fetch OpenAI models if API key is configured
    if app_settings.openai_api_key and len(app_settings.openai_api_key) > 10:
        try:
            result = await _fetch_openai_models(app_settings.openai_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(AvailableModel(
                        id=m.id,
                        name=m.name,
                        provider="openai"
                    ))
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")

    # Fetch Anthropic models if API key is configured
    if app_settings.anthropic_api_key and len(app_settings.anthropic_api_key) > 10:
        try:
            result = await _fetch_anthropic_models(app_settings.anthropic_api_key)
            if result.success:
                for m in result.models:
                    all_models.append(AvailableModel(
                        id=m.id,
                        name=m.name,
                        provider="anthropic"
                    ))
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")

    # Get currently allowed models from settings
    allowed_models = app_settings.allowed_chat_models or []

    return AvailableModelsResponse(
        models=all_models,
        default_model=app_settings.llm_model,
        current_model=app_settings.llm_model,
        allowed_models=allowed_models
    )


@router.get("/conversations", response_model=List[ConversationResponse])
async def list_conversations():
    """List all chat conversations."""
    convs = await repository.list_conversations()
    return [
        ConversationResponse(
            id=c.id,
            title=c.title,
            model=c.model,
            messages=c.messages,
            total_tokens=c.total_tokens,
            created_at=c.created_at,
            updated_at=c.updated_at,
        )
        for c in convs
    ]


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(request: CreateConversationRequest = None):
    """Create a new chat conversation."""
    # Get default model from app settings if not provided
    app_settings = await repository.get_settings()
    default_model = app_settings.llm_model if app_settings else "gpt-4-turbo"

    title = request.title if request and request.title else "New Chat"
    model = request.model if request and request.model else default_model

    conv = await repository.create_conversation(title=title, model=model)
    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get a specific conversation."""
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    success = await repository.delete_conversation(conversation_id)
    if not success:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


@router.patch("/conversations/{conversation_id}/title", response_model=ConversationResponse)
async def update_conversation_title(conversation_id: str, body: dict):
    """Update a conversation's title."""
    title = body.get("title", "").strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title is required")

    conv = await repository.update_conversation_title(conversation_id, title)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.patch("/conversations/{conversation_id}/model", response_model=ConversationResponse)
async def update_conversation_model(conversation_id: str, body: dict):
    """Update a conversation's model."""
    model = body.get("model", "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Model is required")

    conv = await repository.update_conversation_model(conversation_id, model)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.post("/conversations/{conversation_id}/clear", response_model=ConversationResponse)
async def clear_conversation(conversation_id: str):
    """Clear all messages in a conversation."""
    conv = await repository.clear_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.post("/conversations/{conversation_id}/truncate", response_model=ConversationResponse)
async def truncate_conversation(conversation_id: str, keep_count: int):
    """
    Truncate conversation messages to keep only the first N messages.
    Used when editing/resending a message to remove subsequent messages.
    """
    conv = await repository.truncate_messages(conversation_id, keep_count)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=conv.id,
        title=conv.title,
        model=conv.model,
        messages=conv.messages,
        total_tokens=conv.total_tokens,
        created_at=conv.created_at,
        updated_at=conv.updated_at,
    )


@router.post("/conversations/{conversation_id}/messages")
async def send_message(conversation_id: str, request: SendMessageRequest):
    """
    Send a message to a conversation and get a response.
    Non-streaming version.
    """
    from ragtime.rag import rag
    from langchain_core.messages import HumanMessage, AIMessage

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="RAG service initializing, please retry")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Add user message
    conv = await repository.add_message(conversation_id, "user", user_message)

    # Build chat history for RAG
    chat_history = []
    for msg in conv.messages[:-1]:  # Exclude the current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    # Generate response
    try:
        answer = await rag.process_query(user_message, chat_history)
    except Exception as e:
        logger.exception("Error processing message")
        answer = f"Error: {str(e)}"

    # Add assistant response
    conv = await repository.add_message(conversation_id, "assistant", answer)

    # Auto-generate title from first user message if still "New Chat"
    if conv.title == "New Chat" and len(conv.messages) >= 2:
        first_msg = conv.messages[0].content[:50]
        new_title = first_msg + ("..." if len(conv.messages[0].content) > 50 else "")
        conv = await repository.update_conversation_title(conversation_id, new_title)

    return {
        "message": ChatMessage(
            role="assistant",
            content=answer,
            timestamp=conv.messages[-1].timestamp,
        ),
        "conversation": ConversationResponse(
            id=conv.id,
            title=conv.title,
            model=conv.model,
            messages=conv.messages,
            total_tokens=conv.total_tokens,
            created_at=conv.created_at,
            updated_at=conv.updated_at,
        ),
    }


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(conversation_id: str, request: SendMessageRequest):
    """
    Send a message to a conversation and stream the response.
    Returns SSE stream of tokens.
    """
    import json
    import time
    from fastapi.responses import StreamingResponse
    from ragtime.rag import rag
    from langchain_core.messages import HumanMessage, AIMessage

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="RAG service initializing, please retry")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Add user message
    await repository.add_message(conversation_id, "user", user_message)

    # Refresh conversation to get updated messages
    conv = await repository.get_conversation(conversation_id)

    # Build chat history for RAG
    chat_history = []
    for msg in conv.messages[:-1]:  # Exclude current message
        if msg.role == "user":
            chat_history.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            chat_history.append(AIMessage(content=msg.content))

    async def stream_response():
        """Generate streaming response tokens."""
        chunk_id = f"chatcmpl-{int(time.time())}"
        full_response = ""
        tool_calls_collected = []  # Collect tool calls for storage (deprecated)
        chronological_events = []  # Collect events in order (content and tools)
        current_tool_call = None   # Track current tool call being built

        try:
            async for event in rag.process_query_stream(user_message, chat_history):
                # Handle structured tool events
                if isinstance(event, dict):
                    event_type = event.get("type")
                    if event_type == "tool_start":
                        # Start tracking a new tool call
                        current_tool_call = {
                            "type": "tool",
                            "tool": event.get("tool"),
                            "input": event.get("input")
                        }
                        tool_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": conv.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_call": {
                                        "type": "start",
                                        "tool": event.get("tool"),
                                        "input": event.get("input")
                                    }
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(tool_chunk)}\n\n"
                    elif event_type == "tool_end":
                        # Complete the current tool call and save it
                        if current_tool_call:
                            current_tool_call["output"] = event.get("output")
                            chronological_events.append(current_tool_call)
                            # Also keep deprecated tool_calls format for backward compatibility
                            tool_calls_collected.append({
                                "tool": current_tool_call["tool"],
                                "input": current_tool_call.get("input"),
                                "output": current_tool_call.get("output")
                            })
                            current_tool_call = None
                        tool_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": conv.model,
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "tool_call": {
                                        "type": "end",
                                        "tool": event.get("tool"),
                                        "output": event.get("output")
                                    }
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(tool_chunk)}\n\n"
                else:
                    # Handle regular text tokens
                    token = event
                    full_response += token
                    # Add content events when we get text (batch them for efficiency)
                    if chronological_events and chronological_events[-1].get("type") == "content":
                        # Append to last content event
                        chronological_events[-1]["content"] += token
                    else:
                        # Start new content event
                        chronological_events.append({"type": "content", "content": token})

                    chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": conv.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            # Save the full response with tool calls and chronological events
            updated_conv = await repository.add_message(
                conversation_id,
                "assistant",
                full_response,
                tool_calls=tool_calls_collected if tool_calls_collected else None,
                events=chronological_events if chronological_events else None
            )

            # Auto-generate title if needed
            if updated_conv and updated_conv.title == "New Chat" and len(updated_conv.messages) >= 2:
                first_msg = updated_conv.messages[0].content[:50]
                new_title = first_msg + ("..." if len(updated_conv.messages[0].content) > 50 else "")
                await repository.update_conversation_title(conversation_id, new_title)

            # Final chunk
            final_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": conv.model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("Error in streaming response")

            raw_error = str(e)
            friendly_error = "An error occurred while generating the response. Please try again."

            max_iters = None
            if rag and getattr(rag, "agent_executor", None):
                max_iters = getattr(rag.agent_executor, "max_iterations", None)

            if "iteration" in raw_error.lower() or "max iterations" in raw_error.lower():
                limit_text = f" ({max_iters})" if max_iters else ""
                friendly_error = f"Stopped after reaching the max_iterations limit{limit_text}. Please narrow the request or retry."

            # Include any in-progress tool call that didn't complete
            if current_tool_call:
                current_tool_call["output"] = "(interrupted)"
                chronological_events.append(current_tool_call)
                # Also add to deprecated format
                tool_calls_collected.append({
                    "tool": current_tool_call["tool"],
                    "input": current_tool_call.get("input"),
                    "output": "(interrupted)"
                })

            # Persist whatever we have so far, including collected tool calls and events
            combined_response = full_response.strip()
            if friendly_error:
                combined_response = f"{combined_response}\n\n{friendly_error}" if combined_response else friendly_error

            try:
                await repository.add_message(
                    conversation_id,
                    "assistant",
                    combined_response,
                    tool_calls=tool_calls_collected if tool_calls_collected else None,
                    events=chronological_events if chronological_events else None
                )
            except Exception:
                logger.exception("Failed to persist assistant message after error")

            error_chunk = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": conv.model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": f"\n\n{friendly_error}"},
                    "finish_reason": "error"
                }]
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream"
    )


# =============================================================================
# Background Chat Task Endpoints
# =============================================================================

from ragtime.indexer.models import ChatTask, ChatTaskResponse, ChatTaskStatus


@router.post("/conversations/{conversation_id}/messages/background", response_model=ChatTaskResponse)
async def send_message_background(conversation_id: str, request: SendMessageRequest):
    """
    Send a message to a conversation and process it in the background.
    Returns a task object that can be polled for status and results.
    """
    from ragtime.rag import rag
    from ragtime.indexer.background_tasks import background_task_service

    # Get conversation
    conv = await repository.get_conversation(conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not rag.is_ready:
        raise HTTPException(status_code=503, detail="RAG service initializing, please retry")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message is required")

    # Check if there's already an active task
    existing_task = await repository.get_active_task_for_conversation(conversation_id)
    if existing_task:
        # Return the existing task instead of creating a new one
        return ChatTaskResponse(
            id=existing_task.id,
            conversation_id=existing_task.conversation_id,
            status=existing_task.status,
            user_message=existing_task.user_message,
            streaming_state=existing_task.streaming_state,
            response_content=existing_task.response_content,
            error_message=existing_task.error_message,
            created_at=existing_task.created_at,
            started_at=existing_task.started_at,
            completed_at=existing_task.completed_at,
            last_update_at=existing_task.last_update_at,
        )

    # Add user message to conversation first
    await repository.add_message(conversation_id, "user", user_message)

    # Start background task
    task_id = await background_task_service.start_task_async(conversation_id, user_message)

    # Get the created task
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=500, detail="Failed to create background task")

    return ChatTaskResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        user_message=task.user_message,
        streaming_state=task.streaming_state,
        response_content=task.response_content,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_update_at=task.last_update_at,
    )


@router.get("/conversations/{conversation_id}/task", response_model=Optional[ChatTaskResponse])
async def get_conversation_active_task(conversation_id: str):
    """
    Get the active (pending/running) task for a conversation, if any.
    Returns null if no active task.
    """
    task = await repository.get_active_task_for_conversation(conversation_id)
    if not task:
        return None

    return ChatTaskResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        user_message=task.user_message,
        streaming_state=task.streaming_state,
        response_content=task.response_content,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_update_at=task.last_update_at,
    )


@router.get("/tasks/{task_id}", response_model=ChatTaskResponse)
async def get_chat_task(task_id: str):
    """
    Get a chat task by ID.
    Use this to poll for task status and streaming state.
    """
    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return ChatTaskResponse(
        id=task.id,
        conversation_id=task.conversation_id,
        status=task.status,
        user_message=task.user_message,
        streaming_state=task.streaming_state,
        response_content=task.response_content,
        error_message=task.error_message,
        created_at=task.created_at,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_update_at=task.last_update_at,
    )


@router.post("/tasks/{task_id}/cancel", response_model=ChatTaskResponse)
async def cancel_chat_task(task_id: str):
    """
    Cancel a running chat task.
    """
    from ragtime.indexer.background_tasks import background_task_service

    task = await repository.get_chat_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if task.status not in (ChatTaskStatus.pending, ChatTaskStatus.running):
        raise HTTPException(status_code=400, detail="Task is not running")

    # Cancel the task
    background_task_service.cancel_task(task_id)
    updated_task = await repository.cancel_chat_task(task_id)

    if not updated_task:
        raise HTTPException(status_code=500, detail="Failed to cancel task")

    return ChatTaskResponse(
        id=updated_task.id,
        conversation_id=updated_task.conversation_id,
        status=updated_task.status,
        user_message=updated_task.user_message,
        streaming_state=updated_task.streaming_state,
        response_content=updated_task.response_content,
        error_message=updated_task.error_message,
        created_at=updated_task.created_at,
        started_at=updated_task.started_at,
        completed_at=updated_task.completed_at,
        last_update_at=updated_task.last_update_at,
    )
