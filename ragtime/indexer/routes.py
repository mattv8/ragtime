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


@router.patch("/{name}/toggle")
async def toggle_index(name: str, request: ToggleIndexRequest):
    """Toggle an index's enabled status for RAG context."""
    success = await repository.set_index_enabled(name, request.enabled)
    if not success:
        raise HTTPException(status_code=404, detail="Index not found")
    return {"message": f"Index '{name}' {'enabled' if request.enabled else 'disabled'}", "enabled": request.enabled}


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
    """Delete a tool configuration."""
    success = await repository.delete_tool_config(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool configuration not found")
    return {"message": "Tool configuration deleted"}


@router.post("/tools/{tool_id}/toggle", tags=["Tools"])
async def toggle_tool_config(tool_id: str, enabled: bool):
    """Toggle a tool's enabled status."""
    config = await repository.update_tool_config(tool_id, {"enabled": enabled})
    if config is None:
        raise HTTPException(status_code=404, detail="Tool configuration not found")
    return {"enabled": config.enabled}


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
    """Test Odoo shell connection."""
    import asyncio
    import subprocess

    container = config.get("container", "")
    database = config.get("database", "odoo")
    config_path = config.get("config_path", "/etc/odoo/odoo.conf")
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
        # Build command - only include -c config_path if it's specified
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
                    "docker_network": docker_network
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
                message=f"Odoo shell test failed - could not verify shell access",
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
    """Test SSH shell connection."""
    import asyncio
    import subprocess

    host = config.get("host", "")
    port = config.get("port", 22)
    user = config.get("user", "")
    key_path = config.get("key_path")

    if not host or not user:
        return ToolTestResponse(
            success=False,
            message="Host and user are required"
        )

    try:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
        if key_path:
            cmd.extend(["-i", key_path])
        cmd.extend(["-p", str(port), f"{user}@{host}", "echo 'Connection successful'"])

        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                *cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ),
            timeout=15.0
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            return ToolTestResponse(
                success=True,
                message="SSH connection successful",
                details={"host": host, "user": user, "port": port}
            )
        else:
            error = stderr.decode("utf-8", errors="replace").strip()
            return ToolTestResponse(
                success=False,
                message=f"SSH connection failed: {error}"
            )

    except asyncio.TimeoutError:
        return ToolTestResponse(
            success=False,
            message="SSH connection timed out"
        )
    except FileNotFoundError:
        return ToolTestResponse(
            success=False,
            message="SSH command not found"
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

    try:
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
                        if container_name in ["ragtime-dev", "ragtime"]:
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
            current_network=current_network
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
