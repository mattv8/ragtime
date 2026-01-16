"""
MCP Route Configuration API - Manage custom MCP routes.

Provides CRUD endpoints for managing custom MCP routes with tool selection
and authorization configuration.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ragtime.core.database import get_db
from ragtime.core.encryption import decrypt_secret, encrypt_secret
from ragtime.core.logging import get_logger
from ragtime.core.security import require_admin
from ragtime.mcp.server import notify_tools_changed

logger = get_logger(__name__)

router = APIRouter(prefix="/mcp-routes", tags=["MCP Routes"])


# =============================================================================
# Pydantic Models
# =============================================================================


class McpRouteConfig(BaseModel):
    """MCP route configuration response model."""

    id: str
    name: str
    route_path: str
    description: str
    enabled: bool
    require_auth: bool
    has_password: bool = False  # Whether a password is set
    auth_password: str | None = None  # Decrypted password (if requested)
    auth_method: str = "password"  # "password" or "oauth2"
    allowed_ldap_group: str | None = None  # LDAP group DN for OAuth2 auth
    include_knowledge_search: bool
    include_git_history: bool
    selected_document_indexes: List[str] = Field(default_factory=list)
    selected_filesystem_indexes: List[str] = Field(default_factory=list)
    selected_schema_indexes: List[str] = Field(default_factory=list)
    tool_config_ids: List[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class CreateMcpRouteRequest(BaseModel):
    """Request model for creating an MCP route."""

    name: str = Field(
        min_length=1, max_length=100, description="User-friendly name for the route"
    )
    route_path: str = Field(
        min_length=1,
        max_length=50,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Route suffix (e.g., 'my_toolset' for /mcp/my_toolset). Lowercase alphanumeric and underscores only.",
    )
    description: str = Field(
        default="", max_length=500, description="Description for documentation"
    )
    require_auth: bool = Field(
        default=False, description="Require authentication for this route"
    )
    auth_password: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="Password/API key for authentication",
    )
    auth_method: str = Field(
        default="password",
        description="Authentication method: 'password' or 'oauth2'",
    )
    allowed_ldap_group: str | None = Field(
        default=None,
        max_length=500,
        description="LDAP group DN for OAuth2 authorization (e.g., 'CN=MCP Users,OU=Groups,DC=example,DC=com')",
    )
    include_knowledge_search: bool = Field(
        default=True, description="Include search_knowledge tool"
    )
    include_git_history: bool = Field(
        default=True, description="Include git history tools"
    )
    selected_document_indexes: List[str] = Field(
        default_factory=list, description="Document index names to include"
    )
    selected_filesystem_indexes: List[str] = Field(
        default_factory=list, description="Filesystem tool IDs to include"
    )
    selected_schema_indexes: List[str] = Field(
        default_factory=list, description="Schema index tool IDs to include"
    )
    tool_config_ids: List[str] = Field(
        default_factory=list, description="IDs of tool configs to include"
    )


class UpdateMcpRouteRequest(BaseModel):
    """Request model for updating an MCP route."""

    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    enabled: bool | None = None
    require_auth: bool | None = None
    auth_password: str | None = Field(
        default=None,
        min_length=8,
        max_length=128,
        description="Password/API key (set to empty string to clear)",
    )
    clear_password: bool = Field(
        default=False, description="Set to true to clear the password"
    )
    auth_method: str | None = Field(
        default=None,
        description="Authentication method: 'password' or 'oauth2'",
    )
    allowed_ldap_group: str | None = Field(
        default=None,
        max_length=500,
        description="LDAP group DN for OAuth2 authorization",
    )
    clear_allowed_ldap_group: bool = Field(
        default=False, description="Set to true to clear the allowed LDAP group"
    )
    include_knowledge_search: bool | None = None
    include_git_history: bool | None = None
    selected_document_indexes: List[str] | None = None
    selected_filesystem_indexes: List[str] | None = None
    selected_schema_indexes: List[str] | None = None
    tool_config_ids: List[str] | None = None


class McpRouteListResponse(BaseModel):
    """Response model for listing MCP routes."""

    routes: List[McpRouteConfig]
    count: int


# =============================================================================
# API Routes
# =============================================================================


@router.get("", response_model=McpRouteListResponse)
async def list_mcp_routes(_user=Depends(require_admin)):
    """List all custom MCP routes. Admin only."""
    db = await get_db()

    routes = await db.mcprouteconfig.find_many(
        order={"createdAt": "desc"},
        include={"toolSelections": True},
    )

    result = []
    for route in routes:
        tool_ids = (
            [sel.toolConfigId for sel in route.toolSelections]
            if route.toolSelections
            else []
        )
        # Decrypt password for admin display
        decrypted_password = (
            decrypt_secret(route.authPassword) if route.authPassword else None
        )
        result.append(
            McpRouteConfig(
                id=route.id,
                name=route.name,
                route_path=route.routePath,
                description=route.description,
                enabled=route.enabled,
                require_auth=route.requireAuth,
                has_password=bool(route.authPassword),
                auth_password=decrypted_password,
                auth_method=route.authMethod or "password",
                allowed_ldap_group=route.allowedLdapGroup,
                include_knowledge_search=route.includeKnowledgeSearch,
                include_git_history=route.includeGitHistory,
                selected_document_indexes=route.selectedDocumentIndexes or [],
                selected_filesystem_indexes=route.selectedFilesystemIndexes or [],
                selected_schema_indexes=route.selectedSchemaIndexes or [],
                tool_config_ids=tool_ids,
                created_at=route.createdAt.isoformat(),
                updated_at=route.updatedAt.isoformat(),
            )
        )

    return McpRouteListResponse(routes=result, count=len(result))


@router.get("/{route_id}", response_model=McpRouteConfig)
async def get_mcp_route(route_id: str, _user=Depends(require_admin)):
    """Get a specific MCP route by ID. Admin only."""
    db = await get_db()

    route = await db.mcprouteconfig.find_unique(
        where={"id": route_id},
        include={"toolSelections": True},
    )

    if not route:
        raise HTTPException(status_code=404, detail="MCP route not found")

    tool_ids = (
        [sel.toolConfigId for sel in route.toolSelections]
        if route.toolSelections
        else []
    )

    # Decrypt password for admin display
    decrypted_password = (
        decrypt_secret(route.authPassword) if route.authPassword else None
    )

    return McpRouteConfig(
        id=route.id,
        name=route.name,
        route_path=route.routePath,
        description=route.description,
        enabled=route.enabled,
        require_auth=route.requireAuth,
        has_password=bool(route.authPassword),
        auth_password=decrypted_password,
        auth_method=route.authMethod or "password",
        allowed_ldap_group=route.allowedLdapGroup,
        include_knowledge_search=route.includeKnowledgeSearch,
        include_git_history=route.includeGitHistory,
        selected_document_indexes=route.selectedDocumentIndexes or [],
        selected_filesystem_indexes=route.selectedFilesystemIndexes or [],
        selected_schema_indexes=route.selectedSchemaIndexes or [],
        tool_config_ids=tool_ids,
        created_at=route.createdAt.isoformat(),
        updated_at=route.updatedAt.isoformat(),
    )


@router.post("", response_model=McpRouteConfig)
async def create_mcp_route(
    request: CreateMcpRouteRequest, _user=Depends(require_admin)
):
    """Create a new custom MCP route. Admin only."""
    db = await get_db()

    # Check if route_path already exists
    existing = await db.mcprouteconfig.find_unique(
        where={"routePath": request.route_path}
    )
    if existing:
        raise HTTPException(
            status_code=400,
            detail=f"Route path '{request.route_path}' already exists",
        )

    # Validate tool config IDs exist
    if request.tool_config_ids:
        for tool_id in request.tool_config_ids:
            tool = await db.toolconfig.find_unique(where={"id": tool_id})
            if not tool:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tool config '{tool_id}' not found",
                )

    # Create the route
    create_data = {
        "name": request.name,
        "routePath": request.route_path,
        "description": request.description,
        "requireAuth": request.require_auth,
        "authMethod": request.auth_method,
        "allowedLdapGroup": request.allowed_ldap_group,
        "includeKnowledgeSearch": request.include_knowledge_search,
        "includeGitHistory": request.include_git_history,
        "selectedDocumentIndexes": request.selected_document_indexes,
        "selectedFilesystemIndexes": request.selected_filesystem_indexes,
        "selectedSchemaIndexes": request.selected_schema_indexes,
    }

    # Encrypt password if provided
    if request.auth_password:
        create_data["authPassword"] = encrypt_secret(request.auth_password)

    route = await db.mcprouteconfig.create(data=create_data)

    # Add tool selections
    for tool_id in request.tool_config_ids:
        await db.mcproutetoolselection.create(
            data={
                "mcpRouteId": route.id,
                "toolConfigId": tool_id,
            }
        )

    logger.info(f"Created MCP route: {route.name} at /mcp/{route.routePath}")
    notify_tools_changed()

    # Return the decrypted password in the response
    decrypted_password = (
        decrypt_secret(route.authPassword) if route.authPassword else None
    )

    return McpRouteConfig(
        id=route.id,
        name=route.name,
        route_path=route.routePath,
        description=route.description,
        enabled=route.enabled,
        require_auth=route.requireAuth,
        has_password=bool(route.authPassword),
        auth_password=decrypted_password,
        auth_method=route.authMethod or "password",
        allowed_ldap_group=route.allowedLdapGroup,
        include_knowledge_search=route.includeKnowledgeSearch,
        include_git_history=route.includeGitHistory,
        selected_document_indexes=route.selectedDocumentIndexes or [],
        selected_filesystem_indexes=route.selectedFilesystemIndexes or [],
        selected_schema_indexes=route.selectedSchemaIndexes or [],
        tool_config_ids=request.tool_config_ids,
        created_at=route.createdAt.isoformat(),
        updated_at=route.updatedAt.isoformat(),
    )


@router.put("/{route_id}", response_model=McpRouteConfig)
async def update_mcp_route(
    route_id: str, request: UpdateMcpRouteRequest, _user=Depends(require_admin)
):
    """Update an existing MCP route. Admin only."""
    db = await get_db()

    # Check route exists
    existing = await db.mcprouteconfig.find_unique(where={"id": route_id})
    if not existing:
        raise HTTPException(status_code=404, detail="MCP route not found")

    # Build update data
    update_data: dict = {}
    if request.name is not None:
        update_data["name"] = request.name
    if request.description is not None:
        update_data["description"] = request.description
    if request.enabled is not None:
        update_data["enabled"] = request.enabled
    if request.require_auth is not None:
        update_data["requireAuth"] = request.require_auth
    if request.include_knowledge_search is not None:
        update_data["includeKnowledgeSearch"] = request.include_knowledge_search
    if request.include_git_history is not None:
        update_data["includeGitHistory"] = request.include_git_history
    if request.selected_document_indexes is not None:
        update_data["selectedDocumentIndexes"] = request.selected_document_indexes
    if request.selected_filesystem_indexes is not None:
        update_data["selectedFilesystemIndexes"] = request.selected_filesystem_indexes
    if request.selected_schema_indexes is not None:
        update_data["selectedSchemaIndexes"] = request.selected_schema_indexes

    # Handle password update
    if request.clear_password:
        update_data["authPassword"] = None
    elif request.auth_password is not None:
        update_data["authPassword"] = encrypt_secret(request.auth_password)

    # Handle auth method
    if request.auth_method is not None:
        update_data["authMethod"] = request.auth_method

    # Handle allowed LDAP group
    if request.clear_allowed_ldap_group:
        update_data["allowedLdapGroup"] = None
    elif request.allowed_ldap_group is not None:
        update_data["allowedLdapGroup"] = request.allowed_ldap_group

    # Update route
    if update_data:
        await db.mcprouteconfig.update(where={"id": route_id}, data=update_data)

    # Update tool selections if provided
    if request.tool_config_ids is not None:
        # Validate tool config IDs exist
        for tool_id in request.tool_config_ids:
            tool = await db.toolconfig.find_unique(where={"id": tool_id})
            if not tool:
                raise HTTPException(
                    status_code=400,
                    detail=f"Tool config '{tool_id}' not found",
                )

        # Delete existing selections
        await db.mcproutetoolselection.delete_many(where={"mcpRouteId": route_id})

        # Add new selections
        for tool_id in request.tool_config_ids:
            await db.mcproutetoolselection.create(
                data={
                    "mcpRouteId": route_id,
                    "toolConfigId": tool_id,
                }
            )

    # Fetch updated route
    route = await db.mcprouteconfig.find_unique(
        where={"id": route_id},
        include={"toolSelections": True},
    )

    tool_ids = (
        [sel.toolConfigId for sel in route.toolSelections]
        if route and route.toolSelections
        else []
    )

    logger.info(f"Updated MCP route: {route.name if route else route_id}")
    notify_tools_changed()

    if not route:
        return None

    # Decrypt password for admin display
    decrypted_password = (
        decrypt_secret(route.authPassword) if route.authPassword else None
    )

    return McpRouteConfig(
        id=route.id,
        name=route.name,
        route_path=route.routePath,
        description=route.description,
        enabled=route.enabled,
        require_auth=route.requireAuth,
        has_password=bool(route.authPassword),
        auth_password=decrypted_password,
        auth_method=route.authMethod or "password",
        allowed_ldap_group=route.allowedLdapGroup,
        include_knowledge_search=route.includeKnowledgeSearch,
        include_git_history=route.includeGitHistory,
        selected_document_indexes=route.selectedDocumentIndexes or [],
        selected_filesystem_indexes=route.selectedFilesystemIndexes or [],
        selected_schema_indexes=route.selectedSchemaIndexes or [],
        tool_config_ids=tool_ids,
        created_at=route.createdAt.isoformat(),
        updated_at=route.updatedAt.isoformat(),
    )


@router.delete("/{route_id}")
async def delete_mcp_route(route_id: str, _user=Depends(require_admin)):
    """Delete an MCP route. Admin only."""
    db = await get_db()

    # Check route exists
    existing = await db.mcprouteconfig.find_unique(where={"id": route_id})
    if not existing:
        raise HTTPException(status_code=404, detail="MCP route not found")

    await db.mcprouteconfig.delete(where={"id": route_id})

    logger.info(f"Deleted MCP route: {existing.name} at /mcp/{existing.routePath}")
    notify_tools_changed()

    return {"status": "deleted", "id": route_id}


@router.get("/by-path/{route_path}", response_model=McpRouteConfig)
async def get_mcp_route_by_path(route_path: str, _user=Depends(require_admin)):
    """Get a specific MCP route by path. Admin only."""
    db = await get_db()

    route = await db.mcprouteconfig.find_unique(
        where={"routePath": route_path},
        include={"toolSelections": True},
    )

    if not route:
        raise HTTPException(status_code=404, detail="MCP route not found")

    tool_ids = (
        [sel.toolConfigId for sel in route.toolSelections]
        if route.toolSelections
        else []
    )

    # Decrypt password for admin display
    decrypted_password = (
        decrypt_secret(route.authPassword) if route.authPassword else None
    )

    return McpRouteConfig(
        id=route.id,
        name=route.name,
        route_path=route.routePath,
        description=route.description,
        enabled=route.enabled,
        require_auth=route.requireAuth,
        has_password=bool(route.authPassword),
        auth_password=decrypted_password,
        auth_method=route.authMethod or "password",
        allowed_ldap_group=route.allowedLdapGroup,
        include_knowledge_search=route.includeKnowledgeSearch,
        include_git_history=route.includeGitHistory,
        selected_document_indexes=route.selectedDocumentIndexes or [],
        selected_filesystem_indexes=route.selectedFilesystemIndexes or [],
        selected_schema_indexes=route.selectedSchemaIndexes or [],
        tool_config_ids=tool_ids,
        created_at=route.createdAt.isoformat(),
        updated_at=route.updatedAt.isoformat(),
    )
