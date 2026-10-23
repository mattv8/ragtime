# MCP surfaces: `/mcp` BYOH vs `/mcp/<custom_route>`

Scope: MCP transport, route authentication, the bring-your-own-harness (BYOH)
workspace-development lane, and diagnosing "tool missing" reports from external
agents. Read this before recommending a connector URL or changing MCP auth.

Ragtime has three external-agent surfaces. Do not substitute one for another:

| Surface | Purpose | Identity | Tools |
|---|---|---|---|
| `/mcp` (default route) | BYOH workspace development, plus the general tool catalog for admitted users | Depends on default-route auth settings (see below) | General tools when admitted; workspace-development tools only when a development principal resolves |
| `/mcp/<route_path>` (custom route) | A curated general-tool catalog for one integration | The route's own auth method and optional allowed group | Only the route's selected tools; never workspace-development tools |
| `/agent/w/{token}` | External build briefs | Agent token acting as its creator | Manifest-advertised HTTP routes; see `external-agent-tasks.md` |

## `/mcp` BYOH is intentional

- BYOH deliberately uses the shared default `/mcp` endpoint. Setup callers
  (`ConnectYourAgentPanel`, `UserSpacePanel`) pass `${origin}/mcp` to the client
  guides. Never recommend a custom route as the fix for a BYOH connector.
  MCP OAuth tokens are bound to the resource path, so a `/mcp` grant does not
  work on `/mcp/<route>` anyway.
- `MCPTransportEndpoint` (`ragtime/mcp/routes.py`) resolves the caller.
  `_resolve_default_development_principal` is authoritative:
  - `rtdev_` credential: `workspace_development` lane. Bound to one workspace and
    its scopes; general tools are hidden.
  - Default-route auth `oauth2`, user in the allowed group: `oauth2`. General
    tools (narrowed by a matching default-route group filter, if any) plus
    workspace-development tools.
  - Default-route auth `oauth2`, user outside the allowed group: admitted as
    `oauth2_development` on `/mcp` only. Workspace-development tools only;
    general tools are neither listed nor callable.
  - Default-route auth `password`, `client_credentials`, or disabled: no
    development principal, so no workspace-development tools. BYOH OAuth setup
    needs default-route auth enabled with method `oauth2`.
- `general_tools_allowed` on `DevelopmentPrincipal` controls only the general
  catalog. Workspace authorization (membership role, credential scopes,
  workspace tool selection, and tool ACLs on the `workspace` surface) is
  checked independently on every development operation.
- When an `oauth2_development` user is supposed to have general tools, the fix is
  the default route's allowed group. That is an admin decision after confirming
  intent. Do not change group membership or route auth as an agent workaround
  for a BYOH user who only needs workspace resources.

## `/mcp/<custom_route>` is a curated catalog

- `MCPCustomRouteEndpoint` rejects `rtdev_` credentials with 401
  ("Workspace development credentials are accepted only on /mcp").
- An OAuth group mismatch on a custom route is a 401. There is no
  development-only fallback.
- Custom routes register handlers with `development_tools=False`; calling a
  `workspace_development*` name there returns `tool_not_available`.

## Diagnosing "the Odoo/SQL tool disappeared"

1. Establish the connector URL and identity before concluding a tool is
   disabled, unhealthy, or offline. MCP disabled returns 503 on every route.
2. A tool list of only `workspace_development*` names (see
   `_DEVELOPMENT_TOOL_NAMES` in `ragtime/mcp/server.py`) means the caller is in
   a development-only lane, not that general tools are broken. User principals
   without a credential binding (OAuth or session) also get
   `workspace_development_workspaces` and `workspace_development_create`;
   `rtdev_` credentials never do.
3. Check `mcp_request_logs.auth_method`. The values are whatever `_log_mcp`
   receives in `routes.py`, including `oauth2_development` and
   `workspace_development`; read the code rather than trusting a cached list.
4. Workspace tools require a workspace UUID. Setup supplies it in the
   verification prompt or copied instructions; user principals can list
   workspaces with `workspace_development_workspaces`.
5. If a selected tool is absent from `workspace_development_resources`, check
   in this order: the workspace tool selection (directly or via tool group);
   the tool is enabled and healthy; the caller's `workspace`-surface tool ACL is
   `read` or `read_write`; the workspace owner's tool access; and whether the
   canonical runtime tool builds (`_enrich_authorized_tool` drops failures).

## Selected tools inside BYOH

- Workspace-selected tools appear as resources:
  `workspace_development_context` -> `workspace_development_resources` ->
  `workspace_development_resource_contract`. They are not top-level MCP tools.
- `execute_component` has two lanes. `development_service.py` chooses the lane
  by tool type; the type sets live in `ragtime/userspace/service.py`:
  - Browser read-only lane (`_USPACE_EXEC_BROWSER_SUPPORTED_TOOL_TYPES`: SQL and
    HTTP API tools). Needs only `read` scope.
  - Server-direct lane (`_USPACE_DEVELOPMENT_SERVER_LANE_TOOL_TYPES`, currently
    `odoo_shell`) via `execute_component_from_development_server_lane`. Needs
    `exec` scope and workspace editor role, the same gate as `exec_start`. It
    adds no capability beyond the runtime bridge; it skips starting a runtime.
    Writes require tool `allow_write`, the workspace write opt-in, owner
    `read_write`, caller workspace-surface `read_write`, and `write` scope.
    Odoo "read" is best-effort validation of Python code, not a sandbox.
- Every execution lane caps a single call at the proxy-safe HTTP timeout
  (`get_http_proxy_safe_timeout_seconds`); the server-direct lane sets the
  Odoo tool's own timeout just below it so the shell process is terminated.
  Split long work into smaller calls. The runtime bridge does not raise the
  per-call limit.
- The server runtime bridge remains the path for app-embedded server code:
  start the runtime, run server code via `exec_start` that POSTs to
  `${RAGTIME_BRIDGE_URL}/execute-component` with the token read from
  `RAGTIME_BRIDGE_TOKEN_FILE` per request.
- Do not add a separate ACL to "block BYOH Odoo": the bridge grants the same
  callers the same access. The real controls are workspace tool selection,
  `workspace`-surface tool ACLs, workspace roles, and credential scopes.
- SSH uses the dedicated `ssh_execute` / `ssh_transfer` operations, not
  `execute_component`.
- Each resource contract's `execute_component.supported` and
  `execution_lanes.*.supported` fields state which lane applies. Read them
  before calling.
