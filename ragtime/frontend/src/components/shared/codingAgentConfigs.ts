import type { CodingAgentClientId } from './codingAgentClients';

export const WORKSPACE_CREDENTIAL_PLACEHOLDER = '<YOUR_WORKSPACE_CREDENTIAL>';

export function workspaceServerName(workspaceId: string): string {
  return `ragtime-workspace-${workspaceId.replace(/[^a-zA-Z0-9_-]/g, '-').toLowerCase()}`;
}

export function workspaceCredentialEnvName(workspaceId: string): string {
  const suffix = workspaceId.replace(/[^a-zA-Z0-9]/g, '_').toUpperCase() || 'WORKSPACE';
  return `RAGTIME_${suffix}_CREDENTIAL`;
}

export function buildMcpServersConfig(name: string, server: Record<string, unknown>): string {
  return JSON.stringify({ mcpServers: { [name]: server } }, null, 2);
}

export function buildCursorInstallLink(mcpUrl: string, workspaceId: string): string {
  const name = workspaceServerName(workspaceId);
  const config = JSON.stringify({
    url: mcpUrl,
    headers: { Authorization: `Bearer \${env:${workspaceCredentialEnvName(workspaceId)}}` },
  });
  const encoded = btoa(String.fromCharCode(...new TextEncoder().encode(config)));
  return `cursor://anysphere.cursor-deeplink/mcp/install?${new URLSearchParams({ name, config: encoded })}`;
}

export function buildClientConfig(
  clientId: CodingAgentClientId,
  mcpUrl: string,
  workspaceId: string,
): string | null {
  const name = workspaceServerName(workspaceId);
  const envName = workspaceCredentialEnvName(workspaceId);
  const header = `Bearer ${WORKSPACE_CREDENTIAL_PLACEHOLDER}`;
  switch (clientId) {
    case 'claude-desktop':
      return buildMcpServersConfig(name, {
        command: 'npx',
        args: ['-y', 'mcp-remote', mcpUrl, '--header-file', '/ABSOLUTE/PRIVATE/headers.txt'],
      });
    case 'claude-code':
      return buildMcpServersConfig(name, {
        type: 'http',
        url: mcpUrl,
        headers: { Authorization: `Bearer \${${envName}}` },
      });
    case 'cursor':
      return buildMcpServersConfig(name, {
        url: mcpUrl,
        headers: { Authorization: `Bearer \${env:${envName}}` },
      });
    case 'cline':
      return buildMcpServersConfig(name, {
        type: 'streamableHttp',
        url: mcpUrl,
        headers: { Authorization: header },
        disabled: false,
        autoApprove: [],
      });
    case 'continue':
      return `name: ${name}
version: 0.0.1
schema: v1
mcpServers:
  - name: ${name}
    type: streamable-http
    url: ${JSON.stringify(mcpUrl)}
    requestOptions:
      headers:
        Authorization: "Bearer \${{ secrets.${envName} }}"`;
    case 'opencode':
      return JSON.stringify(
        {
          mcp: {
            [name]: {
              type: 'remote',
              url: mcpUrl,
              oauth: false,
              headers: { Authorization: `Bearer {env:${envName}}` },
            },
          },
        },
        null,
        2,
      );
    case 'hermes':
      return `mcp_servers:
  ${name}:
    url: ${JSON.stringify(mcpUrl)}
    headers:
      Authorization: "Bearer \${${envName}}"`;
    case 'codex':
      return `[mcp_servers.${name}]
url = ${JSON.stringify(mcpUrl)}
bearer_token_env_var = "${envName}"`;
    default:
      return null;
  }
}
