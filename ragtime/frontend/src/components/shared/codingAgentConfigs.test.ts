import { describe, expect, it } from 'vitest';
import {
  buildClientConfig,
  buildCursorInstallLink,
  workspaceCredentialEnvName,
  workspaceServerName,
} from './codingAgentConfigs';

const url = 'https://ragtime.example.test/mcp?workspace=one';
const workspaceId = 'Workspace one/2';

describe('coding agent configuration templates', () => {
  it('uses stable workspace-specific names without secrets', () => {
    expect(workspaceServerName(workspaceId)).toBe('ragtime-workspace-workspace-one-2');
    expect(workspaceCredentialEnvName(workspaceId)).toBe('RAGTIME_WORKSPACE_ONE_2_CREDENTIAL');
    expect(buildClientConfig('cline', url, workspaceId)).toContain('<YOUR_WORKSPACE_CREDENTIAL>');
    expect(buildClientConfig('cline', url, workspaceId)).not.toContain('fresh-token');
  });

  it('builds client-specific syntax', () => {
    const desktop = JSON.parse(buildClientConfig('claude-desktop', url, workspaceId)!);
    expect(desktop.mcpServers[workspaceServerName(workspaceId)]).toEqual({
      command: 'npx',
      args: ['-y', 'mcp-remote', url, '--header-file', '/ABSOLUTE/PRIVATE/headers.txt'],
    });
    expect(buildClientConfig('cline', url, workspaceId)).toContain('"streamableHttp"');
    expect(buildClientConfig('continue', url, workspaceId)).toContain('requestOptions:');
    expect(buildClientConfig('opencode', url, workspaceId)).toContain('"servers"');
    expect(buildClientConfig('opencode', url, workspaceId)).toContain('"oauth": false');
    expect(buildClientConfig('hermes', url, workspaceId)).toContain('mcp_servers:');
    expect(buildClientConfig('codex', url, workspaceId)).toContain('bearer_token_env_var');
    expect(buildClientConfig('chatgpt', url, workspaceId)).toBeNull();
  });

  it('encodes a single secret-free Cursor server entry', () => {
    const link = new URL(buildCursorInstallLink(url, workspaceId));
    expect(link.protocol).toBe('cursor:');
    expect(link.searchParams.get('name')).toBe(workspaceServerName(workspaceId));
    const config = JSON.parse(atob(link.searchParams.get('config')!));
    expect(config).toEqual({
      url,
      headers: { Authorization: 'Bearer ${env:RAGTIME_WORKSPACE_ONE_2_CREDENTIAL}' },
    });
    expect(config).not.toHaveProperty('mcpServers');
  });
});
