import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { api } from './client';

function jsonResponse(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

describe('git webhook client normalization', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('normalizes nullable webhook config payloads at the client boundary', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: 1,
        webhook_url: 'https://ragtime.example/webhooks/git/webhook-123',
        provider: null,
        branch: null,
        created_at: '2026-07-16T12:00:00Z',
      }),
    );

    const result = await api.getIndexWebhook('docs/repo');

    expect(result).toEqual({
      enabled: true,
      paused: true,
      webhook_url: 'https://ragtime.example/webhooks/git/webhook-123',
      provider: 'generic',
      branch: '',
      created_at: '2026-07-16T12:00:00Z',
    });
  });

  it('normalizes omitted enable secrets to null', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: null,
        webhook_url: 'https://ragtime.example/webhooks/git/webhook-456',
        provider: 'github',
        branch: 'main',
        created_at: '2026-07-16T12:00:00Z',
      }),
    );

    const result = await api.enableUserSpaceWorkspaceScmWebhook('workspace/123');

    expect(result).toEqual({
      enabled: true,
      paused: false,
      webhook_url: 'https://ragtime.example/webhooks/git/webhook-456',
      provider: 'github',
      branch: 'main',
      created_at: '2026-07-16T12:00:00Z',
      secret: null,
    });
  });

  it('posts to pause the index webhook and normalizes the response', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: true,
        webhook_url: 'https://ragtime.example/webhooks/git/webhook-123',
        provider: 'github',
        branch: 'main',
        created_at: '2026-07-16T12:00:00Z',
      }),
    );

    const result = await api.pauseIndexWebhook('docs/repo');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/docs%2Frepo/webhook/pause',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.paused).toBe(true);
  });

  it('posts to resume the index webhook and normalizes the response', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: false,
        webhook_url: 'https://ragtime.example/webhooks/git/webhook-123',
        provider: 'github',
        branch: 'main',
        created_at: '2026-07-16T12:00:00Z',
      }),
    );

    const result = await api.resumeIndexWebhook('docs/repo');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/docs%2Frepo/webhook/resume',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.paused).toBe(false);
  });

  it('posts to pause the workspace scm webhook and normalizes the response', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: true,
        webhook_url: 'https://ragtime.example/webhooks/git/workspace-123',
        provider: 'github',
        branch: 'main',
        created_at: '2026-07-16T12:00:00Z',
      }),
    );

    const result = await api.pauseUserSpaceWorkspaceScmWebhook('workspace/123');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/workspace%2F123/scm/webhook/pause',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.paused).toBe(true);
  });

  it('posts to resume the workspace scm webhook and normalizes the response', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: false,
        webhook_url: 'https://ragtime.example/webhooks/git/workspace-123',
        provider: 'github',
        branch: 'main',
        created_at: '2026-07-16T12:00:00Z',
      }),
    );

    const result = await api.resumeUserSpaceWorkspaceScmWebhook('workspace/123');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/workspace%2F123/scm/webhook/resume',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.paused).toBe(false);
  });
});

describe('HTTP API OAuth client request shapes', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('posts the issuer to discovery', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ issuer: 'https://issuer.example' }));

    await api.discoverHttpApiOAuth({ issuer_url: 'https://issuer.example' });

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/http-api/oauth/discover',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ issuer_url: 'https://issuer.example' }),
      }),
    );
  });

  it('posts the unsaved connection and optional tool id to start', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: 'pending', session_id: 'session-1', interval: 7 }),
    );
    const connectionConfig = {
      auth_mode: 'oauth2' as const,
      oauth_flow: 'device_code' as const,
      oauth_client_id: 'client-id',
    };

    await api.startHttpApiOAuth({ connection_config: connectionConfig, tool_id: 'tool-1' });

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/http-api/oauth/start',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ connection_config: connectionConfig, tool_id: 'tool-1' }),
      }),
    );
  });

  it('posts only the temporary session id to poll', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ status: 'pending', session_id: 'session-1', retry_after_seconds: 11 }),
    );

    await api.pollHttpApiOAuth({ session_id: 'session-1' });

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/http-api/oauth/poll',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ session_id: 'session-1' }),
      }),
    );
  });

  it('gets the encoded HTTP API edit config endpoint', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        connection_config: {
          base_url: 'https://api.example.com',
          auth_mode: 'api_key',
          api_key: 'decrypted-api-key',
        },
      }),
    );

    await api.getHttpApiEditConfig('tool/http api');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/tool%2Fhttp%20api/http-api-edit-config',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('adds the requested userspace tool surface query when listing available tools', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await api.listUserSpaceAvailableTools('workspace');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/tools?surface=workspace',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('gets the encoded tool access policy endpoint', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        tool_id: 'tool/http api',
        default_chat_access: 'deny',
        default_workspace_access: 'deny',
        users: [],
        groups: [],
      }),
    );

    await api.getToolAccessPolicy('tool/http api');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/tool%2Fhttp%20api/access',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('puts only request-direction tool access fields when updating a policy', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        tool_id: 'tool-1',
        default_chat_access: 'read',
        default_workspace_access: 'deny',
        users: [],
        groups: [],
      }),
    );

    await api.updateToolAccessPolicy('tool-1', {
      tool_id: 'tool-1',
      default_chat_access: 'read',
      default_workspace_access: 'deny',
      users: [
        {
          principal_id: 'user-1',
          chat_access: 'read',
          workspace_access: null,
          display_name: 'Alice',
          principal_detail: '@alice',
          orphaned: false,
        },
      ],
      groups: [
        {
          principal_id: 'group-1',
          chat_access: null,
          workspace_access: 'read_write',
          display_name: 'Engineering',
          principal_detail: 'LDAP',
          orphaned: false,
        },
      ],
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/tool-1/access',
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({
          default_chat_access: 'read',
          default_workspace_access: 'deny',
          users: [
            {
              principal_id: 'user-1',
              chat_access: 'read',
              workspace_access: null,
            },
          ],
          groups: [
            {
              principal_id: 'group-1',
              chat_access: null,
              workspace_access: 'read_write',
            },
          ],
        }),
      }),
    );
  });
});

describe('workspace agent grant client request shapes', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('serializes an explicit sqlite_access_mode when present', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        id: 'grant-1',
        source_workspace_id: 'ws-source',
        target_workspace_id: 'ws-target',
        target_workspace_name: 'Target',
        access_mode: 'read',
        sqlite_access_mode: 'read_write',
        granted_by_user_id: 'user-1',
        created_at: '2026-08-05T12:00:00Z',
        updated_at: '2026-08-05T12:00:00Z',
      }),
    );

    await api.upsertUserSpaceWorkspaceAgentGrant('ws-source', {
      target_workspace_id: 'ws-target',
      access_mode: 'read',
      sqlite_access_mode: 'read_write',
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/ws-source/agent-grants',
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({
          target_workspace_id: 'ws-target',
          access_mode: 'read',
          sqlite_access_mode: 'read_write',
        }),
      }),
    );
  });

  it('preserves omission when sqlite_access_mode is absent', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        id: 'grant-1',
        source_workspace_id: 'ws-source',
        target_workspace_id: 'ws-target',
        target_workspace_name: 'Target',
        access_mode: 'read_write',
        sqlite_access_mode: 'read',
        granted_by_user_id: 'user-1',
        created_at: '2026-08-05T12:00:00Z',
        updated_at: '2026-08-05T12:00:00Z',
      }),
    );

    await api.upsertUserSpaceWorkspaceAgentGrant('ws-source', {
      target_workspace_id: 'ws-target',
      access_mode: 'read_write',
    });

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/ws-source/agent-grants',
      expect.objectContaining({
        method: 'PUT',
        body: JSON.stringify({
          target_workspace_id: 'ws-target',
          access_mode: 'read_write',
        }),
      }),
    );
  });
});
