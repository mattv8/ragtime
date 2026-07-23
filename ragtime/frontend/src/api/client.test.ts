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
});
