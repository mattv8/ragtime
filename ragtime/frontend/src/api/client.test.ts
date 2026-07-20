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
