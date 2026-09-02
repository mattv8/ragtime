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

describe('workspace sqlite inspector owner routing', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each([
    {
      name: 'initializes a database with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database: {
              name: 'app.sqlite3',
              relative_path: '.ragtime/db/app.sqlite3',
              size_bytes: 0,
              table_count: 0,
              last_modified_ms: null,
              owner_workspace_id: 'target/ws',
              owner_workspace_name: 'Target',
              ownership: 'linked',
              access_mode: 'read_write',
              persistence_mode: 'exclude',
              initialized: false,
            },
            mode_promoted: false,
            persistence_mode: 'exclude',
          }),
        );

        await api.initializeUserSpaceSqliteDatabase(
          'source-ws',
          { database_name: 'app.sqlite3' },
          'target/ws',
        );
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ database_name: 'app.sqlite3' }),
      }),
    },
    {
      name: 'imports a database with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database: {
              name: 'app.sqlite3',
              relative_path: '.ragtime/db/app.sqlite3',
              size_bytes: 0,
              table_count: 0,
              last_modified_ms: null,
              owner_workspace_id: 'target/ws',
              owner_workspace_name: 'Target',
              ownership: 'linked',
              access_mode: 'read_write',
              persistence_mode: 'exclude',
              initialized: true,
            },
            mode_promoted: false,
          }),
        );

        await api.importUserSpaceSqliteDatabase(
          'source-ws',
          'app.sqlite3',
          new FormData(),
          'target/ws',
        );
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/import?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({ method: 'POST', body: expect.any(FormData) }),
    },
    {
      name: 'lists tables with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database: {
              name: 'app.sqlite3',
              relative_path: '.ragtime/db/app.sqlite3',
              size_bytes: 0,
              table_count: 0,
              last_modified_ms: null,
              owner_workspace_id: 'target/ws',
              owner_workspace_name: 'Target',
              ownership: 'linked',
              access_mode: 'read',
              persistence_mode: 'exclude',
              initialized: true,
            },
            tables: [],
            persistence_mode: 'exclude',
            mode_promoted: false,
          }),
        );

        await api.listUserSpaceSqliteTables('source-ws', 'app.sqlite3', 'target/ws');
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/tables?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({ credentials: 'include' }),
    },
    {
      name: 'imports a table with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database_name: 'app.sqlite3',
            table: { name: 'items', type: 'table', row_count: 1 },
            mode_promoted: false,
          }),
        );

        await api.importUserSpaceSqliteTable(
          'source-ws',
          'app.sqlite3',
          'items',
          new FormData(),
          'target/ws',
        );
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/tables/items/import?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({ method: 'POST', body: expect.any(FormData) }),
    },
    {
      name: 'patches a row with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database_name: 'app.sqlite3',
            table_name: 'items',
            row: { id: 1 },
            mode_promoted: false,
          }),
        );

        await api.updateUserSpaceSqliteRow(
          'source-ws',
          'app.sqlite3',
          'items',
          { row_key: { id: 1 }, values: { name: 'updated' } },
          'target/ws',
        );
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/tables/items/rows?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify({ row_key: { id: 1 }, values: { name: 'updated' } }),
      }),
    },
    {
      name: 'deletes a row with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database_name: 'app.sqlite3',
            table_name: 'items',
            deleted: true,
            mode_promoted: false,
          }),
        );

        await api.deleteUserSpaceSqliteRow(
          'source-ws',
          'app.sqlite3',
          'items',
          { row_key: { id: 1 } },
          'target/ws',
        );
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/tables/items/rows?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({
        method: 'DELETE',
        body: JSON.stringify({ row_key: { id: 1 } }),
      }),
    },
    {
      name: 'queries a database with owner selection',
      run: async () => {
        fetchMock.mockResolvedValueOnce(
          jsonResponse({
            workspace_id: 'source-ws',
            database_name: 'app.sqlite3',
            columns: ['id'],
            rows: [{ id: 1 }],
            row_count: 1,
            truncated: false,
          }),
        );

        await api.queryUserSpaceSqliteDatabase(
          'source-ws',
          'app.sqlite3',
          { sql: 'select * from items' },
          'target/ws',
        );
      },
      expectedUrl:
        '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/query?owner_workspace_id=target%2Fws',
      expectedOptions: expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ sql: 'select * from items' }),
      }),
    },
  ])('$name', async ({ run, expectedUrl, expectedOptions }) => {
    await run();

    expect(fetchMock).toHaveBeenLastCalledWith(expectedUrl, expectedOptions);
  });

  it('appends owner selection to existing row pagination query parameters', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        workspace_id: 'source-ws',
        database_name: 'app.sqlite3',
        table_name: 'items',
        columns: [],
        rows: [],
        total: 0,
        limit: 25,
        offset: 10,
      }),
    );

    await api.listUserSpaceSqliteRows(
      'source-ws',
      'app.sqlite3',
      'items',
      {
        limit: 25,
        offset: 10,
        order_by: 'id',
        order_direction: 'desc',
      },
      'target/ws',
    );

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/tables/items/rows?limit=25&offset=10&order_by=id&order_direction=desc&owner_workspace_id=target%2Fws',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('preserves owned URLs when owner selection is omitted', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        workspace_id: 'source-ws',
        database: {
          name: 'app.sqlite3',
          relative_path: '.ragtime/db/app.sqlite3',
          size_bytes: 12,
          table_count: 1,
          last_modified_ms: 123,
          owner_workspace_id: 'source-ws',
          owner_workspace_name: 'Source',
          ownership: 'owned',
          access_mode: 'read_write',
          persistence_mode: 'include',
          initialized: true,
        },
        tables: [],
        persistence_mode: 'include',
        mode_promoted: false,
      }),
    );

    await api.listUserSpaceSqliteTables('source-ws', 'app.sqlite3');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/source-ws/sqlite/databases/app.sqlite3/tables',
      expect.objectContaining({ credentials: 'include' }),
    );
  });
});

describe('workspace bridge credential client requests', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('gets bridge credential status for a workspace runtime session', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        state: 'expired',
        bridge_url: 'https://bridge.example',
        token_session_id: 'session-old',
        current_session_id: 'session-new',
        issued_at: '2026-08-05T18:00:00Z',
        expires_at: '2026-08-05T19:00:00Z',
        last_success_at: '2026-08-05T18:30:00Z',
        detail: 'Bridge credentials expired.',
      }),
    );

    const result = await api.getUserSpaceBridgeCredentialStatus('workspace/123');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/runtime/workspaces/workspace%2F123/bridge-credentials/status',
      expect.objectContaining({ credentials: 'include' }),
    );
    expect(result.state).toBe('expired');
    expect(result.detail).toBe('Bridge credentials expired.');
  });

  it('posts to refresh bridge credentials for a workspace runtime session', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        state: 'healthy',
        bridge_url: 'https://bridge.example',
        token_session_id: 'session-new',
        current_session_id: 'session-new',
        issued_at: '2026-08-05T19:00:00Z',
        expires_at: '2026-08-05T20:00:00Z',
        last_success_at: '2026-08-05T19:01:00Z',
        detail: null,
      }),
    );

    const result = await api.refreshUserSpaceBridgeCredentials('workspace/123');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/runtime/workspaces/workspace%2F123/bridge-credentials/refresh',
      expect.objectContaining({ method: 'POST', credentials: 'include' }),
    );
    expect(result.state).toBe('healthy');
    expect(result.token_session_id).toBe('session-new');
  });
});

describe('workspace external API credential client requests', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('deletes a revoked credential record with encoded ids and accepts 204', async () => {
    fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }));

    await expect(
      api.deleteWorkspaceExternalApiCredential('workspace/123', 'cred/abc def'),
    ).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/workspace%2F123/external-api/credentials/cred%2Fabc%20def/record',
      expect.objectContaining({ method: 'DELETE', credentials: 'include' }),
    );
  });

  it('surfaces delete-record JSON errors through handleResponse when the server rejects deletion', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({ detail: 'Only revoked credentials can be deleted' }, 400),
    );

    await expect(
      api.deleteWorkspaceExternalApiCredential('workspace/123', 'cred/abc def'),
    ).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      detail: 'Only revoked credentials can be deleted',
      message: 'Only revoked credentials can be deleted',
    });
  });
});

describe('workspace archive export downloads', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(),
      revokeObjectURL: vi.fn(),
    } as Partial<typeof URL>);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('sends HEAD request and then calls startNativeDownload on success', async () => {
    const taskId = 'export-task-123';
    const expectedUrl = `/indexes/userspace/workspace-archive-export-tasks/${encodeURIComponent(taskId)}/download`;

    // Mock HEAD response for readiness check
    const headResponse = new Response(null, {
      status: 200,
      headers: {
        'content-disposition': 'attachment; filename="workspace-export.zip"',
      },
    });

    // Simulate startNativeDownload spy
    const iframeAppendSpy = vi.spyOn(document.body, 'appendChild');

    // Mock fetch to return HEAD response
    fetchMock.mockResolvedValueOnce(headResponse);

    await api.downloadUserSpaceWorkspaceArchiveExportTask(taskId);

    // Verify HEAD request was made with credentials
    expect(fetchMock).toHaveBeenCalledWith(
      expectedUrl,
      expect.objectContaining({
        method: 'HEAD',
        credentials: 'include',
      }),
    );

    // Verify iframe was created (indicating startNativeDownload was called)
    const iframeCreated = iframeAppendSpy.mock.calls.some((call) => {
      const node = call[0];
      return node instanceof HTMLElement && node.tagName === 'IFRAME';
    });
    expect(iframeCreated).toBe(true);

    iframeAppendSpy.mockRestore();
  });

  it('throws ApiError with detail on non-OK HEAD response', async () => {
    const taskId = 'export-task-456';

    const errorResponse = new Response(JSON.stringify({ detail: 'Archive not ready' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });

    fetchMock.mockResolvedValueOnce(errorResponse);

    const iframeAppendSpy = vi.spyOn(document.body, 'appendChild');

    await expect(api.downloadUserSpaceWorkspaceArchiveExportTask(taskId)).rejects.toMatchObject({
      name: 'ApiError',
      status: 400,
      detail: 'Archive not ready',
    });

    // Verify iframe was never created on error
    const iframeCreated = iframeAppendSpy.mock.calls.some((call) => {
      const node = call[0];
      return node instanceof HTMLElement && node.tagName === 'IFRAME';
    });
    expect(iframeCreated).toBe(false);

    iframeAppendSpy.mockRestore();
  });

  it('throws ApiError with fallback message when error detail is missing', async () => {
    const taskId = 'export-task-789';

    const errorResponse = new Response(null, {
      status: 500,
    });

    fetchMock.mockResolvedValueOnce(errorResponse);

    await expect(api.downloadUserSpaceWorkspaceArchiveExportTask(taskId)).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      message: 'Archive download failed',
    });
  });

  it('does not call blob() or createObjectURL on success', async () => {
    const taskId = 'export-task-blob-test';

    const headResponse = new Response(null, {
      status: 200,
      headers: {
        'content-disposition': 'attachment; filename="test.zip"',
      },
    });

    const blobSpy = vi.spyOn(headResponse, 'blob');
    const createObjectURLSpy = vi.spyOn(window.URL, 'createObjectURL');

    fetchMock.mockResolvedValueOnce(headResponse);

    await api.downloadUserSpaceWorkspaceArchiveExportTask(taskId);

    // Verify blob() was not called
    expect(blobSpy).not.toHaveBeenCalled();

    // Verify createObjectURL was not called
    expect(createObjectURLSpy).not.toHaveBeenCalled();

    blobSpy.mockRestore();
  });
});
