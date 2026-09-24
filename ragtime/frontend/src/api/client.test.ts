import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { api, apiFetch } from './client';
import { sessionLifecycle } from '@/auth/sessionLifecycle';
import { formatPublicErrorDetail } from './publicErrorDetail';

type TransportApi = (typeof import('./client'))['api'];

function jsonResponse(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

async function loadIsolatedTransport() {
  vi.resetModules();
  const [client, lifecycle] = await Promise.all([
    import('./client'),
    import('@/auth/sessionLifecycle'),
  ]);
  return { ...client, sessionLifecycle: lifecycle.sessionLifecycle };
}

describe('auth-aware transport', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => vi.stubGlobal('fetch', fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('keeps anonymous public and challenge 401s local, but expires a current session once', async () => {
    const events: string[] = [];
    const unsubscribe = sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ detail: 'no session' }, 401))
      .mockResolvedValueOnce(jsonResponse({ detail: 'bad password' }, 401))
      .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }))
      .mockResolvedValueOnce(jsonResponse({ detail: 'expired' }, 401));

    await expect(api.getAuthStatus()).rejects.toMatchObject({ status: 401 });
    await expect(api.login({ username: 'bad', password: 'bad' })).rejects.toMatchObject({
      status: 401,
    });
    await api.login({ username: 'ok', password: 'ok' });
    await expect(api.getCurrentUser()).rejects.toMatchObject({ status: 401 });

    expect(events.filter((type) => type === 'expired')).toHaveLength(1);
    unsubscribe();
  });

  it('does not let an old private 401 expire a newer terminal exchange', async () => {
    let resolveOld!: (response: Response) => void;
    const oldResponse = new Promise<Response>((resolve) => {
      resolveOld = resolve;
    });
    const events: string[] = [];
    const unsubscribe = sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock
      .mockReturnValueOnce(oldResponse)
      .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }));

    const oldRequest = apiFetch('/private');
    await api.login({ username: 'new', password: 'new' });
    resolveOld(jsonResponse({ detail: 'old session' }, 401));
    await oldRequest;

    expect(events).not.toContain('expired');
    unsubscribe();
  });

  it('uses browser credentials for logout and treats its 401 as signed-out success', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({}, 401));
    await expect(api.logout()).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith(
      '/auth/logout',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('keeps public share access and shared previews local while private share management remains session-scoped', async () => {
    const events: string[] = [];
    const unsubscribe = sessionLifecycle.subscribe((event) => events.push(event.type));
    const publicCalls = [
      () => api.resolvePublicShareTarget('share'),
      () => api.resolvePublicShareTargetBySlug('owner', 'share'),
      () => api.getSharedConversation('share', 'password'),
      () => api.getSharedConversationBySlug('owner', 'share', 'password'),
      () => api.launchUserSpaceSharedPreview('share', { path: '/' }, 'password'),
      () => api.launchUserSpaceSharedPreviewBySlug('owner', 'share', { path: '/' }, 'password'),
    ];
    for (const call of publicCalls) {
      events.length = 0;
      fetchMock
        .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }))
        .mockResolvedValueOnce(jsonResponse({ detail: 'password required' }, 401));
      await api.login({ username: 'ok', password: 'ok' });
      await expect(call()).rejects.toMatchObject({ status: 401 });
      expect(events).not.toContain('expired');
    }

    events.length = 0;
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'forbidden' }, 401));
    await expect(api.listUserSpaceWorkspaceShareLinks('workspace')).rejects.toMatchObject({
      status: 401,
    });

    expect(events.filter((type) => type === 'expired')).toHaveLength(1);
    unsubscribe();
  });

  it('revalidates an account enrollment code failure without expiring a healthy session', async () => {
    const events: string[] = [];
    const unsubscribe = sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }))
      .mockResolvedValueOnce(jsonResponse({ detail: 'invalid code' }, 401));

    await api.login({ username: 'ok', password: 'ok' });
    await expect(
      api.completeMfaEnrollment({ code: '000000', enrollment_token: 'token' }),
    ).rejects.toMatchObject({ status: 401 });

    expect(events).toContain('revalidate');
    expect(events).not.toContain('expired');
    unsubscribe();
  });

  it('rejects stale successful current-user and pending credential bodies', async () => {
    let resolveBody!: (value: unknown) => void;
    const delayedBody = new Promise<unknown>((resolve) => {
      resolveBody = resolve;
    });
    const delayedResponse = {
      ok: true,
      status: 200,
      json: () => delayedBody,
    } as unknown as Response;
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }))
      .mockResolvedValueOnce(delayedResponse)
      .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }));

    await api.login({ username: 'first', password: 'first' });
    const currentUser = api.getCurrentUser();
    await api.login({ username: 'second', password: 'second' });
    resolveBody({ id: 'u1' });
    await expect(currentUser).rejects.toMatchObject({ name: 'AbortError' });

    let resolvePendingBody!: (value: unknown) => void;
    const delayedPending = new Promise<unknown>((resolve) => {
      resolvePendingBody = resolve;
    });
    const pendingResponse = {
      ok: true,
      status: 200,
      json: () => delayedPending,
    } as unknown as Response;
    fetchMock
      .mockResolvedValueOnce(pendingResponse)
      .mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }));
    const pendingLogin = api.login({ username: 'pending', password: 'pending' });
    await api.login({ username: 'newer', password: 'newer' });
    resolvePendingBody({ success: true, role: 'user', mfa_required: true });
    await expect(pendingLogin).rejects.toMatchObject({ name: 'AbortError' });
  });
});

describe('auth-aware transport isolated authenticated sessions', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => vi.stubGlobal('fetch', fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  async function authenticatedTransport() {
    const transport = await loadIsolatedTransport();
    fetchMock.mockResolvedValueOnce(jsonResponse({ success: true, role: 'user' }));
    await transport.api.login({ username: 'ok', password: 'ok' });
    expect(transport.sessionLifecycle.phase).toBe('establishing');
    const established = transport.sessionLifecycle.adoptSession(
      transport.sessionLifecycle.capture('session'),
      'u1',
    );
    expect(established).not.toBeNull();
    return transport;
  }

  it('captures the old private request as authenticated before a newer terminal exchange', async () => {
    const transport = await authenticatedTransport();
    let resolveOld!: (response: Response) => void;
    const oldResponse = new Promise<Response>((resolve) => {
      resolveOld = resolve;
    });
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock
      .mockReturnValueOnce(oldResponse)
      .mockResolvedValueOnce(jsonResponse({ success: true }));

    const oldRequest = transport.apiFetch('/private');
    await transport.api.login({ username: 'new', password: 'new' });
    resolveOld(jsonResponse({ detail: 'old cookie' }, 401));
    const response = await oldRequest;

    expect(transport.getResponseAuthContext(response)).toMatchObject({
      purpose: 'session',
      hadSession: true,
    });
    expect(events).not.toContain('expired');
    expect(transport.sessionLifecycle.phase).toBe('establishing');
  });

  it.each([
    ['public status', (api: TransportApi) => api.getAuthStatus()],
    ['bad credentials', (api: TransportApi) => api.login({ username: 'bad', password: 'bad' })],
    [
      'MFA verification',
      (api: TransportApi) =>
        api.verifyMfaChallenge({ mfa_challenge_token: 'token', code: '000000' }),
    ],
    [
      'WebAuthn authentication',
      (api: TransportApi) => api.startWebauthnAuthentication('challenge'),
    ],
  ])('keeps authenticated %s 401 local', async (_label, call) => {
    const transport = await authenticatedTransport();
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'rejected' }, 401));

    await expect(call(transport.api)).rejects.toMatchObject({ status: 401 });

    expect(transport.sessionLifecycle.phase).toBe('authenticated');
    expect(events).not.toContain('expired');
  });

  it.each([
    ['JSON', (api: TransportApi) => api.getCurrentUser()],
    ['blob download', (api: TransportApi) => api.downloadIndex('private-index')],
    ['delete', (api: TransportApi) => api.deleteIndex('private-index')],
    ['cancel', (api: TransportApi) => api.cancelJob('job-1')],
  ])('expires an authenticated session exactly once for a %s 401', async (_label, call) => {
    const transport = await authenticatedTransport();
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'expired' }, 401));

    await expect(call(transport.api)).rejects.toMatchObject({ status: 401 });

    expect(events.filter((event) => event === 'expired')).toHaveLength(1);
    expect(transport.sessionLifecycle.phase).toBe('anonymous');
  });

  it.each([
    ['403', () => Promise.resolve(jsonResponse({ detail: 'forbidden' }, 403))],
    ['500', () => Promise.resolve(jsonResponse({ detail: 'unavailable' }, 500))],
    ['network failure', () => Promise.reject(new TypeError('network unavailable'))],
  ])('does not expire a healthy session for %s', async (_label, response) => {
    const transport = await authenticatedTransport();
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock.mockImplementationOnce(response);

    await expect(transport.api.getCurrentUser()).rejects.toBeDefined();

    expect(transport.sessionLifecycle.phase).toBe('authenticated');
    expect(events).not.toContain('expired');
  });

  it.each([
    ['TOTP rotation start', (api: TransportApi) => api.startTotpRotation('000000')],
    [
      'TOTP rotation completion',
      (api: TransportApi) =>
        api.completeTotpRotation({ enrollment_token: 'token', code: '000000' }),
    ],
    ['recovery regeneration', (api: TransportApi) => api.regenerateRecoveryCodes('000000')],
    [
      'account TOTP enrollment',
      (api: TransportApi) =>
        api.completeMfaEnrollment({ code: '000000', enrollment_token: 'token' }),
    ],
    [
      'mount source directory creation',
      (api: TransportApi) => api.createUserspaceMountSourceDirectory('source', { path: '/' }),
    ],
    [
      'cloud mount directory creation',
      (api: TransportApi) =>
        api.createCloudMountSourceDirectory({
          source_type: 'google_drive',
          oauth_account_id: 'account',
          path: '/',
        }),
    ],
    [
      'workspace mount creation',
      (api: TransportApi) =>
        api.createWorkspaceMount('workspace', {
          mount_source_id: 'source',
          source_path: '/',
          target_path: '/',
        }),
    ],
    [
      'workspace mount sync preview',
      (api: TransportApi) => api.previewWorkspaceMountSync('workspace', 'mount'),
    ],
  ])('revalidates but preserves a healthy session after %s 401', async (_label, call) => {
    const transport = await authenticatedTransport();
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'provider or code rejected' }, 401));

    await expect(call(transport.api)).rejects.toMatchObject({ status: 401 });

    expect(transport.sessionLifecycle.phase).toBe('authenticated');
    expect(events).toContain('revalidate');
    expect(events).not.toContain('expired');
  });

  it.each([
    ['token route', (api: TransportApi) => api.joinSharedConversation('share', 'bad-password')],
    [
      'owner/slug route',
      (api: TransportApi) => api.joinSharedConversationBySlug('owner', 'share', 'bad-password'),
    ],
  ])(
    'revalidates an authenticated shared-conversation join after a %s 401',
    async (_label, call) => {
      const transport = await authenticatedTransport();
      const events: string[] = [];
      transport.sessionLifecycle.subscribe((event) => events.push(event.type));
      fetchMock.mockResolvedValueOnce(jsonResponse({ detail: 'password rejected' }, 401));

      await expect(call(transport.api)).rejects.toMatchObject({ status: 401 });

      expect(transport.sessionLifecycle.phase).toBe('authenticated');
      expect(events).toContain('revalidate');
      expect(events).not.toContain('expired');
    },
  );

  it('abandons tokenless renewal when enrollment returns success=false', async () => {
    const transport = await authenticatedTransport();
    let resolveOld!: (response: Response) => void;
    let resolveEnrollmentBody!: (body: unknown) => void;
    const oldResponse = new Promise<Response>((resolve) => {
      resolveOld = resolve;
    });
    const enrollmentBody = new Promise<unknown>((resolve) => {
      resolveEnrollmentBody = resolve;
    });
    const enrollmentResponse = {
      ok: true,
      status: 200,
      json: () => enrollmentBody,
    } as Response;
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock.mockReturnValueOnce(oldResponse).mockResolvedValueOnce(enrollmentResponse);

    const oldRequest = transport.apiFetch('/private');
    const enrollment = transport.api.completeMfaEnrollment({
      code: '000000',
      enrollment_token: 'token',
    });
    resolveOld(jsonResponse({ detail: 'old cookie' }, 401));
    await oldRequest;
    expect(transport.sessionLifecycle.phase).toBe('authenticated');

    resolveEnrollmentBody({ success: false });
    await expect(enrollment).resolves.toEqual({ success: false });
    expect(events.filter((event) => event === 'expired')).toHaveLength(1);
    expect(transport.sessionLifecycle.phase).toBe('anonymous');
  });

  it('renews an in-session enrollment before a concurrent old-cookie 401 can expire it', async () => {
    const transport = await authenticatedTransport();
    let resolveOld!: (response: Response) => void;
    const oldResponse = new Promise<Response>((resolve) => {
      resolveOld = resolve;
    });
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock
      .mockReturnValueOnce(oldResponse)
      .mockResolvedValueOnce(jsonResponse({ success: true, user: { id: 'u1' } }));

    const oldRequest = transport.apiFetch('/private');
    await expect(
      transport.api.completeMfaEnrollment({ code: '000000', enrollment_token: 'token' }),
    ).resolves.toMatchObject({ success: true });
    resolveOld(jsonResponse({ detail: 'old cookie' }, 401));
    await oldRequest;

    expect(events).toContain('renewed');
    expect(events).not.toContain('expired');
    expect(transport.sessionLifecycle.phase).toBe('authenticated');
  });

  it('applies a deferred old-cookie expiry when enrollment fails over the network', async () => {
    const transport = await authenticatedTransport();
    let resolveOld!: (response: Response) => void;
    let rejectEnrollment!: (error: Error) => void;
    const oldResponse = new Promise<Response>((resolve) => {
      resolveOld = resolve;
    });
    const enrollmentResponse = new Promise<Response>((_resolve, reject) => {
      rejectEnrollment = reject;
    });
    const events: string[] = [];
    transport.sessionLifecycle.subscribe((event) => events.push(event.type));
    fetchMock.mockReturnValueOnce(oldResponse).mockReturnValueOnce(enrollmentResponse);

    const oldRequest = transport.apiFetch('/private');
    const enrollment = transport.api.completeMfaEnrollment({
      code: '000000',
      enrollment_token: 'token',
    });
    resolveOld(jsonResponse({ detail: 'old cookie' }, 401));
    await oldRequest;
    expect(transport.sessionLifecycle.phase).toBe('authenticated');
    rejectEnrollment(new TypeError('network unavailable'));
    await expect(enrollment).rejects.toThrow('network unavailable');

    expect(events.filter((event) => event === 'expired')).toHaveLength(1);
    expect(transport.sessionLifecycle.phase).toBe('anonymous');
  });

  it('abandons a timed-out enrollment fence and clears its timeout', async () => {
    vi.useFakeTimers();
    try {
      const transport = await authenticatedTransport();
      fetchMock.mockImplementationOnce(
        (_url, options) =>
          new Promise<Response>((_resolve, reject) => {
            (options?.signal as AbortSignal).addEventListener('abort', () => {
              reject(new DOMException('Timed out', 'AbortError'));
            });
          }),
      );

      const enrollment = transport.api.completeMfaEnrollment({
        code: '000000',
        enrollment_token: 'token',
      });
      const rejected = expect(enrollment).rejects.toMatchObject({ name: 'AbortError' });
      await vi.advanceTimersByTimeAsync(30_000);
      await rejected;

      expect(transport.sessionLifecycle.phase).toBe('authenticated');
      expect(vi.getTimerCount()).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it('distinguishes pending enrollment and registration establishment from in-session renewal', async () => {
    const transport = await loadIsolatedTransport();
    fetchMock.mockResolvedValueOnce(jsonResponse({ success: true, user: { id: 'u1' } }));

    await transport.api.completeMfaEnrollment({
      code: '000000',
      enrollment_token: 'token',
      mfa_challenge_token: 'challenge',
    });
    expect(transport.sessionLifecycle.phase).toBe('establishing');

    const pendingRegistration = await loadIsolatedTransport();
    fetchMock.mockResolvedValueOnce(jsonResponse({ success: true }));
    await pendingRegistration.api.completeWebauthnRegistration({
      registration_token: 'token',
      credential: {},
      mfa_challenge_token: 'challenge',
    });
    expect(pendingRegistration.sessionLifecycle.phase).toBe('establishing');

    const accountRegistration = await authenticatedTransport();
    const generation = accountRegistration.sessionLifecycle.generation;
    fetchMock.mockResolvedValueOnce(jsonResponse({ success: true }));
    await accountRegistration.api.completeWebauthnRegistration({
      registration_token: 'token',
      credential: {},
    });
    expect(accountRegistration.sessionLifecycle.generation).toBe(generation);
  });
});

describe('workspace development operation requests', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it.each([
    ['exec_start', { command: 'npm test' }],
    ['exec_get', { job_id: 'job-1', cursor: 128 }],
    ['exec_cancel', { job_id: 'job-1' }],
  ])(
    'wraps %s arguments in the development operation request envelope',
    async (operation, arguments_) => {
      fetchMock.mockResolvedValueOnce(jsonResponse({}));

      await api.executeWorkspaceDevelopmentOperation('workspace/1', operation, arguments_);

      expect(fetchMock).toHaveBeenCalledWith(
        `/indexes/userspace/development/workspaces/workspace%2F1/operations/${operation}`,
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ arguments: arguments_ }),
        }),
      );
    },
  );
});

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

describe('SQLite history API client', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => vi.stubGlobal('fetch', fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('uses the exact list and preview request contracts', async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ workspace_id: 'ws/1', backups: [], can_manage: true }))
      .mockResolvedValueOnce(
        jsonResponse({
          preview_id: null,
          backup_id: 'backup/1',
          database_name: 'app.sqlite3',
          mode: 'merge',
          conflict_policy: 'keep_current',
          tables: [],
          migrations_applied: [],
          warnings: [],
          blockers: ['blocked'],
          can_apply: false,
          expires_at: null,
        }),
      );

    await api.listUserSpaceSqliteHistory('ws/1', {
      databaseName: 'app.sqlite3',
      snapshotId: 'snap/1',
    });
    await api.previewUserSpaceSqliteHistory('ws/1', 'backup/1', {
      mode: 'merge',
      conflict_policy: 'keep_current',
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/indexes/userspace/workspaces/ws%2F1/sqlite-history?database_name=app.sqlite3&snapshot_id=snap%2F1',
      expect.any(Object),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/indexes/userspace/workspaces/ws%2F1/sqlite-history/backup%2F1/preview',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ mode: 'merge', conflict_policy: 'keep_current' }),
      }),
    );
  });

  it('subscribes to encoded scoped history events with credentials', () => {
    const EventSourceMock = vi.fn();
    vi.stubGlobal('EventSource', EventSourceMock);

    api.subscribeUserSpaceSqliteHistoryEvents('ws/ 1', {
      databaseName: 'app data.sqlite3',
      snapshotId: 'snap/1',
    });

    expect(EventSourceMock).toHaveBeenCalledWith(
      '/indexes/userspace/workspaces/ws%2F%201/sqlite-history/events?database_name=app+data.sqlite3&snapshot_id=snap%2F1',
      { withCredentials: true },
    );
  });

  it('uses the capture, download, delete, restore, and recovery request contracts', async () => {
    const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});
    vi.stubGlobal('URL', {
      createObjectURL: vi.fn(() => 'blob:sqlite-history'),
      revokeObjectURL: vi.fn(),
    } as Partial<typeof URL>);
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ backup: { id: 'backup/1' } }))
      .mockResolvedValueOnce(
        new Response('sqlite bytes', {
          status: 200,
          headers: { 'content-disposition': 'attachment; filename="app.sqlite3"' },
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ success: true }))
      .mockResolvedValueOnce(
        jsonResponse({
          operation_id: 'operation/1',
          restored_backup_id: 'backup/1',
          safety_backup_id: 'safety/1',
          runtime_stopped: true,
          status: 'completed',
        }),
      )
      .mockResolvedValueOnce(jsonResponse({ operation_id: 'operation/1', status: 'aborted' }));

    await api.captureUserSpaceSqliteHistory('ws/1', 'app.sqlite3');
    await api.downloadUserSpaceSqliteHistory('ws/1', 'backup/1');
    await api.deleteUserSpaceSqliteHistory('ws/1', 'backup/1');
    await api.restoreUserSpaceSqliteHistory('ws/1', 'preview/1');
    await api.recoverUserSpaceSqliteHistoryMaintenance('ws/1', 'operation/1', 'abort');

    const base = '/indexes/userspace/workspaces/ws%2F1/sqlite-history';
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      base,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ database_name: 'app.sqlite3' }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(2, `${base}/backup%2F1/download`, expect.any(Object));
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      `${base}/backup%2F1`,
      expect.objectContaining({ method: 'DELETE' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      `${base}/restore`,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ preview_id: 'preview/1' }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      5,
      `${base}/maintenance/operation%2F1/recover`,
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ action: 'abort' }) }),
    );
    clickSpy.mockRestore();
  });

  it('uses queue capture endpoints with encoded paths and optional filters', async () => {
    const job = { id: 'job/1', status: 'pending' };
    fetchMock
      .mockResolvedValueOnce(jsonResponse({ job }))
      .mockResolvedValueOnce(jsonResponse({ jobs: [job] }))
      .mockResolvedValueOnce(jsonResponse({ job }))
      .mockResolvedValueOnce(jsonResponse({ job: { ...job, status: 'cancelled' } }));

    await api.enqueueUserSpaceSqliteBackup('ws/1', 'app.sqlite3', 'request-123');
    await api.listUserSpaceSqliteBackupJobs('ws/1', {
      databaseName: 'app.sqlite3',
      snapshotId: 'snapshot/1',
    });
    await api.getUserSpaceSqliteBackupJob('ws/1', 'job/1');
    await api.cancelUserSpaceSqliteBackupJob('ws/1', 'job/1');

    const base = '/indexes/userspace/workspaces/ws%2F1/sqlite-history/capture-jobs';
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      base,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ database_name: 'app.sqlite3', request_id: 'request-123' }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      `${base}?database_name=app.sqlite3&snapshot_id=snapshot%2F1`,
      expect.objectContaining({ cache: 'no-store' }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(3, `${base}/job%2F1`, expect.any(Object));
    expect(fetchMock).toHaveBeenNthCalledWith(
      4,
      `${base}/job%2F1/cancel`,
      expect.objectContaining({ method: 'POST' }),
    );
  });
});

describe('PDM webhook client', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('uses the saved-tool webhook URL and preserves nullable status fields', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: false,
        paused: false,
        webhook_id: null,
        webhook_url: null,
        created_at: null,
        last_received_at: null,
        pending: false,
        active_job_id: null,
        last_attempt_at: null,
        last_success_at: null,
        last_error: null,
      }),
    );

    const result = await api.getPdmWebhook('pdm/tool');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/pdm%2Ftool/pdm/webhook',
      expect.anything(),
    );
    expect(result.webhook_id).toBeNull();
    expect(result.last_error).toBeNull();
  });

  it('returns the one-time secret only from the enable endpoint', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        paused: false,
        webhook_id: 'pdm-webhook',
        webhook_url: 'https://ragtime.example/webhooks/pdm/pdm-webhook',
        created_at: '2026-09-16T12:00:00Z',
        last_received_at: null,
        pending: false,
        active_job_id: null,
        last_attempt_at: null,
        last_success_at: null,
        last_error: null,
        secret: 'one-time-secret',
      }),
    );

    const result = await api.enablePdmWebhook('pdm-tool');

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/tools/pdm-tool/pdm/webhook',
      expect.objectContaining({ method: 'POST' }),
    );
    expect(result.secret).toBe('one-time-secret');
  });
});

describe('conversation client request shapes', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('keeps summary timestamps verbatim, encodes all pagination options, and forwards its signal', async () => {
    const controller = new AbortController();
    const cursorUpdatedAt = '2026-09-15T12:34:56.123456+00:00';
    fetchMock.mockResolvedValueOnce(jsonResponse([]));

    await api.listConversationSummaries(
      'workspace/a b',
      {
        since: '2026-08-01T00:00:00+00:00',
        until: '2026-09-01T00:00:00+00:00',
        limit: 50,
        cursorUpdatedAt,
        cursorId: 'conversation/a b',
      },
      controller.signal,
    );

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/conversations/summaries?workspace_id=workspace%2Fa%20b&since=2026-08-01T00%3A00%3A00%2B00%3A00&until=2026-09-01T00%3A00%3A00%2B00%3A00&limit=50&cursor_updated_at=2026-09-15T12%3A34%3A56.123456%2B00%3A00&cursor_id=conversation%2Fa%20b',
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it('preserves bare summary responses and optional callers without query parameters', async () => {
    const summaries = [{ id: 'summary-1' }];
    fetchMock.mockResolvedValueOnce(jsonResponse(summaries));

    await expect(api.listConversationSummaries()).resolves.toEqual(summaries);
    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/conversations/summaries',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('forwards abort signals to existing conversation requests', async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValueOnce(jsonResponse({ id: 'conversation-1' }));
    fetchMock.mockResolvedValueOnce(jsonResponse([]));
    fetchMock.mockResolvedValueOnce(jsonResponse({ matched_conversation_ids: [] }));

    await api.getConversation('conversation-1', undefined, controller.signal);
    await api.listConversations(undefined, undefined, controller.signal);
    await api.searchConversationBranches(['conversation-1'], 'needle', controller.signal);

    for (const [, request] of fetchMock.mock.calls) {
      expect(request?.signal).toBe(controller.signal);
    }
  });

  it('uses the frozen window endpoints, workspace query, and abort options', async () => {
    const controller = new AbortController();
    fetchMock.mockResolvedValueOnce(jsonResponse({ entries: [] }));
    fetchMock.mockResolvedValueOnce(jsonResponse({ entries: [] }));
    fetchMock.mockResolvedValueOnce(jsonResponse({ entries: [] }));

    await api.getConversationLatestExchange('conversation/a b', 'workspace/a b', {
      signal: controller.signal,
    });
    await api.getConversationMessageWindow(
      'conversation/a b',
      { cursor: 'before/20', limit: 20 },
      'workspace/a b',
      { signal: controller.signal },
    );
    await api.getConversationWindowMessage(
      'conversation/a b',
      12,
      'revision/a b',
      'workspace/a b',
      { signal: controller.signal },
    );

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      '/indexes/conversations/conversation%2Fa%20b/latest-exchange?workspace_id=workspace%2Fa%20b',
      expect.objectContaining({ signal: controller.signal }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      '/indexes/conversations/conversation%2Fa%20b/message-window?cursor=before%2F20&limit=20&workspace_id=workspace%2Fa%20b',
      expect.objectContaining({ signal: controller.signal }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      '/indexes/conversations/conversation%2Fa%20b/messages/12?revision=revision%2Fa%20b&workspace_id=workspace%2Fa%20b',
      expect.objectContaining({ signal: controller.signal }),
    );
  });

  it('adds owner_scope only when requested without changing older summary calls', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse([]));
    await api.listConversationSummaries(undefined, { owner_scope: 'self' });
    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/conversations/summaries?owner_scope=self',
      expect.anything(),
    );
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

describe('OpenRouter credit monitor client requests', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => vi.stubGlobal('fetch', fetchMock));
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('gets admin credit status without caching', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse({
        enabled: true,
        state: 'low',
        key_remaining_usd: 2,
        wallet_remaining_usd: null,
        threshold_usd: 5,
        checked_at: null,
        stale: false,
        warning: 'OpenRouter key credits are low.',
      }),
    );

    await api.getOpenRouterCreditStatus();

    expect(fetchMock).toHaveBeenCalledWith(
      '/indexes/settings/openrouter-credits',
      expect.objectContaining({ cache: 'no-store', credentials: 'include' }),
    );
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

describe('public content-protection error details', () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.clearAllMocks();
  });

  it('uses the complete nested FastAPI detail message without stringifying the object', async () => {
    fetchMock.mockResolvedValueOnce(
      jsonResponse(
        {
          detail: {
            code: 'content_access_denied',
            message:
              'This request is outside your access profile. Try rephrasing your question within your access profile.',
            reason: 'This request is outside your access profile.',
            next_step: 'Try rephrasing your question within your access profile.',
            request_id: 'request-123',
            reason_code: 'profile_mismatch',
          },
        },
        403,
      ),
    );

    await expect(
      api.deleteWorkspaceExternalApiCredential('workspace-1', 'credential-1'),
    ).rejects.toMatchObject({
      message:
        'This request is outside your access profile. Try rephrasing your question within your access profile.',
      publicDetail: {
        code: 'content_access_denied',
        reason_code: 'profile_mismatch',
        request_id: 'request-123',
      },
    });
  });

  it('keeps generic failures on their existing fallback path when structured reasons are absent', async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse({ detail: { code: 'validation_error' } }, 422));

    await expect(
      api.deleteWorkspaceExternalApiCredential('workspace-1', 'credential-1'),
    ).rejects.toMatchObject({ message: 'Request failed', publicDetail: undefined });
  });

  it('formats SSE classifier fields for plain-text UI display without exposing unused markup', () => {
    expect(
      formatPublicErrorDetail(
        {
          code: 'content_access_denied',
          message: 'This request is outside your access profile. Try a narrower request.',
          reason: '<img src=x onerror=alert(1)>',
          next_step: 'Try a narrower request.',
          request_id: 'request-123',
        },
        'Generation failed',
      ),
    ).toBe('This request is outside your access profile. Try a narrower request.');
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
