import type { ComponentProps, ReactNode } from 'react';

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GitWebhookConfig, GitWebhookEnableResponse, UserSpaceWorkspace } from '@/types';

import { WorkspaceScmWizard } from './WorkspaceScmWizard';

const apiMock = vi.hoisted(() => ({
  getUserSpacePreviewSettings: vi.fn(),
  fetchUserSpaceWorkspaceScmBranches: vi.fn(),
  updateUserSpaceWorkspaceScmSettings: vi.fn(),
  getUserSpaceWorkspaceScmWebhook: vi.fn(),
  enableUserSpaceWorkspaceScmWebhook: vi.fn(),
  pauseUserSpaceWorkspaceScmWebhook: vi.fn(),
  resumeUserSpaceWorkspaceScmWebhook: vi.fn(),
  rotateUserSpaceWorkspaceScmWebhookSecret: vi.fn(),
  disableUserSpaceWorkspaceScmWebhook: vi.fn(),
  queueUserSpaceWorkspaceScmPreviewImport: vi.fn(),
}));

const latestWebhookSettingsProps = vi.hoisted(() => ({
  config: null as GitWebhookConfig | null,
  revealedSecret: null as string | null,
  disabled: false,
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('./DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({
    buttonText,
    onDelete,
    disabled,
  }: {
    buttonText: string;
    onDelete: () => void;
    disabled?: boolean;
  }) => (
    <button type="button" onClick={onDelete} disabled={disabled}>
      {buttonText}
    </button>
  ),
}));

vi.mock('./Popover', () => ({
  Popover: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}));

vi.mock('./ScheduleStartTimeInput', () => ({
  defaultScheduleStartMinute: () => 0,
  defaultScheduleTimezone: () => 'UTC',
  ScheduleStartTimeInput: () => <div>schedule input</div>,
}));

vi.mock('./GitWebhookSettings', () => ({
  GitWebhookSettings: ({
    config,
    revealedSecret,
    disabled,
    onRotate,
    onPause,
    onResume,
  }: {
    config: GitWebhookConfig;
    revealedSecret: string | null;
    disabled: boolean;
    onRotate: () => void;
    onPause: () => void;
    onResume: () => void;
  }) =>
    (() => {
      latestWebhookSettingsProps.config = config;
      latestWebhookSettingsProps.revealedSecret = revealedSecret;
      latestWebhookSettingsProps.disabled = Boolean(disabled);
      return (
        <div data-testid="git-webhook-settings-mock">
          <div>Push webhook</div>
          <div>{`webhook:${config.provider}:${config.branch}:${config.enabled ? 'enabled' : 'disabled'}:${config.paused ? 'paused' : 'active'}`}</div>
          {revealedSecret && <div>{revealedSecret}</div>}
          {config.enabled && (
            <>
              <button type="button" onClick={onRotate} disabled={disabled}>
                Rotate secret
              </button>
              {config.paused ? (
                <button type="button" onClick={onResume} disabled={disabled}>
                  Resume webhook
                </button>
              ) : (
                <button type="button" onClick={onPause} disabled={disabled}>
                  Pause webhook
                </button>
              )}
            </>
          )}
        </div>
      );
    })(),
}));

vi.mock('./shared/MiniLoadingSpinner', () => ({
  MiniLoadingSpinner: () => <span>loading</span>,
}));

const toastApiMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  dismiss: vi.fn(),
}));

vi.mock('./shared/Toast', () => ({
  ToastContainer: () => null,
  useToast: () => [[], toastApiMock],
}));

const baseWorkspace: UserSpaceWorkspace = {
  id: 'workspace-1',
  name: 'Workspace One',
  description: 'Test workspace',
  sqlite_persistence_mode: 'exclude',
  owner_user_id: 'user-1',
  owner_username: 'ada',
  owner_display_name: 'Ada',
  selected_tool_ids: [],
  selected_tool_group_ids: [],
  conversation_ids: [],
  members: [],
  scm: null,
  archive_export_task_id: null,
  archive_export_task_phase: null,
  archive_import_task_id: null,
  archive_import_task_phase: null,
  scm_import_task_id: null,
  scm_import_task_phase: null,
  created_at: '2026-07-16T00:00:00Z',
  updated_at: '2026-07-16T00:00:00Z',
};

const disabledWebhookConfig: GitWebhookConfig = {
  enabled: false,
  paused: false,
  webhook_url: null,
  provider: 'github',
  branch: 'main',
  created_at: null,
};

const enabledWebhookConfig: GitWebhookConfig = {
  enabled: true,
  paused: false,
  webhook_url: 'https://ragtime.example/webhooks/git/workspace-1',
  provider: 'github',
  branch: 'main',
  created_at: '2026-07-16T12:00:00Z',
};

const enabledWebhookWithSecret: GitWebhookEnableResponse = {
  ...enabledWebhookConfig,
  secret: 'secret-once',
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function upstreamWorkspace(
  overrides?: Partial<NonNullable<UserSpaceWorkspace['scm']>>,
): UserSpaceWorkspace {
  return {
    ...baseWorkspace,
    scm: {
      connected: true,
      git_url: 'https://github.com/example/repo.git',
      git_branch: 'main',
      provider: 'github',
      repo_visibility: 'private',
      has_stored_token: true,
      remote_role: 'upstream',
      auto_sync_policy: 'manual',
      auto_pull_enabled: true,
      auto_push_interval_seconds: 3600,
      auto_pull_interval_seconds: 3600,
      ...overrides,
    },
  };
}

function publishWorkspace(): UserSpaceWorkspace {
  return {
    ...upstreamWorkspace({ remote_role: 'publish' }),
  };
}

function scmSettingsResponse(
  scmOverrides?: Partial<NonNullable<UserSpaceWorkspace['scm']>>,
  workspaceId = 'workspace-1',
) {
  return {
    workspace_id: workspaceId,
    scm: upstreamWorkspace(scmOverrides).scm!,
  };
}

function disconnectedWorkspace(): UserSpaceWorkspace {
  return {
    ...baseWorkspace,
    scm: {
      connected: false,
      git_url: null,
      git_branch: null,
      provider: null,
      repo_visibility: null,
      has_stored_token: false,
      remote_role: null,
    },
  };
}

async function openGitSourceTab() {
  fireEvent.click(screen.getAllByRole('button', { name: 'Git Source' })[0]);
  await act(async () => {
    await Promise.resolve();
  });
}

function renderWizard(
  workspace: UserSpaceWorkspace,
  overrides: Partial<ComponentProps<typeof WorkspaceScmWizard>> = {},
) {
  return render(
    <WorkspaceScmWizard
      workspace={workspace}
      onClose={vi.fn()}
      onSyncComplete={vi.fn()}
      onWorkspaceChanged={vi.fn()}
      {...overrides}
    />,
  );
}

beforeEach(() => {
  apiMock.getUserSpacePreviewSettings.mockResolvedValue({
    userspace_sqlite_import_max_bytes: 1000,
  });
  apiMock.fetchUserSpaceWorkspaceScmBranches.mockResolvedValue({ branches: ['main'], error: null });
  apiMock.updateUserSpaceWorkspaceScmSettings.mockResolvedValue(scmSettingsResponse());
  apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(disabledWebhookConfig);
  apiMock.enableUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookWithSecret);
  apiMock.pauseUserSpaceWorkspaceScmWebhook.mockResolvedValue({
    ...enabledWebhookConfig,
    paused: true,
  });
  apiMock.resumeUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookConfig);
  apiMock.rotateUserSpaceWorkspaceScmWebhookSecret.mockResolvedValue({
    ...enabledWebhookConfig,
    secret: 'rotated-secret',
  });
  apiMock.disableUserSpaceWorkspaceScmWebhook.mockResolvedValue(undefined);
  apiMock.queueUserSpaceWorkspaceScmPreviewImport.mockResolvedValue({
    task_id: 'preview-task-1',
    workspace_id: 'workspace-1',
    git_url: 'https://github.com/example/repo.git',
    git_branch: 'main',
    phase: 'queued',
    progress: 0,
    message: null,
    preview: null,
    created_at: '2026-07-16T12:00:00Z',
    started_at: null,
    completed_at: null,
    error: null,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  toastApiMock.success.mockReset();
  toastApiMock.error.mockReset();
  toastApiMock.info.mockReset();
  toastApiMock.dismiss.mockReset();
  latestWebhookSettingsProps.config = null;
  latestWebhookSettingsProps.revealedSecret = null;
  latestWebhookSettingsProps.disabled = false;
});

describe('WorkspaceScmWizard webhook integration', () => {
  it('shows manual pull cadence by default and hides webhook setup until webhook delivery is selected', async () => {
    renderWizard(upstreamWorkspace({ auto_pull_enabled: false }));

    await openGitSourceTab();

    expect(
      (await screen.findByLabelText('Auto-pull interval')) as HTMLSelectElement,
    ).toHaveProperty('value', '0');
    expect(screen.queryByText('Push webhook')).toBeNull();
  });

  it('renders webhook loading text as a muted status message before config resolves', async () => {
    const webhookConfigRequest = deferred<GitWebhookConfig>();
    apiMock.getUserSpaceWorkspaceScmWebhook.mockImplementationOnce(
      () => webhookConfigRequest.promise,
    );

    renderWizard(upstreamWorkspace());

    await openGitSourceTab();

    const loadingMessage = await screen.findByRole('status');
    expect(loadingMessage.textContent).toBe('Loading webhook settings…');
    expect(loadingMessage.className).toContain('userspace-muted');

    webhookConfigRequest.resolve(disabledWebhookConfig);
    await waitFor(() => {
      expect(screen.queryByRole('status')).toBeNull();
    });
  });

  it('does not show webhook controls for publish or disconnected remotes', async () => {
    const { rerender } = renderWizard(publishWorkspace());

    await openGitSourceTab();
    expect(screen.queryByText('Push webhook')).toBeNull();

    rerender(
      <WorkspaceScmWizard
        workspace={disconnectedWorkspace()}
        onClose={vi.fn()}
        onSyncComplete={vi.fn()}
        onWorkspaceChanged={vi.fn()}
      />,
    );

    await openGitSourceTab();
    expect(screen.queryByText('Push webhook')).toBeNull();
  });

  it('enables webhook delivery from the pull cadence selector', async () => {
    renderWizard(upstreamWorkspace({ auto_pull_enabled: true }));

    await openGitSourceTab();

    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: 'webhook' } });

    await waitFor(() => {
      expect(apiMock.enableUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect(latestWebhookSettingsProps.config?.enabled).toBe(true);
  });

  it('requires confirmation before leaving webhook delivery for a scheduled pull', async () => {
    apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookConfig);
    renderWizard(upstreamWorkspace({ auto_pull_enabled: false }));

    await openGitSourceTab();

    await screen.findByLabelText('Auto-pull interval');
    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: '3600' } });
    expect(screen.getByRole('dialog', { name: 'Disable webhook delivery?' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Disable webhook and continue' }));
    await waitFor(() => {
      expect(apiMock.disableUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect((screen.getByLabelText('Auto-pull interval') as HTMLSelectElement).value).toBe('3600');
  });

  it('keeps scheduled pull drafts unsaved until Save is clicked', async () => {
    renderWizard(upstreamWorkspace({ auto_pull_enabled: false }));

    await openGitSourceTab();

    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: '300' } });

    expect(apiMock.updateUserSpaceWorkspaceScmSettings).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(apiMock.updateUserSpaceWorkspaceScmSettings).toHaveBeenCalledWith(
        'workspace-1',
        expect.objectContaining({
          auto_pull_enabled: true,
          auto_pull_interval_seconds: 300,
        }),
      );
    });
  });

  it('restores the prior pull draft when enabling webhook delivery fails', async () => {
    apiMock.enableUserSpaceWorkspaceScmWebhook.mockRejectedValueOnce(new Error('enable failed'));
    renderWizard(upstreamWorkspace({ auto_pull_enabled: true, auto_pull_interval_seconds: 900 }));

    await openGitSourceTab();

    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: 'webhook' } });

    await waitFor(() => {
      expect(toastApiMock.error).toHaveBeenCalledWith('Webhook setup failed: enable failed');
    });
    expect((screen.getByLabelText('Auto-pull interval') as HTMLSelectElement).value).toBe('900');
  });

  it('refreshes deliveries and notifies parent refresh when webhook delivery is enabled', async () => {
    const onWorkspaceChanged = vi.fn();
    renderWizard(upstreamWorkspace(), { onWorkspaceChanged });

    await openGitSourceTab();
    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: 'webhook' } });

    await waitFor(() => {
      expect(apiMock.enableUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect(await screen.findByText('secret-once')).toBeTruthy();
    expect(onWorkspaceChanged).toHaveBeenCalledTimes(1);
  });

  it('pauses and resumes the webhook while keeping webhook cadence selected', async () => {
    apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookConfig);
    const onWorkspaceChanged = vi.fn();
    renderWizard(upstreamWorkspace(), { onWorkspaceChanged });

    await openGitSourceTab();

    fireEvent.click(await screen.findByRole('button', { name: 'Rotate secret' }));
    await waitFor(() => {
      expect(apiMock.rotateUserSpaceWorkspaceScmWebhookSecret).toHaveBeenCalledWith('workspace-1');
    });
    expect(await screen.findByText('rotated-secret')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Pause webhook' }));
    await waitFor(() => {
      expect(apiMock.pauseUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect((screen.getByLabelText('Auto-pull interval') as HTMLSelectElement).value).toBe(
      'webhook',
    );
    expect(screen.getByText('webhook:github:main:enabled:paused')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Resume webhook' }));
    await waitFor(() => {
      expect(apiMock.resumeUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect(screen.getByText('webhook:github:main:enabled:active')).toBeTruthy();
    expect(onWorkspaceChanged).toHaveBeenCalledTimes(3);
  });

  it('shows pull now beside cadence for manual, scheduled, active, and paused webhook states', async () => {
    renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    expect(screen.getByRole('button', { name: 'Pull now' })).toBeTruthy();

    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: '300' } });
    expect(screen.getByRole('button', { name: 'Pull now' })).toBeTruthy();

    cleanup();
    apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookConfig);
    renderWizard(upstreamWorkspace());
    await openGitSourceTab();
    expect(await screen.findByRole('button', { name: 'Pull now' })).toBeTruthy();

    cleanup();
    apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue({
      ...enabledWebhookConfig,
      paused: true,
    });
    renderWizard(upstreamWorkspace());
    await openGitSourceTab();
    expect(await screen.findByRole('button', { name: 'Pull now' })).toBeTruthy();
  });

  it('pulls now by reusing the workspace import preview flow', async () => {
    renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    fireEvent.click(screen.getByRole('button', { name: 'Pull now' }));

    await waitFor(() => {
      expect(apiMock.queueUserSpaceWorkspaceScmPreviewImport).toHaveBeenCalledWith(
        'workspace-1',
        expect.objectContaining({ git_branch: 'main' }),
      );
    });
  });

  it('rejects stale webhook mutation responses after switching workspaces', async () => {
    const enableWebhook = deferred<GitWebhookEnableResponse>();
    apiMock.getUserSpaceWorkspaceScmWebhook
      .mockResolvedValueOnce(disabledWebhookConfig)
      .mockResolvedValueOnce({ ...disabledWebhookConfig, branch: 'develop' });
    apiMock.enableUserSpaceWorkspaceScmWebhook.mockImplementationOnce(() => enableWebhook.promise);

    const { rerender } = renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    fireEvent.change(screen.getByLabelText('Auto-pull interval'), { target: { value: 'webhook' } });

    rerender(
      <WorkspaceScmWizard
        workspace={{ ...upstreamWorkspace({ git_branch: 'develop' }), id: 'workspace-2' }}
        onClose={vi.fn()}
        onSyncComplete={vi.fn()}
        onWorkspaceChanged={vi.fn()}
      />,
    );
    await waitFor(() => {
      expect(apiMock.getUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-2');
    });

    enableWebhook.resolve(enabledWebhookWithSecret);
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.queryByText('secret-once')).toBeNull();
    expect(screen.queryByText('webhook:github:main:enabled')).toBeNull();
    expect(screen.queryByText('Push webhook')).toBeNull();
    expect((screen.getByLabelText('Auto-pull interval') as HTMLSelectElement).value).toBe('3600');
  });

  it('clears the one-time secret when the workspace disconnects', async () => {
    apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookConfig);
    const { rerender } = renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Rotate secret' }));
    expect(await screen.findByText('rotated-secret')).toBeTruthy();

    rerender(
      <WorkspaceScmWizard
        workspace={{ ...upstreamWorkspace({ connected: false, git_url: null, remote_role: null }) }}
        onClose={vi.fn()}
        onSyncComplete={vi.fn()}
        onWorkspaceChanged={vi.fn()}
      />,
    );

    await openGitSourceTab();
    expect(screen.queryByText('rotated-secret')).toBeNull();
    expect(screen.queryByText('Push webhook')).toBeNull();
  });
});
