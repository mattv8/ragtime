import type { ComponentProps, ReactNode } from 'react';

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  GitWebhookConfig,
  GitWebhookDelivery,
  GitWebhookEnableResponse,
  UserSpaceWorkspace,
} from '@/types';

import { WorkspaceScmWizard } from './WorkspaceScmWizard';

const apiMock = vi.hoisted(() => ({
  getUserSpacePreviewSettings: vi.fn(),
  fetchUserSpaceWorkspaceScmBranches: vi.fn(),
  getUserSpaceWorkspaceScmWebhook: vi.fn(),
  enableUserSpaceWorkspaceScmWebhook: vi.fn(),
  rotateUserSpaceWorkspaceScmWebhookSecret: vi.fn(),
  disableUserSpaceWorkspaceScmWebhook: vi.fn(),
  listUserSpaceWorkspaceScmWebhookDeliveries: vi.fn(),
}));

const latestWebhookSettingsProps = vi.hoisted(() => ({
  config: null as GitWebhookConfig | null,
  revealedSecret: null as string | null,
  deliveries: [] as GitWebhookDelivery[],
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
    deliveries,
    disabled,
    onEnable,
    onRotate,
    onDisable,
    onDismissSecret,
  }: {
    config: GitWebhookConfig;
    revealedSecret: string | null;
    deliveries: GitWebhookDelivery[];
    disabled: boolean;
    onEnable: () => void;
    onRotate: () => void;
    onDisable: () => void;
    onDismissSecret: () => void;
  }) =>
    (() => {
      latestWebhookSettingsProps.config = config;
      latestWebhookSettingsProps.revealedSecret = revealedSecret;
      latestWebhookSettingsProps.deliveries = deliveries;
      latestWebhookSettingsProps.disabled = Boolean(disabled);
      return (
        <div data-testid="git-webhook-settings-mock">
          <div>Push webhook</div>
          <div>{`webhook:${config.provider}:${config.branch}:${config.enabled ? 'enabled' : 'disabled'}`}</div>
          <div>{`deliveries:${deliveries.length}`}</div>
          {deliveries[0] && <div>{`delivery-id:${deliveries[0].id}`}</div>}
          {revealedSecret && <div>{revealedSecret}</div>}
          <button type="button" onClick={onDismissSecret} disabled={disabled}>
            Dismiss webhook secret
          </button>
          {config.enabled ? (
            <>
              <button type="button" onClick={onRotate} disabled={disabled}>
                Rotate secret
              </button>
              <button type="button" onClick={onDisable} disabled={disabled}>
                Disable webhook
              </button>
            </>
          ) : (
            <button type="button" onClick={onEnable} disabled={disabled}>
              Enable push webhook
            </button>
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
  webhook_url: null,
  provider: 'github',
  branch: 'main',
  created_at: null,
};

const enabledWebhookConfig: GitWebhookConfig = {
  enabled: true,
  webhook_url: 'https://ragtime.example/webhooks/git/workspace-1',
  provider: 'github',
  branch: 'main',
  created_at: '2026-07-16T12:00:00Z',
};

const enabledWebhookWithSecret: GitWebhookEnableResponse = {
  ...enabledWebhookConfig,
  secret: 'secret-once',
};

const webhookDeliveries: GitWebhookDelivery[] = [
  {
    id: 'delivery-1',
    event_name: 'push',
    branch: 'main',
    head_commit: 'abc123',
    status: 'completed',
    message: 'done',
    received_at: '2026-07-16T12:00:00Z',
    started_at: '2026-07-16T12:00:01Z',
    completed_at: '2026-07-16T12:00:02Z',
  },
];

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
  fireEvent.click(screen.getByRole('button', { name: 'Git Source' }));
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
  apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(disabledWebhookConfig);
  apiMock.enableUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookWithSecret);
  apiMock.rotateUserSpaceWorkspaceScmWebhookSecret.mockResolvedValue({
    ...enabledWebhookConfig,
    secret: 'rotated-secret',
  });
  apiMock.disableUserSpaceWorkspaceScmWebhook.mockResolvedValue(undefined);
  apiMock.listUserSpaceWorkspaceScmWebhookDeliveries.mockResolvedValue(webhookDeliveries);
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
  latestWebhookSettingsProps.deliveries = [];
  latestWebhookSettingsProps.disabled = false;
});

describe('WorkspaceScmWizard webhook integration', () => {
  it('shows webhook controls for a connected upstream remote independently of auto-pull', async () => {
    renderWizard(upstreamWorkspace({ auto_pull_enabled: false }));

    await openGitSourceTab();

    expect(await screen.findByText('Push webhook')).toBeTruthy();
    const hint = screen.getByText(
      'Scheduled auto-pull is off. Push webhooks can still trigger pulls.',
    );
    expect(hint.className).toContain('userspace-muted');
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

  it('enables the workspace webhook, refreshes deliveries, and notifies parent refresh', async () => {
    const onWorkspaceChanged = vi.fn();
    renderWizard(upstreamWorkspace(), { onWorkspaceChanged });

    await openGitSourceTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Enable push webhook' }));

    await waitFor(() => {
      expect(apiMock.enableUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect(await screen.findByText('secret-once')).toBeTruthy();
    expect(await screen.findByText('delivery-id:delivery-1')).toBeTruthy();
    expect(onWorkspaceChanged).toHaveBeenCalledTimes(1);
  });

  it('rotates and disables the webhook while refreshing deliveries after each mutation', async () => {
    apiMock.getUserSpaceWorkspaceScmWebhook.mockResolvedValue(enabledWebhookConfig);
    const onWorkspaceChanged = vi.fn();
    renderWizard(upstreamWorkspace(), { onWorkspaceChanged });

    await openGitSourceTab();

    fireEvent.click(await screen.findByRole('button', { name: 'Rotate secret' }));
    await waitFor(() => {
      expect(apiMock.rotateUserSpaceWorkspaceScmWebhookSecret).toHaveBeenCalledWith('workspace-1');
    });
    expect(await screen.findByText('rotated-secret')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Disable webhook' }));
    await waitFor(() => {
      expect(apiMock.disableUserSpaceWorkspaceScmWebhook).toHaveBeenCalledWith('workspace-1');
    });
    expect(await screen.findByRole('button', { name: 'Enable push webhook' })).toBeTruthy();
    expect(apiMock.listUserSpaceWorkspaceScmWebhookDeliveries).toHaveBeenCalledTimes(3);
    expect(onWorkspaceChanged).toHaveBeenCalledTimes(2);
  });

  it('preserves the visible webhook configuration when a delivery refresh fails after enable', async () => {
    apiMock.listUserSpaceWorkspaceScmWebhookDeliveries
      .mockResolvedValueOnce(webhookDeliveries)
      .mockRejectedValueOnce(new Error('refresh failed'));

    renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Enable push webhook' }));

    expect(await screen.findByText('secret-once')).toBeTruthy();
    expect(screen.getByText('webhook:github:main:enabled')).toBeTruthy();
    await waitFor(() => {
      expect(toastApiMock.error).toHaveBeenCalledWith(
        'Failed to refresh webhook deliveries: refresh failed',
      );
    });
  });

  it('rejects stale delivery responses after a newer mutation refresh', async () => {
    const initialDeliveries = deferred<GitWebhookDelivery[]>();
    const mutationDeliveries = deferred<GitWebhookDelivery[]>();

    apiMock.listUserSpaceWorkspaceScmWebhookDeliveries
      .mockImplementationOnce(() => initialDeliveries.promise)
      .mockImplementationOnce(() => mutationDeliveries.promise);

    renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Enable push webhook' }));

    await act(async () => {
      mutationDeliveries.resolve([{ ...webhookDeliveries[0], id: 'delivery-new' }]);
      await Promise.resolve();
    });

    await act(async () => {
      initialDeliveries.resolve([{ ...webhookDeliveries[0], id: 'delivery-stale' }]);
      await Promise.resolve();
    });

    expect(latestWebhookSettingsProps.deliveries[0]?.id).toBe('delivery-new');
    expect(screen.queryByText('delivery-id:delivery-stale')).toBeNull();
    expect(latestWebhookSettingsProps.revealedSecret).toBe('secret-once');
  });

  it('rejects stale webhook mutation responses after switching workspaces', async () => {
    const enableWebhook = deferred<GitWebhookEnableResponse>();
    apiMock.getUserSpaceWorkspaceScmWebhook
      .mockResolvedValueOnce(disabledWebhookConfig)
      .mockResolvedValueOnce({ ...disabledWebhookConfig, branch: 'develop' });
    apiMock.listUserSpaceWorkspaceScmWebhookDeliveries
      .mockResolvedValueOnce(webhookDeliveries)
      .mockResolvedValueOnce([{ ...webhookDeliveries[0], id: 'delivery-two' }]);
    apiMock.enableUserSpaceWorkspaceScmWebhook.mockImplementationOnce(() => enableWebhook.promise);

    const { rerender } = renderWizard(upstreamWorkspace());

    await openGitSourceTab();
    fireEvent.click(await screen.findByRole('button', { name: 'Enable push webhook' }));

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
    expect(screen.getByText('webhook:github:develop:disabled')).toBeTruthy();
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
