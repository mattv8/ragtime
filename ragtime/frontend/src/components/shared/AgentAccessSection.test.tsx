import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AgentAccessSection } from './AgentAccessSection';

const apiMock = vi.hoisted(() => ({
  getWorkspaceAgentAccess: vi.fn(),
  enableWorkspaceAgentAccess: vi.fn(),
  disableWorkspaceAgentAccess: vi.fn(),
  rotateWorkspaceAgentAccess: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

function createDeferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((innerResolve, innerReject) => {
    resolve = innerResolve;
    reject = innerReject;
  });
  return { promise, resolve, reject };
}

function createDisabledStatus(workspaceId: string) {
  return {
    workspace_id: workspaceId,
    enabled: false,
    allow_task_submission: true,
    token: null,
    agent_url: null,
    created_at: null,
    last_used_at: null,
    hit_count: 0,
  };
}

function createEnabledStatus(workspaceId: string, token: string) {
  return {
    ...createDisabledStatus(workspaceId),
    enabled: true,
    token,
    agent_url: `https://ragtime.example.com/agent/w/${token}`,
    created_at: '2026-07-15T12:00:00Z',
  };
}

const DISABLED_STATUS = createDisabledStatus('ws-1');
const ENABLED_STATUS = createEnabledStatus('ws-1', 'tok-abc');

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('AgentAccessSection', () => {
  it('loads disabled state and enables external agent access', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceAgentAccess.mockResolvedValue(DISABLED_STATUS);
    apiMock.enableWorkspaceAgentAccess.mockResolvedValue(ENABLED_STATUS);

    render(<AgentAccessSection workspaceId="ws-1" />);

    const enableButton = await screen.findByRole('button', { name: 'Enable Agent Access' });
    await user.click(enableButton);

    expect(apiMock.enableWorkspaceAgentAccess).toHaveBeenCalledWith('ws-1', true);
    expect(
      await screen.findByDisplayValue('https://ragtime.example.com/agent/w/tok-abc'),
    ).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy agent manifest URL' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy agent instructions' })).toBeTruthy();
  });

  it('retries initial load failures and copies secret-safe instructions', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceAgentAccess
      .mockRejectedValueOnce(new Error('Failed to load agent access'))
      .mockResolvedValueOnce(ENABLED_STATUS);

    render(<AgentAccessSection workspaceId="ws-1" />);

    expect((await screen.findByRole('alert')).textContent).toContain('Failed to load agent access');
    await user.click(screen.getByRole('button', { name: 'Retry' }));

    expect(
      await screen.findByRole('switch', {
        name: 'Allow external agents to submit build tasks',
      }),
    ).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Copy agent instructions' }));

    const copiedInstructions = await navigator.clipboard.readText();
    expect(copiedInstructions).toContain('collaborate with me in this workspace');
    expect(copiedInstructions).toContain('workspace agent manifest');
    expect(copiedInstructions).toContain('/context');
    expect(copiedInstructions).toContain("Ragtime's builder chat");
    expect(copiedInstructions).toContain('Treat this bearer URL as a secret');
    expect(screen.getByRole('button', { name: 'Agent instructions copied' })).toBeTruthy();
  });

  it('updates task permission, rotates the token, and disables the active token', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceAgentAccess.mockResolvedValue(ENABLED_STATUS);
    apiMock.enableWorkspaceAgentAccess.mockResolvedValue({
      ...ENABLED_STATUS,
      allow_task_submission: false,
    });
    apiMock.rotateWorkspaceAgentAccess.mockResolvedValue({
      ...ENABLED_STATUS,
      token: 'tok-new',
      agent_url: 'https://ragtime.example.com/agent/w/tok-new',
    });
    apiMock.disableWorkspaceAgentAccess.mockResolvedValue(DISABLED_STATUS);

    render(<AgentAccessSection workspaceId="ws-1" />);

    const taskSwitch = await screen.findByRole('switch', {
      name: 'Allow external agents to submit build tasks',
    });
    await user.click(taskSwitch);
    expect(apiMock.enableWorkspaceAgentAccess).toHaveBeenCalledWith('ws-1', false);

    await user.click(screen.getByRole('button', { name: 'Rotate Token' }));
    expect(apiMock.rotateWorkspaceAgentAccess).toHaveBeenCalledWith('ws-1');
    expect(
      await screen.findByDisplayValue('https://ragtime.example.com/agent/w/tok-new'),
    ).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Disable Agent Access' }));
    expect(apiMock.disableWorkspaceAgentAccess).toHaveBeenCalledWith('ws-1');
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Enable Agent Access' })).toBeTruthy();
    });
  });

  it('ignores stale workspace action success and failure after switching workspaces', async () => {
    const user = userEvent.setup();
    const rotateDeferred = createDeferred<ReturnType<typeof createEnabledStatus>>();
    const failingRotateDeferred = createDeferred<ReturnType<typeof createEnabledStatus>>();
    const workspaceBLoad = createDeferred<ReturnType<typeof createDisabledStatus>>();

    apiMock.getWorkspaceAgentAccess
      .mockResolvedValueOnce(createEnabledStatus('ws-a', 'tok-a'))
      .mockImplementationOnce(() => workspaceBLoad.promise)
      .mockResolvedValueOnce(createEnabledStatus('ws-a', 'tok-a'))
      .mockResolvedValueOnce(createDisabledStatus('ws-b'));
    apiMock.rotateWorkspaceAgentAccess
      .mockImplementationOnce(() => rotateDeferred.promise)
      .mockImplementationOnce(() => failingRotateDeferred.promise);

    const { rerender } = render(<AgentAccessSection workspaceId="ws-a" />);

    await screen.findByDisplayValue('https://ragtime.example.com/agent/w/tok-a');
    await user.click(screen.getByRole('button', { name: 'Rotate Token' }));

    rerender(<AgentAccessSection workspaceId="ws-b" />);

    expect(screen.queryByDisplayValue('https://ragtime.example.com/agent/w/tok-a')).toBeNull();
    expect(screen.getByText('Loading agent access...')).toBeTruthy();

    workspaceBLoad.resolve(createDisabledStatus('ws-b'));
    await screen.findByRole('button', { name: 'Enable Agent Access' });

    rotateDeferred.resolve(createEnabledStatus('ws-a', 'tok-a-rotated'));
    await waitFor(() => {
      expect(
        screen.queryByDisplayValue('https://ragtime.example.com/agent/w/tok-a-rotated'),
      ).toBeNull();
      expect(screen.queryByRole('alert')).toBeNull();
      expect(screen.getByRole('button', { name: 'Enable Agent Access' })).toBeTruthy();
    });

    rerender(<AgentAccessSection workspaceId="ws-a" />);
    await screen.findByDisplayValue('https://ragtime.example.com/agent/w/tok-a');
    await user.click(screen.getByRole('button', { name: 'Rotate Token' }));

    rerender(<AgentAccessSection workspaceId="ws-b" />);
    await screen.findByRole('button', { name: 'Enable Agent Access' });

    failingRotateDeferred.reject(new Error('stale rotate failure'));
    await waitFor(() => {
      expect(screen.queryByRole('alert')).toBeNull();
      expect(screen.queryByDisplayValue('https://ragtime.example.com/agent/w/tok-a')).toBeNull();
      expect(screen.getByRole('button', { name: 'Enable Agent Access' })).toBeTruthy();
    });
  });
});
