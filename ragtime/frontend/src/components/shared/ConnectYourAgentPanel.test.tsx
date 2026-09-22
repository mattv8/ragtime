import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConnectYourAgentPanel } from './ConnectYourAgentPanel';

const apiMock = vi.hoisted(() => ({
  listWorkspaceDevelopmentCredentials: vi.fn(),
  createWorkspaceDevelopmentCredential: vi.fn(),
  rotateWorkspaceDevelopmentCredential: vi.fn(),
  revokeWorkspaceDevelopmentCredential: vi.fn(),
  executeWorkspaceDevelopmentOperation: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

const credential = (overrides = {}) => ({
  id: 'credential-1',
  workspace_id: 'workspace-1',
  user_id: 'user-1',
  name: 'External agent',
  scopes: ['read'],
  expires_at: null,
  revoked_at: null,
  created_at: '2026-09-19T00:00:00Z',
  updated_at: '2026-09-19T00:00:00Z',
  ...overrides,
});

describe('ConnectYourAgentPanel', () => {
  afterEach(cleanup);

  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([]);
    apiMock.executeWorkspaceDevelopmentOperation.mockResolvedValue([]);
  });

  it('adds the created credential card immediately and opens its connect step', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential({ name: 'My Agent' }),
      token: 'fresh-token-value',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.clear(screen.getByLabelText('Credential name'));
    await user.type(screen.getByLabelText('Credential name'), 'My Agent');
    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));

    const card = await screen.findByRole('article', { name: 'My Agent credential' });
    expect(within(card).getByText('Active')).toBeTruthy();
    expect(
      within(card)
        .getByRole('list', { name: 'Coding agent setup steps' })
        .querySelector('[aria-current="step"]')?.textContent,
    ).toContain('Connect agent');
    expect(within(card).getByRole('button', { name: /copy setup instructions/i })).toBeTruthy();
  });

  it('keeps the create step open and shows an error when creation fails', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockRejectedValue(new Error('Creation failed'));
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));

    expect((await screen.findByRole('alert')).textContent).toContain('Creation failed');
    expect(screen.queryByRole('article')).toBeNull();
    expect(screen.getByRole('button', { name: /create credential and continue/i })).toBeTruthy();
  });

  it('does not create a second credential when navigating setup steps', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'fresh-token-value',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await screen.findByRole('button', { name: /^next: start working$/i });
    await user.click(screen.getByRole('button', { name: /^next: start working$/i }));
    await user.click(screen.getByRole('button', { name: /back to connect agent/i }));

    expect(apiMock.createWorkspaceDevelopmentCredential).toHaveBeenCalledTimes(1);
  });

  it('preserves a newly created secret when reopening its setup after Done', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'preserved-token',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await user.click(screen.getByRole('button', { name: /^next: start working$/i }));
    await user.click(screen.getByRole('button', { name: /^done$/i }));
    await user.click(screen.getByRole('button', { name: /^setup instructions$/i }));
    await user.click(screen.getByRole('button', { name: /^copy setup instructions$/i }));

    expect(await navigator.clipboard.readText()).toContain('preserved-token');
    expect(apiMock.rotateWorkspaceDevelopmentCredential).not.toHaveBeenCalled();
  });

  it('keeps existing cards visible and cancels explicit new-agent creation', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const card = await screen.findByRole('article', { name: 'External agent credential' });
    await user.click(screen.getByRole('button', { name: /^new agent$/i }));
    expect(screen.getByLabelText('Credential name')).toBeTruthy();
    expect(screen.getByRole('article', { name: 'External agent credential' })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /^cancel creation$/i }));

    expect(screen.queryByLabelText('Credential name')).toBeNull();
    expect(card).toBeTruthy();
    expect(apiMock.createWorkspaceDevelopmentCredential).not.toHaveBeenCalled();
  });

  it('offers setup recovery for a returned credential without exposing a secret', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([
      credential({ name: 'Saved agent' }),
    ]);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const card = await screen.findByRole('article', { name: 'Saved agent credential' });
    await user.click(within(card).getByRole('button', { name: /setup instructions/i }));

    expect(await screen.findByText(/secret is only shown when created or rotated/i)).toBeTruthy();
    expect(
      within(card).getByRole('button', { name: /rotate credential to recover setup/i }),
    ).toBeTruthy();
  });

  it('warns before rotation, replaces setup instructions with the rotated token, and revokes it', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    apiMock.rotateWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'rotated-token',
    });
    apiMock.revokeWorkspaceDevelopmentCredential.mockResolvedValue(
      credential({ revoked_at: '2026-09-19T00:00:01Z' }),
    );
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const card = await screen.findByRole('article', { name: 'External agent credential' });
    await user.click(within(card).getByRole('button', { name: /rotate credential/i }));
    expect(screen.getByText(/old token stops working/i)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: /rotate and continue/i }));
    await user.click(screen.getByRole('button', { name: /^copy setup instructions$/i }));
    expect(await navigator.clipboard.readText()).toContain('rotated-token');

    await user.click(within(card).getByRole('button', { name: /revoke credential/i }));
    expect(await within(card).findByText('Revoked')).toBeTruthy();
    expect(screen.queryByRole('button', { name: /^copy setup instructions$/i })).toBeNull();
  });

  it('clears setup state when workspace or permission changes', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'workspace-one-token',
    });
    const { rerender } = render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);
    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await user.click(screen.getByRole('button', { name: /^copy setup instructions$/i }));
    expect(await navigator.clipboard.readText()).toContain('workspace-one-token');

    rerender(<ConnectYourAgentPanel workspaceId="workspace-2" canManage />);
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: /^copy setup instructions$/i })).toBeNull(),
    );
    rerender(<ConnectYourAgentPanel workspaceId="workspace-2" canManage={false} />);
    expect(await screen.findByText(/only workspace owners and admins/i)).toBeTruthy();
    expect(screen.queryByRole('article')).toBeNull();
  });

  it('keeps activity collapsed until requested and copies setup instructions', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'fresh-token-value',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    expect(
      (screen.getByText('Development activity').closest('details') as HTMLDetailsElement).open,
    ).toBe(false);
    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await screen.findByRole('button', { name: /copy setup instructions/i });
    await user.click(screen.getByRole('button', { name: /copy setup instructions/i }));
    expect(await navigator.clipboard.readText()).toContain('fresh-token-value');
    await user.click(screen.getByText('Development activity'));
    expect(
      (screen.getByText('Development activity').closest('details') as HTMLDetailsElement).open,
    ).toBe(true);
  });

  it('cancels a running development job and refreshes activity', async () => {
    const user = userEvent.setup();
    apiMock.executeWorkspaceDevelopmentOperation
      .mockResolvedValueOnce([{ id: 'job-1', status: 'running' }])
      .mockResolvedValueOnce({ id: 'job-1' })
      .mockResolvedValueOnce([{ id: 'job-1', status: 'cancelled' }]);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByText('Development activity'));
    await user.click(screen.getByRole('button', { name: /^refresh$/i }));
    await screen.findByText('running');
    await user.click(screen.getByRole('button', { name: /^cancel$/i }));

    await waitFor(() => {
      expect(apiMock.executeWorkspaceDevelopmentOperation).toHaveBeenNthCalledWith(
        2,
        'workspace-1',
        'exec_cancel',
        { job_id: 'job-1' },
      );
      expect(apiMock.executeWorkspaceDevelopmentOperation).toHaveBeenNthCalledWith(
        3,
        'workspace-1',
        'exec_list',
      );
    });
    expect(screen.getByText('cancelled')).toBeTruthy();
  });
});
