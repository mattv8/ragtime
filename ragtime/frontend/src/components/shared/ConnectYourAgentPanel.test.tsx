import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConnectYourAgentPanel } from './ConnectYourAgentPanel';

const apiMock = vi.hoisted(() => ({
  listWorkspaceDevelopmentCredentials: vi.fn(),
  createWorkspaceDevelopmentCredential: vi.fn(),
  executeWorkspaceDevelopmentOperation: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));
vi.mock('./InlineCopyButton', () => ({
  InlineCopyButton: () => <button type="button">Copy token</button>,
}));

describe('ConnectYourAgentPanel', () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([]);
    apiMock.executeWorkspaceDevelopmentOperation.mockResolvedValue([]);
  });

  it('provides MCP bootstrap details and creates a credential', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      id: 'credential-1',
      workspace_id: 'workspace-1',
      user_id: 'user-1',
      name: 'External agent',
      scopes: ['read'],
      expires_at: null,
      revoked_at: null,
      created_at: '2026-09-19T00:00:00Z',
      updated_at: '2026-09-19T00:00:00Z',
      token: 'development-token',
    });

    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);
    await user.click(screen.getByRole('button', { name: 'Connect your agent' }));

    expect(screen.getByText('workspace_development_context')).toBeTruthy();
    expect(screen.getByText('workspace_development')).toBeTruthy();
    expect(screen.getByText(/Authorization: Bearer/)).toBeTruthy();
    expect(screen.getByText(`${window.location.origin}/mcp`)).toBeTruthy();

    await user.click(screen.getByRole('button', { name: 'Create credential' }));

    await waitFor(() => {
      expect(apiMock.createWorkspaceDevelopmentCredential).toHaveBeenCalledWith('workspace-1', {
        name: 'External agent',
      });
    });
    expect(screen.getByText('development-token')).toBeTruthy();
  });

  it('starts a sandbox command and refreshes its activity', async () => {
    const user = userEvent.setup();
    apiMock.executeWorkspaceDevelopmentOperation
      .mockResolvedValueOnce({ id: 'job-1' })
      .mockResolvedValueOnce([{ id: 'job-1', status: 'running' }]);

    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);
    await user.click(screen.getByRole('button', { name: 'Connect your agent' }));
    await user.type(screen.getByRole('textbox', { name: 'Sandbox command' }), 'npm test');
    await user.click(screen.getByRole('button', { name: 'Run' }));

    await waitFor(() => {
      expect(apiMock.executeWorkspaceDevelopmentOperation).toHaveBeenNthCalledWith(
        1,
        'workspace-1',
        'exec_start',
        { command: 'npm test' },
      );
      expect(apiMock.executeWorkspaceDevelopmentOperation).toHaveBeenNthCalledWith(
        2,
        'workspace-1',
        'exec_list',
      );
    });
    expect(screen.getByText('running')).toBeTruthy();
  });
});
