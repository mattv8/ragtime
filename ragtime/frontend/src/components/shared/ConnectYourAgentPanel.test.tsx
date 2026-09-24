import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ConnectYourAgentPanel } from './ConnectYourAgentPanel';

vi.mock('../DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({ onDelete, buttonText }: { onDelete: () => void; buttonText: string }) => (
    <button type="button" onClick={onDelete}>
      {buttonText}
    </button>
  ),
}));

const apiMock = vi.hoisted(() => ({
  listWorkspaceDevelopmentCredentials: vi.fn(),
  createWorkspaceDevelopmentCredential: vi.fn(),
  rotateWorkspaceDevelopmentCredential: vi.fn(),
  revokeWorkspaceDevelopmentCredential: vi.fn(),
  deleteWorkspaceDevelopmentCredential: vi.fn(),
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

  it('copies the selected client profile in the canonical setup instructions', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'fresh-token-value',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await user.click(await screen.findByRole('button', { name: /^codex$/i }));
    await user.click(screen.getByRole('button', { name: /^copy setup instructions$/i }));

    expect(await navigator.clipboard.readText()).toContain('**Profile ID:** codex');
  });

  it('hides the setup guide when no credentials exist yet', async () => {
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await screen.findByRole('heading', { name: 'Development credentials' });
    expect(screen.queryByText('MCP endpoint')).toBeNull();
    expect(screen.queryByRole('button', { name: /^opencode$/i })).toBeNull();
  });

  it('applies a matching rail selection when credentials exist', async () => {
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    render(
      <ConnectYourAgentPanel
        workspaceId="workspace-1"
        canManage
        selectionRequest={{ requestId: 1, clientId: 'claude-code', workspaceId: 'workspace-1' }}
      />,
    );

    expect(
      (await screen.findByRole('button', { name: /^claude code$/i })).getAttribute('aria-pressed'),
    ).toBe('true');
    expect(
      screen.getByRole('button', { name: /^claude code$/i }).getAttribute('aria-expanded'),
    ).toBe('true');
  });

  it('collapses the active client guide without changing the credential setup state', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'preserved-token',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    const codexButton = await screen.findByRole('button', { name: /^codex$/i });
    await user.click(codexButton);
    await user.click(codexButton);

    expect(codexButton.getAttribute('aria-pressed')).toBe('false');
    expect(codexButton.getAttribute('aria-expanded')).toBe('false');
    expect(screen.getByRole('button', { name: /^copy setup instructions$/i })).toBeTruthy();

    await user.click(screen.getByRole('button', { name: /^copy setup instructions$/i }));
    expect(await navigator.clipboard.readText()).toContain('preserved-token');
  });

  it('reopens a collapsed matching client guide for a newer rail request', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    const { rerender } = render(
      <ConnectYourAgentPanel
        workspaceId="workspace-1"
        canManage
        selectionRequest={{ requestId: 1, clientId: 'codex', workspaceId: 'workspace-1' }}
      />,
    );

    const codexButton = await screen.findByRole('button', { name: /^codex$/i });
    await user.click(codexButton);
    expect(codexButton.getAttribute('aria-expanded')).toBe('false');

    rerender(
      <ConnectYourAgentPanel
        workspaceId="workspace-1"
        canManage
        selectionRequest={{ requestId: 2, clientId: 'codex', workspaceId: 'workspace-1' }}
      />,
    );

    await waitFor(() => expect(codexButton.getAttribute('aria-expanded')).toBe('true'));
  });

  it('accepts a matching rail request after changing workspaces', async () => {
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    const { rerender } = render(
      <ConnectYourAgentPanel
        workspaceId="workspace-1"
        canManage
        selectionRequest={{ requestId: 1, clientId: 'codex', workspaceId: 'workspace-1' }}
      />,
    );

    await screen.findByRole('button', { name: /^codex$/i });
    rerender(
      <ConnectYourAgentPanel
        workspaceId="workspace-2"
        canManage
        selectionRequest={{ requestId: 1, clientId: 'claude-code', workspaceId: 'workspace-2' }}
      />,
    );

    await waitFor(() =>
      expect(
        screen.getByRole('button', { name: /^claude code$/i }).getAttribute('aria-expanded'),
      ).toBe('true'),
    );
  });

  it('does not reset a manual client selection during unrelated rerenders', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    const { rerender } = render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(await screen.findByRole('button', { name: /^codex$/i }));
    rerender(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    expect(screen.getByRole('button', { name: /^codex$/i }).getAttribute('aria-pressed')).toBe(
      'true',
    );
  });

  it('ignores a rail selection request for a different workspace', async () => {
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    render(
      <ConnectYourAgentPanel
        workspaceId="workspace-1"
        canManage
        selectionRequest={{ requestId: 1, clientId: 'codex', workspaceId: 'workspace-2' }}
      />,
    );

    expect(
      (await screen.findByRole('button', { name: /^opencode$/i })).getAttribute('aria-pressed'),
    ).toBe('false');
  });

  it('hides setup guide for non-managers and skips fetching credentials', async () => {
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage={false} />);

    expect(await screen.findByText(/only workspace owners and admins/i)).toBeTruthy();
    expect(screen.queryByText('MCP endpoint')).toBeNull();
    expect(screen.queryByRole('button', { name: /^opencode$/i })).toBeNull();
    expect(apiMock.listWorkspaceDevelopmentCredentials).not.toHaveBeenCalled();
  });

  it('renders development credentials before MCP client setup', async () => {
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const credentialsHeading = await screen.findByRole('heading', {
      name: 'Development credentials',
    });
    const endpointLabel = await screen.findByText('MCP endpoint');

    expect(
      credentialsHeading.compareDocumentPosition(endpointLabel) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('hides existing cards while the new-agent form is open and restores them on dismiss', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await screen.findByRole('article', { name: 'External agent credential' });
    await user.click(screen.getByRole('button', { name: /^new credential$/i }));

    // Create form is visible; existing card is hidden
    expect(screen.getByLabelText('Credential name')).toBeTruthy();
    expect(screen.queryByRole('article', { name: 'External agent credential' })).toBeNull();

    await user.click(screen.getByRole('button', { name: /^dismiss new agent form$/i }));

    // Form dismissed; card reappears; no credential was created
    expect(screen.queryByLabelText('Credential name')).toBeNull();
    expect(screen.getByRole('article', { name: 'External agent credential' })).toBeTruthy();
    expect(apiMock.createWorkspaceDevelopmentCredential).not.toHaveBeenCalled();
  });

  it('keeps the new-agent form open while credential creation is in flight', async () => {
    const user = userEvent.setup();
    let resolveCreate:
      | ((value: ReturnType<typeof credential> & { token: string }) => void)
      | undefined;
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([
      credential({ id: 'existing-credential' }),
    ]);
    apiMock.createWorkspaceDevelopmentCredential.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveCreate = resolve;
        }),
    );
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await screen.findByRole('article', { name: 'External agent credential' });
    await user.click(screen.getByRole('button', { name: /^new credential$/i }));
    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));

    expect(
      screen.getByRole('button', { name: /dismiss new agent form/i }).hasAttribute('disabled'),
    ).toBe(true);

    resolveCreate?.({ ...credential({ id: 'created-credential' }), token: 'fresh-token-value' });
    await waitFor(() =>
      expect(screen.queryByRole('button', { name: /dismiss new agent form/i })).toBeNull(),
    );
  });

  it('keeps a freshly created manual-client secret available without a recovery prompt', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'cursor-token',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await user.click(await screen.findByRole('button', { name: /^cursor$/i }));

    expect(screen.queryByText(/secret is only shown when created or rotated/i)).toBeNull();
    await user.click(screen.getByText('Manual connection details'));
    await user.click(screen.getByRole('button', { name: /copy development credential/i }));
    expect(await navigator.clipboard.readText()).toBe('cursor-token');
  });

  it('does not offer credential recovery for ChatGPT', async () => {
    const user = userEvent.setup();
    apiMock.createWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'chatgpt-token',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await user.click(screen.getByRole('button', { name: /create credential and continue/i }));
    await user.click(await screen.findByRole('button', { name: /^chatgpt$/i }));

    expect(screen.getByText(/cannot connect ChatGPT/i)).toBeTruthy();
    expect(
      screen.queryByRole('button', { name: /rotate credential to recover setup/i }),
    ).toBeNull();
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

  it('does not expose a secret when an expired credential is selected for recovery', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([
      credential({ expires_at: '2020-01-01T00:00:00Z' }),
    ]);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const card = await screen.findByRole('article', { name: 'External agent credential' });
    expect(within(card).getByText('Expired')).toBeTruthy();
    await user.click(within(card).getByRole('button', { name: /setup instructions/i }));

    expect(screen.queryByRole('button', { name: /^copy setup instructions$/i })).toBeNull();
    expect(
      screen.getByRole('button', { name: /rotate credential to recover setup/i }),
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

    // Advance to start step then dismiss so card actions become accessible
    await user.click(within(card).getByRole('button', { name: /^next: start working$/i }));
    await user.click(within(card).getByRole('button', { name: /^done$/i }));

    await user.click(within(card).getByRole('button', { name: /revoke credential/i }));
    expect(await within(card).findByText('Revoked')).toBeTruthy();
    expect(screen.queryByRole('button', { name: /^copy setup instructions$/i })).toBeNull();
  });

  it('retains the selected client when rotating a credential', async () => {
    const user = userEvent.setup();
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([credential()]);
    apiMock.rotateWorkspaceDevelopmentCredential.mockResolvedValue({
      ...credential(),
      token: 'rotated-token',
    });
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    await screen.findByRole('article', { name: 'External agent credential' });
    await user.click(screen.getByRole('button', { name: /^claude code$/i }));
    await user.click(screen.getByRole('button', { name: /rotate credential/i }));
    await user.click(screen.getByRole('button', { name: /rotate and continue/i }));
    await user.click(screen.getByRole('button', { name: /^copy setup instructions$/i }));

    expect(await navigator.clipboard.readText()).toContain('**Profile ID:** claude-code');
  });

  it('shows Delete button on revoked credentials and deletes via callback', async () => {
    const user = userEvent.setup();
    const revokedCred = credential({ revoked_at: '2026-09-19T00:00:01Z' });
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([revokedCred]);
    apiMock.deleteWorkspaceDevelopmentCredential.mockResolvedValue(undefined);
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const card = await screen.findByRole('article', { name: 'External agent credential' });
    expect(within(card).getByRole('button', { name: /^delete$/i })).toBeTruthy();

    await user.click(within(card).getByRole('button', { name: /^delete$/i }));

    await waitFor(() => {
      expect(apiMock.deleteWorkspaceDevelopmentCredential).toHaveBeenCalledWith(
        'workspace-1',
        revokedCred.id,
      );
    });
    expect(screen.queryByRole('article', { name: 'External agent credential' })).toBeNull();

    const newCredentialButton = screen.getByRole('button', { name: /^new credential$/i });
    await user.click(newCredentialButton);
    expect(screen.getByLabelText('Credential name')).toBeTruthy();
  });

  it('shows error and keeps card when delete call fails', async () => {
    const user = userEvent.setup();
    const revokedCred = credential({ revoked_at: '2026-09-19T00:00:01Z' });
    apiMock.listWorkspaceDevelopmentCredentials.mockResolvedValue([revokedCred]);
    apiMock.deleteWorkspaceDevelopmentCredential.mockRejectedValue(new Error('Delete failed'));
    render(<ConnectYourAgentPanel workspaceId="workspace-1" canManage />);

    const card = await screen.findByRole('article', { name: 'External agent credential' });
    await user.click(within(card).getByRole('button', { name: /^delete$/i }));

    await waitFor(() => {
      expect((screen.getByRole('alert') as HTMLElement).textContent).toContain('Delete failed');
    });
    expect(screen.getByRole('article', { name: 'External agent credential' })).toBeTruthy();
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
});
