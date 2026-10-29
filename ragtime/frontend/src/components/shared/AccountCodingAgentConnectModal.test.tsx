import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AccountCodingAgentConnectModal } from './AccountCodingAgentConnectModal';

const apiMock = vi.hoisted(() => ({
  listWorkspaceDevelopmentCredentials: vi.fn(),
  createWorkspaceDevelopmentCredential: vi.fn(),
  rotateWorkspaceDevelopmentCredential: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

function AccountCodingAgentConnectModalHarness() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <button type="button" onClick={() => setIsOpen(true)}>
        Open account setup
      </button>
      <AccountCodingAgentConnectModal
        isOpen={isOpen}
        onClose={() => setIsOpen(false)}
        mcpUrl="https://ragtime.example.test/mcp"
      />
    </>
  );
}

describe('AccountCodingAgentConnectModal', () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(cleanup);

  it('shows account OAuth guides and the fixed read-only discovery prompt without API calls', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    render(
      <AccountCodingAgentConnectModal
        isOpen
        onClose={onClose}
        mcpUrl="https://ragtime.example.test/mcp"
      />,
    );

    expect(screen.getByRole('dialog', { name: 'Connect your account agent' })).toBeTruthy();
    expect(
      screen.getByText(/Call workspace_development_workspaces to list workspaces I can access/),
    ).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'ChatGPT' }));
    await user.click(screen.getByRole('tab', { name: 'OpenCode V2' }));
    expect(screen.getByText(/opencode mcp add ragtime --global --url/)).toBeTruthy();
    await user.keyboard('{Escape}');
    expect(onClose).toHaveBeenCalledOnce();
    expect(Object.values(apiMock).every((request) => request.mock.calls.length === 0)).toBe(true);
  });

  it('does not register document handling or call APIs while closed', () => {
    const addEventListener = vi.spyOn(document, 'addEventListener');
    render(
      <AccountCodingAgentConnectModal
        isOpen={false}
        onClose={vi.fn()}
        mcpUrl="https://ragtime.example.test/mcp"
      />,
    );

    expect(addEventListener).not.toHaveBeenCalledWith('keydown', expect.any(Function));
    expect(Object.values(apiMock).every((request) => request.mock.calls.length === 0)).toBe(true);
    addEventListener.mockRestore();
  });

  it('closes from its backdrop but not its content and restores trigger focus', async () => {
    const user = userEvent.setup();
    render(<AccountCodingAgentConnectModalHarness />);
    const trigger = screen.getByRole('button', { name: 'Open account setup' });

    await user.click(trigger);
    await user.click(screen.getByRole('dialog', { name: 'Connect your account agent' }));
    expect(screen.getByRole('dialog', { name: 'Connect your account agent' })).toBeTruthy();

    fireEvent.click(
      document.querySelector('[data-userspace-modal="account-coding-agent-connect"]')!,
    );

    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
    expect(document.activeElement).toBe(trigger);
  });

  it('keeps tab focus inside visible panels after every OAuth tab has been visited', async () => {
    const user = userEvent.setup();
    render(
      <AccountCodingAgentConnectModal
        isOpen
        onClose={vi.fn()}
        mcpUrl="https://ragtime.example.test/mcp"
      />,
    );

    await user.click(screen.getByRole('tab', { name: 'ChatGPT' }));
    await user.click(screen.getByRole('tab', { name: 'OpenCode V2' }));
    const connectionCopy = screen.getByRole('button', {
      name: 'Copy OpenCode OAuth connection command',
    });
    connectionCopy.focus();
    await user.keyboard('{Tab}');

    expect(document.activeElement).toBe(
      screen.getByRole('button', { name: 'Copy read-only discovery prompt' }),
    );
    expect(Object.values(apiMock).every((request) => request.mock.calls.length === 0)).toBe(true);
  });
});
