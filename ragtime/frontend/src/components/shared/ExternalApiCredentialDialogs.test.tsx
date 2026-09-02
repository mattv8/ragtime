import { useState } from 'react';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ExternalApiCredentialConfirmDialog,
  ExternalApiCredentialTokenDialog,
} from './ExternalApiCredentialDialogs';

describe('ExternalApiCredentialDialogs', () => {
  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('renders the token dialog in document.body, traps focus, restores focus, and only closes through the saved action', async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();
    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <div data-testid="parent-shell">
          <button type="button" onClick={() => setOpen(true)}>
            Open token dialog
          </button>
          {open ? (
            <ExternalApiCredentialTokenDialog
              workspaceId="ws-123"
              tokenState={{
                token: 'rtws_secret_token',
                prefix: 'rtws_abcd1234',
                label: 'August workpapers',
                operation: 'Created',
              }}
              onClose={() => {
                onClose();
                setOpen(false);
              }}
            />
          ) : null}
        </div>
      );
    }
    render(<Harness />);

    const parentShell = screen.getByTestId('parent-shell');
    const trigger = within(parentShell).getByRole('button', { name: 'Open token dialog' });
    await user.click(trigger);
    const dialog = document.body.querySelector('#userspace-external-api-token-dialog');
    expect(dialog).toBeTruthy();
    expect(parentShell.contains(dialog)).toBe(false);
    expect(screen.getByText('Created credential')).toBeTruthy();
    expect(screen.getByText('August workpapers')).toBeTruthy();
    expect(screen.getByText('rtws_abcd1234')).toBeTruthy();
    expect(screen.getByText('Copy this token now. It cannot be shown again.')).toBeTruthy();

    const copyButton = screen.getByRole('button', { name: /copy token/i });
    const savedButton = screen.getByRole('button', { name: /i saved this token/i });
    await waitFor(() => {
      expect(document.activeElement).toBe(copyButton);
    });

    await user.click(screen.getByTestId('parent-shell'));
    expect(onClose).not.toHaveBeenCalled();

    await user.keyboard('{Escape}');
    expect(onClose).not.toHaveBeenCalled();

    savedButton.focus();
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Tab', bubbles: true }));
    expect(document.activeElement).toBe(copyButton);
    copyButton.focus();
    document.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'Tab', shiftKey: true, bubbles: true }),
    );
    expect(document.activeElement).toBe(savedButton);

    await user.click(copyButton);
    expect(await screen.findByText('Copied')).toBeTruthy();

    await user.click(savedButton);
    expect(onClose).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog', { name: /created credential/i })).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it('switches token usage tabs with required aria contracts and snippets', async () => {
    const user = userEvent.setup();
    render(
      <ExternalApiCredentialTokenDialog
        workspaceId="workspace-9"
        tokenState={{
          token: 'rtws_secret_token',
          prefix: 'rtws_abcd1234',
          label: 'August workpapers',
          operation: 'Rotated',
        }}
        onClose={() => undefined}
      />,
    );

    expect(screen.getByText('Rotated credential')).toBeTruthy();
    const tablist = screen.getByRole('tablist');
    const curlTab = within(tablist).getByRole('tab', { name: 'curl' });
    const powerQueryTab = within(tablist).getByRole('tab', { name: 'Power Query' });
    const curlPanelId = curlTab.getAttribute('aria-controls');
    const powerQueryPanelId = powerQueryTab.getAttribute('aria-controls');
    expect(curlPanelId).toBeTruthy();
    expect(powerQueryPanelId).toBeTruthy();
    expect(curlTab.getAttribute('aria-selected')).toBe('true');
    expect(powerQueryTab.getAttribute('aria-selected')).toBe('false');
    expect(screen.getByRole('tabpanel', { name: /curl/i }).textContent).toContain(
      'Authorization: Bearer rtws_secret_token',
    );
    expect(screen.getByRole('tabpanel', { name: /curl/i }).textContent).toContain(
      '/indexes/userspace/workspaces/workspace-9/external-api',
    );

    await user.click(powerQueryTab);

    expect(curlTab.getAttribute('aria-selected')).toBe('false');
    expect(powerQueryTab.getAttribute('aria-selected')).toBe('true');
    const powerQueryPanel = document.getElementById(powerQueryPanelId ?? '');
    expect(powerQueryPanel?.getAttribute('role')).toBe('tabpanel');
    expect(powerQueryPanel?.textContent).toContain('Web.Contents');
    expect(powerQueryPanel?.textContent).toContain('Authorization="Bearer rtws_secret_token"');
    expect(powerQueryPanel?.textContent).toContain(
      '/indexes/userspace/workspaces/workspace-9/external-api',
    );
  });

  it('supports arrow, home, and end keyboard tab selection while moving focus to the active tab', async () => {
    const user = userEvent.setup();
    render(
      <ExternalApiCredentialTokenDialog
        workspaceId="workspace-9"
        tokenState={{
          token: 'rtws_secret_token',
          prefix: 'rtws_abcd1234',
          label: 'August workpapers',
          operation: 'Rotated',
        }}
        onClose={() => undefined}
      />,
    );

    const curlTab = screen.getByRole('tab', { name: 'curl' });
    const powerQueryTab = screen.getByRole('tab', { name: 'Power Query' });
    await waitFor(() => {
      expect(screen.getByRole('button', { name: /copy token/i })).toBe(document.activeElement);
    });

    curlTab.focus();
    await user.keyboard('{ArrowRight}');
    expect(document.activeElement).toBe(powerQueryTab);
    expect(powerQueryTab.getAttribute('aria-selected')).toBe('true');

    await user.keyboard('{Home}');
    expect(document.activeElement).toBe(curlTab);
    expect(curlTab.getAttribute('aria-selected')).toBe('true');

    await user.keyboard('{End}');
    expect(document.activeElement).toBe(powerQueryTab);
    expect(powerQueryTab.getAttribute('aria-selected')).toBe('true');

    await user.keyboard('{ArrowLeft}');
    expect(document.activeElement).toBe(curlTab);
    expect(curlTab.getAttribute('aria-selected')).toBe('true');
  });

  it('supports confirm dialog dismissal, submission locks, and action-specific descriptions', async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    const onConfirm = vi.fn();
    function Harness() {
      const [open, setOpen] = useState(false);

      return (
        <div>
          <button type="button" onClick={() => setOpen(true)}>
            Rotate trigger
          </button>
          {open ? (
            <ExternalApiCredentialConfirmDialog
              action="rotate"
              credential={{
                id: 'cred-1',
                label: 'August workpapers',
                token_prefix: 'rtws_abcd1234',
                enabled: true,
                expires_at: null,
                last_used_at: null,
                request_count: 0,
                revoked_at: null,
                endpoint_keys: ['accounting-periods'],
              }}
              isSubmitting={false}
              onCancel={() => {
                onCancel();
                setOpen(false);
              }}
              onConfirm={onConfirm}
            />
          ) : null}
        </div>
      );
    }

    render(<Harness />);

    const trigger = screen.getByRole('button', { name: 'Rotate trigger' });
    await user.click(trigger);

    const dialog = document.body.querySelector('#userspace-external-api-confirm-dialog');
    expect(dialog).toBeTruthy();
    expect(screen.getByText('August workpapers')).toBeTruthy();
    expect(screen.getByText('rtws_abcd1234')).toBeTruthy();
    expect(
      screen.getByText(
        'The current token will stop working immediately. You will need to update every client with the replacement token.',
      ),
    ).toBeTruthy();
    const cancelButton = screen.getByRole('button', { name: 'Cancel' });
    const confirmButton = screen.getByRole('button', { name: 'Rotate token' });
    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });

    await user.tab();
    expect(document.activeElement).toBe(confirmButton);
    await user.tab();
    expect(document.activeElement).toBe(cancelButton);

    await user.keyboard('{Escape}');
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole('dialog', { name: /rotate credential/i })).toBeNull();
    expect(document.activeElement).toBe(trigger);

    cleanup();
    const submittingCancel = vi.fn();
    render(
      <ExternalApiCredentialConfirmDialog
        action="delete"
        credential={{
          id: 'cred-1',
          label: 'August workpapers',
          token_prefix: 'rtws_abcd1234',
          enabled: false,
          expires_at: null,
          last_used_at: null,
          request_count: 0,
          revoked_at: '2026-09-02T00:00:00Z',
          endpoint_keys: ['accounting-periods'],
        }}
        isSubmitting
        onCancel={submittingCancel}
        onConfirm={onConfirm}
      />,
    );

    expect(
      screen.getByText(
        'This permanently removes the revoked credential record. Request history and management audit history are preserved.',
      ),
    ).toBeTruthy();
    expect(
      screen.getByRole('button', { name: 'Delete permanently' }).hasAttribute('disabled'),
    ).toBe(true);
    expect(screen.getByRole('button', { name: 'Cancel' }).hasAttribute('disabled')).toBe(true);
    await user.keyboard('{Escape}');
    expect(submittingCancel).not.toHaveBeenCalled();
  });

  it('keeps focus inside a mounted confirm dialog while submission state changes and restores trigger focus after unmount', async () => {
    const trigger = document.createElement('button');
    trigger.type = 'button';
    trigger.textContent = 'Delete trigger';
    document.body.appendChild(trigger);
    trigger.focus();

    const { rerender, unmount } = render(
      <ExternalApiCredentialConfirmDialog
        action="delete"
        credential={{
          id: 'cred-1',
          label: 'August workpapers',
          token_prefix: 'rtws_abcd1234',
          enabled: false,
          expires_at: null,
          last_used_at: null,
          request_count: 0,
          revoked_at: '2026-09-02T00:00:00Z',
          endpoint_keys: ['accounting-periods'],
        }}
        isSubmitting={false}
        onCancel={() => undefined}
        onConfirm={() => undefined}
      />,
    );

    const dialog = screen.getByRole('dialog', { name: /delete credential/i });
    const cancelButton = screen.getByRole('button', { name: 'Cancel' });
    await waitFor(() => {
      expect(document.activeElement).toBe(cancelButton);
    });

    rerender(
      <ExternalApiCredentialConfirmDialog
        action="delete"
        credential={{
          id: 'cred-1',
          label: 'August workpapers',
          token_prefix: 'rtws_abcd1234',
          enabled: false,
          expires_at: null,
          last_used_at: null,
          request_count: 0,
          revoked_at: '2026-09-02T00:00:00Z',
          endpoint_keys: ['accounting-periods'],
        }}
        isSubmitting
        onCancel={() => undefined}
        onConfirm={() => undefined}
      />,
    );

    expect(screen.getByRole('dialog', { name: /delete credential/i })).toBe(dialog);
    expect(dialog.contains(document.activeElement)).toBe(true);
    expect(document.activeElement).not.toBe(trigger);

    unmount();
    expect(document.activeElement).toBe(trigger);
    trigger.remove();
  });
});
