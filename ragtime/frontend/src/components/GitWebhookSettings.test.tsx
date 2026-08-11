import type { ComponentProps } from 'react';

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GitWebhookConfig } from '@/types';

import { GitWebhookSettings } from './GitWebhookSettings';

const githubConfig: GitWebhookConfig = {
  enabled: true,
  paused: false,
  webhook_url: 'https://ragtime.example/webhooks/git/webhook-123',
  provider: 'github',
  branch: 'main',
  created_at: '2026-07-16T12:00:00Z',
};

const gitlabConfig: GitWebhookConfig = {
  enabled: true,
  paused: false,
  webhook_url: 'https://ragtime.example/webhooks/git/webhook-456',
  provider: 'gitlab',
  branch: 'release/2026.07',
  created_at: '2026-07-16T12:00:00Z',
};

const disabledConfig: GitWebhookConfig = {
  enabled: false,
  paused: false,
  webhook_url: null,
  provider: 'generic',
  branch: 'main',
  created_at: null,
};

let writeTextMock: ReturnType<typeof vi.fn>;

function renderComponent(overrides: Partial<ComponentProps<typeof GitWebhookSettings>> = {}) {
  return render(
    <GitWebhookSettings
      config={githubConfig}
      revealedSecret={null}
      disabled={false}
      onRotate={vi.fn()}
      onPause={vi.fn()}
      onResume={vi.fn()}
      {...overrides}
    />,
  );
}

describe('GitWebhookSettings', () => {
  beforeEach(() => {
    writeTextMock = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      get: () => ({
        writeText: writeTextMock,
      }),
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it('renders the one-time secret flat with an inline warning and share-link copy field', async () => {
    const user = userEvent.setup();
    renderComponent({
      config: githubConfig,
      revealedSecret: 'secret-once',
    });

    expect(screen.queryByText('Enabled')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Dismiss webhook secret' })).toBeNull();
    expect(
      screen.getByText(
        'Copy this now. This secret will not be shown again after you close this window.',
      ),
    ).toBeTruthy();

    const secretInput = screen.getByLabelText('One-time secret');
    expect(secretInput.getAttribute('type')).toBe('text');
    expect((secretInput as HTMLInputElement).readOnly).toBe(true);
    expect((secretInput as HTMLInputElement).value).toBe('secret-once');
    expect(secretInput.closest('.userspace-share-url-copy-wrap')).toBeTruthy();
    expect(screen.queryByText('Recent deliveries')).toBeNull();
    expect(screen.getByRole('button', { name: 'Pause webhook' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Disable webhook' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Copy webhook secret' }).className).toContain(
      'is-always-visible',
    );

    expect(screen.getByRole('alert')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Copy webhook secret' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Webhook secret copied' })).toBeTruthy();
    });
  });

  it('uses the share-link inline copy field for the selected webhook URL', () => {
    renderComponent({ revealedSecret: null });

    const urlInput = screen.getByLabelText('Selected webhook URL');
    expect(urlInput.getAttribute('type')).toBe('text');
    expect((urlInput as HTMLInputElement).readOnly).toBe(true);
    expect((urlInput as HTMLInputElement).value).toBe(githubConfig.webhook_url);
    expect(urlInput.closest('.userspace-share-url-copy-wrap')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy selected webhook URL' }).className).toContain(
      'userspace-share-inline-copy',
    );
  });

  it('keeps recent deliveries hidden for GitLab webhooks', () => {
    renderComponent({
      config: gitlabConfig,
      revealedSecret: null,
    });

    expect(screen.queryByText('Recent deliveries')).toBeNull();
  });

  it('defaults to the normal webhook URL and reveals the query-token URL only when selected', async () => {
    const user = userEvent.setup();
    renderComponent({
      revealedSecret: 'secret value/with?symbols',
    });

    expect((screen.getByLabelText('Selected webhook URL') as HTMLInputElement).value).toBe(
      githubConfig.webhook_url,
    );
    expect(screen.queryByText(/\?token=secret%20value/)).toBeNull();

    await user.click(screen.getByRole('radio', { name: 'URL with query token (less secure)' }));
    expect((screen.getByLabelText('Selected webhook URL') as HTMLInputElement).value).toBe(
      'https://ragtime.example/webhooks/git/webhook-123?token=secret%20value%2Fwith%3Fsymbols',
    );
  });

  it('removes the query-token choice when the one-time secret is dismissed', () => {
    const { rerender } = renderComponent({ revealedSecret: 'secret-once' });

    rerender(
      <GitWebhookSettings
        config={githubConfig}
        revealedSecret={null}
        disabled={false}
        onRotate={vi.fn()}
        onPause={vi.fn()}
        onResume={vi.fn()}
      />,
    );

    expect(screen.queryByRole('radio', { name: 'URL with query token (less secure)' })).toBeNull();
    expect((screen.getByLabelText('Selected webhook URL') as HTMLInputElement).value).toBe(
      githubConfig.webhook_url,
    );
  });

  it('does not repeat provider, branch, or creation metadata in enabled setup', () => {
    renderComponent({ revealedSecret: 'secret-once' });

    expect(screen.queryByText('Provider setup')).toBeNull();
    expect(screen.queryByText('Created')).toBeNull();
    expect(screen.queryByText('Provider')).toBeNull();
  });

  it('shows an active status badge when the webhook is running', () => {
    renderComponent({ config: githubConfig });

    expect(screen.getByText('Active')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Pause webhook' })).toBeTruthy();
  });

  it('shows a paused status badge with a resume action', () => {
    const pausedConfig: GitWebhookConfig = { ...githubConfig, paused: true };

    renderComponent({ config: pausedConfig });

    expect(screen.getByText('Paused')).toBeTruthy();
    expect(
      screen.queryByText('Webhook is paused. Pushes are ignored until you resume.'),
    ).toBeNull();
    expect(screen.getByRole('button', { name: 'Resume webhook' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Pause webhook' })).toBeNull();
  });

  it('disables action and copy controls when disabled', () => {
    renderComponent({
      config: githubConfig,
      revealedSecret: 'disabled-secret',
      disabled: true,
    });

    expect(
      (screen.getByRole('button', { name: 'Copy selected webhook URL' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Copy webhook secret' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Rotate secret' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Pause webhook' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('omits enable, secret, and query-token controls for disabled configs', () => {
    renderComponent({
      config: disabledConfig,
    });

    expect(screen.queryByText('Webhook delivery is disabled.')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Enable push webhook' })).toBeNull();
    expect(screen.queryByText(/secret unavailable/i)).toBeNull();
    expect(screen.queryByText(/query token/i)).toBeNull();
  });

  it('requires confirmation before rotating the secret and then invokes the callback', () => {
    vi.useFakeTimers();
    const onRotate = vi.fn();
    renderComponent({ onRotate });

    fireEvent.click(screen.getByRole('button', { name: 'Rotate secret' }));
    expect(screen.getByRole('button', { name: 'Confirm? (3)' })).toBeTruthy();
    expect(onRotate).not.toHaveBeenCalled();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByRole('button', { name: 'Confirm? (2)' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByRole('button', { name: 'Confirm? (1)' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByRole('button', { name: 'Confirm?' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm?' }));
    expect(onRotate).toHaveBeenCalledTimes(1);
  });

  it('requires confirmation before pausing the webhook and then invokes the callback', () => {
    vi.useFakeTimers();
    const onPause = vi.fn();
    renderComponent({ onPause });

    fireEvent.click(screen.getByRole('button', { name: 'Pause webhook' }));
    expect(screen.getByRole('button', { name: 'Confirm? (3)' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByRole('button', { name: 'Confirm? (2)' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByRole('button', { name: 'Confirm? (1)' })).toBeTruthy();

    act(() => {
      vi.advanceTimersByTime(1000);
    });
    expect(screen.getByRole('button', { name: 'Confirm?' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Confirm?' }));
    expect(onPause).toHaveBeenCalledTimes(1);
  });
});
