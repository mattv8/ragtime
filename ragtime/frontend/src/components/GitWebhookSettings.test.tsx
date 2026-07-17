import type { ComponentProps } from 'react';

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GitWebhookConfig, GitWebhookDelivery } from '@/types';

import { GitWebhookSettings } from './GitWebhookSettings';

const githubConfig: GitWebhookConfig = {
  enabled: true,
  webhook_url: 'https://ragtime.example/webhooks/git/webhook-123',
  provider: 'github',
  branch: 'main',
  created_at: '2026-07-16T12:00:00Z',
};

const gitlabConfig: GitWebhookConfig = {
  enabled: true,
  webhook_url: 'https://ragtime.example/webhooks/git/webhook-456',
  provider: 'gitlab',
  branch: 'release/2026.07',
  created_at: '2026-07-16T12:00:00Z',
};

const genericConfig: GitWebhookConfig = {
  enabled: true,
  webhook_url: 'https://ragtime.example/webhooks/git/webhook-789',
  provider: 'generic',
  branch: 'develop',
  created_at: '2026-07-16T12:00:00Z',
};

const disabledConfig: GitWebhookConfig = {
  enabled: false,
  webhook_url: null,
  provider: 'generic',
  branch: 'main',
  created_at: null,
};

const skippedDelivery: GitWebhookDelivery = {
  id: 'delivery-1',
  event_name: 'Push Hook',
  branch: 'release/2026.07',
  head_commit: 'abc123def456',
  status: 'skipped',
  message: 'A newer push arrived while another sync was already queued.',
  received_at: '2026-07-16T12:10:00Z',
  started_at: null,
  completed_at: '2026-07-16T12:10:05Z',
};

const completedDelivery: GitWebhookDelivery = {
  id: 'delivery-2',
  event_name: 'push',
  branch: 'main',
  head_commit: 'fedcba987654',
  status: 'completed',
  message: 'Started a sync job successfully.',
  received_at: '2026-07-16T12:12:00Z',
  started_at: '2026-07-16T12:12:02Z',
  completed_at: '2026-07-16T12:12:30Z',
};

let writeTextMock: ReturnType<typeof vi.fn>;

function renderComponent(overrides: Partial<ComponentProps<typeof GitWebhookSettings>> = {}) {
  return render(
    <GitWebhookSettings
      config={githubConfig}
      revealedSecret={null}
      deliveries={[]}
      disabled={false}
      onEnable={vi.fn()}
      onRotate={vi.fn()}
      onDisable={vi.fn()}
      onDismissSecret={vi.fn()}
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

  it('shows and copies a one-time secret without retaining it after dismissal', async () => {
    const user = userEvent.setup();
    const onDismissSecret = vi.fn();
    const { rerender } = renderComponent({
      config: githubConfig,
      revealedSecret: 'secret-once',
      onDismissSecret,
    });

    expect(screen.getByText('secret-once')).toBeTruthy();
    expect(screen.getByRole('alert')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Copy webhook secret' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Webhook secret copied' })).toBeTruthy();
    });

    await user.click(screen.getByRole('button', { name: 'Dismiss webhook secret' }));
    expect(onDismissSecret).toHaveBeenCalledTimes(1);

    rerender(
      <GitWebhookSettings
        config={githubConfig}
        revealedSecret={null}
        deliveries={[]}
        disabled={false}
        onEnable={vi.fn()}
        onRotate={vi.fn()}
        onDisable={vi.fn()}
        onDismissSecret={onDismissSecret}
      />,
    );

    expect(screen.queryByText('secret-once')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Copy webhook secret' })).toBeNull();
  });

  it('renders skipped deliveries and GitLab setup instructions', () => {
    renderComponent({
      config: gitlabConfig,
      revealedSecret: null,
      deliveries: [skippedDelivery],
    });

    expect(screen.getByText('Use the Secret Token field.')).toBeTruthy();
    expect(screen.getByText('Skipped')).toBeTruthy();
    expect(screen.getAllByText('release/2026.07').length).toBeGreaterThan(0);
    expect(
      screen.getByText('A newer push arrived while another sync was already queued.'),
    ).toBeTruthy();
  });

  it('copies the webhook URL and query-token fallback URL with encoding', async () => {
    const user = userEvent.setup();
    renderComponent({
      config: genericConfig,
      revealedSecret: 'secret value/with?symbols',
    });

    await user.click(screen.getByRole('button', { name: 'Copy webhook URL' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Webhook URL copied' })).toBeTruthy();
    });

    await user.click(screen.getByRole('button', { name: 'Copy query token fallback URL' }));
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Fallback URL copied' })).toBeTruthy();
    });
    expect(
      screen.getByText(
        'https://ragtime.example/webhooks/git/webhook-789?token=secret%20value%2Fwith%3Fsymbols',
      ),
    ).toBeTruthy();
    expect(screen.getByText('Fallback URL with query token')).toBeTruthy();
    expect(
      screen.getByText('This is less secure because URLs may be captured in provider logs.'),
    ).toBeTruthy();
  });

  it('renders GitHub HMAC instructions, branch details, and recent delivery history', () => {
    renderComponent({
      config: githubConfig,
      deliveries: [completedDelivery],
    });

    expect(
      screen.getByText('Configure the secret so GitHub sends X-Hub-Signature-256 HMAC signatures.'),
    ).toBeTruthy();
    expect(screen.getAllByText('Branch').length).toBe(2);
    expect(screen.getAllByText('main').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Completed').length).toBeGreaterThan(0);
    expect(screen.getByText('Started a sync job successfully.')).toBeTruthy();
    expect(screen.getByText('fedcba987654')).toBeTruthy();
  });

  it('renders generic provider instructions without unsupported-provider wording', () => {
    renderComponent({
      config: genericConfig,
      revealedSecret: 'generic-secret',
    });

    expect(
      screen.getByText('Use a token header when your provider supports custom headers.'),
    ).toBeTruthy();
    expect(
      screen.getByText(
        'Accepted options include X-Ragtime-Webhook-Token, Authorization: Bearer, GitHub-style SHA-256 signatures, GitLab Secret Token headers, and compatible Gitea or Gogs SHA-256 headers.',
      ),
    ).toBeTruthy();
    expect(screen.queryByText(/GitLab only/i)).toBeNull();
  });

  it('disables action and copy controls when disabled', () => {
    renderComponent({
      config: githubConfig,
      revealedSecret: 'disabled-secret',
      disabled: true,
    });

    expect(
      (screen.getByRole('button', { name: 'Copy webhook URL' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Copy webhook secret' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Dismiss webhook secret' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Rotate secret' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Disable webhook' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('shows the enable action for disabled configs and no plaintext placeholder when the secret is unavailable', () => {
    const onEnable = vi.fn();
    renderComponent({
      config: disabledConfig,
      onEnable,
    });

    expect(screen.getByText('Webhook delivery is disabled.')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Enable push webhook' })).toBeTruthy();
    expect(screen.queryByText(/secret unavailable/i)).toBeNull();
    expect(screen.queryByText(/query token/i)).toBeNull();
  });

  it('invokes the enable callback when enabling a disabled webhook', () => {
    const onEnable = vi.fn();
    renderComponent({
      config: disabledConfig,
      onEnable,
    });

    fireEvent.click(screen.getByRole('button', { name: 'Enable push webhook' }));

    expect(onEnable).toHaveBeenCalledTimes(1);
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

  it('requires confirmation before disabling the webhook and then invokes the callback', () => {
    vi.useFakeTimers();
    const onDisable = vi.fn();
    renderComponent({ onDisable });

    fireEvent.click(screen.getByRole('button', { name: 'Disable webhook' }));
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
    expect(onDisable).toHaveBeenCalledTimes(1);
  });

  it('renders a single Branch label in the summary section', () => {
    renderComponent({
      config: githubConfig,
      deliveries: [completedDelivery],
    });

    expect(screen.getAllByText('Branch').length).toBe(2);
    expect(screen.queryByText('Branch:')).toBeNull();
  });
});
