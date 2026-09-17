import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PdmWebhookConfig, PdmWebhookEnableResponse } from '@/types';
import { PdmWebhookSettings } from './PdmWebhookSettings';

const apiMock = vi.hoisted(() => ({
  getPdmWebhook: vi.fn(),
  enablePdmWebhook: vi.fn(),
  rotatePdmWebhookSecret: vi.fn(),
  pausePdmWebhook: vi.fn(),
  resumePdmWebhook: vi.fn(),
  disablePdmWebhook: vi.fn(),
}));
vi.mock('@/api', () => ({ api: apiMock }));
const disabledConfig: PdmWebhookConfig = {
  enabled: false,
  paused: false,
  webhook_id: null,
  webhook_url: null,
  created_at: null,
  last_received_at: null,
  pending: false,
  active_job_id: null,
  last_attempt_at: null,
  last_success_at: null,
  last_error: null,
};
function createDeferredPromise<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function renderSettings(overrides: Partial<ComponentProps<typeof PdmWebhookSettings>> = {}) {
  return render(
    <PdmWebhookSettings
      toolId="pdm-1"
      intervalHours={0}
      startMinute={null}
      timezone={null}
      onIntervalChange={vi.fn()}
      onStartMinuteChange={vi.fn()}
      onTimezoneChange={vi.fn()}
      webhookDeliveryRequested={false}
      onWebhookDeliveryRequestedChange={vi.fn()}
      {...overrides}
    />,
  );
}

describe('PdmWebhookSettings', () => {
  beforeEach(() => {
    apiMock.getPdmWebhook.mockResolvedValue(disabledConfig);
    apiMock.enablePdmWebhook.mockResolvedValue({
      ...disabledConfig,
      enabled: true,
      webhook_id: 'webhook-1',
      webhook_url: 'https://ragtime.example/webhooks/pdm/webhook-1',
      secret: 'one-time-secret',
    });
  });
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });
  it('shows webhook delivery in the cadence dropdown and hides metadata in manual mode', async () => {
    renderSettings({ toolId: null });
    expect(screen.getByRole('option', { name: 'Webhook delivery' })).toBeTruthy();
    expect(screen.queryByTestId('pdm-webhook-status')).toBeNull();
    expect(screen.queryByText('Save this PDM tool before enabling webhook delivery.')).toBeNull();
    expect(apiMock.getPdmWebhook).not.toHaveBeenCalled();
  });
  it('defers webhook delivery for a new PDM tool without making an API request', async () => {
    const user = userEvent.setup();
    const onWebhookDeliveryRequestedChange = vi.fn();
    const onIntervalChange = vi.fn();
    const onStartMinuteChange = vi.fn();
    const onTimezoneChange = vi.fn();
    const { rerender } = renderSettings({
      toolId: null,
      onWebhookDeliveryRequestedChange,
      onIntervalChange,
      onStartMinuteChange,
      onTimezoneChange,
    });
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    expect(onWebhookDeliveryRequestedChange).toHaveBeenCalledWith(true);
    expect(onIntervalChange).toHaveBeenCalledWith(0);
    expect(onStartMinuteChange).toHaveBeenCalledWith(null);
    expect(onTimezoneChange).toHaveBeenCalledWith(null);
    rerender(
      <PdmWebhookSettings
        toolId={null}
        intervalHours={0}
        startMinute={null}
        timezone={null}
        onIntervalChange={onIntervalChange}
        onStartMinuteChange={onStartMinuteChange}
        onTimezoneChange={onTimezoneChange}
        webhookDeliveryRequested
        onWebhookDeliveryRequestedChange={onWebhookDeliveryRequestedChange}
      />,
    );
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe(
      'webhook',
    );
    expect(
      screen.getByText('Webhook setup will be generated after this PDM tool is created.'),
    ).toBeTruthy();
    expect(screen.queryByTestId('pdm-webhook-status')).toBeNull();
    expect(apiMock.getPdmWebhook).not.toHaveBeenCalled();
  });
  it('enables a webhook from the cadence selector, reveals its secret once, and clears the schedule after success', async () => {
    const user = userEvent.setup();
    const onIntervalChange = vi.fn();
    const onStartMinuteChange = vi.fn();
    const onTimezoneChange = vi.fn();
    renderSettings({ onIntervalChange, onStartMinuteChange, onTimezoneChange });
    await screen.findByLabelText('Auto Re-index Interval');
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    await waitFor(() => expect(apiMock.enablePdmWebhook).toHaveBeenCalledWith('pdm-1'));
    expect(onIntervalChange).toHaveBeenCalledWith(0);
    expect(onStartMinuteChange).toHaveBeenCalledWith(null);
    expect(onTimezoneChange).toHaveBeenCalledWith(null);
    expect((screen.getByLabelText('PDM one-time secret') as HTMLInputElement).value).toBe(
      'one-time-secret',
    );
    expect((screen.getByLabelText('PDM webhook URL') as HTMLInputElement).value).not.toContain('?');
    await user.click(screen.getByRole('button', { name: 'Dismiss secret' }));
    expect(screen.queryByLabelText('PDM one-time secret')).toBeNull();
  });
  it('shows queued and failed webhook state from nullable response fields', async () => {
    apiMock.getPdmWebhook.mockResolvedValue({
      ...disabledConfig,
      enabled: true,
      pending: true,
      last_received_at: '2026-09-16T12:00:00Z',
      last_error: 'Indexing unavailable',
    });
    renderSettings();
    await screen.findByTestId('pdm-webhook-status');
    expect(screen.getByText('Pending')).toBeTruthy();
    expect(screen.getByText('Indexing unavailable')).toBeTruthy();
    expect(screen.getByText('Never')).toBeTruthy();
  });
  it('updates the visible status after pause and resume actions', async () => {
    const user = userEvent.setup();
    apiMock.getPdmWebhook.mockResolvedValue({
      ...disabledConfig,
      enabled: true,
      webhook_id: 'webhook-1',
    });
    apiMock.pausePdmWebhook.mockResolvedValue({
      ...disabledConfig,
      enabled: true,
      paused: true,
      webhook_id: 'webhook-1',
    });
    apiMock.resumePdmWebhook.mockResolvedValue({
      ...disabledConfig,
      enabled: true,
      webhook_id: 'webhook-1',
    });
    renderSettings();
    await screen.findByRole('button', { name: 'Pause webhook' });
    await user.click(screen.getByRole('button', { name: 'Pause webhook' }));
    await waitFor(() => expect(apiMock.pausePdmWebhook).toHaveBeenCalledWith('pdm-1'));
    expect(screen.getByText('Paused')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Resume webhook' }));
    await waitFor(() => expect(apiMock.resumePdmWebhook).toHaveBeenCalledWith('pdm-1'));
    expect(screen.getByText('Active')).toBeTruthy();
  });
  it('does not reveal a stale secret after changing tools during an enable request', async () => {
    const user = userEvent.setup();
    const enableRequest = createDeferredPromise<PdmWebhookEnableResponse>();
    apiMock.getPdmWebhook.mockImplementation((toolId: string) =>
      Promise.resolve({ ...disabledConfig, webhook_id: toolId }),
    );
    apiMock.enablePdmWebhook.mockReturnValue(enableRequest.promise);
    const { rerender } = renderSettings({ toolId: 'pdm-a' });
    await screen.findByLabelText('Auto Re-index Interval');
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    rerender(
      <PdmWebhookSettings
        toolId="pdm-b"
        intervalHours={0}
        startMinute={null}
        timezone={null}
        onIntervalChange={vi.fn()}
        onStartMinuteChange={vi.fn()}
        onTimezoneChange={vi.fn()}
        webhookDeliveryRequested={false}
        onWebhookDeliveryRequestedChange={vi.fn()}
      />,
    );
    await waitFor(() => expect(apiMock.getPdmWebhook).toHaveBeenCalledWith('pdm-b'));
    enableRequest.resolve({
      ...disabledConfig,
      enabled: true,
      webhook_id: 'pdm-a',
      secret: 'stale-secret',
    });
    await waitFor(() => expect(screen.queryByLabelText('PDM one-time secret')).toBeNull());
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe('0');
  });
  it('hydrates an activation result and renders its one-time secret', () => {
    renderSettings({
      toolId: null,
      webhookDeliveryRequested: true,
      activationResult: {
        ...disabledConfig,
        enabled: true,
        webhook_id: 'webhook-1',
        webhook_url: 'https://ragtime.example/webhooks/pdm/webhook-1',
        secret: 'activation-secret',
      },
    });
    expect((screen.getByLabelText('PDM one-time secret') as HTMLInputElement).value).toBe(
      'activation-secret',
    );
    expect(screen.getByTestId('pdm-webhook-status')).toBeTruthy();
  });
  it('normalizes a legacy enabled webhook with a scheduled interval', async () => {
    const onIntervalChange = vi.fn();
    const onStartMinuteChange = vi.fn();
    const onTimezoneChange = vi.fn();
    apiMock.getPdmWebhook.mockResolvedValue({ ...disabledConfig, enabled: true });
    renderSettings({
      intervalHours: 24,
      startMinute: 60,
      timezone: 'UTC',
      onIntervalChange,
      onStartMinuteChange,
      onTimezoneChange,
    });
    await screen.findByTestId('pdm-webhook-status');
    expect(onIntervalChange).toHaveBeenCalledWith(0);
    expect(onStartMinuteChange).toHaveBeenCalledWith(null);
    expect(onTimezoneChange).toHaveBeenCalledWith(null);
  });
  it('disables the cadence selector while enabling a webhook', async () => {
    const user = userEvent.setup();
    const enableRequest = createDeferredPromise<PdmWebhookEnableResponse>();
    apiMock.enablePdmWebhook.mockReturnValue(enableRequest.promise);
    renderSettings();
    await screen.findByLabelText('Auto Re-index Interval');
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).disabled).toBe(
      true,
    );
    enableRequest.resolve({ ...disabledConfig, enabled: true, secret: 'one-time-secret' });
  });
  it('applies a scheduled cadence only after successfully disabling webhook delivery', async () => {
    const user = userEvent.setup();
    const disableRequest = createDeferredPromise<PdmWebhookConfig>();
    const onIntervalChange = vi.fn();
    const onStartMinuteChange = vi.fn();
    const onTimezoneChange = vi.fn();
    apiMock.getPdmWebhook.mockResolvedValue({ ...disabledConfig, enabled: true });
    apiMock.disablePdmWebhook.mockReturnValue(disableRequest.promise);
    renderSettings({ onIntervalChange, onStartMinuteChange, onTimezoneChange });

    await screen.findByTestId('pdm-webhook-status');
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), '24');
    await user.click(screen.getByRole('button', { name: 'Disable webhook and continue' }));

    expect(apiMock.disablePdmWebhook).toHaveBeenCalledWith('pdm-1');
    expect(onIntervalChange).not.toHaveBeenCalled();
    expect(onStartMinuteChange).not.toHaveBeenCalled();
    expect(onTimezoneChange).not.toHaveBeenCalled();

    disableRequest.resolve(disabledConfig);

    await waitFor(() => expect(onIntervalChange).toHaveBeenCalledWith(24));
    expect(onStartMinuteChange).toHaveBeenCalledTimes(1);
    expect(onTimezoneChange).toHaveBeenCalledTimes(1);
  });
  it('keeps webhook delivery selected when disabling it fails', async () => {
    const user = userEvent.setup();
    const onIntervalChange = vi.fn();
    apiMock.getPdmWebhook.mockResolvedValue({ ...disabledConfig, enabled: true });
    apiMock.disablePdmWebhook.mockRejectedValue(new Error('Disable failed'));
    renderSettings({ onIntervalChange });
    await screen.findByTestId('pdm-webhook-status');
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), '0');
    await user.click(screen.getByRole('button', { name: 'Disable webhook and continue' }));
    await screen.findByRole('alert');
    expect(apiMock.disablePdmWebhook).toHaveBeenCalledWith('pdm-1');
    expect(onIntervalChange).not.toHaveBeenCalled();
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe(
      'webhook',
    );
    expect(screen.getByRole('button', { name: 'Disable webhook and continue' })).toBeTruthy();
  });
});
