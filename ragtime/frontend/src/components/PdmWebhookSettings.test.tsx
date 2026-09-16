import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
  it('gates setup until the PDM tool is saved', () => {
    render(<PdmWebhookSettings toolId={null} />);
    expect(screen.getByText('Save this PDM tool before enabling webhook delivery.')).toBeTruthy();
    expect(apiMock.getPdmWebhook).not.toHaveBeenCalled();
  });
  it('enables a webhook, reveals its secret once, and clears it on dismiss', async () => {
    const user = userEvent.setup();
    render(<PdmWebhookSettings toolId="pdm-1" />);
    await screen.findByRole('button', { name: 'Enable webhook' });
    await user.click(screen.getByRole('button', { name: 'Enable webhook' }));
    await waitFor(() => expect(apiMock.enablePdmWebhook).toHaveBeenCalledWith('pdm-1'));
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
    render(<PdmWebhookSettings toolId="pdm-1" />);
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
    render(<PdmWebhookSettings toolId="pdm-1" />);
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
    const { rerender } = render(<PdmWebhookSettings toolId="pdm-a" />);
    await screen.findByRole('button', { name: 'Enable webhook' });
    await user.click(screen.getByRole('button', { name: 'Enable webhook' }));
    rerender(<PdmWebhookSettings toolId="pdm-b" />);
    await waitFor(() => expect(apiMock.getPdmWebhook).toHaveBeenCalledWith('pdm-b'));
    enableRequest.resolve({
      ...disabledConfig,
      enabled: true,
      webhook_id: 'pdm-a',
      secret: 'stale-secret',
    });
    await waitFor(() => expect(screen.queryByLabelText('PDM one-time secret')).toBeNull());
    expect(screen.getByRole('button', { name: 'Enable webhook' })).toBeTruthy();
  });
});
