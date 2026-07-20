import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ReindexIntervalSelect } from './ReindexIntervalSelect';

afterEach(() => {
  cleanup();
});

describe('ReindexIntervalSelect', () => {
  it('enables webhook delivery and does not expose the schedule while active', async () => {
    const user = userEvent.setup();
    const onWebhookDeliveryChange = vi.fn<(enabled: boolean) => Promise<boolean>>(
      async (_enabled: boolean) => true,
    );

    function Harness() {
      const [webhookDeliveryEnabled, setWebhookDeliveryEnabled] = useState(false);
      return (
        <ReindexIntervalSelect
          value={24}
          onChange={vi.fn()}
          onWebhookDeliveryChange={async (enabled) => {
            const result = await onWebhookDeliveryChange(enabled);
            if (result) setWebhookDeliveryEnabled(enabled);
            return result;
          }}
          webhookDeliveryEnabled={webhookDeliveryEnabled}
          startMinute={60}
          timezone="UTC"
          onStartMinuteChange={vi.fn()}
          onTimezoneChange={vi.fn()}
        />
      );
    }

    render(<Harness />);

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');

    expect(onWebhookDeliveryChange).toHaveBeenCalledWith(true);
    expect(screen.queryByText('Start Time')).toBeNull();
  });

  it('requires confirmation before leaving webhook delivery and preserves it on cancellation', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onWebhookDeliveryChange = vi.fn(async () => true);

    render(
      <ReindexIntervalSelect
        value={24}
        onChange={onChange}
        onWebhookDeliveryChange={onWebhookDeliveryChange}
        webhookDeliveryEnabled
      />,
    );

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), '24');
    expect(screen.getByRole('dialog', { name: 'Disable webhook delivery?' })).toBeTruthy();
    expect(
      screen.getByText('Switching away from webhook delivery will re-enable scheduled updates.'),
    ).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Cancel' }));

    expect(onWebhookDeliveryChange).not.toHaveBeenCalled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it('keeps webhook delivery selected when disabling it fails', async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    const onWebhookDeliveryChange = vi.fn(async () => false);

    render(
      <ReindexIntervalSelect
        value={24}
        onChange={onChange}
        onWebhookDeliveryChange={onWebhookDeliveryChange}
        webhookDeliveryEnabled
      />,
    );

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), '24');
    await user.click(screen.getByRole('button', { name: 'Disable webhook and continue' }));

    expect(onWebhookDeliveryChange).toHaveBeenCalledWith(false);
    expect(onChange).not.toHaveBeenCalled();
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe(
      'webhook',
    );
  });

  it('re-enables the confirmation action when disabling webhook delivery rejects', async () => {
    const user = userEvent.setup();
    const onWebhookDeliveryChange = vi.fn<(enabled: boolean) => Promise<boolean>>(
      async (_enabled: boolean) => {
        throw new Error('disable failed');
      },
    );

    render(
      <ReindexIntervalSelect
        value={24}
        onChange={vi.fn()}
        onWebhookDeliveryChange={onWebhookDeliveryChange}
        webhookDeliveryEnabled
      />,
    );

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), '24');

    const confirmButton = screen.getByRole('button', { name: 'Disable webhook and continue' });
    await user.click(confirmButton);

    await waitFor(() => {
      expect((confirmButton as HTMLButtonElement).disabled).toBe(false);
    });
  });

  it('renders a cadence action beside the selector only when provided', () => {
    const { rerender } = render(
      <ReindexIntervalSelect
        value={0}
        onChange={vi.fn()}
        action={<button type="button">Pull now</button>}
      />,
    );

    expect(screen.getByRole('button', { name: 'Pull now' })).toBeTruthy();

    rerender(<ReindexIntervalSelect value={0} onChange={vi.fn()} />);

    expect(screen.queryByRole('button', { name: 'Pull now' })).toBeNull();
  });
});
