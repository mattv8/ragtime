import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ChatLoadingState } from './ChatLoadingState';

describe('ChatLoadingState', () => {
  afterEach(cleanup);

  it('exposes a busy loading region with hidden skeleton decoration', () => {
    render(<ChatLoadingState kind="main" id="conversation-loading" label="Loading messages" />);

    const region = document.getElementById('conversation-loading');
    expect(region?.getAttribute('data-chat-loading-kind')).toBe('main');
    expect(region?.getAttribute('aria-busy')).toBe('true');
    expect(screen.getByRole('status').textContent).toBe('Loading messages');
    expect(region?.querySelector('[aria-hidden="true"]')).not.toBeNull();
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
  });

  it('clears busy state and invokes retry when an error is retried', async () => {
    const user = userEvent.setup();
    const onRetry = vi.fn();
    const { rerender } = render(<ChatLoadingState kind="history" label="Loading history" />);

    rerender(
      <ChatLoadingState
        kind="history"
        state="error"
        label="History could not load"
        onRetry={onRetry}
      />,
    );

    const region = document.querySelector('[data-chat-loading-kind="history"]');
    expect(region?.tagName).toBe('DIV');
    expect(region?.hasAttribute('aria-busy')).toBe(false);
    expect(screen.getByRole('status').textContent).toBe('History could not load');

    await user.click(screen.getByRole('button', { name: 'Retry' }));
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
