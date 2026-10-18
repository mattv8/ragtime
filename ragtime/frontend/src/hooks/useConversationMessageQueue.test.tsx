import { render, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const { getQueue } = vi.hoisted(() => ({ getQueue: vi.fn() }));

vi.mock('@/api', () => ({
  ApiError: class ApiError extends Error {
    status = 500;
  },
  api: { getConversationMessageQueue: getQueue },
}));

import { useConversationMessageQueue } from './useConversationMessageQueue';

function QueueProbe({ conversationId }: { conversationId: string }) {
  const { state } = useConversationMessageQueue(conversationId);
  return <output>{state.items.map((item) => item.id).join(',')}</output>;
}

describe('useConversationMessageQueue', () => {
  it('does not apply a slower prior conversation snapshot after switching conversations', async () => {
    let resolveFirst: (value: { items: never[]; active_task_id: null; held: false }) => void;
    getQueue
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockResolvedValueOnce({ items: [{ id: 'new-item' }], active_task_id: null, held: false });
    const view = render(<QueueProbe conversationId="first" />);
    view.rerender(<QueueProbe conversationId="second" />);

    await waitFor(() => expect(view.container.textContent).toContain('new-item'));
    resolveFirst!({ items: [], active_task_id: null, held: false });

    await waitFor(() => expect(view.container.textContent).toContain('new-item'));
  });

  it('clears the previous conversation queue while the replacement snapshot is pending', async () => {
    let resolveSecond: (value: { items: never[]; active_task_id: null; held: false }) => void;
    getQueue
      .mockResolvedValueOnce({
        items: [{ id: 'first-item' }],
        active_task_id: 'first-task',
        held: false,
      })
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecond = resolve;
          }),
      );
    const view = render(<QueueProbe conversationId="first" />);
    await waitFor(() => expect(view.container.textContent).toContain('first-item'));

    view.rerender(<QueueProbe conversationId="second" />);

    expect(view.container.textContent).not.toContain('first-item');
    resolveSecond!({ items: [], active_task_id: null, held: false });
  });
});
