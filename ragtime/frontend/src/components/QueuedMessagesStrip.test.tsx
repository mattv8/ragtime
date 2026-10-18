import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { QueuedMessagesStrip } from './QueuedMessagesStrip';

const item = {
  id: 'queue-1',
  conversation_id: 'conversation-1',
  message: 'First queued message',
  segments: [{ type: 'text' as const, text: 'First queued message' }],
  request_context: {},
  seq: '1',
  dispatch_error: null,
  created_at: '2026-10-18T00:00:00Z',
};

describe('QueuedMessagesStrip', () => {
  afterEach(cleanup);
  it('expands multiple queued messages and removes the selected item', async () => {
    const onDelete = vi.fn().mockResolvedValue(undefined);
    render(
      <QueuedMessagesStrip
        conversationId="conversation-1"
        state={{
          items: [item, { ...item, id: 'queue-2', seq: '2' }],
          activeTaskId: 'task-1',
          held: false,
          loading: false,
          error: null,
        }}
        draftIsEmpty
        onDelete={onDelete}
        onEdit={vi.fn()}
        onDispatch={vi.fn()}
        onRetry={vi.fn()}
      />,
    );

    await userEvent.click(screen.getByRole('button', { name: 'Expand queued messages' }));
    await userEvent.click(screen.getAllByRole('button', { name: 'Remove queued message' })[1]);

    expect(onDelete).toHaveBeenCalledWith('queue-2');
  });

  it('announces a single queued item and explains why its edit action is disabled', () => {
    render(
      <QueuedMessagesStrip
        conversationId="conversation-1"
        state={{ items: [item], activeTaskId: 'task-1', held: false, loading: false, error: null }}
        draftIsEmpty={false}
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onDispatch={vi.fn()}
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByRole('status', { name: '1 message queued' })).toBeDefined();
    expect(screen.getByRole('button', { name: 'Edit queued message' })).toHaveProperty(
      'disabled',
      true,
    );
    expect(document.getElementById('queue-edit-disabled-reason-queue-1')?.textContent).toContain(
      'Clear your draft first to edit a queued message',
    );
  });

  it('renders malformed segments without crashing and announces status outside row actions', () => {
    render(
      <QueuedMessagesStrip
        conversationId="conversation-1"
        state={{
          items: [{ ...item, segments: [{ type: 'ref' } as never] }],
          activeTaskId: null,
          held: true,
          loading: false,
          error: null,
        }}
        draftIsEmpty
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onDispatch={vi.fn()}
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByRole('status').textContent).toContain('1 message queued');
    expect(screen.getByRole('button', { name: 'Remove queued message' })).toBeDefined();
  });

  it('uses canonical message text when display segments were omitted', () => {
    render(
      <QueuedMessagesStrip
        conversationId="conversation-1"
        state={{
          items: [{ ...item, message: 'Large pasted message', segments: [] }],
          activeTaskId: null,
          held: true,
          loading: false,
          error: null,
        }}
        draftIsEmpty
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onDispatch={vi.fn()}
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByText('Large pasted message')).toBeDefined();
  });
});
