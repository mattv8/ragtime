import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ReceivedConversationShare } from '@/types';
import { ReceivedConversationShares } from './ReceivedConversationShares';

const useReceivedConversationSharesMock = vi.hoisted(() => vi.fn());

vi.mock('@/hooks/useReceivedConversationShares', () => ({
  useReceivedConversationShares: useReceivedConversationSharesMock,
}));

const share: ReceivedConversationShare = {
  id: 'share-1',
  conversation_id: 'conversation-1',
  title: 'Quarterly planning',
  owner_username: 'alice',
  owner_display_name: 'Alice Example',
  share_token: 'token/with spaces',
  label: 'Budget review',
  granted_role: 'editor',
  scope_anchor_message_idx: 12,
  scope_direction: 'forward',
  created_at: '2026-09-27T12:00:00Z',
};

function mockHook(overrides: Record<string, unknown> = {}) {
  const refresh = vi.fn().mockResolvedValue(undefined);
  useReceivedConversationSharesMock.mockReturnValue({
    shares: [share],
    loading: false,
    error: null,
    refresh,
    ...overrides,
  });
  return refresh;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ReceivedConversationShares', () => {
  it('renders scoped share metadata as a safe new-tab shared route', () => {
    mockHook();

    render(<ReceivedConversationShares currentUserId="recipient-1" searchQuery="" />);

    const link = screen.getByRole('link', { name: /quarterly planning.*opens in a new tab/i });
    expect(link.getAttribute('href')).toBe('/shared/token%2Fwith%20spaces');
    expect(link.getAttribute('target')).toBe('_blank');
    expect(link.getAttribute('rel')).toBe('noopener noreferrer');
    expect(link.getAttribute('data-share-id')).toBe('share-1');
    expect(screen.getByText('Budget review')).toBeDefined();
    expect(screen.getByText('Shared by Alice Example (@alice)')).toBeDefined();
    expect(screen.getByText('Editor · From message 13 onward')).toBeDefined();
    expect(screen.queryByRole('button', { name: /delete|rename|members/i })).toBeNull();
  });

  it('filters title, label, and either owner name and keeps a filtered-empty state', () => {
    mockHook();
    const { rerender } = render(
      <ReceivedConversationShares currentUserId="recipient-1" searchQuery="alice" />,
    );
    expect(screen.getByText('Quarterly planning')).toBeDefined();

    rerender(<ReceivedConversationShares currentUserId="recipient-1" searchQuery="budget" />);
    expect(screen.getByText('Quarterly planning')).toBeDefined();

    rerender(<ReceivedConversationShares currentUserId="recipient-1" searchQuery="missing" />);
    expect(screen.getByText('No shared chats match "missing".')).toBeDefined();
  });

  it('hides an empty idle section but shows local loading, error, and retry states', () => {
    const refresh = mockHook({ shares: [] });
    const { rerender } = render(
      <ReceivedConversationShares currentUserId="recipient-1" searchQuery="" />,
    );
    expect(screen.queryByRole('heading', { name: 'Shared with you' })).toBeNull();

    rerender(<ReceivedConversationShares currentUserId="recipient-1" searchQuery="needle" />);
    expect(screen.getByText('No shared chats match "needle".')).toBeDefined();

    mockHook({ shares: [], loading: true, refresh });
    rerender(<ReceivedConversationShares currentUserId="recipient-1" searchQuery="" />);
    expect(screen.getByLabelText('Loading shared chats')).toBeDefined();

    mockHook({ shares: [], error: new Error('Unable to load'), refresh });
    rerender(<ReceivedConversationShares currentUserId="recipient-1" searchQuery="" />);
    expect(screen.getByRole('alert').textContent).toContain('Unable to load');
    fireEvent.click(screen.getByRole('button', { name: 'Retry shared chats' }));
    expect(refresh).toHaveBeenCalledOnce();
  });
});
