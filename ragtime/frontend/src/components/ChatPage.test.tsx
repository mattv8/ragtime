import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect } from 'react';
import type { ConversationShareLinkStatus } from '@/types/api';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ChatPage } from './ChatPage';

const sseMock = vi.hoisted(() => {
  type Listener = (event: MessageEvent<string>) => void;
  const sources: Array<{
    addEventListener: ReturnType<typeof vi.fn>;
    removeEventListener: ReturnType<typeof vi.fn>;
    close: ReturnType<typeof vi.fn>;
    emit: (type: string, data: unknown) => void;
  }> = [];

  const createSource = () => {
    const listeners = new Map<string, Set<Listener>>();
    const source = {
      addEventListener: vi.fn((type: string, listener: Listener) => {
        const bucket = listeners.get(type) ?? new Set<Listener>();
        bucket.add(listener);
        listeners.set(type, bucket);
      }),
      removeEventListener: vi.fn((type: string, listener: Listener) => {
        listeners.get(type)?.delete(listener);
      }),
      close: vi.fn(),
      emit: (type: string, data: unknown) => {
        const event = { data: JSON.stringify(data) } as MessageEvent<string>;
        listeners.get(type)?.forEach((listener) => listener(event));
      },
    };
    sources.push(source);
    return source;
  };

  return { createSource, sources };
});

const apiMock = vi.hoisted(() => ({
  listConversationShareLinks: vi.fn(),
  listUsersDirectory: vi.fn().mockResolvedValue([]),
  createConversationShareLink: vi.fn(),
  deleteConversationShareLink: vi.fn(),
  subscribeConversationShareLinkAnalytics: vi.fn(() => sseMock.createSource()),
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('./ChatPanel', () => ({
  ChatPanel: ({
    onActiveConversationChange,
    onOpenShareModal,
  }: {
    onActiveConversationChange?: (conversationId: string | null) => void;
    onOpenShareModal?: () => void;
  }) => {
    useEffect(() => {
      onActiveConversationChange?.('conv-1');
    }, [onActiveConversationChange]);

    return (
      <button type="button" onClick={() => onOpenShareModal?.()}>
        Open share modal
      </button>
    );
  },
}));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  sseMock.sources.length = 0;
});

const makeShareLinkStatus = (
  overrides: Partial<ConversationShareLinkStatus> = {},
): ConversationShareLinkStatus => ({
  id: 'share-1',
  conversation_id: 'conv-1',
  has_share_link: true,
  owner_username: 'owner',
  label: 'Alpha link',
  share_slug: 'alpha-link',
  share_token: 'token-1',
  share_url: 'https://example.com/owner/alpha-link',
  anonymous_share_url: 'https://example.com/shared/token-1',
  created_at: '2026-07-14T00:00:00Z',
  share_access_mode: 'token',
  selected_user_ids: [],
  selected_ldap_groups: [],
  has_password: false,
  granted_role: 'viewer',
  scope_anchor_message_idx: null,
  scope_direction: null,
  active_share_style: 'named',
  public_hit_count: 0,
  last_public_hit_at: null,
  ...overrides,
});

const mockConversationShareLinks = (links: ConversationShareLinkStatus[]): void => {
  apiMock.listConversationShareLinks.mockResolvedValue({
    conversation_id: 'conv-1',
    owner_username: 'owner',
    links,
  });
};

const renderAuthenticatedChatPage = () =>
  render(
    <ChatPage
      currentUser={{
        id: 'user-1',
        username: 'local:admin',
        display_name: 'Admin',
        email: null,
        role: 'admin',
        auth_provider: 'local',
      }}
    />,
  );

describe('ChatPage share link analytics', () => {
  it('removes a deleted share link without reloading the list', async () => {
    const user = userEvent.setup();
    mockConversationShareLinks([
      makeShareLinkStatus(),
      makeShareLinkStatus({
        id: 'share-2',
        label: 'Beta link',
        share_slug: 'beta-link',
        share_token: 'token-2',
        share_url: 'https://example.com/owner/beta-link',
        anonymous_share_url: 'https://example.com/shared/token-2',
        created_at: '2026-07-14T00:05:00Z',
      }),
    ]);
    apiMock.deleteConversationShareLink.mockResolvedValue(undefined);

    renderAuthenticatedChatPage();

    await user.click(await screen.findByRole('button', { name: 'Open share modal' }));

    expect(await screen.findByText('Alpha link')).toBeDefined();
    expect(screen.getByText('Beta link')).toBeDefined();

    await user.click(screen.getAllByRole('button', { name: 'Delete link' })[0]);
    await user.click(await screen.findByRole('button', { name: 'Confirm delete' }));

    await waitFor(() => {
      expect(screen.queryByText('Alpha link')).toBeNull();
      expect(screen.getByText('Beta link')).toBeDefined();
    });

    expect(apiMock.deleteConversationShareLink).toHaveBeenCalledWith('conv-1', 'share-1');
    expect(apiMock.listConversationShareLinks).toHaveBeenCalledTimes(1);
  });

  it('restores the original share links and shows an error when delete rolls back', async () => {
    const user = userEvent.setup();
    let rejectDelete: (reason?: unknown) => void = () => {
      throw new Error('Delete promise was not initialized');
    };
    const deletePromise = new Promise<void>((_, reject) => {
      rejectDelete = reject;
    });

    mockConversationShareLinks([
      makeShareLinkStatus(),
      makeShareLinkStatus({
        id: 'share-2',
        label: 'Beta link',
        share_slug: 'beta-link',
        share_token: 'token-2',
        share_url: 'https://example.com/owner/beta-link',
        anonymous_share_url: 'https://example.com/shared/token-2',
        created_at: '2026-07-14T00:05:00Z',
      }),
    ]);
    apiMock.deleteConversationShareLink.mockReturnValue(deletePromise);

    renderAuthenticatedChatPage();

    await user.click(await screen.findByRole('button', { name: 'Open share modal' }));

    expect(await screen.findByText('Alpha link')).toBeDefined();
    expect(screen.getByText('Beta link')).toBeDefined();

    await user.click(screen.getAllByRole('button', { name: 'Delete link' })[0]);
    await user.click(await screen.findByRole('button', { name: 'Confirm delete' }));

    await waitFor(() => {
      expect(screen.queryByText('Alpha link')).toBeNull();
      expect(screen.getByText('Beta link')).toBeDefined();
    });

    rejectDelete(new Error('Delete failed'));

    await waitFor(() => {
      expect(screen.getByText('Alpha link')).toBeDefined();
      expect(screen.getByText('Beta link')).toBeDefined();
      expect(
        screen.getAllByRole('alert').some((alert) => alert.textContent?.includes('Delete failed')),
      ).toBe(true);
    });

    expect(apiMock.deleteConversationShareLink).toHaveBeenCalledWith('conv-1', 'share-1');
    expect(apiMock.listConversationShareLinks).toHaveBeenCalledTimes(1);
  });

  it('renders click counts and merges analytics updates without clobbering local drafts', async () => {
    const user = userEvent.setup();
    mockConversationShareLinks([makeShareLinkStatus()]);
    apiMock.createConversationShareLink.mockResolvedValue({
      id: 'share-1',
      conversation_id: 'conv-1',
      share_token: 'token-1',
      owner_username: 'owner',
      share_slug: 'alpha-link',
      share_url: 'https://example.com/owner/alpha-link',
      anonymous_share_url: 'https://example.com/shared/token-1',
      label: 'Alpha link',
      scope_anchor_message_idx: null,
      scope_direction: null,
    });

    renderAuthenticatedChatPage();

    await user.click(await screen.findByRole('button', { name: 'Open share modal' }));

    expect(await screen.findByRole('columnheader', { name: 'Click Count' })).toBeDefined();
    const columnHeaders = screen.getAllByRole('columnheader').map((header) => header.textContent);
    expect(columnHeaders.indexOf('Access')).toBeLessThan(columnHeaders.indexOf('Click Count'));
    expect(columnHeaders.indexOf('Click Count')).toBeLessThan(columnHeaders.indexOf('URL'));
    expect(screen.getByText('Alpha link')).toBeDefined();
    expect(screen.getByText('https://example.com/owner/alpha-link')).toBeDefined();
    expect(screen.getByText('0')).toBeDefined();

    await user.click(screen.getByRole('button', { name: 'Edit link' }));
    await user.click(screen.getByRole('button', { name: 'Alpha link' }));
    const labelInput = await screen.findByPlaceholderText('Untitled link');
    await user.clear(labelInput);
    await user.type(labelInput, 'Draft label');

    expect(apiMock.subscribeConversationShareLinkAnalytics).toHaveBeenCalledWith('conv-1');
    expect(sseMock.sources).toHaveLength(1);

    sseMock.sources[0].emit('share_links', {
      links: [{ id: 'share-1', public_hit_count: 3, last_public_hit_at: '2026-07-14T01:00:00Z' }],
    });

    await waitFor(() => {
      expect((screen.getByPlaceholderText('Untitled link') as HTMLInputElement).value).toBe(
        'Draft label',
      );
    });

    await user.click(screen.getByRole('button', { name: 'Back to all links' }));

    await waitFor(() => {
      expect(screen.getByText('Alpha link')).toBeDefined();
      expect(screen.getByText('https://example.com/owner/alpha-link')).toBeDefined();
      expect(screen.getByText('3')).toBeDefined();
    });

    await user.click(screen.getByRole('button', { name: '×' }));

    await waitFor(() => {
      expect(sseMock.sources[0].close).toHaveBeenCalledTimes(1);
    });
  });
});
