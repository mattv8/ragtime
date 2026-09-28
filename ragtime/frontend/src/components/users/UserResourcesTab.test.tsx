import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { User, UserSpaceWorkspace } from '@/types';
import { UserResourcesTab } from './UserResourcesTab';

const user = { id: 'u1', username: 'alex' } as User;
const workspace = {
  id: 'w1',
  name: 'Project',
  owner_user_id: 'u1',
  members: [],
  conversation_ids: [],
  selected_tool_ids: [],
  selected_tool_group_ids: [],
} as unknown as UserSpaceWorkspace;
const props = () => ({
  user,
  users: [user],
  workspaces: [workspace],
  chats: [],
  chatsLoaded: false,
  chatsLoading: false,
  chatsError: null,
  deletingWorkspaceIds: new Set<string>(),
  workspaceStateById: {},
  workspaceLastMessageAtById: {},
  workspaceLastConversationById: {},
  workspaceMetaLoading: false,
  storageByWorkspaceId: {},
  storageFailuresByWorkspaceId: { w1: 'offline' },
  storageLoading: false,
  onLoad: vi.fn(),
  onComputeStorage: vi.fn(),
  onWorkspaceDelete: vi.fn(),
  onWorkspaceTransfer: vi.fn(),
  onOpenWorkspace: vi.fn(),
  onOpenChat: vi.fn(),
  onDeleteChat: vi.fn(),
  onCancelChat: vi.fn(),
  formatBytes: (value: number) => `${value} B`,
  formatDateTime: () => 'n/a',
  getConversationContextMeta: () => 'n/a',
});

describe('UserResourcesTab', () => {
  afterEach(cleanup);
  it('shows partial storage failure and retries missing workspace sampling', () => {
    const p = props();
    render(<UserResourcesTab {...p} />);
    expect(screen.getByText(/0 B sampled for 0\/1 workspaces; 1 failed/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry failed storage' }));
    expect(p.onComputeStorage).toHaveBeenCalledOnce();
  });

  it('loads chats instead of claiming no chats before the snapshot arrives', () => {
    const p = props();
    render(<UserResourcesTab {...p} />);
    expect(screen.getByText('Loading chats...')).toBeTruthy();
    expect(screen.queryByText('No standalone chats.')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Load chats' }));
    expect(p.onLoad).toHaveBeenCalledOnce();
  });
});
