import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ConversationSummary, User, UserSpaceWorkspace } from '@/types';
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
  chats: [] as ConversationSummary[],
  chatsLoaded: false,
  chatsLoading: false,
  chatsError: null,
  deletingWorkspaceIds: new Set<string>(),
  workspaceStateById: {},
  workspaceLastMessageAtById: {},
  workspaceLastConversationById: {},
  workspaceMetaLoading: false,
  storageByWorkspaceId: {},
  storageFailuresByWorkspaceId: { w1: 'offline' } as Record<string, string>,
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

// Test-only hook lookup: metadata spans carry data-resource-metadata, not ARIA roles or names.
const metadataFor = (rowTitle: string, kind: 'workspace' | 'chat'): HTMLElement => {
  const row = screen.getByTitle(rowTitle);
  const metadata = row.querySelector<HTMLElement>(`[data-resource-metadata='${kind}']`);
  if (!metadata) throw new Error(`Missing ${kind} metadata in "${rowTitle}"`);
  return metadata;
};

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

  it('binds workspace resource states and pending metadata to labeled details', () => {
    const p = props();
    p.workspaceStateById = { w1: { has_live_task: true } };
    p.workspaceLastConversationById = {
      w1: { model: 'anthropic::claude-opus-4-5' } as ConversationSummary,
    };
    p.workspaceMetaLoading = true;

    render(<UserResourcesTab {...p} />);

    const details = metadataFor('Open workspace: Project', 'workspace');
    expect(details.textContent).toContain('Live');
    expect(details.textContent).toMatch(/Model\s+Claude Opus 4 5/);
    expect(details.textContent).toMatch(/Context\s+n\/a/);
    expect(details.textContent).toMatch(/Last message\s+Loading…/);
    expect(details.textContent).toMatch(/Storage\s+Unavailable/);
    expect(screen.getByTitle('anthropic::claude-opus-4-5')).toBeTruthy();
  });

  it('names the workspace row button with its visible status, field labels, and values', () => {
    const p = props();
    p.workspaceStateById = { w1: { has_live_task: true } };
    p.workspaceLastConversationById = {
      w1: { model: 'anthropic::claude-opus-4-5' } as ConversationSummary,
    };
    p.workspaceLastMessageAtById = { w1: '2026-01-02T03:04:05.000Z' };
    p.formatDateTime = () => 'Jan 2, 2026';

    render(<UserResourcesTab {...p} />);

    expect(
      screen.getByRole('button', {
        name: /Live Model Claude Opus 4 5 Context n\/a Last message Jan 2, 2026 Storage Unavailable/,
      }),
    ).toBeTruthy();
  });

  it('names the chat row button with its visible running state, count, and field values', () => {
    const p = props();
    p.chatsLoaded = true;
    p.chats = [
      {
        id: 'c1',
        title: 'Current chat',
        model: 'openai::gpt-5.3-codex',
        message_count: 0,
        active_task_id: 'task-1',
        updated_at: '2026-01-02T03:04:05.000Z',
      } as ConversationSummary,
    ];
    p.formatDateTime = () => 'Jan 2, 2026';

    render(<UserResourcesTab {...p} />);

    expect(
      screen.getByRole('button', {
        name: /Running 0 messages Model GPT 5\.3 Codex Context n\/a Last message Jan 2, 2026/,
      }),
    ).toBeTruthy();
  });

  it('prefers known storage bytes over a stale storage failure without a warning class', () => {
    const p = props();
    p.storageByWorkspaceId = { w1: 512 };
    p.storageFailuresByWorkspaceId = { w1: 'offline' };

    render(<UserResourcesTab {...p} />);

    const details = metadataFor('Open workspace: Project', 'workspace');
    expect(details.textContent).toMatch(/Storage\s+512 B/);
    expect(details.textContent).not.toMatch(/Unavailable/);
    const storageValue = within(details).getByText('512 B');
    expect(storageValue.classList.contains('user-resource-meta-failure')).toBe(false);
  });

  it('binds chat count and running state while preserving selection callbacks', () => {
    const p = props();
    p.chatsLoaded = true;
    p.chats = [
      {
        id: 'c1',
        title: 'Current chat',
        model: 'openai::gpt-5.3-codex',
        message_count: 0,
        active_task_id: 'task-1',
        updated_at: '2026-01-02T03:04:05.000Z',
      } as ConversationSummary,
    ];

    render(<UserResourcesTab {...p} />);

    const details = metadataFor('Open chat: Current chat', 'chat');
    expect(details.textContent).toContain('Running');
    expect(details.textContent).toContain('0 messages');
    expect(details.textContent).toMatch(/Model\s+GPT 5\.3 Codex/);
    fireEvent.click(screen.getByTitle('Open chat: Current chat'));
    expect(p.onOpenChat).toHaveBeenCalledWith('c1');
  });

  it('shows storage as loading only while storage sampling is in progress', () => {
    const p = props();
    p.storageFailuresByWorkspaceId = {};
    p.storageLoading = true;
    p.workspaceMetaLoading = false;

    render(<UserResourcesTab {...p} />);

    const details = metadataFor('Open workspace: Project', 'workspace');
    expect(details.textContent).toMatch(/Storage\s+Loading…/);
  });

  it('keeps storage not sampled while metadata loads without storage sampling', () => {
    const p = props();
    p.storageFailuresByWorkspaceId = {};
    p.storageLoading = false;
    p.workspaceMetaLoading = true;

    render(<UserResourcesTab {...p} />);

    const details = metadataFor('Open workspace: Project', 'workspace');
    expect(details.textContent).toMatch(/Storage\s+Not sampled/);
    expect(details.textContent).toMatch(/Last message\s+Loading…/);
  });

  it('exposes full workspace context and last-message text for ellipsized fields', () => {
    const p = props();
    p.workspaceLastConversationById = {
      w1: { model: 'anthropic::claude-opus-4-5' } as ConversationSummary,
    };
    p.workspaceLastMessageAtById = { w1: '2026-01-02T03:04:05.000Z' };
    p.getConversationContextMeta = () => 'Context: 12 of 200k tokens used';
    p.formatDateTime = () => 'Jan 2, 2026, 3:04 AM';

    render(<UserResourcesTab {...p} />);

    expect(screen.getByTitle('Context: 12 of 200k tokens used')).toBeTruthy();
    expect(screen.getByTitle('Jan 2, 2026, 3:04 AM')).toBeTruthy();
  });

  it('exposes full chat context and last-message text for ellipsized fields', () => {
    const p = props();
    p.chatsLoaded = true;
    p.chats = [
      {
        id: 'c1',
        title: 'Long chat',
        model: 'openai::gpt-5.3-codex',
        message_count: 2,
        updated_at: '2026-01-02T03:04:05.000Z',
      } as ConversationSummary,
    ];
    p.getConversationContextMeta = () => 'Context: 80 of 128k tokens used';
    p.formatDateTime = () => 'Jan 2, 2026, 3:04 AM';

    render(<UserResourcesTab {...p} />);

    const details = metadataFor('Open chat: Long chat', 'chat');
    expect(within(details).getByTitle('Context: 80 of 128k tokens used').textContent).toBe(
      'Context: 80 of 128k tokens used',
    );
    expect(within(details).getByTitle('Jan 2, 2026, 3:04 AM').textContent).toBe(
      'Jan 2, 2026, 3:04 AM',
    );
  });
});
