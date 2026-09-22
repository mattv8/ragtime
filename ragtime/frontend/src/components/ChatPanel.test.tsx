import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { ReactElement, ReactNode } from 'react';
import userEvent from '@testing-library/user-event';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type {
  Conversation,
  ConversationMessageWindow,
  ConversationSummary,
  ConversationWindowEntry,
  User,
  WorkspaceChatStateResponse,
} from '@/types';
import { AvailableModelsProvider } from '@/contexts/AvailableModelsContext';
import {
  ChatPanel,
  ToolCallDisplay,
  applyConversationToolGroupWriteToggle,
  getConversationToolGroupWriteMenuItem,
  isToolEffectivelyWritableForConversation,
  mergeConversationFromWorkspaceSnapshot,
  type ActiveToolCall,
} from './ChatPanel';
import type { ChatMessageNavigationEntry } from './ChatMessageNavigator';

const apiMock = vi.hoisted(() => {
  const mock = {
    getConversation: vi.fn().mockResolvedValue(null),
    getConversationLatestExchange: vi.fn().mockResolvedValue({
      conversation: null,
      revision: 'test-revision',
      total_message_count: 0,
      entries: [],
      next_cursor: null,
      has_more: false,
      legacy_conversation: null,
    }),
    getConversationMessageWindow: vi.fn(),
    getConversationWindowMessage: vi.fn(),
    searchConversationBranches: vi.fn().mockResolvedValue({ matches: [] }),
    listConversationSummaries: vi.fn().mockResolvedValue([]),
    listConversations: vi.fn().mockResolvedValue([]),
    countConversations: vi.fn().mockResolvedValue({ count: 0 }),
    listUserSpaceWorkspaces: vi.fn().mockResolvedValue([]),
    createConversation: vi.fn(),
    deleteConversation: vi.fn().mockResolvedValue(undefined),
    updateConversationTitle: vi.fn(),
    updateConversationModel: vi.fn(),
    getConversationBranchPoints: vi.fn().mockResolvedValue([]),
    switchConversationBranch: vi.fn(),
    releaseConversationBranch: vi.fn(),
    getConversationMembers: vi.fn().mockResolvedValue([]),
    getConversationTaskState: vi
      .fn()
      .mockResolvedValue({ active_task: null, interrupted_task: null }),
    getConversationTools: vi.fn().mockResolvedValue({
      tool_selection_mode: 'default_all',
      tool_config_ids: [],
      tool_group_ids: [],
      disabled_builtin_tool_ids: [],
      tool_options: {},
    }),
    getSubagentConversationSummaries: vi.fn().mockResolvedValue([] as ConversationSummary[]),
    getConversationEventsUrl: vi.fn().mockReturnValue('/events'),
    listUserSpaceAvailableTools: vi.fn().mockResolvedValue([]),
    listUserSpaceToolGroups: vi.fn().mockResolvedValue([]),
    subscribeToolHealthEvents: vi.fn().mockReturnValue({
      addEventListener: vi.fn(),
      close: vi.fn(),
      onmessage: null,
    }),
    streamChatTask: vi.fn().mockReturnValue(
      (async function* () {
        yield* [];
      })(),
    ),
    sendMessageBackground: vi.fn().mockResolvedValue({
      id: 'task-window-send',
      status: 'running',
    }),
    compactConversation: vi.fn().mockResolvedValue({
      id: 'task-window-compaction',
      status: 'running',
    }),
  };
  mock.getConversationMessageWindow.mockImplementation(async (conversationId: string) => {
    const latestResult = [...mock.getConversationLatestExchange.mock.results]
      .reverse()
      .find(
        (_result, index) =>
          mock.getConversationLatestExchange.mock.calls[
            mock.getConversationLatestExchange.mock.calls.length - 1 - index
          ]?.[0] === conversationId,
      );
    const latest = await latestResult?.value;
    if (!latest?.conversation)
      throw new Error(`Missing latest window fixture for ${conversationId}`);
    return {
      ...latest,
      entries: [],
      next_cursor: null,
      has_more: false,
      legacy_conversation: null,
    };
  });
  mock.getConversationWindowMessage.mockImplementation(
    async (conversationId: string, index: number) => {
      const latestResult = [...mock.getConversationLatestExchange.mock.results]
        .reverse()
        .find(
          (_result, resultIndex) =>
            mock.getConversationLatestExchange.mock.calls[
              mock.getConversationLatestExchange.mock.calls.length - 1 - resultIndex
            ]?.[0] === conversationId,
        );
      const latest = await latestResult?.value;
      const entry = latest?.entries.find(
        (candidate: ConversationWindowEntry) => candidate.index === index,
      );
      if (!entry) throw new Error(`Missing deferred entry fixture for ${conversationId}:${index}`);
      if (entry.state === 'ready') return entry;
      return {
        ...entry,
        state: 'ready' as const,
        message: {
          role: entry.preview.role,
          content: entry.preview.content,
          timestamp: entry.preview.timestamp,
          message_id: entry.preview.message_id ?? undefined,
        },
        preview: null,
      };
    },
  );
  return mock;
});

const chatMessageNavigatorMock = vi.hoisted(() => ({
  renderSpy: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

// Stub the sandboxed iframe component so ToolCallDisplay tests never exercise srcdoc/postMessage.
vi.mock('./HtmlComponentDisplay', () => ({
  HtmlComponentDisplay: ({
    component,
    descriptionNode,
    anchor,
  }: {
    component: { title: string };
    descriptionNode?: ReactNode;
    anchor?: ReactNode;
  }) => (
    <div data-testid="html-component-stub">
      <span>{component.title}</span>
      {descriptionNode}
      {anchor}
    </div>
  ),
}));

vi.mock('./ChatMessageNavigator', () => ({
  ChatMessageNavigator: ({
    entries,
    activeKey,
    onNavigate,
  }: {
    entries: ChatMessageNavigationEntry[];
    activeKey: string | null;
    onNavigate: (entry: ChatMessageNavigationEntry) => void;
  }) => {
    chatMessageNavigatorMock.renderSpy({ entries, activeKey, onNavigate });
    if (entries.length < 2) return null;
    return (
      <nav aria-label="User message navigation" data-active-key={activeKey ?? ''}>
        {entries.map((entry) => (
          <button
            key={entry.key}
            type="button"
            data-entry-key={entry.key}
            data-message-index={entry.messageIndex}
            data-active={entry.key === activeKey ? 'true' : 'false'}
            onClick={() => onNavigate(entry)}
            aria-label={`Jump to user message: ${entry.preview}`}
          >
            {entry.preview}
          </button>
        ))}
      </nav>
    );
  },
}));

class MockEventSource {
  static instances: MockEventSource[] = [];

  url: string;
  withCredentials: boolean;
  onmessage: ((event: MessageEvent<string>) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  close = vi.fn();

  constructor(url: string, init?: { withCredentials?: boolean }) {
    this.url = url;
    this.withCredentials = Boolean(init?.withCredentials);
    MockEventSource.instances.push(this);
  }

  emitMessage(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent<string>);
  }

  static reset() {
    MockEventSource.instances = [];
  }
}

vi.stubGlobal('EventSource', MockEventSource as unknown as typeof EventSource);
vi.stubGlobal(
  'ResizeObserver',
  class ResizeObserverMock {
    observe() {}
    disconnect() {}
    unobserve() {}
  },
);
vi.stubGlobal('localStorage', {
  getItem: vi.fn().mockReturnValue(null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
});
vi.stubGlobal('sessionStorage', {
  getItem: vi.fn().mockReturnValue(null),
  setItem: vi.fn(),
  removeItem: vi.fn(),
});
vi.stubGlobal(
  'fetch',
  vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      models: [],
      models_loading: false,
      copilot_refresh_in_progress: false,
      provider_states: [],
      default_model: null,
      current_model: null,
      discovered_model_identifiers: [],
      allowed_models: null,
    }),
  }),
);

const defaultPrototypeScrollIntoView = vi.fn();
const defaultPrototypeScrollTo = vi.fn();
window.HTMLElement.prototype.scrollIntoView = defaultPrototypeScrollIntoView;
window.HTMLElement.prototype.scrollTo = defaultPrototypeScrollTo;
window.matchMedia = vi.fn().mockImplementation(() => ({
  matches: false,
  media: '',
  onchange: null,
  addListener: vi.fn(),
  removeListener: vi.fn(),
  addEventListener: vi.fn(),
  removeEventListener: vi.fn(),
  dispatchEvent: vi.fn(),
}));

const originalRequestAnimationFrame = window.requestAnimationFrame;
const originalCancelAnimationFrame = window.cancelAnimationFrame;

const currentUser: User = {
  id: 'user-1',
  username: 'ada',
  display_name: 'Ada Lovelace',
  email: 'ada@example.com',
  role: 'admin',
  auth_provider: 'local',
};

function makeConversation(
  id: string,
  content: string,
  overrides: Partial<Conversation> = {},
): Conversation {
  return {
    id,
    title: 'Recovered subagent conversation',
    model: 'gpt-4o',
    messages: [
      {
        role: 'user',
        content: 'Do the work.',
        timestamp: new Date().toISOString(),
        message_id: 'msg-user-1',
      },
      {
        role: 'assistant',
        content,
        timestamp: new Date().toISOString(),
        message_id: 'msg-assistant-1',
      },
    ],
    total_tokens: 12,
    active_task_id: null,
    user_id: currentUser.id,
    tool_output_mode: 'show',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    ...overrides,
  };
}

/**
 * Converts a complete fixture only at the test boundary.  The API mock still
 * returns the bounded window envelope ChatPanel consumes in production.
 */
function makeWindow(
  conversation: Conversation,
  options: {
    entries?: ConversationWindowEntry[];
    nextCursor?: string | null;
    hasMore?: boolean;
  } = {},
): ConversationMessageWindow {
  const { messages: _messages, ...metadata } = conversation;
  return {
    conversation: metadata,
    revision: `${conversation.id}-revision`,
    total_message_count: conversation.messages.length,
    entries:
      options.entries ??
      conversation.messages.map((message, index) => ({
        index,
        key: message.message_id ?? `${conversation.id}-${index}`,
        state: 'ready' as const,
        message,
        preview: null,
      })),
    next_cursor: options.nextCursor ?? null,
    has_more: options.hasMore ?? false,
    legacy_conversation: null,
  };
}

function makeLatestExchange(conversation: Conversation): ConversationMessageWindow {
  const lastIndex = conversation.messages.length - 1;
  const entries = conversation.messages.slice(Math.max(0, lastIndex - 1)).map((message, offset) => {
    const index = Math.max(0, lastIndex - 1) + offset;
    return {
      index,
      key: message.message_id ?? `${conversation.id}-${index}`,
      state: 'deferred' as const,
      message: null,
      preview: {
        role: message.role,
        content:
          typeof message.content === 'string' ? message.content : JSON.stringify(message.content),
        timestamp: message.timestamp,
        message_id: message.message_id ?? null,
        content_truncated: false,
        has_details: true,
      },
    };
  });
  return makeWindow(conversation, {
    entries,
    nextCursor: `${conversation.id}-before`,
    hasMore: true,
  });
}

function makeSummary(
  id: string,
  title: string,
  overrides: Partial<ConversationSummary> = {},
): ConversationSummary {
  return {
    id,
    title,
    model: 'gpt-4o',
    message_count: 2,
    total_tokens: 12,
    active_task_id: null,
    user_id: currentUser.id,
    created_at: '2026-09-11T12:00:00.000Z',
    updated_at: '2026-09-11T12:00:00.000Z',
    ...overrides,
  };
}

function makeWorkspaceChatState(conversation: Conversation): WorkspaceChatStateResponse {
  return {
    conversations: [conversation],
    interrupted_conversation_ids: [],
    selected_conversation_id: conversation.id,
    active_task: {
      id: 'task-parent-1',
      conversation_id: conversation.id,
      status: 'running',
      user_message: 'Coordinate the subagents.',
      streaming_state: null,
      response_content: null,
      error_message: null,
      created_at: new Date().toISOString(),
      started_at: new Date().toISOString(),
      completed_at: null,
      last_update_at: new Date().toISOString(),
    },
    interrupted_task: null,
  };
}

describe('mergeConversationFromWorkspaceSnapshot', () => {
  it('keeps a stale incoming branch and its messages from splitting the active pair', () => {
    const current = makeConversation('conversation-1', 'Current branch reply', {
      active_branch_id: 'branch-current',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const staleIncoming = makeConversation('conversation-1', 'Older, longer branch reply', {
      active_branch_id: 'branch-stale',
      updated_at: '2026-09-11T12:00:01.000Z',
      messages: [
        ...current.messages,
        {
          role: 'assistant',
          content: 'Older extra reply',
          timestamp: '2026-09-11T12:00:01.000Z',
        },
      ],
    });

    const merged = mergeConversationFromWorkspaceSnapshot(current, staleIncoming);

    expect(merged.active_branch_id).toBe('branch-current');
    expect(merged.messages).toBe(current.messages);
  });

  it('accepts an authoritative empty branch when its identity changes', () => {
    const current = makeConversation('conversation-1', 'Current branch reply', {
      active_branch_id: 'branch-current',
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    const incoming = makeConversation('conversation-1', '', {
      active_branch_id: 'branch-empty',
      messages: [],
      updated_at: '2026-09-11T12:00:02.000Z',
    });

    const merged = mergeConversationFromWorkspaceSnapshot(current, incoming);

    expect(merged.active_branch_id).toBe('branch-empty');
    expect(merged.messages).toEqual([]);
  });
});

describe('ChatPanel branch navigation', () => {
  const branchPoint = {
    branch_point_index: 0,
    branches: [
      {
        id: 'branch-a',
        conversation_id: 'conversation-branches',
        branch_point_index: 0,
        branch_kind: 'edit' as const,
        message_count: 2,
        created_at: '2026-09-11T12:00:00.000Z',
      },
      {
        id: 'branch-current',
        conversation_id: 'conversation-branches',
        branch_point_index: 0,
        branch_kind: null,
        message_count: 2,
        created_at: '2026-09-11T12:00:01.000Z',
      },
      {
        id: 'branch-b',
        conversation_id: 'conversation-branches',
        branch_point_index: 0,
        branch_kind: 'replay' as const,
        message_count: 2,
        created_at: '2026-09-11T12:00:02.000Z',
      },
    ],
  };

  it('renders an explicit live Current option and blocks switching during generation', async () => {
    apiMock.getConversationBranchPoints.mockResolvedValue([branchPoint]);
    const conversation = makeConversation('conversation-branches', 'Reply', {
      workspace_id: 'ws-1',
      active_task_id: 'running-task',
      active_branch_id: null,
    });
    apiMock.getConversation.mockResolvedValue(conversation);

    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={makeWorkspaceChatState(conversation)}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => expect(screen.getByText('4/4')).toBeDefined());
    expect(screen.getByRole('button', { name: 'Previous branch' }).hasAttribute('disabled')).toBe(
      true,
    );
    expect(screen.getByRole('button', { name: 'Next branch' }).hasAttribute('disabled')).toBe(true);
  });

  it('keeps the release option reachable for a legacy branch without saved Current', async () => {
    apiMock.getConversationBranchPoints.mockResolvedValue([
      { ...branchPoint, branches: [branchPoint.branches[0]] },
    ]);
    const conversation = makeConversation('conversation-branches', 'Legacy reply', {
      workspace_id: 'ws-1',
      active_branch_id: 'branch-a',
    });
    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );
    await waitFor(() => expect(screen.getByText('1/2')).toBeDefined());
    expect(screen.getByRole('button', { name: 'Next branch' }).hasAttribute('disabled')).toBe(
      false,
    );
  });

  it('commits a delayed switch after an intervening same-conversation refresh', async () => {
    apiMock.getConversationBranchPoints.mockResolvedValue([branchPoint]);
    let resolveSwitch: ((conversation: Conversation) => void) | undefined;
    apiMock.switchConversationBranch.mockImplementation(
      () => new Promise<Conversation>((resolve) => (resolveSwitch = resolve)),
    );
    const conversation = makeConversation('conversation-branches', 'Current reply', {
      workspace_id: 'ws-1',
      active_branch_id: 'branch-current',
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    const { rerender } = renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => expect(screen.getByText('2/3')).toBeDefined());
    fireEvent.click(screen.getByRole('button', { name: 'Previous branch' }));
    expect(apiMock.switchConversationBranch).toHaveBeenCalledTimes(1);

    rerender(
      <AvailableModelsProvider>
        <ChatPanel
          currentUser={currentUser}
          workspaceId="ws-1"
          workspaceChatState={{
            ...makeWorkspaceChatState({ ...conversation, title: 'Refreshed title' }),
            active_task: null,
          }}
          workspaceAvailableTools={[]}
          workspaceSelectedToolIds={[]}
          embedded
        />
      </AvailableModelsProvider>,
    );
    resolveSwitch?.(
      makeConversation('conversation-branches', 'Switched branch reply', {
        workspace_id: 'ws-1',
        active_branch_id: 'branch-a',
        updated_at: '2026-09-11T12:00:02.000Z',
      }),
    );

    await waitFor(() => expect(screen.getByText('Switched branch reply')).toBeDefined());
  });

  it('sends only one request for rapid repeated branch clicks', async () => {
    apiMock.getConversationBranchPoints.mockResolvedValue([branchPoint]);
    apiMock.switchConversationBranch.mockReturnValue(new Promise(() => undefined));
    const conversation = makeConversation('conversation-branches', 'Current reply', {
      workspace_id: 'ws-1',
      active_branch_id: 'branch-current',
    });
    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => expect(screen.getByText('2/3')).toBeDefined());
    const previous = screen.getByRole('button', { name: 'Previous branch' });
    fireEvent.click(previous);
    fireEvent.click(previous);
    expect(apiMock.switchConversationBranch).toHaveBeenCalledTimes(1);
  });

  it('keeps the displayed branch unchanged when a switch fails', async () => {
    apiMock.getConversationBranchPoints.mockResolvedValue([branchPoint]);
    apiMock.switchConversationBranch.mockRejectedValue(new Error('Branch switch failed'));
    const conversation = makeConversation('conversation-branches', 'Current reply', {
      workspace_id: 'ws-1',
      active_branch_id: 'branch-current',
    });
    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => expect(screen.getByText('2/3')).toBeDefined());
    fireEvent.click(screen.getByRole('button', { name: 'Previous branch' }));
    await waitFor(() => expect(screen.getByText('Branch switch failed')).toBeDefined());
    expect(screen.getByText('Current reply')).toBeDefined();
    expect(screen.getByText('2/3')).toBeDefined();
  });

  it('drops a delayed switch after leaving and returning to the conversation', async () => {
    apiMock.getConversationBranchPoints.mockResolvedValue([branchPoint]);
    let resolveSwitch: ((conversation: Conversation) => void) | undefined;
    apiMock.switchConversationBranch.mockImplementation(
      () => new Promise<Conversation>((resolve) => (resolveSwitch = resolve)),
    );
    const conversation = makeConversation('conversation-branches', 'Current reply', {
      workspace_id: 'ws-1',
      active_branch_id: 'branch-current',
    });
    const otherConversation = makeConversation('conversation-other', 'Other reply', {
      workspace_id: 'ws-1',
    });
    const stateFor = (selectedConversationId: string): WorkspaceChatStateResponse => ({
      ...makeWorkspaceChatState(conversation),
      conversations: [conversation, otherConversation],
      selected_conversation_id: selectedConversationId,
      active_task: null,
    });
    const { rerender } = renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={stateFor(conversation.id)}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => expect(screen.getByText('2/3')).toBeDefined());
    fireEvent.click(screen.getByRole('button', { name: 'Previous branch' }));
    rerender(
      <AvailableModelsProvider>
        <ChatPanel
          currentUser={currentUser}
          workspaceId="ws-1"
          workspaceChatState={stateFor(otherConversation.id)}
          workspaceAvailableTools={[]}
          workspaceSelectedToolIds={[]}
          embedded
        />
      </AvailableModelsProvider>,
    );
    await waitFor(() => expect(screen.getByText('Other reply')).toBeDefined());
    rerender(
      <AvailableModelsProvider>
        <ChatPanel
          currentUser={currentUser}
          workspaceId="ws-1"
          workspaceChatState={stateFor(conversation.id)}
          workspaceAvailableTools={[]}
          workspaceSelectedToolIds={[]}
          embedded
        />
      </AvailableModelsProvider>,
    );
    await waitFor(() => expect(screen.getByText('Current reply')).toBeDefined());
    resolveSwitch?.(
      makeConversation('conversation-branches', 'Late switched reply', {
        workspace_id: 'ws-1',
        active_branch_id: 'branch-a',
      }),
    );

    await waitFor(() => expect(screen.queryByText('Late switched reply')).toBeNull());
    expect(screen.getByText('Current reply')).toBeDefined();
  });
});

function renderChatPanel(ui: ReactElement) {
  return render(<AvailableModelsProvider>{ui}</AvailableModelsProvider>);
}

function setChatLayoutCookie(userId: string, layout: Record<string, unknown>) {
  document.cookie = `${encodeURIComponent(`chat_layout_${userId}`)}=${encodeURIComponent(JSON.stringify(layout))}; path=/`;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  document.cookie
    .split(';')
    .map((entry) => entry.trim())
    .filter(Boolean)
    .forEach((entry) => {
      const separatorIndex = entry.indexOf('=');
      const key = separatorIndex >= 0 ? entry.slice(0, separatorIndex) : entry;
      document.cookie = `${key}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
    });
  window.HTMLElement.prototype.scrollIntoView = defaultPrototypeScrollIntoView;
  window.HTMLElement.prototype.scrollTo = defaultPrototypeScrollTo;
  window.requestAnimationFrame = originalRequestAnimationFrame;
  window.cancelAnimationFrame = originalCancelAnimationFrame;
  MockEventSource.reset();
});

describe('ChatPanel standalone first-paint loading', () => {
  it('shows the main loading state instead of the welcome screen while bootstrap selection is pending', async () => {
    let releaseBootstrap: ((rows: ConversationSummary[]) => void) | undefined;
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { limit?: number; owner_scope?: string }) => {
        if (options.owner_scope === 'self' && options.limit === 1) {
          return new Promise<ConversationSummary[]>((resolve) => (releaseBootstrap = resolve));
        }
        return Promise.resolve([]);
      },
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} />);

    expect(await screen.findByText('Loading conversation')).toBeDefined();
    expect(screen.queryByText('Start a conversation')).toBeNull();
    releaseBootstrap?.([]);
    expect(await screen.findByText('Start a conversation')).toBeDefined();
  });

  it('shows latest-exchange previews while the bootstrap message page is still loading', async () => {
    let releasePage: ((page: ConversationMessageWindow) => void) | undefined;
    const selected = makeConversation('bootstrap-preview', 'Deferred bootstrap preview', {
      title: 'Bootstrap preview',
    });
    apiMock.listConversationSummaries.mockImplementationOnce(
      (_workspaceId: unknown, options: { limit?: number; owner_scope?: string }) => {
        if (options.owner_scope === 'self' && options.limit === 1) {
          return Promise.resolve([makeSummary(selected.id, selected.title)]);
        }
        return Promise.resolve([]);
      },
    );
    apiMock.getConversationLatestExchange.mockResolvedValueOnce(makeLatestExchange(selected));
    apiMock.getConversationMessageWindow.mockImplementationOnce(
      () => new Promise<ConversationMessageWindow>((resolve) => (releasePage = resolve)),
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} />);

    expect(await screen.findByText('Deferred bootstrap preview')).toBeDefined();
    expect(document.querySelector('#chat-window-main-loading')).toBeNull();
    expect(screen.queryByText('Start a conversation')).toBeNull();
    await waitFor(() => expect(releasePage).toBeDefined());

    releasePage?.({
      ...makeLatestExchange(selected),
      entries: [],
      next_cursor: null,
      has_more: false,
    });
  });

  it('renders the selected detail and usable composer before the deferred sidebar page completes', async () => {
    let releaseSidebarPage: ((rows: ConversationSummary[]) => void) | undefined;
    const selected = makeConversation('recent-1', 'First paint response', {
      title: 'Recent detail',
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { limit?: number; owner_scope?: string }) => {
        if (options.owner_scope === 'self' && options.limit === 1) {
          return Promise.resolve([makeSummary(selected.id, selected.title)]);
        }
        if (options.owner_scope === 'self') {
          return new Promise<ConversationSummary[]>((resolve) => (releaseSidebarPage = resolve));
        }
        return Promise.resolve([]);
      },
    );
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(selected));

    renderChatPanel(<ChatPanel currentUser={currentUser} />);

    expect(await screen.findByText('First paint response')).toBeDefined();
    expect(screen.getByLabelText('Message').hasAttribute('disabled')).toBe(false);
    expect(apiMock.listConversationSummaries).toHaveBeenNthCalledWith(
      1,
      undefined,
      expect.objectContaining({ limit: 1, owner_scope: 'self' }),
      expect.any(AbortSignal),
    );
    expect(apiMock.listConversations).not.toHaveBeenCalled();
    expect(apiMock.listUserSpaceWorkspaces).not.toHaveBeenCalled();

    releaseSidebarPage?.([]);
  });

  it('lands a standalone conversation at the newest message after deferred details hydrate', async () => {
    const conversation = makeConversation('initial-bottom', 'Newest response');
    let resolveDetails: ((entry: ConversationWindowEntry) => void) | undefined;
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationWindowMessage.mockImplementation(
      (_id: string, index: number) =>
        new Promise<ConversationWindowEntry>((resolve) => {
          if (index === 1) resolveDetails = resolve;
        }),
    );
    window.requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );
    await waitFor(() => expect(resolveDetails).toBeDefined());
    const messagesRoot = document.querySelector('.chat-messages') as HTMLElement;
    let scrollTop = 0;
    Object.defineProperties(messagesRoot, {
      clientHeight: { configurable: true, value: 200 },
      scrollHeight: { configurable: true, value: 800 },
      scrollTop: {
        configurable: true,
        get: () => scrollTop,
        set: (value: number) => {
          scrollTop = value;
        },
      },
    });
    const scrollTo = vi.fn((options?: ScrollToOptions | number, y?: number) => {
      scrollTop = typeof options === 'number' ? Number(y) : Number(options?.top);
    });
    messagesRoot.scrollTo = scrollTo;

    resolveDetails?.({ ...makeWindow(conversation).entries[1], index: 1 });
    await waitFor(() => expect(scrollTop).toBe(800));
    expect(scrollTo).not.toHaveBeenCalled();
    expect(scrollTop).toBe(800);
  });

  it('prioritizes an authorized shared preferred id and skips an older preferred detail', async () => {
    const older = makeConversation('older-preferred', 'Old response', {
      updated_at: '2000-01-01T00:00:00.000Z',
      user_id: 'another-user',
    });
    const shared = makeConversation('shared-preferred', 'Shared response', {
      updated_at: '2026-09-11T12:00:00.000Z',
      user_id: 'another-user',
    });
    apiMock.getConversationLatestExchange.mockImplementation((id: string) => {
      if (id === older.id) return Promise.resolve(makeLatestExchange(older));
      if (id === shared.id) return Promise.resolve(makeLatestExchange(shared));
      return Promise.resolve(makeLatestExchange(older));
    });
    apiMock.listConversationSummaries.mockResolvedValue([]);

    const { rerender } = renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={older.id} />,
    );
    await waitFor(() => expect(apiMock.listConversationSummaries).toHaveBeenCalled());

    rerender(
      <AvailableModelsProvider>
        <ChatPanel currentUser={currentUser} initialConversationId={shared.id} />
      </AvailableModelsProvider>,
    );

    expect(await screen.findByText('Shared response')).toBeDefined();
    expect(apiMock.getConversationLatestExchange.mock.calls.map(([id]) => id)).toContain(shared.id);
  });

  it('does not let a late initial detail replace explicit later navigation', async () => {
    let resolveInitial: ((window: ConversationMessageWindow) => void) | undefined;
    const initial = makeConversation('initial-race', 'Initial late response', {
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    const explicit = makeConversation('explicit-race', 'Explicit response wins', {
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    apiMock.getConversationLatestExchange.mockImplementation((id: string) => {
      if (id === initial.id) {
        return new Promise<ConversationMessageWindow>((resolve) => (resolveInitial = resolve));
      }
      return Promise.resolve(makeLatestExchange(explicit));
    });
    apiMock.listConversationSummaries.mockResolvedValue([]);

    const { rerender } = renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={initial.id} />,
    );
    await waitFor(() =>
      expect(apiMock.getConversationLatestExchange).toHaveBeenCalledWith(
        initial.id,
        undefined,
        expect.anything(),
      ),
    );

    rerender(
      <AvailableModelsProvider>
        <ChatPanel currentUser={currentUser} initialConversationId={explicit.id} />
      </AvailableModelsProvider>,
    );
    expect(await screen.findByText('Explicit response wins')).toBeDefined();

    resolveInitial?.(makeLatestExchange(initial));
    await waitFor(() => expect(screen.queryByText('Initial late response')).toBeNull());
  });

  it('hydrates a selected summary while preserving its sidebar row and recency', async () => {
    const selected = makeConversation('summary-select', 'Hydrated selected response', {
      title: 'Summary-only chat',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const initial = makeConversation('bootstrap', 'Bootstrap response', {
      title: 'Bootstrap chat',
      updated_at: '2026-09-11T12:00:03.000Z',
    });
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { limit?: number; owner_scope?: string }) => {
        if (options.owner_scope === 'self' && options.limit === 1) {
          return Promise.resolve([makeSummary(initial.id, initial.title)]);
        }
        if (options.owner_scope === 'self')
          return Promise.resolve([makeSummary(selected.id, selected.title)]);
        return Promise.resolve([]);
      },
    );
    apiMock.getConversationLatestExchange.mockImplementation((id: string) =>
      Promise.resolve(makeLatestExchange(id === selected.id ? selected : initial)),
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} />);
    expect(await screen.findByText('Bootstrap response')).toBeDefined();
    const row = await waitFor(() => {
      const matching = Array.from(document.querySelectorAll('.chat-conversation-item')).find(
        (item) => item.textContent?.includes(selected.title),
      );
      expect(matching).toBeTruthy();
      return matching as HTMLElement;
    });

    fireEvent.click(row);
    expect(await screen.findByText('Hydrated selected response')).toBeDefined();
    expect(
      Array.from(document.querySelectorAll('.chat-conversation-item')).filter((item) =>
        item.textContent?.includes(selected.title),
      ),
    ).toHaveLength(1);
  });

  it('keeps a deleted direct-detail row tombstoned when the pending sidebar page returns it', async () => {
    let releaseSidebarPage: ((rows: ConversationSummary[]) => void) | undefined;
    const detail = makeConversation('delete-tombstone', 'Delete me', {
      title: 'Delete pending sidebar row',
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(detail));
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { limit?: number }) => {
        if (options.limit === 1) return Promise.resolve([]);
        return new Promise<ConversationSummary[]>((resolve) => (releaseSidebarPage = resolve));
      },
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={detail.id} />);
    expect(await screen.findByText('Delete me')).toBeDefined();
    await waitFor(() => expect(releaseSidebarPage).toBeDefined());
    await userEvent.setup().click(screen.getByTitle('Delete'));
    await userEvent.setup().click(screen.getByTitle('Confirm delete'));
    await waitFor(() =>
      expect(apiMock.deleteConversation).toHaveBeenCalledWith(detail.id, undefined),
    );

    releaseSidebarPage?.([makeSummary(detail.id, detail.title)]);
    await waitFor(() =>
      expect(
        Array.from(document.querySelectorAll('.chat-conversation-item')).filter((item) =>
          item.textContent?.includes(detail.title),
        ),
      ).toHaveLength(0),
    );
  });

  it('continues cursor hydration after a user creates a chat while the first sidebar page is pending', async () => {
    let releaseFirstPage: ((rows: ConversationSummary[]) => void) | undefined;
    const bootstrap = makeConversation('cursor-bootstrap', 'Bootstrap stays visible', {
      title: 'Bootstrap chat',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const created = makeConversation('cursor-created', 'Created chat stays active', {
      title: 'Created while loading',
      updated_at: '2026-09-11T12:00:03.000Z',
    });
    const later = makeSummary('cursor-later', 'Later cursor row', {
      updated_at: '2026-09-10T12:00:00.000Z',
    });
    const firstPage = Array.from({ length: 50 }, (_, index) =>
      makeSummary(`cursor-page-${index}`, `Cursor page ${index}`, {
        updated_at: `2026-09-11T11:${String(index).padStart(2, '0')}:00.000Z`,
      }),
    );
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(bootstrap));
    apiMock.createConversation.mockResolvedValue(created);
    apiMock.listConversationSummaries
      .mockImplementationOnce(
        () => new Promise<ConversationSummary[]>((resolve) => (releaseFirstPage = resolve)),
      )
      .mockResolvedValueOnce([later]);

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={bootstrap.id} />);
    expect(await screen.findByText('Bootstrap stays visible')).toBeDefined();
    await waitFor(() => expect(releaseFirstPage).toBeDefined());
    await userEvent.setup().click(screen.getByTitle('Start a new conversation'));
    expect(await screen.findByText('Created chat stays active')).toBeDefined();
    expect(apiMock.listConversationSummaries).toHaveBeenCalledTimes(1);

    releaseFirstPage?.(firstPage);
    await waitFor(() =>
      expect(apiMock.listConversationSummaries).toHaveBeenNthCalledWith(
        2,
        undefined,
        expect.objectContaining({
          limit: 50,
          cursorId: firstPage[firstPage.length - 1].id,
          cursorUpdatedAt: firstPage[firstPage.length - 1].updated_at,
        }),
        expect.any(AbortSignal),
      ),
    );
    expect(await screen.findByText(later.title)).toBeDefined();
    expect(screen.getByText('Created chat stays active')).toBeDefined();
  });

  it('keeps fresh-chat bubble skeletons until an adopted conversation is ready', async () => {
    const current = makeConversation('fresh-current', 'Current reply', {
      model: 'openai::gpt-test',
    });
    const created = makeConversation('fresh-created', 'Fresh reply', { model: 'openai::gpt-test' });
    let resolveCreate: ((conversation: Conversation) => void) | undefined;
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { id: 'gpt-test', name: 'Test model', provider: 'openai', context_limit: 128_000 },
        ],
        default_model: 'gpt-test',
        current_model: 'gpt-test',
        allowed_models: ['gpt-test'],
        allowed_openapi_models: [],
      }),
    } as Response);
    apiMock.getConversationLatestExchange.mockImplementation((id: string) => {
      if (id === current.id) return Promise.resolve(makeLatestExchange(current));
      throw new Error(`Unexpected latest-exchange request for ${id}`);
    });
    apiMock.createConversation.mockImplementation(
      () => new Promise<Conversation>((resolve) => (resolveCreate = resolve)),
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={current.id} />);
    expect(await screen.findByText('Current reply')).toBeDefined();

    await userEvent.setup().click(screen.getByTitle('Start a new conversation'));
    expect(document.querySelector('.chat-message-skeleton-list')).not.toBeNull();
    expect(document.querySelector('#chat-window-main-loading')).toBeNull();
    expect(apiMock.createConversation).toHaveBeenCalledWith(
      { model: 'openai::gpt-test' },
      undefined,
    );

    resolveCreate?.(created);
    expect(await screen.findByText('Fresh reply')).toBeDefined();
    expect(document.querySelector('#chat-window-main-loading')).toBeNull();
    expect(apiMock.getConversationLatestExchange).not.toHaveBeenCalledWith(
      created.id,
      undefined,
      expect.anything(),
    );
    expect(screen.getByLabelText('Message').hasAttribute('disabled')).toBe(false);
  });

  it('uses the server fallback when the active standalone model is unavailable', async () => {
    const current = makeConversation('missing-model', 'Current reply', {
      model: 'personal-default',
    });
    const created = makeConversation('server-default', 'Fresh reply');
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { id: 'gpt-test', name: 'Test model', provider: 'openai', context_limit: 128_000 },
        ],
        default_model: 'gpt-test',
        current_model: 'gpt-test',
        allowed_models: ['gpt-test'],
        allowed_openapi_models: [],
      }),
    } as Response);
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(current));
    apiMock.createConversation.mockResolvedValue(created);

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={current.id} />);
    expect(await screen.findByText('Current reply')).toBeDefined();

    await userEvent.setup().click(screen.getByTitle('Start a new conversation'));

    expect(apiMock.createConversation).toHaveBeenCalledWith(undefined, undefined);
  });

  it('updates the standalone window selector after a successful model change without full hydration', async () => {
    const current = makeConversation('window-only-model', 'Window-only reply', {
      model: 'old-model',
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(current));
    apiMock.updateConversationModel.mockResolvedValue({
      ...current,
      model: 'sol-model',
      updated_at: '2026-09-11T12:01:00.000Z',
    });
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { id: 'old-model', name: 'Old', selector_label: 'Old', provider: 'openai' },
          { id: 'sol-model', name: 'Sol', selector_label: 'Sol', provider: 'openai' },
        ],
        default_model: 'old-model',
        current_model: 'old-model',
        allowed_models: ['old-model', 'sol-model'],
        allowed_openapi_models: [],
      }),
    } as Response);

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={current.id} />);

    expect(await screen.findByTitle('Old')).toBeDefined();
    await userEvent.setup().click(screen.getByTitle('Old'));
    await userEvent.setup().type(screen.getByLabelText('Filter models'), 'Sol');
    await userEvent.setup().click(screen.getByTitle('sol-model'));

    await waitFor(() => expect(apiMock.updateConversationModel).toHaveBeenCalledTimes(1));
    expect(await screen.findByTitle('Sol')).toBeDefined();
    expect(apiMock.getConversation).not.toHaveBeenCalled();
  });

  it('defers workspace new-chat model selection to the server', async () => {
    const current = makeConversation('workspace-current', 'Workspace reply', {
      model: 'openai::gpt-test',
      workspace_id: 'ws-model-policy',
    });
    const created = makeConversation('workspace-created', 'Fresh workspace reply', {
      workspace_id: 'ws-model-policy',
    });
    vi.mocked(globalThis.fetch).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        models: [
          { id: 'gpt-test', name: 'Test model', provider: 'openai', context_limit: 128_000 },
        ],
        default_model: 'gpt-test',
        current_model: 'gpt-test',
        allowed_models: ['gpt-test'],
        allowed_openapi_models: [],
      }),
    } as Response);
    apiMock.createConversation.mockResolvedValue(created);

    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-model-policy"
        workspaceChatState={makeWorkspaceChatState(current)}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );
    expect(await screen.findByText('Workspace reply')).toBeDefined();

    await userEvent.setup().click(screen.getByTitle('Start a new conversation'));

    expect(apiMock.createConversation).toHaveBeenCalledWith(undefined, 'ws-model-policy');
  });

  it('preserves the first visible ready message position when programmatic anchor restoration emits a scroll event', async () => {
    window.requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const conversation = makeConversation('history-anchor', 'Newest ready response', {
      messages: [
        {
          role: 'user',
          content: 'Oldest question',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'anchor-oldest-user',
        },
        {
          role: 'assistant',
          content: 'Oldest response',
          timestamp: '2026-09-11T12:00:01.000Z',
          message_id: 'anchor-oldest-assistant',
        },
        {
          role: 'user',
          content: 'Visible question',
          timestamp: '2026-09-11T12:00:02.000Z',
          message_id: 'anchor-visible-user',
        },
        {
          role: 'assistant',
          content: 'Newest ready response',
          timestamp: '2026-09-11T12:00:03.000Z',
          message_id: 'anchor-visible-assistant',
        },
      ],
    });
    const firstPage = makeWindow(conversation, {
      entries: conversation.messages.slice(2).map((message, index) => ({
        index: index + 2,
        key: message.message_id ?? `anchor-${index + 2}`,
        state: 'ready' as const,
        message,
        preview: null,
      })),
      nextCursor: 'anchor-earlier',
      hasMore: true,
    });
    const earlierPage = makeWindow(conversation, {
      entries: conversation.messages.slice(0, 2).map((message, index) => ({
        index,
        key: message.message_id ?? `anchor-${index}`,
        state: 'ready' as const,
        message,
        preview: null,
      })),
      nextCursor: null,
      hasMore: false,
    });
    let releaseEarlierPage: ((page: ConversationMessageWindow) => void) | undefined;
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow
      .mockResolvedValueOnce(firstPage)
      .mockImplementationOnce(
        () => new Promise<ConversationMessageWindow>((resolve) => (releaseEarlierPage = resolve)),
      );

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    const loadEarlier = (await screen.findByRole('button', {
      name: 'Load earlier messages',
    })) as HTMLButtonElement;
    await waitFor(() => {
      expect(apiMock.getConversationMessageWindow).toHaveBeenCalledTimes(1);
      expect(loadEarlier.disabled).toBe(false);
      expect(document.querySelector('.chat-message-deferred')).toBeNull();
    });
    const messagesRoot = document.querySelector('.chat-messages') as HTMLElement;
    let scrollTop = 200;
    let emitProgrammaticScroll = false;
    Object.defineProperty(messagesRoot, 'scrollTop', {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        scrollTop = value;
        if (emitProgrammaticScroll) fireEvent.scroll(messagesRoot);
      },
    });
    Object.defineProperties(messagesRoot, {
      clientHeight: { configurable: true, value: 400 },
      scrollHeight: { configurable: true, value: 1_000 },
    });
    messagesRoot.getBoundingClientRect = () =>
      ({
        top: 100,
        bottom: 500,
        left: 0,
        right: 300,
        width: 300,
        height: 400,
        x: 0,
        y: 100,
        toJSON: () => ({}),
      }) as DOMRect;
    let prepended = false;
    let userScrolled = false;
    const messageWrappers = Array.from(
      document.querySelectorAll<HTMLElement>('[data-chat-message-key]'),
    );
    expect(messageWrappers.map((element) => element.dataset.chatMessageKey)).toEqual([
      'anchor-visible-user',
      'anchor-visible-assistant',
    ]);
    messageWrappers.forEach((wrapper) => {
      const key = wrapper.dataset.chatMessageKey;
      wrapper.getBoundingClientRect = () => {
        const top = key === 'anchor-visible-user' ? 20 : prepended ? 260 : userScrolled ? 220 : 160;
        return {
          top,
          bottom: top + 60,
          left: 0,
          right: 300,
          width: 300,
          height: 60,
          x: 0,
          y: top,
          toJSON: () => ({}),
        } as DOMRect;
      };
    });

    fireEvent.click(loadEarlier);
    userScrolled = true;
    scrollTop = 180;
    fireEvent.scroll(messagesRoot);
    emitProgrammaticScroll = true;
    prepended = true;
    releaseEarlierPage?.(earlierPage);

    await waitFor(() => expect(apiMock.getConversationMessageWindow).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(scrollTop).toBe(220));
    expect(apiMock.getConversationMessageWindow).toHaveBeenLastCalledWith(
      conversation.id,
      { cursor: 'anchor-earlier', limit: 20 },
      undefined,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('keeps the active detail selected when summary backfill fails and retries only the backfill', async () => {
    const detail = makeConversation('backfill-active', 'Active detail remains visible', {
      title: 'Active while backfill fails',
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(detail));
    let selfAttempts = 0;
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { owner_scope?: string }) => {
        if (options.owner_scope === 'self') {
          selfAttempts += 1;
          if (selfAttempts === 1) return Promise.reject(new Error('Sidebar request failed'));
        }
        return Promise.resolve([]);
      },
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={detail.id} />);

    expect(await screen.findByText('Active detail remains visible')).toBeDefined();
    const retry = await screen.findByRole('button', { name: /retry/i });
    await userEvent.setup().click(retry);

    await waitFor(() => expect(selfAttempts).toBeGreaterThanOrEqual(2));
    expect(screen.getByText('Active detail remains visible')).toBeDefined();
    expect(apiMock.getConversationLatestExchange).toHaveBeenCalledTimes(1);
  });

  it('resumes a failed second summary page from its cursor when retrying', async () => {
    const detail = makeConversation('retry-cursor-active', 'Active detail remains selected', {
      title: 'Retry cursor active',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const firstPage = Array.from({ length: 50 }, (_, index) =>
      makeSummary(`retry-cursor-${index}`, `Retry cursor ${index}`, {
        updated_at: `2026-09-11T11:${String(index).padStart(2, '0')}:00.000Z`,
      }),
    );
    const remaining = makeSummary('retry-cursor-remaining', 'Retry loaded remaining row', {
      updated_at: '2026-09-10T12:00:00.000Z',
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(detail));
    let retriedCursor = false;
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { owner_scope?: string; cursorId?: string | null }) => {
        if (options.owner_scope !== 'self') return Promise.resolve([]);
        if (!options.cursorId) return Promise.resolve(firstPage);
        if (!retriedCursor) {
          retriedCursor = true;
          return Promise.reject(new Error('Second page failed'));
        }
        return Promise.resolve([remaining]);
      },
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={detail.id} />);
    expect(await screen.findByText('Active detail remains selected')).toBeDefined();
    const retry = await screen.findByRole('button', { name: /retry chats/i });
    await userEvent.setup().click(retry);

    await waitFor(() =>
      expect(
        apiMock.listConversationSummaries.mock.calls.some(
          ([, options]) =>
            options.owner_scope === 'self' &&
            options.cursorId === firstPage[firstPage.length - 1].id &&
            options.cursorUpdatedAt === firstPage[firstPage.length - 1].updated_at,
        ),
      ).toBe(true),
    );
    expect(await screen.findByText(remaining.title)).toBeDefined();
    expect(screen.getByText('Active detail remains selected')).toBeDefined();
  });

  it('renames a summary-only sidebar row without fetching its detail', async () => {
    const bootstrap = makeConversation('rename-bootstrap', 'Bootstrap response', {
      title: 'Bootstrap chat',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const summary = makeSummary('rename-summary', 'Before summary rename', {
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(bootstrap));
    apiMock.listConversationSummaries.mockResolvedValue([summary]);
    apiMock.updateConversationTitle.mockResolvedValue({
      id: summary.id,
      title: 'After summary rename',
      updated_at: '2026-09-11T12:00:03.000Z',
    });

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={bootstrap.id} />);
    expect(await screen.findByText('Bootstrap response')).toBeDefined();
    const row = await waitFor(() => {
      const matching = Array.from(document.querySelectorAll('.chat-conversation-item')).find(
        (item) => item.textContent?.includes(summary.title),
      );
      expect(matching).toBeTruthy();
      return matching as HTMLElement;
    });
    const rename = row.querySelector('[title="Rename"]') as HTMLButtonElement | null;
    expect(rename).toBeTruthy();
    await userEvent.setup().click(rename as HTMLButtonElement);
    await userEvent.setup().clear(row.querySelector('textarea') as HTMLTextAreaElement);
    await userEvent
      .setup()
      .type(row.querySelector('textarea') as HTMLTextAreaElement, 'After summary rename');
    fireEvent.keyDown(row.querySelector('textarea') as HTMLTextAreaElement, { key: 'Enter' });

    await waitFor(() =>
      expect(apiMock.updateConversationTitle).toHaveBeenCalledWith(
        summary.id,
        'After summary rename',
        undefined,
      ),
    );
    expect(await screen.findByText('After summary rename')).toBeDefined();
    expect(apiMock.getConversationLatestExchange).toHaveBeenCalledTimes(1);
  });

  it('retries an initial summaries failure and selects the recovered recent detail', async () => {
    const recovered = makeConversation('initial-retry-recovered', 'Recovered after initial retry', {
      title: 'Recovered recent chat',
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    let bootstrapAttempts = 0;
    apiMock.listConversationSummaries.mockImplementation(
      (_workspaceId: unknown, options: { limit?: number; owner_scope?: string }) => {
        if (options.owner_scope === 'self' && options.limit === 1) {
          bootstrapAttempts += 1;
          if (bootstrapAttempts === 1) {
            return Promise.reject(
              Object.assign(new Error('Temporary service failure'), { status: 503 }),
            );
          }
          return Promise.resolve([makeSummary(recovered.id, recovered.title)]);
        }
        return Promise.resolve([]);
      },
    );
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(recovered));

    renderChatPanel(<ChatPanel currentUser={currentUser} />);

    const retry = await screen.findByRole('button', { name: /retry/i });
    await userEvent.setup().click(retry);

    expect(await screen.findByText('Recovered after initial retry')).toBeDefined();
    expect(
      apiMock.listConversationSummaries.mock.calls.filter(
        ([, options]) => options.limit === 1 && options.owner_scope === 'self',
      ),
    ).toHaveLength(2);
    expect(apiMock.getConversationLatestExchange).toHaveBeenCalledWith(
      recovered.id,
      undefined,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('advances past a missing top summary to select the next recent detail', async () => {
    const missing = makeSummary('missing-top-summary', 'Missing top summary', {
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    const fallback = makeSummary('fallback-summary', 'Fallback summary', {
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    const fallbackDetail = makeConversation(fallback.id, 'Fallback detail selected', {
      title: fallback.title,
      updated_at: fallback.updated_at,
    });
    apiMock.listConversationSummaries.mockImplementation(
      (
        _workspaceId: unknown,
        options: { limit?: number; owner_scope?: string; cursorId?: string | null },
      ) => {
        if (options.owner_scope !== 'self' || options.limit !== 1) return Promise.resolve([]);
        return Promise.resolve(options.cursorId === missing.id ? [fallback] : [missing]);
      },
    );
    apiMock.getConversationLatestExchange.mockImplementation((id: string) => {
      if (id === missing.id) {
        return Promise.reject(Object.assign(new Error('Not found'), { status: 404 }));
      }
      return Promise.resolve(makeLatestExchange(fallbackDetail));
    });

    renderChatPanel(<ChatPanel currentUser={currentUser} />);

    expect(await screen.findByText('Fallback detail selected')).toBeDefined();
    expect(apiMock.getConversationLatestExchange.mock.calls.map(([id]) => id)).toEqual([
      missing.id,
      fallback.id,
    ]);
    expect(apiMock.listConversationSummaries).toHaveBeenNthCalledWith(
      2,
      undefined,
      expect.objectContaining({ limit: 1, cursorId: missing.id }),
      expect.any(AbortSignal),
    );
  });

  it('hydrates body-only search matches and aborts a later page when cleared', async () => {
    let resolveFirstSearchPage: ((conversations: Conversation[]) => void) | undefined;
    let resolveLateSearchPage: ((conversations: Conversation[]) => void) | undefined;
    let lateSearchSignal: AbortSignal | undefined;
    const initial = makeConversation('search-bootstrap', 'Bootstrap response', {
      title: 'Bootstrap chat',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const bodyMatch = makeConversation('body-match', 'Needle only exists in this message body', {
      title: 'Unrelated title',
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    const lateMatch = makeConversation(
      'late-body-match',
      'Late needle only exists in this message body',
      {
        title: 'Late unrelated title',
        updated_at: '2026-09-11T12:00:00.000Z',
      },
    );
    apiMock.listConversationSummaries
      .mockResolvedValueOnce([makeSummary(initial.id, initial.title)])
      .mockResolvedValueOnce([]);
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(initial));
    apiMock.listConversations.mockImplementation(
      (
        _workspaceId: unknown,
        options: { until?: string | null } | undefined,
        signal?: AbortSignal,
      ) => {
        if (options?.until) return Promise.resolve([]);
        if (!resolveFirstSearchPage) {
          return new Promise<Conversation[]>((resolve) => (resolveFirstSearchPage = resolve));
        }
        lateSearchSignal = signal;
        return new Promise<Conversation[]>((resolve) => (resolveLateSearchPage = resolve));
      },
    );

    renderChatPanel(<ChatPanel currentUser={currentUser} />);
    expect(await screen.findByText('Bootstrap response')).toBeDefined();
    const search = await screen.findByLabelText('Search conversations by title or content');
    fireEvent.change(search, { target: { value: 'needle' } });
    await waitFor(() => expect(resolveFirstSearchPage).toBeDefined());
    resolveFirstSearchPage?.([bodyMatch]);
    expect(await screen.findByText(bodyMatch.title)).toBeDefined();

    fireEvent.change(search, { target: { value: '' } });
    fireEvent.change(search, { target: { value: 'late needle' } });
    await waitFor(() => expect(lateSearchSignal).toBeDefined());
    fireEvent.change(search, { target: { value: '' } });
    await waitFor(() => expect(lateSearchSignal?.aborted).toBe(true));

    resolveLateSearchPage?.([lateMatch]);
    await waitFor(() => expect(screen.queryByText(lateMatch.title)).toBeNull());
  });

  it('does not repeat a full-prefix branch search while body pages hydrate', async () => {
    const initial = makeConversation('branch-bootstrap', 'Bootstrap response', {
      title: 'Bootstrap chat',
      updated_at: '2026-09-11T12:00:02.000Z',
    });
    const firstPage = makeConversation('branch-page-1', 'branch needle one', {
      title: 'First branch page',
      updated_at: '2026-09-11T12:00:01.000Z',
    });
    const secondPage = makeConversation('branch-page-2', 'branch needle two', {
      title: 'Second branch page',
      updated_at: '2026-09-11T12:00:00.000Z',
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(initial));
    apiMock.listConversationSummaries.mockResolvedValue([]);
    const firstHistoryPage = [
      firstPage,
      ...Array.from({ length: 49 }, (_, index) =>
        makeConversation(`branch-page-1-${index}`, `branch filler ${index}`, {
          title: `Branch filler ${index}`,
          updated_at: `2026-09-10T12:${String(index).padStart(2, '0')}:00.000Z`,
        }),
      ),
    ];
    let releaseSecondHistoryPage: ((conversations: Conversation[]) => void) | undefined;
    apiMock.listConversations.mockResolvedValueOnce(firstHistoryPage).mockImplementationOnce(
      () =>
        new Promise<Conversation[]>((resolve) => {
          releaseSecondHistoryPage = resolve;
        }),
    );
    apiMock.searchConversationBranches.mockResolvedValue({ matches: [] });

    renderChatPanel(<ChatPanel currentUser={currentUser} initialConversationId={initial.id} />);
    expect(await screen.findByText('Bootstrap response')).toBeDefined();
    await userEvent
      .setup()
      .click(await screen.findByRole('button', { name: 'Search chat branches' }));
    apiMock.searchConversationBranches.mockClear();
    fireEvent.change(await screen.findByLabelText('Search conversations by title or content'), {
      target: { value: 'branch needle' },
    });

    await waitFor(() => expect(releaseSecondHistoryPage).toBeDefined());
    expect(
      apiMock.searchConversationBranches.mock.calls.filter(
        ([, query]) => query === 'branch needle',
      ),
    ).toHaveLength(0);

    releaseSecondHistoryPage?.([secondPage]);
    expect(await screen.findByText(secondPage.title)).toBeDefined();
    await waitFor(() =>
      expect(
        apiMock.searchConversationBranches.mock.calls.filter(
          ([, query]) => query === 'branch needle',
        ),
      ).toHaveLength(1),
    );
    const [finalIds] = apiMock.searchConversationBranches.mock.calls.find(
      ([, query]) => query === 'branch needle',
    ) as [string[], string];
    expect(finalIds).toEqual(
      expect.arrayContaining([
        ...firstHistoryPage.map((conversation) => conversation.id),
        secondPage.id,
      ]),
    );
  });

  it('shows deferred latest previews before the first page and starts no sidebar request until it settles', async () => {
    let releaseFirstPage: ((window: ConversationMessageWindow) => void) | undefined;
    const conversation = makeConversation(
      'window-first-paint',
      'Deferred latest assistant response',
      {
        title: 'Window first paint',
      },
    );
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockImplementation(
      () => new Promise<ConversationMessageWindow>((resolve) => (releaseFirstPage = resolve)),
    );

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    expect(await screen.findByText('Deferred latest assistant response')).toBeDefined();
    expect(screen.getByLabelText('Message').hasAttribute('disabled')).toBe(false);
    expect(apiMock.listConversationSummaries).not.toHaveBeenCalled();
    expect(document.getElementById('chat-workbench-sidebar-loading')).toBeTruthy();
    expect(screen.queryByLabelText('Search conversations by title or content')).toBeNull();
    expect(document.querySelector('.chat-conversation-item')).toBeNull();

    await waitFor(() => expect(releaseFirstPage).toBeDefined());
    releaseFirstPage?.(makeWindow(conversation, { entries: [], nextCursor: null, hasMore: false }));
    await waitFor(() =>
      expect(apiMock.listConversationSummaries).toHaveBeenCalledWith(
        undefined,
        expect.objectContaining({ owner_scope: 'self', limit: 50 }),
        expect.any(AbortSignal),
      ),
    );
    expect(screen.getByLabelText('Search conversations by title or content')).toBeDefined();
    expect(document.querySelector('.chat-conversation-item')).toBeTruthy();
  });

  it('hydrates sidebar owner scopes independently, completing self before others', async () => {
    const conversation = makeConversation('owner-scopes', 'Owner scope preview');
    const selfPage = Array.from({ length: 50 }, (_, index) =>
      makeSummary(`self-${index}`, `Self ${index}`),
    );
    const otherPage = Array.from({ length: 50 }, (_, index) =>
      makeSummary(`other-${index}`, `Other ${index}`, { user_id: `other-${index}` }),
    );
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, { entries: [], nextCursor: null, hasMore: false }),
    );
    apiMock.listConversationSummaries
      .mockResolvedValueOnce(selfPage)
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce(otherPage)
      .mockResolvedValueOnce([]);

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    await waitFor(() => expect(apiMock.listConversationSummaries).toHaveBeenCalledTimes(4));
    expect(
      apiMock.listConversationSummaries.mock.calls.map(([, options]) => options.owner_scope),
    ).toEqual(['self', 'self', 'others', 'others']);
  });

  it('keeps the deferred transcript and composer usable when the first older page fails', async () => {
    const conversation = makeConversation('older-page-error', 'Still usable after history error');
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockRejectedValue(new Error('Older page unavailable'));

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    expect(await screen.findByText('Still usable after history error')).toBeDefined();
    expect(screen.getByLabelText('Message').hasAttribute('disabled')).toBe(false);
    expect(await screen.findByText('Older page unavailable')).toBeDefined();
    await waitFor(() => expect(apiMock.listConversationSummaries).toHaveBeenCalled());
  });

  it('automatically uses absolute message indexes for deferred details after an older page prepends', async () => {
    const conversation = makeConversation('absolute-index', 'Latest preview');
    const latest = makeLatestExchange(conversation);
    const deferredAtNineteen: ConversationWindowEntry = {
      ...latest.entries[1],
      index: 19,
      key: 'absolute-19',
    };
    apiMock.getConversationLatestExchange.mockResolvedValue({
      ...latest,
      total_message_count: 20,
      entries: [deferredAtNineteen],
    });
    apiMock.getConversationMessageWindow.mockResolvedValue({
      ...makeWindow(conversation),
      total_message_count: 20,
      entries: [{ ...makeWindow(conversation).entries[0], index: 0, key: 'absolute-0' }],
      next_cursor: 'older',
      has_more: true,
    });
    apiMock.getConversationWindowMessage.mockResolvedValue({
      index: 19,
      key: 'absolute-19',
      state: 'ready',
      message: conversation.messages[1],
      preview: null,
    });

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    await waitFor(() =>
      expect(apiMock.getConversationWindowMessage).toHaveBeenCalledWith(
        conversation.id,
        19,
        `${conversation.id}-revision`,
        undefined,
        expect.objectContaining({ signal: expect.any(AbortSignal) }),
      ),
    );
  });

  it('shows queued deferred details as loading until each newest-first hydration settles', async () => {
    const conversation = makeConversation('sequential-details', 'Newest assistant response', {
      messages: [
        {
          role: 'user',
          content: 'Oldest question',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'sequential-0',
        },
        {
          role: 'assistant',
          content: 'Middle response',
          timestamp: '2026-09-11T12:00:01.000Z',
          message_id: 'sequential-1',
        },
        {
          role: 'user',
          content: 'Newest question',
          timestamp: '2026-09-11T12:00:02.000Z',
          message_id: 'sequential-2',
        },
      ],
    });
    const deferredEntries = conversation.messages.map((message, index) => ({
      index,
      key: message.message_id ?? `sequential-${index}`,
      state: 'deferred' as const,
      message: null,
      preview: {
        role: message.role,
        content: String(message.content),
        timestamp: message.timestamp,
        message_id: message.message_id ?? null,
        content_truncated: false,
        has_details: true,
      },
    }));
    let resolveNewest: ((entry: ConversationWindowEntry) => void) | undefined;
    let resolveMiddle: ((entry: ConversationWindowEntry) => void) | undefined;
    apiMock.getConversationLatestExchange.mockResolvedValue(
      makeWindow(conversation, { entries: deferredEntries }),
    );
    apiMock.getConversationWindowMessage.mockImplementation((_: string, index: number) => {
      if (index === 2) {
        return new Promise<ConversationWindowEntry>((resolve) => (resolveNewest = resolve));
      }
      if (index === 1) {
        return new Promise<ConversationWindowEntry>((resolve) => (resolveMiddle = resolve));
      }
      return Promise.resolve({ ...makeWindow(conversation).entries[index], index });
    });

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    await waitFor(() => expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(1));
    expect(apiMock.getConversationWindowMessage.mock.calls[0]?.[1]).toBe(2);
    await waitFor(() => expect(screen.getAllByText('Loading details')).toHaveLength(3));
    expect(screen.queryByRole('button', { name: /details/i })).toBeNull();
    await waitFor(() =>
      expect(document.querySelectorAll('.chat-message-deferred[aria-busy="true"]')).toHaveLength(3),
    );
    await act(async () => {
      resolveNewest?.({ ...makeWindow(conversation).entries[2], index: 2 });
    });
    await waitFor(() => expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(2));
    expect(apiMock.getConversationWindowMessage.mock.calls.map(([, index]) => index)).toEqual([
      2, 1,
    ]);
    await waitFor(() => expect(screen.getAllByText('Loading details')).toHaveLength(2));
    await act(async () => {
      resolveMiddle?.({ ...makeWindow(conversation).entries[1], index: 1 });
    });
    await waitFor(() => expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(3));
    expect(apiMock.getConversationWindowMessage.mock.calls.map(([, index]) => index)).toEqual([
      2, 1, 0,
    ]);
  });

  it('offers failed deferred hydration as a manual retry without automatically retrying it', async () => {
    const conversation = makeConversation('failed-details', 'Deferred retry preview');
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationWindowMessage.mockRejectedValue(new Error('Detail unavailable'));

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    await waitFor(() => expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(screen.getAllByRole('button', { name: 'Retry details' })).toHaveLength(1),
    );
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(2);
  });

  it('hydrates the complete transcript only when editing a ready window message', async () => {
    const conversation = makeConversation('intent-hydration', 'Partial window reply');
    const readyUser: ConversationWindowEntry = {
      index: 0,
      key: 'intent-user',
      state: 'ready',
      message: conversation.messages[0],
      preview: null,
    };
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, { entries: [readyUser], nextCursor: null, hasMore: false }),
    );
    apiMock.getConversation.mockResolvedValue(conversation);

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    expect(await screen.findByText('Do the work.')).toBeDefined();
    expect(apiMock.getConversation).not.toHaveBeenCalled();
    await userEvent.setup().click(await screen.findByTitle('Edit and resend'));
    await waitFor(() =>
      expect(apiMock.getConversation).toHaveBeenCalledWith(
        conversation.id,
        undefined,
        expect.any(AbortSignal),
      ),
    );
  });

  it('sends from a partial window with known totals without a full detail request', async () => {
    const conversation = makeConversation('partial-send', 'Prior assistant reply', {
      total_tokens: 100,
    });
    let releaseStream: (() => void) | undefined;
    const streamCompletion = new Promise<void>((resolve) => {
      releaseStream = resolve;
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, { entries: [], nextCursor: null, hasMore: false }),
    );
    apiMock.streamChatTask.mockReturnValue(
      (async function* () {
        yield {
          type: 'state',
          content: 'Streaming partial-window reply',
          version: 1,
          tool_calls: [],
          events: [],
        };
        await streamCompletion;
      })(),
    );

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    try {
      const composer = await screen.findByLabelText('Message');
      await userEvent.setup().type(composer, 'A new message from partial state');
      await userEvent.setup().click(await screen.findByTitle('Send message'));
      await waitFor(() =>
        expect(apiMock.sendMessageBackground).toHaveBeenCalledWith(
          conversation.id,
          'A new message from partial state',
          undefined,
        ),
      );
      expect(apiMock.getConversation).not.toHaveBeenCalled();
      expect(await screen.findByText('Streaming partial-window reply')).toBeDefined();
    } finally {
      await act(async () => {
        releaseStream?.();
      });
    }
  });

  it('keeps a standalone partial-window stream connected when the task safety interval runs', async () => {
    const conversation = makeConversation('partial-stream-interval', 'Prior assistant reply');
    let streamSignal: AbortSignal | undefined;
    let releaseSecondState: (() => void) | undefined;
    let releaseCompletion: (() => void) | undefined;
    const secondState = new Promise<void>((resolve) => (releaseSecondState = resolve));
    const completion = new Promise<void>((resolve) => (releaseCompletion = resolve));
    const setIntervalSpy = vi.spyOn(window, 'setInterval');
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, { entries: [], nextCursor: null, hasMore: false }),
    );
    apiMock.streamChatTask.mockImplementation(
      (_taskId: string, _version: number, signal: AbortSignal) => {
        streamSignal = signal;
        return (async function* () {
          yield { type: 'state', content: 'First streamed state', version: 1, events: [] };
          await secondState;
          yield { type: 'state', content: 'Second streamed state', version: 2, events: [] };
          await completion;
          yield { type: 'completion', completed: true, status: 'completed' };
        })();
      },
    );

    try {
      renderChatPanel(
        <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
      );
      const composer = await screen.findByLabelText('Message');
      await userEvent.setup().type(composer, 'Keep streaming');
      await userEvent.setup().click(await screen.findByTitle('Send message'));
      expect(await screen.findByText('First streamed state')).toBeDefined();
      const taskCheck = setIntervalSpy.mock.calls.find(([, delay]) => delay === 30000)?.[0] as
        | (() => void)
        | undefined;
      expect(taskCheck).toBeDefined();

      await act(async () => {
        taskCheck?.();
        await Promise.resolve();
      });
      expect(streamSignal?.aborted).toBe(false);

      await act(async () => releaseSecondState?.());
      expect(await screen.findByText('Second streamed state')).toBeDefined();
      await act(async () => releaseCompletion?.());
      await waitFor(() => expect(apiMock.getConversationLatestExchange).toHaveBeenCalledTimes(2));
      expect(apiMock.getConversation).not.toHaveBeenCalled();
    } finally {
      releaseSecondState?.();
      releaseCompletion?.();
      setIntervalSpy.mockRestore();
    }
  });

  it('ignores a stale full hydration result after explicit navigation selects another conversation', async () => {
    let resolveFull: ((conversation: Conversation) => void) | undefined;
    const first = makeConversation('stale-full-first', 'First ready reply');
    const second = makeConversation('stale-full-second', 'Second selected reply');
    apiMock.getConversationLatestExchange.mockImplementation((id: string) =>
      Promise.resolve(makeLatestExchange(id === first.id ? first : second)),
    );
    apiMock.getConversationMessageWindow.mockImplementation((id: string) => {
      const conversation = id === first.id ? first : second;
      return Promise.resolve(
        makeWindow(conversation, {
          entries: [
            {
              index: 0,
              key: `${conversation.id}-user`,
              state: 'ready',
              message: conversation.messages[0],
              preview: null,
            },
          ],
          nextCursor: null,
          hasMore: false,
        }),
      );
    });
    apiMock.getConversation.mockImplementation(
      () => new Promise<Conversation>((resolve) => (resolveFull = resolve)),
    );

    const { rerender } = renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={first.id} />,
    );
    await screen.findByText('First ready reply');
    await userEvent.setup().click(await screen.findByTitle('Edit and resend'));
    await waitFor(() =>
      expect(apiMock.getConversation).toHaveBeenCalledWith(
        first.id,
        undefined,
        expect.any(AbortSignal),
      ),
    );
    rerender(
      <AvailableModelsProvider>
        <ChatPanel currentUser={currentUser} initialConversationId={second.id} />
      </AvailableModelsProvider>,
    );
    expect(await screen.findByText('Second selected reply')).toBeDefined();
    resolveFull?.(first);
    await waitFor(() => expect(screen.queryByText('First ready reply')).toBeNull());
  });

  it('hydrates a partial ready compaction marker before opening and retrying it at its absolute index', async () => {
    let resolveFull: ((conversation: Conversation) => void) | undefined;
    const conversation = makeConversation('compaction-window', 'After compaction', {
      messages: [
        {
          role: 'user',
          content: 'Before compaction',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'before-compaction',
        },
        {
          role: 'compaction',
          content: 'Compacted summary text',
          timestamp: '2026-09-11T12:01:00.000Z',
          message_id: 'compaction-marker',
        },
        {
          role: 'assistant',
          content: 'After compaction',
          timestamp: '2026-09-11T12:02:00.000Z',
          message_id: 'after-compaction',
        },
      ],
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, {
        entries: [
          {
            index: 1,
            key: 'compaction-marker',
            state: 'ready',
            message: conversation.messages[1],
            preview: null,
          },
        ],
        nextCursor: null,
        hasMore: false,
      }),
    );
    apiMock.getConversation.mockImplementation(
      () => new Promise<Conversation>((resolve) => (resolveFull = resolve)),
    );

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    await userEvent
      .setup()
      .click(await screen.findByRole('button', { name: 'Review chat compaction' }));
    expect(screen.queryByRole('dialog', { name: 'Compaction Result' })).toBeNull();
    await waitFor(() =>
      expect(apiMock.getConversation).toHaveBeenCalledWith(
        conversation.id,
        undefined,
        expect.any(AbortSignal),
      ),
    );
    resolveFull?.(conversation);
    const modal = await screen.findByRole('dialog', { name: 'Compaction Result' });
    expect(modal).toBeDefined();
    await userEvent.setup().click(screen.getByRole('button', { name: 'Regenerate compaction' }));
    await waitFor(() =>
      expect(apiMock.compactConversation).toHaveBeenCalledWith(conversation.id, undefined, 4, {
        replaceMessageId: 'compaction-marker',
        replaceMessageIndex: 1,
        createRevisionBranch: true,
      }),
    );
  });

  it('hydrates a partial transcript only after Find in chat and searches older loaded content', async () => {
    const conversation = makeConversation('find-window', 'Latest answer', {
      messages: [
        {
          role: 'user',
          content: 'Older needle text',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'older-needle',
        },
        {
          role: 'assistant',
          content: 'Latest answer',
          timestamp: '2026-09-11T12:01:00.000Z',
          message_id: 'latest-answer',
        },
      ],
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, { entries: [], nextCursor: null, hasMore: false }),
    );
    apiMock.getConversation.mockResolvedValue(conversation);

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );

    await screen.findByText('Latest answer');
    expect(apiMock.getConversation).not.toHaveBeenCalled();
    await userEvent.setup().click(await screen.findByRole('button', { name: 'Find in chat' }));
    await waitFor(() =>
      expect(apiMock.getConversation).toHaveBeenCalledWith(
        conversation.id,
        undefined,
        expect.any(AbortSignal),
      ),
    );
    const find = await screen.findByRole('textbox', { name: 'Find in chat' });
    await userEvent.setup().type(find, 'needle');
    expect(await screen.findByText('1/1')).toBeDefined();
    expect(screen.getByRole('button', { name: 'Next match' })).toBeDefined();
  });

  it('does not open a stale compaction marker after navigation during its hydration', async () => {
    let resolveFull: ((conversation: Conversation) => void) | undefined;
    const first = makeConversation('stale-compaction', 'First reply', {
      messages: [
        {
          role: 'compaction',
          content: 'Stale summary',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'stale-marker',
        },
        {
          role: 'assistant',
          content: 'First reply',
          timestamp: '2026-09-11T12:01:00.000Z',
          message_id: 'first-reply',
        },
      ],
    });
    const second = makeConversation('stale-compaction-next', 'Second reply');
    apiMock.getConversationLatestExchange.mockImplementation((id: string) =>
      Promise.resolve(makeLatestExchange(id === first.id ? first : second)),
    );
    apiMock.getConversationMessageWindow.mockImplementation((id: string) => {
      const conversation = id === first.id ? first : second;
      return Promise.resolve(
        makeWindow(conversation, {
          entries:
            id === first.id
              ? [
                  {
                    index: 0,
                    key: 'stale-marker',
                    state: 'ready',
                    message: first.messages[0],
                    preview: null,
                  },
                ]
              : [],
          nextCursor: null,
          hasMore: false,
        }),
      );
    });
    apiMock.getConversation.mockImplementation(
      () => new Promise<Conversation>((resolve) => (resolveFull = resolve)),
    );

    const { rerender } = renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={first.id} />,
    );
    await userEvent
      .setup()
      .click(await screen.findByRole('button', { name: 'Review chat compaction' }));
    rerender(
      <AvailableModelsProvider>
        <ChatPanel currentUser={currentUser} initialConversationId={second.id} />
      </AvailableModelsProvider>,
    );
    expect(await screen.findByText('Second reply')).toBeDefined();
    resolveFull?.(first);
    await waitFor(() =>
      expect(screen.queryByRole('dialog', { name: 'Compaction Result' })).toBeNull(),
    );
  });
});

describe('ToolCallDisplay screenshot rendering', () => {
  it('renders screenshot image from MCP metadata when streamed output JSON is truncated', () => {
    const previewImageUrl = '/indexes/userspace/runtime/workspaces/ws/screenshots/capture.png';
    const toolCall: ActiveToolCall = {
      tool: 'capture_userspace_screenshot',
      status: 'complete',
      input: {
        path: 'admin/users',
        reason: 'Verify admin users page renders correctly',
      },
      output: '{"ok":true,"preview_image_url":"/stale"... (truncated)',
      mcp: {
        ok: true,
        server_id: 'runtime-playwright',
        server_name: 'Runtime Playwright',
        tool_name: 'playwright_capture_screenshot',
        request: { path: 'admin/users' },
        response: {
          ok: true,
          preview_image_url: previewImageUrl,
          effective_width: 1440,
          effective_height: 900,
          effective_wait_after_load_ms: 1800,
        },
      },
    };

    render(<ToolCallDisplay toolCall={toolCall} defaultExpanded />);

    const image = screen.getByAltText('Captured User Space screenshot') as HTMLImageElement;
    expect(image.getAttribute('src')).toBe(previewImageUrl);
    expect(screen.queryByText('Query:')).toBeNull();
    expect(screen.queryByText('MCP:')).toBeNull();
  });

  it('renders screenshot image from MCP metadata when output is empty', () => {
    const previewImageUrl = '/indexes/userspace/runtime/workspaces/ws/screenshots/capture.png';
    const toolCall: ActiveToolCall = {
      tool: 'capture_userspace_screenshot',
      status: 'complete',
      output: '',
      mcp: {
        ok: true,
        server_id: 'runtime-playwright',
        server_name: 'Runtime Playwright',
        tool_name: 'playwright_capture_screenshot',
        request: { path: 'admin/users' },
        response: {
          ok: true,
          preview_image_url: previewImageUrl,
        },
      },
    };

    render(<ToolCallDisplay toolCall={toolCall} defaultExpanded />);

    const image = screen.getByAltText('Captured User Space screenshot') as HTMLImageElement;
    expect(image.getAttribute('src')).toBe(previewImageUrl);
  });

  it('portals the enlarged screenshot outside the embedded chat container', async () => {
    const previewImageUrl = '/indexes/userspace/runtime/workspaces/ws/screenshots/capture.png';
    const toolCall: ActiveToolCall = {
      tool: 'capture_userspace_screenshot',
      status: 'complete',
      output: '',
      mcp: {
        ok: true,
        server_id: 'runtime-playwright',
        server_name: 'Runtime Playwright',
        tool_name: 'playwright_capture_screenshot',
        request: { path: 'admin/users' },
        response: {
          ok: true,
          preview_image_url: previewImageUrl,
        },
      },
    };

    const { container } = render(
      <div data-testid="embedded-chat-wrapper">
        <ToolCallDisplay toolCall={toolCall} defaultExpanded />
      </div>,
    );

    fireEvent.click(screen.getByAltText('Captured User Space screenshot'));

    const modal = await screen.findByRole('dialog');
    expect(modal).toBeInstanceOf(HTMLElement);
    expect(container.contains(modal)).toBe(false);
    expect(modal.parentElement).toBe(document.body);
    expect(document.body.contains(modal)).toBe(true);
    expect(document.body.querySelector('[data-chat-image-modal]')).toBe(modal);
  });
});

describe('ToolCallDisplay load_tool_skills rendering', () => {
  it('renders a flat singular loaded-tool row with a wrench icon and no expandable controls', () => {
    const { container } = render(
      <ToolCallDisplay
        toolCall={{
          tool: 'load_tool_skills',
          status: 'complete',
          output: JSON.stringify({
            status: 'ok',
            transition_kind: 'load',
            bindings_changed: true,
            loaded_tool_names: ['search_git_history'],
          }),
        }}
        defaultExpanded
      />,
    );

    expect(screen.getByText('Loaded tool Search Git History')).toBeDefined();
    expect(container.querySelector('.tool-call-load-tools-flat')).not.toBeNull();
    expect(container.querySelector('.tool-call-load-tools-flat .lucide-wrench')).not.toBeNull();
    expect(
      container.querySelector('.tool-call-load-tools-flat-icon')?.getAttribute('aria-hidden'),
    ).toBe('true');
    expect(container.querySelector('.tool-call-header')).toBeNull();
    expect(container.querySelector('.tool-call-toggle')).toBeNull();
    expect(container.querySelector('.tool-call-details')).toBeNull();
    expect(container.querySelector('button')).toBeNull();
  });

  it('renders plural loaded-tool names in a single flat row', () => {
    render(
      <ToolCallDisplay
        toolCall={{
          tool: 'load_tool_skills',
          status: 'complete',
          output: JSON.stringify({
            status: 'ok',
            transition_kind: 'load',
            bindings_changed: true,
            loaded_tool_names: ['query_demo_sql', 'search_demo_sql_schema'],
          }),
        }}
      />,
    );

    expect(screen.getByText('Loaded tools Query Demo SQL, Search Demo SQL Schema')).toBeDefined();
  });

  it.each([
    {
      name: 'malformed empty-name payload',
      toolCall: {
        tool: 'load_tool_skills',
        status: 'complete' as const,
        output: JSON.stringify({
          status: 'ok',
          transition_kind: 'load',
          bindings_changed: true,
          loaded_tool_names: [''],
        }),
      },
    },
    {
      name: 'no-op bindings_changed false payload',
      toolCall: {
        tool: 'load_tool_skills',
        status: 'complete' as const,
        output: JSON.stringify({
          status: 'ok',
          transition_kind: 'load',
          bindings_changed: false,
          loaded_tool_names: ['search_git_history'],
        }),
      },
    },
    {
      name: 'json status error payload',
      toolCall: {
        tool: 'load_tool_skills',
        status: 'complete' as const,
        output: JSON.stringify({
          status: 'error',
          transition_kind: 'load',
          bindings_changed: true,
          loaded_tool_names: ['search_git_history'],
        }),
      },
    },
    {
      name: 'unload_tool_skills payload',
      toolCall: {
        tool: 'unload_tool_skills',
        status: 'complete' as const,
        output: JSON.stringify({
          status: 'ok',
          transition_kind: 'load',
          bindings_changed: true,
          loaded_tool_names: ['search_git_history'],
        }),
      },
    },
  ])('keeps $name on the generic expandable card', ({ toolCall }) => {
    const { container } = render(<ToolCallDisplay toolCall={toolCall} />);

    expect(container.querySelector('.tool-call-header')).not.toBeNull();
    expect(container.querySelector('.tool-call-load-tools-flat')).toBeNull();
  });
});

describe('ToolCallDisplay userspace validation rendering', () => {
  it('renders validate_userspace_code failure output from production-shaped validation payloads', () => {
    const payload = {
      message: 'Validation failed. Fix the reported diagnostics before finalizing.',
      action_required:
        'Fix the diagnostics in this response, then run validate_userspace_code again.',
      diagnostics: {
        live_data: [
          'dashboard/main.ts must include live_data_connections metadata.',
          'dashboard/main.ts must include live_data_checks metadata.',
          'dashboard/main.ts must call context.components.sales.execute().',
          'dashboard/main.ts must call context.components.inventory.execute().',
          'dashboard/main.ts must record execution proof for sales.',
        ],
        runtime: [
          'Runtime validation failed: preview is returning a directory listing instead of rendering the app.',
          'Runtime validation failed: browser console reported a JavaScript exception during preview.',
          'Runtime validation failed: preview renders a blank page with no visible content.',
          'Runtime validation failed: preview is rendering an error page.',
          'Runtime strict validation failed: devserver is not running.',
        ],
      },
      validation: {
        ok: false,
        validated_files: [
          'dashboard/main.ts',
          'dashboard/widgets/chart.ts',
          'dashboard/widgets/table.ts',
          'dashboard/lib/runtime.ts',
          'dashboard/lib/probe.ts',
        ],
        error_count: 10,
        errors: ['Validation failed.'],
        runtime_error_count: 1,
        runtime_errors: ['Runtime strict validation failed: devserver is not running.'],
        runtime_warning_count: 1,
        runtime_warnings: ['Screenshot capture skipped because preview was unavailable.'],
        contract_error_count: 2,
        contract_errors: [
          'dashboard/main.ts must include live_data_connections metadata.',
          'dashboard/main.ts must include live_data_checks metadata.',
        ],
        runtime_probe: {
          attempted: true,
          devserver_running: false,
          preview_status_code: 200,
          directory_listing_detected: true,
          blank_screen_detected: true,
          error_page_detected: true,
          console_error_count: 3,
          upstream_url: 'http://runtime.internal/workspace/ws-1',
          console_errors: ['ReferenceError: window is not defined'],
          content_probe: { body_text_preview: 'Internal error' },
        },
      },
    };

    render(
      <ToolCallDisplay
        toolCall={{
          tool: 'validate_userspace_code',
          status: 'complete',
          output: JSON.stringify(payload),
        }}
        defaultExpanded
      />,
    );

    expect(screen.getByText('Code validation:')).toBeDefined();
    expect(screen.getByText('Validation failed.')).toBeDefined();
    expect(screen.getByText('Live data')).toBeDefined();
    expect(screen.getByText('Runtime')).toBeDefined();
    expect(screen.getByText('dashboard/main.ts')).toBeDefined();
    expect(screen.getByText('dashboard/widgets/chart.ts')).toBeDefined();
    expect(screen.getAllByText('2 more omitted.')).toHaveLength(3);
    expect(screen.queryByText('dashboard/lib/runtime.ts')).toBeNull();
    expect(screen.getByText('Attempted')).toBeDefined();
    expect(screen.getByText('Not running')).toBeDefined();
    expect(screen.getByText('200')).toBeDefined();
    expect(screen.getByText('Directory listing')).toBeDefined();
    expect(screen.getByText('Blank screen')).toBeDefined();
    expect(screen.getByText('Error page')).toBeDefined();
    expect(screen.getByText('Console errors')).toBeDefined();
    expect(screen.queryByText('http://runtime.internal/workspace/ws-1')).toBeNull();
    expect(screen.queryByText('ReferenceError: window is not defined')).toBeNull();
    expect(screen.queryByText('Internal error')).toBeNull();
    expect(screen.queryByText('Result:')).toBeNull();
  });

  it('renders validate_userspace_code success status from validation.ok without top-level ok', () => {
    const payload = {
      message: 'Validation passed.',
      action_required: 'Create a snapshot for this completed change loop.',
      diagnostics: {},
      validation: {
        ok: true,
        validated_files: ['dashboard/main.ts', 'dashboard/lib/runtime.ts'],
        error_count: 0,
        errors: [],
        runtime_error_count: 0,
        runtime_errors: [],
        runtime_warning_count: 0,
        runtime_warnings: [],
        contract_error_count: 0,
        contract_errors: [],
        runtime_probe: {
          attempted: true,
          devserver_running: true,
          preview_status_code: 200,
          directory_listing_detected: false,
          blank_screen_detected: false,
          error_page_detected: false,
          console_error_count: 0,
        },
      },
    };

    const { container } = render(
      <ToolCallDisplay
        toolCall={{
          tool: 'validate_userspace_code',
          status: 'complete',
          output: JSON.stringify(payload),
        }}
        defaultExpanded
      />,
    );

    const status = screen.getByText('Validation passed.');
    expect(status).toBeDefined();
    expect(status.className).toContain('tool-call-userspace-json-status-pass');
    expect(container.querySelector('.tool-call-userspace-json-status-fail')).toBeNull();
    expect(screen.getByText('Create a snapshot for this completed change loop.')).toBeDefined();
  });
});

describe('ToolCallDisplay subagent rendering', () => {
  it('expands completed subagent transcripts before opening the child conversation', async () => {
    const user = userEvent.setup();
    const onOpenSubagentConversation = vi.fn();
    const toolCall: ActiveToolCall = {
      tool: 'spawn_subagents',
      status: 'complete',
      input: {
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            instructions: 'Update the type definitions and report the exact files changed.',
          },
        ],
      },
      output: JSON.stringify({
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            status: 'completed',
            conversation_id: 'child-conversation-1',
            task_id: 'task-1',
            final_output: 'Changed api.ts and ChatPanel.tsx, then ran the frontend typecheck.',
          },
        ],
      }),
    };

    render(
      <ToolCallDisplay
        toolCall={toolCall}
        defaultExpanded
        onOpenSubagentConversation={onOpenSubagentConversation}
      />,
    );

    expect(
      screen.getByText('Changed api.ts and ChatPanel.tsx, then ran the frontend typecheck.'),
    ).toBeDefined();
    expect(screen.queryByText('Prompt')).toBeNull();

    await user.click(screen.getByLabelText('Expand Types update subagent transcript'));

    expect(screen.getByText('Prompt')).toBeDefined();
    expect(
      screen.getByText('Update the type definitions and report the exact files changed.'),
    ).toBeDefined();
    expect(onOpenSubagentConversation).not.toHaveBeenCalled();

    await user.click(screen.getByLabelText('Open Types update chat session'));

    expect(onOpenSubagentConversation).toHaveBeenCalledWith('child-conversation-1');
  });

  it('collapses a running subagent card when it completes', () => {
    const baseToolCall: ActiveToolCall = {
      tool: 'spawn_subagents',
      status: 'running',
      input: {
        subagents: [
          {
            name: 'Toolbar search',
            role: 'worker',
            instructions: 'Inspect the toolbar search input behavior.',
          },
        ],
      },
      output: JSON.stringify({
        subagents: [
          {
            name: 'Toolbar search',
            role: 'worker',
            status: 'running',
            conversation_id: 'child-conversation-2',
            task_id: 'task-2',
          },
        ],
      }),
    };

    const { rerender } = render(<ToolCallDisplay toolCall={baseToolCall} defaultExpanded />);

    expect(screen.getByText('Prompt')).toBeDefined();

    rerender(
      <ToolCallDisplay
        toolCall={{
          ...baseToolCall,
          status: 'complete',
          output: JSON.stringify({
            subagents: [
              {
                name: 'Toolbar search',
                role: 'worker',
                status: 'completed',
                conversation_id: 'child-conversation-2',
                task_id: 'task-2',
                final_output: 'Confirmed the toolbar search input now filters results correctly.',
              },
            ],
          }),
        }}
        defaultExpanded
      />,
    );

    expect(screen.queryByText('Prompt')).toBeNull();
    expect(
      screen.getByText('Confirmed the toolbar search input now filters results correctly.'),
    ).toBeDefined();
  });

  it('dedupes repeated handoff tool segments inside the subagent transcript', async () => {
    const user = userEvent.setup();
    const toolCall: ActiveToolCall = {
      tool: 'spawn_subagents',
      status: 'complete',
      input: {
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            instructions: 'Update the type definitions and hand the result back once.',
          },
        ],
      },
      output: JSON.stringify({
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            status: 'completed',
            conversation_id: 'child-conversation-3',
            task_id: 'task-3',
            final_output: 'Final handoff from the child.',
          },
        ],
      }),
    };

    apiMock.getConversation.mockResolvedValue(
      makeConversation('child-conversation-3', 'Recovered child response', {
        messages: [
          {
            role: 'assistant',
            content: '',
            timestamp: new Date().toISOString(),
            message_id: 'msg-assistant-2',
            events: [
              {
                type: 'tool',
                channel: 'commentary',
                tool: 'submit_subagent_handoff',
                input: { final_output: 'First draft handoff.' },
                output: 'First draft handoff.',
              },
              {
                type: 'tool',
                channel: 'commentary',
                tool: 'submit_subagent_handoff',
                input: { final_output: 'Final handoff from the child.' },
                output: 'Final handoff from the child.',
              },
            ],
          },
        ],
      }),
    );

    const { container } = render(
      <ToolCallDisplay toolCall={toolCall} defaultExpanded onOpenSubagentConversation={vi.fn()} />,
    );

    await user.click(screen.getByLabelText('Expand Types update subagent transcript'));

    await screen.findByText('Final handoff from the child.');
    expect(screen.queryByText('First draft handoff.')).toBeNull();
    expect(container.querySelectorAll('.subagent-handoff-output')).toHaveLength(1);
  });
});

describe('ToolCallDisplay truncated subagent recovery', () => {
  it('recovers a child conversation id from parent summaries when spawn_subagents output is truncated', async () => {
    const user = userEvent.setup();
    const onOpenSubagentConversation = vi.fn();
    const toolCall: ActiveToolCall = {
      tool: 'spawn_subagents',
      status: 'complete',
      input: {
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            instructions: 'Update the type definitions.',
          },
        ],
      },
      output:
        '{"subagents": [{"name": "Types update", "role": "worker", "status": "completed", "final_output": "cut off before ids"... (truncated)',
    };
    const summary: ConversationSummary = {
      id: 'child-recovered-1',
      title: 'Types update (worker)',
      model: 'gpt-4o',
      message_count: 2,
      total_tokens: 12,
      active_task_id: null,
      parent_conversation_id: 'parent-1',
      subagent_role: 'worker',
      subagent_index: 0,
      created_at: '',
      updated_at: '',
    };
    const child = makeConversation('child-recovered-1', 'Recovered child response');

    apiMock.getSubagentConversationSummaries.mockResolvedValue([summary]);
    apiMock.getConversation.mockResolvedValue(child);

    render(
      <ToolCallDisplay
        toolCall={toolCall}
        defaultExpanded
        conversationId="parent-1"
        workspaceId="ws-1"
        onOpenSubagentConversation={onOpenSubagentConversation}
      />,
    );

    await waitFor(() => {
      expect(apiMock.getSubagentConversationSummaries).toHaveBeenCalledWith('parent-1', 'ws-1');
    });

    await user.click(screen.getByLabelText('Expand Types update subagent transcript'));
    expect(await screen.findByText('Recovered child response')).toBeDefined();

    await user.click(screen.getByLabelText('Open Types update chat session'));
    expect(onOpenSubagentConversation).toHaveBeenCalledWith('child-recovered-1');
  });

  it('prefers conversation ids recovered from malformed output over duplicate title summary matches', async () => {
    const user = userEvent.setup();
    const onOpenSubagentConversation = vi.fn();
    const toolCall: ActiveToolCall = {
      tool: 'spawn_subagents',
      status: 'complete',
      input: {
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            instructions: 'Update the type definitions in the second spawn call.',
          },
        ],
      },
      output:
        '{"subagents": [{"name": "Types update", "role": "worker", "status": "completed", "conversation_id": "second-child", "task_id": "second-task", "final_output": "cut off"... (truncated)',
    };
    const duplicateSummaries: ConversationSummary[] = [
      {
        id: 'first-child',
        title: 'Types update (worker)',
        model: 'gpt-4o',
        message_count: 2,
        total_tokens: 12,
        active_task_id: null,
        parent_conversation_id: 'parent-1',
        subagent_role: 'worker',
        subagent_index: 1,
        created_at: '',
        updated_at: '',
      },
      {
        id: 'second-child',
        title: 'Types update (worker)',
        model: 'gpt-4o',
        message_count: 2,
        total_tokens: 12,
        active_task_id: null,
        parent_conversation_id: 'parent-1',
        subagent_role: 'worker',
        subagent_index: 1,
        created_at: '',
        updated_at: '',
      },
    ];
    const firstChild = makeConversation('first-child', 'Recovered first child response');
    const secondChild = makeConversation('second-child', 'Recovered second child response');

    apiMock.getSubagentConversationSummaries.mockResolvedValue(duplicateSummaries);
    apiMock.getConversation.mockImplementation((conversationId: string) =>
      Promise.resolve(conversationId === 'second-child' ? secondChild : firstChild),
    );

    render(
      <ToolCallDisplay
        toolCall={toolCall}
        defaultExpanded
        conversationId="parent-1"
        workspaceId="ws-1"
        onOpenSubagentConversation={onOpenSubagentConversation}
      />,
    );

    await user.click(screen.getByLabelText('Expand Types update subagent transcript'));
    expect(await screen.findByText('Recovered second child response')).toBeDefined();
    expect(screen.queryByText('Recovered first child response')).toBeNull();
    expect(apiMock.getConversation).toHaveBeenCalledWith('second-child', 'ws-1');
  });

  it('shows a terminal unavailable message for truncated output without parent recovery', async () => {
    const user = userEvent.setup();
    const toolCall: ActiveToolCall = {
      tool: 'spawn_subagents',
      status: 'complete',
      input: {
        subagents: [
          {
            name: 'Types update',
            role: 'worker',
            instructions: 'Update the type definitions.',
          },
        ],
      },
      output:
        '{"subagents": [{"name": "Types update", "role": "worker", "status": "completed"... (truncated)',
    };

    apiMock.getSubagentConversationSummaries.mockResolvedValue([]);
    apiMock.getConversation.mockResolvedValue(null);

    render(<ToolCallDisplay toolCall={toolCall} defaultExpanded workspaceId="ws-1" />);

    await user.click(screen.getByLabelText('Expand Types update subagent transcript'));

    expect(
      screen.getByText('Subagent transcript is unavailable for this archived tool result.'),
    ).toBeDefined();
    expect(screen.queryByText('Loading subagent transcript...')).toBeNull();
    expect(apiMock.getSubagentConversationSummaries).not.toHaveBeenCalled();
  });
});

describe('ChatPanel streaming subagent placement', () => {
  it('renders one wrapped typing indicator before streaming content arrives', async () => {
    const conversation = makeConversation('pending-1', '', {
      title: 'Pending conversation',
      workspace_id: 'ws-1',
      messages: [],
      active_task_id: 'task-parent-1',
    });
    let releaseStream: (() => void) | null = null;
    apiMock.streamChatTask.mockImplementation(() =>
      (async function* () {
        await new Promise<void>((resolve) => {
          releaseStream = resolve;
        });
        yield { type: 'done' };
      })(),
    );

    const { container, unmount } = renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={makeWorkspaceChatState(conversation)}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => {
      expect(container.querySelectorAll('.chat-typing-indicator')).toHaveLength(1);
    });

    expect(
      container.querySelector('.chat-branch-wrapper-assistant .chat-typing-indicator'),
    ).not.toBeNull();

    const release = releaseStream as unknown as (() => void) | null;
    unmount();
    if (typeof release === 'function') {
      release();
    }
  });

  it('keeps the active subagent run at the spawn position, skips parent handoff cards, and renders parent final content after it', async () => {
    const parentConversation = makeConversation('parent-1', '', {
      title: 'Parent conversation',
      workspace_id: 'ws-1',
      messages: [],
      active_task_id: 'task-parent-1',
    });

    let releaseStream: (() => void) | null = null;
    apiMock.streamChatTask.mockImplementation((taskId: string) => {
      if (taskId === 'child-task-1') {
        return (async function* () {
          yield {
            type: 'state',
            state: {
              content: '',
              version: 1,
              content_length: 0,
              tool_calls: [],
              events: [
                {
                  type: 'tool',
                  channel: 'commentary',
                  tool: 'submit_subagent_handoff',
                  input: {
                    final_output: 'Child handoff that should stay out of the parent tool list.',
                  },
                  output: 'Child handoff that should stay out of the parent tool list.',
                },
              ],
            },
          };
        })();
      }

      return (async function* () {
        yield {
          type: 'state',
          state: {
            content: '',
            version: 1,
            content_length: 0,
            tool_calls: [],
            events: [
              {
                type: 'tool',
                channel: 'commentary',
                tool: 'spawn_subagents',
                input: {
                  subagents: [
                    {
                      name: 'Analyzer',
                      role: 'worker',
                      instructions: 'Inspect the toolbar search input behavior.',
                    },
                  ],
                },
                output: JSON.stringify({
                  subagents: [
                    {
                      name: 'Analyzer',
                      role: 'worker',
                      status: 'running',
                      conversation_id: 'child-live-1',
                      task_id: 'child-task-1',
                    },
                  ],
                }),
              },
              {
                type: 'content',
                channel: 'final',
                content: 'Parent final summary after the active handoff anchor.',
              },
              {
                type: 'tool',
                channel: 'commentary',
                tool: 'submit_subagent_handoff',
                input: {
                  final_output: 'Child handoff that should stay out of the parent tool list.',
                },
                output: 'Child handoff that should stay out of the parent tool list.',
              },
            ],
          },
        };
        await new Promise<void>((resolve) => {
          releaseStream = resolve;
        });
      })();
    });

    const { container, unmount } = renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={makeWorkspaceChatState(parentConversation)}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => {
      expect(apiMock.streamChatTask).toHaveBeenCalledWith(
        'task-parent-1',
        0,
        expect.anything(),
        'ws-1',
      );
    });

    await waitFor(() => {
      expect(MockEventSource.instances).toHaveLength(1);
    });

    MockEventSource.instances[0].emitMessage({
      event: 'subagent_spawned',
      conversation_id: 'child-live-1',
      task_id: 'child-task-1',
      name: 'Analyzer',
      role: 'worker',
      index: 0,
    });

    const activeRuns = await screen.findByLabelText('Subagents');
    const finalContent = await screen.findByText(
      'Parent final summary after the active handoff anchor.',
    );

    const chatMessageContent = container.querySelector(
      '.chat-message-streaming-active .chat-message-content',
    );

    expect(chatMessageContent?.firstElementChild).toBe(activeRuns);
    expect(chatMessageContent?.contains(finalContent)).toBe(true);
    expect(
      activeRuns.compareDocumentPosition(finalContent) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();

    const parentStandaloneHandoff = Array.from(chatMessageContent?.children ?? []).find(
      (child) => child !== activeRuns && child.querySelector('.subagent-handoff-output'),
    );

    expect(parentStandaloneHandoff).toBeUndefined();

    await waitFor(() => {
      expect(activeRuns.querySelector('.subagent-handoff-output')?.textContent).toContain(
        'Child handoff that should stay out of the parent tool list.',
      );
    });

    const release = releaseStream as unknown as (() => void) | null;
    unmount();
    if (typeof release === 'function') {
      release();
    }
  });
});

describe('ChatPanel ACL-aware conversation write helpers', () => {
  it('fails closed when chat tool ACL context is missing', () => {
    expect(
      isToolEffectivelyWritableForConversation(
        {
          id: 'tool-1',
          name: 'CRM',
          tool_type: 'http_api',
          allow_write: true,
        },
        {},
      ),
    ).toBe(false);
  });

  it('requires read_write ACL before a globally writable tool is writable in conversation', () => {
    expect(
      isToolEffectivelyWritableForConversation(
        {
          id: 'tool-1',
          name: 'CRM',
          tool_type: 'http_api',
          allow_write: true,
          access_level: 'read',
        },
        {},
      ),
    ).toBe(false);
  });

  it('allows per-conversation write enablement for globally read-only tools when ACL grants read_write', () => {
    expect(
      isToolEffectivelyWritableForConversation(
        {
          id: 'tool-1',
          name: 'ERP',
          tool_type: 'odoo',
          allow_write: false,
          access_level: 'read_write',
        },
        { write_access_enabled: true },
      ),
    ).toBe(true);
  });

  it('only persists group write options for ACL read_write tools and labels partial eligibility', () => {
    const tools = [
      {
        id: 'tool-rw-global',
        name: 'RW Global',
        tool_type: 'postgres',
        allow_write: true,
        access_level: 'read_write' as const,
      },
      {
        id: 'tool-rw-acl',
        name: 'RW ACL',
        tool_type: 'postgres',
        allow_write: false,
        access_level: 'read_write' as const,
      },
      {
        id: 'tool-read',
        name: 'Read Only ACL',
        tool_type: 'postgres',
        allow_write: true,
        access_level: 'read' as const,
      },
    ];

    expect(getConversationToolGroupWriteMenuItem(tools, {}, false)?.label).toBe(
      'Enable write access for 2 eligible tools in this group',
    );

    expect(applyConversationToolGroupWriteToggle(tools, {}, false)).toEqual({
      'tool-rw-acl': { write_access_enabled: true },
    });
  });
});

describe('ChatPanel tool group menu refresh', () => {
  it('tracks conversationToolOptions in the group menu callback dependencies', () => {
    const source = readFileSync(join(cwd(), 'src/components/ChatPanel.tsx'), 'utf8');

    expect(source).toMatch(
      /const getToolGroupMenuItems = useCallback\([\s\S]*?\[\s*conversationForTools,\s*conversationToolOptions,\s*isConversationViewer,\s*saveConversationToolOptions,\s*savingTools,\s*\]/,
    );
  });
});

describe('ChatPanel bounded-window operation boundaries', () => {
  it('keeps manual deferred-detail retry behind the automatic hydration flight', async () => {
    const conversation = makeConversation('hydrate-retry-race', 'Newest reply', {
      messages: [
        {
          role: 'user',
          content: 'old',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'race-0',
        },
        {
          role: 'assistant',
          content: 'middle',
          timestamp: '2026-09-11T12:00:01.000Z',
          message_id: 'race-1',
        },
        {
          role: 'user',
          content: 'new',
          timestamp: '2026-09-11T12:00:02.000Z',
          message_id: 'race-2',
        },
      ],
    });
    const entries: ConversationWindowEntry[] = conversation.messages.map((message, index) => {
      if (index === 0) {
        return {
          index,
          key: message.message_id || String(index),
          state: 'ready' as const,
          message,
          preview: null,
        };
      }
      return {
        index,
        key: message.message_id || String(index),
        state: 'deferred' as const,
        message: null,
        preview: {
          role: message.role,
          content: String(message.content),
          timestamp: message.timestamp,
          message_id: message.message_id || null,
          content_truncated: false,
          has_details: true,
        },
      };
    });
    let rejectNewest: ((reason?: unknown) => void) | undefined;
    let resolveMiddle: ((entry: ConversationWindowEntry) => void) | undefined;
    apiMock.getConversationLatestExchange.mockResolvedValue(makeWindow(conversation, { entries }));
    apiMock.getConversationWindowMessage.mockImplementation((_id: string, index: number) => {
      if (
        index === 2 &&
        apiMock.getConversationWindowMessage.mock.calls.filter(([, i]) => i === 2).length === 1
      ) {
        return new Promise<ConversationWindowEntry>((_resolve, reject) => {
          rejectNewest = reject;
        });
      }
      if (index === 1) {
        return new Promise<ConversationWindowEntry>((resolve) => {
          resolveMiddle = resolve;
        });
      }
      return new Promise<ConversationWindowEntry>(() => undefined);
    });

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );
    await waitFor(() =>
      expect(apiMock.getConversationWindowMessage.mock.calls.map(([, index]) => index)).toEqual([
        2,
      ]),
    );
    await act(async () => {
      rejectNewest?.(new Error('newest failed'));
    });
    await waitFor(() =>
      expect(apiMock.getConversationWindowMessage.mock.calls.map(([, index]) => index)).toEqual([
        2, 1,
      ]),
    );
    const retryDetails = await screen.findByRole('button', { name: 'Retry details' });
    expect(retryDetails.hasAttribute('disabled')).toBe(true);
    await userEvent.setup().click(retryDetails);
    expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(2);
    await act(async () => {
      resolveMiddle?.({
        ...entries[1],
        state: 'ready',
        message: conversation.messages[1],
        preview: null,
      });
    });
    await waitFor(() => expect(retryDetails.hasAttribute('disabled')).toBe(false));
    await userEvent.setup().click(retryDetails);
    await waitFor(() => expect(apiMock.getConversationWindowMessage).toHaveBeenCalledTimes(3));
  });

  it('restores a pending older-page anchor after deferred detail hydration completes', async () => {
    window.requestAnimationFrame = vi.fn((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    });
    const conversation = makeConversation('anchor-detail-race', 'latest', {
      messages: [
        {
          role: 'user',
          content: 'old question',
          timestamp: '2026-09-11T12:00:00.000Z',
          message_id: 'anchor-0',
        },
        {
          role: 'assistant',
          content: 'old response',
          timestamp: '2026-09-11T12:00:01.000Z',
          message_id: 'anchor-1',
        },
        {
          role: 'user',
          content: 'visible question',
          timestamp: '2026-09-11T12:00:02.000Z',
          message_id: 'anchor-2',
        },
        {
          role: 'assistant',
          content: 'deferred latest',
          timestamp: '2026-09-11T12:00:03.000Z',
          message_id: 'anchor-3',
        },
      ],
    });
    const deferred: ConversationWindowEntry = {
      index: 3,
      key: 'anchor-3',
      state: 'deferred',
      message: null,
      preview: {
        role: 'assistant',
        content: 'deferred latest',
        timestamp: conversation.messages[3].timestamp,
        message_id: 'anchor-3',
        content_truncated: false,
        has_details: true,
      },
    };
    const firstPage = makeWindow(conversation, {
      entries: [
        {
          index: 2,
          key: 'anchor-2',
          state: 'ready',
          message: conversation.messages[2],
          preview: null,
        },
      ],
      nextCursor: 'older',
      hasMore: true,
    });
    const olderPage = makeWindow(conversation, {
      entries: [
        {
          index: 0,
          key: 'anchor-0',
          state: 'ready',
          message: conversation.messages[0],
          preview: null,
        },
        {
          index: 1,
          key: 'anchor-1',
          state: 'ready',
          message: conversation.messages[1],
          preview: null,
        },
      ],
      nextCursor: null,
      hasMore: false,
    });
    let resolveDetail: ((entry: ConversationWindowEntry) => void) | undefined;
    let resolveOlder: ((page: ConversationMessageWindow) => void) | undefined;
    apiMock.getConversationLatestExchange.mockResolvedValue(
      makeWindow(conversation, { entries: [deferred], nextCursor: 'older', hasMore: true }),
    );
    apiMock.getConversationMessageWindow.mockResolvedValueOnce(firstPage).mockImplementationOnce(
      () =>
        new Promise<ConversationMessageWindow>((resolve) => {
          resolveOlder = resolve;
        }),
    );
    apiMock.getConversationWindowMessage.mockImplementation(
      () =>
        new Promise<ConversationWindowEntry>((resolve) => {
          resolveDetail = resolve;
        }),
    );
    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );
    const loadEarlier = await screen.findByRole('button', { name: 'Load earlier messages' });
    await waitFor(() => expect(resolveDetail).toBeDefined());
    const root = document.querySelector('.chat-messages') as HTMLElement;
    let scrollTop = 200;
    Object.defineProperty(root, 'scrollTop', {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        scrollTop = value;
      },
    });
    Object.defineProperties(root, {
      clientHeight: { configurable: true, value: 400 },
      scrollHeight: { configurable: true, value: 1000 },
    });
    root.getBoundingClientRect = () =>
      ({
        top: 100,
        bottom: 500,
        left: 0,
        right: 300,
        width: 300,
        height: 400,
        x: 0,
        y: 100,
        toJSON: () => ({}),
      }) as DOMRect;
    let prepended = false;
    for (const element of Array.from(
      document.querySelectorAll<HTMLElement>('[data-chat-message-key]'),
    )) {
      const key = element.dataset.chatMessageKey;
      element.getBoundingClientRect = () => {
        const top = key === 'anchor-2' ? (prepended ? 360 : 120) : key === 'anchor-3' ? 220 : 160;
        return {
          top,
          bottom: top + 60,
          left: 0,
          right: 300,
          width: 300,
          height: 60,
          x: 0,
          y: top,
          toJSON: () => ({}),
        } as DOMRect;
      };
    }
    fireEvent.scroll(root);
    fireEvent.click(loadEarlier);
    await waitFor(() => expect(resolveOlder).toBeDefined());
    resolveDetail?.({
      index: 3,
      key: 'anchor-3',
      state: 'ready',
      message: conversation.messages[3],
      preview: null,
    });
    await waitFor(() => expect(document.querySelector('.chat-message-deferred')).toBeNull());
    prepended = true;
    resolveOlder?.(olderPage);
    await waitFor(() => expect(scrollTop).toBe(440));
  });

  it('loads earlier messages when no message intersects the viewport for anchoring', async () => {
    const conversation = makeConversation('unanchored-older-page', 'latest');
    const latestEntry = makeWindow(conversation).entries[1];
    const firstPage = makeWindow(conversation, {
      entries: [latestEntry],
      nextCursor: 'older',
      hasMore: true,
    });
    const olderPage = makeWindow(conversation, {
      entries: [makeWindow(conversation).entries[0]],
      nextCursor: null,
      hasMore: false,
    });
    apiMock.getConversationLatestExchange.mockResolvedValue(firstPage);
    apiMock.getConversationMessageWindow
      .mockResolvedValueOnce(firstPage)
      .mockResolvedValueOnce(olderPage);

    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );
    const loadEarlier = await screen.findByRole('button', { name: 'Load earlier messages' });
    const root = document.querySelector('.chat-messages') as HTMLElement;
    root.getBoundingClientRect = () =>
      ({
        top: 100,
        bottom: 200,
        left: 0,
        right: 300,
        width: 300,
        height: 100,
        x: 0,
        y: 100,
        toJSON: () => ({}),
      }) as DOMRect;
    for (const element of Array.from(
      document.querySelectorAll<HTMLElement>('[data-chat-message-key]'),
    )) {
      element.getBoundingClientRect = () =>
        ({
          top: 300,
          bottom: 360,
          left: 0,
          right: 300,
          width: 300,
          height: 60,
          x: 0,
          y: 300,
          toJSON: () => ({}),
        }) as DOMRect;
    }

    await userEvent.setup().click(loadEarlier);
    await waitFor(() => expect(apiMock.getConversationMessageWindow).toHaveBeenCalledTimes(2));
    expect(document.querySelector('#chat-window-history-error')).toBeNull();
  });

  it('shows usable conversation tools for a standalone partial window without fetching its full transcript', async () => {
    const conversation = makeConversation('partial-tools', 'partial reply');
    apiMock.getConversationLatestExchange.mockResolvedValue(makeLatestExchange(conversation));
    apiMock.getConversationMessageWindow.mockResolvedValue(
      makeWindow(conversation, { entries: [], nextCursor: null, hasMore: false }),
    );
    renderChatPanel(
      <ChatPanel currentUser={currentUser} initialConversationId={conversation.id} />,
    );
    await screen.findByText('partial reply');
    expect((await screen.findAllByTitle(/Conversation Tools/)).length).toBeGreaterThan(0);
    expect(apiMock.getConversation).not.toHaveBeenCalled();
  });
});

describe('ChatPanel user message navigator integration', () => {
  it('derives chronological user-only entries with normalized previews and attachment fallback', async () => {
    const conversation = makeConversation('navigator-1', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
      messages: [
        {
          role: 'user',
          content: '  First\n\n   message  ',
          timestamp: '2026-07-29T12:00:00.000Z',
          message_id: 'msg-user-1',
        },
        {
          role: 'assistant',
          content: 'First response',
          timestamp: '2026-07-29T12:00:01.000Z',
          message_id: 'msg-assistant-1',
        },
        {
          role: 'user',
          content: JSON.stringify([
            { type: 'text', text: 'Second\nline   with   spaces' },
            { type: 'image_url', image_url: { url: 'https://example.com/two.png' } },
          ]),
          timestamp: '2026-07-29T12:00:02.000Z',
          message_id: 'msg-user-2',
        },
        {
          role: 'assistant',
          content: 'Second response',
          timestamp: '2026-07-29T12:00:03.000Z',
          message_id: 'msg-assistant-2',
        },
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: { url: 'https://example.com/only-attachment.png' },
            },
          ],
          timestamp: '2026-07-29T12:00:04.000Z',
        },
      ],
    });

    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => {
      expect(chatMessageNavigatorMock.renderSpy).toHaveBeenCalled();
    });

    const latestCalls = chatMessageNavigatorMock.renderSpy.mock.calls;
    const latestCall = latestCalls[latestCalls.length - 1]?.[0];
    expect(latestCall?.entries).toEqual([
      {
        key: 'msg-user-1',
        messageIndex: 0,
        preview: 'First message',
      },
      {
        key: 'msg-user-2',
        messageIndex: 2,
        preview: 'Second line with spaces',
      },
      {
        key: expect.any(String),
        messageIndex: 4,
        preview: '1 attachment',
      },
    ]);
  });

  it('keeps the latest navigator destination selected through intermediate smooth-scroll frames, then resumes geometry tracking after completion and cancellation', async () => {
    const user = userEvent.setup();
    const conversation = makeConversation('navigator-2', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
      messages: [
        {
          role: 'user',
          content: 'First question',
          timestamp: '2026-07-29T12:10:00.000Z',
          message_id: 'msg-user-a',
        },
        {
          role: 'assistant',
          content: 'First answer',
          timestamp: '2026-07-29T12:10:01.000Z',
          message_id: 'msg-assistant-a',
        },
        {
          role: 'user',
          content: 'Second question',
          timestamp: '2026-07-29T12:10:02.000Z',
          message_id: 'msg-user-b',
        },
        {
          role: 'assistant',
          content: 'Second answer',
          timestamp: '2026-07-29T12:10:03.000Z',
          message_id: 'msg-assistant-b',
        },
        {
          role: 'user',
          content: 'Third question',
          timestamp: '2026-07-29T12:10:04.000Z',
          message_id: 'msg-user-c',
        },
      ],
    });

    const frameCallbacks: FrameRequestCallback[] = [];
    const flushAnimationFrames = () => {
      const pending = frameCallbacks.splice(0, frameCallbacks.length);
      pending.forEach((callback) => callback(0));
    };
    const requestAnimationFrameSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((callback: FrameRequestCallback) => {
        frameCallbacks.push(callback);
        return frameCallbacks.length;
      });

    try {
      renderChatPanel(
        <ChatPanel
          currentUser={currentUser}
          workspaceId="ws-1"
          workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
          workspaceAvailableTools={[]}
          workspaceSelectedToolIds={[]}
          embedded
        />,
      );

      const messagesRoot = await waitFor(() => {
        const element = document.querySelector('.chat-messages') as HTMLElement | null;
        expect(element).toBeTruthy();
        return element as HTMLElement;
      });
      let scrollTop = 120;
      Object.defineProperty(messagesRoot, 'scrollTop', {
        configurable: true,
        get: () => scrollTop,
        set: (value: number) => {
          scrollTop = value;
        },
      });
      Object.defineProperty(messagesRoot, 'clientHeight', {
        configurable: true,
        value: 400,
      });
      Object.defineProperty(messagesRoot, 'scrollHeight', {
        configurable: true,
        get: () => 1500,
      });
      messagesRoot.getBoundingClientRect = () =>
        ({
          top: 100,
          bottom: 500,
          left: 0,
          right: 300,
          width: 300,
          height: 400,
          x: 0,
          y: 100,
          toJSON: () => ({}),
        }) as DOMRect;

      const scrollToMock = vi.fn();
      messagesRoot.scrollTo = scrollToMock;

      const wrappers = await waitFor(() => {
        const elements = document.querySelectorAll('.chat-branch-wrapper-user');
        expect(elements).toHaveLength(3);
        return elements;
      });
      const wrapperTops = [120, 220, 430];
      wrappers.forEach((wrapper, index) => {
        wrapper.getBoundingClientRect = () =>
          ({
            top: wrapperTops[index],
            bottom: wrapperTops[index] + 60,
            left: 0,
            right: 300,
            width: 300,
            height: 60,
            x: 0,
            y: wrapperTops[index],
            toJSON: () => ({}),
          }) as DOMRect;
      });
      flushAnimationFrames();

      await waitFor(() => {
        expect(
          screen.getByRole('button', { name: 'Jump to user message: Second question' }),
        ).toBeDefined();
      });

      await user.click(
        screen.getByRole('button', { name: 'Jump to user message: First question' }),
      );
      await user.click(
        screen.getByRole('button', { name: 'Jump to user message: Second question' }),
      );

      expect(screen.getByLabelText('User message navigation').getAttribute('data-active-key')).toBe(
        'msg-user-b',
      );
      expect(scrollToMock).toHaveBeenLastCalledWith({
        top: scrollTop + wrapperTops[1] - 100 - 400 * 0.25,
        behavior: 'smooth',
      });

      const renderCountBeforeFirstIntermediateScroll =
        chatMessageNavigatorMock.renderSpy.mock.calls.length;
      wrapperTops[0] = 120;
      wrapperTops[1] = 260;
      wrapperTops[2] = 420;
      fireEvent.scroll(messagesRoot);
      flushAnimationFrames();

      await waitFor(() => {
        expect(
          screen.getByLabelText('User message navigation').getAttribute('data-active-key'),
        ).toBe('msg-user-b');
      });
      const firstIntermediateKeys = chatMessageNavigatorMock.renderSpy.mock.calls
        .slice(renderCountBeforeFirstIntermediateScroll)
        .map((call) => call[0].activeKey);
      expect(firstIntermediateKeys).not.toContain('msg-user-a');

      await user.click(
        screen.getByRole('button', { name: 'Jump to user message: Third question' }),
      );

      expect(screen.getByLabelText('User message navigation').getAttribute('data-active-key')).toBe(
        'msg-user-c',
      );
      expect(scrollToMock).toHaveBeenLastCalledWith({
        top: scrollTop + wrapperTops[2] - 100 - 400 * 0.25,
        behavior: 'smooth',
      });

      const renderCountBeforeSecondIntermediateScroll =
        chatMessageNavigatorMock.renderSpy.mock.calls.length;
      wrapperTops[0] = -40;
      wrapperTops[1] = 120;
      wrapperTops[2] = 260;
      fireEvent.scroll(messagesRoot);
      flushAnimationFrames();

      await waitFor(() => {
        expect(
          screen.getByLabelText('User message navigation').getAttribute('data-active-key'),
        ).toBe('msg-user-c');
      });
      const secondIntermediateKeys = chatMessageNavigatorMock.renderSpy.mock.calls
        .slice(renderCountBeforeSecondIntermediateScroll)
        .map((call) => call[0].activeKey);
      expect(secondIntermediateKeys).not.toContain('msg-user-b');

      scrollTop = 340;
      wrapperTops[0] = -260;
      wrapperTops[1] = -20;
      wrapperTops[2] = 150;
      fireEvent.scroll(messagesRoot);
      flushAnimationFrames();

      await waitFor(() => {
        expect(
          screen.getByLabelText('User message navigation').getAttribute('data-active-key'),
        ).toBe('msg-user-c');
      });

      await user.click(
        screen.getByRole('button', { name: 'Jump to user message: Third question' }),
      );
      fireEvent.wheel(messagesRoot);
      wrapperTops[0] = -160;
      wrapperTops[1] = 150;
      wrapperTops[2] = 340;
      fireEvent.scroll(messagesRoot);
      flushAnimationFrames();

      await waitFor(() => {
        expect(
          screen.getByLabelText('User message navigation').getAttribute('data-active-key'),
        ).toBe('msg-user-b');
      });
    } finally {
      requestAnimationFrameSpy.mockRestore();
    }
  });

  it('disables auto-follow after a navigator jump until normal scrolling re-enables it', async () => {
    const user = userEvent.setup();
    const baseConversation = makeConversation('navigator-2b', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
      messages: [
        {
          role: 'user',
          content: 'First question',
          timestamp: '2026-07-29T12:10:00.000Z',
          message_id: 'msg-user-a',
        },
        {
          role: 'assistant',
          content: 'First answer',
          timestamp: '2026-07-29T12:10:01.000Z',
          message_id: 'msg-assistant-a',
        },
        {
          role: 'user',
          content: 'Second question',
          timestamp: '2026-07-29T12:10:02.000Z',
          message_id: 'msg-user-b',
        },
      ],
    });

    const { rerender } = renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(baseConversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    const messagesRoot = await waitFor(() => {
      const element = document.querySelector('.chat-messages') as HTMLElement | null;
      expect(element).toBeTruthy();
      return element as HTMLElement;
    });

    let scrollTop = 300;
    Object.defineProperty(messagesRoot, 'scrollTop', {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        scrollTop = value;
      },
    });
    Object.defineProperty(messagesRoot, 'clientHeight', {
      configurable: true,
      value: 400,
    });
    Object.defineProperty(messagesRoot, 'scrollHeight', {
      configurable: true,
      get: () => 1200,
    });

    await user.click(screen.getByRole('button', { name: 'Jump to user message: Second question' }));
    defaultPrototypeScrollTo.mockClear();

    const updatedConversation = {
      ...baseConversation,
      messages: [
        ...baseConversation.messages,
        {
          role: 'assistant' as const,
          content: 'Follow-up answer',
          timestamp: '2026-07-29T12:10:03.000Z',
          message_id: 'msg-assistant-b',
        },
      ],
    };

    rerender(
      <AvailableModelsProvider>
        <ChatPanel
          currentUser={currentUser}
          workspaceId="ws-1"
          workspaceChatState={{ ...makeWorkspaceChatState(updatedConversation), active_task: null }}
          workspaceAvailableTools={[]}
          workspaceSelectedToolIds={[]}
          embedded
        />
      </AvailableModelsProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Follow-up answer')).toBeDefined();
    });
    expect(defaultPrototypeScrollTo).not.toHaveBeenCalled();

    scrollTop = 760;
    fireEvent.scroll(messagesRoot);

    const afterBottomConversation = {
      ...updatedConversation,
      messages: [
        ...updatedConversation.messages,
        {
          role: 'assistant' as const,
          content: 'Newest answer',
          timestamp: '2026-07-29T12:10:04.000Z',
          message_id: 'msg-assistant-c',
        },
      ],
    };

    defaultPrototypeScrollTo.mockClear();
    rerender(
      <AvailableModelsProvider>
        <ChatPanel
          currentUser={currentUser}
          workspaceId="ws-1"
          workspaceChatState={{
            ...makeWorkspaceChatState(afterBottomConversation),
            active_task: null,
          }}
          workspaceAvailableTools={[]}
          workspaceSelectedToolIds={[]}
          embedded
        />
      </AvailableModelsProvider>,
    );

    await waitFor(() => {
      expect(screen.getByText('Newest answer')).toBeDefined();
    });
    await waitFor(() => {
      expect(defaultPrototypeScrollTo).toHaveBeenCalledWith({
        top: 1200,
        behavior: 'smooth',
      });
    });
  });

  it('tracks the latest user message at or above the focus line on chat scroll', async () => {
    const conversation = makeConversation('navigator-3', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
      messages: [
        {
          role: 'user',
          content: 'First question',
          timestamp: '2026-07-29T12:20:00.000Z',
          message_id: 'msg-user-1',
        },
        {
          role: 'assistant',
          content: 'First answer',
          timestamp: '2026-07-29T12:20:01.000Z',
          message_id: 'msg-assistant-1',
        },
        {
          role: 'user',
          content: 'Second question',
          timestamp: '2026-07-29T12:20:02.000Z',
          message_id: 'msg-user-2',
        },
        {
          role: 'assistant',
          content: 'Second answer',
          timestamp: '2026-07-29T12:20:03.000Z',
          message_id: 'msg-assistant-2',
        },
        {
          role: 'user',
          content: 'Third question',
          timestamp: '2026-07-29T12:20:04.000Z',
          message_id: 'msg-user-3',
        },
      ],
    });

    const requestAnimationFrameSpy = vi
      .spyOn(window, 'requestAnimationFrame')
      .mockImplementation((callback: FrameRequestCallback) => {
        callback(0);
        return 1;
      });
    const cancelAnimationFrameSpy = vi
      .spyOn(window, 'cancelAnimationFrame')
      .mockImplementation(() => undefined);

    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    const messagesRoot = await waitFor(() => {
      const element = document.querySelector('.chat-messages') as HTMLElement | null;
      expect(element).toBeTruthy();
      return element as HTMLElement;
    });

    let scrollTop = 0;
    Object.defineProperty(messagesRoot, 'scrollTop', {
      configurable: true,
      get: () => scrollTop,
      set: (value: number) => {
        scrollTop = value;
      },
    });
    Object.defineProperty(messagesRoot, 'clientHeight', {
      configurable: true,
      value: 400,
    });
    Object.defineProperty(messagesRoot, 'scrollHeight', {
      configurable: true,
      value: 1200,
    });
    messagesRoot.getBoundingClientRect = () =>
      ({
        top: 0,
        bottom: 400,
        left: 0,
        right: 300,
        width: 300,
        height: 400,
        x: 0,
        y: 0,
        toJSON: () => ({}),
      }) as DOMRect;

    const userWrappers = Array.from(
      document.querySelectorAll('.chat-branch-wrapper-user'),
    ) as HTMLElement[];
    const wrapperTops = [20, 180, 340];
    userWrappers.forEach((wrapper, index) => {
      wrapper.getBoundingClientRect = () =>
        ({
          top: wrapperTops[index],
          bottom: wrapperTops[index] + 60,
          left: 0,
          right: 300,
          width: 300,
          height: 60,
          x: 0,
          y: wrapperTops[index],
          toJSON: () => ({}),
        }) as DOMRect;
    });

    fireEvent.scroll(messagesRoot);

    await waitFor(() => {
      expect(screen.getByLabelText('User message navigation').getAttribute('data-active-key')).toBe(
        'msg-user-1',
      );
    });

    expect(requestAnimationFrameSpy).toHaveBeenCalled();
    expect(cancelAnimationFrameSpy).not.toHaveBeenCalled();
  });

  it('does not render the navigator before there are two persisted user entries', async () => {
    const conversation = makeConversation('navigator-4', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
      messages: [
        {
          role: 'user',
          content: 'Only user question',
          timestamp: '2026-07-29T12:30:00.000Z',
          message_id: 'msg-user-only',
        },
        {
          role: 'assistant',
          content: 'Assistant answer',
          timestamp: '2026-07-29T12:30:01.000Z',
          message_id: 'msg-assistant-only',
        },
      ],
    });

    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
        embedded
      />,
    );

    await waitFor(() => {
      expect(screen.getByText('Only user question')).toBeDefined();
    });

    expect(screen.queryByLabelText('User message navigation')).toBeNull();
    expect(chatMessageNavigatorMock.renderSpy).not.toHaveBeenCalled();
  });
});

describe('ChatPanel resize and mobile sidebar integration', () => {
  it('restores the messages region from the keyboard when the composer is maximized', async () => {
    setChatLayoutCookie(currentUser.id, {
      showSidebar: true,
      sidebarWidth: 280,
      inputAreaHeight: 160,
      isInputAreaCollapsed: false,
      isMessagesCollapsed: true,
    });

    const conversation = makeConversation('layout-restore-1', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
    });

    renderChatPanel(
      <ChatPanel
        currentUser={currentUser}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
      />,
    );

    const separator = await screen.findByRole('separator', { name: 'Restore chat messages' });
    expect(document.getElementById('chat-workbench-main')).toBeNull();

    fireEvent.keyDown(separator, { key: 'Enter' });

    await waitFor(() => {
      expect(
        screen.getByRole('separator', { name: 'Resize chat messages and composer' }),
      ).toBeDefined();
      expect(document.getElementById('chat-workbench-main')).toBeTruthy();
    });
  });

  it('exposes a stable mobile sidebar restore control and reopens the overlay sidebar', async () => {
    const matchMediaSpy = vi.fn().mockImplementation(() => ({
      matches: true,
      media: '(max-width: 768px)',
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn(),
    }));
    window.matchMedia = matchMediaSpy;

    const conversation = makeConversation('mobile-sidebar-1', 'Assistant reply', {
      workspace_id: 'ws-1',
      active_task_id: null,
    });

    renderChatPanel(
      <ChatPanel
        currentUser={{ ...currentUser, role: 'admin' }}
        workspaceId="ws-1"
        workspaceChatState={{ ...makeWorkspaceChatState(conversation), active_task: null }}
        workspaceAvailableTools={[]}
        workspaceSelectedToolIds={[]}
      />,
    );

    const toggle = await screen.findByRole('button', { name: 'Open chat sidebar' });
    expect(toggle.id).toBe('chat-mobile-sidebar-toggle');
    expect(document.getElementById('chat-workbench-sidebar')).toBeNull();

    fireEvent.click(toggle);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Close chat sidebar' })).toBeDefined();
      expect(document.getElementById('chat-workbench-sidebar')).toBeTruthy();
      expect(
        document.getElementById('chat-mobile-sidebar-toggle')?.closest('.chat-sidebar'),
      ).toBeNull();
    });
  });
});

describe('ToolCallDisplay create_html_component rendering', () => {
  afterEach(() => {
    cleanup();
  });

  const validEnvelope = JSON.stringify({
    __html_component__: true,
    title: 'Shipments by origin',
    html: '<!doctype html><html><head></head><body><div id="map"></div></body></html>',
    data: {
      columns: ['lat', 'lng', 'shipments'],
      rows: [{ lat: 31.9, lng: -99.9, shipments: 412 }],
      row_count: 1,
    },
    description: 'Shipments by origin state, last 30 days',
    height: 480,
    data_connection: null,
  });

  it('renders a valid envelope inline through HtmlComponentDisplay with export controls', () => {
    const toolCall: ActiveToolCall = {
      tool: 'create_html_component',
      status: 'complete',
      input: { title: 'Shipments by origin' },
      output: validEnvelope,
    };

    const { container } = render(<ToolCallDisplay toolCall={toolCall} defaultExpanded />);

    const wrapper = container.querySelector('.tool-call.tool-call-html-component');
    expect(wrapper).not.toBeNull();
    expect(wrapper?.classList.contains('tool-call-complete')).toBe(true);
    const stub = screen.getByTestId('html-component-stub');
    expect(stub.textContent).toContain('Shipments by origin');
    expect(stub.textContent).toContain('Shipments by origin state, last 30 days');
    expect(stub.querySelector('.chart-description')).toBeNull();
    // Tabular `data` exposes the export menu inside the injected anchor.
    expect(stub.querySelector('.viz-version-anchor')).not.toBeNull();
    expect(container.querySelector('.tool-call-failed')).toBeNull();
    expect(container.querySelector('.tool-call-retry-btn')).toBeNull();
  });

  it('hides the export menu when component data is not tabular', () => {
    const toolCall: ActiveToolCall = {
      tool: 'create_html_component',
      status: 'complete',
      output: JSON.stringify({
        __html_component__: true,
        title: 'Gauge',
        html: '<html><body>gauge</body></html>',
        data: { threshold: 42 },
      }),
    };

    const { container } = render(<ToolCallDisplay toolCall={toolCall} defaultExpanded />);

    expect(container.querySelector('.tool-call-html-component')).not.toBeNull();
    expect(
      screen.getByTestId('html-component-stub').querySelector('.viz-version-anchor'),
    ).toBeNull();
  });

  it('marks malformed output as failed without offering the visualization retry button', () => {
    const toolCall: ActiveToolCall = {
      tool: 'create_html_component',
      status: 'complete',
      input: { title: 'Broken' },
      output: JSON.stringify({ __html_component__: true, title: 'Broken', html: 42 }),
    };

    const { container } = render(
      <ToolCallDisplay toolCall={toolCall} conversationId="conv-1" allowRerun defaultExpanded />,
    );

    expect(container.querySelector('.tool-call-html-component')).toBeNull();
    expect(screen.queryByTestId('html-component-stub')).toBeNull();
    expect(container.querySelector('.tool-call.tool-call-failed')).not.toBeNull();
    expect(container.querySelector('.tool-call-error-icon')).not.toBeNull();
    expect(container.querySelector('.tool-call-retry-btn')).toBeNull();
  });

  it('still offers the retry button for malformed chart output', () => {
    const toolCall: ActiveToolCall = {
      tool: 'create_chart',
      status: 'complete',
      output: JSON.stringify({ __chart__: true }),
    };

    const { container } = render(
      <ToolCallDisplay toolCall={toolCall} conversationId="conv-1" allowRerun defaultExpanded />,
    );

    expect(container.querySelector('.tool-call.tool-call-failed')).not.toBeNull();
    expect(container.querySelector('.tool-call-retry-btn')).not.toBeNull();
  });
});
