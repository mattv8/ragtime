import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import type { ReactElement, ReactNode } from 'react';
import userEvent from '@testing-library/user-event';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { Conversation, ConversationSummary, User, WorkspaceChatStateResponse } from '@/types';
import { AvailableModelsProvider } from '@/contexts/AvailableModelsContext';
import {
  ChatPanel,
  ToolCallDisplay,
  applyConversationToolGroupWriteToggle,
  getConversationToolGroupWriteMenuItem,
  isToolEffectivelyWritableForConversation,
  type ActiveToolCall,
} from './ChatPanel';
import type { ChatMessageNavigationEntry } from './ChatMessageNavigator';

const apiMock = vi.hoisted(() => ({
  getConversation: vi.fn().mockResolvedValue(null),
  getConversationBranchPoints: vi.fn().mockResolvedValue([]),
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
}));

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
    tool_output_mode: 'show',
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
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
      /const getToolGroupMenuItems = useCallback\([\s\S]*?\[\s*activeConversation,\s*conversationToolOptions,\s*isConversationViewer,\s*saveConversationToolOptions,\s*savingTools,\s*\]/,
    );
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
