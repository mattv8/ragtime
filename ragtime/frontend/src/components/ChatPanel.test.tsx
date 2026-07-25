import { cleanup, render, screen, waitFor } from '@testing-library/react';
import type { ReactElement } from 'react';
import userEvent from '@testing-library/user-event';
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

vi.mock('@/api', () => ({ api: apiMock }));

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

window.HTMLElement.prototype.scrollIntoView = vi.fn();
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

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
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
