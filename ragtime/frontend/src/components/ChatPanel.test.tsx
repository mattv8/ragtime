import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { api } from '@/api';
import type { Conversation, ConversationSummary } from '@/types';
import { ToolCallDisplay, type ActiveToolCall } from './ChatPanel';

vi.mock('@/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/api')>();
  return {
    api: {
      ...actual.api,
      getConversation: vi.fn().mockResolvedValue(null),
      getSubagentConversationSummaries: vi.fn().mockResolvedValue([] as ConversationSummary[]),
      streamChatTask: vi.fn().mockReturnValue(
        (async function* () {
          return;
        })(),
      ),
    },
  };
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
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
});

describe('ToolCallDisplay truncated subagent recovery', () => {
  function makeConversation(id: string, content: string): Conversation {
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
    };
  }

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

    (api as any).getSubagentConversationSummaries.mockResolvedValue([summary]);
    (api as any).getConversation.mockResolvedValue(child);

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
      expect((api as any).getSubagentConversationSummaries).toHaveBeenCalledWith(
        'parent-1',
        'ws-1',
      );
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

    (api as any).getSubagentConversationSummaries.mockResolvedValue(duplicateSummaries);
    (api as any).getConversation.mockImplementation((conversationId: string) =>
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
    expect((api as any).getConversation).toHaveBeenCalledWith('second-child', 'ws-1');
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

    (api as any).getSubagentConversationSummaries.mockResolvedValue([]);
    (api as any).getConversation.mockResolvedValue(null);

    render(<ToolCallDisplay toolCall={toolCall} defaultExpanded workspaceId="ws-1" />);

    await user.click(screen.getByLabelText('Expand Types update subagent transcript'));

    expect(
      screen.getByText('Subagent transcript is unavailable for this archived tool result.'),
    ).toBeDefined();
    expect(screen.queryByText('Loading subagent transcript...')).toBeNull();
    expect((api as any).getSubagentConversationSummaries).not.toHaveBeenCalled();
  });
});
