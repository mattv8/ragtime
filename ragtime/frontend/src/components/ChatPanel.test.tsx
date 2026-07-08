import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ToolCallDisplay, type ActiveToolCall } from './ChatPanel';

afterEach(() => {
  cleanup();
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
