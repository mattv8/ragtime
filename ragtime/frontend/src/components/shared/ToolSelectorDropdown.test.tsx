import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useMemo, useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ToolSelectorDropdown,
  type ToolSelectorFocusRequest,
  type ToolSelectorTool,
} from './ToolSelectorDropdown';
import {
  getEffectiveUserSpaceToolIdSet,
  type UserSpaceToolSelection,
} from '@/utils/userSpaceTools';

afterEach(() => {
  cleanup();
});

const groupedTool: ToolSelectorTool = {
  id: 'tool-grouped',
  name: 'Grouped Tool',
  tool_type: 'ssh_shell',
  group_id: 'group-1',
  group_name: 'Alpha Group',
  available: true,
};

const ungroupedTool: ToolSelectorTool = {
  id: 'tool-ungrouped',
  name: 'Ungrouped Tool',
  tool_type: 'postgres',
  available: true,
};

const builtInTool: ToolSelectorTool = {
  id: 'builtin-web-search',
  name: 'Web Search',
  tool_type: 'built-in',
};

function expectCheckbox(name: string | RegExp, checked: boolean) {
  expect((screen.getByRole('checkbox', { name }) as HTMLInputElement).checked).toBe(checked);
}

interface ControlledDropdownProps {
  availableTools?: ToolSelectorTool[];
  builtInTools?: ToolSelectorTool[];
  initialSelection?: UserSpaceToolSelection;
  initialBuiltInToolIds?: string[];
  focusRequest?: ToolSelectorFocusRequest | null;
  onRequestEnableWorkspaceTool?: (toolId: string) => void;
}

function ControlledDropdown({
  availableTools: availableToolsProp = [groupedTool, ungroupedTool],
  builtInTools = [builtInTool],
  initialSelection = {
    mode: 'default_all',
    toolIds: [],
    toolGroupIds: [],
  },
  initialBuiltInToolIds = [builtInTool.id],
  focusRequest = null,
  onRequestEnableWorkspaceTool,
}: ControlledDropdownProps) {
  const availableTools = useMemo(() => availableToolsProp, [availableToolsProp]);
  const [selection, setSelection] = useState<UserSpaceToolSelection>({
    ...initialSelection,
  });
  const [selectedBuiltInToolIds, setSelectedBuiltInToolIds] = useState(
    () => new Set(initialBuiltInToolIds),
  );
  const effectiveSelectedToolIds = useMemo(
    () => getEffectiveUserSpaceToolIdSet(selection, availableTools),
    [availableTools, selection],
  );

  return (
    <>
      <ToolSelectorDropdown
        availableTools={availableTools}
        selectedToolIds={effectiveSelectedToolIds}
        toolSelectionMode={selection.mode}
        selectedToolGroupIds={new Set(selection.toolGroupIds)}
        onSelectionChange={setSelection}
        builtInTools={builtInTools}
        selectedBuiltInToolIds={selectedBuiltInToolIds}
        onToggleBuiltInTool={(toolId) => {
          setSelectedBuiltInToolIds((previous) => {
            const next = new Set(previous);
            if (next.has(toolId)) {
              next.delete(toolId);
            } else {
              next.add(toolId);
            }
            return next;
          });
        }}
        onBulkBuiltInToggle={(selected) => {
          setSelectedBuiltInToolIds(
            selected ? new Set(builtInTools.map((tool) => tool.id)) : new Set(),
          );
        }}
        toolGroups={[{ id: 'group-1', name: 'Alpha Group' }]}
        title="Conversation Tools"
        focusRequest={focusRequest}
        onRequestEnableWorkspaceTool={onRequestEnableWorkspaceTool}
      />
      <output data-testid="selection-state">{JSON.stringify(selection)}</output>
    </>
  );
}

describe('ToolSelectorDropdown bulk selection', () => {
  it('deselects configured tools before built-in tools, then reselects all presented tools', async () => {
    const user = userEvent.setup();
    render(<ControlledDropdown />);

    await user.click(screen.getByTitle('Conversation Tools (3/3 selected)'));

    await user.click(screen.getByRole('button', { name: 'Deselect all' }));
    expectCheckbox('Select all tools in Alpha Group', false);
    expectCheckbox(/Ungrouped Tool/, false);
    expectCheckbox('Web Search', true);
    expect(screen.getByRole('button', { name: 'Deselect all' })).toBeDefined();

    await user.click(screen.getByRole('button', { name: 'Deselect all' }));
    expectCheckbox('Web Search', false);
    expect(screen.getByRole('button', { name: 'Select all' })).toBeDefined();

    await user.click(screen.getByRole('button', { name: 'Select all' }));
    expectCheckbox('Select all tools in Alpha Group', true);
    expectCheckbox(/Ungrouped Tool/, true);
    expectCheckbox('Web Search', true);
  });

  it('falls back to a two-step cycle when only built-in tools are presented', async () => {
    const user = userEvent.setup();
    render(<ControlledDropdown availableTools={[]} />);

    await user.click(screen.getByTitle('Conversation Tools (1/1 selected)'));

    await user.click(screen.getByRole('button', { name: 'Deselect all' }));
    expectCheckbox('Web Search', false);
    expect(screen.getByRole('button', { name: 'Select all' })).toBeDefined();

    await user.click(screen.getByRole('button', { name: 'Select all' }));
    expectCheckbox('Web Search', true);
  });

  it('falls back to a two-step cycle when only configured tools are presented', async () => {
    const user = userEvent.setup();
    render(<ControlledDropdown builtInTools={[]} initialBuiltInToolIds={[]} />);

    await user.click(screen.getByTitle('Conversation Tools (2/2 selected)'));

    await user.click(screen.getByRole('button', { name: 'Deselect all' }));
    expectCheckbox('Select all tools in Alpha Group', false);
    expectCheckbox(/Ungrouped Tool/, false);
    expect(screen.getByRole('button', { name: 'Select all' })).toBeDefined();

    await user.click(screen.getByRole('button', { name: 'Select all' }));
    expectCheckbox('Select all tools in Alpha Group', true);
    expectCheckbox(/Ungrouped Tool/, true);
  });

  it('selects all presented tools from a partial custom selection', async () => {
    const user = userEvent.setup();
    render(
      <ControlledDropdown
        initialSelection={{
          mode: 'custom',
          toolIds: [ungroupedTool.id],
          toolGroupIds: [],
        }}
        initialBuiltInToolIds={[]}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (1/3 selected)'));

    await user.click(screen.getByRole('button', { name: 'Select all' }));
    expectCheckbox('Select all tools in Alpha Group', true);
    expectCheckbox(/Ungrouped Tool/, true);
    expectCheckbox('Web Search', true);
  });

  it('does not preserve out-of-scope hidden tool ids when selecting visible tools', async () => {
    const user = userEvent.setup();
    render(
      <ControlledDropdown
        availableTools={[groupedTool]}
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        initialSelection={{
          mode: 'custom',
          toolIds: ['tool-hidden'],
          toolGroupIds: [],
        }}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (0/1 selected)'));
    expect(screen.queryByText('tool-hidden')).toBeNull();

    await user.click(screen.getByRole('button', { name: 'Select all' }));

    expect(screen.getByTestId('selection-state').textContent).toBe(
      JSON.stringify({ mode: 'default_all', toolIds: [], toolGroupIds: [] }),
    );
  });

  it('shows unavailable configured tools with the disabled explanation in the status row only', async () => {
    const user = userEvent.setup();
    render(
      <ControlledDropdown
        availableTools={[
          groupedTool,
          {
            ...ungroupedTool,
            available: false,
            disabled_reason: 'Disabled in Workspace',
          },
        ]}
        builtInTools={[]}
        initialBuiltInToolIds={[]}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (1/1 selected)'));

    const disabledCheckbox = screen.getByRole('checkbox', {
      name: /Ungrouped Tool/,
    }) as HTMLInputElement;
    expect(disabledCheckbox.disabled).toBe(true);
    expect(screen.getByText('postgres - Disabled in Workspace')).toBeDefined();

    await user.hover(screen.getByText('Ungrouped Tool'));

    expect(screen.queryByRole('tooltip')).toBeNull();
  });

  it('requests the workspace tools menu when the disabled enable link is clicked', async () => {
    const user = userEvent.setup();
    const onRequestEnableWorkspaceTool = vi.fn();
    render(
      <ControlledDropdown
        availableTools={[
          groupedTool,
          {
            ...ungroupedTool,
            available: false,
            disabled_reason: 'Disabled in Workspace',
          },
        ]}
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        onRequestEnableWorkspaceTool={onRequestEnableWorkspaceTool}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (1/1 selected)'));
    await user.click(
      screen.getByRole('button', { name: 'Enable Ungrouped Tool in Workspace Tools' }),
    );

    expect(onRequestEnableWorkspaceTool).toHaveBeenCalledWith(ungroupedTool.id);
  });

  it('opens the menu and highlights a requested workspace tool', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ControlledDropdown builtInTools={[]} initialBuiltInToolIds={[]} />,
    );

    expect(screen.queryByText('Grouped Tool')).toBeNull();

    rerender(
      <ControlledDropdown
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        focusRequest={{ toolId: groupedTool.id, requestId: 1 }}
      />,
    );

    const row = screen.getByText('Grouped Tool').closest('.userspace-tool-item');
    await waitFor(() => expect(row?.classList.contains('highlight-setting')).toBe(true));

    await user.click(screen.getByTitle('Conversation Tools (2/2 selected)'));
    expect(screen.queryByText('Grouped Tool')).toBeNull();
  });

  it('clears search before highlighting a requested workspace tool', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <ControlledDropdown builtInTools={[]} initialBuiltInToolIds={[]} />,
    );

    await user.click(screen.getByTitle('Conversation Tools (2/2 selected)'));
    await user.type(screen.getByRole('textbox', { name: 'Filter tools' }), 'Ungrouped');
    expect(screen.queryByText('Grouped Tool')).toBeNull();

    rerender(
      <ControlledDropdown
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        focusRequest={{ toolId: groupedTool.id, requestId: 1 }}
      />,
    );

    const row = await screen.findByText('Grouped Tool');
    await waitFor(() =>
      expect(row.closest('.userspace-tool-item')?.classList.contains('highlight-setting')).toBe(
        true,
      ),
    );
  });

  it('allows outside clicks to close a focused menu before the target row is processed', async () => {
    const user = userEvent.setup();
    const { rerender } = render(
      <>
        <ControlledDropdown builtInTools={[]} initialBuiltInToolIds={[]} />
        <button type="button">Outside target</button>
      </>,
    );

    await user.click(screen.getByTitle('Conversation Tools (2/2 selected)'));
    await user.type(screen.getByRole('textbox', { name: 'Filter tools' }), 'Ungrouped');

    rerender(
      <>
        <ControlledDropdown
          builtInTools={[]}
          initialBuiltInToolIds={[]}
          focusRequest={{ toolId: groupedTool.id, requestId: 1 }}
        />
        <button type="button">Outside target</button>
      </>,
    );

    await user.click(screen.getByRole('button', { name: 'Outside target' }));

    await waitFor(() => expect(screen.queryByText('Conversation Tools')).toBeNull());
  });

  it('places the access badge in the same title row as the tool name', async () => {
    const user = userEvent.setup();
    render(
      <ControlledDropdown
        availableTools={[{ ...ungroupedTool, allow_write: true }]}
        builtInTools={[]}
        initialBuiltInToolIds={[]}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (1/1 selected)'));

    const titleRow = screen.getByText('Ungrouped Tool').closest('.userspace-tool-item-title-row');
    expect(titleRow).not.toBeNull();
    expect(titleRow?.textContent).toContain('Write');
  });
});
