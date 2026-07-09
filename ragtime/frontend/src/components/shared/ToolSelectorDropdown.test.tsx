import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useMemo, useState } from 'react';
import { afterEach, describe, expect, it } from 'vitest';

import { ToolSelectorDropdown, type ToolSelectorTool } from './ToolSelectorDropdown';
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
    />
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
});
