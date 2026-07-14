import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useMemo, useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ToolSelectorDropdown,
  type ToolSelectorFocusRequest,
  type ToolSelectorStatusBadgeContext,
  type ToolSelectorMenuItem,
  type ToolSelectorStatusBadge,
  type ToolSelectorTool,
} from './ToolSelectorDropdown';
import {
  getEffectiveUserSpaceToolIdSet,
  type UserSpaceToolSelection,
} from '@/utils/userSpaceTools';

afterEach(() => {
  cleanup();
  vi.useRealTimers();
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

const disabledWriteAccessDescription =
  'Ask an admin to enable Allow Write Operations for this tool.';

function expectCheckbox(name: string | RegExp, checked: boolean) {
  expect((screen.getByRole('checkbox', { name }) as HTMLInputElement).checked).toBe(checked);
}

function renderDropdownWithDisabledContextMenu(onChange = vi.fn()) {
  render(
    <ControlledDropdown
      builtInTools={[]}
      initialBuiltInToolIds={[]}
      getToolMenuItems={() => [
        {
          label: 'Write access unavailable',
          description: disabledWriteAccessDescription,
          checked: false,
          disabled: true,
          onChange,
        },
      ]}
    />,
  );

  return { onChange };
}

async function openUngroupedToolContextMenu(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTitle('Conversation Tools (2/2 selected)'));
  fireEvent.contextMenu(screen.getByText('Ungrouped Tool').closest('.userspace-tool-item')!);
}

interface ControlledDropdownProps {
  availableTools?: ToolSelectorTool[];
  builtInTools?: ToolSelectorTool[];
  initialSelection?: UserSpaceToolSelection;
  initialBuiltInToolIds?: string[];
  focusRequest?: ToolSelectorFocusRequest | null;
  onRequestEnableWorkspaceTool?: (toolId: string) => void;
  getToolMenuItems?: (tool: ToolSelectorTool) => ToolSelectorMenuItem[];
  getToolStatusBadge?: (
    tool: ToolSelectorTool,
    context: ToolSelectorStatusBadgeContext,
  ) => ToolSelectorStatusBadge | null;
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
  getToolMenuItems,
  getToolStatusBadge,
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
        getToolMenuItems={getToolMenuItems}
        getToolStatusBadge={getToolStatusBadge}
      />
      <output data-testid="selection-state">{JSON.stringify(selection)}</output>
    </>
  );
}

describe('ToolSelectorDropdown bulk selection', () => {
  it.each([
    {
      label: 'deselects configured tools before built-in tools, then reselects all presented tools',
      props: {},
      triggerTitle: 'Conversation Tools (3/3 selected)',
      firstAction: 'Deselect all',
      firstExpected: [
        ['Select all tools in Alpha Group', false],
        [/Ungrouped Tool/, false],
        ['Web Search', true],
      ] as const,
      secondAction: 'Deselect all',
      secondExpected: [['Web Search', false]] as const,
      finalAction: 'Select all',
      finalExpected: [
        ['Select all tools in Alpha Group', true],
        [/Ungrouped Tool/, true],
        ['Web Search', true],
      ] as const,
    },
    {
      label: 'falls back to a two-step cycle when only built-in tools are presented',
      props: { availableTools: [] },
      triggerTitle: 'Conversation Tools (1/1 selected)',
      firstAction: 'Deselect all',
      firstExpected: [['Web Search', false]] as const,
      finalAction: 'Select all',
      finalExpected: [['Web Search', true]] as const,
    },
    {
      label: 'falls back to a two-step cycle when only configured tools are presented',
      props: { builtInTools: [], initialBuiltInToolIds: [] },
      triggerTitle: 'Conversation Tools (2/2 selected)',
      firstAction: 'Deselect all',
      firstExpected: [
        ['Select all tools in Alpha Group', false],
        [/Ungrouped Tool/, false],
      ] as const,
      finalAction: 'Select all',
      finalExpected: [
        ['Select all tools in Alpha Group', true],
        [/Ungrouped Tool/, true],
      ] as const,
    },
    {
      label: 'selects all presented tools from a partial custom selection',
      props: {
        initialSelection: {
          mode: 'custom' as const,
          toolIds: [ungroupedTool.id],
          toolGroupIds: [],
        },
        initialBuiltInToolIds: [],
      },
      triggerTitle: 'Conversation Tools (1/3 selected)',
      firstAction: 'Select all',
      firstExpected: [
        ['Select all tools in Alpha Group', true],
        [/Ungrouped Tool/, true],
        ['Web Search', true],
      ] as const,
    },
  ])(
    '$label',
    async ({
      props,
      triggerTitle,
      firstAction,
      firstExpected,
      secondAction,
      secondExpected,
      finalAction,
      finalExpected,
    }) => {
      const user = userEvent.setup();
      render(<ControlledDropdown {...props} />);

      await user.click(screen.getByTitle(triggerTitle));

      await user.click(screen.getByRole('button', { name: firstAction }));
      firstExpected.forEach(([name, checked]) => expectCheckbox(name, checked));

      if (secondAction && secondExpected) {
        expect(screen.getByRole('button', { name: secondAction })).toBeDefined();
        await user.click(screen.getByRole('button', { name: secondAction }));
        secondExpected.forEach(([name, checked]) => expectCheckbox(name, checked));
      }

      if (finalAction && finalExpected) {
        expect(screen.getByRole('button', { name: finalAction })).toBeDefined();
        await user.click(screen.getByRole('button', { name: finalAction }));
        finalExpected.forEach(([name, checked]) => expectCheckbox(name, checked));
      }
    },
  );

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

  it.each([
    {
      label: 'renders a workspace-scope write badge without the global icon',
      props: {
        availableTools: [{ ...ungroupedTool, allow_write: true }],
        builtInTools: [],
        initialBuiltInToolIds: [],
        getToolStatusBadge: () => ({
          label: 'Write',
          tone: 'write' as const,
          scope: 'workspace' as const,
          title: 'Write access enabled for this workspace',
        }),
      },
      triggerTitle: 'Conversation Tools (1/1 selected)',
      expectedTitle: 'Write access enabled for this workspace',
      expectedLabel: 'Write',
      expectWorkspaceClass: true,
      expectIcon: false,
    },
    {
      label: 'renders a read-only status badge label from getToolStatusBadge',
      props: {
        availableTools: [ungroupedTool],
        builtInTools: [],
        initialBuiltInToolIds: [],
        getToolStatusBadge: () => ({
          label: 'Read only',
          tone: 'read' as const,
          title: 'Write access unavailable',
        }),
      },
      triggerTitle: 'Conversation Tools (1/1 selected)',
      expectedTitle: 'Write access unavailable',
      expectedLabel: 'Read only',
      expectWorkspaceClass: false,
      expectIcon: false,
    },
    {
      label: 'omits status badges when getToolStatusBadge returns null',
      props: {
        availableTools: [ungroupedTool],
        builtInTools: [],
        initialBuiltInToolIds: [],
        getToolStatusBadge: () => null,
      },
      triggerTitle: 'Conversation Tools (1/1 selected)',
      expectedLabel: null,
    },
  ])(
    '$label',
    async ({
      props,
      triggerTitle,
      expectedTitle,
      expectedLabel,
      expectWorkspaceClass,
      expectIcon,
    }) => {
      const user = userEvent.setup();
      render(<ControlledDropdown {...props} />);

      await user.click(screen.getByTitle(triggerTitle));

      const titleRow = screen.getByText('Ungrouped Tool').closest('.userspace-tool-item-title-row');
      if (expectedLabel === null) {
        expect(titleRow?.querySelector('.userspace-tool-status-badge')).toBeNull();
        expect(titleRow?.textContent).toBe('Ungrouped Tool');
        return;
      }

      expect(titleRow?.textContent).toContain(expectedLabel);
      const badge = screen.getByTitle(expectedTitle!);
      expect(badge.classList.contains('userspace-tool-status-badge-workspace')).toBe(
        expectWorkspaceClass,
      );
      if (expectIcon) {
        expect(badge.querySelector('svg')).not.toBeNull();
      } else {
        expect(badge.querySelector('svg')).toBeNull();
      }
    },
  );

  it('lets status badges depend on selected workspace tool state', async () => {
    const user = userEvent.setup();
    const otherTool: ToolSelectorTool = {
      id: 'tool-other',
      name: 'Other Tool',
      tool_type: 'postgres',
      available: true,
    };
    render(
      <ControlledDropdown
        availableTools={[ungroupedTool, otherTool]}
        initialSelection={{ mode: 'custom', toolIds: [ungroupedTool.id], toolGroupIds: [] }}
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        getToolStatusBadge={(_tool, context) =>
          context.selected
            ? {
                label: 'WORKSPACE READ',
                tone: 'read',
                scope: 'workspace',
                title: 'Selected for this workspace with read access',
              }
            : null
        }
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (1/2 selected)'));

    const selectedTitleRow = screen
      .getByText('Ungrouped Tool')
      .closest('.userspace-tool-item-title-row');
    const unselectedTitleRow = screen
      .getByText('Other Tool')
      .closest('.userspace-tool-item-title-row');
    expect(selectedTitleRow?.textContent).toContain('WORKSPACE READ');
    expect(unselectedTitleRow?.querySelector('.userspace-tool-status-badge')).toBeNull();
  });

  it('omits status badges for unavailable workspace tools', async () => {
    const user = userEvent.setup();
    render(
      <ControlledDropdown
        availableTools={[
          { ...ungroupedTool, available: false, disabled_reason: 'No recent heartbeat' },
        ]}
        initialSelection={{ mode: 'custom', toolIds: [ungroupedTool.id], toolGroupIds: [] }}
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        getToolStatusBadge={(_tool, context) =>
          context.available
            ? {
                label: 'WORKSPACE READ',
                tone: 'read',
                scope: 'workspace',
              }
            : null
        }
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (0/0 selected)'));

    const titleRow = screen.getByText('Ungrouped Tool').closest('.userspace-tool-item-title-row');
    expect(titleRow?.querySelector('.userspace-tool-status-badge')).toBeNull();
  });

  it('shows a disabled context-menu item and does not fire onChange when clicked', async () => {
    const user = userEvent.setup();
    const { onChange } = renderDropdownWithDisabledContextMenu();

    await openUngroupedToolContextMenu(user);

    const disabledItem = screen.getByText('Write access unavailable').closest('[role="button"]');
    expect(disabledItem?.classList.contains('disabled')).toBe(true);
    expect(
      (disabledItem?.querySelector('input[type="checkbox"]') as HTMLInputElement | null)?.disabled,
    ).toBe(true);

    await user.click(screen.getByText('Write access unavailable'));

    expect(onChange).not.toHaveBeenCalled();
  });

  it('dismisses the context menu when clicking outside it', async () => {
    const user = userEvent.setup();
    renderDropdownWithDisabledContextMenu();

    await openUngroupedToolContextMenu(user);
    expect(screen.getByText('Write access unavailable')).toBeTruthy();

    await user.click(document.body);

    expect(screen.queryByText('Write access unavailable')).toBeNull();
  });

  it('dismisses the context menu when clicking inside the dropdown but outside the menu', async () => {
    const user = userEvent.setup();
    renderDropdownWithDisabledContextMenu();

    await openUngroupedToolContextMenu(user);
    expect(screen.getByText('Write access unavailable')).toBeTruthy();

    await user.click(screen.getByLabelText('Filter tools'));

    expect(screen.queryByText('Write access unavailable')).toBeNull();
  });

  it('dismisses the context menu when focus leaves the window', async () => {
    const user = userEvent.setup();
    renderDropdownWithDisabledContextMenu();

    await openUngroupedToolContextMenu(user);
    expect(screen.getByText('Write access unavailable')).toBeTruthy();

    fireEvent.blur(window);

    expect(screen.queryByText('Write access unavailable')).toBeNull();
  });

  it('renders a link inside a disabled context-menu description and still lets the link handle clicks', async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    render(
      <ControlledDropdown
        builtInTools={[]}
        initialBuiltInToolIds={[]}
        getToolMenuItems={(tool) => [
          {
            label: 'Write access unavailable',
            description: (
              <>
                Ask an admin to enable "Allow Write Operations" for this tool in{' '}
                <a
                  href="?view=tools"
                  className="btn-link"
                  onClick={(event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    onNavigate(`tool:${tool.id}`);
                  }}
                >
                  Settings &gt; Tools
                </a>
                .
              </>
            ),
            checked: false,
            disabled: true,
            onChange: vi.fn(),
          },
        ]}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (2/2 selected)'));
    fireEvent.contextMenu(screen.getByText('Ungrouped Tool').closest('.userspace-tool-item')!);

    const link = screen.getByRole('link', { name: 'Settings > Tools' });
    await user.click(link);

    expect(onNavigate).toHaveBeenCalledWith('tool:tool-ungrouped');
  });

  it('hints that tools support right-click above the pointer when hovering the open dropdown', async () => {
    vi.useFakeTimers();
    userEvent.setup({ advanceTimers: vi.advanceTimersByTimeAsync });
    render(
      <ControlledDropdown
        getToolMenuItems={() => [
          {
            label: 'Enable write access for this workspace',
            checked: false,
            onChange: vi.fn(),
          },
        ]}
      />,
    );

    fireEvent.click(screen.getByTitle('Conversation Tools (3/3 selected)'));
    const dropdown = document.querySelector('.userspace-tool-dropdown') as HTMLElement;
    dropdown.style.zIndex = '9000';
    fireEvent.mouseEnter(dropdown, { clientX: 300, clientY: 250 });
    fireEvent.mouseMove(dropdown, { clientX: 300, clientY: 250 });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(999);
    });
    expect(screen.queryByText('Right-click a tool for more options')).toBeNull();

    fireEvent.mouseMove(dropdown, { clientX: 302, clientY: 252 });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(999);
    });
    expect(screen.queryByText('Right-click a tool for more options')).toBeNull();
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1);
    });

    const tooltip = screen.getByText('Right-click a tool for more options');
    const popover = tooltip.closest('.popover') as HTMLElement;
    expect(popover.style.left).toBe('302px');
    expect(popover.style.top).toBe('236px');
    expect(popover.style.zIndex).toBe('9001');
  });

  it('does not hint at right-click when the menu has no right-click options', async () => {
    const user = userEvent.setup();
    render(<ControlledDropdown />);

    await user.click(screen.getByTitle('Conversation Tools (3/3 selected)'));
    await user.hover(document.querySelector('.userspace-tool-dropdown-surface') as HTMLElement);

    await new Promise((resolve) => setTimeout(resolve, 400));
    expect(screen.queryByText('Right-click a tool for more options')).toBeNull();
  });

  it('does not hint at right-click when hovering the trigger while the menu is open', async () => {
    const user = userEvent.setup();
    render(
      <ControlledDropdown
        getToolMenuItems={() => [
          {
            label: 'Enable write access for this workspace',
            checked: false,
            onChange: vi.fn(),
          },
        ]}
      />,
    );

    await user.click(screen.getByTitle('Conversation Tools (3/3 selected)'));
    await user.hover(screen.getByTitle('Conversation Tools (3/3 selected)'));

    await waitFor(() =>
      expect(screen.queryByText('Right-click a tool for more options')).toBeNull(),
    );
  });
});
