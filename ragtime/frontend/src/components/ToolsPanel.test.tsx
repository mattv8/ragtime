import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';

import { ToolsPanel } from './ToolsPanel';
import type { ToolConfig, ToolGroup, User } from '@/types';

import {
  apiMock,
  toastMock,
  toolFilterState,
  wizardState,
  groupedTool,
  ungroupedTool,
  pdmTool,
  pairedWindowsTool,
  linkedSolidworksTool,
  skillFixture,
  searchFilterBarMock,
  setupToolsPanelTest,
  teardownToolsPanelTest,
  toolGroup,
} from './ToolsPanel.testSupport';

type GroupApiMock = {
  createToolGroup: Mock;
  updateToolGroup: Mock;
  deleteToolGroup: Mock;
  reorderToolGroups: Mock;
};
const groupApiMock = apiMock as typeof apiMock & GroupApiMock;

const testUser = { id: 'user-1' } as User;

function clearCollapsedSectionCookies(): void {
  for (const entry of document.cookie.split('; ')) {
    const name = entry.split('=')[0];
    if (name.startsWith('tools_sections_collapsed_')) {
      document.cookie = `${name}=; path=/; max-age=0`;
    }
  }
}

function sectionBody(sectionId: string): HTMLElement {
  return document.getElementById(`tools-section-${sectionId}-body`)!;
}

async function setupGroupApis(): Promise<void> {
  Object.assign(apiMock, {
    createToolGroup: vi.fn(),
    updateToolGroup: vi.fn(),
    deleteToolGroup: vi.fn().mockResolvedValue(undefined),
    reorderToolGroups: vi.fn().mockResolvedValue(undefined),
  });
  const actual = await vi.importActual<typeof import('./shared/SearchFilterBar')>(
    './shared/SearchFilterBar',
  );
  // The shared mock always matches; filter tests need the real query matcher.
  Object.assign(searchFilterBarMock, {
    searchFilterTextMatchesQuery: actual.searchFilterTextMatchesQuery,
  });
}

async function setupTest(): Promise<void> {
  setupToolsPanelTest();
  clearCollapsedSectionCookies();
  await setupGroupApis();
}

vi.mock('@/api', async () => ({ api: (await import('./ToolsPanel.testSupport')).apiMock }));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: {
    getConfig: (await import('./ToolsPanel.testSupport')).contentProtectionMock.getConfig,
  },
  updateContentProtectionConfigSlice: (await import('./ToolsPanel.testSupport'))
    .contentProtectionMock.updateContentProtectionConfigSlice,
}));
vi.mock('./ToolWizard', async () => (await import('./ToolsPanel.testSupport')).toolWizardMock);
vi.mock(
  './MountSourceWizard',
  async () => (await import('./ToolsPanel.testSupport')).mountSourceWizardMock,
);
vi.mock(
  './InstructionSkillsDialog',
  async () => (await import('./ToolsPanel.testSupport')).instructionSkillsDialogMock,
);
vi.mock(
  './ToolAccessModal',
  async () => (await import('./ToolsPanel.testSupport')).toolAccessModalMock,
);
vi.mock('./Popover', async () => (await import('./ToolsPanel.testSupport')).popoverMock);
vi.mock(
  './AnimatedCreateButton',
  async () => (await import('./ToolsPanel.testSupport')).animatedCreateButtonMock,
);
vi.mock(
  './DeleteConfirmButton',
  async () => (await import('./ToolsPanel.testSupport')).deleteConfirmButtonMock,
);
vi.mock(
  './shared/Toast',
  async () => (await import('./ToolsPanel.testSupport')).toastComponentsMock,
);
vi.mock(
  './shared/SearchFilterBar',
  async () => (await import('./ToolsPanel.testSupport')).searchFilterBarMock,
);

describe('ToolsPanel', () => {
  beforeEach(setupTest);
  afterEach(() => {
    clearCollapsedSectionCookies();
    teardownToolsPanelTest();
  });

  it('renders the grouped and ungrouped tools in separate sections', async () => {
    render(<ToolsPanel />);
    await screen.findByText(groupedTool.name);
    const sections = [...document.querySelectorAll('[data-tool-section]')];
    expect(sections.map((section) => section.getAttribute('data-tool-section'))).toEqual([
      groupedTool.group_id,
      'ungrouped',
    ]);
  });

  it.each([
    ['healthy', [pdmTool], 'PDM: PDM Tool'],
    ['missing', [], 'PDM unavailable'],
    ['disabled', [{ ...pdmTool, enabled: false }], 'PDM unavailable'],
  ])('derives a %s link from loaded tools without API metadata', async (_state, targets, label) => {
    apiMock.listToolConfigs.mockResolvedValue([linkedSolidworksTool, ...targets]);
    render(<ToolsPanel />);
    await screen.findByRole('heading', { name: 'CAD workstation' });
    expect(document.querySelector('[data-solidworks-pdm-badge]')?.textContent).toBe(label);
  });

  it.each([
    [
      true,
      ['\\Designs', '\\Parts'],
      'Can add files, check out and check in in 2 approved vault folders',
    ],
    [true, [], 'No approved folders: changes are refused'],
    [false, ['\\Designs'], null],
  ])(
    'shows write badge for allow_write=%s and folders=%j',
    async (allowWrite, folders, content) => {
      apiMock.listToolConfigs.mockResolvedValue([
        {
          ...linkedSolidworksTool,
          allow_write: allowWrite,
          connection_config: {
            ...linkedSolidworksTool.connection_config,
            pdm_approved_folders: folders,
          },
        },
        pdmTool,
      ]);
      render(<ToolsPanel />);
      await screen.findByRole('heading', { name: 'CAD workstation' });
      const badge = document.querySelector('[data-solidworks-write-badge]');
      if (content === null) expect(badge).toBeNull();
      else {
        expect(badge?.textContent).toBe('Write');
        expect(badge?.parentElement?.getAttribute('data-popover-content')).toBe(content);
      }
    },
  );

  it.each([true, false])('offers the changes kill switch only when on=%s', async (allowWrite) => {
    apiMock.listToolConfigs.mockResolvedValue([
      { ...linkedSolidworksTool, allow_write: allowWrite },
      pdmTool,
    ]);
    apiMock.updateToolConfig.mockResolvedValue({ ...linkedSolidworksTool, allow_write: false });
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByRole('heading', { name: 'CAD workstation' }));
    const toggle = screen.queryByRole('button', { name: 'Allow changes' });
    if (!allowWrite) {
      expect(toggle).toBeNull();
      return;
    }
    expect(toggle?.getAttribute('aria-checked')).toBe('true');
    fireEvent.click(toggle!);
    await waitFor(() =>
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(linkedSolidworksTool.id, {
        allow_write: false,
      }),
    );
    expect(screen.queryByRole('heading', { name: 'Enable Write Access' })).toBeNull();
  });

  it.each([true, false])('shows Used by only for a linked PDM tool: %s', async (linked) => {
    apiMock.listToolConfigs.mockResolvedValue(linked ? [linkedSolidworksTool, pdmTool] : [pdmTool]);
    render(<ToolsPanel />);
    await screen.findByText('PDM Tool');
    const row = screen.queryByTestId(`pdm-linked-from-${pdmTool.id}`);
    if (linked) expect(row?.textContent).toBe('Used byCAD workstation');
    else expect(row).toBeNull();
  });

  it.each([true, false])(
    'confirms linked PDM card deletion with changes on=%s',
    async (allowWrite) => {
      apiMock.getSettings.mockResolvedValue({ settings: { show_tool_card_footer_actions: true } });
      apiMock.listToolConfigs.mockResolvedValue([
        { ...linkedSolidworksTool, allow_write: allowWrite },
        pdmTool,
      ]);
      render(<ToolsPanel />);
      await screen.findByText('PDM Tool');
      const card = document.getElementById(`tools-tool-${pdmTool.id}`)!;
      const deleteButton = card.querySelector('[title="Delete tool"]')!;
      expect(deleteButton.hasAttribute('data-delete-confirm-button')).toBe(false);
      fireEvent.click(deleteButton);
      expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();
      expect(screen.getByRole('heading', { name: 'Delete Tool' })).toBeTruthy();
      const note = screen.getByRole('note');
      expect(note.id).toBe('tool-delete-linked-note');
      expect(note.textContent).toContain(
        '“PDM Tool” is linked to the SolidWorks tool “CAD workstation”. If you delete it, that tool can no longer search the vault or open, add, check out or check in vault files until you link another PDM tool. The SolidWorks tool keeps its other settings and shows a warning.',
      );
      expect(note.textContent?.includes('Changes are currently on in 2 approved folders.')).toBe(
        allowWrite,
      );
      fireEvent.click(screen.getByRole('button', { name: 'Delete Tool' }));
      await waitFor(() => expect(apiMock.deleteToolConfig).toHaveBeenCalledWith(pdmTool.id));
      expect(apiMock.updateToolConfig).not.toHaveBeenCalled();
      expect(document.querySelector('[data-solidworks-pdm-badge]')?.textContent).toBe(
        'PDM unavailable',
      );
    },
  );

  it('shows every linked SolidWorks tool in the PDM row and context-menu delete note', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      linkedSolidworksTool,
      { ...linkedSolidworksTool, id: 'cad-second', name: 'Second workstation', allow_write: true },
      pdmTool,
    ]);
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByRole('heading', { name: 'PDM Tool' }));
    expect(screen.getByTestId(`pdm-linked-from-${pdmTool.id}`).textContent).toBe(
      'Used byCAD workstation, Second workstation',
    );
    fireEvent.click(
      within(document.querySelector('.userspace-tool-context-menu') as HTMLElement).getByRole(
        'button',
        { name: 'Delete' },
      ),
    );
    const note = screen.getByRole('note');
    expect(note.textContent).toContain('SolidWorks tool “CAD workstation”');
    expect(note.textContent).toContain('SolidWorks tool “Second workstation”');
    expect(note.textContent).toContain('Changes are currently on in 2 approved folders.');
    expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();
  });

  it('keeps the unlinked PDM card deletion path unchanged', async () => {
    apiMock.getSettings.mockResolvedValue({ settings: { show_tool_card_footer_actions: true } });
    apiMock.listToolConfigs.mockResolvedValue([pdmTool]);
    render(<ToolsPanel />);
    await screen.findByText('PDM Tool');
    const deleteButton = document.querySelector('[title="Delete tool"]')!;
    expect(deleteButton.hasAttribute('data-delete-confirm-button')).toBe(true);
    fireEvent.click(deleteButton);
    await waitFor(() => expect(apiMock.deleteToolConfig).toHaveBeenCalledWith(pdmTool.id));
    expect(screen.queryByRole('heading', { name: 'Delete Tool' })).toBeNull();
  });

  it('shows linked-tool consequences after disabling PDM succeeds', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...linkedSolidworksTool,
        pdm_link: { tool_id: pdmTool.id, tool_name: pdmTool.name, enabled: true, exists: true },
      },
      pdmTool,
    ]);
    apiMock.toggleToolConfig.mockResolvedValue({ enabled: false });
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByText('PDM Tool'));
    fireEvent.click(screen.getByRole('button', { name: 'Enabled' }));
    await waitFor(() =>
      expect(toastMock.info).toHaveBeenCalledWith(
        '“PDM Tool” is disabled. The SolidWorks tool “CAD workstation” cannot search the vault until you enable it again.',
      ),
    );
    expect(apiMock.toggleToolConfig).toHaveBeenCalledWith(pdmTool.id, false);
    expect(document.querySelector('[data-solidworks-pdm-badge]')?.textContent).toBe(
      'PDM unavailable',
    );
  });

  it('duplicates a tool from the right-click card menu and keeps the copy after the original', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith({
        name: 'Ungrouped Tool Copy',
        tool_type: 'ssh_shell',
        description: 'Ungrouped tool',
        connection_config: { host: 'example.org', user: 'deploy', port: 22 },
        max_results: 10,
        timeout_max_seconds: 30,
        allow_write: false,
        group_id: null,
      });
    });

    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-ungrouped', 'tool-ungrouped-copy'],
    });
    expect(await screen.findByText('Ungrouped Tool Copy')).toBeTruthy();
    expect(toastMock.success).toHaveBeenCalledWith('Tool duplicated');
  });

  it('opens the access modal from the card menu and saves ACL changes', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    await screen.findByText('Grouped Tool');

    fireEvent.contextMenu(screen.getByText('Grouped Tool'));
    await user.click(await screen.findByRole('button', { name: 'Edit Users' }));

    await waitFor(() => {
      expect(apiMock.getToolAccessPolicy).toHaveBeenCalledWith('tool-grouped');
    });
    expect(apiMock.listUsers).toHaveBeenCalledOnce();
    await user.click(screen.getByRole('button', { name: 'Save Access' }));

    await waitFor(() => {
      expect(apiMock.updateToolAccessPolicy).toHaveBeenCalledWith(
        'tool-grouped',
        expect.objectContaining({
          default_chat_access: 'read',
          default_workspace_access: 'deny',
        }),
      );
    });
  });

  it('shows delete confirmation modal before deleting a tool', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await screen.findByRole('button', { name: /Duplicate/ });
    const deleteButtons = screen.getAllByRole('button', { name: /Delete/ });
    await user.click(deleteButtons[deleteButtons.length - 1]);

    await screen.findByRole('heading', { name: 'Delete Tool' });
    expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Delete Tool' }));

    await waitFor(() => {
      expect(apiMock.deleteToolConfig).toHaveBeenCalledWith('tool-ungrouped');
    });
  });

  it('opens Windows cleanup confirmation from both tool deletion entrypoints', async () => {
    const user = userEvent.setup();
    apiMock.getSettings.mockResolvedValue({
      settings: { show_tool_card_footer_actions: true },
    });
    apiMock.listToolConfigs.mockResolvedValue([pairedWindowsTool]);

    render(<ToolsPanel />);

    await screen.findByText('Paired Windows Tool');
    await user.click(screen.getByTitle('Delete tool'));
    expect(await screen.findByRole('heading', { name: 'Delete paired Windows tool' })).toBeTruthy();
    expect(apiMock.deleteToolConfig).not.toHaveBeenCalled();
    expect(document.querySelector('[title="Delete tool"][data-delete-confirm-button]')).toBeNull();

    await user.click(screen.getByRole('button', { name: 'Close' }));
    fireEvent.contextMenu(screen.getByText('Paired Windows Tool'));
    const contextDeleteButtons = await screen.findAllByRole('button', { name: /^Delete$/ });
    await user.click(contextDeleteButtons[contextDeleteButtons.length - 1]);
    expect(await screen.findByRole('heading', { name: 'Delete paired Windows tool' })).toBeTruthy();
  });

  it('highlights the requested tool card when navigated to a tool-specific section target', async () => {
    render(<ToolsPanel highlightSection="tool:tool-ungrouped" />);

    const toolName = await screen.findByText('Ungrouped Tool');
    const toolCard = toolName.closest('.tool-card');

    await waitFor(() => expect(toolCard?.classList.contains('highlight-setting')).toBe(true));
  });

  it('highlights tool connections when opened with its hash', async () => {
    const originalHash = window.location.hash;
    window.location.hash = '#tools-connections';
    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    await waitFor(() =>
      expect(
        document.getElementById('tools-connections')?.classList.contains('highlight-setting'),
      ).toBe(true),
    );
    window.location.hash = originalHash;
  });

  it('uses the next copy suffix when duplicating a tool with an existing copy', async () => {
    const user = userEvent.setup();
    const existingCopy: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-existing-copy',
      name: 'Ungrouped Tool Copy',
      sort_order: 300,
      undecryptable_fields: [],
    };
    apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool, existingCopy]);
    apiMock.createToolConfig.mockResolvedValue({
      ...ungroupedTool,
      id: 'tool-ungrouped-copy-2',
      name: 'Ungrouped Tool Copy 2',
      sort_order: 400,
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith(
        expect.objectContaining({ name: 'Ungrouped Tool Copy 2' }),
      );
    });
    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-ungrouped', 'tool-ungrouped-copy-2', 'tool-existing-copy'],
    });
  });

  it('preserves disabled state after duplicating a disabled tool', async () => {
    const user = userEvent.setup();
    const disabledTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-disabled',
      name: 'Disabled Tool',
      enabled: false,
      sort_order: 300,
      undecryptable_fields: [],
    };
    apiMock.listToolConfigs.mockResolvedValue([groupedTool, disabledTool]);
    apiMock.createToolConfig.mockResolvedValue({
      ...disabledTool,
      id: 'tool-disabled-copy',
      name: 'Disabled Tool Copy',
      enabled: true,
      sort_order: 400,
    });
    apiMock.updateToolConfig.mockResolvedValue({
      ...disabledTool,
      id: 'tool-disabled-copy',
      name: 'Disabled Tool Copy',
      enabled: false,
      sort_order: 400,
    });

    render(<ToolsPanel />);

    await screen.findByText('Disabled Tool');

    fireEvent.contextMenu(screen.getByText('Disabled Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith('tool-disabled-copy', {
        enabled: false,
      });
    });
    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-disabled', 'tool-disabled-copy'],
    });
  });

  it('shows a warning badge when a tool has undecryptable credential fields', async () => {
    const brokenTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-broken',
      name: 'Broken Tool',
      undecryptable_fields: ['password'],
    };
    apiMock.listToolConfigs.mockResolvedValue([ungroupedTool, brokenTool]);

    render(<ToolsPanel />);

    await screen.findByText('Broken Tool');
    expect(screen.getByText('Key Mismatch')).toBeTruthy();
    expect(
      screen.getByTitle(/credentials cannot be decrypted with the current server key/i),
    ).toBeTruthy();
  });

  it('clears broken credentials and opens the edit wizard after confirmation', async () => {
    const user = userEvent.setup();
    const brokenTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-broken',
      name: 'Broken Tool',
      undecryptable_fields: ['password'],
    };
    const clearedTool: ToolConfig = {
      ...brokenTool,
      undecryptable_fields: [],
      updated_at: '2026-01-02T00:00:00Z',
    };
    apiMock.listToolConfigs.mockResolvedValue([ungroupedTool, brokenTool]);
    apiMock.clearToolUndecryptableCredentials.mockResolvedValue(clearedTool);

    render(<ToolsPanel />);

    await screen.findByText('Broken Tool');

    fireEvent.contextMenu(screen.getByText('Broken Tool'));
    await user.click(await screen.findByRole('button', { name: /Clear broken credentials/ }));

    await screen.findByRole('heading', { name: 'Clear Broken Credentials' });
    expect(screen.getByText(/cannot be decrypted with the current server key/i)).toBeTruthy();
    expect(screen.getByText('password')).toBeTruthy();
    expect(apiMock.clearToolUndecryptableCredentials).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Clear Credentials' }));

    await waitFor(() => {
      expect(apiMock.clearToolUndecryptableCredentials).toHaveBeenCalledWith('tool-broken');
    });
    expect(toastMock.success).toHaveBeenCalledWith(
      'Broken tool credentials cleared. Re-enter them to restore the connection.',
    );
    expect(wizardState.existingTool).toEqual(clearedTool);
  });

  it('shows live password requirements in export modal and enables export only for a strong matching password', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /^Export$/ }));

    await screen.findByRole('heading', { name: 'Export Tool Config' });
    // With an empty password every requirement is still outstanding.
    expect(screen.getByText('12+ characters')).toBeTruthy();
    expect(screen.getByText('Uppercase')).toBeTruthy();
    expect(screen.getByText('Lowercase')).toBeTruthy();
    expect(screen.getByText('Number')).toBeTruthy();
    expect(screen.getByText('Special character')).toBeTruthy();

    const passwordInput = screen.getByLabelText(/^Password$/i) as HTMLInputElement;
    const confirmInput = screen.getByLabelText(/^Confirm Password$/i) as HTMLInputElement;
    const exportButton = screen.getByRole('button', { name: /^Export$/ }) as HTMLButtonElement;

    await user.type(passwordInput, 'short1!');
    expect(exportButton.disabled).toBe(true);
    // Met requirements drop out of the "still needed" hint.
    expect(screen.queryByText('Lowercase')).toBeNull();
    expect(screen.getByText('12+ characters')).toBeTruthy();

    await user.clear(passwordInput);
    await user.type(passwordInput, 'StrongPass123!');
    expect(exportButton.disabled).toBe(true);
    // All requirements satisfied collapses to the success line.
    expect(screen.getByText('Password meets all requirements')).toBeTruthy();

    await user.type(confirmInput, 'StrongPass123!');
    await waitFor(() => expect(exportButton.disabled).toBe(false));
  });

  it('lists a Skills item after Edit Users with the attachment summary', async () => {
    const user = userEvent.setup();
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 3, enabled: 2 } },
      routes: {},
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));

    const item = await screen.findByRole('button', { name: /^Skills/ });
    expect(item.textContent).toContain('3 attached, 2 enabled');

    await user.click(item);

    expect(await screen.findByRole('dialog', { name: 'Skills · Ungrouped Tool' })).toBeTruthy();
  });

  it('shows None yet on the Skills menu item when the tool has no skills', async () => {
    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));

    const item = await screen.findByRole('button', { name: /^Skills/ });
    expect(item.textContent).toContain('None yet');
  });

  it('keeps the tool list usable when the skill summary cannot be loaded', async () => {
    apiMock.getInstructionSkillSummary.mockRejectedValue(new Error('offline'));

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    expect(document.querySelector('[data-tool-skills-badge]')).toBeNull();
    expect(toastMock.error).not.toHaveBeenCalled();
  });

  it('copies the source skills to the duplicate and reports success', async () => {
    const user = userEvent.setup();
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 2, enabled: 2 } },
      routes: {},
    });
    apiMock.copyToolInstructionSkills.mockResolvedValue([
      skillFixture('one', true),
      skillFixture('two', true),
    ]);

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(apiMock.copyToolInstructionSkills).toHaveBeenCalledWith(
        'tool-ungrouped-copy',
        'tool-ungrouped',
      );
    });
    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith('Tool duplicated'));
    expect(await screen.findAllByRole('button', { name: /^2 skills,/ })).toHaveLength(2);
  });

  it('keeps the duplicate and warns when its skills cannot be copied', async () => {
    const user = userEvent.setup();
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 2, enabled: 2 } },
      routes: {},
    });
    apiMock.copyToolInstructionSkills.mockRejectedValue(new Error('Name clash'));

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => {
      expect(toastMock.error).toHaveBeenCalledWith(
        'Tool duplicated, but its skills could not be copied',
      );
    });
    expect(toastMock.success).not.toHaveBeenCalledWith('Tool duplicated');
    expect(apiMock.createToolConfig).toHaveBeenCalledOnce();
    expect(await screen.findByText('Ungrouped Tool Copy')).toBeTruthy();
  });

  it('requests skill copying even when the summary has no source entry', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

    await waitFor(() => expect(toastMock.success).toHaveBeenCalledWith('Tool duplicated'));
    expect(apiMock.copyToolInstructionSkills).toHaveBeenCalledWith(
      'tool-ungrouped-copy',
      'tool-ungrouped',
    );
  });

  it.each(['failed', 'stale'] as const)(
    'copies source skills when the optional summary is %s',
    async (summaryState) => {
      const user = userEvent.setup();
      if (summaryState === 'failed') {
        apiMock.getInstructionSkillSummary.mockRejectedValue(new Error('offline'));
      } else {
        apiMock.getInstructionSkillSummary.mockResolvedValue({
          tools: { 'tool-ungrouped': { total: 0, enabled: 0 } },
          routes: {},
        });
      }
      apiMock.copyToolInstructionSkills.mockResolvedValue([skillFixture('one', true)]);

      render(<ToolsPanel />);

      await screen.findByText('Ungrouped Tool');
      expect(document.querySelector('[data-tool-skills-badge]')).toBeNull();
      fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
      await user.click(await screen.findByRole('button', { name: /Duplicate/ }));

      await waitFor(() =>
        expect(apiMock.copyToolInstructionSkills).toHaveBeenCalledWith(
          'tool-ungrouped-copy',
          'tool-ungrouped',
        ),
      );
      expect(await screen.findByRole('button', { name: /^1 skill,/ })).toBeTruthy();
      expect(toastMock.success).toHaveBeenCalledWith('Tool duplicated');
      expect(toastMock.error).not.toHaveBeenCalled();
    },
  );

  it('orders group sections by sort order with Ungrouped last', async () => {
    apiMock.listToolGroups.mockResolvedValue([
      { ...toolGroup, id: 'group-late', name: 'Late group', sort_order: 300 },
      toolGroup,
      { ...toolGroup, id: 'group-early', name: 'Early group', sort_order: 50 },
    ]);
    render(<ToolsPanel />);
    await screen.findByText(groupedTool.name);
    const sections = [...document.querySelectorAll('[data-tool-section]')];
    expect(sections.map((section) => section.getAttribute('data-tool-section'))).toEqual([
      'group-early',
      'group-1',
      'group-late',
      'ungrouped',
    ]);
  });

  it('renders a plain tool grid without section chrome when there are no groups', async () => {
    apiMock.listToolGroups.mockResolvedValue([]);
    apiMock.listToolConfigs.mockResolvedValue([ungroupedTool]);
    render(<ToolsPanel />);
    await screen.findByText(ungroupedTool.name);
    expect(document.querySelector('[data-tool-section]')).toBeNull();
    expect(document.querySelector('#tools-toolbar .tools-toolbar-summary')?.textContent).toBe(
      '1 tool',
    );
    expect(document.querySelectorAll('.tools-grid > [role="listitem"]')).toHaveLength(1);
  });

  it('collapses a section, stores the choice for the same user, and restores it on re-render', async () => {
    const { unmount } = render(<ToolsPanel currentUser={testUser} />);
    await screen.findByText(groupedTool.name);
    fireEvent.click(document.getElementById('tools-section-group-1-toggle')!);
    expect(sectionBody('group-1').hidden).toBe(true);
    expect(document.cookie).toContain('tools_sections_collapsed_user-1=group-1');
    unmount();

    render(<ToolsPanel currentUser={testUser} />);
    await screen.findByText(groupedTool.name);
    expect(sectionBody('group-1').hidden).toBe(true);
    expect(sectionBody('ungrouped').hidden).toBe(false);
  });

  it('filters sections in memory: matching sections open, others hide, and clearing restores the saved state', async () => {
    document.cookie = 'tools_sections_collapsed_user-1=ungrouped; path=/';
    toolFilterState.queries = ['ungrouped'];
    toolFilterState.hasActiveFilters = true;
    const { rerender } = render(<ToolsPanel currentUser={testUser} />);
    await screen.findByText(ungroupedTool.name);
    expect(document.querySelector('[data-tool-section="group-1"]')).toBeNull();
    expect(sectionBody('ungrouped').hidden).toBe(false);

    toolFilterState.queries = [];
    toolFilterState.hasActiveFilters = false;
    rerender(<ToolsPanel currentUser={testUser} />);
    expect(document.querySelector('[data-tool-section="group-1"]')).not.toBeNull();
    expect(sectionBody('ungrouped').hidden).toBe(true);
  });

  it('opens a collapsed section when a link targets one of its tools and scrolls to it', async () => {
    const scrollIntoView = vi.fn();
    Object.defineProperty(Element.prototype, 'scrollIntoView', {
      value: scrollIntoView,
      configurable: true,
      writable: true,
    });
    document.cookie = 'tools_sections_collapsed_user-1=ungrouped; path=/';
    render(<ToolsPanel currentUser={testUser} highlightSection="tool:tool-ungrouped" />);
    const toolCard = (await screen.findByText(ungroupedTool.name)).closest('.tool-card');
    await waitFor(() => expect(toolCard?.classList.contains('highlight-setting')).toBe(true));
    expect(sectionBody('ungrouped').hidden).toBe(false);
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it('moves a tool into a group from the card menu, updating the group and then the full order', async () => {
    const user = userEvent.setup();
    apiMock.updateToolConfig.mockResolvedValue({
      ...ungroupedTool,
      group_id: 'group-1',
      group_name: 'Alpha Group',
    });
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByText(ungroupedTool.name));
    await user.click(await screen.findByRole('button', { name: 'Move to group…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Move “Ungrouped Tool” to group' });
    await user.click(within(dialog).getByRole('radio', { name: /Alpha Group/ }));
    await user.click(within(dialog).getByRole('button', { name: 'Move' }));

    await waitFor(() =>
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith('tool-ungrouped', {
        group_id: 'group-1',
      }),
    );
    expect(apiMock.reorderTools).toHaveBeenCalledWith({
      tool_ids: ['tool-grouped', 'tool-ungrouped'],
    });
    expect(toastMock.success).toHaveBeenCalledWith('Moved “Ungrouped Tool” to “Alpha Group”');
  });

  it('moves a tool earlier or later within its section and disables the moves at the section ends', async () => {
    const user = userEvent.setup();
    const secondUngrouped: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-ungrouped-2',
      name: 'Second Ungrouped',
      sort_order: 300,
    };
    apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool, secondUngrouped]);
    render(<ToolsPanel />);
    await screen.findByText('Second Ungrouped');

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));
    const menu = await screen.findByRole('group', { name: 'Tool actions for Ungrouped Tool' });
    expect(within(menu).getByRole('button', { name: 'Move earlier' }).classList).toContain(
      'disabled',
    );
    await user.click(within(menu).getByRole('button', { name: 'Move later' }));

    await waitFor(() =>
      expect(apiMock.reorderTools).toHaveBeenCalledWith({
        tool_ids: ['tool-grouped', 'tool-ungrouped-2', 'tool-ungrouped'],
      }),
    );
    expect(apiMock.updateToolConfig).not.toHaveBeenCalled();
  });

  it('moves a group down with the full group order and announces its new position', async () => {
    const user = userEvent.setup();
    const betaGroup: ToolGroup = {
      ...toolGroup,
      id: 'group-beta',
      name: 'Beta group',
      sort_order: 200,
    };
    apiMock.listToolGroups.mockResolvedValue([toolGroup, betaGroup]);
    render(<ToolsPanel />);
    await screen.findByText(groupedTool.name);

    await user.click(screen.getByRole('button', { name: 'Group actions for Alpha Group' }));
    await user.click(await screen.findByRole('button', { name: 'Move down' }));

    await waitFor(() =>
      expect(groupApiMock.reorderToolGroups).toHaveBeenCalledWith({
        group_ids: ['group-beta', 'group-1'],
      }),
    );
    expect(await screen.findByText('Group “Alpha Group” moved to position 2 of 2.')).toBeTruthy();
  });

  it('deletes a group after confirmation and moves its tools to Ungrouped', async () => {
    const user = userEvent.setup();
    const betaGroup: ToolGroup = {
      ...toolGroup,
      id: 'group-beta',
      name: 'Beta group',
      sort_order: 200,
    };
    apiMock.listToolGroups.mockResolvedValue([toolGroup, betaGroup]);
    render(<ToolsPanel />);
    await screen.findByText(groupedTool.name);

    await user.click(screen.getByRole('button', { name: 'Group actions for Alpha Group' }));
    await user.click(await screen.findByRole('button', { name: 'Delete group…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Delete group “Alpha Group”?' });
    expect(
      within(dialog).getByText('The 1 tool in this group move to Ungrouped. No tools are deleted.'),
    ).toBeTruthy();
    await user.click(within(dialog).getByRole('button', { name: 'Delete group' }));

    await waitFor(() => expect(groupApiMock.deleteToolGroup).toHaveBeenCalledWith('group-1'));
    await waitFor(() =>
      expect(
        within(document.querySelector('[data-tool-section="ungrouped"]') as HTMLElement).getByText(
          groupedTool.name,
        ),
      ).toBeTruthy(),
    );
    expect(document.querySelector('[data-tool-section="group-1"]')).toBeNull();
  });

  it('keeps a group that lost its last tool instead of deleting it', async () => {
    const user = userEvent.setup();
    apiMock.updateToolConfig.mockResolvedValue({
      ...groupedTool,
      group_id: null,
      group_name: null,
    });
    render(<ToolsPanel />);
    fireEvent.contextMenu(await screen.findByText(groupedTool.name));
    await user.click(await screen.findByRole('button', { name: 'Move to group…' }));
    const dialog = await screen.findByRole('dialog', { name: 'Move “Grouped Tool” to group' });
    await user.click(within(dialog).getByRole('radio', { name: /No group \(Ungrouped\)/ }));
    await user.click(within(dialog).getByRole('button', { name: 'Move' }));

    await waitFor(() =>
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith('tool-grouped', { group_id: '' }),
    );
    expect(groupApiMock.deleteToolGroup).not.toHaveBeenCalled();
    const emptyGroup = document.querySelector('[data-tool-section="group-1"]') as HTMLElement;
    expect(within(emptyGroup).getByText(/No tools in this group yet/)).toBeTruthy();
  });

  it('creates a group after the last sort order and focuses its section toggle', async () => {
    const user = userEvent.setup();
    groupApiMock.createToolGroup.mockResolvedValue({
      ...toolGroup,
      id: 'group-new',
      name: 'Beta',
      description: '',
      sort_order: 200,
    });
    render(<ToolsPanel />);
    await screen.findByText(groupedTool.name);

    await user.click(screen.getByRole('button', { name: 'New group' }));
    const dialog = await screen.findByRole('dialog', { name: 'New group' });
    await user.type(within(dialog).getByLabelText('Name'), 'Beta');
    await user.click(within(dialog).getByRole('button', { name: 'Create group' }));

    await waitFor(() =>
      expect(groupApiMock.createToolGroup).toHaveBeenCalledWith({
        name: 'Beta',
        description: '',
        sort_order: 200,
      }),
    );
    await waitFor(() =>
      expect(document.activeElement).toBe(
        document.getElementById('tools-section-group-new-toggle'),
      ),
    );
  });

  it('opens tool actions from the card ⋯ button with focus inside and returns focus to it on Escape', async () => {
    const user = userEvent.setup();
    render(<ToolsPanel />);
    await screen.findByText(ungroupedTool.name);
    const trigger = document.querySelector<HTMLButtonElement>(
      '#tools-tool-tool-ungrouped [data-tool-actions-button]',
    )!;

    await user.click(trigger);
    const menu = await screen.findByRole('group', { name: 'Tool actions for Ungrouped Tool' });
    await waitFor(() => expect(menu.contains(document.activeElement)).toBe(true));

    fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
    await waitFor(() =>
      expect(screen.queryByRole('group', { name: 'Tool actions for Ungrouped Tool' })).toBeNull(),
    );
    expect(document.activeElement).toBe(trigger);
  });

  describe('DG-5 fixes', () => {
    const betaGroup: ToolGroup = {
      ...toolGroup,
      id: 'group-beta',
      name: 'Beta group',
      sort_order: 200,
    };
    const betaTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-beta',
      name: 'Beta Tool',
      description: 'Other',
      group_id: 'group-beta',
      group_name: 'Beta group',
      sort_order: 500,
    };
    const secondUngrouped: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-ungrouped-2',
      name: 'Second Ungrouped',
      sort_order: 300,
    };

    function deferred<T = void>() {
      let resolve!: (value: T) => void;
      let reject!: (reason: unknown) => void;
      const promise = new Promise<T>((res, rej) => {
        resolve = res;
        reject = rej;
      });
      return { promise, resolve, reject };
    }

    async function openToolMenu(name: string) {
      fireEvent.contextMenu(await screen.findByText(name));
      return screen.findByRole('group', { name: `Tool actions for ${name}` });
    }

    it('blocks a second tool move until the first one settles', async () => {
      const user = userEvent.setup();
      apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool, secondUngrouped]);
      const pending = deferred();
      apiMock.reorderTools.mockReturnValueOnce(pending.promise);
      render(<ToolsPanel />);
      await screen.findByText('Second Ungrouped');

      const firstMenu = await openToolMenu('Ungrouped Tool');
      await user.click(within(firstMenu).getByRole('button', { name: 'Move later' }));
      await waitFor(() => expect(apiMock.reorderTools).toHaveBeenCalledTimes(1));

      expect(
        document
          .getElementById('tools-tool-tool-grouped')!
          .parentElement!.getAttribute('draggable'),
      ).toBe('false');
      // Move later is a legal move for Second Ungrouped; only the lock disables it here.
      const secondMenu = await openToolMenu('Second Ungrouped');
      expect(within(secondMenu).getByRole('button', { name: 'Move later' }).classList).toContain(
        'disabled',
      );
      await user.click(within(secondMenu).getByRole('button', { name: 'Move later' }));
      expect(apiMock.reorderTools).toHaveBeenCalledTimes(1);

      pending.resolve();
      const settledMenu = await openToolMenu('Second Ungrouped');
      await waitFor(() =>
        expect(
          within(settledMenu).getByRole('button', { name: 'Move later' }).classList,
        ).not.toContain('disabled'),
      );
    });

    it('re-enables moves after a failed reorder reloads the list', async () => {
      const user = userEvent.setup();
      apiMock.listToolConfigs.mockResolvedValue([groupedTool, ungroupedTool, secondUngrouped]);
      apiMock.reorderTools.mockRejectedValueOnce(new Error('Reorder failed'));
      render(<ToolsPanel />);
      await screen.findByText('Second Ungrouped');

      const menu = await openToolMenu('Ungrouped Tool');
      await user.click(within(menu).getByRole('button', { name: 'Move later' }));
      await waitFor(() => expect(toastMock.error).toHaveBeenCalledWith('Reorder failed'));
      await waitFor(() => expect(apiMock.listToolConfigs.mock.calls.length).toBeGreaterThan(1));
      await screen.findByText('Ungrouped Tool');

      const retryMenu = await openToolMenu('Ungrouped Tool');
      expect(within(retryMenu).getByRole('button', { name: 'Move later' }).classList).not.toContain(
        'disabled',
      );
    });

    it('blocks a second group move until the first one settles', async () => {
      const user = userEvent.setup();
      apiMock.listToolGroups.mockResolvedValue([toolGroup, betaGroup]);
      const pending = deferred();
      groupApiMock.reorderToolGroups.mockReturnValueOnce(pending.promise);
      render(<ToolsPanel />);
      await screen.findByText(groupedTool.name);

      await user.click(screen.getByRole('button', { name: 'Group actions for Alpha Group' }));
      await user.click(await screen.findByRole('button', { name: 'Move down' }));
      await waitFor(() => expect(groupApiMock.reorderToolGroups).toHaveBeenCalledTimes(1));

      await user.click(screen.getByRole('button', { name: 'Group actions for Beta group' }));
      const moveDown = await screen.findByRole('button', { name: 'Move down' });
      expect(moveDown.classList).toContain('disabled');
      await user.click(moveDown);
      expect(groupApiMock.reorderToolGroups).toHaveBeenCalledTimes(1);

      pending.resolve();
      await waitFor(() =>
        expect(screen.getByText('Group “Alpha Group” moved to position 2 of 2.')).toBeTruthy(),
      );
      // The Beta menu is still open from above; read its state rather than toggling it shut.
      await waitFor(() =>
        expect(screen.getByRole('button', { name: 'Move down' }).classList).not.toContain(
          'disabled',
        ),
      );
    });

    it('clears an active filter when a group is created so the new section takes focus', async () => {
      const user = userEvent.setup();
      groupApiMock.createToolGroup.mockResolvedValue({
        ...toolGroup,
        id: 'group-new',
        name: 'Beta',
        description: '',
        sort_order: 200,
      });
      toolFilterState.queries = ['ungrouped'];
      toolFilterState.hasActiveFilters = true;
      render(<ToolsPanel />);
      await screen.findByText(ungroupedTool.name);

      await user.click(screen.getByRole('button', { name: 'New group' }));
      const dialog = await screen.findByRole('dialog', { name: 'New group' });
      await user.type(within(dialog).getByLabelText('Name'), 'Beta');
      await user.click(within(dialog).getByRole('button', { name: 'Create group' }));

      await waitFor(() => expect(toolFilterState.clear).toHaveBeenCalled());
      await waitFor(() =>
        expect(document.activeElement).toBe(
          document.getElementById('tools-section-group-new-toggle'),
        ),
      );
    });

    it('after deleting a group under a filter, focuses the next rendered section toggle', async () => {
      const user = userEvent.setup();
      apiMock.listToolGroups.mockResolvedValue([toolGroup, betaGroup]);
      apiMock.listToolConfigs.mockResolvedValue([groupedTool, betaTool, ungroupedTool]);
      toolFilterState.queries = ['group'];
      toolFilterState.hasActiveFilters = true;
      render(<ToolsPanel />);
      await screen.findByText(groupedTool.name);

      await user.click(screen.getByRole('button', { name: 'Group actions for Alpha Group' }));
      await user.click(await screen.findByRole('button', { name: 'Delete group…' }));
      const dialog = await screen.findByRole('dialog', { name: 'Delete group “Alpha Group”?' });
      await user.click(within(dialog).getByRole('button', { name: 'Delete group' }));

      await waitFor(() => expect(groupApiMock.deleteToolGroup).toHaveBeenCalledWith('group-1'));
      await waitFor(() =>
        expect(document.activeElement).toBe(
          document.getElementById('tools-section-group-beta-toggle'),
        ),
      );
    });

    it('after deleting a group under a filter, falls back to New group when the next section is hidden', async () => {
      const user = userEvent.setup();
      apiMock.listToolGroups.mockResolvedValue([toolGroup, betaGroup]);
      apiMock.listToolConfigs.mockResolvedValue([groupedTool, betaTool, ungroupedTool]);
      toolFilterState.queries = ['alpha group'];
      toolFilterState.hasActiveFilters = true;
      render(<ToolsPanel />);
      await screen.findByText(groupedTool.name);

      await user.click(screen.getByRole('button', { name: 'Group actions for Alpha Group' }));
      await user.click(await screen.findByRole('button', { name: 'Delete group…' }));
      const dialog = await screen.findByRole('dialog', { name: 'Delete group “Alpha Group”?' });
      await user.click(within(dialog).getByRole('button', { name: 'Delete group' }));

      await waitFor(() => expect(groupApiMock.deleteToolGroup).toHaveBeenCalledWith('group-1'));
      await waitFor(() =>
        expect(document.activeElement).toBe(document.getElementById('tools-new-group-button')),
      );
    });

    it('drops forced-open sections when the signed-in user changes', async () => {
      document.cookie = 'tools_sections_collapsed_user-1=ungrouped; path=/';
      document.cookie = 'tools_sections_collapsed_user-2=ungrouped; path=/';
      const { rerender } = render(
        <ToolsPanel currentUser={testUser} highlightSection="tool:tool-ungrouped" />,
      );
      await screen.findByText(ungroupedTool.name);
      await waitFor(() => expect(sectionBody('ungrouped').hidden).toBe(false));

      rerender(<ToolsPanel currentUser={{ id: 'user-2' } as User} />);
      await waitFor(() => expect(sectionBody('ungrouped').hidden).toBe(true));
    });

    it('drops in-memory filter toggles when the signed-in user changes', async () => {
      toolFilterState.queries = ['ungrouped'];
      toolFilterState.hasActiveFilters = true;
      const { rerender } = render(<ToolsPanel currentUser={testUser} />);
      await screen.findByText(ungroupedTool.name);

      fireEvent.click(document.getElementById('tools-section-ungrouped-toggle')!);
      expect(sectionBody('ungrouped').hidden).toBe(true);

      rerender(<ToolsPanel currentUser={{ id: 'user-2' } as User} />);
      await waitFor(() => expect(sectionBody('ungrouped').hidden).toBe(false));
    });

    it('gives each section grid a stable id hook', async () => {
      render(<ToolsPanel />);
      await screen.findByText(groupedTool.name);
      expect(document.getElementById('tools-section-group-1-grid')?.getAttribute('role')).toBe(
        'list',
      );
      expect(document.getElementById('tools-section-ungrouped-grid')).not.toBeNull();
    });

    it('gives the plain no-groups grid the tools-grid-all hook', async () => {
      apiMock.listToolGroups.mockResolvedValue([]);
      apiMock.listToolConfigs.mockResolvedValue([ungroupedTool]);
      render(<ToolsPanel />);
      await screen.findByText(ungroupedTool.name);
      expect(document.getElementById('tools-grid-all')?.getAttribute('role')).toBe('list');
    });

    it('shows visible-of-total counts while filtering', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        groupedTool,
        { ...groupedTool, id: 'tool-grouped-2', name: 'Grouped Other', description: 'x' },
      ]);
      toolFilterState.queries = ['grouped tool'];
      toolFilterState.hasActiveFilters = true;
      render(<ToolsPanel />);
      await screen.findByText(groupedTool.name);
      const section = document.querySelector('[data-tool-section="group-1"]') as HTMLElement;
      expect(within(section).getByText('1 of 2 tools')).toBeTruthy();
    });
  });
});
