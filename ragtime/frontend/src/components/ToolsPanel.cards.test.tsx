import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ToolsPanel } from './ToolsPanel';
import { hostUpdateFailureText } from '@/utils/solidworksHostUpdate';
import type { SolidworksHostAutoUpdate, ToolConfig } from '@/types';

import {
  apiMock,
  skillsDialogMock,
  contentProtectionMock,
  ungroupedTool,
  pdmTool,
  httpApiTool,
  pairedWindowsTool,
  linkedSolidworksTool,
  skillFixture,
  setupToolsPanelTest,
  teardownToolsPanelTest,
} from './ToolsPanel.testSupport';

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
  beforeEach(setupToolsPanelTest);
  afterEach(teardownToolsPanelTest);

  it.each([
    ['healthy', true, true, 'PDM: PDM Tool', 'Linked to SolidWorks PDM tool PDM Tool, vault PDM'],
    ['missing', false, false, 'PDM unavailable', 'The linked PDM tool no longer exists.'],
    [
      'disabled',
      true,
      false,
      'PDM unavailable',
      '“PDM Tool” is disabled, so the AI cannot search it. Enable it or choose another tool.',
    ],
  ])('shows the %s PDM link badge', async (_state, exists, enabled, label, description) => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...linkedSolidworksTool,
        pdm_link: { tool_id: pdmTool.id, tool_name: pdmTool.name, exists, enabled },
      },
      pdmTool,
    ]);
    render(<ToolsPanel />);
    await screen.findByRole('heading', { name: 'CAD workstation' });
    const badge = document.querySelector('[data-solidworks-pdm-badge]')!;
    expect(badge.textContent).toBe(label);
    expect(badge.parentElement?.getAttribute('data-popover-content')).toBe(description);
    expect(badge.classList.contains('badge-warning')).toBe(!exists || !enabled);
  });

  it('shows SolidWorks connection and relevant badges without write or working directory', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...pairedWindowsTool,
        id: 'solidworks-pinned',
        name: 'SolidWorks pinned',
        tool_type: 'solidworks',
        allow_write: true,
        connection_config: {
          host: 'cad.example.org',
          user: 'cad',
          port: 2222,
          solidworks_version: 2024,
          instance_mode: 'attach',
          working_directory: 'C:\\work',
        },
      },
      {
        ...pairedWindowsTool,
        id: 'solidworks-default',
        name: 'SolidWorks default',
        tool_type: 'solidworks',
        connection_config: { host: 'other.example.org', user: 'cad', port: 22 },
      },
    ]);
    render(<ToolsPanel />);
    await screen.findByText('SolidWorks pinned');
    const cards = document.querySelectorAll('[data-tool-type="solidworks"]');
    expect(cards).toHaveLength(2);
    const pinned = Array.from(cards).find((card) =>
      card.textContent?.includes('SolidWorks pinned'),
    )!;
    const defaults = Array.from(cards).find((card) =>
      card.textContent?.includes('SolidWorks default'),
    )!;
    expect(pinned.querySelector('[data-tool-summary]')?.textContent).toBe(
      'cad@cad.example.org:2222',
    );
    expect(pinned.querySelector('[data-solidworks-version-badge]')?.textContent).toBe(
      'SolidWorks 2024',
    );
    expect(pinned.querySelector('[data-solidworks-mode-badge]')?.textContent).toBe('Attach');
    expect(pinned.textContent).not.toContain('Write');
    expect(pinned.textContent).not.toContain('C:\\work');
    expect(defaults.querySelector('[data-solidworks-version-badge]')).toBeNull();
    expect(defaults.querySelector('[data-solidworks-mode-badge]')).toBeNull();
    fireEvent.contextMenu(pinned);
    expect(screen.queryByRole('button', { name: 'Write Access' })).toBeNull();
  });

  it('shows scripts only on SolidWorks tools that allow them, even when disabled', async () => {
    apiMock.listToolConfigs.mockResolvedValue([
      {
        ...pairedWindowsTool,
        id: 'scripts-on',
        name: 'Scripts on',
        tool_type: 'solidworks',
        enabled: false,
        connection_config: { allow_scripts: true },
      },
      {
        ...pairedWindowsTool,
        id: 'scripts-off',
        name: 'Scripts off',
        tool_type: 'solidworks',
        connection_config: { allow_scripts: false },
      },
      {
        ...pairedWindowsTool,
        id: 'scripts-missing',
        name: 'Scripts missing',
        tool_type: 'solidworks',
        connection_config: {},
      },
      {
        ...pairedWindowsTool,
        id: 'other',
        name: 'Other tool',
        connection_config: { allow_scripts: true },
      },
    ]);
    render(<ToolsPanel />);
    await screen.findByRole('heading', { name: 'Scripts on' });
    const badges = document.querySelectorAll('[data-badge="scripts-allowed"]');
    expect(badges).toHaveLength(1);
    expect(badges[0].textContent).toBe('Scripts on');
    expect(badges[0].parentElement?.getAttribute('data-popover-content')).toBe(
      'Can run C# scripts on the SolidWorks computer as its Windows user. Not sandboxed.',
    );
  });

  describe('agent update and tunnel client badges', () => {
    const hostUpdate = (
      overrides: Partial<SolidworksHostAutoUpdate> = {},
    ): SolidworksHostAutoUpdate => ({
      enabled: true,
      state: 'pending',
      installed_version: '0.3.0',
      target_version: '0.4.0',
      reason: 'in_use',
      last_attempt_at: null,
      next_attempt_at: null,
      message: null,
      ...overrides,
    });

    const solidworksTool = (
      id: string,
      name: string,
      extra: Partial<ToolConfig> = {},
    ): ToolConfig => ({
      ...ungroupedTool,
      id,
      name,
      tool_type: 'solidworks',
      connection_config: { host: 'cad.example.org', user: 'cad', port: 22 },
      ...extra,
    });

    const cardFor = async (name: string): Promise<HTMLElement> => {
      const heading = await screen.findByRole('heading', { name });
      return heading.closest('[data-tool-type]') as HTMLElement;
    };

    const badgeLabel = (badge: HTMLElement): string | null =>
      badge.querySelector('.path-text')?.textContent ?? null;
    const badgeDetail = (badge: HTMLElement): string | null =>
      badge.querySelector('.rc-visually-hidden')?.textContent ?? null;

    const hostBadgesIn = (card: HTMLElement): Element[] =>
      Array.from(
        card.querySelectorAll(
          '[data-tool-badge^="host-update"], [data-tool-badge="host-updating"]',
        ),
      );

    it('shows a pending agent update badge with the installed and target versions and reason', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        solidworksTool('sw-pending', 'SW pending', {
          solidworks_host_update: hostUpdate({ reason: 'documents_open' }),
        }),
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('SW pending');
      const badge = card.querySelector('[data-tool-badge="host-update-pending"]') as HTMLElement;
      expect(badgeLabel(badge)).toBe('Agent update pending');
      expect(badge.className).toBe('tool-badge');
      expect(badge.tabIndex).toBe(0);
      const detail =
        'Version 0.3.0 is running; 0.4.0 will be installed automatically. A document is open in a SolidWorks instance the agent started. Ragtime updates the agent after that instance closes.';
      expect(badge.parentElement?.getAttribute('data-popover-content')).toBe(detail);
      expect(badgeDetail(badge)).toBe(detail);
    });

    it('shows an updating badge with a decorative spinner and the target version in its popover', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        solidworksTool('sw-updating', 'SW updating', {
          solidworks_host_update: hostUpdate({ state: 'updating', reason: null }),
        }),
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('SW updating');
      const badge = card.querySelector('[data-tool-badge="host-updating"]') as HTMLElement;
      expect(badgeLabel(badge)).toBe('Updating agent…');
      expect(badge.className).toBe('tool-badge');
      expect(badge.tabIndex).toBe(0);
      expect(badge.querySelector('.rc-polling-indicator')?.getAttribute('aria-hidden')).toBe(
        'true',
      );
      const detail =
        'Ragtime is updating the agent to version 0.4.0. Requests wait until it finishes.';
      expect(badge.parentElement?.getAttribute('data-popover-content')).toBe(detail);
      expect(badgeDetail(badge)).toBe(detail);
    });

    it('shows a warning failed badge whose popover gives the message and next try', async () => {
      const failedUpdate = hostUpdate({
        state: 'failed',
        reason: null,
        message: 'The installer could not replace the agent.',
        last_attempt_at: null,
        next_attempt_at: null,
      });
      apiMock.listToolConfigs.mockResolvedValue([
        solidworksTool('sw-failed', 'SW failed', { solidworks_host_update: failedUpdate }),
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('SW failed');
      const badge = card.querySelector('[data-tool-badge="host-update-failed"]') as HTMLElement;
      expect(badgeLabel(badge)).toBe('Agent update failed');
      expect(badge.className).toBe('tool-badge badge-warning');
      expect(badge.tabIndex).toBe(0);
      const detail = `${hostUpdateFailureText(failedUpdate)} Open the tool to update the agent yourself.`;
      expect(detail).toContain('The installer could not replace the agent.');
      expect(badge.parentElement?.getAttribute('data-popover-content')).toBe(detail);
      expect(badgeDetail(badge)).toBe(detail);
    });

    it('shows at most one agent update badge per SolidWorks tool', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        solidworksTool('sw-one', 'SW one', {
          solidworks_host_update: hostUpdate({ state: 'failed', reason: null }),
          tunnel_client: { connected: true, version: 1, current_version: 2, outdated: true },
        }),
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('SW one');
      expect(hostBadgesIn(card)).toHaveLength(1);
      expect(card.querySelector('[data-tool-badge="tunnel-client-outdated"]')).not.toBeNull();
    });

    it.each([
      ['off', hostUpdate({ state: 'off', reason: null })],
      ['current', hostUpdate({ state: 'current', reason: null })],
      ['newer', hostUpdate({ state: 'newer', reason: null })],
      ['absent', null],
    ])('shows no agent update badge when the agent state is %s', async (label, update) => {
      const name = `SW ${label}`;
      apiMock.listToolConfigs.mockResolvedValue([
        solidworksTool('sw-quiet', name, { solidworks_host_update: update }),
      ]);
      render(<ToolsPanel />);

      const card = await cardFor(name);
      expect(hostBadgesIn(card)).toHaveLength(0);
    });

    it('never shows an agent update badge for a tool that is not SolidWorks', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        { ...ungroupedTool, name: 'Plain shell', solidworks_host_update: hostUpdate() },
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('Plain shell');
      expect(hostBadgesIn(card)).toHaveLength(0);
    });

    it('shows a tunnel client outdated badge with the version and newest version in its popover', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        {
          ...pairedWindowsTool,
          name: 'Tunnel old',
          tunnel_client: { connected: false, version: 1, current_version: 2, outdated: true },
        },
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('Tunnel old');
      const badge = card.querySelector('[data-tool-badge="tunnel-client-outdated"]') as HTMLElement;
      expect(badgeLabel(badge)).toBe('Tunnel client outdated');
      expect(badge.className).toBe('tool-badge');
      expect(badge.tabIndex).toBe(0);
      const detail =
        "The tunnel keeps working. This machine's tunnel client is version 1; the newest is version 2. Ragtime cannot update tunnel clients yet.";
      expect(badge.parentElement?.getAttribute('data-popover-content')).toBe(detail);
      expect(badgeDetail(badge)).toBe(detail);
    });

    it('explains an unreported tunnel client version as older than the newest version', async () => {
      apiMock.listToolConfigs.mockResolvedValue([
        {
          ...pairedWindowsTool,
          name: 'Tunnel unknown',
          tunnel_client: { connected: true, version: null, current_version: 2, outdated: true },
        },
      ]);
      render(<ToolsPanel />);

      const card = await cardFor('Tunnel unknown');
      const badge = card.querySelector('[data-tool-badge="tunnel-client-outdated"]') as HTMLElement;
      expect(badgeDetail(badge)).toBe(
        "The tunnel keeps working. This machine's tunnel client did not report a version, so it is older than version 2. Ragtime cannot update tunnel clients yet.",
      );
    });

    it.each([
      ['false', { connected: true, version: 2, current_version: 2, outdated: false }],
      ['undefined', undefined],
    ])('shows no tunnel client badge when outdated is %s', async (label, tunnelClient) => {
      const name = `Tunnel ${label}`;
      apiMock.listToolConfigs.mockResolvedValue([
        { ...pairedWindowsTool, name, tunnel_client: tunnelClient },
      ]);
      render(<ToolsPanel />);

      const card = await cardFor(name);
      expect(card.querySelector('[data-tool-badge="tunnel-client-outdated"]')).toBeNull();
    });
  });

  it('adds soft wrap opportunities only to heartbeat error tokens longer than 30 characters', async () => {
    apiMock.getToolHeartbeats.mockResolvedValue({
      statuses: {
        [ungroupedTool.id]: {
          tool_id: ungroupedTool.id,
          alive: false,
          latency_ms: null,
          error: `Error: ${'a'.repeat(30)} ${'b'.repeat(31)}`,
          checked_at: '2026-01-01T00:00:00Z',
        },
      },
    });

    const { container } = render(<ToolsPanel />);

    await screen.findByText(`Error: ${'a'.repeat(30)} ${'b'.repeat(31)}`);

    expect(container.querySelectorAll('.tool-card-heartbeat-error wbr')).toHaveLength(1);
  });

  it('does not add soft wrap opportunities to a 30-character heartbeat error token', async () => {
    apiMock.getToolHeartbeats.mockResolvedValue({
      statuses: {
        [ungroupedTool.id]: {
          tool_id: ungroupedTool.id,
          alive: false,
          latency_ms: null,
          error: `Error: ${'a'.repeat(30)}`,
          checked_at: '2026-01-01T00:00:00Z',
        },
      },
    });

    const { container } = render(<ToolsPanel />);

    await screen.findByText(`Error: ${'a'.repeat(30)}`);

    expect(container.querySelectorAll('.tool-card-heartbeat-error wbr')).toHaveLength(0);
  });

  it('hides card footer actions by default and exposes them in the right-click menu', async () => {
    const user = userEvent.setup();

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');

    expect(screen.queryByRole('button', { name: /^Test$/ })).toBeNull();
    expect(screen.queryByRole('button', { name: /^Edit$/ })).toBeNull();
    expect(screen.queryAllByRole('button', { name: /^Delete$/ })).toHaveLength(0);

    fireEvent.contextMenu(screen.getByText('Ungrouped Tool'));

    expect(await screen.findByRole('button', { name: /^Test$/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Edit$/ })).toBeTruthy();
    expect(screen.getByRole('button', { name: /^Duplicate$/ })).toBeTruthy();
    expect(screen.getAllByRole('button', { name: /^Delete$/ })).toHaveLength(1);

    await user.click(screen.getByRole('button', { name: /^Edit$/ }));

    await waitFor(() => {
      expect(screen.queryByRole('button', { name: /^Test$/ })).toBeNull();
    });
  });

  it('shows card footer actions when the global legacy setting is enabled', async () => {
    apiMock.getSettings.mockResolvedValue({
      settings: { show_tool_card_footer_actions: true },
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    const card = within(document.getElementById('tools-tool-tool-ungrouped') as HTMLElement);

    expect(card.getByRole('button', { name: /^Test$/ })).toBeTruthy();
    expect(card.getByRole('button', { name: /^Edit$/ })).toBeTruthy();
    expect(card.getAllByRole('button', { name: /^Delete$/ })).toHaveLength(1);
  });

  it('shows write badge on the card when write access is enabled', async () => {
    const writeTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-write',
      name: 'Write Tool',
      allow_write: true,
      undecryptable_fields: [],
      connection_config: { host: 'example.org', user: 'deploy', port: 22 },
    };
    apiMock.listToolConfigs.mockResolvedValue([writeTool]);

    render(<ToolsPanel />);

    await screen.findByText('Write Tool');
    expect(screen.getByText('Write')).toBeTruthy();
  });

  it('shows a Classified badge only for required tools while protection is enabled', async () => {
    contentProtectionMock.getConfig.mockResolvedValue({
      revision: 1,
      enabled: true,
      classifier_model: null,
      coverage_mode: 'selected_scopes',
      profiles: [],
      group_profiles: [],
      requirements: [{ scope_kind: 'tool', scope_key: 'tool-ungrouped', mode: 'require' }],
      user_overrides: [],
    });

    render(<ToolsPanel />);

    expect(await screen.findByText('Classified')).toBeTruthy();
  });

  it('hides the Classified badge while content protection is disabled', async () => {
    contentProtectionMock.getConfig.mockResolvedValue({
      revision: 1,
      enabled: false,
      classifier_model: null,
      coverage_mode: 'selected_scopes',
      profiles: [],
      group_profiles: [],
      requirements: [{ scope_kind: 'tool', scope_key: 'tool-ungrouped', mode: 'require' }],
      user_overrides: [],
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    expect(screen.queryByText('Classified')).toBeNull();
  });

  it('shows PDM index document stats on SolidWorks PDM cards', async () => {
    apiMock.listToolConfigs.mockResolvedValue([pdmTool]);
    apiMock.getPdmIndexStats.mockResolvedValue({
      document_count: 1160,
      embedding_count: 3480,
      last_indexed_at: '2026-01-02T00:00:00Z',
    });

    render(<ToolsPanel />);

    await screen.findByText('PDM Tool');

    expect(apiMock.getPdmIndexStats).toHaveBeenCalledWith('tool-pdm');
    expect(await screen.findByText('PDM Index:')).toBeTruthy();
    expect(screen.getByText(`${new Intl.NumberFormat().format(1160)} documents`)).toBeTruthy();
  });

  it('shows / in the working directory badge when unset', async () => {
    const noWdTool: ToolConfig = {
      ...ungroupedTool,
      id: 'tool-nowd',
      name: 'No Work Dir Tool',
      undecryptable_fields: [],
      connection_config: { host: 'example.org', user: 'deploy', port: 22 },
    };
    apiMock.listToolConfigs.mockResolvedValue([noWdTool]);

    render(<ToolsPanel />);

    await screen.findByText('No Work Dir Tool');
    expect(screen.getByText('/')).toBeTruthy();
  });

  it('shows an HTTP API summary with base URL and auth mode without secrets', async () => {
    apiMock.listToolConfigs.mockResolvedValue([httpApiTool]);

    render(<ToolsPanel />);

    expect(await screen.findByText('Orders API')).toBeTruthy();
    expect(screen.getByText(/https:\/\/api\.example\.com/i)).toBeTruthy();
    expect(screen.getByText(/bearer/i)).toBeTruthy();
    expect(screen.queryByText(/token/i)).toBeNull();
  });

  it('hides the skills badge and footer button when the tool has no skills', async () => {
    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    expect(document.querySelector('[data-tool-skills-badge]')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Skills' })).toBeNull();
  });

  it('shows a skill count badge whose title gives the enabled count', async () => {
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 3, enabled: 2 } },
      routes: {},
    });

    render(<ToolsPanel />);

    const badge = await screen.findByRole('button', { name: /^3 skills,/ });
    expect(badge.getAttribute('title')).toBe('2 of 3 skills enabled');
    expect(badge.hasAttribute('data-tool-skills-badge')).toBe(true);
  });

  it('uses the singular label for a single skill', async () => {
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 1, enabled: 0 } },
      routes: {},
    });

    render(<ToolsPanel />);

    const badge = await screen.findByRole('button', { name: /^1 skill,/ });
    expect(badge.textContent).toBe('1 skill');
    expect(badge.getAttribute('title')).toBe('0 of 1 skills enabled');
  });

  it('opens the skills dialog for the tool from its badge without opening the card menu', async () => {
    const user = userEvent.setup();
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 1, enabled: 1 } },
      routes: {},
    });

    render(<ToolsPanel />);

    await user.click(await screen.findByRole('button', { name: /^1 skill,/ }));

    const dialog = await screen.findByRole('dialog', { name: 'Skills · Ungrouped Tool' });
    expect(dialog.getAttribute('data-owner-kind')).toBe('tool');
    expect(dialog.getAttribute('data-owner-id')).toBe('tool-ungrouped');
    expect(screen.queryByRole('button', { name: /^Duplicate$/ })).toBeNull();
  });

  it('shows a secondary Skills button in footer actions that opens the dialog', async () => {
    const user = userEvent.setup();
    apiMock.getSettings.mockResolvedValue({
      settings: { show_tool_card_footer_actions: true },
    });

    render(<ToolsPanel />);

    await screen.findByText('Ungrouped Tool');
    const skillsButton = within(
      document.getElementById('tools-tool-tool-ungrouped') as HTMLElement,
    ).getByRole('button', { name: 'Skills' });
    expect(skillsButton.classList.contains('btn-secondary')).toBe(true);

    await user.click(skillsButton);

    const dialog = await screen.findByRole('dialog', { name: 'Skills · Ungrouped Tool' });
    expect(dialog.getAttribute('data-owner-id')).toBe('tool-ungrouped');
  });

  it('updates the badge from the skills the dialog reports', async () => {
    const user = userEvent.setup();
    apiMock.getInstructionSkillSummary.mockResolvedValue({
      tools: { 'tool-ungrouped': { total: 1, enabled: 1 } },
      routes: {},
    });
    skillsDialogMock.nextSkills = [skillFixture('one', true), skillFixture('two', false)];

    render(<ToolsPanel />);

    await user.click(await screen.findByRole('button', { name: /^1 skill,/ }));
    await user.click(await screen.findByRole('button', { name: 'Save Skills Mock' }));

    const badge = await screen.findByRole('button', { name: /^2 skills,/ });
    expect(badge.getAttribute('title')).toBe('1 of 2 skills enabled');
  });
});
