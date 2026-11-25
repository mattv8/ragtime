import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { MCPRoutesPanel } from './MCPRoutesPanel';
import type {
  IndexInfo,
  InstructionSkill,
  InstructionSkillOwner,
  InstructionSkillSummary,
  McpRouteConfig,
  ToolConfig,
} from '@/types';

const apiMock = vi.hoisted(() => ({
  listMcpRoutes: vi.fn(),
  listMcpDefaultFilters: vi.fn(),
  listToolConfigs: vi.fn(),
  listIndexes: vi.fn(),
  getSettings: vi.fn(),
  getInstructionSkillSummary: vi.fn(),
  createMcpRoute: vi.fn(),
  updateMcpRoute: vi.fn(),
  toggleMcpRoute: vi.fn(),
  deleteMcpRoute: vi.fn(),
  createMcpDefaultFilter: vi.fn(),
  updateMcpDefaultFilter: vi.fn(),
  toggleMcpDefaultFilter: vi.fn(),
  deleteMcpDefaultFilter: vi.fn(),
}));

const toastMock = {
  success: vi.fn(),
  error: vi.fn(),
  dismiss: vi.fn(),
};

vi.mock('@/api', () => ({ api: apiMock }));
const contentProtectionMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
}));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getConfig: contentProtectionMock.getConfig },
  updateContentProtectionConfigSlice: contentProtectionMock.updateContentProtectionConfigSlice,
}));

vi.mock('./shared/Toast', () => ({
  useToast: () => [[], toastMock] as const,
  ToastContainer: () => null,
}));

const skillsDialogMock = vi.hoisted(() => ({
  nextSkills: [] as InstructionSkill[],
}));

vi.mock('./InstructionSkillsDialog', () => ({
  InstructionSkillsDialog: ({
    open,
    owner,
    ownerName,
    onClose,
    onSkillsChanged,
  }: {
    open: boolean;
    owner: InstructionSkillOwner;
    ownerName: string;
    onClose: () => void;
    onSkillsChanged?: (skills: InstructionSkill[]) => void;
  }) =>
    open ? (
      <div
        role="dialog"
        aria-label={`Skills · ${ownerName}`}
        data-owner-kind={owner.kind}
        data-owner-id={owner.id}
      >
        <button type="button" onClick={onClose}>
          Close Skills Mock
        </button>
        <button type="button" onClick={() => onSkillsChanged?.(skillsDialogMock.nextSkills)}>
          Save Skills Mock
        </button>
      </div>
    ) : null,
}));

vi.mock('./DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({
    onDelete,
    buttonText,
    className,
  }: {
    onDelete: () => void;
    buttonText: string;
    className?: string;
  }) => (
    <button type="button" className={className} onClick={onDelete}>
      {buttonText}
    </button>
  ),
}));

const TOOL_CONFIGS: ToolConfig[] = [
  {
    id: 'tool-1',
    name: 'Orders SQL',
    tool_type: 'postgres',
    enabled: true,
    description: 'Orders data',
    connection_config: {},
    max_results: 10,
    timeout_max_seconds: 30,
    allow_write: false,
    sort_order: 10,
    group_id: null,
    group_name: null,
    undecryptable_fields: [],
    configured_secret_fields: [],
    last_test_at: null,
    last_test_result: null,
    last_test_error: null,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  },
];

const INDEXES: IndexInfo[] = [];

const LDAP_GROUPS = [
  {
    dn: 'cn=engineering,dc=example,dc=com',
    name: 'engineering',
    display_name: 'Engineering',
  },
];

const CONTENT_PROTECTION_CONFIG = {
  revision: 1,
  enabled: true,
  schema_version: 2,
  share_with_assistant: false,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: 'jev-latest' }, llm_model: null },
  strictness: 'strict',
  categories: [
    {
      id: 'operational',
      name: 'Operational',
      description: 'Operational information.',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: 'Restricted.',
      threshold_override: null,
      system: false,
    },
    {
      id: 'rule_override',
      name: 'Rule override',
      description: 'Policy override requests.',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: 'Cannot override.',
      threshold_override: null,
      system: true,
    },
  ],
  access_levels: [
    { id: 'standard', name: 'Standard', granted_category_ids: ['operational'], guidance: '' },
  ],
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'selected_scopes' as const,
  requirements: [],
  user_overrides: [],
};

function makeRoute(overrides: Partial<McpRouteConfig> = {}): McpRouteConfig {
  return {
    id: 'route-1',
    name: 'Custom Route',
    route_path: 'custom_route',
    description: 'Custom route description',
    enabled: true,
    require_auth: true,
    has_password: false,
    auth_password: undefined,
    auth_client_id: undefined,
    auth_method: 'oauth2',
    allowed_ldap_group: null,
    include_knowledge_search: false,
    include_git_history: false,
    selected_document_indexes: [],
    selected_filesystem_indexes: [],
    selected_schema_indexes: [],
    tool_config_ids: ['tool-1'],
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    ...overrides,
  };
}

function skillFixture(id: string, enabled: boolean): InstructionSkill {
  return {
    id,
    owner_kind: 'route',
    owner_id: 'route-1',
    name: `skill-${id}`,
    description: 'Describes the skill',
    content: 'Skill body',
    mode: 'on_demand',
    enabled,
    sort_order: 0,
    uri: `skill://routes/route-1/skill-${id}/SKILL.md`,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  };
}

async function renderPanel({
  routes = [],
  ldapConfigured = false,
  contentProtectionConfig = CONTENT_PROTECTION_CONFIG,
  skillSummary = { tools: {}, routes: {} },
  rejectSkillSummary = false,
}: {
  routes?: McpRouteConfig[];
  ldapConfigured?: boolean;
  contentProtectionConfig?: typeof CONTENT_PROTECTION_CONFIG;
  skillSummary?: InstructionSkillSummary | null;
  rejectSkillSummary?: boolean;
} = {}) {
  apiMock.listMcpRoutes.mockResolvedValue({ routes, count: routes.length });
  apiMock.listMcpDefaultFilters.mockResolvedValue({ filters: [], count: 0 });
  apiMock.listToolConfigs.mockResolvedValue(TOOL_CONFIGS);
  apiMock.listIndexes.mockResolvedValue(INDEXES);
  if (rejectSkillSummary)
    apiMock.getInstructionSkillSummary.mockRejectedValue(new Error('offline'));
  else apiMock.getInstructionSkillSummary.mockResolvedValue(skillSummary);
  apiMock.getSettings.mockResolvedValue({
    settings: {
      aggregate_search: true,
      mcp_default_route_auth: false,
      mcp_default_route_auth_method: 'password',
    },
  });
  contentProtectionMock.getConfig.mockResolvedValue(contentProtectionConfig);
  contentProtectionMock.updateContentProtectionConfigSlice.mockImplementation(async (mutate) =>
    mutate(CONTENT_PROTECTION_CONFIG),
  );

  render(<MCPRoutesPanel ldapConfigured={ldapConfigured} ldapGroups={LDAP_GROUPS} />);

  await screen.findByRole('button', { name: /add custom route/i });
}

async function openCreateRoute(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: /add custom route/i }));
  await user.type(screen.getByPlaceholderText('My Custom Tools'), 'OAuth Route');
  await user.type(screen.getByPlaceholderText('my_tools'), 'oauth_route');
}

describe('MCPRoutesPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.createMcpRoute.mockResolvedValue(makeRoute());
    apiMock.updateMcpRoute.mockResolvedValue(makeRoute());
    apiMock.toggleMcpRoute.mockResolvedValue(undefined);
    apiMock.deleteMcpRoute.mockResolvedValue(undefined);
    apiMock.createMcpDefaultFilter.mockResolvedValue(undefined);
    apiMock.updateMcpDefaultFilter.mockResolvedValue(undefined);
    apiMock.toggleMcpDefaultFilter.mockResolvedValue(undefined);
    apiMock.deleteMcpDefaultFilter.mockResolvedValue(undefined);
    skillsDialogMock.nextSkills = [];
    vi.spyOn(globalThis.crypto, 'getRandomValues').mockImplementation((array) => {
      const values = array as Uint8Array;
      values.fill(7);
      return array;
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('shows OAuth2 without LDAP and selects it by default for new routes', async () => {
    const user = userEvent.setup();

    await renderPanel();
    await openCreateRoute(user);
    await user.click(screen.getByRole('checkbox', { name: /require authentication/i }));

    const oauthRadio = screen.getByRole('radio', { name: 'OAuth2' });

    expect((oauthRadio as HTMLInputElement).checked).toBe(true);
    expect(screen.queryByText('OAuth2 (LDAP)')).toBeNull();
    expect(screen.getAllByRole('radio').map((radio) => radio.getAttribute('value'))).toEqual([
      'oauth2',
      'password',
      'client_credentials',
    ]);
  });

  it('includes an OAuth2 fallback password in create payloads', async () => {
    const user = userEvent.setup();

    await renderPanel();
    await openCreateRoute(user);
    await user.click(screen.getByRole('checkbox', { name: /require authentication/i }));
    await user.click(screen.getByRole('button', { name: /generate password/i }));
    await user.click(screen.getByRole('button', { name: /create route/i }));

    await waitFor(() => {
      expect(apiMock.createMcpRoute).toHaveBeenCalledWith(
        expect.objectContaining({
          auth_method: 'oauth2',
          auth_password: 'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH',
          require_auth: true,
        }),
      );
    });
  });

  it('shows the exact group bypass warning when OAuth2 password fallback and group restriction coexist', async () => {
    const user = userEvent.setup();

    await renderPanel({ ldapConfigured: true });
    await openCreateRoute(user);
    await user.click(screen.getByRole('checkbox', { name: /require authentication/i }));
    await user.selectOptions(screen.getByRole('combobox'), LDAP_GROUPS[0].dn);
    await user.click(screen.getByRole('button', { name: /generate password/i }));

    expect(screen.getByLabelText('MCP Password Fallback (Optional)')).toBeTruthy();
    expect(
      screen.getByText('MCP-Password bypasses this group restriction.').closest('.field-warning'),
    ).toBeTruthy();
  });

  it('clears an OAuth2 fallback password in update payloads when requested', async () => {
    const user = userEvent.setup();

    await renderPanel({
      routes: [
        makeRoute({
          id: 'route-oauth-password',
          has_password: true,
          auth_password: 'storedpass',
          allowed_ldap_group: LDAP_GROUPS[0].dn,
        }),
      ],
      ldapConfigured: true,
    });

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    await user.click(screen.getByRole('button', { name: 'Clear' }));
    await user.click(screen.getByRole('button', { name: /update route/i }));

    await waitFor(() => {
      expect(apiMock.updateMcpRoute).toHaveBeenCalledWith(
        'route-oauth-password',
        expect.objectContaining({
          auth_method: 'oauth2',
          clear_password: true,
        }),
      );
    });
  });

  it('renders OAuth2 + Password on route summaries when an OAuth2 route has a stored password', async () => {
    await renderPanel({
      routes: [
        makeRoute({
          id: 'route-oauth-password',
          name: 'OAuth Hybrid Route',
          has_password: true,
        }),
      ],
    });

    expect(screen.getByText('OAuth2 + Password')).toBeTruthy();
  });

  it('saves a classification requirement immediately for an existing route', async () => {
    const user = userEvent.setup();
    await renderPanel({ routes: [makeRoute()] });

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    const select = await screen.findByLabelText('Require classification');
    await user.selectOptions(select, 'require');

    await waitFor(() =>
      expect(contentProtectionMock.updateContentProtectionConfigSlice).toHaveBeenCalledOnce(),
    );
    expect(
      contentProtectionMock.updateContentProtectionConfigSlice.mock.calls[0][0](
        CONTENT_PROTECTION_CONFIG,
      ).requirements,
    ).toEqual([{ scope_kind: 'mcp_route', scope_key: 'route-1', mode: 'require' }]);
  });

  it('explains that a route must be saved before classification can be required', async () => {
    const user = userEvent.setup();
    await renderPanel();
    await user.click(screen.getByRole('button', { name: /add custom route/i }));

    expect(
      screen.getByText('Save the route first, then edit it to require classification.'),
    ).toBeTruthy();
    expect(screen.queryByLabelText('Require classification')).toBeNull();
  });

  it('hides route classification controls and the new-route hint when protection is disabled', async () => {
    const user = userEvent.setup();
    await renderPanel({
      routes: [makeRoute()],
      contentProtectionConfig: { ...CONTENT_PROTECTION_CONFIG, enabled: false },
    });

    await user.click(screen.getByRole('button', { name: 'Edit' }));
    await waitFor(() => expect(contentProtectionMock.getConfig).toHaveBeenCalled());
    expect(screen.queryByLabelText('Require classification')).toBeNull();
    expect(screen.queryByText(/Save the route first/)).toBeNull();
    expect(contentProtectionMock.updateContentProtectionConfigSlice).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: 'Cancel' }));
    await user.click(screen.getByRole('button', { name: /add custom route/i }));
    await waitFor(() => expect(contentProtectionMock.getConfig).toHaveBeenCalledTimes(2));
    expect(screen.queryByText(/Save the route first/)).toBeNull();
  });

  it('shows the route skill count and a Skills button first in the card actions', async () => {
    await renderPanel({
      routes: [makeRoute()],
      skillSummary: { tools: {}, routes: { 'route-1': { total: 2, enabled: 1 } } },
    });

    const count = screen.getByText('2 route skills');
    expect(count.hasAttribute('data-mcp-route-skills-count')).toBe(true);
    expect(count.closest('.tool-card-meta')).toBeTruthy();

    const actions = document.querySelector('#mcp-route-route-1 .tool-card-actions') as HTMLElement;
    const buttons = Array.from(actions.querySelectorAll('button'));
    expect(buttons[0].textContent).toBe('Skills');
    expect(buttons[0].hasAttribute('data-mcp-route-skills-button')).toBe(true);
    expect(buttons[0].classList.contains('btn-secondary')).toBe(true);
  });

  it('uses the singular label for a single route skill', async () => {
    await renderPanel({
      routes: [makeRoute()],
      skillSummary: { tools: {}, routes: { 'route-1': { total: 1, enabled: 1 } } },
    });

    expect(screen.getByText('1 route skill')).toBeTruthy();
  });

  it('hides the route skill count when the route has no skills', async () => {
    await renderPanel({ routes: [makeRoute()] });

    expect(screen.queryByText(/route skill/)).toBeNull();
    expect(screen.getByRole('button', { name: 'Skills' })).toBeTruthy();
  });

  it('gives each route card a stable id hook', async () => {
    await renderPanel({ routes: [makeRoute()] });

    expect(document.getElementById('mcp-route-route-1')?.classList.contains('tool-card')).toBe(
      true,
    );
  });

  it('opens the skills dialog with the route as owner', async () => {
    const user = userEvent.setup();
    await renderPanel({ routes: [makeRoute()] });

    await user.click(screen.getByRole('button', { name: 'Skills' }));

    const dialog = await screen.findByRole('dialog', { name: 'Skills · Custom Route' });
    expect(dialog.getAttribute('data-owner-kind')).toBe('route');
    expect(dialog.getAttribute('data-owner-id')).toBe('route-1');
  });

  it('refreshes the route count from the skills the dialog reports', async () => {
    const user = userEvent.setup();
    skillsDialogMock.nextSkills = [skillFixture('one', true), skillFixture('two', false)];
    await renderPanel({
      routes: [makeRoute()],
      skillSummary: { tools: {}, routes: { 'route-1': { total: 1, enabled: 1 } } },
    });

    await user.click(screen.getByRole('button', { name: 'Skills' }));
    await user.click(await screen.findByRole('button', { name: 'Save Skills Mock' }));

    expect(await screen.findByText('2 route skills')).toBeTruthy();
  });

  it('keeps the route list usable when the skill summary cannot be loaded', async () => {
    await renderPanel({ routes: [makeRoute()], skillSummary: null });

    expect(screen.getByText('Custom Route')).toBeTruthy();
    expect(screen.queryByText(/route skill/)).toBeNull();
    expect(toastMock.error).not.toHaveBeenCalled();
  });

  it('renders the default section before custom routes without LDAP', async () => {
    await renderPanel();
    const section = document.getElementById('mcp-default-route-section')!;
    expect(section.querySelector('#mcp-default-route')).toBeTruthy();
    expect(
      section.compareDocumentPosition(document.getElementById('mcp-custom-routes-heading')!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('hides the default section in the route wizard', async () => {
    const user = userEvent.setup();
    await renderPanel();
    await user.click(screen.getByRole('button', { name: /add custom route/i }));
    expect(document.getElementById('mcp-default-route-section')).toBeNull();
  });

  it('shows the zero skills state', async () => {
    await renderPanel();
    expect(
      screen.getByText('No skills yet').hasAttribute('data-mcp-default-route-skills-count'),
    ).toBe(true);
  });

  it('formats singular and partially enabled counts', async () => {
    await renderPanel({
      skillSummary: { tools: {}, routes: { default: { total: 3, enabled: 2 } } },
    });
    expect(screen.getByText('3 skills, 2 of 3 enabled')).toBeTruthy();
    skillsDialogMock.nextSkills = [skillFixture('one', true)];
    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Skills for the default /mcp route' }));
    await user.click(screen.getByRole('button', { name: 'Save Skills Mock' }));
    expect(screen.getByText('1 skill')).toBeTruthy();
  });

  it('shows the eligibility helper', async () => {
    await renderPanel();
    expect(
      screen.getByText(
        "Skills reach only clients that can use the general tool catalog. Workspace-only development credentials (such as rtdev_ credentials) and sign-ins outside the default route's allowed group don't receive them. Who can connect is set in Settings → MCP Configuration.",
      ),
    ).toBeTruthy();
  });

  it('opens the default route skills dialog', async () => {
    const user = userEvent.setup();
    await renderPanel();
    await user.click(screen.getByRole('button', { name: 'Skills for the default /mcp route' }));
    const dialog = screen.getByRole('dialog', { name: 'Skills · Default /mcp route' });
    expect(dialog.getAttribute('data-owner-kind')).toBe('route');
    expect(dialog.getAttribute('data-owner-id')).toBe('default');
  });

  it('refreshes the default count without affecting custom counts', async () => {
    const user = userEvent.setup();
    skillsDialogMock.nextSkills = [skillFixture('one', true), skillFixture('two', false)];
    await renderPanel({
      routes: [makeRoute()],
      skillSummary: { tools: {}, routes: { 'route-1': { total: 1, enabled: 1 } } },
    });
    await user.click(screen.getByRole('button', { name: 'Skills for the default /mcp route' }));
    await user.click(screen.getByRole('button', { name: 'Save Skills Mock' }));
    expect(screen.getByText('2 skills, 1 of 2 enabled')).toBeTruthy();
    expect(screen.getByText('1 route skill')).toBeTruthy();
  });

  it('shows unavailable count without a toast on summary failure', async () => {
    await renderPanel({ skillSummary: null });
    expect(screen.getByText('Skill count unavailable')).toBeTruthy();
    expect(toastMock.error).not.toHaveBeenCalled();
  });

  it('keeps the default count unavailable after saving custom route skills', async () => {
    const user = userEvent.setup();
    await renderPanel({ routes: [makeRoute()], rejectSkillSummary: true });
    expect(screen.getByText('Skill count unavailable')).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Skills' }));
    await user.click(screen.getByRole('button', { name: 'Save Skills Mock' }));
    expect(screen.getByText('Skill count unavailable')).toBeTruthy();
  });

  it('shows zero enabled when all default skills are disabled', async () => {
    await renderPanel({
      skillSummary: { tools: {}, routes: { default: { total: 3, enabled: 0 } } },
    });
    expect(screen.getByText('3 skills, 0 of 3 enabled')).toBeTruthy();
  });

  it('returns focus to the default card when the opener loses focus', async () => {
    const user = userEvent.setup();
    await renderPanel();
    await user.click(screen.getByRole('button', { name: 'Skills for the default /mcp route' }));
    await user.click(screen.getByRole('button', { name: 'Close Skills Mock' }));
    await waitFor(() => expect(document.activeElement?.id).toBe('mcp-default-route'));
  });
});
