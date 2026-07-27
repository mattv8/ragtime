import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { MCPRoutesPanel } from './MCPRoutesPanel';
import type { IndexInfo, McpRouteConfig, ToolConfig } from '@/types';

const apiMock = vi.hoisted(() => ({
  listMcpRoutes: vi.fn(),
  listMcpDefaultFilters: vi.fn(),
  listToolConfigs: vi.fn(),
  listIndexes: vi.fn(),
  getSettings: vi.fn(),
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

vi.mock('./shared/Toast', () => ({
  useToast: () => [[], toastMock] as const,
  ToastContainer: () => null,
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

async function renderPanel({
  routes = [],
  ldapConfigured = false,
}: {
  routes?: McpRouteConfig[];
  ldapConfigured?: boolean;
} = {}) {
  apiMock.listMcpRoutes.mockResolvedValue({ routes, count: routes.length });
  apiMock.listMcpDefaultFilters.mockResolvedValue({ filters: [], count: 0 });
  apiMock.listToolConfigs.mockResolvedValue(TOOL_CONFIGS);
  apiMock.listIndexes.mockResolvedValue(INDEXES);
  apiMock.getSettings.mockResolvedValue({
    settings: {
      aggregate_search: true,
      mcp_default_route_auth: false,
      mcp_default_route_auth_method: 'password',
    },
  });

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
});
