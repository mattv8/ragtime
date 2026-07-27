import { useEffect } from 'react';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const modalRenderSpy = vi.hoisted(() => vi.fn());
const searchFilterBarSpy = vi.hoisted(() => vi.fn());
const modalState = vi.hoisted(() => ({
  latestAllowedChatModelsProps: null as null | {
    title?: string;
    allModels?: Array<Record<string, unknown>>;
  },
}));
const serverBackupSectionSpy = vi.hoisted(() => vi.fn());
const mcpSectionState = vi.hoisted(() => ({
  latestProps: null as null | Record<string, unknown>,
  run: null as null | ((props: Record<string, unknown>) => void),
}));

const apiMock = vi.hoisted(() => ({
  getSettings: vi.fn(),
  getUserSpacePreviewSettings: vi.fn(),
  getAuthProviderConfig: vi.fn(),
  listAuthGroups: vi.fn(),
  getCopilotAuthStatus: vi.fn(),
  getOpenAICodexAuthStatus: vi.fn(),
  getClaudeCodeAuthStatus: vi.fn(),
  getAvailableModels: vi.fn(),
  getAllModels: vi.fn(),
  fetchLLMModels: vi.fn(),
  listMcpRoutes: vi.fn(),
  listCloudOAuthProviders: vi.fn(),
  getLdapConfig: vi.fn(),
  updateSettings: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('@/contexts/AvailableModelsContext', () => ({
  useAvailableModels: () => ({ refresh: vi.fn() }),
}));

vi.mock('./settings/ChatModelsSettingsSection', () => ({
  ChatModelsSettingsSection: ({ openModelFilterModal }: { openModelFilterModal: () => void }) => {
    useEffect(() => {
      void openModelFilterModal();
    }, [openModelFilterModal]);

    return <button type="button">Open chat models</button>;
  },
}));

vi.mock('./settings/ServerBackupRestoreSettingsSection', () => ({
  ServerBackupRestoreSettingsSection: (props: unknown) => {
    serverBackupSectionSpy(props);
    return (
      <fieldset>
        <legend>Server Backup & Restore</legend>
        <div className="form-group">
          <label>Encrypt backup archive</label>
        </div>
      </fieldset>
    );
  },
}));

vi.mock('./ModelFilterModal', () => ({
  ModelFilterModal: (props: unknown) => {
    modalRenderSpy(props);
    if (
      props &&
      typeof props === 'object' &&
      'title' in props &&
      (props as { title?: string }).title === 'Allowed Chat Models'
    ) {
      modalState.latestAllowedChatModelsProps = props as {
        title?: string;
        allModels?: Array<Record<string, unknown>>;
      };
    }
    return null;
  },
}));

vi.mock('./shared/Toast', () => ({
  ToastContainer: () => null,
  useToast: () => [[], { success: vi.fn(), error: vi.fn(), clear: vi.fn() }] as const,
}));

vi.mock('./settings/SearchSettingsSection', () => ({
  SearchSettingsSection: () => (
    <fieldset>
      <legend>Search Configuration</legend>
      <div className="form-group">
        <label>Results per Search (k)</label>
      </div>
      <details>
        <summary>Advanced Settings</summary>
        <div className="form-group">
          <label>Archive Max Size</label>
        </div>
      </details>
    </fieldset>
  ),
}));
vi.mock('./settings/AppearanceSettingsSection', () => ({
  AppearanceSettingsSection: () => (
    <fieldset>
      <legend>Appearance</legend>
      <div className="form-group">
        <label>Theme pack</label>
      </div>
    </fieldset>
  ),
}));
vi.mock('./settings/SecuritySettingsSection', () => ({
  SecuritySettingsSection: () => (
    <fieldset>
      <legend>Security</legend>
      <div className="form-group">
        <label>Minimum Password Length</label>
      </div>
    </fieldset>
  ),
}));
vi.mock('./settings/McpSettingsSection', () => ({
  McpSettingsSection: (props: Record<string, unknown>) => {
    mcpSectionState.latestProps = props;
    useEffect(() => {
      mcpSectionState.run?.(props);
    }, [props]);

    return null;
  },
}));
vi.mock('./shared/SearchFilterBar', () => ({
  SearchFilterBar: (props: unknown) => {
    searchFilterBarSpy(props);
    return null;
  },
  normalizeSearchFilterText: (value: string) => value,
  searchFilterTextMatchesQuery: () => true,
  useUrlSearchFilterState: () => ({ queries: [] }),
}));
vi.mock('./shared/AuthAdminModals', () => ({ AuthAdminModalHost: () => null }));

beforeEach(() => {
  modalRenderSpy.mockClear();
  searchFilterBarSpy.mockClear();
  serverBackupSectionSpy.mockClear();
  modalState.latestAllowedChatModelsProps = null;
  mcpSectionState.latestProps = null;
  mcpSectionState.run = null;

  apiMock.getSettings.mockResolvedValue({
    settings: {
      llm_provider: 'openai',
      default_theme_pack: 'default',
      has_openai_codex_auth: true,
      has_claude_code_auth: true,
      authenticated_webgl_background_enabled: true,
      openapi_model_prefix_enabled: true,
      show_tool_card_footer_actions: false,
      mcp_enabled: true,
      mcp_default_route_auth: true,
      mcp_default_route_auth_method: 'oauth2',
      mcp_default_route_allowed_group: null,
      mcp_default_route_client_id: '',
      has_mcp_default_password: true,
      mcp_default_route_password: '',
      updated_at: null,
    },
  });
  apiMock.getUserSpacePreviewSettings.mockResolvedValue({});
  apiMock.getAuthProviderConfig.mockResolvedValue({ provider: 'local_managed' });
  apiMock.listAuthGroups.mockResolvedValue([]);
  apiMock.getCopilotAuthStatus.mockResolvedValue({
    connected: false,
    base_url: '',
    enterprise_url: null,
  });
  apiMock.getOpenAICodexAuthStatus.mockResolvedValue({
    connected: true,
    base_url: 'https://codex.example.com',
  });
  apiMock.getClaudeCodeAuthStatus.mockResolvedValue({ connected: true });
  apiMock.getAvailableModels.mockResolvedValue({ automatic_default_model: null });
  apiMock.getAllModels.mockResolvedValue({
    models: [
      {
        id: 'gpt-5.6-terra',
        name: 'gpt-5.6-terra',
        provider: 'openai_codex',
        context_limit: 400000,
        display_name: 'gpt-5.6-terra',
      },
      {
        id: 'claude-sonnet-4-6',
        name: 'claude-sonnet-4-6',
        provider: 'claude_code',
        context_limit: 200000,
        display_name: 'claude-sonnet-4-6',
      },
    ],
    default_model: null,
    current_model: null,
    allowed_models: [],
    allowed_openapi_models: [],
  });
  apiMock.fetchLLMModels.mockResolvedValue({ success: true, models: [] });
  apiMock.listMcpRoutes.mockResolvedValue({ routes: [] });
  apiMock.listCloudOAuthProviders.mockResolvedValue([]);
  apiMock.getLdapConfig.mockResolvedValue({ server_url: '', allow_self_signed: false });
  apiMock.updateSettings.mockImplementation(async (payload: Record<string, unknown>) => ({
    id: 'settings-1',
    server_name: 'Ragtime',
    default_theme_pack: 'default',
    authenticated_webgl_background_enabled: true,
    openapi_model_prefix_enabled: true,
    show_tool_card_footer_actions: false,
    llm_provider: 'openai',
    mcp_enabled: true,
    mcp_default_route_auth: true,
    mcp_default_route_auth_method: 'oauth2',
    mcp_default_route_allowed_group: null,
    mcp_default_route_client_id: '',
    has_mcp_default_password: payload.mcp_default_route_password !== '',
    mcp_default_route_password:
      typeof payload.mcp_default_route_password === 'string'
        ? payload.mcp_default_route_password
        : '',
    updated_at: null,
  }));
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('SettingsPanel', () => {
  it('renders the authentication provider selector as a form field, not an action row', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    const selector = await screen.findByDisplayValue('Internal Users');
    const wrapper = selector.closest('div');

    expect(wrapper?.classList.contains('form-group')).toBe(true);
    expect(wrapper?.classList.contains('form-actions')).toBe(false);
  });

  it('does not render an API-key warning from a stale unauthenticated status', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    render(
      <SettingsPanel
        authStatus={{
          authenticated: false,
          ldap_configured: false,
          local_admin_enabled: true,
          debug_mode: false,
          api_key_configured: false,
          session_cookie_secure: false,
          allowed_origins_open: false,
        }}
      />,
    );

    await screen.findByRole('button', { name: 'Open chat models' });

    await waitFor(() => {
      expect(document.querySelector('.field-warning')?.textContent).not.toContain(
        'The API endpoint accepts an API Key for authentication',
      );
    });
  });

  it('preserves enriched Codex and Claude Code labels when refreshing the allowed chat models modal', async () => {
    apiMock.fetchLLMModels.mockImplementation(({ provider }: { provider: string }) => {
      if (provider === 'openai_codex') {
        return Promise.resolve({
          success: true,
          models: [
            {
              id: 'gpt-5.6-terra',
              name: 'gpt-5.6-terra',
              provider: 'openai_codex',
              context_limit: 123456,
              display_name: '5.6 Terra',
              model_variant: '5.6 Terra',
              model_provider_label: 'OpenAI Codex',
              model_family: 'Codex',
              selector_label: 'OpenAI Codex / Codex / 5.6 Terra',
              host_provider_label: 'OpenAI Codex',
              group: 'Codex',
              is_latest: true,
            },
          ],
        });
      }

      if (provider === 'claude_code') {
        return Promise.resolve({
          success: true,
          models: [
            {
              id: 'claude-sonnet-4-6',
              name: 'claude-sonnet-4-6',
              provider: 'anthropic',
              context_limit: 654321,
              display_name: 'Sonnet 4.6',
              model_variant: 'Sonnet 4.6',
              model_provider_label: 'Anthropic',
              model_family: 'Claude',
              selector_label: 'Claude Code / Anthropic / Sonnet 4.6',
              host_provider_label: 'Claude Code',
              group: 'Claude',
            },
          ],
        });
      }

      return Promise.resolve({ success: true, models: [] });
    });

    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);
    await screen.findByRole('button', { name: 'Open chat models' });

    await waitFor(() => {
      expect(apiMock.fetchLLMModels).toHaveBeenCalledWith({ provider: 'openai_codex' });
      expect(apiMock.fetchLLMModels).toHaveBeenCalledWith({ provider: 'claude_code' });

      const allModels = modalState.latestAllowedChatModelsProps?.allModels ?? [];
      expect(allModels).toHaveLength(2);

      expect(allModels).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            id: 'gpt-5.6-terra',
            provider: 'openai_codex',
            context_limit: 400000,
            display_name: '5.6 Terra',
            model_variant: '5.6 Terra',
            model_provider_label: 'OpenAI Codex',
            model_family: 'Codex',
            selector_label: 'OpenAI Codex / Codex / 5.6 Terra',
            host_provider_label: 'OpenAI Codex',
          }),
          expect.objectContaining({
            id: 'claude-sonnet-4-6',
            provider: 'claude_code',
            context_limit: 200000,
            display_name: 'Sonnet 4.6',
            model_variant: 'Sonnet 4.6',
            model_provider_label: 'Anthropic',
            model_family: 'Claude',
            selector_label: 'Claude Code / Anthropic / Sonnet 4.6',
            host_provider_label: 'Claude Code',
          }),
        ]),
      );
    });
  });

  it('passes the server-backup accordion state and reminder callback through to the dedicated section', async () => {
    const onEncryptedArtifactDelivered = vi.fn();
    const onServerBackupJobObserved = vi.fn();
    const onServerRestoreJobObserved = vi.fn();
    const onServerOperationError = vi.fn();
    const { SettingsPanel } = await import('./SettingsPanel');

    render(
      <SettingsPanel
        currentUser={{
          id: 'user-1',
          username: 'local:admin',
          display_name: 'Admin',
          email: null,
          auth_provider: 'local_managed',
          role: 'admin',
        }}
        onEncryptedArtifactDelivered={onEncryptedArtifactDelivered}
        onServerBackupJobObserved={onServerBackupJobObserved}
        onServerRestoreJobObserved={onServerRestoreJobObserved}
        onServerOperationError={onServerOperationError}
      />,
    );

    await waitFor(() => {
      expect(serverBackupSectionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          open: false,
          onEncryptedArtifactDelivered,
          onServerBackupJobObserved,
          onServerRestoreJobObserved,
          onServerOperationError,
        }),
      );
    });
  });

  it('passes deduplicated rendered settings search candidates into the shared search bar', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);
    await screen.findByRole('button', { name: 'Open chat models' });

    await waitFor(() => {
      const latestCall = searchFilterBarSpy.mock.calls[
        searchFilterBarSpy.mock.calls.length - 1
      ]?.[0] as {
        completionCandidates?: string[];
      };

      expect(latestCall?.completionCandidates).toEqual(
        expect.arrayContaining([
          'Search Configuration',
          'Results per Search (k)',
          'Advanced Settings',
          'Appearance',
          'Theme pack',
          'Security',
          'Minimum Password Length',
          'Server Backup & Restore',
          'Encrypt backup archive',
          'OpenAI-Compatible API',
          'Cloud Drive OAuth',
        ]),
      );
      expect(
        latestCall?.completionCandidates?.filter((value) => value === 'Search Configuration'),
      ).toHaveLength(1);
    });
  });

  it('includes the OAuth2 fallback password in the MCP save payload when it is set', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');
    let stage: 'wait-settings' | 'set-password' | 'save' | 'done' = 'wait-settings';

    mcpSectionState.run = (props) => {
      if (!props.settings || stage === 'done') {
        return;
      }
      const formData = (props.formData as Record<string, unknown>) || {};
      if (stage === 'wait-settings') {
        stage = 'set-password';
        setTimeout(() => {
          (props.setFormData as (value: unknown) => void)((prev: Record<string, unknown>) => ({
            ...prev,
            mcp_enabled: true,
            mcp_default_route_auth: true,
            mcp_default_route_auth_method: 'oauth2',
            mcp_default_route_password: 'fallback-pass1',
          }));
        }, 0);
        return;
      }
      if (stage === 'set-password' && formData.mcp_default_route_password === 'fallback-pass1') {
        stage = 'save';
        void (props.handleSaveMcp as () => Promise<void>)();
        return;
      }
      if (stage === 'save') {
        stage = 'done';
      }
    };

    render(<SettingsPanel />);

    await waitFor(() => {
      expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
    });

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        mcp_default_route_auth_method: 'oauth2',
        mcp_default_route_password: 'fallback-pass1',
      }),
    );
  });

  it('retains an existing OAuth2 fallback password when the admin saves without changing it', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    mcpSectionState.run = (props) => {
      if (!props.settings || apiMock.updateSettings.mock.calls.length > 0) {
        return;
      }
      void (props.handleSaveMcp as () => Promise<void>)();
    };

    render(<SettingsPanel />);

    await waitFor(() => {
      expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
    });

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).not.toHaveProperty(
      'mcp_default_route_password',
    );
  });

  it('clears an existing OAuth2 fallback password when the admin removes it and saves', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');
    let stage: 'wait-settings' | 'clear-password' | 'save' | 'done' = 'wait-settings';

    mcpSectionState.run = (props) => {
      if (!props.settings || stage === 'done') {
        return;
      }
      const formData = (props.formData as Record<string, unknown>) || {};
      if (stage === 'wait-settings') {
        stage = 'clear-password';
        setTimeout(() => {
          (props.setFormData as (value: unknown) => void)((prev: Record<string, unknown>) => ({
            ...prev,
            mcp_enabled: true,
            mcp_default_route_auth: true,
            mcp_default_route_auth_method: 'oauth2',
            mcp_default_route_password: '',
          }));
        }, 0);
        return;
      }
      if (stage === 'clear-password' && formData.mcp_default_route_password === '') {
        stage = 'save';
        void (props.handleSaveMcp as () => Promise<void>)();
        return;
      }
      if (stage === 'save') {
        stage = 'done';
      }
    };

    render(<SettingsPanel />);

    await waitFor(() => {
      expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
    });

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        mcp_default_route_auth_method: 'oauth2',
        mcp_default_route_password: '',
      }),
    );
  });

  it('shows the updated default MCP route tooltip for OAuth2 routes with a password fallback', async () => {
    apiMock.listMcpRoutes.mockResolvedValue({
      routes: [
        {
          id: 'route-oauth',
          name: 'OAuth only',
          route_path: 'oauth_only',
          description: '',
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
          tool_config_ids: [],
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
        },
        {
          id: 'route-hybrid',
          name: 'OAuth hybrid',
          route_path: 'oauth_hybrid',
          description: '',
          enabled: true,
          require_auth: true,
          has_password: true,
          auth_password: undefined,
          auth_client_id: undefined,
          auth_method: 'oauth2',
          allowed_ldap_group: null,
          include_knowledge_search: false,
          include_git_history: false,
          selected_document_indexes: [],
          selected_filesystem_indexes: [],
          selected_schema_indexes: [],
          tool_config_ids: [],
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
        },
      ],
    });

    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);
    await screen.findByRole('button', { name: 'Open chat models' });

    await waitFor(() => {
      expect(screen.getByTitle('OAuth2 + Password protected')).toBeTruthy();
      expect(screen.queryByTitle('OAuth2 (LDAP)')).toBeNull();
    });
  });
});
