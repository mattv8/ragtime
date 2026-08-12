import { useEffect } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ChatModelsSettingsSectionProps } from './settings/ChatModelsSettingsSection';

const modalRenderSpy = vi.hoisted(() => vi.fn());
const searchFilterBarSpy = vi.hoisted(() => vi.fn());
const searchFilterState = vi.hoisted(() => ({ queries: [] as string[] }));
const toastSuccessSpy = vi.hoisted(() => vi.fn());
const toastErrorSpy = vi.hoisted(() => vi.fn());
const toastInfoSpy = vi.hoisted(() => vi.fn());
const toastClearSpy = vi.hoisted(() => vi.fn());
const toastDismissSpy = vi.hoisted(() => vi.fn());
const toastActions = vi.hoisted(() => ({
  success: toastSuccessSpy,
  error: toastErrorSpy,
  info: toastInfoSpy,
  clear: toastClearSpy,
  dismiss: toastDismissSpy,
}));
const modalState = vi.hoisted(() => ({
  latestAllowedChatModelsProps: null as null | {
    title?: string;
    allModels?: Array<Record<string, unknown>>;
  },
}));
const chatModelsSectionState = vi.hoisted(() => ({
  latestProps: null as null | ChatModelsSettingsSectionProps,
  openedModal: false,
  autoOpenModal: true,
}));
const serverBackupSectionSpy = vi.hoisted(() => vi.fn());
const mcpSectionState = vi.hoisted(() => ({
  latestProps: null as null | Record<string, unknown>,
  run: null as null | ((props: Record<string, unknown>) => void),
}));
const agentBehaviorSectionState = vi.hoisted(() => ({
  latestProps: null as null | Record<string, unknown>,
}));
const searchSectionState = vi.hoisted(() => ({
  latestProps: null as null | Record<string, unknown>,
  run: null as null | ((props: Record<string, unknown>) => void),
}));
const sectionRenderOrder = vi.hoisted(() => [] as string[]);

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
  ChatModelsSettingsSection: (props: ChatModelsSettingsSectionProps) => {
    chatModelsSectionState.latestProps = props;
    sectionRenderOrder.push('chat-models');
    const { openModelFilterModal } = props;

    useEffect(() => {
      if (chatModelsSectionState.autoOpenModal && !chatModelsSectionState.openedModal) {
        chatModelsSectionState.openedModal = true;
        void openModelFilterModal();
      }
    }, [openModelFilterModal]);

    return (
      <section data-settings-accordion-section="chat-models">
        <span className="settings-accordion-title">Chat Models</span>
        <button type="button">Open chat models</button>
      </section>
    );
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
  useToast: () => [[], toastActions] as const,
}));

vi.mock('./settings/AgentBehaviorSettingsSection', () => ({
  AgentBehaviorSettingsSection: (props: Record<string, unknown>) => {
    agentBehaviorSectionState.latestProps = props;
    sectionRenderOrder.push('agent-behavior');
    const formData = (props.formData as Record<string, unknown>) || {};
    const setFormData = props.setFormData as (value: unknown) => void;

    return (
      <section data-settings-accordion-section="agent-behavior">
        <button type="button" aria-expanded={Boolean(props.open)}>
          Agent Behavior
        </button>
        <fieldset>
          <legend>Agent Behavior</legend>
          <div className="form-group" id="setting-tool_skills_enabled">
            <label htmlFor="agent-behavior-tool-skills-enabled">Load tools on demand</label>
            <input
              id="agent-behavior-tool-skills-enabled"
              type="checkbox"
              checked={formData.tool_skills_enabled !== false}
              onChange={(event) =>
                setFormData((prev: Record<string, unknown>) => ({
                  ...prev,
                  tool_skills_enabled: event.target.checked,
                }))
              }
            />
          </div>
          <div className="form-group">
            <label htmlFor="agent-behavior-max-iterations">Max Tool Iterations</label>
            <input
              id="agent-behavior-max-iterations"
              type="range"
              aria-label="Max Tool Iterations"
              min="1"
              max="100"
              value={Number(formData.max_iterations ?? 30)}
              onChange={(event) =>
                setFormData((prev: Record<string, unknown>) => ({
                  ...prev,
                  max_iterations: parseInt(event.target.value, 10),
                }))
              }
            />
          </div>
          <div className="form-group">
            <label htmlFor="agent-behavior-max-tool-output">Max Tool Output (chars)</label>
            <input
              id="agent-behavior-max-tool-output"
              type="range"
              aria-label="Max Tool Output (chars)"
              min="0"
              max="50000"
              value={Number(formData.max_tool_output_chars ?? 5000)}
              onChange={(event) =>
                setFormData((prev: Record<string, unknown>) => ({
                  ...prev,
                  max_tool_output_chars: parseInt(event.target.value, 10),
                }))
              }
            />
          </div>
          <div className="form-group">
            <label htmlFor="agent-behavior-context-window">Context Window (steps)</label>
            <input
              id="agent-behavior-context-window"
              type="range"
              aria-label="Context Window (steps)"
              min="0"
              max="30"
              value={Number(formData.scratchpad_window_size ?? 6)}
              onChange={(event) =>
                setFormData((prev: Record<string, unknown>) => ({
                  ...prev,
                  scratchpad_window_size: parseInt(event.target.value, 10),
                }))
              }
            />
          </div>
          <button
            type="button"
            onClick={() =>
              void (props.handleSaveAgentBehavior as (() => void | Promise<void>) | undefined)?.()
            }
          >
            Save Agent Behavior
          </button>
        </fieldset>
      </section>
    );
  },
}));

vi.mock('./settings/SearchSettingsSection', () => ({
  SearchSettingsSection: (props: Record<string, unknown>) => {
    searchSectionState.latestProps = props;
    sectionRenderOrder.push('search');
    useEffect(() => {
      searchSectionState.run?.(props);
    }, [props]);

    const formData = (props.formData as Record<string, unknown>) || {};
    const settings = (props.settings as Record<string, unknown> | null) || null;

    return (
      <fieldset>
        <legend>Search Configuration</legend>
        <div className="form-group" id="setting-faiss_search_concurrency_mode">
          <label htmlFor="search-faiss-concurrency-mode">Global FAISS search gate</label>
          <input
            id="search-faiss-concurrency-mode"
            type="checkbox"
            aria-label="Global FAISS search gate"
            checked={
              (formData.faiss_search_concurrency_mode ??
                settings?.faiss_search_concurrency_mode ??
                'per_index') === 'global'
            }
            onChange={(event) =>
              (props.setFormData as (value: unknown) => void)((prev: Record<string, unknown>) => ({
                ...prev,
                faiss_search_concurrency_mode: event.target.checked ? 'global' : 'per_index',
              }))
            }
          />
        </div>
        <button
          type="button"
          onClick={() =>
            void (props.handleSaveSearch as (() => void | Promise<void>) | undefined)?.()
          }
        >
          Save Search Configuration
        </button>
        <details>
          <summary>Advanced Settings</summary>
          <div className="form-group">
            <label>Results per Search (k)</label>
          </div>
          <div className="form-group">
            <label>Archive Max Size</label>
          </div>
        </details>
      </fieldset>
    );
  },
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
    sectionRenderOrder.push('mcp');
    useEffect(() => {
      mcpSectionState.run?.(props);
    }, [props]);

    return (
      <section data-settings-accordion-section="mcp">
        <span className="settings-accordion-title">MCP Configuration</span>
      </section>
    );
  },
}));
vi.mock('./shared/SearchFilterBar', () => ({
  SearchFilterBar: (props: unknown) => {
    searchFilterBarSpy(props);
    return null;
  },
  normalizeSearchFilterText: (value: string) => value,
  searchFilterTextMatchesQuery: () => true,
  useUrlSearchFilterState: () => searchFilterState,
}));
vi.mock('./shared/AuthAdminModals', () => ({ AuthAdminModalHost: () => null }));

function buildSettingsResponse(
  overrides: Record<string, unknown> = {},
  options: { omitToolSkillsEnabled?: boolean } = {},
): { settings: Record<string, unknown> } {
  const settings: Record<string, unknown> = {
    llm_provider: 'openai',
    default_theme_pack: 'default',
    has_openai_codex_auth: true,
    has_claude_code_auth: true,
    authenticated_webgl_background_enabled: true,
    tool_skills_enabled: true,
    openapi_model_prefix_enabled: true,
    show_tool_card_footer_actions: false,
    mcp_enabled: true,
    mcp_default_route_auth: true,
    mcp_default_route_auth_method: 'oauth2',
    mcp_default_route_allowed_group: null,
    mcp_default_route_client_id: '',
    has_mcp_default_password: true,
    mcp_default_route_password: '',
    faiss_search_concurrency_mode: 'per_index',
    updated_at: null,
    ...overrides,
  };

  if (options.omitToolSkillsEnabled) {
    delete settings.tool_skills_enabled;
  }

  return { settings };
}

beforeEach(() => {
  modalRenderSpy.mockClear();
  searchFilterBarSpy.mockClear();
  serverBackupSectionSpy.mockClear();
  modalState.latestAllowedChatModelsProps = null;
  chatModelsSectionState.latestProps = null;
  chatModelsSectionState.openedModal = false;
  chatModelsSectionState.autoOpenModal = true;
  mcpSectionState.latestProps = null;
  mcpSectionState.run = null;
  agentBehaviorSectionState.latestProps = null;
  searchSectionState.latestProps = null;
  searchSectionState.run = null;
  sectionRenderOrder.length = 0;
  toastSuccessSpy.mockClear();
  toastErrorSpy.mockClear();
  toastInfoSpy.mockClear();
  toastClearSpy.mockClear();
  toastDismissSpy.mockClear();

  apiMock.getSettings.mockResolvedValue(buildSettingsResponse());
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
  apiMock.updateSettings.mockImplementation(async (payload: Record<string, unknown>) => {
    const current = buildSettingsResponse().settings;
    return {
      ...current,
      ...payload,
      tool_skills_enabled:
        typeof payload.tool_skills_enabled === 'boolean'
          ? payload.tool_skills_enabled
          : current.tool_skills_enabled,
      max_iterations:
        typeof payload.max_iterations === 'number'
          ? payload.max_iterations
          : current.max_iterations,
      max_tool_output_chars:
        typeof payload.max_tool_output_chars === 'number'
          ? payload.max_tool_output_chars
          : current.max_tool_output_chars,
      scratchpad_window_size:
        typeof payload.scratchpad_window_size === 'number'
          ? payload.scratchpad_window_size
          : current.scratchpad_window_size,
      mcp_default_route_password:
        typeof payload.mcp_default_route_password === 'string'
          ? payload.mcp_default_route_password
          : current.mcp_default_route_password,
      has_mcp_default_password:
        typeof payload.mcp_default_route_password === 'string'
          ? payload.mcp_default_route_password !== ''
          : current.has_mcp_default_password,
      faiss_search_concurrency_mode:
        payload.faiss_search_concurrency_mode === 'global'
          ? 'global'
          : current.faiss_search_concurrency_mode,
      updated_at: null,
    };
  });
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

    await waitFor(
      () => {
        expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
      },
      { timeout: 3000 },
    );

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        mcp_default_route_auth_method: 'oauth2',
        mcp_default_route_password: 'fallback-pass1',
      }),
    );
  }, 10000);

  it('retains an existing OAuth2 fallback password when the admin saves without changing it', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    mcpSectionState.run = (props) => {
      if (!props.settings || apiMock.updateSettings.mock.calls.length > 0) {
        return;
      }
      void (props.handleSaveMcp as () => Promise<void>)();
    };

    render(<SettingsPanel />);

    await waitFor(
      () => {
        expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
      },
      { timeout: 3000 },
    );

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).not.toHaveProperty(
      'mcp_default_route_password',
    );
  }, 10000);

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

    await waitFor(
      () => {
        expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
      },
      { timeout: 3000 },
    );

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({
        mcp_default_route_auth_method: 'oauth2',
        mcp_default_route_password: '',
      }),
    );
  }, 10000);

  it('hydrates and saves the FAISS concurrency mode through the search configuration flow', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');
    let stage: 'wait-settings' | 'toggle' | 'save' | 'done' = 'wait-settings';

    searchSectionState.run = (props) => {
      if (!props.settings || stage === 'done') {
        return;
      }
      const formData = (props.formData as Record<string, unknown>) || {};
      if (stage === 'wait-settings') {
        expect(props.settings).toEqual(
          expect.objectContaining({ faiss_search_concurrency_mode: 'per_index' }),
        );
        expect(formData.faiss_search_concurrency_mode).toBe('per_index');
        stage = 'toggle';
        setTimeout(() => {
          (props.setFormData as (value: unknown) => void)((prev: Record<string, unknown>) => ({
            ...prev,
            faiss_search_concurrency_mode: 'global',
          }));
        }, 0);
        return;
      }
      if (stage === 'toggle' && formData.faiss_search_concurrency_mode === 'global') {
        stage = 'save';
        void (props.handleSaveSearch as () => Promise<void>)();
        return;
      }
      if (stage === 'save') {
        stage = 'done';
      }
    };

    render(<SettingsPanel />);

    await waitFor(
      () => {
        expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
      },
      { timeout: 3000 },
    );

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).toEqual(
      expect.objectContaining({ faiss_search_concurrency_mode: 'global' }),
    );
  }, 10000);

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

  it('defaults the load-tools-on-demand toggle to checked when the API omits the value', async () => {
    apiMock.getSettings.mockResolvedValue(
      buildSettingsResponse({}, { omitToolSkillsEnabled: true }),
    );

    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    const checkbox = (await screen.findByLabelText('Load tools on demand')) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);
  });

  it('renders the load-tools-on-demand toggle as unchecked when the API returns false', async () => {
    apiMock.getSettings.mockResolvedValue(buildSettingsResponse({ tool_skills_enabled: false }));

    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    await screen.findByRole('button', { name: 'Open chat models' });
    const checkbox = (await screen.findByLabelText('Load tools on demand')) as HTMLInputElement;
    expect(checkbox.checked).toBe(false);
  });

  it('updates the load-tools-on-demand toggle state when clicked', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    const checkbox = (await screen.findByLabelText('Load tools on demand')) as HTMLInputElement;
    expect(checkbox.checked).toBe(true);

    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(false);

    fireEvent.click(checkbox);
    expect(checkbox.checked).toBe(true);
  });

  it('renders the load-tools-on-demand toggle in its own row with one clear label association', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    const checkbox = (await screen.findByLabelText('Load tools on demand')) as HTMLInputElement;
    const section = checkbox.closest('#setting-tool_skills_enabled');
    const formGroup = checkbox.closest('.form-group');

    expect(section).toBeTruthy();
    expect(section?.textContent).not.toContain('Image Max Bytes');
    expect(section?.textContent).not.toContain('Max Tool Output (chars)');
    expect(checkbox.id).toBe('agent-behavior-tool-skills-enabled');
    expect(
      formGroup?.querySelector('label[for="agent-behavior-tool-skills-enabled"]'),
    ).toBeTruthy();
  });

  it('renders Agent Behavior after Chat Models and before MCP, closed by default, with the expected wiring', async () => {
    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    await waitFor(() => {
      expect(chatModelsSectionState.latestProps).toBeTruthy();
      expect(mcpSectionState.latestProps).toBeTruthy();
      expect(agentBehaviorSectionState.latestProps).toBeTruthy();
    });

    const agentBehaviorHeader = await screen.findByRole('button', { name: 'Agent Behavior' });

    expect(sectionRenderOrder.slice(0, 3)).toEqual(['chat-models', 'agent-behavior', 'mcp']);
    expect(agentBehaviorHeader.getAttribute('aria-expanded')).toBe('false');
    expect(agentBehaviorSectionState.latestProps).toEqual(
      expect.objectContaining({
        open: false,
        agentBehaviorSaving: false,
      }),
    );
  });

  it('saves Agent Behavior with exactly the isolated four-field payload and success feedback', async () => {
    apiMock.getSettings.mockResolvedValue(buildSettingsResponse({ tool_skills_enabled: false }));

    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    await waitFor(() => {
      expect(agentBehaviorSectionState.latestProps).toBeTruthy();
    });

    apiMock.updateSettings.mockClear();

    const setFormData = agentBehaviorSectionState.latestProps?.setFormData as
      | ((value: unknown) => void)
      | undefined;

    setFormData?.((prev: Record<string, unknown>) => ({
      ...prev,
      tool_skills_enabled: true,
      max_iterations: 44,
      max_tool_output_chars: 12000,
      scratchpad_window_size: 9,
    }));
    await waitFor(() => {
      expect(agentBehaviorSectionState.latestProps?.formData).toEqual(
        expect.objectContaining({
          tool_skills_enabled: true,
          max_iterations: 44,
          max_tool_output_chars: 12000,
          scratchpad_window_size: 9,
        }),
      );
    });
    await (
      agentBehaviorSectionState.latestProps?.handleSaveAgentBehavior as
        | (() => Promise<void>)
        | undefined
    )?.();

    await waitFor(() => {
      expect(apiMock.updateSettings).toHaveBeenCalledTimes(1);
    });

    expect(apiMock.updateSettings.mock.calls[0]?.[0]).toEqual({
      tool_skills_enabled: true,
      max_iterations: 44,
      max_tool_output_chars: 12000,
      scratchpad_window_size: 9,
    });
    expect(toastSuccessSpy).toHaveBeenCalledWith('Agent behavior settings saved');
  });

  it('does not include agent behavior fields in the LLM save payload', async () => {
    apiMock.getSettings.mockResolvedValue(
      buildSettingsResponse({
        tool_skills_enabled: false,
        max_iterations: 17,
        max_tool_output_chars: 7777,
        scratchpad_window_size: 4,
      }),
    );

    const { SettingsPanel } = await import('./SettingsPanel');

    render(<SettingsPanel />);

    await screen.findByRole('button', { name: 'Open chat models' });
    await (
      chatModelsSectionState.latestProps?.handleSaveLlm as (() => Promise<void>) | undefined
    )?.();

    await waitFor(() => {
      expect(apiMock.updateSettings.mock.calls.length).toBeGreaterThan(0);
    });

    const latestPayload =
      apiMock.updateSettings.mock.calls[apiMock.updateSettings.mock.calls.length - 1]?.[0];
    expect(latestPayload).not.toHaveProperty('tool_skills_enabled');
    expect(latestPayload).not.toHaveProperty('max_iterations');
    expect(latestPayload).not.toHaveProperty('max_tool_output_chars');
    expect(latestPayload).not.toHaveProperty('scratchpad_window_size');
  });
});
