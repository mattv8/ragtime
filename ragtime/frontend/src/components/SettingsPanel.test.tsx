import { useEffect } from 'react';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const modalRenderSpy = vi.hoisted(() => vi.fn());
const modalState = vi.hoisted(() => ({
  latestAllowedChatModelsProps: null as null | {
    title?: string;
    allModels?: Array<Record<string, unknown>>;
  },
}));
const serverBackupSectionSpy = vi.hoisted(() => vi.fn());

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
    return <div>Server backup section</div>;
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

vi.mock('./settings/SearchSettingsSection', () => ({ SearchSettingsSection: () => null }));
vi.mock('./settings/AppearanceSettingsSection', () => ({ AppearanceSettingsSection: () => null }));
vi.mock('./settings/SecuritySettingsSection', () => ({ SecuritySettingsSection: () => null }));
vi.mock('./settings/McpSettingsSection', () => ({ McpSettingsSection: () => null }));
vi.mock('./shared/SearchFilterBar', () => ({
  SearchFilterBar: () => null,
  normalizeSearchFilterText: (value: string) => value,
  searchFilterTextMatchesQuery: () => true,
  useUrlSearchFilterState: () => ({ queries: [] }),
}));
vi.mock('./shared/AuthAdminModals', () => ({ AuthAdminModalHost: () => null }));

beforeEach(() => {
  modalRenderSpy.mockClear();
  serverBackupSectionSpy.mockClear();
  modalState.latestAllowedChatModelsProps = null;

  apiMock.getSettings.mockResolvedValue({
    settings: {
      llm_provider: 'openai',
      default_theme_pack: 'default',
      has_openai_codex_auth: true,
      has_claude_code_auth: true,
      authenticated_webgl_background_enabled: true,
      openapi_model_prefix_enabled: true,
      show_tool_card_footer_actions: false,
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
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('SettingsPanel server backup section wiring', () => {
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
      />,
    );

    await waitFor(() => {
      expect(serverBackupSectionSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          open: false,
          onEncryptedArtifactDelivered,
        }),
      );
    });
  });
});
