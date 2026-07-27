import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { McpSettingsSection } from './McpSettingsSection';
import type { AppSettings, UpdateSettingsRequest } from '@/types';
import type { LdapGroup } from '../LdapGroupSelect';

afterEach(() => {
  cleanup();
});

const ldapGroups: LdapGroup[] = [
  {
    dn: 'CN=MCP Users,OU=Groups,DC=example,DC=com',
    name: 'MCP Users',
    display_name: 'MCP Users',
  },
];

function buildSettings(overrides: Partial<AppSettings> = {}): AppSettings {
  return {
    id: 'settings-1',
    server_name: 'Ragtime',
    default_theme_pack: 'default',
    authenticated_webgl_background_enabled: true,
    openapi_model_prefix_enabled: true,
    show_tool_card_footer_actions: false,
    embedding_provider: 'openai',
    embedding_model: 'text-embedding-3-small',
    ollama_protocol: 'http',
    ollama_host: '',
    ollama_port: 11434,
    ollama_base_url: '',
    llama_cpp_protocol: 'http',
    llama_cpp_host: '',
    llama_cpp_port: 8080,
    llama_cpp_base_url: '',
    lmstudio_protocol: 'http',
    lmstudio_host: '',
    lmstudio_port: 1234,
    lmstudio_base_url: '',
    lmstudio_api_key: '',
    omlx_protocol: 'http',
    omlx_host: '',
    omlx_port: 11435,
    omlx_base_url: '',
    omlx_api_key: '',
    llm_provider: 'openai',
    llm_model: 'gpt-5',
    llm_ollama_protocol: 'http',
    llm_ollama_host: '',
    llm_ollama_port: 11434,
    llm_ollama_base_url: '',
    llm_llama_cpp_protocol: 'http',
    llm_llama_cpp_host: '',
    llm_llama_cpp_port: 8080,
    llm_llama_cpp_base_url: '',
    llm_lmstudio_protocol: 'http',
    llm_lmstudio_host: '',
    llm_lmstudio_port: 1234,
    llm_lmstudio_base_url: '',
    llm_omlx_protocol: 'http',
    llm_omlx_host: '',
    llm_omlx_port: 11435,
    llm_omlx_base_url: '',
    openai_api_key: '',
    anthropic_api_key: '',
    openrouter_api_key: '',
    github_models_api_token: '',
    github_copilot_access_token: '',
    github_copilot_refresh_token: '',
    github_copilot_base_url: '',
    include_copilot_third_party_models: false,
    has_github_copilot_auth: false,
    allowed_chat_models: [],
    allowed_openapi_models: [],
    openapi_sync_chat_models: true,
    available_models_cache_enabled: true,
    max_iterations: 10,
    chat_compaction_threshold_percent: 80,
    chat_auto_compaction_threshold_percent: 50,
    max_tool_output_chars: 10000,
    scratchpad_window_size: 20,
    search_results_k: 5,
    aggregate_search: true,
    search_use_mmr: false,
    search_mmr_lambda: 0.5,
    context_token_budget: 10000,
    chunking_use_tokens: true,
    ivfflat_lists: 100,
    default_ocr_mode: 'disabled',
    ocr_concurrency_limit: 1,
    ollama_embedding_timeout_seconds: 60,
    sequential_index_loading: true,
    chunking_max_workers: 1,
    chunking_max_batch_size: 1,
    tool_output_mode: 'default',
    mcp_enabled: true,
    mcp_default_route_auth: true,
    mcp_default_route_auth_method: 'oauth2',
    mcp_default_route_allowed_group: null,
    mcp_default_route_client_id: '',
    has_mcp_default_password: false,
    enabled_tools: [],
    odoo_container: '',
    postgres_container: '',
    postgres_host: '',
    postgres_port: 5432,
    postgres_user: '',
    postgres_password: '',
    postgres_database: '',
    max_query_results: 100,
    query_timeout: 30,
    http_proxy_safe_timeout_seconds: 30,
    enable_write_ops: false,
    snapshot_retention_days: 30,
    snapshot_stale_branch_threshold: 30,
    userspace_preview_sandbox_flags: [],
    userspace_duplicate_copy_files_default: false,
    userspace_duplicate_copy_metadata_default: false,
    userspace_duplicate_copy_chats_default: false,
    userspace_duplicate_copy_mounts_default: false,
    userspace_mount_sync_interval_seconds: 60,
    userspace_mount_sync_start_minute: null,
    userspace_mount_sync_timezone: null,
    userspace_sqlite_import_max_bytes: 1048576,
    userspace_primitive_upload_max_bytes: 1048576,
    userspace_primitive_archive_max_entries: 100,
    userspace_code_index_enabled: true,
    userspace_code_index_debounce_seconds: 5,
    userspace_code_index_reconcile_interval_seconds: 60,
    userspace_code_index_max_attempts: 3,
    userspace_code_index_max_concurrency: 1,
    archive_max_total_size_bytes: 1048576,
    archive_max_file_count: 100,
    updated_at: null,
    ...overrides,
  };
}

function renderSection({
  settings = buildSettings(),
  formData = {},
  ldapConfigured = false,
}: {
  settings?: AppSettings;
  formData?: UpdateSettingsRequest;
  ldapConfigured?: boolean;
}) {
  function Wrapper(): JSX.Element {
    const [currentFormData, setCurrentFormData] = useState<UpdateSettingsRequest>(formData);
    const [showMcpPassword, setShowMcpPassword] = useState(false);
    const [mcpError, setMcpError] = useState<string | null>(null);

    return (
      <McpSettingsSection
        open
        onToggle={() => {}}
        formData={currentFormData}
        settings={settings}
        setFormData={setCurrentFormData}
        ldapConfigured={ldapConfigured}
        ldapDiscoveredGroups={ldapGroups}
        showMcpPassword={showMcpPassword}
        setShowMcpPassword={setShowMcpPassword}
        mcpError={mcpError}
        setMcpError={setMcpError}
        mcpSaving={false}
        handleSaveMcp={() => {}}
        setShowMcpRoutesPanel={() => {}}
        toast={{ success: vi.fn() }}
        generateMcpClientId={() => 'cid-generated'}
        generateMcpSecret={() => 'generated-secret'}
      />
    );
  }

  return render(<Wrapper />);
}

describe('McpSettingsSection', () => {
  it('shows OAuth2 as the default selectable auth method without LDAP and explains local plus LDAP sign-in', () => {
    renderSection({
      settings: buildSettings({
        mcp_default_route_auth_method: undefined as never,
      }),
    });

    expect((screen.getByRole('radio', { name: 'OAuth2' }) as HTMLInputElement).checked).toBe(true);
    expect(screen.getByRole('radio', { name: 'Password' })).toBeTruthy();
    expect(screen.getByRole('radio', { name: 'Client Credentials' })).toBeTruthy();
    expect(screen.getAllByRole('radio').map((radio) => radio.getAttribute('value'))).toEqual([
      'oauth2',
      'password',
      'client_credentials',
    ]);
    expect(screen.getByText(/Local and LDAP users can sign in through OAuth2/i)).toBeTruthy();
  });

  it('shows optional OAuth2 password fallback controls with existing password actions', () => {
    renderSection({
      settings: buildSettings({
        has_mcp_default_password: true,
      }),
    });

    expect(screen.getByLabelText('MCP Password Fallback (Optional)')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Copy password' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Show password' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Generate Password' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Clear' })).toBeTruthy();
    expect(screen.getByText(/enter a new one to change it/i)).toBeTruthy();
  });

  it('shows the exact LDAP bypass warning when an OAuth2 group restriction and fallback password are both configured', async () => {
    const user = userEvent.setup();

    renderSection({
      settings: buildSettings({
        has_mcp_default_password: true,
      }),
      ldapConfigured: true,
    });

    await user.selectOptions(
      screen.getByLabelText('Allowed LDAP Group (Optional)'),
      'CN=MCP Users,OU=Groups,DC=example,DC=com',
    );

    expect(
      screen.getByText('MCP-Password bypasses this group restriction.').closest('.field-warning'),
    ).toBeTruthy();
  });
});
