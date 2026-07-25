import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ToolWizard } from './ToolWizard';
import type { ToolConfig } from '@/types';
import { api as clientApi } from '@/api/client';

const apiMock = vi.hoisted(() => ({
  discoverDocker: vi.fn(),
  connectToNetwork: vi.fn(),
  discoverPostgresDatabases: vi.fn(),
  discoverMssqlDatabases: vi.fn(),
  discoverMysqlDatabases: vi.fn(),
  discoverInfluxdbBuckets: vi.fn(),
  generateSSHKeypair: vi.fn(),
  updateToolConfig: vi.fn(),
  testToolConnection: vi.fn(),
  testSavedToolConnection: vi.fn(),
  createToolConfig: vi.fn(),
  startFilesystemAnalysis: vi.fn(),
  getFilesystemAnalysisJob: vi.fn(),
  triggerSchemaIndex: vi.fn(),
  triggerPdmIndex: vi.fn(),
  discoverMounts: vi.fn(),
  browseFilesystem: vi.fn(),
  browseSSHFilesystem: vi.fn(),
  discoverNfsExports: vi.fn(),
  browseNfsExport: vi.fn(),
  discoverSmbShares: vi.fn(),
  browseSmbShare: vi.fn(),
  getHttpApiEditConfig: vi.fn(),
  getToolAccessPolicy: vi.fn(),
  updateToolAccessPolicy: vi.fn(),
  listUsers: vi.fn(),
  listUsersDirectory: vi.fn(),
  listAuthGroups: vi.fn(),
  getSettings: vi.fn(),
  checkContainerCapabilities: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

const getTokenEndpointInput = () =>
  screen.queryByLabelText('Token endpoint') ?? screen.getByLabelText('Login path');

function createDeferredPromise<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

async function waitForHttpApiEditHydration(toolId: string): Promise<void> {
  await waitFor(() => {
    expect(apiMock.getHttpApiEditConfig).toHaveBeenCalledWith(toolId);
  });
  await waitFor(() => {
    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(false);
  });
}

const postgresContainerTool: ToolConfig = {
  id: 'tool-postgres-container',
  name: 'Container Postgres',
  tool_type: 'postgres',
  enabled: true,
  description: 'PostgreSQL in Docker',
  connection_config: {
    container: 'postgres-1',
    docker_network: '',
    docker_ssh_enabled: false,
    database: '',
    host: '',
    port: 5432,
    user: '',
    password: '',
  },
  max_results: 100,
  timeout_max_seconds: 300,
  allow_write: false,
  sort_order: 100,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: [],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const existingHttpApiTool: ToolConfig = {
  id: 'tool-http-api',
  name: 'Orders API',
  tool_type: 'http_api',
  enabled: true,
  description: 'HTTP API tool',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'api_key',
    api_key_location: 'header',
    api_key_name: 'X-API-Key',
    openapi_source_url: 'https://api.example.com/openapi.json',
    openapi_source_name: 'Existing API catalog',
    openapi_source_hash: 'existing-hash',
    openapi_catalog: {
      title: 'Existing API',
      version: '1.0.0',
      operations: [],
    },
  },
  max_results: 25,
  timeout_max_seconds: 300,
  allow_write: false,
  sort_order: 100,
  group_id: null,
  group_name: null,
  undecryptable_fields: [],
  configured_secret_fields: ['api_key'],
  last_test_at: null,
  last_test_result: null,
  last_test_error: null,
  created_at: '2026-01-01T00:00:00Z',
  updated_at: '2026-01-01T00:00:00Z',
};

const existingHttpApiLoginTool: ToolConfig = {
  ...existingHttpApiTool,
  id: 'tool-http-api-login',
  name: 'Login API',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'login_exchange',
    login_path: '/session',
    login_username: 'demo',
    login_username_field: 'username',
  },
  configured_secret_fields: ['login_password'],
};

const existingHttpApiTokenTool: ToolConfig = {
  ...existingHttpApiTool,
  id: 'tool-http-api-token',
  name: 'Token API',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'token_exchange',
    login_path: '/session',
    token_request_fields: [
      { name: 'client_secret', value: 'existing-client-secret', secret: true },
    ],
    token_request_headers: [{ name: 'X-Key', value: 'existing-token-header-secret' }],
    request_body_format: 'form',
    request_body_fields: [
      { name: 'grant_type', value: 'client_credentials', secret: true },
      { name: 'client_secret', value: 'existing-body-secret', secret: true },
    ],
  },
  configured_secret_fields: [
    'token_request_fields.client_secret',
    'token_request_headers.X-Key',
    'request_body_fields.client_secret',
  ],
};

const existingHttpApiHeadersTool: ToolConfig = {
  ...existingHttpApiTool,
  id: 'tool-http-api-headers',
  name: 'Headers API',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'headers',
    request_headers: [{ name: 'X-Tenant', value: 'existing-tenant-secret' }],
  },
  configured_secret_fields: ['request_headers.x-tenant'],
};

const existingHttpApiOAuthTool: ToolConfig = {
  ...existingHttpApiTool,
  id: 'tool-http-api-oauth',
  name: 'OAuth API',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'oauth2',
    oauth_flow: 'authorization_code_pkce',
    oauth_issuer_url: 'https://issuer.example.test',
    oauth_authorization_url: 'https://issuer.example.test/authorize',
    oauth_token_url: 'https://issuer.example.test/token',
    oauth_client_id: 'client-id',
    oauth_client_auth_method: 'none',
    oauth_scopes: ['openid'],
  },
  configured_secret_fields: ['oauth_access_token'],
};

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  apiMock.getHttpApiEditConfig.mockImplementation(async (toolId: string) => {
    switch (toolId) {
      case 'tool-http-api':
        return {
          connection_config: {
            ...existingHttpApiTool.connection_config,
            api_key: 'decrypted-api-key',
          },
        };
      case 'tool-http-api-login':
        return {
          connection_config: {
            ...existingHttpApiLoginTool.connection_config,
            login_password: 'decrypted-login-password',
          },
        };
      case 'tool-http-api-token':
        return {
          connection_config: {
            ...existingHttpApiTokenTool.connection_config,
          },
        };
      case 'tool-http-api-headers':
        return {
          connection_config: {
            ...existingHttpApiHeadersTool.connection_config,
          },
        };
      case 'tool-http-api-oauth':
        return {
          connection_config: {
            ...existingHttpApiOAuthTool.connection_config,
          },
        };
      default:
        throw new Error(`Unexpected tool id: ${toolId}`);
    }
  });
  apiMock.testToolConnection.mockResolvedValue({
    success: true,
    message: 'Configuration is valid - no live request was sent.',
  });
  apiMock.testSavedToolConnection.mockResolvedValue({
    success: true,
    message: 'Configuration is valid - no live request was sent.',
  });
  apiMock.createToolConfig.mockResolvedValue({
    ...existingHttpApiTool,
    id: 'tool-http-api-created',
    name: 'Demo HTTP API',
    configured_secret_fields: [],
  });
  apiMock.updateToolConfig.mockResolvedValue(existingHttpApiTool);
  apiMock.getToolAccessPolicy.mockImplementation(async (toolId: string) => ({
    tool_id: toolId,
    default_chat_access: 'deny',
    default_workspace_access: 'deny',
    users: [],
    groups: [],
  }));
  apiMock.updateToolAccessPolicy.mockImplementation(async (_toolId: string, policy: unknown) => ({
    tool_id: 'tool-http-api-created',
    ...(policy as Record<string, unknown>),
  }));
  apiMock.listUsers.mockResolvedValue([]);
  apiMock.listAuthGroups.mockResolvedValue([]);
  apiMock.getSettings.mockResolvedValue({ settings: {} });
  apiMock.checkContainerCapabilities.mockResolvedValue({
    privileged: false,
    has_sys_admin: false,
    can_mount: false,
    message: 'ok',
  });
  apiMock.discoverMounts.mockResolvedValue({
    mounts: [{ container_path: '/workspace', host_path: '/host/workspace' }],
    docker_compose_example: '',
  });
  apiMock.browseFilesystem.mockResolvedValue({ entries: [] });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe('ToolWizard', () => {
  it('adds an Access step before review for every wizard variant', async () => {
    const cases: Array<{
      label: string;
      existingTool: ToolConfig | null;
      defaultToolType?: ToolConfig['tool_type'];
      mountOnly?: boolean;
    }> = [
      { label: 'base', existingTool: null, defaultToolType: 'postgres' },
      { label: 'http api', existingTool: null, defaultToolType: 'http_api' },
      { label: 'pdm', existingTool: null, defaultToolType: 'solidworks_pdm' },
      { label: 'ssh', existingTool: null, defaultToolType: 'ssh_shell' },
      { label: 'odoo', existingTool: null, defaultToolType: 'odoo_shell' },
      { label: 'filesystem', existingTool: null, defaultToolType: 'filesystem_indexer' },
      {
        label: 'mount only',
        existingTool: null,
        defaultToolType: 'filesystem_indexer',
        mountOnly: true,
      },
      { label: 'edit', existingTool: existingHttpApiTool },
    ];

    for (const testCase of cases) {
      const { unmount } = render(
        <ToolWizard
          existingTool={testCase.existingTool}
          onClose={vi.fn()}
          onSave={vi.fn()}
          defaultToolType={testCase.defaultToolType}
          mountOnly={testCase.mountOnly}
        />,
      );

      if (testCase.existingTool?.tool_type === 'http_api') {
        await waitForHttpApiEditHydration(testCase.existingTool.id);
      }

      const progressTitles = Array.from(document.querySelectorAll('.wizard-step .step-title')).map(
        (node) => node.textContent?.trim(),
      );
      const accessIndex = progressTitles.indexOf('User Access');
      const reviewIndex = progressTitles.indexOf('Review & Save');
      expect(accessIndex, `${testCase.label} access step`).toBeGreaterThanOrEqual(0);
      expect(reviewIndex, `${testCase.label} review step`).toBeGreaterThan(accessIndex);

      unmount();
    }
  });

  it('saves tool access after creating a tool and keeps the wizard open if ACL save fails', async () => {
    const onSave = vi.fn();
    apiMock.updateToolAccessPolicy.mockRejectedValueOnce(new Error('ACL failed closed'));

    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={onSave}
        defaultToolType="ssh_shell"
      />,
    );

    fireEvent.change(screen.getByPlaceholderText('server.example.com'), {
      target: { value: 'server.example.com' },
    });
    fireEvent.change(screen.getByPlaceholderText('ubuntu'), { target: { value: 'deploy' } });
    fireEvent.change(screen.getByPlaceholderText('Enter SSH password'), {
      target: { value: 'secret' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByPlaceholderText('e.g., Production Database, Staging Odoo'), {
      target: { value: 'Deploy Shell' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Create Tool' }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(apiMock.updateToolAccessPolicy).toHaveBeenCalledWith(
        'tool-http-api-created',
        expect.objectContaining({
          default_chat_access: 'deny',
          default_workspace_access: 'deny',
          users: [],
          groups: [],
        }),
      );
    });
    expect(onSave).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: 'Create Tool' })).toBeTruthy();
    expect(screen.getByText('ACL failed closed')).toBeTruthy();
  });

  it('applies access policy to filesystem tools auto-created during analysis when saving later', async () => {
    apiMock.createToolConfig.mockResolvedValueOnce({
      ...postgresContainerTool,
      id: 'tool-filesystem-created',
      name: 'Workspace Files',
      tool_type: 'filesystem_indexer',
      connection_config: { mount_type: 'docker_volume', base_path: '/workspace' },
    });
    apiMock.startFilesystemAnalysis.mockResolvedValue({
      id: 'job-1',
      status: 'completed',
      result: {
        suggested_exclusions: [],
        total_files: 0,
        total_size_mb: 0,
        estimated_chunks: 0,
        analysis_duration_seconds: 0,
        directories_scanned: 0,
        file_type_stats: [],
        warnings: [],
      },
    });
    apiMock.getFilesystemAnalysisJob.mockResolvedValue({
      id: 'job-1',
      status: 'completed',
      result: {
        suggested_exclusions: [],
        total_files: 0,
        total_size_mb: 0,
        estimated_chunks: 0,
        analysis_duration_seconds: 0,
        directories_scanned: 0,
        file_type_stats: [],
        warnings: [],
      },
    });

    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultToolType="filesystem_indexer"
      />,
    );

    await screen.findByText('/workspace');
    fireEvent.click(screen.getByText('/workspace'));
    await waitFor(() => {
      expect(apiMock.browseFilesystem).toHaveBeenCalledWith('/workspace');
    });
    fireEvent.click(screen.getByRole('button', { name: 'Select' }));
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Filesystem' }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith(
        expect.objectContaining({ tool_type: 'filesystem_indexer' }),
      );
    });

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Create Tool' }));

    await waitFor(() => {
      expect(apiMock.updateToolAccessPolicy).toHaveBeenCalledWith(
        'tool-filesystem-created',
        expect.objectContaining({
          default_chat_access: 'deny',
          default_workspace_access: 'deny',
        }),
      );
    });
  });
  it('groups SSH host, port, and user in one compact row with a flat auth panel', () => {
    const { container } = render(
      <ToolWizard existingTool={postgresContainerTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    fireEvent.click(screen.getByLabelText('Remote Docker host via SSH'));

    const remoteSshRow = container.querySelector('.remote-docker-ssh-row');
    expect(remoteSshRow).not.toBeNull();
    expect(remoteSshRow?.querySelectorAll('input')).toHaveLength(3);
    const rowLabels = Array.from(remoteSshRow?.querySelectorAll('label') ?? []).map((label) =>
      label.textContent?.trim(),
    );
    expect(rowLabels).toEqual(['SSH Host', 'SSH Port', 'SSH User']);

    expect(container.querySelector('.ssh-auth-panel.compact')).not.toBeNull();
    expect(container.querySelector('.ssh-key-panel.flat')).not.toBeNull();
  });

  it('creates a new headers-auth HTTP API tool with documentation metadata and redacted review output', async () => {
    const onSave = vi.fn();

    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={onSave}
        defaultToolType="http_api"
      />,
    );

    fireEvent.change(screen.getByLabelText('Base URL'), {
      target: { value: 'https://api.example.com' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'headers' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'X-Tenant' },
    });
    fireEvent.change(screen.getByLabelText('Configured header value 1'), {
      target: { value: 'tenant-secret' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

    await waitFor(() => {
      expect(apiMock.testToolConnection).toHaveBeenCalledWith({
        tool_type: 'http_api',
        connection_config: expect.objectContaining({
          base_url: 'https://api.example.com',
          auth_mode: 'headers',
          request_headers: [{ name: 'X-Tenant', value: 'tenant-secret' }],
        }),
      });
    });
    expect(screen.getByText('Configuration is valid - no live request was sent.')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    fireEvent.change(screen.getByLabelText(/API documentation URL/), {
      target: { value: 'https://api.example.com/docs' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByPlaceholderText('e.g., Production Database, Staging Odoo'), {
      target: { value: 'Demo HTTP API' },
    });
    fireEvent.change(
      screen.getByPlaceholderText(/Describe what data is available through this connection/i),
      {
        target: { value: 'Order management API.' },
      },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(screen.getAllByRole('spinbutton')[0]).toBeTruthy();
    expect(screen.getByText(/write methods still require this tool option/i)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(screen.queryByText('tenant-secret')).toBeNull();
    const review = document.querySelector('.review-config')?.textContent ?? '';
    expect(review).toContain('"documentation_url": "https://api.example.com/docs"');
    expect(review).not.toContain('"documentation_url": "[redacted]"');

    fireEvent.click(screen.getByRole('button', { name: 'Create Tool' }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          tool_type: 'http_api',
          name: 'Demo HTTP API',
          connection_config: expect.objectContaining({
            documentation_url: 'https://api.example.com/docs',
          }),
        }),
      );
    });
    const createPayload = apiMock.createToolConfig.mock.calls[0][0];
    expect(createPayload.connection_config).not.toHaveProperty('document');
    expect(createPayload.connection_config).not.toHaveProperty('document_name');
    expect(onSave).toHaveBeenCalledTimes(1);
  });

  it('tests an existing HTTP API tool through the saved-tool endpoint after updating without re-sending omitted secrets', async () => {
    render(<ToolWizard existingTool={existingHttpApiTool} onClose={vi.fn()} onSave={vi.fn()} />);
    await waitForHttpApiEditHydration('tool-http-api');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('API key name'), {
      target: { value: 'X-Auth-Key' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(
        'tool-http-api',
        expect.objectContaining({
          connection_config: expect.objectContaining({
            api_key_name: 'X-Auth-Key',
          }),
        }),
      );
    });

    const updatePayload = apiMock.updateToolConfig.mock.calls[0][1];
    expect(updatePayload.connection_config.api_key).toBe('decrypted-api-key');
    expect(updatePayload.connection_config).toEqual(
      expect.objectContaining({
        openapi_source_url: 'https://api.example.com/openapi.json',
        openapi_source_name: 'Existing API catalog',
        openapi_source_hash: 'existing-hash',
        openapi_catalog: expect.objectContaining({ title: 'Existing API' }),
      }),
    );
    expect(apiMock.testSavedToolConnection).toHaveBeenCalledWith('tool-http-api');
    expect(apiMock.testToolConnection).not.toHaveBeenCalled();
  });

  it('hydrates an existing HTTP API tool before allowing edit navigation', async () => {
    render(<ToolWizard existingTool={existingHttpApiTool} onClose={vi.fn()} onSave={vi.fn()} />);
    await waitForHttpApiEditHydration('tool-http-api');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    await waitFor(() => {
      expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe(
        'decrypted-api-key',
      );
    });
  });

  it('does not request HTTP API edit config for create mode or non-HTTP tools', () => {
    const { rerender } = render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultToolType="http_api"
      />,
    );

    rerender(
      <ToolWizard existingTool={postgresContainerTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    expect(apiMock.getHttpApiEditConfig).not.toHaveBeenCalled();
  });

  it('shows a generic load error and blocks progression when HTTP API edit hydration fails', async () => {
    apiMock.getHttpApiEditConfig.mockRejectedValueOnce(new Error('decryption exploded'));

    render(<ToolWizard existingTool={existingHttpApiTool} onClose={vi.fn()} onSave={vi.fn()} />);

    await waitFor(() => {
      expect(screen.getByText('Failed to load saved HTTP API credentials.')).toBeTruthy();
    });

    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(true);
    expect(screen.getByRole('group').hasAttribute('disabled')).toBe(true);
  });

  it('ignores stale HTTP API edit hydration responses after switching tools', async () => {
    const firstHydration = createDeferredPromise<{
      connection_config: typeof existingHttpApiTool.connection_config & { api_key: string };
    }>();
    const secondHydration = createDeferredPromise<{
      connection_config: typeof existingHttpApiTool.connection_config & { api_key: string };
    }>();
    const replacementTool: ToolConfig = {
      ...existingHttpApiTool,
      id: 'tool-http-api-replacement',
      name: 'Replacement API',
      connection_config: {
        ...existingHttpApiTool.connection_config,
        base_url: 'https://replacement.example.com',
      },
    };

    apiMock.getHttpApiEditConfig.mockImplementation((toolId: string) => {
      if (toolId === 'tool-http-api') {
        return firstHydration.promise;
      }
      if (toolId === 'tool-http-api-replacement') {
        return secondHydration.promise;
      }
      return Promise.reject(new Error(`Unexpected tool id: ${toolId}`));
    });

    const { rerender } = render(
      <ToolWizard existingTool={existingHttpApiTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    rerender(<ToolWizard existingTool={replacementTool} onClose={vi.fn()} onSave={vi.fn()} />);

    secondHydration.resolve({
      connection_config: {
        ...replacementTool.connection_config,
        api_key: 'replacement-api-key',
      },
    });
    await waitForHttpApiEditHydration('tool-http-api-replacement');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    await waitFor(() => {
      expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe(
        'replacement-api-key',
      );
    });

    firstHydration.resolve({
      connection_config: {
        ...existingHttpApiTool.connection_config,
        api_key: 'stale-api-key',
      },
    });

    await waitFor(() => {
      expect((screen.getByLabelText('API key') as HTMLInputElement).value).toBe(
        'replacement-api-key',
      );
    });
  });

  it('keeps Continue and Test disabled for duplicate modern auth rows until corrected', async () => {
    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultToolType="http_api"
      />,
    );

    fireEvent.change(screen.getByLabelText('Base URL'), {
      target: { value: 'https://api.example.com' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'token_exchange' },
    });

    fireEvent.change(getTokenEndpointInput(), {
      target: { value: '/oauth/token' },
    });
    fireEvent.change(screen.getByLabelText('Token response path'), {
      target: { value: 'access_token' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add token request field' }));
    fireEvent.change(screen.getByLabelText('Token request field name 1'), {
      target: { value: 'client_secret' },
    });
    fireEvent.change(screen.getByLabelText('Token request field value 1'), {
      target: { value: 'secret-a' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add token request field' }));
    fireEvent.change(screen.getByLabelText('Token request field name 2'), {
      target: { value: 'client_secret' },
    });
    fireEvent.change(screen.getByLabelText('Token request field value 2'), {
      target: { value: 'client-b' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add token request header' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add token request header' }));
    fireEvent.change(screen.getByLabelText('Token request header name 1'), {
      target: { value: 'x-key' },
    });
    fireEvent.change(screen.getByLabelText('Token request header value 1'), {
      target: { value: 'header-a' },
    });
    fireEvent.change(screen.getByLabelText('Token request header name 2'), {
      target: { value: 'X-Key' },
    });
    fireEvent.change(screen.getByLabelText('Token request header value 2'), {
      target: { value: 'header-b' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'x-tenant' },
    });
    fireEvent.change(screen.getByLabelText('Configured header value 1'), {
      target: { value: 'tenant-a' },
    });
    fireEvent.change(screen.getByLabelText('Configured header name 2'), {
      target: { value: 'X-Tenant' },
    });
    fireEvent.change(screen.getByLabelText('Configured header value 2'), {
      target: { value: 'tenant-b' },
    });

    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(true);
    expect(screen.getByRole('button', { name: 'Test connection' }).hasAttribute('disabled')).toBe(
      true,
    );

    fireEvent.change(screen.getByLabelText('Token request field name 2'), {
      target: { value: 'client_id' },
    });
    fireEvent.change(screen.getByLabelText('Token request header name 2'), {
      target: { value: 'X-Extra-Key' },
    });
    fireEvent.change(screen.getByLabelText('Configured header name 2'), {
      target: { value: 'X-Region' },
    });

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(false);
    });
    expect(screen.getByRole('button', { name: 'Test connection' }).hasAttribute('disabled')).toBe(
      false,
    );
  });

  it('allows login exchange progression and testing with visible backend defaults and redacts a newly typed secret in review', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiLoginTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-login');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    fireEvent.change(getTokenEndpointInput(), {
      target: { value: '/session' },
    });
    fireEvent.change(screen.getByLabelText('Login username'), {
      target: { value: 'alice' },
    });
    fireEvent.change(screen.getByLabelText('Login password'), {
      target: { value: 'login-secret' },
    });

    const testButton = screen.getByRole('button', { name: 'Test connection' });
    expect(testButton.hasAttribute('disabled')).toBe(false);
    fireEvent.click(testButton);

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(
        'tool-http-api-login',
        expect.objectContaining({
          connection_config: expect.objectContaining({
            auth_mode: 'login_exchange',
            login_path: '/session',
            login_username: 'alice',
            login_password: 'login-secret',
          }),
        }),
      );
    });
    expect(apiMock.testSavedToolConnection).toHaveBeenCalledWith('tool-http-api-login');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(screen.queryByText('login-secret')).toBeNull();
    expect(document.querySelector('.review-config')?.textContent).toContain(
      '"login_password": "[redacted]"',
    );
  });

  it('sends an explicit empty string when clearing a saved login secret on an existing HTTP API tool', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiLoginTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-login');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Login password'), { target: { value: '' } });
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'none' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(
        'tool-http-api-login',
        expect.objectContaining({
          connection_config: expect.objectContaining({
            login_password: '',
          }),
        }),
      );
    });
  });

  it('allows token exchange progression and testing with saved scoped secrets, exact payloads, and redacted review fields', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    fireEvent.change(getTokenEndpointInput(), {
      target: { value: '/oauth/token' },
    });
    fireEvent.change(screen.getByLabelText('Token response path'), {
      target: { value: 'data.access_token' },
    });
    fireEvent.change(screen.getByLabelText('Token request field name 1'), {
      target: { value: 'client_secret' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add token request field' }));
    fireEvent.change(screen.getByLabelText('Token request field name 2'), {
      target: { value: 'client_id' },
    });
    fireEvent.change(screen.getByLabelText('Token request field value 2'), {
      target: { value: 'dummy-client-id' },
    });
    fireEvent.change(screen.getByLabelText('Token request header name 1'), {
      target: { value: 'X-Key' },
    });

    const testButton = screen.getByRole('button', { name: 'Test connection' });
    expect(testButton.hasAttribute('disabled')).toBe(false);
    fireEvent.click(testButton);

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(
        'tool-http-api-token',
        expect.objectContaining({
          connection_config: {
            base_url: 'https://api.example.com',
            auth_mode: 'token_exchange',
            login_path: '',
            token_url: '/oauth/token',
            login_method: 'POST',
            login_body_format: 'json',
            token_request_fields: [
              { name: 'client_secret', value: 'existing-client-secret', secret: true },
              { name: 'client_id', value: 'dummy-client-id', secret: true },
            ],
            token_request_headers: [{ name: 'X-Key', value: 'existing-token-header-secret' }],
            token_response_path: 'data.access_token',
            token_expires_in_path: '',
            token_header_name: 'Authorization',
            token_prefix: 'Bearer',
            request_headers: [],
            request_body_format: 'form',
            request_body_fields: [
              { name: 'grant_type', value: 'client_credentials', secret: true },
              { name: 'client_secret', value: 'existing-body-secret', secret: true },
            ],
          },
        }),
      );
    });
    expect(apiMock.updateToolConfig.mock.calls[0][1].connection_config.token_url).toBe(
      '/oauth/token',
    );
    expect(apiMock.testSavedToolConnection).toHaveBeenCalledWith('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(document.querySelector('.review-config')?.textContent).toContain('"name": "client_id"');
    expect(document.querySelector('.review-config')?.textContent).toContain(
      '"value": "[redacted]"',
    );
    expect(document.querySelector('.review-config')?.textContent).not.toContain('dummy-client-id');
    expect(document.querySelector('.review-config')?.textContent).not.toContain(
      'Saved value not shown',
    );
    expect(document.querySelector('.review-config')?.textContent).not.toContain(
      'dummy-client-secret',
    );
  });

  it('shows saved configured header markers in review after a case-only header rename', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiHeadersTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-headers');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'X-TENANT' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(document.querySelector('.review-config')?.textContent).toContain('"name": "X-TENANT"');
    expect(document.querySelector('.review-config')?.textContent).toContain(
      '"value": "[redacted]"',
    );
    expect(document.querySelector('.review-config')?.textContent).not.toContain(
      'Saved value not shown',
    );
    expect(document.querySelector('.review-config')?.textContent).not.toContain(
      'existing-tenant-secret',
    );
  });

  it('allows a body-only Headers configuration without requiring a static header', async () => {
    const bodyOnlyHeadersTool: ToolConfig = {
      ...existingHttpApiTokenTool,
      connection_config: {
        ...existingHttpApiTokenTool.connection_config,
        auth_mode: 'headers',
        request_headers: [],
      },
    };
    apiMock.getHttpApiEditConfig.mockResolvedValueOnce({
      connection_config: bodyOnlyHeadersTool.connection_config,
    });

    render(<ToolWizard existingTool={bodyOnlyHeadersTool} onClose={vi.fn()} onSave={vi.fn()} />);
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(false);
    expect(screen.getByRole('button', { name: 'Test connection' }).hasAttribute('disabled')).toBe(
      false,
    );
  });

  it('preserves resource body fields across Headers and token exchange, then clears them for Basic', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'headers' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

    await waitFor(() => expect(apiMock.updateToolConfig).toHaveBeenCalledTimes(1));
    expect(apiMock.updateToolConfig.mock.calls[0][1].connection_config).toEqual(
      expect.objectContaining({
        auth_mode: 'headers',
        request_body_format: 'form',
        request_body_fields: expect.any(Array),
      }),
    );

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'token_exchange' },
    });
    fireEvent.change(getTokenEndpointInput(), { target: { value: '/session' } });
    fireEvent.change(screen.getByLabelText('Token response path'), {
      target: { value: 'access_token' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add token request field' }));
    fireEvent.change(screen.getByLabelText('Token request field name 1'), {
      target: { value: 'client_id' },
    });
    fireEvent.change(screen.getByLabelText('Token request field value 1'), {
      target: { value: 'client-id' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));

    await waitFor(() => expect(apiMock.updateToolConfig).toHaveBeenCalledTimes(2));
    expect(apiMock.updateToolConfig.mock.calls[1][1].connection_config).toEqual(
      expect.objectContaining({
        auth_mode: 'token_exchange',
        request_body_format: 'form',
        request_body_fields: expect.any(Array),
      }),
    );

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'basic' },
    });
    fireEvent.change(screen.getByLabelText('Basic username'), { target: { value: 'user' } });
    fireEvent.change(screen.getByLabelText('Basic password'), { target: { value: 'password' } });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(apiMock.updateToolConfig).toHaveBeenCalledTimes(3));
    const basicConfig = apiMock.updateToolConfig.mock.calls[2][1].connection_config;
    expect(basicConfig.request_body_fields).toEqual([]);
    expect(basicConfig).not.toHaveProperty('request_body_format');
  });

  it('clears resource body fields and format when switching from token exchange to None', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), { target: { value: 'none' } });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(apiMock.updateToolConfig).toHaveBeenCalledTimes(1));
    const noneConfig = apiMock.updateToolConfig.mock.calls[0][1].connection_config;
    expect(noneConfig.request_body_fields).toEqual([]);
    expect(noneConfig).not.toHaveProperty('request_body_format');
  });

  it('preserves and redacts saved resource body secrets while editing an unrelated field', async () => {
    const toolWithTypedBodySecret: ToolConfig = {
      ...existingHttpApiTokenTool,
      connection_config: {
        ...existingHttpApiTokenTool.connection_config,
        request_body_fields: [
          { name: 'grant_type', value: 'client_credentials', secret: true },
          { name: 'client_secret', value: 'client-secret-value', secret: true },
        ],
      },
    };
    apiMock.getHttpApiEditConfig.mockResolvedValueOnce({
      connection_config: toolWithTypedBodySecret.connection_config,
    });
    render(
      <ToolWizard existingTool={toolWithTypedBodySecret} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Token response path'), {
      target: { value: 'data.access_token' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    const review = document.querySelector('.review-config')?.textContent ?? '';
    expect(review).toContain('"value": "[redacted]"');
    expect(review).not.toContain('Saved value not shown');
    expect(review).not.toContain('client-secret-value');
  });

  it('clears modern-only headers mode rows when switching away before save', async () => {
    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultToolType="http_api"
      />,
    );

    fireEvent.change(screen.getByLabelText('Base URL'), {
      target: { value: 'https://api.example.com' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'headers' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'X-Tenant' },
    });
    fireEvent.change(screen.getByLabelText('Configured header value 1'), {
      target: { value: 'tenant-a' },
    });
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'basic' },
    });
    fireEvent.change(screen.getByLabelText('Basic username'), {
      target: { value: 'alfred' },
    });
    fireEvent.change(screen.getByLabelText('Basic password'), {
      target: { value: 's3cret' },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByPlaceholderText('e.g., Production Database, Staging Odoo'), {
      target: { value: 'Basic HTTP API' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Create Tool' }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalled();
    });

    const createPayload = apiMock.createToolConfig.mock.calls[0][0];
    expect(createPayload.connection_config.auth_mode).toBe('basic');
    expect(createPayload.connection_config.request_headers).toEqual([]);
  });

  it('clears modern-only token exchange rows when switching away before save', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'none' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(
        'tool-http-api-token',
        expect.objectContaining({
          connection_config: expect.objectContaining({
            token_request_fields: [],
            token_request_headers: [],
            request_headers: [],
          }),
        }),
      );
    });
    const updatePayload = apiMock.updateToolConfig.mock.calls[0][1];
    expect(updatePayload.connection_config.auth_mode).toBe('none');
    expect(updatePayload.connection_config.token_request_fields).toEqual([]);
    expect(updatePayload.connection_config.token_request_headers).toEqual([]);
    expect(updatePayload.connection_config.request_headers).toEqual([]);
    expect(updatePayload.connection_config).not.toHaveProperty('login_path');
    expect(updatePayload.connection_config).not.toHaveProperty('token_response_path');
  });

  it('keeps saved token body fields secret-only and redacted in review', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(screen.queryByLabelText('Token request field secret 1')).toBeNull();
    expect((screen.getByLabelText('Token request field value 1') as HTMLInputElement).type).toBe(
      'password',
    );
    expect((screen.getByLabelText('Token request field value 1') as HTMLInputElement).value).toBe(
      'existing-client-secret',
    );
    expect((screen.getByLabelText('Token request header value 1') as HTMLInputElement).type).toBe(
      'password',
    );
    expect((screen.getByLabelText('Token request header value 1') as HTMLInputElement).value).toBe(
      'existing-token-header-secret',
    );
    expect((screen.getByLabelText('Request body field value 2') as HTMLInputElement).type).toBe(
      'password',
    );
    expect((screen.getByLabelText('Request body field value 2') as HTMLInputElement).value).toBe(
      'existing-body-secret',
    );
    expect(screen.queryByText('Saved value not shown')).toBeNull();
    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(false);

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    const review = document.querySelector('.review-config')?.textContent ?? '';
    expect(review).toContain('"value": "[redacted]"');
    expect(review).not.toContain('Saved value not shown');
    expect(review).not.toContain('client-secret');
    expect(review).not.toContain('existing-client-secret');
    expect(review).not.toContain('existing-token-header-secret');
    expect(review).not.toContain('existing-body-secret');
  });

  it('normalizes legacy token body fields to secret-only before saving', async () => {
    apiMock.getHttpApiEditConfig.mockResolvedValueOnce({
      connection_config: {
        ...existingHttpApiTokenTool.connection_config,
        token_request_fields: [{ name: 'client_id', value: 'public-id', secret: false }],
      },
    });
    render(
      <ToolWizard
        existingTool={{
          ...existingHttpApiTokenTool,
          connection_config: {
            ...existingHttpApiTokenTool.connection_config,
            token_request_fields: [{ name: 'client_id', value: 'public-id', secret: false }],
          },
        }}
        onClose={vi.fn()}
        onSave={vi.fn()}
      />,
    );
    await waitForHttpApiEditHydration('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(apiMock.updateToolConfig).toHaveBeenCalledWith(
        'tool-http-api-token',
        expect.objectContaining({
          connection_config: expect.objectContaining({
            token_request_fields: [{ name: 'client_id', value: 'public-id', secret: true }],
          }),
        }),
      );
    });
  });

  it('accepts saved OAuth credentials, omits token values, and hides generic testing for new OAuth', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiOAuthTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );
    await waitForHttpApiEditHydration('tool-http-api-oauth');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    expect(screen.getByRole('region', { name: 'OAuth 2.0 connection' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Test connection' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => expect(apiMock.updateToolConfig).toHaveBeenCalled());
    const payload = apiMock.updateToolConfig.mock.calls[0][1].connection_config;
    expect(payload).not.toHaveProperty('oauth_access_token');
    expect(payload).not.toHaveProperty('oauth_refresh_token');
  });

  it('does not allow a fully configured new OAuth tool to continue before connecting', () => {
    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultToolType="http_api"
      />,
    );

    fireEvent.change(screen.getByLabelText('Base URL'), {
      target: { value: 'https://api.example.com' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), { target: { value: 'oauth2' } });
    fireEvent.change(screen.getByLabelText('Issuer URL'), {
      target: { value: 'https://issuer.example.test' },
    });
    fireEvent.change(screen.getByLabelText('Client ID'), { target: { value: 'client-id' } });
    fireEvent.change(screen.getByLabelText('Scopes'), { target: { value: 'openid profile' } });
    fireEvent.change(screen.getByLabelText('Token endpoint'), {
      target: { value: 'https://issuer.example.test/token' },
    });
    fireEvent.change(screen.getByLabelText('Device authorization endpoint'), {
      target: { value: 'https://issuer.example.test/device' },
    });

    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(true);
  });

  it('connects a new OAuth tool, polls to connected, and saves only its opaque session ID', async () => {
    vi.spyOn(clientApi, 'startHttpApiOAuth').mockResolvedValue({
      status: 'pending',
      session_id: 'opaque-session-id',
      verification_uri: 'https://issuer.example.test/device',
      user_code: 'ABCD-EFGH',
      interval: 0,
    });
    vi.spyOn(clientApi, 'pollHttpApiOAuth').mockResolvedValue({
      status: 'connected',
      session_id: 'opaque-session-id',
    });
    vi.spyOn(window, 'open').mockReturnValue(null);

    render(
      <ToolWizard
        existingTool={null}
        onClose={vi.fn()}
        onSave={vi.fn()}
        defaultToolType="http_api"
      />,
    );
    fireEvent.change(screen.getByLabelText('Base URL'), {
      target: { value: 'https://api.example.com' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), { target: { value: 'oauth2' } });
    fireEvent.change(screen.getByLabelText('Issuer URL'), {
      target: { value: 'https://issuer.example.test' },
    });
    fireEvent.change(screen.getByLabelText('Client ID'), { target: { value: 'client-id' } });
    fireEvent.change(screen.getByLabelText('Scopes'), { target: { value: 'openid profile' } });
    fireEvent.change(screen.getByLabelText('Token endpoint'), {
      target: { value: 'https://issuer.example.test/token' },
    });
    fireEvent.change(screen.getByLabelText('Device authorization endpoint'), {
      target: { value: 'https://issuer.example.test/device' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() =>
      expect(screen.getByText('Connected. Save the tool to keep this credential.')),
    );
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByPlaceholderText('e.g., Production Database, Staging Odoo'), {
      target: { value: 'OAuth API' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Create Tool' }));

    await waitFor(() => expect(apiMock.createToolConfig).toHaveBeenCalled());
    const payload = apiMock.createToolConfig.mock.calls[0][0].connection_config;
    expect(payload).toEqual(
      expect.objectContaining({ auth_mode: 'oauth2', oauth_session_id: 'opaque-session-id' }),
    );
    expect(payload).not.toHaveProperty('oauth_access_token');
    expect(payload).not.toHaveProperty('oauth_refresh_token');
  });
});
