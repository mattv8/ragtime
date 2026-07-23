import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ToolWizard } from './ToolWizard';
import type { ToolConfig } from '@/types';

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
  normalizeHttpApiOpenApi: vi.fn(),
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
}));

vi.mock('@/api', () => ({ api: apiMock }));

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
    token_request_fields: [{ name: 'client_secret', value: '', secret: true }],
    token_request_headers: [{ name: 'X-Key', value: '' }],
  },
  configured_secret_fields: ['token_request_fields.client_secret', 'token_request_headers.X-Key'],
};

const existingHttpApiHeadersTool: ToolConfig = {
  ...existingHttpApiTool,
  id: 'tool-http-api-headers',
  name: 'Headers API',
  connection_config: {
    base_url: 'https://api.example.com',
    auth_mode: 'headers',
    request_headers: [{ name: 'X-Tenant', value: '' }],
  },
  configured_secret_fields: ['request_headers.x-tenant'],
};

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
  apiMock.testToolConnection.mockResolvedValue({
    success: true,
    message: 'Configuration is valid - no live request was sent.',
  });
  apiMock.testSavedToolConnection.mockResolvedValue({
    success: true,
    message: 'Configuration is valid - no live request was sent.',
  });
  apiMock.normalizeHttpApiOpenApi.mockResolvedValue({
    openapi_source_url: 'https://api.example.com/openapi.json',
    openapi_source_name: 'Demo API',
    openapi_source_hash: 'hash-123',
    openapi_catalog: {
      title: 'Demo API',
      version: '1.0.0',
      operations: [
        {
          operation_id: 'listOrders',
          method: 'GET',
          path: '/orders',
          summary: 'List orders',
          description: 'Returns orders',
          tags: ['orders'],
        },
      ],
    },
  });
  apiMock.createToolConfig.mockResolvedValue({
    ...existingHttpApiTool,
    id: 'tool-http-api-created',
    name: 'Demo HTTP API',
    configured_secret_fields: [],
  });
  apiMock.updateToolConfig.mockResolvedValue(existingHttpApiTool);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ToolWizard', () => {
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

  it('creates a new headers-auth HTTP API tool with normalized OpenAPI metadata and redacted review output', async () => {
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

    fireEvent.change(screen.getByLabelText('OpenAPI URL'), {
      target: { value: 'https://api.example.com/openapi.json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Normalize OpenAPI' }));

    await waitFor(() => {
      expect(apiMock.normalizeHttpApiOpenApi).toHaveBeenCalledWith({
        spec_url: 'https://api.example.com/openapi.json',
        document: undefined,
        document_name: undefined,
      });
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

    expect(screen.queryByText('tenant-secret')).toBeNull();
    expect(screen.getByText(/OpenAPI catalog: 1 operation/i)).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Create Tool' }));

    await waitFor(() => {
      expect(apiMock.createToolConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          tool_type: 'http_api',
          name: 'Demo HTTP API',
          connection_config: expect.objectContaining({
            openapi_source_url: 'https://api.example.com/openapi.json',
            openapi_source_name: 'Demo API',
            openapi_source_hash: 'hash-123',
            openapi_catalog: expect.objectContaining({
              title: 'Demo API',
              version: '1.0.0',
            }),
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
    expect(updatePayload.connection_config).not.toHaveProperty('api_key');
    expect(apiMock.testSavedToolConnection).toHaveBeenCalledWith('tool-http-api');
    expect(apiMock.testToolConnection).not.toHaveBeenCalled();
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

    fireEvent.change(screen.getByLabelText('Login path'), {
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
    fireEvent.click(screen.getByLabelText('Token request field secret 2'));
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

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    fireEvent.change(screen.getByLabelText('Login path'), {
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

    expect(screen.queryByText('login-secret')).toBeNull();
    expect(document.querySelector('.review-config')?.textContent).toContain(
      '"login_password": "[redacted]"',
    );
  });

  it('sends an explicit empty string when clearing a saved login secret on an existing HTTP API tool', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiLoginTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Clear saved Login password' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'none' },
    });
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

  it('allows token exchange progression and testing with saved scoped secrets, exact payloads, and visible non-secret review fields', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    fireEvent.change(screen.getByLabelText('Login path'), {
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
    fireEvent.click(screen.getByLabelText('Token request field secret 2'));
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
            login_path: '/oauth/token',
            login_method: 'POST',
            login_body_format: 'json',
            token_request_fields: [
              { name: 'client_secret', value: '', secret: true },
              { name: 'client_id', value: 'dummy-client-id', secret: false },
            ],
            token_request_headers: [{ name: 'X-Key', value: '' }],
            token_response_path: 'data.access_token',
            token_expires_in_path: '',
            token_header_name: 'Authorization',
            token_prefix: 'Bearer',
            request_headers: [],
          },
        }),
      );
    });
    expect(apiMock.testSavedToolConnection).toHaveBeenCalledWith('tool-http-api-token');

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(document.querySelector('.review-config')?.textContent).toContain('"name": "client_id"');
    expect(document.querySelector('.review-config')?.textContent).toContain(
      '"value": "dummy-client-id"',
    );
    expect(document.querySelector('.review-config')?.textContent).toContain(
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

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'X-TENANT' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));

    expect(document.querySelector('.review-config')?.textContent).toContain('"name": "X-TENANT"');
    expect(document.querySelector('.review-config')?.textContent).toContain(
      '"value": "Saved value not shown"',
    );
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

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'none' },
    });
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

  it('requires a replacement value when toggling a saved token secret field to non-secret', async () => {
    render(
      <ToolWizard existingTool={existingHttpApiTokenTool} onClose={vi.fn()} onSave={vi.fn()} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.click(screen.getByLabelText('Token request field secret 1'));

    expect(screen.getByRole('button', { name: 'Continue' }).hasAttribute('disabled')).toBe(true);
    expect(screen.getByText('Value is required when Secret is off.')).toBeTruthy();
  });

  it('shows a controlled normalize error when the OpenAPI normalize response is malformed', async () => {
    apiMock.normalizeHttpApiOpenApi.mockResolvedValueOnce({ openapi_catalog: null });

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
    fireEvent.click(screen.getByRole('button', { name: 'Continue' }));
    fireEvent.change(screen.getByLabelText('OpenAPI URL'), {
      target: { value: 'https://api.example.com/openapi.json' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Normalize OpenAPI' }));

    expect(await screen.findByText('Malformed OpenAPI normalize response.')).toBeTruthy();
  });
});
