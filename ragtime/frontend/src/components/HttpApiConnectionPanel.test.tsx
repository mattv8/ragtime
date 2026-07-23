import { useState } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { HttpApiConnectionPanel } from './HttpApiConnectionPanel';
import type { HttpApiConnectionConfig, OpenApiCatalog } from '@/types';

const baseValue: HttpApiConnectionConfig = {
  base_url: 'https://api.example.com',
  auth_mode: 'none',
  approved_request_headers: ['X-Trace-Id'],
};

afterEach(() => {
  cleanup();
});

describe('HttpApiConnectionPanel', () => {
  it('renders only the supplied section and no internal tabs', () => {
    render(<HttpApiConnectionPanel section="connection" value={baseValue} onChange={vi.fn()} />);

    expect(screen.queryByRole('tablist')).toBeNull();
    expect(screen.getByLabelText('Base URL')).toBeTruthy();
    expect(screen.queryByLabelText('Authentication mode')).toBeNull();
    expect(screen.queryByLabelText('OpenAPI URL')).toBeNull();
  });

  it('shows only the new auth choices for new configs and hides legacy editors', () => {
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={baseValue}
        onChange={vi.fn()}
        configuredSecretFields={[]}
      />,
    );

    const options = Array.from(
      (screen.getByLabelText('Authentication mode') as HTMLSelectElement).options,
    ).map((option) => option.textContent);

    expect(options).toEqual(['None', 'Headers', 'Basic authentication', 'Token exchange']);
    expect(screen.queryByLabelText('API key location')).toBeNull();
    expect(screen.queryByLabelText('Bearer token')).toBeNull();
    expect(screen.queryByLabelText('Login username')).toBeNull();
  });

  it('keeps loaded legacy auth choices available after switching to a modern mode', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'api_key',
        api_key_name: 'X-API-Key',
      });

      return (
        <HttpApiConnectionPanel
          section="authentication"
          value={value}
          onChange={setValue}
          configuredSecretFields={[]}
        />
      );
    }

    render(<Harness />);

    let options = Array.from(
      (screen.getByLabelText('Authentication mode') as HTMLSelectElement).options,
    ).map((option) => option.textContent);
    expect(options).toEqual([
      'None',
      'Headers',
      'Basic authentication',
      'Token exchange',
      'Legacy API key',
      'Legacy Bearer token',
      'Legacy login exchange',
    ]);

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'headers' },
    });

    options = Array.from(
      (screen.getByLabelText('Authentication mode') as HTMLSelectElement).options,
    ).map((option) => option.textContent);
    expect(options).toEqual([
      'None',
      'Headers',
      'Basic authentication',
      'Token exchange',
      'Legacy API key',
      'Legacy Bearer token',
      'Legacy login exchange',
    ]);

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'api_key' },
    });

    expect(screen.getByLabelText('API key location')).toBeTruthy();
  });

  it('renders API key auth using persisted query-or-header fields and no clear flags', () => {
    const onChange = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'api_key',
          api_key_location: 'query',
          api_key_name: 'api_key',
          api_key_prefix: 'Token',
        }}
        onChange={onChange}
        configuredSecretFields={[]}
      />,
    );

    expect((screen.getByLabelText('API key location') as HTMLSelectElement).value).toBe('query');
    expect((screen.getByLabelText('API key name') as HTMLInputElement).value).toBe('api_key');
    expect((screen.getByLabelText('API key prefix') as HTMLInputElement).value).toBe('Token');

    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'secret-key' } });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        api_key: 'secret-key',
        api_key_location: 'query',
        api_key_name: 'api_key',
      }),
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).not.toHaveProperty(
      'clear_api_key',
    );
  });

  it('renders basic and bearer auth fields', () => {
    const { rerender } = render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{ ...baseValue, auth_mode: 'basic', basic_username: 'alfred' }}
        onChange={vi.fn()}
        configuredSecretFields={[]}
      />,
    );

    expect((screen.getByLabelText('Basic username') as HTMLInputElement).value).toBe('alfred');
    expect(screen.getByLabelText('Basic password')).toBeTruthy();

    rerender(
      <HttpApiConnectionPanel
        section="authentication"
        value={{ ...baseValue, auth_mode: 'bearer', token_header_name: 'Authorization' }}
        onChange={vi.fn()}
        configuredSecretFields={[]}
      />,
    );

    expect(screen.getByLabelText('Bearer token')).toBeTruthy();
    expect((screen.getByLabelText('Token header name') as HTMLInputElement).value).toBe(
      'Authorization',
    );
  });

  it('renders login exchange fields including optional API key and request toggles', () => {
    const onChange = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'login_exchange',
          api_key_location: 'header',
          api_key_name: 'X-API-Key',
          login_path: '/session',
          login_method: 'POST',
          login_body_format: 'json',
          login_username: 'demo',
          login_username_field: 'email',
          login_password_field: 'password',
          send_api_key_to_login: true,
          send_api_key_to_requests: true,
          token_response_path: 'data.access_token',
          token_expires_in_path: 'data.expires_in',
          token_header_name: 'Authorization',
          token_prefix: 'Bearer',
        }}
        onChange={onChange}
        configuredSecretFields={[]}
      />,
    );

    expect((screen.getByLabelText('Login path') as HTMLInputElement).value).toBe('/session');
    expect((screen.getByLabelText('Login method') as HTMLSelectElement).value).toBe('POST');
    expect((screen.getByLabelText('Login body format') as HTMLSelectElement).value).toBe('json');
    expect((screen.getByLabelText('API key location') as HTMLSelectElement).value).toBe('header');
    expect((screen.getByLabelText('API key name') as HTMLInputElement).value).toBe('X-API-Key');
    expect(
      (screen.getByLabelText('Send API key to login exchange') as HTMLInputElement).checked,
    ).toBe(true);
    expect(
      (screen.getByLabelText('Send API key to authenticated requests') as HTMLInputElement).checked,
    ).toBe(true);
    expect((screen.getByLabelText('Token response path') as HTMLInputElement).value).toBe(
      'data.access_token',
    );
    expect((screen.getByLabelText('Token expires-in path') as HTMLInputElement).value).toBe(
      'data.expires_in',
    );
    expect((screen.getByLabelText('Token header name') as HTMLInputElement).value).toBe(
      'Authorization',
    );
    expect((screen.getByLabelText('Token prefix') as HTMLInputElement).value).toBe('Bearer');

    fireEvent.change(screen.getByLabelText('Login password'), { target: { value: 's3cret' } });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        login_password: 's3cret',
      }),
    );
  });

  it('shows explicit clear only for saved fixed secrets and clears by sending an empty secret value', () => {
    const onChange = vi.fn();

    const { rerender } = render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{ ...baseValue, auth_mode: 'api_key', api_key_name: 'X-API-Key' }}
        onChange={onChange}
        configuredSecretFields={['api_key']}
      />,
    );

    expect(screen.getByText('Saved value not shown')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Clear saved API key' }));

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        api_key: '',
      }),
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).not.toHaveProperty(
      'clear_api_key',
    );

    rerender(
      <HttpApiConnectionPanel
        section="authentication"
        value={{ ...baseValue, auth_mode: 'basic', basic_username: 'alfred' }}
        onChange={vi.fn()}
        configuredSecretFields={['basic_password']}
      />,
    );

    expect(screen.getByRole('button', { name: 'Clear saved Basic password' })).toBeTruthy();
  });

  it('does not inject an omitted saved fixed secret when an unrelated field changes', () => {
    const onChange = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{ ...baseValue, auth_mode: 'api_key', api_key_name: 'X-API-Key' }}
        onChange={onChange}
        configuredSecretFields={['api_key']}
      />,
    );

    fireEvent.change(screen.getByLabelText('API key name'), { target: { value: 'X-Auth-Key' } });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        api_key_name: 'X-Auth-Key',
      }),
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).not.toHaveProperty('api_key');
  });

  it('shows effective backend defaults without forcing those legacy keys into payload until changed', async () => {
    const onChange = vi.fn();

    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'login_exchange',
      });

      return (
        <HttpApiConnectionPanel
          section="authentication"
          value={value}
          onChange={(next) => {
            onChange(next);
            setValue(next);
          }}
          configuredSecretFields={[]}
        />
      );
    }

    render(<Harness />);

    expect((screen.getByLabelText('Login method') as HTMLSelectElement).value).toBe('POST');
    expect((screen.getByLabelText('Login body format') as HTMLSelectElement).value).toBe('json');
    expect((screen.getByLabelText('Username field') as HTMLInputElement).value).toBe('username');
    expect((screen.getByLabelText('Password field') as HTMLInputElement).value).toBe('password');
    expect((screen.getByLabelText('Token response path') as HTMLInputElement).value).toBe(
      'access_token',
    );
    expect((screen.getByLabelText('Token header name') as HTMLInputElement).value).toBe(
      'Authorization',
    );
    expect((screen.getByLabelText('Token prefix') as HTMLInputElement).value).toBe('Bearer');
    expect(screen.queryByLabelText('API key location')).toBeNull();
    expect(screen.queryByLabelText('API key name')).toBeNull();
    expect(
      (screen.getByLabelText('Send API key to authenticated requests') as HTMLInputElement).checked,
    ).toBe(false);

    fireEvent.click(screen.getByLabelText('Send API key to authenticated requests'));

    await waitFor(() => {
      expect((screen.getByLabelText('API key location') as HTMLSelectElement).value).toBe('header');
      expect(screen.getByLabelText('API key name')).toBeTruthy();
    });
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).toHaveProperty(
      'send_api_key_to_requests',
      true,
    );

    fireEvent.change(screen.getByLabelText('Login username'), { target: { value: 'alice' } });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        login_username: 'alice',
      }),
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).toHaveProperty(
      'send_api_key_to_requests',
      true,
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).not.toHaveProperty(
      'login_method',
    );
  });

  it('edits headers auth rows with case-insensitive saved paths and contextual add labels', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'headers',
        request_headers: [{ name: 'X-Tenant', value: '' }],
      });

      return (
        <HttpApiConnectionPanel
          section="authentication"
          value={value}
          onChange={setValue}
          configuredSecretFields={['request_headers.x-tenant']}
        />
      );
    }

    const { container } = render(<Harness />);

    expect(screen.getByText('Configured headers')).toBeTruthy();
    expect(screen.getByText('Saved')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Add configured header' })).toBeTruthy();
    expect(
      (screen.getByLabelText('Configured header value 1') as HTMLInputElement).placeholder,
    ).toBe('Saved value not shown');

    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'X-TENANT' },
    });

    expect(
      (screen.getByLabelText('Configured header value 1') as HTMLInputElement).placeholder,
    ).toBe('Saved value not shown');

    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    fireEvent.change(screen.getByLabelText('Configured header name 2'), {
      target: { value: 'X-Region' },
    });
    fireEvent.change(screen.getByLabelText('Configured header value 2'), {
      target: { value: 'emea' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Remove configured header 1' }));

    expect(container.querySelector('.http-api-header-row')).not.toBeNull();
    expect((screen.getByLabelText('Configured header name 1') as HTMLInputElement).value).toBe(
      'X-Region',
    );
  });

  it('shows inline validation and aria-invalid for incomplete or duplicate modern rows', async () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'token_exchange',
        login_path: '/oauth/token',
        token_request_fields: [{ name: 'client_secret', value: '', secret: true }],
        token_request_headers: [{ name: 'X-Key', value: '' }],
      });

      return (
        <HttpApiConnectionPanel
          section="authentication"
          value={value}
          onChange={setValue}
          configuredSecretFields={[
            'token_request_fields.client_secret',
            'token_request_headers.X-Key',
          ]}
        />
      );
    }

    render(<Harness />);

    expect(screen.getByRole('button', { name: 'Add token request field' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Add token request header' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Add configured header' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Add token request field' }));
    fireEvent.change(screen.getByLabelText('Token request field name 2'), {
      target: { value: 'client_secret' },
    });
    fireEvent.click(screen.getByLabelText('Token request field secret 2'));
    fireEvent.change(screen.getByLabelText('Token request field value 1'), {
      target: { value: '' },
    });
    fireEvent.change(screen.getByLabelText('Token request header name 1'), {
      target: { value: 'x-key' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Add token request header' }));
    fireEvent.change(screen.getByLabelText('Token request header name 2'), {
      target: { value: 'X-Key' },
    });

    await waitFor(() => {
      expect(screen.getByText('Value is required when Secret is off.')).toBeTruthy();
    });
    expect(screen.getAllByText('Duplicate token request field name.').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Duplicate token request header name.').length).toBeGreaterThan(0);
    expect(
      (screen.getByLabelText('Token request field value 2') as HTMLInputElement).getAttribute(
        'aria-invalid',
      ),
    ).toBe('true');
    expect(
      (screen.getByLabelText('Token request field name 2') as HTMLInputElement).getAttribute(
        'aria-invalid',
      ),
    ).toBe('true');
    expect(
      (screen.getByLabelText('Token request header name 2') as HTMLInputElement).getAttribute(
        'aria-invalid',
      ),
    ).toBe('true');
  });

  it('sends an explicit empty request_headers array when switching from headers mode to a different mode', () => {
    const onChange = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'headers',
          request_headers: [{ name: 'X-Tenant', value: 'tenant-a' }],
        }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'basic' },
    });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        auth_mode: 'basic',
        request_headers: [],
      }),
    );
  });

  it('sends explicit empty modern nested arrays when switching away from token exchange', () => {
    const onChange = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'token_exchange',
          login_path: '/oauth/token',
          request_headers: [{ name: 'X-Tenant', value: 'tenant-a' }],
          token_request_headers: [{ name: 'X-Key', value: 'key-a' }],
          token_request_fields: [{ name: 'client_secret', value: 'secret-a', secret: true }],
        }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'none' },
    });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        auth_mode: 'none',
        request_headers: [],
        token_request_headers: [],
        token_request_fields: [],
      }),
    );
  });

  it('keeps test connection controls in the authentication section', () => {
    const onTestConnection = vi.fn();

    const { rerender } = render(
      <HttpApiConnectionPanel
        section="authentication"
        value={baseValue}
        onChange={vi.fn()}
        onTestConnection={onTestConnection}
        testStatus={{
          state: 'success',
          message: 'Configuration is valid - no live request was sent.',
        }}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Test connection' }));
    expect(onTestConnection).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('status').textContent).toContain(
      'Configuration is valid - no live request was sent.',
    );

    rerender(<HttpApiConnectionPanel section="connection" value={baseValue} onChange={vi.fn()} />);
    expect(screen.queryByRole('button', { name: 'Test connection' })).toBeNull();
  });

  it('keeps selector and OpenAPI normalize controls in API details and sends uploaded document text to the callback', async () => {
    const onNormalizeOpenApi = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="api_details"
        value={{
          ...baseValue,
          openapi_source_url: 'https://api.example.com/openapi.json',
          default_response_selector: 'items',
        }}
        onChange={vi.fn()}
        onNormalizeOpenApi={onNormalizeOpenApi}
        openApiNormalizeStatus={{
          state: 'success',
          message: 'Normalized 8 operations.',
          operationCount: 8,
        }}
      />,
    );

    expect((screen.getByLabelText('OpenAPI URL') as HTMLInputElement).value).toBe(
      'https://api.example.com/openapi.json',
    );
    expect((screen.getByLabelText('Default response selector') as HTMLInputElement).value).toBe(
      'items',
    );
    expect(screen.getByText(/agent-settable header names/i)).toBeTruthy();

    const file = new File(['openapi: 3.1.0\ninfo:\n  title: Demo'], 'demo.yaml', {
      type: 'application/yaml',
    });
    fireEvent.change(screen.getByLabelText('OpenAPI file'), {
      target: { files: [file] },
    });

    fireEvent.click(screen.getByRole('button', { name: 'Normalize OpenAPI' }));

    await waitFor(() => {
      expect(onNormalizeOpenApi).toHaveBeenCalledWith({
        spec_url: 'https://api.example.com/openapi.json',
        document: 'openapi: 3.1.0\ninfo:\n  title: Demo',
        document_name: 'demo.yaml',
      });
    });

    expect(screen.getByText('Normalized 8 operations.')).toBeTruthy();
    expect(screen.getByText('8 operations')).toBeTruthy();
  });

  it('validates the base URL only after blur and clears the error after a valid http(s) blur', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        base_url: '',
      });

      return <HttpApiConnectionPanel section="connection" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    const input = screen.getByLabelText('Base URL');
    fireEvent.change(input, { target: { value: 'ftp://api.example.com' } });

    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');

    fireEvent.blur(input);

    expect(screen.getByText('Enter a valid http:// or https:// URL.')).toBeTruthy();
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('http-api-base-url-error');

    fireEvent.change(input, { target: { value: 'https://api.example.com' } });
    fireEvent.blur(input);

    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');
    expect(input.getAttribute('aria-describedby')).toBeNull();
  });

  it('normalizes base URL whitespace on blur and shows a required error for whitespace-only input', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        base_url: 'https://api.example.com',
      });

      return <HttpApiConnectionPanel section="connection" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    const input = screen.getByLabelText('Base URL') as HTMLInputElement;

    fireEvent.change(input, { target: { value: '   https://trimmed.example.com/path   ' } });
    fireEvent.blur(input);

    expect(input.value).toBe('https://trimmed.example.com/path');
    expect(screen.queryByText('Base URL is required.')).toBeNull();
    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();

    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.blur(input);

    expect(input.value).toBe('');
    expect(screen.getByText('Base URL is required.')).toBeTruthy();
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('http-api-base-url-error');
  });

  it('validates the OpenAPI URL only after blur, ignores empty values, and clears malformed URL errors on valid blur', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        openapi_source_url: '',
      });

      return <HttpApiConnectionPanel section="api_details" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    const input = screen.getByLabelText('OpenAPI URL');
    fireEvent.blur(input);

    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');

    fireEvent.change(input, { target: { value: 'not a url' } });
    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();

    fireEvent.blur(input);

    expect(screen.getByText('Enter a valid http:// or https:// URL.')).toBeTruthy();
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('http-api-openapi-url-error');

    fireEvent.change(input, { target: { value: 'http://api.example.com/openapi.json' } });
    fireEvent.blur(input);

    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');
  });

  it('normalizes optional OpenAPI URL whitespace to empty or a trimmed valid value on blur', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        openapi_source_url: '',
      });

      return <HttpApiConnectionPanel section="api_details" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    const input = screen.getByLabelText('OpenAPI URL') as HTMLInputElement;

    fireEvent.change(input, { target: { value: '   ' } });
    fireEvent.blur(input);

    expect(input.value).toBe('');
    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');

    fireEvent.change(input, {
      target: { value: '   https://api.example.com/openapi.json   ' },
    });
    fireEvent.blur(input);

    expect(input.value).toBe('https://api.example.com/openapi.json');
    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
  });

  it('matches the backend-compatible OpenAPI catalog shape', () => {
    const catalog: OpenApiCatalog = {
      title: 'Demo API',
      version: '1.0.0',
      operations: [
        {
          operation_id: 'listWidgets',
          method: 'GET',
          path: '/widgets',
          summary: 'List widgets',
          description: 'Returns widgets',
          tags: ['widgets'],
        },
      ],
    };

    expect(catalog).toEqual({
      title: 'Demo API',
      version: '1.0.0',
      operations: [
        {
          operation_id: 'listWidgets',
          method: 'GET',
          path: '/widgets',
          summary: 'List widgets',
          description: 'Returns widgets',
          tags: ['widgets'],
        },
      ],
    });
  });
});
