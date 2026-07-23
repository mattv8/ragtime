import { useState } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { HttpApiConnectionPanel } from './HttpApiConnectionPanel';
import type { HttpApiConnectionConfig, HttpApiSecretField, OpenApiCatalog } from '@/types';

type FixedSecretFieldKey = 'api_key' | 'basic_password' | 'login_password';

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
    expect(screen.getByText(/separate from the optional documentation URL/i)).toBeTruthy();
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

    expect(options).toEqual([
      'None',
      'Headers',
      'Basic authentication',
      'OAuth 2.0 / Token exchange',
      'OAuth 2.0 / Interactive',
    ]);
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
      'OAuth 2.0 / Token exchange',
      'OAuth 2.0 / Interactive',
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
      'OAuth 2.0 / Token exchange',
      'OAuth 2.0 / Interactive',
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

  it.each([
    {
      label: 'API key',
      value: {
        ...baseValue,
        auth_mode: 'api_key' as const,
        api_key_name: 'X-API-Key',
        api_key: 'stored-api-key',
      },
      configuredSecretFields: ['api_key'],
      toggleName: 'Show API key',
      nextToggleName: 'Hide API key',
      updatedSecret: 'updated-api-key',
      expectedSecretKey: 'api_key' as FixedSecretFieldKey,
    },
    {
      label: 'Basic password',
      value: {
        ...baseValue,
        auth_mode: 'basic' as const,
        basic_username: 'alfred',
        basic_password: 'stored-basic-password',
      },
      configuredSecretFields: ['basic_password'],
      toggleName: 'Show Basic password',
      nextToggleName: 'Hide Basic password',
      updatedSecret: 'updated-basic-password',
      expectedSecretKey: 'basic_password' as FixedSecretFieldKey,
    },
    {
      label: 'Login password',
      value: {
        ...baseValue,
        auth_mode: 'login_exchange' as const,
        login_password: 'stored-login-password',
      },
      configuredSecretFields: ['login_password'],
      toggleName: 'Show Login password',
      nextToggleName: 'Hide Login password',
      updatedSecret: 'updated-login-password',
      expectedSecretKey: 'login_password' as FixedSecretFieldKey,
    },
  ])(
    'masks loaded $label values, reveals locally, and keeps reveal out of onChange payloads',
    ({
      label,
      value,
      configuredSecretFields,
      toggleName,
      nextToggleName,
      updatedSecret,
      expectedSecretKey,
    }) => {
      const onChange = vi.fn();

      render(
        <HttpApiConnectionPanel
          section="authentication"
          value={value}
          onChange={onChange}
          configuredSecretFields={configuredSecretFields as HttpApiSecretField[]}
        />,
      );

      const input = screen.getByLabelText(label) as HTMLInputElement;
      expect(input.value).toBe(value[expectedSecretKey] ?? '');
      expect(input.type).toBe('password');
      expect(screen.queryByText('Saved')).toBeNull();
      expect(screen.queryByText('Saved value not shown')).toBeNull();
      expect(
        screen.queryByRole('button', { name: new RegExp(`Clear saved ${label}`, 'i') }),
      ).toBeNull();

      fireEvent.click(screen.getByRole('button', { name: toggleName }));

      expect(input.type).toBe('text');
      expect(screen.getByRole('button', { name: nextToggleName })).toBeTruthy();
      expect(onChange).not.toHaveBeenCalled();

      fireEvent.change(input, { target: { value: updatedSecret } });

      expect(onChange).toHaveBeenLastCalledWith(
        expect.objectContaining({
          [expectedSecretKey]: updatedSecret,
        }),
      );
    },
  );

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

  it('keeps empty resource header and body sections compact and reveals them per collection', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'headers',
        request_headers: [],
        request_body_fields: [],
      });

      return <HttpApiConnectionPanel section="authentication" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    expect(screen.queryByText('Configured headers')).toBeNull();
    expect(screen.queryByText('Request body')).toBeNull();
    expect(screen.queryByLabelText('Request body content type')).toBeNull();
    expect(
      screen
        .getByRole('button', { name: 'Add configured header' })
        .closest('.http-api-optional-action'),
    ).toBeTruthy();
    expect(
      screen
        .getByRole('button', { name: 'Add request body field' })
        .closest('.http-api-optional-action'),
    ).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    expect(screen.getByText('Configured headers')).toBeTruthy();
    expect(screen.queryByText('Request body')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Remove configured header 1' }));
    expect(screen.queryByText('Configured headers')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Add request body field' }));
    expect(screen.queryByText('Request body')).toBeNull();
    expect(screen.getByLabelText('Request body content type')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Remove request body field 1' }));
    expect(screen.queryByText('Request body')).toBeNull();
    expect(screen.queryByLabelText('Request body content type')).toBeNull();
  });

  it('keeps empty token request header and body sections compact and reveals them per collection', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'token_exchange',
        token_request_headers: [],
        token_request_fields: [],
        request_headers: [],
        request_body_fields: [],
      });

      return <HttpApiConnectionPanel section="authentication" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    expect(screen.queryByText('Token exchange')).toBeNull();
    expect(screen.queryByText('Token request headers')).toBeNull();
    expect(screen.queryByText('Token request body')).toBeNull();
    expect(screen.queryByLabelText('Token request body content type')).toBeNull();
    expect(screen.getByRole('button', { name: 'Add token request header' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Add token request field' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Add token request header' }));
    expect(screen.getByText('Token request headers')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Remove token request header 1' }));
    expect(screen.queryByText('Token request headers')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Add token request field' }));
    expect(screen.queryByText('Token request body')).toBeNull();
    expect(screen.getByLabelText('Token request body content type')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Remove token request field 1' }));
    expect(screen.queryByText('Token request body')).toBeNull();
    expect(screen.queryByLabelText('Token request body content type')).toBeNull();
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

  it('masks loaded configured header values, reveals one row locally, and keeps row editing stable', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'headers',
        request_headers: [
          { name: 'X-Tenant', value: 'tenant-secret' },
          { name: 'X-Region', value: 'region-secret' },
        ],
      });

      return (
        <HttpApiConnectionPanel
          section="authentication"
          value={value}
          onChange={setValue}
          configuredSecretFields={['request_headers.x-tenant', 'request_headers.x-region']}
        />
      );
    }

    const { container } = render(<Harness />);
    const firstValue = screen.getByLabelText('Configured header value 1') as HTMLInputElement;
    const secondValue = screen.getByLabelText('Configured header value 2') as HTMLInputElement;

    expect(screen.getByText('Configured headers')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Add configured header' })).toBeTruthy();
    expect(firstValue.value).toBe('tenant-secret');
    expect(secondValue.value).toBe('region-secret');
    expect(firstValue.type).toBe('password');
    expect(secondValue.type).toBe('password');
    expect(screen.queryByText('Saved')).toBeNull();
    expect(screen.queryByText('Saved value not shown')).toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Show configured header value 1' }));

    expect(firstValue.type).toBe('text');
    expect(secondValue.type).toBe('password');

    fireEvent.change(screen.getByLabelText('Configured header name 1'), {
      target: { value: 'X-TENANT' },
    });

    expect((screen.getByLabelText('Configured header value 1') as HTMLInputElement).type).toBe(
      'password',
    );
    expect((screen.getByLabelText('Configured header value 2') as HTMLInputElement).type).toBe(
      'password',
    );

    fireEvent.click(screen.getByRole('button', { name: 'Add configured header' }));
    fireEvent.change(screen.getByLabelText('Configured header name 3'), {
      target: { value: 'X-Env' },
    });
    fireEvent.change(screen.getByLabelText('Configured header value 3'), {
      target: { value: 'prod' },
    });
    fireEvent.click(screen.getByRole('button', { name: 'Remove configured header 1' }));

    expect(container.querySelector('.http-api-header-row')).not.toBeNull();
    expect((screen.getByLabelText('Configured header name 1') as HTMLInputElement).value).toBe(
      'X-Region',
    );
  });

  it('renders a compact request body editor after configured headers', () => {
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'headers',
          request_headers: [{ name: 'X-Tenant', value: 'tenant-a' }],
          request_body_format: 'json',
          request_body_fields: [{ name: 'tenant', value: 'configured', secret: false }],
        }}
        onChange={vi.fn()}
      />,
    );

    expect((screen.getByLabelText('Request body content type') as HTMLSelectElement).value).toBe(
      'json',
    );
    expect(screen.getByRole('button', { name: 'Add request body field' }).textContent).toContain(
      '+ Body',
    );
    expect(
      screen.getByRole('button', { name: 'Remove request body field 1' }).querySelector('svg'),
    ).toBeTruthy();
    const headerRows = document.querySelector('.http-api-header-row') as HTMLElement;
    const bodyAdd = screen.getByRole('button', { name: 'Add request body field' });
    expect(
      headerRows.compareDocumentPosition(bodyAdd) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('shows resource body validation errors while keeping loaded masked values revealable', () => {
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'headers',
          request_body_fields: [
            { name: '', value: '', secret: false },
            { name: 'tenant', value: 'tenant-secret', secret: false },
            { name: 'tenant', value: 'tenant-secret-2', secret: true },
            { name: 'Tenant', value: '', secret: false },
          ],
        }}
        configuredSecretFields={['request_body_fields.tenant']}
        onChange={vi.fn()}
      />,
    );

    expect(screen.getByText('Request body field name is required.')).toBeTruthy();
    expect(screen.getAllByText('Duplicate request body field name.')).toHaveLength(2);
    expect(screen.getAllByText('Field value is required.')).toHaveLength(2);
    expect(screen.queryByText('Saved')).toBeNull();
    expect(screen.queryByText('Saved value not shown')).toBeNull();
    expect(
      (screen.getByLabelText('Request body field value 3') as HTMLInputElement).getAttribute(
        'aria-invalid',
      ),
    ).toBe('false');
    expect(document.querySelectorAll('input[type="password"]')).toHaveLength(4);
    expect(screen.queryByLabelText('Request body field secret 1')).toBeNull();
    expect((screen.getByLabelText('Request body field value 2') as HTMLInputElement).value).toBe(
      'tenant-secret',
    );
    expect((screen.getByLabelText('Request body field value 3') as HTMLInputElement).value).toBe(
      'tenant-secret-2',
    );
    expect(
      (screen.getByLabelText('Request body field name 4') as HTMLInputElement).getAttribute(
        'aria-invalid',
      ),
    ).toBe('false');
  });

  it('masks body values by default and reveals one row without changing its secret payload', () => {
    const onChange = vi.fn();
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'headers',
          request_body_fields: [{ name: 'tenant', value: 'configured', secret: false }],
        }}
        onChange={onChange}
      />,
    );

    const valueInput = screen.getByLabelText('Request body field value 1') as HTMLInputElement;
    const toggle = screen.getByRole('button', { name: 'Show request body field value 1' });
    expect(valueInput.type).toBe('password');

    fireEvent.click(toggle);

    expect(valueInput.type).toBe('text');
    expect(screen.getByRole('button', { name: 'Hide request body field value 1' })).toBeTruthy();
    expect(onChange).not.toHaveBeenCalled();

    fireEvent.change(valueInput, { target: { value: 'updated' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        request_body_fields: [{ name: 'tenant', value: 'updated', secret: true }],
      }),
    );
  });

  it.each([
    {
      valueKey: 'Configured header' as const,
      collectionKey: 'request_headers' as const,
      updatedValue: 'updated-header-secret',
      value: {
        ...baseValue,
        auth_mode: 'headers' as const,
        request_headers: [{ name: 'X-Key', value: 'header-secret' }],
      },
      toggleName: 'Show configured header value 1',
    },
    {
      valueKey: 'Token request header' as const,
      collectionKey: 'token_request_headers' as const,
      updatedValue: 'updated-token-secret',
      value: {
        ...baseValue,
        auth_mode: 'token_exchange' as const,
        token_request_headers: [{ name: 'X-Client', value: 'token-secret' }],
      },
      toggleName: 'Show token request header value 1',
    },
  ])(
    'masks and locally reveals $valueKey values without changing the payload',
    ({ valueKey, collectionKey, updatedValue, value, toggleName }) => {
      const onChange = vi.fn();
      render(<HttpApiConnectionPanel section="authentication" value={value} onChange={onChange} />);

      const valueInput = screen.getByLabelText(`${valueKey} value 1`) as HTMLInputElement;
      const toggle = screen.getByRole('button', { name: toggleName });
      expect(valueInput.type).toBe('password');

      fireEvent.click(toggle);

      expect(valueInput.type).toBe('text');
      expect(screen.getByRole('button', { name: toggleName.replace('Show', 'Hide') })).toBeTruthy();
      expect(onChange).not.toHaveBeenCalled();

      fireEvent.change(valueInput, { target: { value: updatedValue } });

      const changedValue = onChange.mock.lastCall?.[0] as HttpApiConnectionConfig;
      expect(changedValue[collectionKey]).toEqual([
        { name: value[collectionKey]?.[0]?.name, value: updatedValue },
      ]);
    },
  );

  it.each([
    {
      value: {
        ...baseValue,
        auth_mode: 'headers' as const,
        request_headers: [{ name: 'X-Key', value: 'header-secret' }],
      },
      inputLabel: 'Configured header value 1',
      toggleName: 'Show configured header value 1',
    },
    {
      value: {
        ...baseValue,
        auth_mode: 'token_exchange' as const,
        token_request_headers: [{ name: 'X-Client', value: 'token-secret' }],
      },
      inputLabel: 'Token request header value 1',
      toggleName: 'Show token request header value 1',
    },
    {
      value: {
        ...baseValue,
        auth_mode: 'headers' as const,
        request_body_fields: [{ name: 'tenant', value: 'body-secret', secret: true }],
      },
      inputLabel: 'Request body field value 1',
      toggleName: 'Show request body field value 1',
    },
  ])(
    'keeps $inputLabel revealed when editing through stateful rerender',
    ({ value, inputLabel, toggleName }) => {
      function Harness() {
        const [currentValue, setCurrentValue] = useState<HttpApiConnectionConfig>(value);
        return (
          <HttpApiConnectionPanel
            section="authentication"
            value={currentValue}
            onChange={setCurrentValue}
          />
        );
      }

      render(<Harness />);

      fireEvent.click(screen.getByRole('button', { name: toggleName }));
      expect((screen.getByLabelText(inputLabel) as HTMLInputElement).type).toBe('text');

      fireEvent.change(screen.getByLabelText(inputLabel), { target: { value: 'updated-secret' } });

      expect((screen.getByLabelText(inputLabel) as HTMLInputElement).type).toBe('text');
    },
  );

  it('keeps reveal state scoped to a row and resets it when a row is renamed or removed', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'headers',
        request_body_fields: [
          { name: 'first', value: 'one', secret: true },
          { name: 'second', value: 'two', secret: true },
        ],
      });
      return <HttpApiConnectionPanel section="authentication" value={value} onChange={setValue} />;
    }

    render(<Harness />);
    fireEvent.click(screen.getByRole('button', { name: 'Show request body field value 2' }));
    expect((screen.getByLabelText('Request body field value 2') as HTMLInputElement).type).toBe(
      'text',
    );

    fireEvent.change(screen.getByLabelText('Request body field name 1'), {
      target: { value: 'renamed' },
    });
    expect((screen.getByLabelText('Request body field value 1') as HTMLInputElement).type).toBe(
      'password',
    );
    expect((screen.getByLabelText('Request body field value 2') as HTMLInputElement).type).toBe(
      'text',
    );

    fireEvent.click(screen.getByRole('button', { name: 'Remove request body field 1' }));
    expect((screen.getByLabelText('Request body field value 1') as HTMLInputElement).type).toBe(
      'text',
    );
  });

  it('renders token endpoint and keeps token and resource body sections distinct', () => {
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'token_exchange',
          login_path: '/legacy-token',
          token_url: '/oauth/token',
          token_request_headers: [{ name: 'X-Client', value: 'client-a' }],
          token_request_fields: [
            { name: 'grant_type', value: 'client_credentials', secret: false },
          ],
          request_headers: [{ name: 'X-Tenant', value: 'tenant-a' }],
          request_body_format: 'form',
          request_body_fields: [{ name: 'tenant', value: 'configured', secret: false }],
        }}
        onChange={vi.fn()}
      />,
    );

    expect((screen.getByLabelText('Token endpoint') as HTMLInputElement).value).toBe(
      '/oauth/token',
    );
    expect(screen.getByText('Token request headers')).toBeTruthy();
    expect(screen.queryByText('Token request body')).toBeNull();
    expect(screen.getByText('Configured headers')).toBeTruthy();
    expect(screen.queryByText('Request body')).toBeNull();
    expect((screen.getByLabelText('Request body content type') as HTMLSelectElement).value).toBe(
      'form',
    );
  });

  it('migrates serialized legacy token endpoints when token_url is an empty string', () => {
    const onChange = vi.fn();

    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'token_exchange',
          token_url: '',
          login_path: '/session',
        }}
        onChange={onChange}
      />,
    );

    expect((screen.getByLabelText('Token endpoint') as HTMLInputElement).value).toBe('/session');

    fireEvent.change(screen.getByLabelText('Token endpoint'), {
      target: { value: '/oauth/token' },
    });

    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ token_url: '/oauth/token', login_path: '' }),
    );
  });

  it('preserves resource body fields between headers and token exchange but clears them for basic auth', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        auth_mode: 'headers',
        request_body_format: 'json',
        request_body_fields: [{ name: 'tenant', value: 'configured', secret: false }],
      });
      return <HttpApiConnectionPanel section="authentication" value={value} onChange={setValue} />;
    }

    render(<Harness />);
    fireEvent.change(screen.getByLabelText('Authentication mode'), {
      target: { value: 'token_exchange' },
    });
    expect((screen.getByLabelText('Request body field name 1') as HTMLInputElement).value).toBe(
      'tenant',
    );
    fireEvent.change(screen.getByLabelText('Authentication mode'), { target: { value: 'basic' } });
    expect(screen.queryByLabelText('Request body field name 1')).toBeNull();
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
      expect(screen.queryByText('Field value is required.')).toBeNull();
    });
    expect(screen.getAllByText('Duplicate token request field name.').length).toBeGreaterThan(0);
    expect(screen.getAllByText('Duplicate token request header name.').length).toBeGreaterThan(0);
    expect(
      (screen.getByLabelText('Token request field value 2') as HTMLInputElement).getAttribute(
        'aria-invalid',
      ),
    ).toBe('false');
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
    expect(
      screen
        .getByRole('button', { name: 'Remove token request header 2' })
        .closest('.http-api-row-actions-input-aligned'),
    ).toBeTruthy();
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

  it('renders optional documentation and request policy details without OpenAPI controls', () => {
    render(
      <HttpApiConnectionPanel
        section="api_details"
        value={{
          ...baseValue,
          documentation_url: 'https://api.example.com/docs',
          default_response_selector: 'items',
        }}
        onChange={vi.fn()}
      />,
    );

    expect(
      (screen.getByLabelText('API documentation URL (optional)') as HTMLInputElement).value,
    ).toBe('https://api.example.com/docs');
    expect(
      (screen.getByLabelText('Default response selector (optional)') as HTMLInputElement).value,
    ).toBe('items');
    expect(screen.getByText(/agent-settable header names/i)).toBeTruthy();
    expect(screen.getByLabelText('Approved request headers (optional)')).toBeTruthy();
    expect(screen.getByText(/not used as the request Base URL/i)).toBeTruthy();
    expect(screen.getByText(/does not configure fixed header values/i)).toBeTruthy();
    expect(screen.getByText(/optional dot-path used to select/i)).toBeTruthy();
    expect(screen.queryByLabelText('OpenAPI URL')).toBeNull();
    expect(screen.queryByLabelText('OpenAPI file')).toBeNull();
    expect(screen.queryByRole('button', { name: 'Normalize OpenAPI' })).toBeNull();
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

  it('validates the documentation URL only after blur, ignores empty values, and clears malformed URL errors on valid blur', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        documentation_url: '',
      });

      return <HttpApiConnectionPanel section="api_details" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    const input = screen.getByLabelText('API documentation URL (optional)');
    fireEvent.blur(input);

    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');

    fireEvent.change(input, { target: { value: 'not a url' } });
    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();

    fireEvent.blur(input);

    expect(screen.getByText('Enter a valid http:// or https:// URL.')).toBeTruthy();
    expect(input.getAttribute('aria-invalid')).toBe('true');
    expect(input.getAttribute('aria-describedby')).toBe('http-api-documentation-url-error');

    fireEvent.change(input, { target: { value: 'http://api.example.com/openapi.json' } });
    fireEvent.blur(input);

    expect(screen.queryByText('Enter a valid http:// or https:// URL.')).toBeNull();
    expect(input.getAttribute('aria-invalid')).toBe('false');
  });

  it('normalizes optional documentation URL whitespace to empty or a trimmed valid value on blur', () => {
    function Harness() {
      const [value, setValue] = useState<HttpApiConnectionConfig>({
        ...baseValue,
        documentation_url: '',
      });

      return <HttpApiConnectionPanel section="api_details" value={value} onChange={setValue} />;
    }

    render(<Harness />);

    const input = screen.getByLabelText('API documentation URL (optional)') as HTMLInputElement;

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

  it('renders interactive OAuth in the authentication panel and clears it when leaving', () => {
    const onChange = vi.fn();
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'oauth2',
          oauth_issuer_url: 'https://issuer.example.test',
          oauth_client_id: 'client-id',
        }}
        onChange={onChange}
      />,
    );

    expect(screen.getByRole('region', { name: 'OAuth 2.0 connection' })).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Authentication mode'), { target: { value: 'none' } });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({
        auth_mode: 'none',
        oauth_client_secret: '',
        oauth_access_token: '',
        oauth_refresh_token: '',
      }),
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).not.toHaveProperty(
      'oauth_session_id',
    );
  });

  it('invalidates a pending OAuth session when provider configuration changes', () => {
    const onChange = vi.fn();
    render(
      <HttpApiConnectionPanel
        section="authentication"
        value={{
          ...baseValue,
          auth_mode: 'oauth2',
          oauth_session_id: 'pending-session',
        }}
        onChange={onChange}
      />,
    );

    fireEvent.change(screen.getByLabelText('Issuer URL'), {
      target: { value: 'https://other.example.test' },
    });
    expect(onChange).toHaveBeenLastCalledWith(
      expect.objectContaining({ oauth_issuer_url: 'https://other.example.test' }),
    );
    expect(onChange.mock.calls[onChange.mock.calls.length - 1]?.[0]).not.toHaveProperty(
      'oauth_session_id',
    );
  });
});
