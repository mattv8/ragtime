import { useEffect, useState } from 'react';

import {
  DEFAULT_HTTP_API_METHOD_POLICIES,
  HTTP_API_METHODS,
  type HttpApiAuthMode,
  type HttpApiConfiguredHeader,
  type HttpApiConnectionConfig,
  type HttpApiFixedSecretField,
  type HttpApiHttpMethod,
  type HttpApiMethodPolicy,
  type HttpApiSecretField,
} from '@/types';

type HttpApiPanelSection = 'connection' | 'authentication' | 'api_details';

interface ActionStatus {
  state: 'idle' | 'pending' | 'success' | 'error';
  message?: string | null;
  operationCount?: number | null;
}

interface NormalizeOpenApiInput {
  spec_url?: string;
  document?: string;
  document_name?: string;
}

interface HttpApiConnectionPanelProps {
  section: HttpApiPanelSection;
  value: HttpApiConnectionConfig;
  onChange: (value: HttpApiConnectionConfig) => void;
  configuredSecretFields?: HttpApiSecretField[];
  testStatus?: ActionStatus;
  onTestConnection?: () => void;
  testDisabled?: boolean;
  openApiNormalizeStatus?: ActionStatus;
  onNormalizeOpenApi?: (input: NormalizeOpenApiInput) => void | Promise<void>;
  normalizeDisabled?: boolean;
}

type UrlFieldKey = 'base_url' | 'openapi_source_url';
type HeaderRowKey = 'request_headers' | 'token_request_headers';

interface RowValidationState {
  nameError?: string;
  valueError?: string;
}

const URL_FORMAT_ERROR = 'Enter a valid http:// or https:// URL.';
const BASE_URL_REQUIRED_ERROR = 'Base URL is required.';

const MODERN_AUTH_MODE_OPTIONS: Array<{ value: HttpApiAuthMode; label: string }> = [
  { value: 'none', label: 'None' },
  { value: 'headers', label: 'Headers' },
  { value: 'basic', label: 'Basic authentication' },
  { value: 'token_exchange', label: 'Token exchange' },
];

const LEGACY_AUTH_MODE_OPTIONS: Array<{ value: HttpApiAuthMode; label: string }> = [
  { value: 'api_key', label: 'Legacy API key' },
  { value: 'bearer', label: 'Legacy Bearer token' },
  { value: 'login_exchange', label: 'Legacy login exchange' },
];

const LEGACY_AUTH_MODES: HttpApiAuthMode[] = ['api_key', 'bearer', 'login_exchange'];
const METHOD_POLICY_OPTIONS: HttpApiMethodPolicy[] = ['disabled', 'read', 'write'];
const DEFAULT_LOGIN_USERNAME_FIELD = 'username';
const DEFAULT_LOGIN_PASSWORD_FIELD = 'password';
const DEFAULT_TOKEN_RESPONSE_PATH = 'access_token';
const DEFAULT_TOKEN_HEADER_NAME = 'Authorization';
const DEFAULT_TOKEN_PREFIX = 'Bearer';

function toDelimitedList(value: string): string[] {
  return value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function getMethodPolicy(
  config: HttpApiConnectionConfig,
  method: HttpApiHttpMethod,
): HttpApiMethodPolicy {
  return config.method_policies?.[method] ?? DEFAULT_HTTP_API_METHOD_POLICIES[method];
}

async function readTextFile(file: File): Promise<string> {
  if (typeof file.text === 'function') {
    return await file.text();
  }

  return await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file'));
    reader.readAsText(file);
  });
}

function validateHttpUrl(value: string | undefined): string | null {
  const trimmedValue = value?.trim() ?? '';
  if (!trimmedValue) {
    return null;
  }

  try {
    const parsedUrl = new URL(trimmedValue);
    if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
      return URL_FORMAT_ERROR;
    }
  } catch {
    return URL_FORMAT_ERROR;
  }

  return null;
}

function normalizeUrlValue(value: string | undefined): string {
  return value?.trim() ?? '';
}

function removeConfigKeys(
  config: HttpApiConnectionConfig,
  keys: Array<keyof HttpApiConnectionConfig>,
): HttpApiConnectionConfig {
  const nextConfig = { ...config } as Partial<HttpApiConnectionConfig>;
  keys.forEach((key) => {
    delete nextConfig[key];
  });
  return nextConfig as HttpApiConnectionConfig;
}

function getClearedConfigForAuthModeChange(
  currentValue: HttpApiConnectionConfig,
  nextAuthMode: HttpApiAuthMode,
): HttpApiConnectionConfig {
  const currentMode = currentValue.auth_mode ?? 'none';
  if (currentMode === nextAuthMode) {
    return currentValue;
  }

  let nextValue: HttpApiConnectionConfig = { ...currentValue, auth_mode: nextAuthMode };

  if (
    currentMode === 'headers' &&
    nextAuthMode !== 'headers' &&
    nextAuthMode !== 'token_exchange'
  ) {
    nextValue = { ...nextValue, request_headers: [] };
  }

  if (currentMode === 'token_exchange' && nextAuthMode !== 'token_exchange') {
    const keysToClear: Array<keyof HttpApiConnectionConfig> = [
      'login_path',
      'login_method',
      'login_body_format',
      'token_response_path',
      'token_expires_in_path',
      'token_header_name',
      'token_prefix',
    ];
    nextValue = {
      ...nextValue,
      token_request_fields: [],
      token_request_headers: [],
      ...(nextAuthMode !== 'headers' ? { request_headers: [] } : {}),
    };
    if (nextAuthMode !== 'headers') {
      nextValue = { ...nextValue, request_headers: [] };
    }
    nextValue = removeConfigKeys(nextValue, keysToClear);
  }

  return nextValue;
}

export function HttpApiConnectionPanel({
  section,
  value,
  onChange,
  configuredSecretFields = [],
  testStatus,
  onTestConnection,
  testDisabled = false,
  openApiNormalizeStatus,
  onNormalizeOpenApi,
  normalizeDisabled = false,
}: HttpApiConnectionPanelProps) {
  const [uploadedOpenApiFile, setUploadedOpenApiFile] = useState<File>();
  const [uploadedOpenApiDocument, setUploadedOpenApiDocument] = useState<string>();
  const [uploadedOpenApiDocumentName, setUploadedOpenApiDocumentName] = useState<string>();
  const [urlErrors, setUrlErrors] = useState<Partial<Record<UrlFieldKey, string>>>({});
  const [legacyOptionsUnlocked, setLegacyOptionsUnlocked] = useState(() =>
    LEGACY_AUTH_MODES.includes(value.auth_mode ?? 'none'),
  );
  const authMode = value.auth_mode ?? 'none';

  useEffect(() => {
    if (LEGACY_AUTH_MODES.includes(authMode)) {
      setLegacyOptionsUnlocked(true);
    }
  }, [authMode]);

  const updateValue = <K extends keyof HttpApiConnectionConfig>(
    key: K,
    nextValue: HttpApiConnectionConfig[K],
  ) => {
    onChange({ ...value, [key]: nextValue });
  };

  const updateSecret = (key: HttpApiFixedSecretField, nextValue: string) => {
    onChange({ ...value, [key]: nextValue });
  };

  const clearSecret = (key: HttpApiFixedSecretField) => {
    onChange({ ...value, [key]: '' });
  };

  const handleAuthModeChange = (nextAuthMode: HttpApiAuthMode) => {
    if (LEGACY_AUTH_MODES.includes(nextAuthMode)) {
      setLegacyOptionsUnlocked(true);
    }
    onChange(getClearedConfigForAuthModeChange(value, nextAuthMode));
  };

  const hasSavedHeaderSecret = (key: HeaderRowKey, name: string): boolean => {
    if (!name.trim()) {
      return false;
    }
    const targetPathPrefix = `${key}.`;
    const lowerName = name.trim().toLowerCase();
    return configuredSecretFields.some((field) => {
      if (!field.startsWith(targetPathPrefix)) {
        return false;
      }
      return field.slice(targetPathPrefix.length).toLowerCase() === lowerName;
    });
  };

  const hasSavedTokenFieldSecret = (name: string): boolean => {
    if (!name.trim()) {
      return false;
    }
    return configuredSecretFields.includes(`token_request_fields.${name.trim()}`);
  };

  const buildHeaderRowValidation = (
    rows: HttpApiConfiguredHeader[] | undefined,
    key: HeaderRowKey,
  ): RowValidationState[] => {
    const nextValidation: RowValidationState[] = [];
    const counts = new Map<string, number>();

    (rows ?? []).forEach((row) => {
      const trimmedName = row.name.trim().toLowerCase();
      if (!trimmedName) {
        return;
      }
      counts.set(trimmedName, (counts.get(trimmedName) ?? 0) + 1);
    });

    (rows ?? []).forEach((row, index) => {
      const trimmedName = row.name.trim();
      const loweredName = trimmedName.toLowerCase();
      const hasSavedValue = hasSavedHeaderSecret(key, trimmedName);
      const errors: RowValidationState = {};

      if (!trimmedName) {
        errors.nameError = 'Header name is required.';
      } else if ((counts.get(loweredName) ?? 0) > 1) {
        errors.nameError =
          key === 'token_request_headers'
            ? 'Duplicate token request header name.'
            : 'Duplicate configured header name.';
      }

      if (!row.value && !hasSavedValue) {
        errors.valueError = 'Header value is required.';
      }

      nextValidation[index] = errors;
    });

    return nextValidation;
  };

  const headerRowValidation = {
    request_headers: buildHeaderRowValidation(value.request_headers, 'request_headers'),
    token_request_headers: buildHeaderRowValidation(
      value.token_request_headers,
      'token_request_headers',
    ),
  };

  const tokenFieldValidation = (() => {
    const counts = new Map<string, number>();
    (value.token_request_fields ?? []).forEach((row) => {
      const trimmedName = row.name.trim();
      if (!trimmedName) {
        return;
      }
      counts.set(trimmedName, (counts.get(trimmedName) ?? 0) + 1);
    });

    return (value.token_request_fields ?? []).map((row) => {
      const trimmedName = row.name.trim();
      const hasSavedValue = row.secret && hasSavedTokenFieldSecret(trimmedName);
      const errors: RowValidationState = {};

      if (!trimmedName) {
        errors.nameError = 'Token request field name is required.';
      } else if ((counts.get(trimmedName) ?? 0) > 1) {
        errors.nameError = 'Duplicate token request field name.';
      }

      if (!row.value && (!row.secret || !hasSavedValue)) {
        errors.valueError = row.secret
          ? 'Field value is required.'
          : 'Value is required when Secret is off.';
      }

      return errors;
    });
  })();

  const handleUrlBlur = (key: UrlFieldKey, nextValue: string | undefined) => {
    const normalizedValue = normalizeUrlValue(nextValue);
    if (normalizedValue !== (value[key] ?? '')) {
      updateValue(key, normalizedValue);
    }

    const error =
      key === 'base_url' && !normalizedValue
        ? BASE_URL_REQUIRED_ERROR
        : validateHttpUrl(normalizedValue);

    setUrlErrors((current) => {
      if (!error) {
        const { [key]: _removed, ...remainingErrors } = current;
        return remainingErrors;
      }
      return { ...current, [key]: error };
    });
  };

  const renderSecretField = ({
    label,
    inputId,
    field,
    value: fieldValue,
  }: {
    label: string;
    inputId: string;
    field: HttpApiFixedSecretField;
    value: string | undefined;
  }) => {
    const hasSavedSecret = configuredSecretFields.includes(field) && !fieldValue;

    return (
      <div className="form-group">
        <label htmlFor={inputId}>{label}</label>
        <div className="http-api-secret-row">
          <input
            id={inputId}
            type="password"
            value={fieldValue ?? ''}
            onChange={(event) => updateSecret(field, event.target.value)}
            placeholder={hasSavedSecret ? 'Saved value not shown' : `Enter ${label.toLowerCase()}`}
            autoComplete="off"
          />
          {hasSavedSecret && (
            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => clearSecret(field)}
            >
              Clear saved {label}
            </button>
          )}
        </div>
        {hasSavedSecret && <p className="field-help">Saved value not shown</p>}
      </div>
    );
  };

  const renderActionStatus = (status: ActionStatus | undefined) => {
    if (!status || (!status.message && !status.operationCount)) {
      return null;
    }

    const className = ['http-api-status', status.state !== 'idle' ? `is-${status.state}` : '']
      .filter(Boolean)
      .join(' ');

    return (
      <div className={className} role="status" aria-live="polite">
        {status.message && <span>{status.message}</span>}
        {typeof status.operationCount === 'number' && (
          <span className="http-api-status-count">{status.operationCount} operations</span>
        )}
      </div>
    );
  };

  const renderHeaderRows = ({
    groupLabel,
    itemLabel,
    addAriaLabel,
    valueKey,
    namePrefix,
  }: {
    groupLabel: string;
    itemLabel: string;
    addAriaLabel: string;
    valueKey: HeaderRowKey;
    namePrefix: string;
  }) => {
    const rows = value[valueKey] ?? [];
    const validation = headerRowValidation[valueKey];

    return (
      <div className="http-api-subsection">
        <div className="http-api-subsection-header">
          <p className="http-api-subsection-label">{groupLabel}</p>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            aria-label={addAriaLabel}
            onClick={() => onChange({ ...value, [valueKey]: [...rows, { name: '', value: '' }] })}
          >
            + Header
          </button>
        </div>
        {rows.map((row, index) => {
          const position = index + 1;
          const hasSavedValue = hasSavedHeaderSecret(valueKey, row.name) && !row.value;
          const rowValidation = validation[index] ?? {};

          return (
            <div key={`${namePrefix}-${position}`} className="http-api-header-row">
              <div className="form-group">
                <label htmlFor={`http-api-${namePrefix}-name-${position}`}>Header name</label>
                <input
                  id={`http-api-${namePrefix}-name-${position}`}
                  aria-label={`${itemLabel} name ${position}`}
                  aria-invalid={rowValidation.nameError ? 'true' : 'false'}
                  type="text"
                  value={row.name}
                  onChange={(event) => {
                    const nextRows = rows.map((currentRow, currentIndex) =>
                      currentIndex === index
                        ? { ...currentRow, name: event.target.value }
                        : currentRow,
                    );
                    onChange({ ...value, [valueKey]: nextRows });
                  }}
                  autoComplete="off"
                />
                {rowValidation.nameError && (
                  <p className="field-error">{rowValidation.nameError}</p>
                )}
              </div>
              <div className="form-group">
                <label htmlFor={`http-api-${namePrefix}-value-${position}`}>Header value</label>
                <input
                  id={`http-api-${namePrefix}-value-${position}`}
                  aria-label={`${itemLabel} value ${position}`}
                  aria-invalid={rowValidation.valueError ? 'true' : 'false'}
                  type="password"
                  value={row.value}
                  onChange={(event) => {
                    const nextRows = rows.map((currentRow, currentIndex) =>
                      currentIndex === index
                        ? { ...currentRow, value: event.target.value }
                        : currentRow,
                    );
                    onChange({ ...value, [valueKey]: nextRows });
                  }}
                  placeholder={hasSavedValue ? 'Saved value not shown' : ''}
                  autoComplete="off"
                />
                {rowValidation.valueError && (
                  <p className="field-error">{rowValidation.valueError}</p>
                )}
                {!rowValidation.valueError && hasSavedValue && (
                  <p className="field-help">Saved value not shown</p>
                )}
              </div>
              <div className="http-api-row-actions">
                {hasSavedValue && <span className="http-api-saved-badge">Saved</span>}
                <button
                  type="button"
                  className="btn btn-danger btn-sm btn-icon"
                  aria-label={`Remove ${itemLabel.toLowerCase()} ${position}`}
                  onClick={() =>
                    onChange({
                      ...value,
                      [valueKey]: rows.filter((_, currentIndex) => currentIndex !== index),
                    })
                  }
                >
                  ×
                </button>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const renderTokenFieldRows = () => {
    const rows = value.token_request_fields ?? [];

    return (
      <div className="http-api-subsection">
        <div className="http-api-subsection-header">
          <p className="http-api-subsection-label">Token request fields</p>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            aria-label="Add token request field"
            onClick={() =>
              onChange({
                ...value,
                token_request_fields: [...rows, { name: '', value: '', secret: true }],
              })
            }
          >
            + Field
          </button>
        </div>
        {rows.map((row, index) => {
          const position = index + 1;
          const hasSavedValue = row.secret && hasSavedTokenFieldSecret(row.name) && !row.value;
          const rowValidation = tokenFieldValidation[index] ?? {};

          return (
            <div key={`token-request-field-${position}`} className="http-api-field-row">
              <div className="form-group">
                <label htmlFor={`http-api-token-request-field-name-${position}`}>Field name</label>
                <input
                  id={`http-api-token-request-field-name-${position}`}
                  aria-label={`Token request field name ${position}`}
                  aria-invalid={rowValidation.nameError ? 'true' : 'false'}
                  type="text"
                  value={row.name}
                  onChange={(event) => {
                    const nextRows = rows.map((currentRow, currentIndex) =>
                      currentIndex === index
                        ? { ...currentRow, name: event.target.value }
                        : currentRow,
                    );
                    onChange({ ...value, token_request_fields: nextRows });
                  }}
                  autoComplete="off"
                />
                {rowValidation.nameError && (
                  <p className="field-error">{rowValidation.nameError}</p>
                )}
              </div>
              <div className="form-group">
                <label htmlFor={`http-api-token-request-field-value-${position}`}>
                  Field value
                </label>
                <input
                  id={`http-api-token-request-field-value-${position}`}
                  aria-label={`Token request field value ${position}`}
                  aria-invalid={rowValidation.valueError ? 'true' : 'false'}
                  type={row.secret ? 'password' : 'text'}
                  value={row.value}
                  onChange={(event) => {
                    const nextRows = rows.map((currentRow, currentIndex) =>
                      currentIndex === index
                        ? { ...currentRow, value: event.target.value }
                        : currentRow,
                    );
                    onChange({ ...value, token_request_fields: nextRows });
                  }}
                  placeholder={hasSavedValue ? 'Saved value not shown' : ''}
                  autoComplete="off"
                />
                {rowValidation.valueError && (
                  <p className="field-error">{rowValidation.valueError}</p>
                )}
                {!rowValidation.valueError && hasSavedValue && (
                  <p className="field-help">Saved value not shown</p>
                )}
              </div>
              <label
                className="http-api-secret-toggle"
                htmlFor={`http-api-token-request-field-secret-${position}`}
              >
                <input
                  id={`http-api-token-request-field-secret-${position}`}
                  aria-label={`Token request field secret ${position}`}
                  type="checkbox"
                  checked={row.secret}
                  onChange={(event) => {
                    const nextRows = rows.map((currentRow, currentIndex) =>
                      currentIndex === index
                        ? {
                            ...currentRow,
                            secret: event.target.checked,
                            value: event.target.checked ? currentRow.value : '',
                          }
                        : currentRow,
                    );
                    onChange({ ...value, token_request_fields: nextRows });
                  }}
                />
                Secret
              </label>
              <div className="http-api-row-actions">
                {hasSavedValue && <span className="http-api-saved-badge">Saved</span>}
                <button
                  type="button"
                  className="btn btn-danger btn-sm btn-icon"
                  aria-label={`Remove token request field ${position}`}
                  onClick={() =>
                    onChange({
                      ...value,
                      token_request_fields: rows.filter(
                        (_, currentIndex) => currentIndex !== index,
                      ),
                    })
                  }
                >
                  ×
                </button>
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  const handleOpenApiFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      setUploadedOpenApiFile(undefined);
      setUploadedOpenApiDocument(undefined);
      setUploadedOpenApiDocumentName(undefined);
      return;
    }

    setUploadedOpenApiFile(file);
    setUploadedOpenApiDocument(await readTextFile(file));
    setUploadedOpenApiDocumentName(file.name);
  };

  const handleNormalizeOpenApi = async () => {
    if (!onNormalizeOpenApi) {
      return;
    }
    const document =
      uploadedOpenApiDocument ??
      (uploadedOpenApiFile ? await readTextFile(uploadedOpenApiFile) : undefined);
    void onNormalizeOpenApi({
      spec_url: value.openapi_source_url,
      document,
      document_name: uploadedOpenApiDocumentName ?? uploadedOpenApiFile?.name,
    });
  };

  const baseUrlError = urlErrors.base_url;
  const openApiUrlError = urlErrors.openapi_source_url;
  const hasConfiguredApiKey = Boolean(
    value.api_key_name ||
    value.api_key_prefix ||
    value.api_key ||
    configuredSecretFields.includes('api_key'),
  );
  const showSharedApiKeyConfig =
    authMode === 'api_key' ||
    (authMode === 'login_exchange' &&
      (value.send_api_key_to_login === true ||
        value.send_api_key_to_requests === true ||
        hasConfiguredApiKey));

  return (
    <section className="http-api-panel">
      {section === 'connection' && (
        <div className="http-api-panel-body">
          <div className="form-group">
            <label htmlFor="http-api-base-url">Base URL</label>
            <input
              id="http-api-base-url"
              type="url"
              value={value.base_url ?? ''}
              onChange={(event) => updateValue('base_url', event.target.value)}
              onBlur={(event) => handleUrlBlur('base_url', event.target.value)}
              placeholder="https://api.example.com"
              autoComplete="off"
              aria-invalid={baseUrlError ? 'true' : 'false'}
              aria-describedby={baseUrlError ? 'http-api-base-url-error' : undefined}
            />
            {baseUrlError && (
              <p id="http-api-base-url-error" className="field-error">
                {baseUrlError}
              </p>
            )}
            <p className="field-help">
              Required. Requests will use relative paths under this origin only.
            </p>
          </div>
        </div>
      )}

      {section === 'authentication' && (
        <div className="http-api-panel-body">
          <div className="form-group">
            <label htmlFor="http-api-auth-mode">Authentication mode</label>
            <select
              id="http-api-auth-mode"
              value={authMode}
              onChange={(event) => handleAuthModeChange(event.target.value as HttpApiAuthMode)}
            >
              {[
                ...MODERN_AUTH_MODE_OPTIONS,
                ...(legacyOptionsUnlocked ? LEGACY_AUTH_MODE_OPTIONS : []),
              ].map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </div>

          {authMode === 'headers' &&
            renderHeaderRows({
              groupLabel: 'Configured headers',
              itemLabel: 'Configured header',
              addAriaLabel: 'Add configured header',
              valueKey: 'request_headers',
              namePrefix: 'configured-header',
            })}

          {showSharedApiKeyConfig && (
            <div className="http-api-grid">
              <div className="form-group">
                <label htmlFor="http-api-api-key-location">API key location</label>
                <select
                  id="http-api-api-key-location"
                  value={value.api_key_location ?? 'header'}
                  onChange={(event) =>
                    updateValue('api_key_location', event.target.value as 'header' | 'query')
                  }
                >
                  <option value="header">header</option>
                  <option value="query">query</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="http-api-api-key-name">API key name</label>
                <input
                  id="http-api-api-key-name"
                  type="text"
                  value={value.api_key_name ?? ''}
                  onChange={(event) => updateValue('api_key_name', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-api-key-prefix">API key prefix</label>
                <input
                  id="http-api-api-key-prefix"
                  type="text"
                  value={value.api_key_prefix ?? ''}
                  onChange={(event) => updateValue('api_key_prefix', event.target.value)}
                  placeholder="Optional"
                  autoComplete="off"
                />
              </div>
              {renderSecretField({
                label: 'API key',
                inputId: 'http-api-api-key',
                field: 'api_key',
                value: value.api_key,
              })}
            </div>
          )}

          {authMode === 'basic' && (
            <div className="http-api-grid">
              <div className="form-group">
                <label htmlFor="http-api-basic-username">Basic username</label>
                <input
                  id="http-api-basic-username"
                  type="text"
                  value={value.basic_username ?? ''}
                  onChange={(event) => updateValue('basic_username', event.target.value)}
                  autoComplete="off"
                />
              </div>
              {renderSecretField({
                label: 'Basic password',
                inputId: 'http-api-basic-password',
                field: 'basic_password',
                value: value.basic_password,
              })}
            </div>
          )}

          {authMode === 'bearer' && (
            <div className="http-api-grid">
              {renderSecretField({
                label: 'Bearer token',
                inputId: 'http-api-bearer-token',
                field: 'bearer_token',
                value: value.bearer_token,
              })}
              <div className="form-group">
                <label htmlFor="http-api-token-header-name">Token header name</label>
                <input
                  id="http-api-token-header-name"
                  type="text"
                  value={value.token_header_name ?? ''}
                  onChange={(event) => updateValue('token_header_name', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-token-prefix">Token prefix</label>
                <input
                  id="http-api-token-prefix"
                  type="text"
                  value={value.token_prefix ?? ''}
                  onChange={(event) => updateValue('token_prefix', event.target.value)}
                  placeholder="Optional"
                  autoComplete="off"
                />
              </div>
            </div>
          )}

          {authMode === 'login_exchange' && (
            <div className="http-api-grid">
              <div className="form-group">
                <label htmlFor="http-api-login-path">Login path</label>
                <input
                  id="http-api-login-path"
                  type="text"
                  value={value.login_path ?? ''}
                  onChange={(event) => updateValue('login_path', event.target.value)}
                  placeholder="/session"
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-login-method">Login method</label>
                <select
                  id="http-api-login-method"
                  value={value.login_method ?? 'POST'}
                  onChange={(event) =>
                    updateValue('login_method', event.target.value as HttpApiHttpMethod)
                  }
                >
                  <option value="POST">POST</option>
                  <option value="PUT">PUT</option>
                  <option value="PATCH">PATCH</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="http-api-login-body-format">Login body format</label>
                <select
                  id="http-api-login-body-format"
                  value={value.login_body_format ?? 'json'}
                  onChange={(event) =>
                    updateValue('login_body_format', event.target.value as 'json' | 'form')
                  }
                >
                  <option value="json">json</option>
                  <option value="form">form</option>
                </select>
              </div>
              <div className="form-group">
                <label htmlFor="http-api-login-username">Login username</label>
                <input
                  id="http-api-login-username"
                  type="text"
                  value={value.login_username ?? ''}
                  onChange={(event) => updateValue('login_username', event.target.value)}
                  autoComplete="off"
                />
              </div>
              {renderSecretField({
                label: 'Login password',
                inputId: 'http-api-login-password',
                field: 'login_password',
                value: value.login_password,
              })}
              <div className="form-group">
                <label htmlFor="http-api-login-username-field">Username field</label>
                <input
                  id="http-api-login-username-field"
                  type="text"
                  value={value.login_username_field ?? DEFAULT_LOGIN_USERNAME_FIELD}
                  onChange={(event) => updateValue('login_username_field', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-login-password-field">Password field</label>
                <input
                  id="http-api-login-password-field"
                  type="text"
                  value={value.login_password_field ?? DEFAULT_LOGIN_PASSWORD_FIELD}
                  onChange={(event) => updateValue('login_password_field', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-token-response-path">Token response path</label>
                <input
                  id="http-api-token-response-path"
                  type="text"
                  value={value.token_response_path ?? DEFAULT_TOKEN_RESPONSE_PATH}
                  onChange={(event) => updateValue('token_response_path', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-token-expires-in-path">Token expires-in path</label>
                <input
                  id="http-api-token-expires-in-path"
                  type="text"
                  value={value.token_expires_in_path ?? ''}
                  onChange={(event) => updateValue('token_expires_in_path', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-token-header-name">Token header name</label>
                <input
                  id="http-api-token-header-name"
                  type="text"
                  value={value.token_header_name ?? DEFAULT_TOKEN_HEADER_NAME}
                  onChange={(event) => updateValue('token_header_name', event.target.value)}
                  autoComplete="off"
                />
              </div>
              <div className="form-group">
                <label htmlFor="http-api-token-prefix">Token prefix</label>
                <input
                  id="http-api-token-prefix"
                  type="text"
                  value={value.token_prefix ?? DEFAULT_TOKEN_PREFIX}
                  onChange={(event) => updateValue('token_prefix', event.target.value)}
                  placeholder="Optional"
                  autoComplete="off"
                />
              </div>
              <label className="http-api-checkbox-row" htmlFor="http-api-send-api-key-to-login">
                <input
                  id="http-api-send-api-key-to-login"
                  type="checkbox"
                  checked={Boolean(value.send_api_key_to_login)}
                  onChange={(event) => updateValue('send_api_key_to_login', event.target.checked)}
                />
                Send API key to login exchange
              </label>
              <label className="http-api-checkbox-row" htmlFor="http-api-send-api-key-to-requests">
                <input
                  id="http-api-send-api-key-to-requests"
                  type="checkbox"
                  checked={Boolean(value.send_api_key_to_requests)}
                  onChange={(event) =>
                    updateValue('send_api_key_to_requests', event.target.checked)
                  }
                />
                Send API key to authenticated requests
              </label>
            </div>
          )}

          {authMode === 'token_exchange' && (
            <>
              <div className="http-api-subsection http-api-token-login-row">
                <div className="http-api-subsection-label">Token exchange</div>
                <div className="http-api-grid">
                  <div className="form-group">
                    <label htmlFor="http-api-login-path">Login path</label>
                    <input
                      id="http-api-login-path"
                      type="text"
                      value={value.login_path ?? ''}
                      onChange={(event) => updateValue('login_path', event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="http-api-login-method">Login method</label>
                    <select
                      id="http-api-login-method"
                      value={value.login_method ?? 'POST'}
                      onChange={(event) =>
                        updateValue('login_method', event.target.value as HttpApiHttpMethod)
                      }
                    >
                      <option value="POST">POST</option>
                      <option value="PUT">PUT</option>
                      <option value="PATCH">PATCH</option>
                    </select>
                  </div>
                  <div className="form-group">
                    <label htmlFor="http-api-login-body-format">Login body format</label>
                    <select
                      id="http-api-login-body-format"
                      value={value.login_body_format ?? 'json'}
                      onChange={(event) =>
                        updateValue('login_body_format', event.target.value as 'json' | 'form')
                      }
                    >
                      <option value="json">json</option>
                      <option value="form">form</option>
                    </select>
                  </div>
                </div>
              </div>

              {renderTokenFieldRows()}
              {renderHeaderRows({
                groupLabel: 'Token request headers',
                itemLabel: 'Token request header',
                addAriaLabel: 'Add token request header',
                valueKey: 'token_request_headers',
                namePrefix: 'token-request-header',
              })}

              <div className="http-api-subsection">
                <p className="http-api-subsection-label">Token response</p>
                <div className="http-api-grid">
                  <div className="form-group">
                    <label htmlFor="http-api-token-response-path">Token response path</label>
                    <input
                      id="http-api-token-response-path"
                      type="text"
                      value={value.token_response_path ?? DEFAULT_TOKEN_RESPONSE_PATH}
                      onChange={(event) => updateValue('token_response_path', event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="http-api-token-expires-in-path">Token expires-in path</label>
                    <input
                      id="http-api-token-expires-in-path"
                      type="text"
                      value={value.token_expires_in_path ?? ''}
                      onChange={(event) => updateValue('token_expires_in_path', event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="http-api-token-header-name">Token header name</label>
                    <input
                      id="http-api-token-header-name"
                      type="text"
                      value={value.token_header_name ?? DEFAULT_TOKEN_HEADER_NAME}
                      onChange={(event) => updateValue('token_header_name', event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                  <div className="form-group">
                    <label htmlFor="http-api-token-prefix">Token prefix</label>
                    <input
                      id="http-api-token-prefix"
                      type="text"
                      value={value.token_prefix ?? DEFAULT_TOKEN_PREFIX}
                      onChange={(event) => updateValue('token_prefix', event.target.value)}
                      autoComplete="off"
                    />
                  </div>
                </div>
              </div>

              {renderHeaderRows({
                groupLabel: 'Configured headers',
                itemLabel: 'Configured header',
                addAriaLabel: 'Add configured header',
                valueKey: 'request_headers',
                namePrefix: 'configured-header',
              })}
            </>
          )}

          <div className="http-api-actions-row">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={onTestConnection}
              disabled={!onTestConnection || testDisabled}
            >
              Test connection
            </button>
          </div>
          {renderActionStatus(testStatus)}
        </div>
      )}

      {section === 'api_details' && (
        <div className="http-api-panel-body">
          <div className="http-api-grid">
            <div className="form-group">
              <label htmlFor="http-api-openapi-url">OpenAPI URL</label>
              <input
                id="http-api-openapi-url"
                type="url"
                value={value.openapi_source_url ?? ''}
                onChange={(event) => updateValue('openapi_source_url', event.target.value)}
                onBlur={(event) => handleUrlBlur('openapi_source_url', event.target.value)}
                placeholder="https://api.example.com/openapi.json"
                autoComplete="off"
                aria-invalid={openApiUrlError ? 'true' : 'false'}
                aria-describedby={openApiUrlError ? 'http-api-openapi-url-error' : undefined}
              />
              {openApiUrlError && (
                <p id="http-api-openapi-url-error" className="field-error">
                  {openApiUrlError}
                </p>
              )}
            </div>
            <div className="form-group">
              <label htmlFor="http-api-openapi-file">OpenAPI file</label>
              <input
                id="http-api-openapi-file"
                type="file"
                accept=".json,.yaml,.yml,application/json,application/yaml,text/yaml,text/x-yaml"
                onChange={handleOpenApiFileChange}
              />
              {uploadedOpenApiDocumentName && (
                <p className="field-help">Selected file: {uploadedOpenApiDocumentName}</p>
              )}
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="http-api-approved-request-headers">Approved request headers</label>
            <textarea
              id="http-api-approved-request-headers"
              value={(value.approved_request_headers ?? []).join('\n')}
              onChange={(event) =>
                updateValue('approved_request_headers', toDelimitedList(event.target.value))
              }
              rows={4}
              placeholder={'X-Trace-Id\nX-Request-Id'}
            />
            <p className="field-help">
              Agent-settable header names only. This does not configure fixed header values.
            </p>
          </div>

          <div className="form-group">
            <label htmlFor="http-api-default-response-selector">Default response selector</label>
            <input
              id="http-api-default-response-selector"
              type="text"
              value={value.default_response_selector ?? ''}
              onChange={(event) => updateValue('default_response_selector', event.target.value)}
              placeholder="items"
              autoComplete="off"
            />
          </div>

          <div className="http-api-method-policy-grid">
            {HTTP_API_METHODS.map((method) => (
              <div key={method} className="form-group">
                <label htmlFor={`http-api-policy-${method}`}>{method} method policy</label>
                <select
                  id={`http-api-policy-${method}`}
                  aria-label={`${method} method policy`}
                  value={getMethodPolicy(value, method)}
                  onChange={(event) =>
                    updateValue('method_policies', {
                      ...value.method_policies,
                      [method]: event.target.value as HttpApiMethodPolicy,
                    })
                  }
                >
                  {METHOD_POLICY_OPTIONS.map((policy) => (
                    <option key={policy} value={policy}>
                      {policy}
                    </option>
                  ))}
                </select>
              </div>
            ))}
          </div>

          <div className="http-api-actions-row">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={handleNormalizeOpenApi}
              disabled={!onNormalizeOpenApi || normalizeDisabled}
            >
              Normalize OpenAPI
            </button>
          </div>
          {renderActionStatus(openApiNormalizeStatus)}
        </div>
      )}
    </section>
  );
}
