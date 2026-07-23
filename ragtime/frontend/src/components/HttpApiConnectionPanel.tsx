import { useEffect, useRef, useState } from 'react';
import { Eye, EyeOff, Trash2 } from 'lucide-react';

import {
  DEFAULT_HTTP_API_METHOD_POLICIES,
  HTTP_API_METHODS,
  type HttpApiAuthMode,
  type HttpApiBodyFormat,
  type HttpApiConfiguredHeader,
  type HttpApiConnectionConfig,
  type HttpApiFixedSecretField,
  type HttpApiHttpMethod,
  type HttpApiMethodPolicy,
  type HttpApiSecretField,
  type HttpApiTokenField,
} from '@/types';

import {
  hasOAuthProviderConfigChanged,
  HttpApiOAuthConnectPanel,
} from './HttpApiOAuthConnectPanel';

type HttpApiPanelSection = 'connection' | 'authentication' | 'api_details';

interface ActionStatus {
  state: 'idle' | 'pending' | 'success' | 'error';
  message?: string | null;
}

interface HttpApiConnectionPanelProps {
  section: HttpApiPanelSection;
  value: HttpApiConnectionConfig;
  onChange: (value: HttpApiConnectionConfig) => void;
  configuredSecretFields?: HttpApiSecretField[];
  testStatus?: ActionStatus;
  onTestConnection?: () => void;
  testDisabled?: boolean;
  toolId?: string;
  onOAuthConnected?: (sessionId: string) => void;
}

type UrlFieldKey = 'base_url' | 'documentation_url';
type HeaderRowKey = 'request_headers' | 'token_request_headers';
type BodyRowKey = 'token_request_fields' | 'request_body_fields';

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
  { value: 'token_exchange', label: 'OAuth 2.0 / Token exchange' },
  { value: 'oauth2', label: 'OAuth 2.0 / Interactive' },
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

  if (currentMode === 'oauth2' && nextAuthMode !== 'oauth2') {
    nextValue = {
      ...nextValue,
      oauth_client_secret: '',
      oauth_access_token: '',
      oauth_refresh_token: '',
    };
    nextValue = removeConfigKeys(nextValue, [
      'oauth_flow',
      'oauth_issuer_url',
      'oauth_authorization_url',
      'oauth_device_authorization_url',
      'oauth_token_url',
      'oauth_client_id',
      'oauth_client_auth_method',
      'oauth_scopes',
      'oauth_token_type',
      'oauth_token_expires_at',
      'oauth_session_id',
    ]);
  }

  if (
    currentMode === 'headers' &&
    nextAuthMode !== 'headers' &&
    nextAuthMode !== 'token_exchange'
  ) {
    nextValue = { ...nextValue, request_headers: [], request_body_fields: [] };
    nextValue = removeConfigKeys(nextValue, ['request_body_format']);
  }

  if (currentMode === 'token_exchange' && nextAuthMode !== 'token_exchange') {
    const keysToClear: Array<keyof HttpApiConnectionConfig> = [
      'login_path',
      'token_url',
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
      request_body_fields: nextAuthMode === 'headers' ? currentValue.request_body_fields : [],
      token_request_headers: [],
      ...(nextAuthMode !== 'headers' ? { request_headers: [] } : {}),
    };
    if (nextAuthMode !== 'headers') {
      nextValue = { ...nextValue, request_headers: [] };
      nextValue = removeConfigKeys(nextValue, ['request_body_format']);
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
  toolId,
  onOAuthConnected,
}: HttpApiConnectionPanelProps) {
  const [urlErrors, setUrlErrors] = useState<Partial<Record<UrlFieldKey, string>>>({});
  const [legacyOptionsUnlocked, setLegacyOptionsUnlocked] = useState(() =>
    LEGACY_AUTH_MODES.includes(value.auth_mode ?? 'none'),
  );
  const [revealedFixedSecrets, setRevealedFixedSecrets] = useState<
    Partial<Record<HttpApiFixedSecretField, boolean>>
  >({});
  const [revealedSecretRows, setRevealedSecretRows] = useState<Record<string, boolean>>({});
  const secretRowIds = useRef(new WeakMap<HttpApiConfiguredHeader | HttpApiTokenField, string>());
  const nextSecretRowId = useRef(0);
  const authMode = value.auth_mode ?? 'none';

  const getSecretRowId = (
    prefix: string,
    row: HttpApiConfiguredHeader | HttpApiTokenField,
  ): string => {
    const existingId = secretRowIds.current.get(row);
    if (existingId) {
      return existingId;
    }
    const id = `${prefix}-${nextSecretRowId.current++}`;
    secretRowIds.current.set(row, id);
    return id;
  };

  const preserveSecretRowId = <T extends HttpApiConfiguredHeader | HttpApiTokenField>(
    previousRow: T,
    nextRow: T,
  ): T => {
    const rowId = secretRowIds.current.get(previousRow);
    if (rowId) {
      secretRowIds.current.set(nextRow, rowId);
    }
    return nextRow;
  };

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

  const handleAuthModeChange = (nextAuthMode: HttpApiAuthMode) => {
    if (LEGACY_AUTH_MODES.includes(nextAuthMode)) {
      setLegacyOptionsUnlocked(true);
    }
    onChange(getClearedConfigForAuthModeChange(value, nextAuthMode));
  };

  const handleOAuthChange = (nextValue: HttpApiConnectionConfig) => {
    if (hasOAuthProviderConfigChanged(value, nextValue)) {
      const invalidated = { ...nextValue };
      delete invalidated.oauth_session_id;
      onChange(invalidated);
      return;
    }
    onChange(nextValue);
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

  const hasSavedBodyFieldSecret = (key: BodyRowKey, name: string): boolean => {
    if (!name.trim()) return false;
    return configuredSecretFields.includes(`${key}.${name.trim()}` as HttpApiSecretField);
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

  const buildBodyFieldValidation = (
    rows: HttpApiTokenField[] | undefined,
    key: BodyRowKey,
  ): RowValidationState[] => {
    const counts = new Map<string, number>();
    (rows ?? []).forEach((row) => {
      const trimmedName = row.name.trim();
      if (!trimmedName) {
        return;
      }
      counts.set(trimmedName, (counts.get(trimmedName) ?? 0) + 1);
    });

    return (rows ?? []).map((row) => {
      const trimmedName = row.name.trim();
      const hasSavedValue =
        key === 'token_request_fields'
          ? hasSavedTokenFieldSecret(trimmedName)
          : hasSavedBodyFieldSecret(key, trimmedName);
      const errors: RowValidationState = {};

      if (!trimmedName) {
        errors.nameError = `${key === 'token_request_fields' ? 'Token request' : 'Request body'} field name is required.`;
      } else if ((counts.get(trimmedName) ?? 0) > 1) {
        errors.nameError = `Duplicate ${key === 'token_request_fields' ? 'token request' : 'request body'} field name.`;
      }

      if (!row.value && !hasSavedValue) {
        errors.valueError = 'Field value is required.';
      }

      return errors;
    });
  };

  const tokenFieldValidation = buildBodyFieldValidation(
    value.token_request_fields,
    'token_request_fields',
  );
  const bodyFieldValidation = buildBodyFieldValidation(
    value.request_body_fields,
    'request_body_fields',
  );

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
    const isRevealed = Boolean(revealedFixedSecrets[field]);

    return (
      <div className="form-group">
        <label htmlFor={inputId}>{label}</label>
        <div className="http-api-secret-input-wrap">
          <input
            id={inputId}
            type={isRevealed ? 'text' : 'password'}
            value={fieldValue ?? ''}
            onChange={(event) => updateSecret(field, event.target.value)}
            placeholder={`Enter ${label.toLowerCase()}`}
            autoComplete="off"
          />
          <button
            type="button"
            className="settings-inline-copy settings-inline-copy-secondary http-api-secret-toggle-button"
            onClick={() =>
              setRevealedFixedSecrets((current) => ({ ...current, [field]: !isRevealed }))
            }
            title={`${isRevealed ? 'Hide' : 'Show'} ${label}`}
            aria-label={`${isRevealed ? 'Hide' : 'Show'} ${label}`}
          >
            {isRevealed ? (
              <EyeOff size={14} aria-hidden="true" />
            ) : (
              <Eye size={14} aria-hidden="true" />
            )}
          </button>
        </div>
      </div>
    );
  };

  const renderActionStatus = (status: ActionStatus | undefined) => {
    if (!status || !status.message) {
      return null;
    }

    const className = ['http-api-status', status.state !== 'idle' ? `is-${status.state}` : '']
      .filter(Boolean)
      .join(' ');

    return (
      <div className={className} role="status" aria-live="polite">
        {status.message && <span>{status.message}</span>}
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
    const hasRows = rows.length > 0;

    return (
      <div
        className={`http-api-subsection${hasRows ? '' : ' http-api-optional-subsection-empty http-api-optional-action'}`}
      >
        {hasRows && (
          <div className="http-api-subsection-header">
            <p className="http-api-subsection-label">{groupLabel}</p>
          </div>
        )}
        {hasRows && (
          <div className="http-api-row-list">
            {rows.map((row, index) => {
              const position = index + 1;
              const rowValidation = validation[index] ?? {};
              const rowId = getSecretRowId(namePrefix, row);
              const isRevealed = Boolean(revealedSecretRows[rowId]);

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
                    <div className="http-api-secret-input-wrap">
                      <input
                        id={`http-api-${namePrefix}-value-${position}`}
                        aria-label={`${itemLabel} value ${position}`}
                        aria-invalid={rowValidation.valueError ? 'true' : 'false'}
                        type={isRevealed ? 'text' : 'password'}
                        value={row.value}
                        onChange={(event) => {
                          const nextRows = rows.map((currentRow, currentIndex) =>
                            currentIndex === index
                              ? preserveSecretRowId(row, {
                                  ...currentRow,
                                  value: event.target.value,
                                })
                              : currentRow,
                          );
                          onChange({ ...value, [valueKey]: nextRows });
                        }}
                        autoComplete="off"
                      />
                      <button
                        type="button"
                        className="settings-inline-copy settings-inline-copy-secondary http-api-secret-toggle-button"
                        onClick={() =>
                          setRevealedSecretRows((current) => ({ ...current, [rowId]: !isRevealed }))
                        }
                        title={`${isRevealed ? 'Hide' : 'Show'} ${itemLabel.toLowerCase()} value ${position}`}
                        aria-label={`${isRevealed ? 'Hide' : 'Show'} ${itemLabel.toLowerCase()} value ${position}`}
                      >
                        {isRevealed ? (
                          <EyeOff size={14} aria-hidden="true" />
                        ) : (
                          <Eye size={14} aria-hidden="true" />
                        )}
                      </button>
                    </div>
                    {rowValidation.valueError && (
                      <p className="field-error">{rowValidation.valueError}</p>
                    )}
                  </div>
                  <div className="http-api-row-actions http-api-row-actions-input-aligned">
                    <button
                      type="button"
                      className="btn btn-danger btn-icon http-api-row-remove"
                      aria-label={`Remove ${itemLabel.toLowerCase()} ${position}`}
                      onClick={() =>
                        onChange({
                          ...value,
                          [valueKey]: rows.filter((_, currentIndex) => currentIndex !== index),
                        })
                      }
                    >
                      <Trash2 size={14} aria-hidden="true" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        <div className="http-api-add-row">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            aria-label={addAriaLabel}
            onClick={() => onChange({ ...value, [valueKey]: [...rows, { name: '', value: '' }] })}
          >
            + Header
          </button>
        </div>
      </div>
    );
  };

  const renderBodyRows = ({
    itemLabel,
    addAriaLabel,
    valueKey,
    format,
    formatLabel,
    formatOptions,
    onFormatChange,
    namePrefix,
  }: {
    itemLabel: string;
    addAriaLabel: string;
    valueKey: BodyRowKey;
    format: HttpApiBodyFormat;
    formatLabel: string;
    formatOptions: HttpApiBodyFormat[];
    onFormatChange: (format: HttpApiBodyFormat) => void;
    namePrefix: string;
  }) => {
    const rows = value[valueKey] ?? [];
    const tokenRows = valueKey === 'token_request_fields';
    const validation: RowValidationState[] = tokenRows ? tokenFieldValidation : bodyFieldValidation;
    const hasRows = rows.length > 0;
    return (
      <div
        className={`http-api-subsection${hasRows ? '' : ' http-api-optional-subsection-empty http-api-optional-action'}`}
      >
        {hasRows && (
          <div className="http-api-body-toolbar">
            <div className="form-group">
              <label htmlFor={`http-api-${namePrefix}-format`}>{formatLabel}</label>
              <select
                id={`http-api-${namePrefix}-format`}
                value={format}
                onChange={(event) => onFormatChange(event.target.value as HttpApiBodyFormat)}
              >
                {formatOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
        {hasRows && (
          <div className="http-api-row-list">
            {rows.map((row, index) => {
              const position = index + 1;
              const rowValidation = validation[index] ?? {};
              const rowId = getSecretRowId(namePrefix, row);
              const isRevealed = Boolean(revealedSecretRows[rowId]);
              const normalizedRows = () =>
                rows.map((currentRow) =>
                  currentRow.secret
                    ? currentRow
                    : preserveSecretRowId(currentRow, { ...currentRow, secret: true }),
                );
              return (
                <div key={`${namePrefix}-${position}`} className="http-api-field-row">
                  <div className="form-group">
                    <label htmlFor={`http-api-${namePrefix}-name-${position}`}>Field name</label>
                    <input
                      id={`http-api-${namePrefix}-name-${position}`}
                      aria-label={`${itemLabel} name ${position}`}
                      aria-invalid={rowValidation.nameError ? 'true' : 'false'}
                      type="text"
                      value={row.name}
                      onChange={(event) =>
                        onChange({
                          ...value,
                          [valueKey]: normalizedRows().map((currentRow, currentIndex) =>
                            currentIndex === index
                              ? { ...currentRow, name: event.target.value, secret: true }
                              : currentRow,
                          ),
                        })
                      }
                      autoComplete="off"
                    />
                    {rowValidation.nameError && (
                      <p className="field-error">{rowValidation.nameError}</p>
                    )}
                  </div>
                  <div className="form-group">
                    <label htmlFor={`http-api-${namePrefix}-value-${position}`}>Field value</label>
                    <div className="http-api-secret-input-wrap">
                      <input
                        id={`http-api-${namePrefix}-value-${position}`}
                        aria-label={`${itemLabel} value ${position}`}
                        aria-invalid={rowValidation.valueError ? 'true' : 'false'}
                        type={isRevealed ? 'text' : 'password'}
                        value={row.value}
                        onChange={(event) =>
                          onChange({
                            ...value,
                            [valueKey]: normalizedRows().map((currentRow, currentIndex) =>
                              currentIndex === index
                                ? preserveSecretRowId(currentRow, {
                                    ...currentRow,
                                    value: event.target.value,
                                  })
                                : currentRow,
                            ),
                          })
                        }
                        autoComplete="off"
                      />
                      <button
                        type="button"
                        className="settings-inline-copy settings-inline-copy-secondary http-api-secret-toggle-button"
                        onClick={() =>
                          setRevealedSecretRows((current) => ({ ...current, [rowId]: !isRevealed }))
                        }
                        title={`${isRevealed ? 'Hide' : 'Show'} ${itemLabel.toLowerCase()} value ${position}`}
                        aria-label={`${isRevealed ? 'Hide' : 'Show'} ${itemLabel.toLowerCase()} value ${position}`}
                      >
                        {isRevealed ? (
                          <EyeOff size={14} aria-hidden="true" />
                        ) : (
                          <Eye size={14} aria-hidden="true" />
                        )}
                      </button>
                    </div>
                    {rowValidation.valueError && (
                      <p className="field-error">{rowValidation.valueError}</p>
                    )}
                  </div>
                  <div className="http-api-row-actions http-api-row-actions-input-aligned">
                    <button
                      type="button"
                      className="btn btn-danger btn-icon http-api-row-remove"
                      aria-label={`Remove ${itemLabel.toLowerCase()} ${position}`}
                      onClick={() =>
                        onChange({
                          ...value,
                          [valueKey]: normalizedRows().filter(
                            (_, currentIndex) => currentIndex !== index,
                          ),
                        })
                      }
                    >
                      <Trash2 size={14} aria-hidden="true" />
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
        <div className="http-api-add-row">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            aria-label={addAriaLabel}
            onClick={() =>
              onChange({ ...value, [valueKey]: [...rows, { name: '', value: '', secret: true }] })
            }
          >
            + Body
          </button>
        </div>
      </div>
    );
  };

  const baseUrlError = urlErrors.base_url;
  const documentationUrlError = urlErrors.documentation_url;
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
              Required. Requests use relative paths under this origin; this is separate from the
              optional documentation URL in API Details.
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

          {authMode === 'headers' &&
            renderBodyRows({
              itemLabel: 'Request body field',
              addAriaLabel: 'Add request body field',
              valueKey: 'request_body_fields',
              format: value.request_body_format ?? 'json',
              formatLabel: 'Request body content type',
              formatOptions: ['json', 'form', 'multipart'],
              onFormatChange: (format) => updateValue('request_body_format', format),
              namePrefix: 'request-body',
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
                <div className="http-api-grid">
                  <div className="form-group">
                    <label htmlFor="http-api-token-url">Token endpoint</label>
                    <input
                      id="http-api-token-url"
                      type="text"
                      value={value.token_url || value.login_path || ''}
                      onChange={(event) =>
                        onChange({
                          ...value,
                          token_url: event.target.value,
                          ...(!value.token_url ? { login_path: '' } : {}),
                        })
                      }
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
                </div>
              </div>

              {renderHeaderRows({
                groupLabel: 'Token request headers',
                itemLabel: 'Token request header',
                addAriaLabel: 'Add token request header',
                valueKey: 'token_request_headers',
                namePrefix: 'token-request-header',
              })}
              {renderBodyRows({
                itemLabel: 'Token request field',
                addAriaLabel: 'Add token request field',
                valueKey: 'token_request_fields',
                format: value.login_body_format ?? 'json',
                formatLabel: 'Token request body content type',
                formatOptions: ['json', 'form'],
                onFormatChange: (format) =>
                  updateValue('login_body_format', format as 'json' | 'form'),
                namePrefix: 'token-request-field',
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
              {renderBodyRows({
                itemLabel: 'Request body field',
                addAriaLabel: 'Add request body field',
                valueKey: 'request_body_fields',
                format: value.request_body_format ?? 'json',
                formatLabel: 'Request body content type',
                formatOptions: ['json', 'form', 'multipart'],
                onFormatChange: (format) => updateValue('request_body_format', format),
                namePrefix: 'request-body',
              })}
            </>
          )}

          {authMode === 'oauth2' && (
            <HttpApiOAuthConnectPanel
              value={value}
              toolId={toolId}
              configuredSecretFields={configuredSecretFields}
              onChange={handleOAuthChange}
              onConnected={(sessionId) => onOAuthConnected?.(sessionId)}
            />
          )}

          {(authMode !== 'oauth2' ||
            configuredSecretFields.includes('oauth_access_token') ||
            configuredSecretFields.includes('oauth_refresh_token')) && (
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
          )}
          {renderActionStatus(testStatus)}
        </div>
      )}

      {section === 'api_details' && (
        <div className="http-api-panel-body">
          <div className="form-group">
            <label htmlFor="http-api-documentation-url">API documentation URL (optional)</label>
            <input
              id="http-api-documentation-url"
              type="url"
              value={value.documentation_url ?? ''}
              onChange={(event) => updateValue('documentation_url', event.target.value)}
              onBlur={(event) => handleUrlBlur('documentation_url', event.target.value)}
              placeholder="https://api.example.com/docs"
              autoComplete="off"
              aria-invalid={documentationUrlError ? 'true' : 'false'}
              aria-describedby={
                documentationUrlError ? 'http-api-documentation-url-error' : undefined
              }
            />
            {documentationUrlError && (
              <p id="http-api-documentation-url-error" className="field-error">
                {documentationUrlError}
              </p>
            )}
            <p className="field-help">
              Optional reference for endpoint paths, parameters, and response details. It is not
              used as the request Base URL.
            </p>
          </div>

          <div className="form-group">
            <label htmlFor="http-api-approved-request-headers">
              Approved request headers (optional)
            </label>
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
              Optional agent-settable header names only. This does not configure fixed header values
              or expose secrets.
            </p>
          </div>

          <div className="form-group">
            <label htmlFor="http-api-default-response-selector">
              Default response selector (optional)
            </label>
            <input
              id="http-api-default-response-selector"
              type="text"
              value={value.default_response_selector ?? ''}
              onChange={(event) => updateValue('default_response_selector', event.target.value)}
              placeholder="items"
              autoComplete="off"
            />
            <p className="field-help">
              Optional dot-path used to select the useful collection from a response.
            </p>
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
        </div>
      )}
    </section>
  );
}
