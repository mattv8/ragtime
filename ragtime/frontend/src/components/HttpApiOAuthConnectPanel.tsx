import { useCallback, useEffect, useRef, useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';

import { api } from '@/api/client';
import type {
  HttpApiConnectionConfig,
  HttpApiOAuthClientAuthMethod,
  HttpApiOAuthDiscoveryResponse,
  HttpApiOAuthFlow,
  HttpApiOAuthPollResponse,
  HttpApiOAuthStartResponse,
} from '@/types';

import { ToastContainer, useToast } from './shared/Toast';
import { InlineCopyButton } from './shared/InlineCopyButton';

export interface HttpApiOAuthConnectPanelProps {
  value: HttpApiConnectionConfig;
  toolId?: string;
  configuredSecretFields: string[];
  onChange: (next: HttpApiConnectionConfig) => void;
  onConnected: (sessionId: string) => void;
}

const OAUTH_PROVIDER_CONFIG_KEYS: Array<keyof HttpApiConnectionConfig> = [
  'oauth_flow',
  'oauth_issuer_url',
  'oauth_authorization_url',
  'oauth_device_authorization_url',
  'oauth_token_url',
  'oauth_client_id',
  'oauth_client_secret',
  'oauth_client_auth_method',
  'oauth_scopes',
];

export function hasOAuthProviderConfigChanged(
  previous: HttpApiConnectionConfig,
  next: HttpApiConnectionConfig,
): boolean {
  return OAUTH_PROVIDER_CONFIG_KEYS.some((key) =>
    key === 'oauth_scopes'
      ? JSON.stringify(previous[key] ?? []) !== JSON.stringify(next[key] ?? [])
      : previous[key] !== next[key],
  );
}

type PanelStatus =
  | 'idle'
  | 'discovering'
  | 'connecting'
  | 'pending'
  | 'connected'
  | 'expired'
  | 'error';

const GENERIC_ERROR = 'OAuth connection failed. Check the configuration and try again.';
const INCOMPLETE_RESPONSE_ERROR = 'OAuth provider returned an incomplete response. Try again.';
const BLOCKED_POPUP_ERROR = 'Your browser blocked the OAuth popup. Allow popups and try again.';

function safeResponseMessage(message: string | null | undefined, fallback: string): string {
  return typeof message === 'string' && message.trim() ? message.trim() : fallback;
}

function hasText(value: string | null | undefined): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

function supportedFlow(value: string): HttpApiOAuthFlow | null {
  if (value === 'device_code' || value === 'urn:ietf:params:oauth:grant-type:device_code') {
    return 'device_code';
  }
  if (value === 'authorization_code' || value === 'authorization_code_pkce') {
    return 'authorization_code_pkce';
  }
  return null;
}

function scopesToText(scopes: string[] | undefined): string {
  return (scopes ?? []).join(', ');
}

function pollDelay(response: HttpApiOAuthPollResponse): number {
  return Math.max(250, (response.retry_after_seconds ?? response.interval ?? 5) * 1000);
}

export function HttpApiOAuthConnectPanel({
  value,
  toolId,
  configuredSecretFields,
  onChange,
  onConnected,
}: HttpApiOAuthConnectPanelProps) {
  const [status, setStatus] = useState<PanelStatus>('idle');
  const [message, setMessage] = useState('');
  const [discovery, setDiscovery] = useState<HttpApiOAuthDiscoveryResponse | null>(null);
  const [deviceCode, setDeviceCode] = useState<string | null>(null);
  const [deviceUrl, setDeviceUrl] = useState<string | null>(null);
  const [clientSecretRevealed, setClientSecretRevealed] = useState(false);
  const timerRef = useRef<number | null>(null);
  const generationRef = useRef(0);
  const lastConfigRef = useRef(value);
  const [toasts, toast] = useToast();

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(
    () => () => {
      generationRef.current += 1;
      clearTimer();
    },
    [clearTimer],
  );

  useEffect(() => {
    const changed = hasOAuthProviderConfigChanged(lastConfigRef.current, value);
    if (changed && status !== 'connecting') {
      generationRef.current += 1;
      clearTimer();
      setStatus('idle');
      setMessage('');
      setDeviceCode(null);
      setDeviceUrl(null);
    }
    lastConfigRef.current = value;
  }, [clearTimer, status, value]);

  const update = useCallback(
    <K extends keyof HttpApiConnectionConfig>(key: K, next: HttpApiConnectionConfig[K]) => {
      onChange({ ...value, [key]: next });
    },
    [onChange, value],
  );

  const applyDiscovery = (result: HttpApiOAuthDiscoveryResponse) => {
    setDiscovery(result);
    const next: HttpApiConnectionConfig = {
      ...value,
      oauth_issuer_url: result.issuer,
      oauth_authorization_url: result.authorization_endpoint ?? value.oauth_authorization_url ?? '',
      oauth_device_authorization_url:
        result.device_authorization_endpoint ?? value.oauth_device_authorization_url ?? '',
      oauth_token_url: result.token_endpoint ?? value.oauth_token_url ?? '',
      oauth_scopes: result.scopes_supported.length ? result.scopes_supported : value.oauth_scopes,
    };
    const currentFlow = value.oauth_flow ?? 'device_code';
    if (
      currentFlow === 'device_code' &&
      !result.grant_types_supported.some((grant) => supportedFlow(grant) === 'device_code')
    ) {
      next.oauth_flow = 'authorization_code_pkce';
    }
    if (
      currentFlow === 'authorization_code_pkce' &&
      !result.grant_types_supported.some(
        (grant) => supportedFlow(grant) === 'authorization_code_pkce',
      )
    ) {
      next.oauth_flow = 'device_code';
    }
    onChange(next);
  };

  const discover = async () => {
    setStatus('discovering');
    setMessage('');
    try {
      applyDiscovery(await api.discoverHttpApiOAuth({ issuer_url: value.oauth_issuer_url ?? '' }));
      setStatus('idle');
    } catch {
      setStatus('error');
      setMessage('OAuth discovery failed. Check the issuer URL and try again.');
    }
  };

  const finish = (
    response: HttpApiOAuthStartResponse | HttpApiOAuthPollResponse,
    generation: number,
    popup?: Window | null,
    options?: { requireStartPayload?: boolean },
  ) => {
    if (generation !== generationRef.current) return;
    if (!hasText(response.session_id)) {
      popup?.close();
      clearTimer();
      setStatus('error');
      setMessage(INCOMPLETE_RESPONSE_ERROR);
      setDeviceCode(null);
      setDeviceUrl(null);
      toast.error(INCOMPLETE_RESPONSE_ERROR);
      return;
    }
    if (response.status === 'connected') {
      clearTimer();
      setStatus('connected');
      setMessage('Connected. Save the tool to keep this credential.');
      onConnected(response.session_id);
      return;
    }
    if (response.status === 'expired' || response.status === 'failed') {
      const errorMessage =
        response.status === 'expired'
          ? 'This OAuth session expired. Reconnect to try again.'
          : safeResponseMessage(response.message, GENERIC_ERROR);
      popup?.close();
      clearTimer();
      setStatus('error');
      setMessage(errorMessage);
      setDeviceCode(null);
      setDeviceUrl(null);
      toast.error(errorMessage);
      return;
    }
    if (
      (options?.requireStartPayload === true &&
        value.oauth_flow === 'authorization_code_pkce' &&
        !hasText(response.authorization_url)) ||
      (value.oauth_flow === 'device_code' &&
        !hasText(response.verification_uri) &&
        !hasText(response.verification_uri_complete) &&
        !hasText(response.user_code))
    ) {
      popup?.close();
      clearTimer();
      setStatus('error');
      setMessage(INCOMPLETE_RESPONSE_ERROR);
      setDeviceCode(null);
      setDeviceUrl(null);
      toast.error(INCOMPLETE_RESPONSE_ERROR);
      return;
    }
    setStatus('pending');
    setMessage('Waiting for authorization…');
    timerRef.current = window.setTimeout(
      () => void poll(response.session_id, generation),
      pollDelay(response),
    );
  };

  const poll = async (sessionId: string, generation = generationRef.current) => {
    if (generation !== generationRef.current) return;
    try {
      finish(await api.pollHttpApiOAuth({ session_id: sessionId }), generation);
    } catch {
      if (generation === generationRef.current) {
        clearTimer();
        setStatus('error');
        setMessage(GENERIC_ERROR);
        setDeviceCode(null);
        setDeviceUrl(null);
        toast.error(GENERIC_ERROR);
      }
    }
  };

  const connect = async () => {
    clearTimer();
    const generation = ++generationRef.current;
    const popup = value.oauth_flow === 'authorization_code_pkce' ? window.open('', '_blank') : null;
    if (popup) {
      popup.opener = null;
    }
    setStatus('connecting');
    setMessage('Starting OAuth connection…');
    setDeviceCode(null);
    setDeviceUrl(null);
    try {
      const response = await api.startHttpApiOAuth({
        connection_config: value,
        ...(toolId ? { tool_id: toolId } : {}),
      });
      if (response.authorization_url) {
        if (!popup) {
          setStatus('error');
          setMessage(BLOCKED_POPUP_ERROR);
          setDeviceCode(null);
          setDeviceUrl(null);
          toast.error(BLOCKED_POPUP_ERROR);
          return;
        }
        popup.location.href = response.authorization_url;
      } else if (response.verification_uri || response.verification_uri_complete) {
        const verificationUrl = response.verification_uri_complete ?? response.verification_uri;
        setDeviceCode(response.user_code ?? null);
        setDeviceUrl(verificationUrl ?? null);
        if (verificationUrl) window.open(verificationUrl, '_blank', 'noopener,noreferrer');
      }
      finish(response, generation, popup, { requireStartPayload: true });
    } catch {
      popup?.close();
      setStatus('error');
      setMessage(GENERIC_ERROR);
      setDeviceCode(null);
      setDeviceUrl(null);
      toast.error(GENERIC_ERROR);
    }
  };

  const deviceAvailable = discovery
    ? discovery.grant_types_supported.some((grant) => supportedFlow(grant) === 'device_code')
    : Boolean(value.oauth_device_authorization_url);
  const pkceAvailable = discovery
    ? discovery.grant_types_supported.some(
        (grant) => supportedFlow(grant) === 'authorization_code_pkce',
      ) &&
      (!discovery.code_challenge_methods_supported.length ||
        discovery.code_challenge_methods_supported.includes('S256'))
    : Boolean(value.oauth_authorization_url);
  const hasSavedCredential =
    configuredSecretFields.includes('oauth_access_token') ||
    configuredSecretFields.includes('oauth_refresh_token');

  return (
    <section className="http-api-oauth-panel" aria-label="OAuth 2.0 connection">
      <div className="http-api-oauth-discover-row">
        <div className="form-group">
          <label htmlFor="oauth-issuer-url">Issuer URL</label>
          <input
            id="oauth-issuer-url"
            className="form-input"
            type="url"
            value={value.oauth_issuer_url ?? ''}
            onChange={(event) => update('oauth_issuer_url', event.target.value)}
          />
        </div>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={() => void discover()}
          disabled={status === 'discovering'}
        >
          {status === 'discovering' ? 'Discovering…' : 'Discover'}
        </button>
      </div>

      <div className="http-api-oauth-grid">
        <div className="form-group">
          <label htmlFor="oauth-flow">Grant</label>
          <select
            id="oauth-flow"
            className="form-input"
            value={value.oauth_flow ?? 'device_code'}
            onChange={(event) => update('oauth_flow', event.target.value as HttpApiOAuthFlow)}
          >
            <option value="device_code" disabled={!deviceAvailable}>
              Device authorization
            </option>
            <option value="authorization_code_pkce" disabled={!pkceAvailable}>
              Authorization code + PKCE
            </option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="oauth-client-id">Client ID</label>
          <input
            id="oauth-client-id"
            className="form-input"
            type="text"
            value={value.oauth_client_id ?? ''}
            onChange={(event) => update('oauth_client_id', event.target.value)}
          />
        </div>
        <div className="form-group">
          <label htmlFor="oauth-client-secret">Client secret (optional)</label>
          <div className="http-api-secret-input-wrap">
            <input
              id="oauth-client-secret"
              className="form-input"
              type={clientSecretRevealed ? 'text' : 'password'}
              value={value.oauth_client_secret ?? ''}
              onChange={(event) => update('oauth_client_secret', event.target.value)}
            />
            <button
              type="button"
              className="settings-inline-copy settings-inline-copy-secondary http-api-secret-toggle-button"
              onClick={() => setClientSecretRevealed((current) => !current)}
              title={`${clientSecretRevealed ? 'Hide' : 'Show'} client secret`}
              aria-label={`${clientSecretRevealed ? 'Hide' : 'Show'} client secret`}
            >
              {clientSecretRevealed ? (
                <EyeOff size={14} aria-hidden="true" />
              ) : (
                <Eye size={14} aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
      </div>

      <div className="http-api-oauth-grid">
        <div className="form-group http-api-oauth-scopes">
          <label htmlFor="oauth-client-auth-method">Client authentication</label>
          <select
            id="oauth-client-auth-method"
            className="form-input"
            value={value.oauth_client_auth_method ?? 'none'}
            onChange={(event) =>
              update('oauth_client_auth_method', event.target.value as HttpApiOAuthClientAuthMethod)
            }
          >
            <option value="none">None</option>
            <option value="client_secret_post">Client secret in body</option>
            <option value="client_secret_basic">HTTP Basic</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="oauth-scopes">Scopes</label>
          <input
            id="oauth-scopes"
            className="form-input"
            type="text"
            value={scopesToText(value.oauth_scopes)}
            onChange={(event) =>
              update('oauth_scopes', event.target.value.split(/[\s,]+/).filter(Boolean))
            }
          />
        </div>
      </div>

      <details>
        <summary>Advanced endpoints</summary>
        <div className="http-api-oauth-grid">
          <label>
            Authorization endpoint
            <input
              className="form-input"
              type="url"
              value={value.oauth_authorization_url ?? ''}
              onChange={(event) => update('oauth_authorization_url', event.target.value)}
            />
          </label>
          <label>
            Device authorization endpoint
            <input
              className="form-input"
              type="url"
              value={value.oauth_device_authorization_url ?? ''}
              onChange={(event) => update('oauth_device_authorization_url', event.target.value)}
            />
          </label>
          <label>
            Token endpoint
            <input
              className="form-input"
              type="url"
              value={value.oauth_token_url ?? ''}
              onChange={(event) => update('oauth_token_url', event.target.value)}
            />
          </label>
        </div>
      </details>

      {status === 'pending' && value.oauth_flow === 'device_code' && deviceCode && (
        <div className="http-api-oauth-device" aria-label="Device authorization">
          <strong>Code: {deviceCode}</strong>
          <InlineCopyButton
            copyText={deviceCode}
            className="btn btn-secondary"
            title="Copy code"
            ariaLabel="Copy device code"
            label="Copy code"
          />
          {deviceUrl && (
            <a href={deviceUrl} target="_blank" rel="noreferrer">
              Open provider
            </a>
          )}
        </div>
      )}
      <div className="http-api-oauth-status-row">
        {(message || hasSavedCredential) && (
          <p role={status === 'error' ? 'alert' : 'status'}>
            {message || 'Connected. Save the tool to keep this credential.'}
          </p>
        )}
        <button
          type="button"
          className="btn btn-primary"
          onClick={() => void connect()}
          disabled={status === 'connecting' || status === 'pending'}
        >
          {status === 'connected' || hasSavedCredential ? 'Reconnect' : 'Connect'}
        </button>
      </div>
      <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
    </section>
  );
}
