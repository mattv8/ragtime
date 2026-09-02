import { useEffect, useMemo, useRef, useState } from 'react';

import { api } from '@/api';
import type {
  CreateWorkspaceExternalApiCredentialRequest,
  WorkspaceExternalApiCredentialItem,
  WorkspaceExternalApiCredentialSecretResponse,
  WorkspaceExternalApiEndpointItem,
  WorkspaceExternalApiManifestCandidate,
  WorkspaceExternalApiManifestResponse,
  WorkspaceExternalApiRequestHistoryItem,
} from '@/types';

import { InlineCopyButton } from './InlineCopyButton';

interface ExternalApiAccessSectionProps {
  workspaceId: string;
  previewOrigin?: string | null;
}

interface TokenRevealState {
  token: string;
  tokenPrefix: string;
  label: string;
  endpoint: WorkspaceExternalApiManifestCandidate | WorkspaceExternalApiEndpointItem | null;
  expiresAt: string | null;
}

interface ExternalApiAccessState {
  workspaceId: string;
  manifest: WorkspaceExternalApiManifestResponse | null;
  endpoints: WorkspaceExternalApiEndpointItem[];
  credentials: WorkspaceExternalApiCredentialItem[];
  requests: WorkspaceExternalApiRequestHistoryItem[];
  loading: boolean;
  saving: boolean;
  error: string | null;
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'Never';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function buildCurlExample(
  origin: string,
  endpointPath: string,
  token: string,
  method: string,
): string {
  return `curl -X ${method} "${origin}${endpointPath}" -H "Authorization: Bearer ${token}"`;
}

function buildPowerQueryExample(origin: string, endpointPath: string, token: string): string {
  return `let\n    Source = Json.Document(Web.Contents("${origin}${endpointPath}", [Headers=[Authorization="Bearer ${token}"]]))\nin\n    Source`;
}

function toCredentialListItem(
  credential: WorkspaceExternalApiCredentialSecretResponse,
): WorkspaceExternalApiCredentialItem {
  return {
    id: credential.id,
    label: credential.label,
    token_prefix: credential.token_prefix,
    enabled: credential.enabled,
    expires_at: credential.expires_at,
    last_used_at: null,
    request_count: 0,
    revoked_at: null,
    endpoint_keys: credential.endpoint_keys,
  };
}

function getErrorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message.trim() ? error.message : fallback;
}

export function ExternalApiAccessSection({
  workspaceId,
  previewOrigin,
}: ExternalApiAccessSectionProps) {
  const [state, setState] = useState<ExternalApiAccessState>({
    workspaceId,
    manifest: null,
    endpoints: [],
    credentials: [],
    requests: [],
    loading: true,
    saving: false,
    error: null,
  });
  const [selectedEndpointKeys, setSelectedEndpointKeys] = useState<string[]>([]);
  const [credentialLabel, setCredentialLabel] = useState('');
  const [expiresAt, setExpiresAt] = useState('');
  const [tokenReveal, setTokenReveal] = useState<TokenRevealState | null>(null);
  const requestGenerationRef = useRef(0);

  const loadSection = async (nextWorkspaceId: string) => {
    requestGenerationRef.current += 1;
    const generation = requestGenerationRef.current;
    setState((current) => ({
      ...current,
      workspaceId: nextWorkspaceId,
      loading: true,
      error: null,
    }));

    try {
      const [manifest, endpointsResponse, credentialsResponse, requestsResponse] =
        await Promise.all([
          api.getWorkspaceExternalApiManifest(nextWorkspaceId),
          api.listWorkspaceExternalApiEndpoints(nextWorkspaceId),
          api.listWorkspaceExternalApiCredentials(nextWorkspaceId),
          api.listWorkspaceExternalApiRequests(nextWorkspaceId),
        ]);
      if (requestGenerationRef.current !== generation) {
        return;
      }
      setState({
        workspaceId: nextWorkspaceId,
        manifest,
        endpoints: endpointsResponse.items,
        credentials: credentialsResponse.items,
        requests: requestsResponse.items,
        loading: false,
        saving: false,
        error: null,
      });
    } catch (error) {
      if (requestGenerationRef.current !== generation) {
        return;
      }
      setState({
        workspaceId: nextWorkspaceId,
        manifest: null,
        endpoints: [],
        credentials: [],
        requests: [],
        loading: false,
        saving: false,
        error: getErrorMessage(error, 'Failed to load external API access'),
      });
    }
  };

  useEffect(() => {
    setSelectedEndpointKeys([]);
    setCredentialLabel('');
    setExpiresAt('');
    setTokenReveal(null);
    void loadSection(workspaceId);
  }, [workspaceId]);

  const candidateByKey = useMemo(() => {
    const next = new Map<string, WorkspaceExternalApiManifestCandidate>();
    for (const candidate of state.manifest?.candidates ?? []) {
      if (candidate.valid) {
        next.set(candidate.key, candidate);
      }
    }
    return next;
  }, [state.manifest]);

  const endpointByKey = useMemo(() => {
    const next = new Map<string, WorkspaceExternalApiEndpointItem>();
    for (const endpoint of state.endpoints) {
      next.set(endpoint.key, endpoint);
    }
    return next;
  }, [state.endpoints]);

  const publishableCandidates = useMemo(
    () => (state.manifest?.candidates ?? []).filter((candidate) => candidate.valid),
    [state.manifest],
  );

  const selectablePublishedEndpoints = useMemo(
    () => state.endpoints.filter((endpoint) => endpoint.enabled && !endpoint.stale),
    [state.endpoints],
  );

  const resolvedPreviewOrigin =
    previewOrigin ?? state.manifest?.preview_origin ?? 'https://workspace-preview.invalid';

  const revealCredential = (
    credential: WorkspaceExternalApiCredentialSecretResponse,
    endpointKey: string | null,
  ) => {
    setTokenReveal({
      token: credential.token,
      tokenPrefix: credential.token_prefix,
      label: credential.label,
      endpoint:
        (endpointKey ? endpointByKey.get(endpointKey) : null) ??
        (endpointKey ? candidateByKey.get(endpointKey) : null) ??
        null,
      expiresAt: credential.expires_at,
    });
  };

  const runPublish = async (key: string) => {
    setState((current) => ({ ...current, saving: true, error: null }));
    try {
      const endpoint = await api.createWorkspaceExternalApiEndpoint(workspaceId, key);
      setState((current) => {
        const withoutCurrent = current.endpoints.filter((item) => item.key !== key);
        return {
          ...current,
          saving: false,
          endpoints: [...withoutCurrent, endpoint].sort((left, right) =>
            left.label.localeCompare(right.label),
          ),
        };
      });
    } catch (error) {
      setState((current) => ({
        ...current,
        saving: false,
        error: getErrorMessage(error, 'Failed to publish endpoint'),
      }));
    }
  };

  const runRevoke = async (credentialId: string) => {
    if (!window.confirm('Revoke this service credential? This cannot be undone.')) {
      return;
    }
    setState((current) => ({ ...current, saving: true, error: null }));
    try {
      const credential = await api.revokeWorkspaceExternalApiCredential(workspaceId, credentialId);
      setState((current) => ({
        ...current,
        saving: false,
        credentials: current.credentials.map((item) =>
          item.id === credentialId ? credential : item,
        ),
      }));
    } catch (error) {
      setState((current) => ({
        ...current,
        saving: false,
        error: getErrorMessage(error, 'Failed to revoke credential'),
      }));
    }
  };

  const runRotate = async (credential: WorkspaceExternalApiCredentialItem) => {
    if (
      !window.confirm(
        'Rotate this service credential? Existing clients will stop working immediately.',
      )
    ) {
      return;
    }
    setState((current) => ({ ...current, saving: true, error: null }));
    try {
      const rotated = await api.rotateWorkspaceExternalApiCredential(workspaceId, credential.id);
      revealCredential(rotated, credential.endpoint_keys[0] ?? rotated.endpoint_keys[0] ?? null);
      setState((current) => ({
        ...current,
        saving: false,
        credentials: current.credentials.map((item) =>
          item.id === credential.id ? { ...item, ...toCredentialListItem(rotated) } : item,
        ),
      }));
    } catch (error) {
      setState((current) => ({
        ...current,
        saving: false,
        error: getErrorMessage(error, 'Failed to rotate credential'),
      }));
    }
  };

  const runCreateCredential = async () => {
    const request: CreateWorkspaceExternalApiCredentialRequest = {
      label: credentialLabel.trim(),
      endpoint_keys: selectedEndpointKeys,
      expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
    };
    setState((current) => ({ ...current, saving: true, error: null }));
    try {
      const created = await api.createWorkspaceExternalApiCredential(workspaceId, request);
      revealCredential(created, request.endpoint_keys[0] ?? null);
      setState((current) => ({
        ...current,
        saving: false,
        credentials: [...current.credentials, toCredentialListItem(created)].sort((left, right) =>
          left.label.localeCompare(right.label),
        ),
      }));
      setCredentialLabel('');
      setSelectedEndpointKeys([]);
      setExpiresAt('');
    } catch (error) {
      setState((current) => ({
        ...current,
        saving: false,
        error: getErrorMessage(error, 'Failed to create credential'),
      }));
    }
  };

  const toggleEndpointKey = (key: string) => {
    setSelectedEndpointKeys((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key],
    );
  };

  const canCreateCredential = credentialLabel.trim().length > 0 && selectedEndpointKeys.length > 0;
  const revealEndpointPath = tokenReveal?.endpoint?.path ?? '/';
  const revealEndpointMethod = tokenReveal?.endpoint?.method ?? 'GET';
  const curlExample = tokenReveal
    ? buildCurlExample(
        resolvedPreviewOrigin,
        revealEndpointPath,
        tokenReveal.token,
        revealEndpointMethod,
      )
    : '';
  const powerQueryExample = tokenReveal
    ? buildPowerQueryExample(resolvedPreviewOrigin, revealEndpointPath, tokenReveal.token)
    : '';

  if (state.loading) {
    return (
      <section id="workspace-external-api-access" className="userspace-share-controls">
        <h4>External API Access</h4>
        <p className="userspace-muted">Loading external API access...</p>
      </section>
    );
  }

  return (
    <section
      id="workspace-external-api-access"
      className="userspace-share-controls userspace-external-api-access"
      aria-label="External API Access"
    >
      <h4>External API Access</h4>
      <p className="userspace-muted">
        Publish machine-readable GET or HEAD routes, issue service credentials, and copy
        bearer-token examples for Power Query or curl.
      </p>

      {state.error && (
        <p className="userspace-error" role="alert">
          {state.error}
        </p>
      )}

      {state.manifest && !state.manifest.valid && state.manifest.errors.length > 0 && (
        <div className="userspace-external-api-callout" role="alert">
          <strong>Manifest errors</strong>
          <ul>
            {state.manifest.errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="userspace-external-api-grid">
        <section className="userspace-external-api-card" aria-label="Candidate endpoints">
          <h5>Candidate endpoints</h5>
          {publishableCandidates.length === 0 ? (
            <p className="userspace-muted">No published endpoints yet.</p>
          ) : (
            <div className="userspace-external-api-list">
              {publishableCandidates.map((candidate) => {
                const endpoint = endpointByKey.get(candidate.key);
                const statusLabel = !endpoint
                  ? 'Not published'
                  : endpoint.stale
                    ? 'Stale'
                    : 'Published';
                const actionLabel = !endpoint ? 'Publish' : endpoint.stale ? 'Reapprove' : null;
                return (
                  <article key={candidate.key} className="userspace-external-api-item">
                    <div className="userspace-external-api-item-header">
                      <div>
                        <label className="userspace-external-api-item-title">
                          <span>{candidate.label}</span>
                        </label>
                        <div className="userspace-external-api-item-meta">
                          <code>{candidate.method}</code>
                          <code>{candidate.path}</code>
                        </div>
                      </div>
                      <span
                        className={`userspace-external-api-status${endpoint?.stale ? ' is-stale' : ''}`}
                      >
                        {statusLabel}
                      </span>
                    </div>
                    <p className="userspace-muted">{candidate.description}</p>
                    {actionLabel && (
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        disabled={state.saving}
                        onClick={() => {
                          void runPublish(candidate.key);
                        }}
                      >
                        {actionLabel}
                      </button>
                    )}
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <section className="userspace-external-api-card" aria-label="Create service credential">
          <h5>Create service credential</h5>
          {selectablePublishedEndpoints.length === 0 ? (
            <p className="userspace-muted">Publish and approve at least one endpoint first.</p>
          ) : (
            <>
              <div className="userspace-external-api-endpoint-checklist">
                {selectablePublishedEndpoints.map((endpoint) => (
                  <label key={endpoint.id} className="userspace-external-api-checkbox-row">
                    <input
                      type="checkbox"
                      checked={selectedEndpointKeys.includes(endpoint.key)}
                      onChange={() => toggleEndpointKey(endpoint.key)}
                    />
                    <span>
                      <strong>{endpoint.label}</strong>
                      <span className="userspace-external-api-checkbox-meta">
                        {endpoint.method} {endpoint.path}
                      </span>
                    </span>
                  </label>
                ))}
              </div>

              <div className="userspace-share-access-row">
                <label
                  htmlFor="workspace-external-api-credential-label"
                  className="userspace-share-label"
                >
                  Credential label
                </label>
                <input
                  id="workspace-external-api-credential-label"
                  value={credentialLabel}
                  onChange={(event) => setCredentialLabel(event.target.value)}
                  placeholder="August workpapers"
                />
              </div>

              <div className="userspace-share-access-row">
                <label
                  htmlFor="workspace-external-api-credential-expiry"
                  className="userspace-share-label"
                >
                  Expiry (optional)
                </label>
                <input
                  id="workspace-external-api-credential-expiry"
                  type="datetime-local"
                  value={expiresAt}
                  onChange={(event) => setExpiresAt(event.target.value)}
                />
              </div>

              <div className="userspace-share-actions userspace-share-actions-single">
                <button
                  type="button"
                  className="btn btn-primary"
                  disabled={!canCreateCredential || state.saving}
                  onClick={() => {
                    void runCreateCredential();
                  }}
                >
                  Create Credential
                </button>
              </div>
            </>
          )}
        </section>
      </div>

      {tokenReveal && (
        <section className="userspace-external-api-card" aria-label="Credential secret reveal">
          <div className="userspace-external-api-reveal-header">
            <div>
              <h5>Copy this token now</h5>
              <p className="userspace-muted">
                This secret is shown once for {tokenReveal.label}. Store it in the client now.
              </p>
            </div>
          </div>

          <div className="userspace-share-access-row userspace-share-link-pane">
            <label htmlFor="workspace-external-api-token" className="userspace-share-label">
              Bearer token
            </label>
            <div className="userspace-share-url-copy-wrap">
              <input id="workspace-external-api-token" value={tokenReveal.token} readOnly />
              <InlineCopyButton
                copyText={tokenReveal.token}
                className="userspace-share-inline-copy"
                title="Copy bearer token"
                ariaLabel="Copy bearer token"
                copiedTitle="Bearer token copied"
                copiedAriaLabel="Bearer token copied"
                iconSize={12}
              />
            </div>
          </div>

          <div className="userspace-external-api-examples">
            <div>
              <div className="userspace-external-api-example-header">
                <label className="userspace-share-label">curl example</label>
                <InlineCopyButton
                  copyText={curlExample}
                  className="btn btn-secondary btn-sm"
                  title="Copy curl example"
                  ariaLabel="Copy curl example"
                  copiedTitle="Curl example copied"
                  copiedAriaLabel="Curl example copied"
                  label="Copy"
                  copiedLabel="Copied"
                />
              </div>
              <pre className="userspace-external-api-code-block">{curlExample}</pre>
            </div>

            <div>
              <div className="userspace-external-api-example-header">
                <label className="userspace-share-label">Power Query (M)</label>
                <InlineCopyButton
                  copyText={powerQueryExample}
                  className="btn btn-secondary btn-sm"
                  title="Copy Power Query example"
                  ariaLabel="Copy Power Query example"
                  copiedTitle="Power Query example copied"
                  copiedAriaLabel="Power Query example copied"
                  label="Copy"
                  copiedLabel="Copied"
                />
              </div>
              <pre className="userspace-external-api-code-block">{powerQueryExample}</pre>
            </div>
          </div>
        </section>
      )}

      <section
        className="userspace-external-api-card"
        role="region"
        aria-label="Service credentials"
      >
        <h5>Service credentials</h5>
        {state.credentials.length === 0 ? (
          <p className="userspace-muted">No service credentials yet.</p>
        ) : (
          <div className="userspace-external-api-list">
            {state.credentials.map((credential) => (
              <article key={credential.id} className="userspace-external-api-item">
                <div className="userspace-external-api-item-header">
                  <div>
                    <strong>{credential.label}</strong>
                    <div className="userspace-external-api-item-meta">
                      <code>{credential.token_prefix}</code>
                      <span>{credential.endpoint_keys.join(', ')}</span>
                    </div>
                  </div>
                  <span
                    className={`userspace-external-api-status${credential.revoked_at ? ' is-revoked' : ''}`}
                  >
                    {credential.revoked_at
                      ? 'Revoked'
                      : credential.enabled
                        ? 'Enabled'
                        : 'Disabled'}
                  </span>
                </div>
                <div className="userspace-external-api-credential-meta">
                  <span>Created access retained</span>
                  <span>Expires {formatTimestamp(credential.expires_at)}</span>
                  <span>Last used {formatTimestamp(credential.last_used_at)}</span>
                  <span>{credential.request_count} requests</span>
                </div>
                <div className="userspace-share-actions userspace-share-actions-edit">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={state.saving || Boolean(credential.revoked_at)}
                    onClick={() => {
                      void runRotate(credential);
                    }}
                  >
                    Rotate Credential
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={state.saving || Boolean(credential.revoked_at)}
                    onClick={() => {
                      void runRevoke(credential.id);
                    }}
                  >
                    Revoke Credential
                  </button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <section className="userspace-external-api-card" role="region" aria-label="Request history">
        <h5>Request history</h5>
        {state.requests.length === 0 ? (
          <p className="userspace-muted">No requests recorded yet.</p>
        ) : (
          <div className="userspace-external-api-table-wrap">
            <table className="jobs-table userspace-external-api-table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>Credential</th>
                  <th>Endpoint</th>
                  <th>Path</th>
                  <th>Status</th>
                  <th>Duration</th>
                </tr>
              </thead>
              <tbody>
                {state.requests.map((request) => (
                  <tr key={request.id}>
                    <td>{formatTimestamp(request.created_at)}</td>
                    <td>{request.credential_label ?? 'Unknown credential'}</td>
                    <td>{request.endpoint_label ?? request.endpoint_key ?? 'Unknown endpoint'}</td>
                    <td>{request.path_template}</td>
                    <td>{request.status_code}</td>
                    <td>{request.duration_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </section>
  );
}
