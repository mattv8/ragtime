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

import {
  ExternalApiCredentialConfirmDialog,
  ExternalApiCredentialTokenDialog,
  type CredentialAction,
  type ExternalApiCredentialDialogTokenState,
} from './ExternalApiCredentialDialogs';

interface ExternalApiAccessSectionProps {
  workspaceId: string;
  previewOrigin?: string | null;
}

interface PendingCredentialAction {
  action: CredentialAction;
  credential: WorkspaceExternalApiCredentialItem;
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

export function ExternalApiAccessSection({ workspaceId }: ExternalApiAccessSectionProps) {
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
  const [tokenReveal, setTokenReveal] = useState<ExternalApiCredentialDialogTokenState | null>(
    null,
  );
  const [pendingAction, setPendingAction] = useState<PendingCredentialAction | null>(null);
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
    setPendingAction(null);
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

  const selectableEndpointKeySet = useMemo(
    () => new Set(selectablePublishedEndpoints.map((endpoint) => endpoint.key)),
    [selectablePublishedEndpoints],
  );

  const revealCredential = (
    credential: WorkspaceExternalApiCredentialSecretResponse,
    endpointKey: string | null,
    operation: 'Created' | 'Rotated' = 'Created',
  ) => {
    const endpoint =
      (endpointKey ? endpointByKey.get(endpointKey) : null) ??
      (endpointKey ? candidateByKey.get(endpointKey) : null) ??
      null;
    setTokenReveal({
      token: credential.token,
      prefix: credential.token_prefix,
      label: credential.label,
      operation,
      endpointPath: endpoint?.path ?? null,
      method: endpoint?.method ?? 'GET',
    });
  };

  const loadRequests = async (nextWorkspaceId: string) => {
    const requestsResponse = await api.listWorkspaceExternalApiRequests(nextWorkspaceId);
    setState((current) => ({ ...current, requests: requestsResponse.items }));
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
    } finally {
      setPendingAction(null);
    }
  };

  const runRotate = async (credential: WorkspaceExternalApiCredentialItem) => {
    setState((current) => ({ ...current, saving: true, error: null }));
    try {
      const rotated = await api.rotateWorkspaceExternalApiCredential(workspaceId, credential.id);
      revealCredential(
        rotated,
        credential.endpoint_keys[0] ?? rotated.endpoint_keys[0] ?? null,
        'Rotated',
      );
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
    } finally {
      setPendingAction(null);
    }
  };

  const runDelete = async (credentialId: string) => {
    setState((current) => ({ ...current, saving: true, error: null }));
    try {
      await api.deleteWorkspaceExternalApiCredential(workspaceId, credentialId);
      setState((current) => ({
        ...current,
        saving: false,
        credentials: current.credentials.filter((item) => item.id !== credentialId),
      }));
      await loadRequests(workspaceId);
    } catch (error) {
      setState((current) => ({
        ...current,
        saving: false,
        error: getErrorMessage(error, 'Failed to delete credential'),
      }));
    } finally {
      setPendingAction(null);
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
      className="userspace-share-controls userspace-external-api-access userspace-external-api-section"
      aria-label="External API Access"
    >
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

      <section
        id="workspace-external-api-endpoints"
        className="userspace-external-api-section"
        aria-label="Candidate endpoints"
      >
        <div className="userspace-external-api-section-header">
          <h5 className="userspace-external-api-section-title userspace-external-api-section-heading">
            Endpoints
          </h5>
        </div>
        {publishableCandidates.length === 0 ? (
          <p className="userspace-muted">No published endpoints yet.</p>
        ) : (
          <div className="userspace-external-api-list">
            {publishableCandidates.map((candidate) => {
              const endpoint = endpointByKey.get(candidate.key);
              const isSelectable = selectableEndpointKeySet.has(candidate.key);
              const isSelected = selectedEndpointKeys.includes(candidate.key);
              const statusLabel = !endpoint
                ? 'Not published'
                : endpoint.stale
                  ? 'Stale'
                  : 'Published';
              const actionLabel = !endpoint ? 'Publish' : endpoint.stale ? 'Reapprove' : null;
              return (
                <article
                  key={candidate.key}
                  className={`userspace-external-api-row${isSelected ? ' is-credential-selected' : ''}`}
                  data-endpoint-key={candidate.key}
                >
                  <div className="userspace-external-api-row-main">
                    <div className="userspace-external-api-row-header">
                      <div className="userspace-external-api-row-copy">
                        <strong className="userspace-external-api-row-title">
                          {candidate.label}
                        </strong>
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
                    <p className="userspace-muted userspace-external-api-row-description">
                      {candidate.description}
                    </p>
                  </div>
                  <div className="userspace-external-api-row-actions">
                    {isSelectable ? (
                      <label className="userspace-external-api-checkbox-row userspace-external-api-endpoint-selection-control">
                        <input
                          type="checkbox"
                          aria-label={`Use ${candidate.label} for credential`}
                          checked={isSelected}
                          onChange={() => toggleEndpointKey(candidate.key)}
                        />
                        <span>Use for credential</span>
                      </label>
                    ) : null}
                    {actionLabel ? (
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
                    ) : null}
                  </div>
                </article>
              );
            })}
          </div>
        )}
        {selectedEndpointKeys.length > 0 ? (
          <div id="workspace-external-api-credential-details">
            <div className="userspace-external-api-section-header">
              <h5 className="userspace-external-api-section-title userspace-external-api-section-heading">
                Create credential
              </h5>
            </div>
            <div className="userspace-external-api-credential-fields">
              <div className="userspace-share-access-row userspace-external-api-field">
                <label
                  htmlFor="workspace-external-api-credential-label"
                  className="userspace-share-label userspace-external-api-field-label"
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

              <div className="userspace-share-access-row userspace-external-api-field">
                <label
                  htmlFor="workspace-external-api-credential-expiry"
                  className="userspace-share-label userspace-external-api-field-label"
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
            </div>

            <div className="userspace-external-api-create-actions">
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
          </div>
        ) : null}
      </section>

      <section
        id="workspace-external-api-credentials"
        className="userspace-external-api-section"
        role="region"
        aria-label="Service credentials"
      >
        <div className="userspace-external-api-section-header">
          <h5 className="userspace-external-api-section-title userspace-external-api-section-heading">
            Service credentials
          </h5>
        </div>
        {state.credentials.length === 0 ? (
          <p className="userspace-muted">No service credentials yet.</p>
        ) : (
          <div className="userspace-external-api-list userspace-external-api-credential-list">
            {state.credentials.map((credential) => (
              <article
                key={credential.id}
                className="userspace-external-api-row userspace-external-api-credential-row"
                data-credential-id={credential.id}
              >
                <div className="userspace-external-api-row-main">
                  <div className="userspace-external-api-row-header">
                    <div className="userspace-external-api-row-copy">
                      <strong className="userspace-external-api-row-title">
                        {credential.label}
                      </strong>
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
                    <span>Expires {formatTimestamp(credential.expires_at)}</span>
                    <span>Last used {formatTimestamp(credential.last_used_at)}</span>
                    <span>{credential.request_count} requests</span>
                  </div>
                </div>
                <div className="userspace-external-api-row-actions userspace-external-api-credential-actions">
                  {credential.revoked_at ? (
                    <button
                      type="button"
                      className="btn btn-danger btn-sm"
                      disabled={state.saving}
                      onClick={() => setPendingAction({ action: 'delete', credential })}
                    >
                      Delete Credential
                    </button>
                  ) : (
                    <>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        disabled={state.saving}
                        onClick={() => setPendingAction({ action: 'rotate', credential })}
                      >
                        Rotate Credential
                      </button>
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        disabled={state.saving}
                        onClick={() => setPendingAction({ action: 'revoke', credential })}
                      >
                        Revoke Credential
                      </button>
                    </>
                  )}
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <section
        id="workspace-external-api-history"
        className="userspace-external-api-section"
        role="region"
        aria-label="Request history"
      >
        <div className="userspace-external-api-section-header">
          <h5 className="userspace-external-api-section-title userspace-external-api-section-heading">
            Request history
          </h5>
        </div>
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

      {tokenReveal ? (
        <ExternalApiCredentialTokenDialog
          workspaceId={workspaceId}
          tokenState={tokenReveal}
          onClose={() => setTokenReveal(null)}
        />
      ) : null}

      {pendingAction ? (
        <ExternalApiCredentialConfirmDialog
          action={pendingAction.action}
          credential={pendingAction.credential}
          isSubmitting={state.saving}
          onCancel={() => {
            if (!state.saving) {
              setPendingAction(null);
            }
          }}
          onConfirm={() => {
            if (state.saving) {
              return;
            }
            if (pendingAction.action === 'rotate') {
              void runRotate(pendingAction.credential);
              return;
            }
            if (pendingAction.action === 'revoke') {
              void runRevoke(pendingAction.credential.id);
              return;
            }
            void runDelete(pendingAction.credential.id);
          }}
        />
      ) : null}
    </section>
  );
}
