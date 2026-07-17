import { X } from 'lucide-react';

import type { GitWebhookConfig, GitWebhookDelivery } from '@/types';

import { DeleteConfirmButton } from './DeleteConfirmButton';
import { InlineCopyButton } from './shared/InlineCopyButton';

interface GitWebhookSettingsProps {
  config: GitWebhookConfig;
  revealedSecret: string | null;
  deliveries: GitWebhookDelivery[];
  disabled: boolean;
  onEnable: () => void;
  onRotate: () => void;
  onDisable: () => void;
  onDismissSecret: () => void;
}

function getProviderInstruction(provider: GitWebhookConfig['provider']): string {
  if (provider === 'github') {
    return 'Configure the secret so GitHub sends X-Hub-Signature-256 HMAC signatures.';
  }
  if (provider === 'gitlab') {
    return 'Use the Secret Token field.';
  }
  return 'Use a token header when your provider supports custom headers.';
}

function getGenericProviderDetail(provider: GitWebhookConfig['provider']): string | null {
  if (provider !== 'generic') {
    return null;
  }
  return 'Accepted options include X-Ragtime-Webhook-Token, Authorization: Bearer, GitHub-style SHA-256 signatures, GitLab Secret Token headers, and compatible Gitea or Gogs SHA-256 headers.';
}

function getStatusLabel(status: GitWebhookDelivery['status']): string {
  return status.charAt(0).toUpperCase() + status.slice(1);
}

function getStatusClassName(status: GitWebhookDelivery['status']): string {
  switch (status) {
    case 'pending':
    case 'processing':
      return 'userspace-status-pill userspace-status-pill-info';
    case 'completed':
      return 'userspace-status-pill userspace-status-pill-success';
    case 'failed':
      return 'userspace-status-pill userspace-status-pill-danger';
    case 'skipped':
      return 'userspace-status-pill userspace-status-pill-warning';
    default:
      return 'userspace-status-pill userspace-status-pill-muted';
  }
}

function formatTimestamp(value: string | null): string | null {
  if (!value) {
    return null;
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return null;
  }
  return date.toLocaleString();
}

export function GitWebhookSettings({
  config,
  revealedSecret,
  deliveries,
  disabled,
  onEnable,
  onRotate,
  onDisable,
  onDismissSecret,
}: GitWebhookSettingsProps) {
  const queryTokenUrl =
    config.webhook_url && revealedSecret
      ? `${config.webhook_url}?token=${encodeURIComponent(revealedSecret)}`
      : null;
  const createdAt = formatTimestamp(config.created_at);

  return (
    <section className="git-webhook-settings">
      <div className="git-webhook-header">
        <div className="git-webhook-header-copy">
          <h3 className="git-webhook-title">Push webhook</h3>
          <p className="git-webhook-subtitle">
            Accepts push notifications for the configured branch and triggers the next sync.
          </p>
        </div>
        <span
          className={`userspace-status-pill ${config.enabled ? 'userspace-status-pill-success' : 'userspace-status-pill-muted'}`}
        >
          {config.enabled ? 'Enabled' : 'Disabled'}
        </span>
      </div>

      {!config.enabled ? (
        <div className="git-webhook-empty-state">
          <p className="git-webhook-empty-title">Webhook delivery is disabled.</p>
          <p className="field-help">
            Enable this webhook to receive push events for branch <strong>{config.branch}</strong>.
          </p>
          <div className="git-webhook-actions">
            <button type="button" className="btn btn-sm" onClick={onEnable} disabled={disabled}>
              Enable push webhook
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="git-webhook-grid">
            <div className="git-webhook-card">
              <div className="git-webhook-card-header">
                <h4>Provider setup</h4>
              </div>
              <dl className="git-webhook-meta">
                <div>
                  <dt>Provider</dt>
                  <dd>{config.provider}</dd>
                </div>
                <div>
                  <dt>Branch</dt>
                  <dd>
                    <strong>{config.branch}</strong>
                  </dd>
                </div>
                {createdAt && (
                  <div>
                    <dt>Created</dt>
                    <dd>{createdAt}</dd>
                  </div>
                )}
              </dl>
              <p className="field-help">{getProviderInstruction(config.provider)}</p>
              {getGenericProviderDetail(config.provider) && (
                <p className="field-help">{getGenericProviderDetail(config.provider)}</p>
              )}
            </div>

            <div className="git-webhook-card">
              <div className="git-webhook-card-header">
                <h4>Webhook URL</h4>
              </div>
              {config.webhook_url ? (
                <div className="git-webhook-code-row">
                  <code className="git-webhook-code">{config.webhook_url}</code>
                  <InlineCopyButton
                    copyText={config.webhook_url}
                    className="git-webhook-copy-btn"
                    title="Copy webhook URL"
                    ariaLabel="Copy webhook URL"
                    copiedTitle="Webhook URL copied"
                    copiedAriaLabel="Webhook URL copied"
                    disabled={disabled}
                  />
                </div>
              ) : (
                <p className="field-help">Enable the webhook to generate a registration URL.</p>
              )}
            </div>
          </div>

          {revealedSecret && (
            <div className="git-webhook-secret-panel" role="alert">
              <div className="git-webhook-secret-header">
                <div>
                  <h4>One-time secret</h4>
                  <p className="field-help">
                    Copy this now. Ragtime will not show the plaintext secret again after you
                    dismiss it.
                  </p>
                </div>
                <button
                  type="button"
                  className="git-webhook-dismiss-btn"
                  onClick={onDismissSecret}
                  disabled={disabled}
                  aria-label="Dismiss webhook secret"
                  title="Dismiss webhook secret"
                >
                  <X size={14} />
                  <span>Dismiss</span>
                </button>
              </div>

              <div className="git-webhook-code-row">
                <code className="git-webhook-code">{revealedSecret}</code>
                <InlineCopyButton
                  copyText={revealedSecret}
                  className="git-webhook-copy-btn"
                  title="Copy webhook secret"
                  ariaLabel="Copy webhook secret"
                  copiedTitle="Webhook secret copied"
                  copiedAriaLabel="Webhook secret copied"
                  disabled={disabled}
                />
              </div>

              {queryTokenUrl && (
                <div className="git-webhook-query-token">
                  <div className="git-webhook-query-token-header">
                    <h5>Fallback URL with query token</h5>
                    <InlineCopyButton
                      copyText={queryTokenUrl}
                      className="git-webhook-copy-btn"
                      title="Copy query token fallback URL"
                      ariaLabel="Copy query token fallback URL"
                      copiedTitle="Fallback URL copied"
                      copiedAriaLabel="Fallback URL copied"
                      disabled={disabled}
                    />
                  </div>
                  <code className="git-webhook-code">{queryTokenUrl}</code>
                  <p className="field-help">
                    This is less secure because URLs may be captured in provider logs.
                  </p>
                </div>
              )}
            </div>
          )}

          <div className="git-webhook-actions">
            <DeleteConfirmButton
              onDelete={onRotate}
              disabled={disabled}
              className="btn btn-sm btn-secondary"
              title="Rotate secret"
              buttonText="Rotate secret"
            />
            <DeleteConfirmButton
              onDelete={onDisable}
              disabled={disabled}
              className="btn btn-sm btn-danger"
              title="Disable webhook"
              buttonText="Disable webhook"
            />
          </div>
        </>
      )}

      <div className="git-webhook-deliveries">
        <div className="git-webhook-card-header">
          <h4>Recent deliveries</h4>
        </div>
        {deliveries.length === 0 ? (
          <p className="field-help">No deliveries yet.</p>
        ) : (
          <ul className="git-webhook-delivery-list">
            {deliveries.map((delivery) => {
              const receivedAt = formatTimestamp(delivery.received_at);
              const startedAt = formatTimestamp(delivery.started_at);
              const completedAt = formatTimestamp(delivery.completed_at);
              return (
                <li key={delivery.id} className="git-webhook-delivery-item">
                  <div className="git-webhook-delivery-topline">
                    <div className="git-webhook-delivery-title-wrap">
                      <strong>{delivery.event_name}</strong>
                      <span className={getStatusClassName(delivery.status)}>
                        {getStatusLabel(delivery.status)}
                      </span>
                    </div>
                    <span className="git-webhook-delivery-id">{delivery.id}</span>
                  </div>
                  <div className="git-webhook-delivery-grid">
                    {delivery.branch && (
                      <div>
                        <span className="git-webhook-delivery-label">Branch</span>
                        <code className="git-webhook-inline-code">{delivery.branch}</code>
                      </div>
                    )}
                    {delivery.head_commit && (
                      <div>
                        <span className="git-webhook-delivery-label">Head commit</span>
                        <code className="git-webhook-inline-code">{delivery.head_commit}</code>
                      </div>
                    )}
                    {receivedAt && (
                      <div>
                        <span className="git-webhook-delivery-label">Received</span>
                        <span>{receivedAt}</span>
                      </div>
                    )}
                    {startedAt && (
                      <div>
                        <span className="git-webhook-delivery-label">Started</span>
                        <span>{startedAt}</span>
                      </div>
                    )}
                    {completedAt && (
                      <div>
                        <span className="git-webhook-delivery-label">Completed</span>
                        <span>{completedAt}</span>
                      </div>
                    )}
                  </div>
                  {delivery.message && (
                    <p className="git-webhook-delivery-message">{delivery.message}</p>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </section>
  );
}
