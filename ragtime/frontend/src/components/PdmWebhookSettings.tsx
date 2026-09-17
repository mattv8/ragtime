import { useEffect, useRef, useState } from 'react';

import { api } from '@/api';
import type { PdmWebhookConfig } from '@/types';

import { DeleteConfirmButton } from './DeleteConfirmButton';
import { InlineCopyButton } from './shared/InlineCopyButton';

interface PdmWebhookSettingsProps {
  toolId: string | null;
  disabled?: boolean;
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'Never';
  return new Date(value).toLocaleString();
}

export function PdmWebhookSettings({ toolId, disabled = false }: PdmWebhookSettingsProps) {
  const [config, setConfig] = useState<PdmWebhookConfig | null>(null);
  const [secret, setSecret] = useState<string | null>(null);
  const [loading, setLoading] = useState(Boolean(toolId));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestGenerationRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    const generation = ++requestGenerationRef.current;
    setSecret(null);
    setError(null);
    if (!toolId) {
      setConfig(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    void api
      .getPdmWebhook(toolId)
      .then((result) => {
        if (!cancelled && requestGenerationRef.current === generation) setConfig(result);
      })
      .catch((reason) => {
        if (!cancelled && requestGenerationRef.current === generation)
          setError(reason instanceof Error ? reason.message : 'Failed to load webhook');
      })
      .finally(() => {
        if (!cancelled && requestGenerationRef.current === generation) setLoading(false);
      });
    return () => {
      cancelled = true;
      requestGenerationRef.current += 1;
      setSecret(null);
    };
  }, [toolId]);

  const runAction = async (
    action: () => Promise<PdmWebhookConfig & { secret?: string | null }>,
  ) => {
    if (!toolId) return;
    const generation = ++requestGenerationRef.current;
    setBusy(true);
    setError(null);
    try {
      const result = await action();
      if (requestGenerationRef.current !== generation) return;
      setConfig(result);
      setSecret(typeof result.secret === 'string' ? result.secret : null);
    } catch (reason) {
      if (requestGenerationRef.current !== generation) return;
      setError(reason instanceof Error ? reason.message : 'Webhook action failed');
    } finally {
      if (requestGenerationRef.current === generation) setBusy(false);
    }
  };

  if (!toolId) {
    return (
      <section
        id="pdm-webhook-settings"
        className="git-webhook-settings"
        data-testid="pdm-webhook-settings"
      >
        <h3 className="git-webhook-title">PDM webhook</h3>
        <p className="field-help">Save this PDM tool before enabling webhook delivery.</p>
      </section>
    );
  }

  if (loading) return <p className="field-help">Loading webhook settings…</p>;

  return (
    <section
      id="pdm-webhook-settings"
      className="git-webhook-settings"
      data-testid="pdm-webhook-settings"
    >
      <div className="git-webhook-header">
        <div className="git-webhook-header-copy">
          <h3 className="git-webhook-title">PDM webhook</h3>
          <p className="git-webhook-subtitle">
            Accepts authenticated notifications from your external PDM script.
          </p>
        </div>
        {config && (
          <span
            className={`userspace-status-pill ${config.paused ? 'userspace-status-pill-warning' : config.enabled ? 'userspace-status-pill-success' : ''}`}
          >
            {config.paused ? 'Paused' : config.enabled ? 'Active' : 'Disabled'}
          </span>
        )}
      </div>
      {error && (
        <p className="field-help" role="alert">
          {error}
        </p>
      )}
      {config?.webhook_url && (
        <div className="userspace-share-url-copy-wrap">
          <input aria-label="PDM webhook URL" type="text" value={config.webhook_url} readOnly />
          <InlineCopyButton
            copyText={config.webhook_url}
            className="userspace-share-inline-copy"
            title="Copy PDM webhook URL"
            ariaLabel="Copy PDM webhook URL"
            copiedTitle="PDM webhook URL copied"
            copiedAriaLabel="PDM webhook URL copied"
            iconSize={12}
            disabled={disabled || busy}
          />
        </div>
      )}
      {secret && (
        <div className="git-webhook-secret" role="alert">
          <h4>One-time secret</h4>
          <p className="field-help">Copy this now. It is not shown again after dismissal.</p>
          <div className="userspace-share-url-copy-wrap">
            <input aria-label="PDM one-time secret" type="text" value={secret} readOnly />
            <InlineCopyButton
              copyText={secret}
              className="userspace-share-inline-copy is-always-visible"
              title="Copy PDM webhook secret"
              ariaLabel="Copy PDM webhook secret"
              copiedTitle="PDM webhook secret copied"
              copiedAriaLabel="PDM webhook secret copied"
              iconSize={12}
              disabled={disabled || busy}
            />
          </div>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => setSecret(null)}
            disabled={busy}
          >
            Dismiss secret
          </button>
        </div>
      )}
      {config && (
        <dl className="tool-card-schema-stats" data-testid="pdm-webhook-status">
          <div>
            <dt>Last event</dt>
            <dd>{formatTimestamp(config.last_received_at)}</dd>
          </div>
          <div>
            <dt>Queue</dt>
            <dd>{config.active_job_id ? 'Running' : config.pending ? 'Pending' : 'Idle'}</dd>
          </div>
          <div>
            <dt>Last success</dt>
            <dd>{formatTimestamp(config.last_success_at)}</dd>
          </div>
          {config.last_error && (
            <div>
              <dt>Last error</dt>
              <dd>{config.last_error}</dd>
            </div>
          )}
        </dl>
      )}
      <div className="git-webhook-actions">
        {!config?.enabled ? (
          <button
            id="pdm-webhook-enable"
            type="button"
            className="btn btn-sm btn-primary"
            onClick={() => void runAction(() => api.enablePdmWebhook(toolId))}
            disabled={disabled || busy}
          >
            Enable webhook
          </button>
        ) : (
          <>
            <DeleteConfirmButton
              onDelete={() => void runAction(() => api.disablePdmWebhook(toolId))}
              disabled={disabled || busy}
              className="btn btn-sm btn-danger"
              title="Disable webhook"
              buttonText="Disable webhook"
            />
            <DeleteConfirmButton
              onDelete={() => void runAction(() => api.rotatePdmWebhookSecret(toolId))}
              disabled={disabled || busy}
              className="btn btn-sm btn-secondary"
              title="Rotate secret"
              buttonText="Rotate secret"
            />
            <button
              id="pdm-webhook-pause-toggle"
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() =>
                void runAction(() =>
                  config.paused ? api.resumePdmWebhook(toolId) : api.pausePdmWebhook(toolId),
                )
              }
              disabled={disabled || busy}
            >
              {config.paused ? 'Resume webhook' : 'Pause webhook'}
            </button>
          </>
        )}
      </div>
    </section>
  );
}
