import { useEffect, useRef, useState } from 'react';

import { api } from '@/api';
import type { PdmWebhookConfig, PdmWebhookEnableResponse } from '@/types';

import { DeleteConfirmButton } from './DeleteConfirmButton';
import { InlineCopyButton } from './shared/InlineCopyButton';
import { ReindexIntervalSelect } from './ReindexIntervalSelect';

interface PdmWebhookSettingsProps {
  toolId: string | null;
  disabled?: boolean;
  cadenceDisabled?: boolean;
  intervalHours: number;
  startMinute: number | null;
  timezone: string | null;
  onIntervalChange: (value: number) => void;
  onStartMinuteChange: (value: number | null) => void;
  onTimezoneChange: (value: string | null) => void;
  webhookDeliveryRequested: boolean;
  onWebhookDeliveryRequestedChange: (enabled: boolean) => void;
  activationResult?: PdmWebhookEnableResponse | null;
  onBusyChange?: (busy: boolean) => void;
}

function formatTimestamp(value: string | null): string {
  if (!value) return 'Never';
  return new Date(value).toLocaleString();
}

export function PdmWebhookSettings({
  toolId,
  disabled = false,
  cadenceDisabled = false,
  intervalHours,
  startMinute,
  timezone,
  onIntervalChange,
  onStartMinuteChange,
  onTimezoneChange,
  webhookDeliveryRequested,
  onWebhookDeliveryRequestedChange,
  activationResult = null,
  onBusyChange,
}: PdmWebhookSettingsProps) {
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
    setBusy(false);
    onBusyChange?.(false);
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
  }, [toolId, onBusyChange]);

  useEffect(() => {
    if (!activationResult) return;
    requestGenerationRef.current += 1;
    setConfig(activationResult);
    setSecret(activationResult.secret);
    setLoading(false);
    setBusy(false);
    onBusyChange?.(false);
  }, [activationResult, onBusyChange]);

  useEffect(() => {
    if (!config?.enabled || intervalHours <= 0) return;
    onIntervalChange(0);
    onStartMinuteChange(null);
    onTimezoneChange(null);
  }, [config?.enabled, intervalHours, onIntervalChange, onStartMinuteChange, onTimezoneChange]);

  const runAction = async (
    action: () => Promise<PdmWebhookConfig & { secret?: string | null }>,
    secretHandling: 'clear' | 'preserve' | 'replace',
  ): Promise<boolean> => {
    if (!toolId) return false;
    const generation = ++requestGenerationRef.current;
    setBusy(true);
    onBusyChange?.(true);
    setError(null);
    try {
      const result = await action();
      if (requestGenerationRef.current !== generation) return false;
      setConfig(result);
      if (secretHandling === 'clear') setSecret(null);
      if (secretHandling === 'replace') {
        setSecret(typeof result.secret === 'string' ? result.secret : null);
      }
      return true;
    } catch (reason) {
      if (requestGenerationRef.current !== generation) return false;
      setError(reason instanceof Error ? reason.message : 'Webhook action failed');
      return false;
    } finally {
      if (requestGenerationRef.current === generation) {
        setBusy(false);
        onBusyChange?.(false);
      }
    }
  };

  const handleWebhookDeliveryChange = async (enabled: boolean): Promise<boolean> => {
    if (!toolId) {
      onWebhookDeliveryRequestedChange(enabled);
      if (enabled) {
        onIntervalChange(0);
        onStartMinuteChange(null);
        onTimezoneChange(null);
      }
      return true;
    }

    const succeeded = enabled
      ? await runAction(() => api.enablePdmWebhook(toolId), 'replace')
      : await runAction(() => api.disablePdmWebhook(toolId), 'clear');
    if (succeeded && enabled) {
      onIntervalChange(0);
      onStartMinuteChange(null);
      onTimezoneChange(null);
    }
    return succeeded;
  };

  const webhookDeliveryEnabled = Boolean(config?.enabled) || webhookDeliveryRequested;
  const showWebhookDetails = webhookDeliveryEnabled && !loading;

  return (
    <section
      id="pdm-webhook-settings"
      className="git-webhook-settings"
      data-testid="pdm-webhook-settings"
    >
      <ReindexIntervalSelect
        value={intervalHours}
        onChange={onIntervalChange}
        webhookDeliveryEnabled={webhookDeliveryEnabled}
        onWebhookDeliveryChange={handleWebhookDeliveryChange}
        startMinute={startMinute}
        timezone={timezone}
        onStartMinuteChange={onStartMinuteChange}
        onTimezoneChange={onTimezoneChange}
        disabled={disabled || cadenceDisabled || loading || busy}
      />
      {loading && <p className="field-help">Loading webhook settings…</p>}
      {webhookDeliveryRequested && !toolId && !activationResult && (
        <p className="field-help">
          Webhook setup will be generated after this PDM tool is created.
        </p>
      )}
      {error && (
        <p className="field-help" role="alert">
          {error}
        </p>
      )}
      {showWebhookDetails && (
        <>
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
            {config?.enabled && toolId && (
              <>
                <DeleteConfirmButton
                  onDelete={() =>
                    void runAction(() => api.rotatePdmWebhookSecret(toolId), 'replace')
                  }
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
                    void runAction(
                      () =>
                        config.paused ? api.resumePdmWebhook(toolId) : api.pausePdmWebhook(toolId),
                      'preserve',
                    )
                  }
                  disabled={disabled || busy}
                >
                  {config.paused ? 'Resume webhook' : 'Pause webhook'}
                </button>
              </>
            )}
          </div>
        </>
      )}
    </section>
  );
}
