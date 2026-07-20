import { useEffect, useState } from 'react';

import type { GitWebhookConfig } from '@/types';

import { DeleteConfirmButton } from './DeleteConfirmButton';
import { InlineCopyButton } from './shared/InlineCopyButton';

interface GitWebhookSettingsProps {
  config: GitWebhookConfig;
  revealedSecret: string | null;
  disabled: boolean;
  onRotate: () => void;
  onPause: () => void;
  onResume: () => void;
}

function getProviderInstruction(provider: GitWebhookConfig['provider']): string {
  if (provider === 'github') {
    return "Add this URL and secret in your GitHub repository's webhook settings.";
  }
  if (provider === 'gitlab') {
    return "Add this URL and secret in your GitLab project's webhook settings.";
  }
  return "Add this URL and secret in your Git provider's webhook settings.";
}

export function GitWebhookSettings({
  config,
  revealedSecret,
  disabled,
  onRotate,
  onPause,
  onResume,
}: GitWebhookSettingsProps) {
  type WebhookUrlMode = 'webhook' | 'query-token';

  const [urlMode, setUrlMode] = useState<WebhookUrlMode>('webhook');
  const queryTokenUrl =
    config.webhook_url && revealedSecret
      ? `${config.webhook_url}?token=${encodeURIComponent(revealedSecret)}`
      : null;
  const canUseQueryToken = Boolean(queryTokenUrl && revealedSecret);
  const displayedUrl =
    urlMode === 'query-token' && canUseQueryToken ? queryTokenUrl : config.webhook_url;

  useEffect(() => {
    if (!canUseQueryToken) {
      setUrlMode('webhook');
    }
  }, [canUseQueryToken]);

  return (
    <section className="git-webhook-settings">
      <div className="git-webhook-header">
        <div className="git-webhook-header-copy">
          <h3 className="git-webhook-title">Push webhook</h3>
          <p className="git-webhook-subtitle">
            Automatically re-indexes this repository whenever you push to the configured branch.
          </p>
        </div>
        <span
          className={`userspace-status-pill ${
            config.paused ? 'userspace-status-pill-warning' : 'userspace-status-pill-success'
          }`}
        >
          {config.paused ? 'Paused' : 'Active'}
        </span>
      </div>

      <>
        <div className="git-webhook-setup">
          <p className="field-help">{getProviderInstruction(config.provider)}</p>

          {revealedSecret && (
            <div className="git-webhook-secret" role="alert">
              <h4>One-time secret</h4>
              <p className="field-help">
                Copy this now. This secret will not be shown again after you close this window.
              </p>
              <div className="userspace-share-url-copy-wrap">
                <input type="text" aria-label="One-time secret" value={revealedSecret} readOnly />
                <InlineCopyButton
                  copyText={revealedSecret}
                  className="userspace-share-inline-copy is-always-visible"
                  title="Copy webhook secret"
                  ariaLabel="Copy webhook secret"
                  copiedTitle="Webhook secret copied"
                  copiedAriaLabel="Webhook secret copied"
                  iconSize={12}
                  disabled={disabled}
                />
              </div>
            </div>
          )}

          {canUseQueryToken && (
            <div
              className="git-webhook-url-options"
              role="radiogroup"
              aria-label="Webhook URL mode"
            >
              <label>
                <input
                  type="radio"
                  name="git-webhook-url-mode"
                  value="webhook"
                  checked={urlMode === 'webhook'}
                  onChange={() => setUrlMode('webhook')}
                  disabled={disabled}
                />
                <span>Webhook URL</span>
              </label>
              <label>
                <input
                  type="radio"
                  name="git-webhook-url-mode"
                  value="query-token"
                  checked={urlMode === 'query-token'}
                  onChange={() => setUrlMode('query-token')}
                  disabled={disabled}
                />
                <span>URL with query token (less secure)</span>
              </label>
            </div>
          )}

          {displayedUrl ? (
            <div className="userspace-share-url-copy-wrap">
              <input type="text" aria-label="Selected webhook URL" value={displayedUrl} readOnly />
              <InlineCopyButton
                copyText={displayedUrl}
                className="userspace-share-inline-copy"
                title="Copy selected webhook URL"
                ariaLabel="Copy selected webhook URL"
                copiedTitle="Selected webhook URL copied"
                copiedAriaLabel="Selected webhook URL copied"
                iconSize={12}
                disabled={disabled}
              />
            </div>
          ) : (
            <p className="field-help">Enable the webhook to generate a registration URL.</p>
          )}

          {urlMode === 'query-token' && canUseQueryToken && (
            <p className="field-help">
              This is less secure because URLs may be captured in provider logs.
            </p>
          )}

          {!canUseQueryToken && config.webhook_url && (
            <p className="field-help">
              Need a URL with an embedded query token? Rotate the secret to reveal one; Ragtime does
              not store the plaintext secret after it is shown.
            </p>
          )}
        </div>

        <div className="git-webhook-actions">
          <DeleteConfirmButton
            onDelete={onRotate}
            disabled={disabled}
            className="btn btn-sm btn-secondary"
            title="Rotate secret"
            buttonText="Rotate secret"
          />
          <DeleteConfirmButton
            onDelete={config.paused ? onResume : onPause}
            disabled={disabled}
            className={`btn btn-sm ${config.paused ? 'btn-primary' : 'btn-secondary'}`}
            title={config.paused ? 'Resume webhook' : 'Pause webhook'}
            buttonText={config.paused ? 'Resume webhook' : 'Pause webhook'}
          />
        </div>
      </>
    </section>
  );
}
