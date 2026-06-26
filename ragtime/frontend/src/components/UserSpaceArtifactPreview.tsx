import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { ExecuteComponentResponse, UserSpaceLiveDataConnection } from '@/types';
import { api } from '@/api/client';
import {
  USERSPACE_EXEC_BRIDGE,
  USERSPACE_EXEC_MESSAGE_TYPES,
} from '@/utils/userspacePreview/constants';
import { buildUserSpacePreviewSandboxAttribute } from '@/utils/userspacePreview/sandbox';

interface UserSpaceArtifactPreviewProps {
  entryPath: string;
  workspaceFiles: Record<string, string>;
  liveDataConnections?: UserSpaceLiveDataConnection[];
  runtimePreviewUrl?: string;
  runtimePreviewOrigin?: string;
  runtimeAuthorizationPending?: boolean;
  runtimeAvailable?: boolean;
  runtimeError?: string;
  previewInstanceKey?: string;
  workspaceId?: string;
  shareToken?: string;
  ownerUsername?: string;
  shareSlug?: string;
  onExecutionStateChange?: (isExecuting: boolean) => void;
  onNetworkActivityChange?: (isActive: boolean) => void;
  onLiveDataWarningChange?: (warning: string | null) => void;
  onLiveDataTimeout?: (message: string, timeoutSeconds?: number | null) => void;
  onPreviewSessionExpired?: () => void;
  previewNotice?: {
    id: number;
    message: string;
    tone?: 'success' | 'error';
  } | null;
}

export function UserSpaceArtifactPreview({
  runtimePreviewUrl,
  runtimePreviewOrigin,
  runtimeAuthorizationPending,
  runtimeAvailable,
  runtimeError,
  previewInstanceKey,
  workspaceId,
  shareToken,
  ownerUsername,
  shareSlug,
  onExecutionStateChange,
  onNetworkActivityChange,
  onLiveDataWarningChange,
  onLiveDataTimeout,
  onPreviewSessionExpired,
  previewNotice,
}: UserSpaceArtifactPreviewProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [pendingExecutions, setPendingExecutions] = useState(0);
  const [pendingNetworkRequests, setPendingNetworkRequests] = useState(0);
  const [sandboxFlags, setSandboxFlags] = useState<string[]>([]);
  const [sandboxSettingsStatus, setSandboxSettingsStatus] = useState<'loading' | 'ready' | 'error'>(
    'loading',
  );
  const [sandboxBlockedMessage, setSandboxBlockedMessage] = useState<string | null>(null);
  const [liveDataExecutionError, setLiveDataExecutionError] = useState<string | null>(null);
  const [activePreviewNotice, setActivePreviewNotice] = useState<{
    id: number;
    message: string;
    tone?: 'success' | 'error';
  } | null>(null);

  const normalizeOrigin = useCallback((value: string | null | undefined): string => {
    const raw = (value || '').trim();
    if (!raw) return '';
    try {
      const parsed = new URL(raw);
      let port = parsed.port;
      if (
        (parsed.protocol === 'https:' && port === '443') ||
        (parsed.protocol === 'http:' && port === '80')
      ) {
        port = '';
      }
      return `${parsed.protocol}//${parsed.hostname}${port ? `:${port}` : ''}`;
    } catch {
      return raw;
    }
  }, []);

  const expectedPreviewOrigin = useMemo(() => {
    if (runtimePreviewOrigin) {
      return runtimePreviewOrigin;
    }
    if (!runtimePreviewUrl) return null;
    try {
      return new URL(runtimePreviewUrl, window.location.origin).origin;
    } catch {
      return null;
    }
  }, [runtimePreviewOrigin, runtimePreviewUrl]);

  const normalizedExpectedPreviewOrigin = useMemo(
    () => normalizeOrigin(expectedPreviewOrigin),
    [expectedPreviewOrigin, normalizeOrigin],
  );

  const getTimeoutMessage = useCallback(
    (
      result:
        | { error?: string | null; error_kind?: string | null; timeout_seconds?: number | null }
        | null
        | undefined,
    ): string | null => {
      if (!result) return null;
      const rawError = typeof result.error === 'string' ? result.error.trim() : '';
      const timeoutSeconds =
        typeof result.timeout_seconds === 'number' ? result.timeout_seconds : null;
      const looksTimedOut =
        result.error_kind === 'timeout' ||
        /(?:timed out|timeout|statement timeout)/i.test(rawError);
      if (!looksTimedOut) return null;
      const timeoutText = timeoutSeconds != null ? ` of ${timeoutSeconds}s` : '';
      return rawError || `Live data query exceeded the platform-configured timeout${timeoutText}.`;
    },
    [],
  );

  const handleIframeMessage = useCallback(
    async (event: MessageEvent) => {
      const frameWindow = iframeRef.current?.contentWindow;
      if (!frameWindow || event.source !== frameWindow) return;

      const normalizedEventOrigin = normalizeOrigin(event.origin);

      const isExpectedOrigin =
        event.origin === 'null' ||
        (normalizedExpectedPreviewOrigin
          ? normalizedEventOrigin === normalizedExpectedPreviewOrigin
          : false);
      if (!isExpectedOrigin) return;

      if (!event.data || event.data.bridge !== USERSPACE_EXEC_BRIDGE) return;

      if (event.data.type === USERSPACE_EXEC_MESSAGE_TYPES.SANDBOX_BLOCKED) {
        const message = typeof event.data.message === 'string' ? event.data.message.trim() : '';
        setSandboxBlockedMessage(
          message || 'This action was blocked by the current preview sandbox policy.',
        );
        return;
      }

      if (event.data.type === USERSPACE_EXEC_MESSAGE_TYPES.ERROR) {
        const timeoutMessage = getTimeoutMessage({
          error: typeof event.data.error === 'string' ? event.data.error : null,
          error_kind: typeof event.data.error_kind === 'string' ? event.data.error_kind : null,
          timeout_seconds:
            typeof event.data.timeout_seconds === 'number' ? event.data.timeout_seconds : null,
        });
        const warning =
          typeof event.data.error === 'string' && event.data.error.trim()
            ? event.data.error.trim()
            : null;
        if (timeoutMessage) {
          setLiveDataExecutionError(timeoutMessage);
          onLiveDataTimeout?.(
            timeoutMessage,
            typeof event.data.timeout_seconds === 'number' ? event.data.timeout_seconds : null,
          );
        } else {
          setLiveDataExecutionError(null);
        }
        onLiveDataWarningChange?.(warning);
        console.error('[UserSpacePreview] iframe execute error:', {
          component_id: event.data.component_id,
          error: event.data.error,
        });
        return;
      }

      if (event.data.type === USERSPACE_EXEC_MESSAGE_TYPES.SESSION_EXPIRED) {
        onPreviewSessionExpired?.();
        return;
      }

      if (event.data.type === USERSPACE_EXEC_MESSAGE_TYPES.NETWORK_ACTIVITY) {
        const pending =
          typeof event.data.pending === 'number' ? Math.max(0, event.data.pending) : 0;
        setPendingNetworkRequests(pending);
        return;
      }

      if (event.data.type !== USERSPACE_EXEC_MESSAGE_TYPES.EXECUTE) return;

      const { callId, component_id, request } = event.data;
      if (typeof callId !== 'string' || typeof component_id !== 'string') return;

      setPendingExecutions((c) => c + 1);

      const sendResult = (result: unknown) => {
        try {
          frameWindow.postMessage(
            {
              bridge: USERSPACE_EXEC_BRIDGE,
              type: USERSPACE_EXEC_MESSAGE_TYPES.RESULT,
              callId,
              result,
            },
            '*',
          );
        } catch (postErr) {
          // Iframe may have unmounted or navigated mid-execute. Swallow so we
          // always reach the pending-counter decrement; bridge child has its
          // own timeout fallback for unanswered calls.
          console.warn('[UserSpacePreview] failed to deliver execute result:', postErr);
        } finally {
          setPendingExecutions((c) => Math.max(0, c - 1));
        }
      };

      if (!workspaceId && !shareToken && !(ownerUsername && shareSlug)) {
        sendResult({
          rows: [],
          columns: [],
          row_count: 0,
          error: 'No workspace context available',
        });
        return;
      }

      try {
        let result: ExecuteComponentResponse;
        if (shareToken) {
          result = await api.executeUserSpaceSharedComponent(shareToken, { component_id, request });
        } else if (ownerUsername && shareSlug) {
          result = await api.executeUserSpaceSharedComponentBySlug(ownerUsername, shareSlug, {
            component_id,
            request,
          });
        } else {
          result = await api.executeWorkspaceComponent(workspaceId!, {
            component_id,
            request,
          });
        }
        const timeoutMessage = getTimeoutMessage(result);
        if (timeoutMessage) {
          setLiveDataExecutionError(timeoutMessage);
          onLiveDataTimeout?.(timeoutMessage, result.timeout_seconds ?? null);
        } else {
          setLiveDataExecutionError(null);
        }
        onLiveDataWarningChange?.(
          typeof result.error === 'string' && result.error.trim() ? result.error : null,
        );
        sendResult(result);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : String(err);
        const timeoutMessage = getTimeoutMessage({ error: errorMessage });
        if (timeoutMessage) {
          setLiveDataExecutionError(timeoutMessage);
          onLiveDataTimeout?.(timeoutMessage, null);
        } else {
          setLiveDataExecutionError(null);
        }
        onLiveDataWarningChange?.(errorMessage || null);
        sendResult({
          rows: [],
          columns: [],
          row_count: 0,
          error: errorMessage,
        });
      }
    },
    [
      normalizeOrigin,
      normalizedExpectedPreviewOrigin,
      workspaceId,
      shareToken,
      ownerUsername,
      shareSlug,
      getTimeoutMessage,
      onLiveDataWarningChange,
      onLiveDataTimeout,
      onPreviewSessionExpired,
    ],
  );

  useEffect(() => {
    window.addEventListener('message', handleIframeMessage);
    return () => window.removeEventListener('message', handleIframeMessage);
  }, [handleIframeMessage]);

  useEffect(() => {
    onExecutionStateChange?.(pendingExecutions > 0);
  }, [pendingExecutions, onExecutionStateChange]);

  useEffect(() => {
    onNetworkActivityChange?.(pendingNetworkRequests > 0);
  }, [pendingNetworkRequests, onNetworkActivityChange]);

  useEffect(
    () => () => {
      onExecutionStateChange?.(false);
      onNetworkActivityChange?.(false);
    },
    [onExecutionStateChange, onNetworkActivityChange],
  );

  useEffect(() => {
    setPendingExecutions(0);
    setPendingNetworkRequests(0);
    setSandboxBlockedMessage(null);
    setLiveDataExecutionError(null);
    setActivePreviewNotice(null);
  }, [previewInstanceKey, runtimePreviewUrl]);

  useEffect(() => {
    if (!previewNotice) {
      return;
    }

    setActivePreviewNotice(previewNotice);
    const timer = window.setTimeout(() => {
      setActivePreviewNotice((current) => (current?.id === previewNotice.id ? null : current));
    }, 6000);

    return () => {
      window.clearTimeout(timer);
    };
  }, [previewNotice]);

  useEffect(() => {
    let cancelled = false;

    api
      .getUserSpacePreviewSettings()
      .then((response) => {
        if (!cancelled) {
          setSandboxFlags(response.userspace_preview_sandbox_flags);
          setSandboxSettingsStatus('ready');
        }
      })
      .catch(() => {
        if (!cancelled) {
          setSandboxFlags([]);
          setSandboxSettingsStatus('error');
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const unavailableMessage = useMemo(() => {
    if (runtimePreviewUrl) {
      if (runtimeAvailable === false) {
        return (
          runtimeError ||
          'Runtime preview is unavailable. Start or restart the workspace runtime and try again.'
        );
      }
      return null;
    }
    return 'Runtime preview is not available. Start or restart the workspace runtime and try again.';
  }, [runtimeError, runtimeAvailable, runtimePreviewUrl]);

  const sandboxAttribute = useMemo(
    () =>
      buildUserSpacePreviewSandboxAttribute(
        sandboxFlags.includes('allow-top-navigation-by-user-activation')
          ? sandboxFlags
          : [...sandboxFlags, 'allow-top-navigation-by-user-activation'],
      ),
    [sandboxFlags],
  );

  if (runtimeAuthorizationPending) {
    return (
      <div className="userspace-preview-card">
        <h4>Loading preview</h4>
        <p>Authorizing workspace preview access...</p>
      </div>
    );
  }

  if (unavailableMessage) {
    return (
      <div className="userspace-preview-card">
        <h4>Runtime preview unavailable</h4>
        <p>{unavailableMessage}</p>
      </div>
    );
  }

  if (sandboxSettingsStatus === 'loading') {
    return (
      <div className="userspace-preview-card">
        <h4>Loading preview</h4>
        <p>Loading preview sandbox configuration...</p>
      </div>
    );
  }

  if (sandboxSettingsStatus === 'error') {
    return (
      <div className="userspace-preview-card">
        <h4>Runtime preview unavailable</h4>
        <p>Preview sandbox configuration could not be loaded. Refresh and try again.</p>
      </div>
    );
  }

  return (
    <div className="userspace-preview-frame-wrap">
      {activePreviewNotice ? (
        <div
          className={`userspace-preview-exec-notice userspace-preview-exec-notice-${activePreviewNotice.tone ?? 'success'}`}
          role="status"
          aria-live="polite"
        >
          {activePreviewNotice.message}
        </div>
      ) : null}
      {sandboxBlockedMessage ? (
        <div className="userspace-preview-exec-error" role="alert">
          {sandboxBlockedMessage}
        </div>
      ) : null}
      {liveDataExecutionError ? (
        <div className="userspace-preview-exec-error" role="alert">
          {liveDataExecutionError}
        </div>
      ) : null}
      <iframe
        ref={iframeRef}
        key={`${previewInstanceKey ?? ''}:${runtimePreviewUrl ?? ''}`}
        title="Runtime preview"
        className="userspace-preview-frame"
        sandbox={sandboxAttribute}
        src={runtimePreviewUrl}
        onLoad={() => setPendingNetworkRequests(0)}
      />
    </div>
  );
}
