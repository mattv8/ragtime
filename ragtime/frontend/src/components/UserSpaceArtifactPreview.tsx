import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { UserSpaceLiveDataConnection } from '@/types';
import { api } from '@/api/client';
import {
  USERSPACE_EXEC_BRIDGE,
  USERSPACE_EXEC_MESSAGE_TYPES,
} from '@/utils/userspacePreview';

interface UserSpaceArtifactPreviewProps {
  entryPath: string;
  workspaceFiles: Record<string, string>;
  liveDataConnections?: UserSpaceLiveDataConnection[];
  runtimePreviewUrl?: string;
  runtimeAvailable?: boolean;
  runtimeError?: string;
  previewInstanceKey?: string;
  workspaceId?: string;
  shareToken?: string;
  ownerUsername?: string;
  shareSlug?: string;
  sharePassword?: string;
  onExecutionStateChange?: (isExecuting: boolean) => void;
}

export function UserSpaceArtifactPreview({
  runtimePreviewUrl,
  runtimeAvailable,
  runtimeError,
  previewInstanceKey,
  workspaceId,
  shareToken,
  ownerUsername,
  shareSlug,
  sharePassword,
  onExecutionStateChange,
}: UserSpaceArtifactPreviewProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [pendingExecutions, setPendingExecutions] = useState(0);

  const handleIframeMessage = useCallback(
    async (event: MessageEvent) => {
      const frameWindow = iframeRef.current?.contentWindow;
      if (!frameWindow || event.source !== frameWindow) return;

      const isExpectedOrigin =
        event.origin === 'null' || event.origin === window.location.origin;
      if (!isExpectedOrigin) return;

      if (!event.data || event.data.bridge !== USERSPACE_EXEC_BRIDGE) return;

      if (event.data.type === USERSPACE_EXEC_MESSAGE_TYPES.ERROR) {
        console.error('[UserSpacePreview] iframe execute error:', {
          component_id: event.data.component_id,
          error: event.data.error,
        });
        return;
      }

      if (event.data.type !== USERSPACE_EXEC_MESSAGE_TYPES.EXECUTE) return;

      const { callId, component_id, request } = event.data;
      if (typeof callId !== 'string' || typeof component_id !== 'string') return;

      setPendingExecutions((c) => c + 1);

      const sendResult = (result: unknown) => {
        frameWindow.postMessage(
          {
            bridge: USERSPACE_EXEC_BRIDGE,
            type: USERSPACE_EXEC_MESSAGE_TYPES.RESULT,
            callId,
            result,
          },
          '*',
        );
        setPendingExecutions((c) => Math.max(0, c - 1));
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
        let result;
        if (shareToken) {
          result = await api.executeUserSpaceSharedComponent(
            shareToken,
            { component_id, request },
            sharePassword,
          );
        } else if (ownerUsername && shareSlug) {
          result = await api.executeUserSpaceSharedComponentBySlug(
            ownerUsername,
            shareSlug,
            { component_id, request },
            sharePassword,
          );
        } else {
          result = await api.executeWorkspaceComponent(workspaceId!, {
            component_id,
            request,
          });
        }
        sendResult(result);
      } catch (err) {
        sendResult({
          rows: [],
          columns: [],
          row_count: 0,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    },
    [workspaceId, shareToken, ownerUsername, shareSlug, sharePassword],
  );

  useEffect(() => {
    window.addEventListener('message', handleIframeMessage);
    return () => window.removeEventListener('message', handleIframeMessage);
  }, [handleIframeMessage]);

  useEffect(() => {
    onExecutionStateChange?.(pendingExecutions > 0);
  }, [pendingExecutions, onExecutionStateChange]);

  useEffect(() => {
    setPendingExecutions(0);
  }, [previewInstanceKey, runtimePreviewUrl]);

  const unavailableMessage = useMemo(() => {
    if (runtimePreviewUrl) {
      if (runtimeAvailable === false) {
        return runtimeError || 'Runtime preview is unavailable. Start or restart the workspace runtime and try again.';
      }
      return null;
    }
    return 'Runtime preview is not available. Start or restart the workspace runtime and try again.';
  }, [runtimeError, runtimeAvailable, runtimePreviewUrl]);

  if (unavailableMessage) {
    return (
      <div className="userspace-preview-card">
        <h4>Runtime preview unavailable</h4>
        <p>{unavailableMessage}</p>
      </div>
    );
  }

  return (
    <div className="userspace-preview-frame-wrap">
      <iframe
        ref={iframeRef}
        key={`${previewInstanceKey ?? ''}:${runtimePreviewUrl ?? ''}`}
        title="Runtime preview"
        className="userspace-preview-frame"
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        src={runtimePreviewUrl}
      />
    </div>
  );
}
