import { useEffect, useMemo } from 'react';

import type { UserSpaceLiveDataConnection } from '@/types';

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
  onExecutionStateChange,
}: UserSpaceArtifactPreviewProps) {
  useEffect(() => {
    onExecutionStateChange?.(false);
  }, [onExecutionStateChange, runtimePreviewUrl, previewInstanceKey]);

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
        key={`${previewInstanceKey ?? ''}:${runtimePreviewUrl ?? ''}`}
        title="Runtime preview"
        className="userspace-preview-frame"
        sandbox="allow-scripts allow-same-origin allow-forms allow-popups"
        src={runtimePreviewUrl}
      />
    </div>
  );
}
