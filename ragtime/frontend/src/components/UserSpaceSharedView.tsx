import { useEffect, useState } from 'react';

import { api } from '@/api';
import type { UserSpaceSharedPreviewResponse } from '@/types';

import { UserSpaceArtifactPreview } from './UserSpaceArtifactPreview';

interface UserSpaceSharedViewProps {
  shareToken?: string;
  ownerUsername?: string;
  shareSlug?: string;
}

export function UserSpaceSharedView({ shareToken, ownerUsername, shareSlug }: UserSpaceSharedViewProps) {
  const [previewData, setPreviewData] = useState<UserSpaceSharedPreviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const data = shareToken
          ? await api.getUserSpaceSharedPreview(shareToken)
          : await api.getUserSpaceSharedPreviewBySlug(ownerUsername as string, shareSlug as string);
        if (cancelled) return;
        setPreviewData(data);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : 'Failed to load shared preview');
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    load();
    return () => {
      cancelled = true;
    };
  }, [ownerUsername, shareSlug, shareToken]);

  if (loading) {
    return <div className="userspace-shared-status">Loading shared dashboard...</div>;
  }

  if (error || !previewData) {
    return <div className="userspace-shared-status userspace-error">{error || 'Shared dashboard not found'}</div>;
  }

  return (
    <div className="userspace-shared-layout">
      <UserSpaceArtifactPreview
        entryPath={previewData.entry_path}
        workspaceFiles={previewData.workspace_files}
        liveDataConnections={previewData.live_data_connections ?? []}
        previewInstanceKey={shareToken || `${ownerUsername}/${shareSlug}`}
        workspaceId={previewData.workspace_id}
        shareToken={shareToken}
        ownerUsername={ownerUsername}
        shareSlug={shareSlug}
      />
    </div>
  );
}
