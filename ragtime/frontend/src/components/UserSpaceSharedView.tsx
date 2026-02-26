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
  const [sharePasswordDraft, setSharePasswordDraft] = useState('');
  const [submittedSharePassword, setSubmittedSharePassword] = useState<string | undefined>(undefined);
  const [passwordRequired, setPasswordRequired] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      try {
        const data = shareToken
          ? await api.getUserSpaceSharedPreview(shareToken, submittedSharePassword)
          : await api.getUserSpaceSharedPreviewBySlug(ownerUsername as string, shareSlug as string, submittedSharePassword);
        if (cancelled) return;
        setPreviewData(data);
        setPasswordRequired(false);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Failed to load shared preview';
        if (message.toLowerCase().includes('password required') || message.toLowerCase().includes('invalid password')) {
          setPasswordRequired(true);
        }
        setError(message);
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
  }, [ownerUsername, shareSlug, shareToken, submittedSharePassword]);

  if (loading) {
    return <div className="userspace-shared-status">Loading shared dashboard...</div>;
  }

  if (error || !previewData) {
    if (passwordRequired) {
      return (
        <div className="userspace-shared-layout">
          <div className="userspace-shared-status">
            <div className="userspace-share-access-row" style={{ maxWidth: 420, margin: '0 auto', textAlign: 'left' }}>
              <label htmlFor="userspace-shared-password" className="userspace-share-label">Enter share password</label>
              <input
                id="userspace-shared-password"
                type="password"
                value={sharePasswordDraft}
                onChange={(event) => setSharePasswordDraft(event.target.value)}
                autoComplete="current-password"
              />
              <div className="userspace-toolbar-actions" style={{ marginTop: 8 }}>
                <button
                  type="button"
                  className="userspace-button"
                  onClick={() => setSubmittedSharePassword(sharePasswordDraft || undefined)}
                  disabled={loading}
                >
                  Load Preview
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }
    return <div className="userspace-shared-status userspace-error">{error || 'Shared dashboard not found'}</div>;
  }

  return (
    <div className="userspace-shared-layout">
      <UserSpaceArtifactPreview
        entryPath={previewData.entry_path}
        workspaceFiles={previewData.workspace_files}
        liveDataConnections={previewData.live_data_connections ?? []}
        runtimePreviewUrl={
          !submittedSharePassword
            ? shareToken
              ? api.getUserSpaceSharedTokenPreviewProxyUrl(shareToken)
              : ownerUsername && shareSlug
                ? api.getUserSpaceSharedPreviewProxyUrl(ownerUsername, shareSlug)
                : undefined
            : undefined
        }
        previewInstanceKey={shareToken || `${ownerUsername}/${shareSlug}`}
        workspaceId={previewData.workspace_id}
        shareToken={shareToken}
        ownerUsername={ownerUsername}
        shareSlug={shareSlug}
        sharePassword={submittedSharePassword}
      />
    </div>
  );
}
