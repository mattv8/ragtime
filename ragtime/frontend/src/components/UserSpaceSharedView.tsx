import { useEffect, useState } from 'react';

import { api } from '@/api';

interface UserSpaceSharedViewProps {
  shareToken?: string;
  ownerUsername?: string;
  shareSlug?: string;
}

export function UserSpaceSharedView({ shareToken, ownerUsername, shareSlug }: UserSpaceSharedViewProps) {
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
        const launch = shareToken
          ? await api.launchUserSpaceSharedPreview(shareToken, {
            path: '/',
            parent_origin: window.location.origin,
            prefer_root_proxy: true,
          }, submittedSharePassword)
          : await api.launchUserSpaceSharedPreviewBySlug(ownerUsername as string, shareSlug as string, {
            path: '/',
            parent_origin: window.location.origin,
            prefer_root_proxy: true,
          }, submittedSharePassword);
        if (cancelled) return;
        setPasswordRequired(false);
        setError(null);
        window.location.replace(launch.preview_url);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Failed to load shared preview';
        if (message.toLowerCase().includes('password required') || message.toLowerCase().includes('invalid password')) {
          setPasswordRequired(true);
        } else {
          setPasswordRequired(false);
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
    return <div className="userspace-shared-status">Opening shared preview...</div>;
  }

  if (error) {
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

  return <div className="userspace-shared-status">Opening shared preview...</div>;
}
