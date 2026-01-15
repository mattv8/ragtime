import { useState, useEffect } from 'react';
import type { AuthStatus } from '@/types';

const DISMISS_KEY = 'ragtime_security_banner_dismissed';

interface SecurityBannerProps {
  authStatus: AuthStatus | null;
  isAdmin: boolean;
  onNavigateToSettings?: () => void;
}

export function SecurityBanner({ authStatus, isAdmin, onNavigateToSettings }: SecurityBannerProps) {
  const [dismissed, setDismissed] = useState(false);

  // Check sessionStorage on mount
  useEffect(() => {
    const wasDismissed = sessionStorage.getItem(DISMISS_KEY) === 'true';
    setDismissed(wasDismissed);
  }, []);

  // Don't show banner if we don't have auth status yet
  if (!authStatus) return null;

  // Only show to admins - regular users can't fix these issues
  if (!isAdmin) return null;

  // Check security issues
  const showApiKeyWarning = !authStatus.api_key_configured;
  const showCorsWarning = authStatus.allowed_origins_open;

  // Don't show banner if everything is configured or dismissed
  if (!showApiKeyWarning && !showCorsWarning) return null;
  if (dismissed) return null;

  const handleDismiss = () => {
    sessionStorage.setItem(DISMISS_KEY, 'true');
    setDismissed(true);
  };

  const isHttp = window.location.protocol === 'http:';

  return (
    <div className="security-banner">
      <div className="security-banner-content">
        <strong>Security:</strong>
        <span>
          {showApiKeyWarning && (
            <span>
              {' '}The API endpoint accepts an API Key for authentication (set via <code>API_KEY</code> environment variable).
              Without an API key, anyone with network access can use your LLM and tools.
            </span>
          )}
          {showCorsWarning && (
            <span>
              {' '}<code>ALLOWED_ORIGINS=*</code> allows requests from any website.
              Consider restricting to specific domains.
            </span>
          )}
          {isHttp && (
            <span>
              {' '}You are accessing over HTTP - credentials will be transmitted in plaintext.
              Consider <code>ENABLE_HTTPS=true</code> or using a reverse proxy.
            </span>
          )}
        </span>
        <div className="security-banner-actions">
          {isAdmin && onNavigateToSettings && (
            <button
              type="button"
              className="security-banner-link"
              onClick={onNavigateToSettings}
            >
              View in Settings
            </button>
          )}
          <button
            type="button"
            className="security-banner-dismiss"
            onClick={handleDismiss}
            title="Dismiss for this session"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}
