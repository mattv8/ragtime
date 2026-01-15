import type { AuthStatus } from '@/types';

interface SecurityBannerProps {
  authStatus: AuthStatus | null;
  isAdmin: boolean;
  onNavigateToSettings?: () => void;
}

export function SecurityBanner({ authStatus, isAdmin, onNavigateToSettings }: SecurityBannerProps) {
  // Don't show banner if we don't have auth status yet
  if (!authStatus) return null;

  // Only show to admins - regular users can't fix these issues
  if (!isAdmin) return null;

  // Check if API key is not configured
  const showApiKeyWarning = !authStatus.api_key_configured;

  // Don't show banner if everything is configured
  if (!showApiKeyWarning) return null;

  return (
    <div className="security-banner">
      <div className="security-banner-content">
        <strong>Security Notice:</strong>
        {showApiKeyWarning && (
          <span>
            {' '}API Key is not configured. The OpenAI-compatible API endpoint is unprotected.
            Anyone with network access can use your LLM (which may incur costs) and access tools.
          </span>
        )}
        {isAdmin && onNavigateToSettings && (
          <button
            type="button"
            className="security-banner-link"
            onClick={onNavigateToSettings}
          >
            Configure in Settings
          </button>
        )}
      </div>
    </div>
  );
}
