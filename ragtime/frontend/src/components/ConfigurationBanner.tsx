import { useState, useEffect } from 'react';
import type { ConfigurationWarning } from '@/types';

const DISMISS_KEY = 'ragtime_config_banner_dismissed';

interface ConfigurationBannerProps {
  warnings: ConfigurationWarning[];
  isAdmin: boolean;
  onNavigateToSettings?: () => void;
}

export function ConfigurationBanner({ warnings, isAdmin, onNavigateToSettings }: ConfigurationBannerProps) {
  const [dismissed, setDismissed] = useState(false);

  // Check sessionStorage on mount
  useEffect(() => {
    const wasDismissed = sessionStorage.getItem(DISMISS_KEY) === 'true';
    setDismissed(wasDismissed);
  }, []);

  // Only show to admins - regular users can't fix these issues
  if (!isAdmin) return null;

  // Filter to only show warnings and errors (not info level)
  const significantWarnings = warnings.filter(w => w.level === 'warning' || w.level === 'error');

  // Don't show banner if no significant warnings or dismissed
  if (significantWarnings.length === 0) return null;
  if (dismissed) return null;

  const handleDismiss = () => {
    sessionStorage.setItem(DISMISS_KEY, 'true');
    setDismissed(true);
  };

  // Get the most severe warning level
  const hasError = significantWarnings.some(w => w.level === 'error');
  const bannerClass = hasError ? 'config-banner config-banner-error' : 'config-banner config-banner-warning';

  return (
    <div className={bannerClass}>
      <div className="config-banner-content">
        <strong>Performance:</strong>
        <span>
          {significantWarnings.map((warning, idx) => (
            <span key={idx}>
              {idx > 0 && ' '}
              {warning.message}
              {warning.recommendation && (
                <span className="config-recommendation"> {warning.recommendation}</span>
              )}
            </span>
          ))}
        </span>
        <div className="config-banner-actions">
          {isAdmin && onNavigateToSettings && (
            <button
              type="button"
              className="config-banner-link"
              onClick={onNavigateToSettings}
            >
              View in Settings
            </button>
          )}
          <button
            type="button"
            className="config-banner-dismiss"
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
