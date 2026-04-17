import { useEffect, useState } from 'react';

interface WarningsBannerProps {
  /** Array of warning messages to display */
  warnings: string[];
  /** Optional custom title */
  title?: string;
  /** Hide banner even when warnings exist */
  hidden?: boolean;
  /** Optional session storage key for dismissible warnings */
  dismissKey?: string;
  /** Render warnings as a compact summary instead of a bullet list */
  compact?: boolean;
}

/**
 * Reusable banner for displaying analysis warnings.
 * Used by GitIndexWizard, UploadForm, and ToolWizard during analysis review.
 */
export function WarningsBanner({
  warnings,
  title = 'Warnings:',
  hidden = false,
  dismissKey,
  compact = false,
}: WarningsBannerProps) {
  const [dismissed, setDismissed] = useState(false);

  useEffect(() => {
    if (!dismissKey) {
      setDismissed(false);
      return;
    }
    setDismissed(sessionStorage.getItem(dismissKey) === 'true');
  }, [dismissKey]);

  if (warnings.length === 0) return null;
  if (hidden || dismissed) return null;

  const summary = warnings.join(' ');
  const bannerClassName = compact ? 'warnings-banner warnings-banner-compact' : 'warnings-banner';

  const handleDismiss = () => {
    if (dismissKey) {
      sessionStorage.setItem(dismissKey, 'true');
    }
    setDismissed(true);
  };

  return (
    <div className={bannerClassName}>
      <div className="warnings-banner-content">
        <strong className="warnings-banner-title">{title}</strong>
        {compact ? <span className="warnings-banner-summary">{summary}</span> : null}
        {dismissKey ? (
          <button
            type="button"
            onClick={handleDismiss}
            className="warnings-banner-dismiss"
          >
            Dismiss
          </button>
        ) : null}
      </div>
      {!compact ? (
        <ul className="warnings-banner-list">
          {warnings.map((warning, i) => (
            <li key={i} className="warnings-banner-item">
              {warning}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
