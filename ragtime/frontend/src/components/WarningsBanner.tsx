import { useEffect, useState } from 'react';

interface WarningsBannerProps {
  /** Array of warning messages to display */
  warnings: string[];
  /** Optional custom title */
  title?: string;
  /** Hide banner even when warnings exist */
  hidden?: boolean;
  /** Optional storage key for dismissible warnings */
  dismissKey?: string;
  /** Persist dismissal across sessions (localStorage) instead of per-session (sessionStorage) */
  persistDismiss?: boolean;
  /** Render warnings as a compact summary instead of a bullet list */
  compact?: boolean;
}

function readDismissed(dismissKey: string | undefined, persistDismiss: boolean): boolean {
  if (!dismissKey || typeof window === 'undefined') return false;

  try {
    const storage = persistDismiss ? window.localStorage : window.sessionStorage;
    return storage.getItem(dismissKey) === 'true';
  } catch {
    return false;
  }
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
  persistDismiss = false,
  compact = false,
}: WarningsBannerProps) {
  const [dismissed, setDismissed] = useState<boolean>(() =>
    readDismissed(dismissKey, persistDismiss),
  );

  useEffect(() => {
    setDismissed(readDismissed(dismissKey, persistDismiss));
  }, [dismissKey, persistDismiss]);

  if (warnings.length === 0) return null;
  if (hidden || dismissed) return null;

  const summary = warnings.join(' ');
  const bannerClassName = compact ? 'warnings-banner warnings-banner-compact' : 'warnings-banner';

  const handleDismiss = () => {
    if (dismissKey && typeof window !== 'undefined') {
      try {
        const storage = persistDismiss ? window.localStorage : window.sessionStorage;
        storage.setItem(dismissKey, 'true');
      } catch {
        // Ignore unavailable storage.
      }
    }
    setDismissed(true);
  };

  return (
    <div className={bannerClassName}>
      <div className="warnings-banner-content">
        <strong className="warnings-banner-title">{title}</strong>
        {compact ? <span className="warnings-banner-summary">{summary}</span> : null}
        {dismissKey ? (
          <button type="button" onClick={handleDismiss} className="warnings-banner-dismiss">
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
