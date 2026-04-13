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
}

/**
 * Reusable banner for displaying analysis warnings.
 * Used by GitIndexWizard, UploadForm, and ToolWizard during analysis review.
 */
export function WarningsBanner({ warnings, title = 'Warnings:', hidden = false, dismissKey }: WarningsBannerProps) {
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

  const handleDismiss = () => {
    if (dismissKey) {
      sessionStorage.setItem(dismissKey, 'true');
    }
    setDismissed(true);
  };

  return (
    <div
      style={{
        background: 'rgba(251, 191, 36, 0.1)',
        border: '1px solid rgba(251, 191, 36, 0.3)',
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '16px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
        }}
      >
        <strong style={{ color: '#fbbf24' }}>{title}</strong>
        {dismissKey ? (
          <button
            type="button"
            onClick={handleDismiss}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#fbbf24',
              cursor: 'pointer',
              fontSize: '0.9rem',
              padding: 0,
            }}
          >
            Dismiss
          </button>
        ) : null}
      </div>
      <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
        {warnings.map((warning, i) => (
          <li key={i} style={{ color: '#fbbf24', fontSize: '0.9rem' }}>
            {warning}
          </li>
        ))}
      </ul>
    </div>
  );
}
