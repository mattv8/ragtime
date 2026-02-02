interface WarningsBannerProps {
  /** Array of warning messages to display */
  warnings: string[];
}

/**
 * Reusable banner for displaying analysis warnings.
 * Used by GitIndexWizard, UploadForm, and ToolWizard during analysis review.
 */
export function WarningsBanner({ warnings }: WarningsBannerProps) {
  if (warnings.length === 0) return null;

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
      <strong style={{ color: '#fbbf24' }}>Warnings:</strong>
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
