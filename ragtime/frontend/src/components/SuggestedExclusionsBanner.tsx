interface SuggestedExclusionsBannerProps {
  /** Array of suggested exclusion patterns */
  exclusions: string[];
  /** Whether exclusions have been applied */
  applied: boolean;
  /** Callback when "Apply All" is clicked */
  onApply: () => void;
}

/**
 * Reusable banner for displaying suggested exclusion patterns with Apply All functionality.
 * Used by GitIndexWizard, UploadForm, and ToolWizard during analysis review.
 */
export function SuggestedExclusionsBanner({
  exclusions,
  applied,
  onApply,
}: SuggestedExclusionsBannerProps) {
  // Don't render if no exclusions or already applied
  if (exclusions.length === 0) return null;

  if (applied) {
    return (
      <div
        style={{
          background: 'var(--color-success-light)',
          border: '1px solid var(--color-success-border)',
          borderRadius: '8px',
          padding: '12px',
          marginBottom: '16px',
        }}
      >
        <strong style={{ color: 'var(--color-success)' }}>Exclusions applied!</strong>
        <span
          style={{ marginLeft: '8px', color: 'var(--color-text-secondary)', fontSize: '0.9rem' }}
        >
          Click "Re-analyze" to see the updated estimates.
        </span>
      </div>
    );
  }

  return (
    <div
      style={{
        background: 'var(--color-info-light)',
        border: '1px solid var(--color-info-border)',
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '16px',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
        <strong style={{ color: 'var(--color-accent)' }}>Suggested Exclusions:</strong>
        <button
          type="button"
          className="btn btn-sm"
          onClick={onApply}
          style={{ fontSize: '0.8rem', padding: '4px 12px' }}
        >
          Apply All
        </button>
      </div>
      <code
        style={{
          fontSize: '0.85rem',
          color: 'var(--color-text-secondary)',
          wordBreak: 'break-word',
        }}
      >
        {exclusions.join(', ')}
      </code>
    </div>
  );
}
