import type { IndexAnalysisResult } from '@/types';

interface AnalysisStatsProps {
  result: IndexAnalysisResult;
  /** Called when user clicks the sequential loading link */
  onNavigateToSettings?: () => void;
}

/**
 * Shared component for displaying index analysis statistics.
 * Used by GitIndexWizard and UploadForm.
 */
export function AnalysisStats({ result, onNavigateToSettings }: AnalysisStatsProps) {
  const formatMemory = (mb: number): string => {
    if (mb < 1024) return `${mb.toLocaleString()} MB`;
    return `${(mb / 1024).toFixed(1)} GB`;
  };

  // Build array of stat items to render
  const stats: Array<{ value: string; label: string; color?: string }> = [
    { value: result.total_files.toLocaleString(), label: 'Files' },
    { value: `${result.total_size_mb.toLocaleString()} MB`, label: 'Source Size' },
    { value: result.estimated_chunks.toLocaleString(), label: 'Est. Chunks' },
    {
      value: `${result.estimated_index_size_mb.toLocaleString()} MB`,
      label: 'Est. Index Size',
      color: result.estimated_index_size_mb > 500 ? '#f87171' : undefined,
    },
  ];

  // Add commit history if available
  if (result.commit_history && result.commit_history.total_commits > 0) {
    stats.push({
      value: result.commit_history.total_commits.toLocaleString(),
      label: 'Total Commits',
    });
  }

  // Add memory estimates if available
  if (result.memory_estimate) {
    stats.push({
      value: formatMemory(result.memory_estimate.steady_memory_mb),
      label: 'Steady-State RAM',
      color: '#22c55e',
    });
    stats.push({
      value: formatMemory(result.memory_estimate.peak_memory_mb),
      label: 'Peak RAM (loading)',
      color: '#f59e0b',
    });
    if (result.total_memory_with_existing_mb) {
      stats.push({
        value: formatMemory(result.total_memory_with_existing_mb),
        label: 'Total w/ Existing',
        color: '#60a5fa',
      });
    }
  }

  // Split stats into two rows for balanced display
  const midpoint = Math.ceil(stats.length / 2);
  const topRow = stats.slice(0, midpoint);
  const bottomRow = stats.slice(midpoint);

  const statBoxStyle: React.CSSProperties = {
    background: 'var(--bg-tertiary)',
    padding: '12px',
    borderRadius: '8px',
    textAlign: 'center',
    flex: '1 1 0',
    minWidth: '80px',
  };

  return (
    <div style={{ marginBottom: '16px' }}>
      {/* Stats in two balanced rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '12px' }}>
        <div style={{ display: 'flex', gap: '12px' }}>
          {topRow.map((stat, i) => (
            <div key={i} style={statBoxStyle}>
              <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: stat.color || 'var(--accent)' }}>
                {stat.value}
              </div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{stat.label}</div>
            </div>
          ))}
        </div>
        {bottomRow.length > 0 && (
          <div style={{ display: 'flex', gap: '12px' }}>
            {bottomRow.map((stat, i) => (
              <div key={i} style={statBoxStyle}>
                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: stat.color || 'var(--accent)' }}>
                  {stat.value}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{stat.label}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Memory note with dimension info */}
      {result.memory_estimate && (
        <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '8px' }}>
          RAM estimates based on {result.memory_estimate.embedding_dimension}-dim embeddings.
          Peak memory occurs during index loading.{' '}
          {onNavigateToSettings ? (
            <a
              href="#settings"
              onClick={(e) => {
                e.preventDefault();
                onNavigateToSettings();
              }}
              style={{ color: '#60a5fa', textDecoration: 'underline', cursor: 'pointer' }}
            >
              Use sequential loading
            </a>
          ) : (
            <span>Use sequential loading in Settings</span>
          )}{' '}
          to reduce peak.
        </div>
      )}
    </div>
  );
}
