import { formatBytes } from '@/utils';

interface FileTypeStat {
  extension: string;
  file_count: number;
  total_size_bytes: number;
  estimated_chunks: number;
}

interface FileTypeStatsTableProps {
  /** Array of file type statistics */
  stats: FileTypeStat[];
  /** Maximum number of rows to display (default: 15) */
  maxRows?: number;
  /** Whether to wrap in a details/summary element (default: false) */
  collapsible?: boolean;
  /** Label for the collapsible summary */
  summaryLabel?: string;
}

/**
 * Reusable table component for displaying file type breakdown statistics.
 * Used by GitIndexWizard, UploadForm, and ToolWizard for analysis results.
 */
export function FileTypeStatsTable({
  stats,
  maxRows = 15,
  collapsible = false,
  summaryLabel,
}: FileTypeStatsTableProps) {
  if (stats.length === 0) return null;

  const displayedStats = stats.slice(0, maxRows);
  const remainingCount = stats.length - maxRows;

  const tableContent = (
    <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
      <table style={{ width: '100%', fontSize: '0.85rem' }}>
        <thead>
          <tr style={{ textAlign: 'left', borderBottom: '1px solid #444' }}>
            <th style={{ padding: '4px 8px' }}>Extension</th>
            <th style={{ padding: '4px 8px' }}>Files</th>
            <th style={{ padding: '4px 8px' }}>Size</th>
            <th style={{ padding: '4px 8px' }}>Est. Chunks</th>
          </tr>
        </thead>
        <tbody>
          {displayedStats.map((stat) => (
            <tr key={stat.extension} style={{ borderBottom: '1px solid #333' }}>
              <td style={{ padding: '4px 8px', fontFamily: 'var(--font-mono)' }}>{stat.extension}</td>
              <td style={{ padding: '4px 8px' }}>{stat.file_count}</td>
              <td style={{ padding: '4px 8px' }}>{formatBytes(stat.total_size_bytes)}</td>
              <td style={{ padding: '4px 8px' }}>{stat.estimated_chunks.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {remainingCount > 0 && (
        <div style={{ padding: '8px', color: '#888', fontSize: '0.85rem' }}>
          ... and {remainingCount} more types
        </div>
      )}
    </div>
  );

  if (collapsible) {
    return (
      <details style={{ marginBottom: '16px' }}>
        <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>
          {summaryLabel || `File Type Breakdown (${stats.length} types)`}
        </summary>
        {tableContent}
      </details>
    );
  }

  return tableContent;
}
