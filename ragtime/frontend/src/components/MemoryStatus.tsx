import { useState, useEffect, useCallback, useRef } from 'react';
import { HardDrive, Loader, Check, X, Pause, AlertTriangle } from 'lucide-react';
import { api } from '@/api';
import { formatSizeMB } from '@/utils';
import type { HealthResponse, IndexLoadingDetail } from '@/types';

interface MemoryStatusProps {
  /** Whether to actively poll for updates */
  active?: boolean;
  /** Polling interval in ms (default 500ms during loading, 5000ms otherwise) */
  pollInterval?: number;
  /** Callback when loading completes */
  onLoadingComplete?: () => void;
}

/**
 * Real-time memory status display showing:
 * - Current process memory usage
 * - Index loading progress (during startup)
 * - Per-index loading details
 */
export function MemoryStatus({ active = true, pollInterval, onLoadingComplete }: MemoryStatusProps) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const wasLoadingRef = useRef(false);

  const fetchHealth = useCallback(async () => {
    try {
      const data = await api.getHealth();
      setHealth(data);
      setError(null);

      // Detect loading completion
      if (wasLoadingRef.current && data.indexes_ready && !data.indexes_loading) {
        wasLoadingRef.current = false;
        onLoadingComplete?.();
      }
      if (data.indexes_loading) {
        wasLoadingRef.current = true;
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch health');
    }
  }, [onLoadingComplete]);

  useEffect(() => {
    if (!active) return;

    // Initial fetch
    fetchHealth();

    // Determine poll interval: faster during loading
    const interval = pollInterval ?? (health?.indexes_loading ? 500 : 5000);

    const timer = setInterval(fetchHealth, interval);
    return () => clearInterval(timer);
  }, [active, fetchHealth, health?.indexes_loading, pollInterval]);

  if (!health) {
    if (error) {
      return (
        <div className="memory-status memory-status-error" title={error}>
          <span className="memory-status-icon"><AlertTriangle size={14} /></span>
          <span className="memory-status-text">Error</span>
        </div>
      );
    }
    // Show loading state while fetching initial health
    return (
      <div className="memory-status">
        <span className="memory-status-toggle">
          <span className="memory-status-icon"><Loader size={14} className="spinning" /></span>
          <span className="memory-status-text">...</span>
        </span>
      </div>
    );
  }

  const { memory, indexes_loading, indexes_ready, indexes_loaded_count, indexes_total, index_details, sequential_loading, loading_index } = health;

  if (!memory) return null;

  const isLoading = indexes_loading && !indexes_ready;
  const loadingProgress = indexes_total && indexes_total > 0
    ? Math.round((indexes_loaded_count ?? 0) / indexes_total * 100)
    : 0;

  // Sort index details: loading first, then pending, then loaded, then errors
  const sortedDetails = [...(index_details || [])].sort((a, b) => {
    const order: Record<string, number> = { loading: 0, pending: 1, loaded: 2, error: 3 };
    return (order[a.status] ?? 4) - (order[b.status] ?? 4);
  });

  return (
    <div className={`memory-status ${isLoading ? 'memory-status-loading' : ''}`}>
      <button
        type="button"
        className="memory-status-toggle"
        onClick={() => setExpanded(!expanded)}
        title={`Process RAM: ${formatSizeMB(memory.rss_mb)} / System: ${formatSizeMB(memory.total_mb)}`}
      >
        <span className="memory-status-icon">
          {isLoading ? <Loader size={14} className="spinning" /> : <HardDrive size={14} />}
        </span>
        <span className="memory-status-text">
          {formatSizeMB(memory.rss_mb)}
          {isLoading && indexes_total && indexes_total > 0 && (
            <span className="memory-status-progress">
              {' '}({indexes_loaded_count}/{indexes_total})
            </span>
          )}
        </span>
        {isLoading && (
          <span className="memory-status-bar">
            <span
              className="memory-status-bar-fill"
              style={{ width: `${loadingProgress}%` }}
            />
          </span>
        )}
      </button>

      {expanded && (
        <div className="memory-status-dropdown">
          <div className="memory-status-section">
            <h4>Process Memory</h4>
            <div className="memory-status-row">
              <span>RAM Used</span>
              <span className="memory-status-value">{formatSizeMB(memory.rss_mb)}</span>
            </div>
            <div className="memory-status-row">
              <span>System RAM</span>
              <span className="memory-status-value">
                {formatSizeMB(memory.total_mb - memory.available_mb)} / {formatSizeMB(memory.total_mb)}
              </span>
            </div>
            <div className="memory-status-row">
              <span>Available</span>
              <span className="memory-status-value">{formatSizeMB(memory.available_mb)}</span>
            </div>
          </div>

          {(isLoading || sortedDetails.length > 0) && (
            <div className="memory-status-section">
              <h4>
                {isLoading ? 'Index Loading' : 'Loaded Indexes'}
                {sequential_loading && <span className="memory-status-badge">Sequential</span>}
                {!sequential_loading && isLoading && <span className="memory-status-badge">Parallel</span>}
              </h4>

              {isLoading && loading_index && (
                <div className="memory-status-current">
                  Loading: <strong>{loading_index}</strong>
                </div>
              )}

              {sortedDetails.map((idx) => (
                <IndexLoadingItem key={idx.name} detail={idx} isLoading={isLoading} />
              ))}

              {!isLoading && indexes_ready && sortedDetails.length === 0 && (
                <div className="memory-status-row muted">No indexes configured</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

interface IndexLoadingItemProps {
  detail: IndexLoadingDetail;
  isLoading?: boolean;
}

function IndexLoadingItem({ detail, isLoading }: IndexLoadingItemProps) {
  const statusIcon: Record<string, React.ReactNode> = {
    pending: <Pause size={12} />,
    loading: <Loader size={12} className="spinning" />,
    loaded: <Check size={12} />,
    error: <X size={12} />,
  };

  const statusClass = `memory-status-index memory-status-index-${detail.status}`;

  return (
    <div className={statusClass}>
      <span className="memory-status-index-icon">{statusIcon[detail.status] || '?'}</span>
      <span className="memory-status-index-name">{detail.name}</span>
      {detail.size_mb !== null && detail.size_mb !== undefined && (
        <span className="memory-status-index-size">{formatSizeMB(detail.size_mb)}</span>
      )}
      {detail.chunk_count !== null && detail.chunk_count !== undefined && !isLoading && (
        <span className="memory-status-index-chunks">{detail.chunk_count.toLocaleString()} chunks</span>
      )}
      {detail.load_time_seconds !== null && detail.load_time_seconds !== undefined && isLoading && (
        <span className="memory-status-index-time">{detail.load_time_seconds.toFixed(1)}s</span>
      )}
      {detail.error && (
        <span className="memory-status-index-error" title={detail.error}>Error</span>
      )}
    </div>
  );
}
