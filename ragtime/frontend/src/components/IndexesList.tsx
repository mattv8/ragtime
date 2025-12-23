import { useState } from 'react';
import { api } from '@/api';
import type { IndexInfo } from '@/types';

interface IndexesListProps {
  indexes: IndexInfo[];
  loading: boolean;
  error: string | null;
  onDelete: () => void;
  onToggle?: () => void;
}

export function IndexesList({ indexes, loading, error, onDelete, onToggle }: IndexesListProps) {
  const [deleting, setDeleting] = useState<string | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);

  const handleDelete = async (name: string) => {
    if (!confirm(`Are you sure you want to delete "${name}"?`)) return;

    setDeleting(name);
    try {
      await api.deleteIndex(name);
      onDelete();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Delete failed'}`);
    } finally {
      setDeleting(null);
    }
  };

  const handleToggle = async (name: string, currentEnabled: boolean) => {
    setToggling(name);
    try {
      await api.toggleIndex(name, !currentEnabled);
      onToggle?.();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Toggle failed'}`);
    } finally {
      setToggling(null);
    }
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return null;
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div className="card">
      <div className="section-header">
        <h2>Available Indexes</h2>
      </div>

      {loading && indexes.length === 0 && (
        <div className="empty-state">Loading...</div>
      )}

      {error && (
        <div className="empty-state" style={{ color: '#f87171' }}>
          Error loading indexes: {error}
        </div>
      )}

      {!loading && !error && indexes.length === 0 && (
        <div className="empty-state">No indexes created yet</div>
      )}

      {indexes.map((idx) => (
        <div key={idx.name} className={`index-item ${!idx.enabled ? 'index-disabled' : ''}`}>
          <div className="index-toggle">
            <label className="toggle-switch" title={idx.enabled ? 'Enabled for RAG' : 'Disabled from RAG'}>
              <input
                type="checkbox"
                checked={idx.enabled}
                onChange={() => handleToggle(idx.name, idx.enabled)}
                disabled={toggling === idx.name}
              />
              <span className="toggle-slider"></span>
            </label>
          </div>
          <div className="index-info">
            <h3>{idx.name}</h3>
            <div className="index-meta">
              {idx.document_count} documents • {idx.size_mb} MB
              {idx.last_modified && ` • Updated ${formatDate(idx.last_modified)}`}
              {!idx.enabled && <span className="index-status-disabled"> • Excluded from RAG</span>}
            </div>
          </div>
          <button
            className="btn btn-sm btn-danger"
            onClick={() => handleDelete(idx.name)}
            disabled={deleting === idx.name}
          >
            {deleting === idx.name ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      ))}
    </div>
  );
}
