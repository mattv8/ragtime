import { useState } from 'react';
import { api } from '@/api';
import type { IndexInfo } from '@/types';

interface IndexesListProps {
  indexes: IndexInfo[];
  loading: boolean;
  error: string | null;
  onDelete: () => void;
  onToggle?: () => void;
  onDescriptionUpdate?: () => void;
}

interface EditModalProps {
  index: IndexInfo;
  onSave: (name: string, description: string) => Promise<void>;
  onClose: () => void;
  saving: boolean;
}

function EditDescriptionModal({ index, onSave, onClose, saving }: EditModalProps) {
  const [description, setDescription] = useState(index.description);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await onSave(index.name, description);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Edit Description: {index.name}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '12px' }}>
              This description helps the AI understand what knowledge is available in this index.
              It was auto-generated during indexing and can be customized.
            </p>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this index contains for AI context..."
              rows={4}
              style={{ width: '100%', resize: 'vertical', minHeight: '100px' }}
              disabled={saving}
              autoFocus
            />
          </div>
          <div className="modal-footer">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
              disabled={saving}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="btn"
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export function IndexesList({ indexes, loading, error, onDelete, onToggle, onDescriptionUpdate }: IndexesListProps) {
  const [deleting, setDeleting] = useState<string | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);
  const [editingIndex, setEditingIndex] = useState<IndexInfo | null>(null);
  const [savingDescription, setSavingDescription] = useState(false);

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

  const handleSaveDescription = async (name: string, description: string) => {
    setSavingDescription(true);
    try {
      await api.updateIndexDescription(name, description);
      setEditingIndex(null);
      onDescriptionUpdate?.();
    } catch (err) {
      alert(`Error: ${err instanceof Error ? err.message : 'Save failed'}`);
    } finally {
      setSavingDescription(false);
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
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => setEditingIndex(idx)}
              title="Edit description for AI context"
            >
              Edit
            </button>
            <button
              className="btn btn-sm btn-danger"
              onClick={() => handleDelete(idx.name)}
              disabled={deleting === idx.name}
            >
              {deleting === idx.name ? 'Deleting...' : 'Delete'}
            </button>
          </div>
        </div>
      ))}

      {editingIndex && (
        <EditDescriptionModal
          index={editingIndex}
          onSave={handleSaveDescription}
          onClose={() => setEditingIndex(null)}
          saving={savingDescription}
        />
      )}
    </div>
  );
}
