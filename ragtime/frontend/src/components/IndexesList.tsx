import { useState, useCallback, useEffect, type DragEvent, type ChangeEvent } from 'react';
import { api } from '@/api';
import type { IndexInfo, IndexJob } from '@/types';
import { GitIndexWizard } from './GitIndexWizard';

interface IndexesListProps {
  indexes: IndexInfo[];
  loading: boolean;
  error: string | null;
  onDelete: () => void;
  onToggle?: () => void;
  onDescriptionUpdate?: () => void;
  onJobCreated?: () => void;
}

type SourceType = 'upload' | 'git';

/** Extract index name from archive filename (strip extension) */
function getIndexNameFromFile(filename: string): string {
  return filename
    .replace(/\.(zip|tar|tar\.gz|tgz|tar\.bz2|tbz2)$/i, '')
    .replace(/[^a-zA-Z0-9_-]/g, '-')
    .toLowerCase();
}

interface EditModalProps {
  index: IndexInfo;
  onSave: (name: string, description: string) => Promise<void>;
  onClose: () => void;
  saving: boolean;
}

interface WeightModalProps {
  index: IndexInfo;
  onSave: (name: string, weight: number) => Promise<void>;
  onClose: () => void;
  saving: boolean;
}

function EditWeightModal({ index, onSave, onClose, saving }: WeightModalProps) {
  const [weight, setWeight] = useState(index.search_weight);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const clampedWeight = Math.max(0, Math.min(10, weight));
    await onSave(index.name, clampedWeight);
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '400px' }}>
        <div className="modal-header">
          <h3>Search Weight: {index.name}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <p style={{ fontSize: '0.9rem', color: '#888', marginBottom: '16px' }}>
              Search weight influences how the AI prioritizes results from this index relative to others.
            </p>
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '8px' }}>
                Weight: <strong>{weight.toFixed(1)}</strong>
              </label>
              <input
                type="range"
                min="0"
                max="10"
                step="0.1"
                value={weight}
                onChange={(e) => setWeight(parseFloat(e.target.value))}
                style={{ width: '100%' }}
                disabled={saving}
              />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#666' }}>
                <span>0 (lowest)</span>
                <span>1 (default)</span>
                <span>10 (highest)</span>
              </div>
            </div>
            <div style={{ fontSize: '0.85rem', color: '#888', background: 'rgba(0,0,0,0.2)', padding: '12px', borderRadius: '6px' }}>
              <div><strong>1.0</strong> = Default/equal weighting</div>
              <div><strong>&gt;1.0</strong> = Higher priority (preferred by AI)</div>
              <div><strong>&lt;1.0</strong> = Lower priority</div>
              <div><strong>0.0</strong> = Deprioritized (still searchable)</div>
            </div>
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

export function IndexesList({ indexes, loading, error, onDelete, onToggle, onDescriptionUpdate, onJobCreated }: IndexesListProps) {
  const [deleting, setDeleting] = useState<string | null>(null);
  const [toggling, setToggling] = useState<string | null>(null);
  const [editingIndex, setEditingIndex] = useState<IndexInfo | null>(null);
  const [savingDescription, setSavingDescription] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [deleteConfirmName, setDeleteConfirmName] = useState<string | null>(null);

  // Weight modal state
  const [weightEditIndex, setWeightEditIndex] = useState<IndexInfo | null>(null);
  const [savingWeight, setSavingWeight] = useState(false);

  // Reindex modal state
  const [reindexingIndex, setReindexingIndex] = useState<IndexInfo | null>(null);
  const [reindexToken, setReindexToken] = useState('');
  const [reindexing, setReindexing] = useState(false);

  // Create wizard state
  const [showCreateWizard, setShowCreateWizard] = useState(false);
  const [activeSource, setActiveSource] = useState<SourceType>('git');

  // Upload form state
  const [file, setFile] = useState<File | null>(null);
  const [indexName, setIndexName] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState<{ type: 'info' | 'success' | 'error' | null; message: string }>({
    type: null,
    message: '',
  });

  // Auto-fill index name when file is selected
  useEffect(() => {
    if (file) {
      setIndexName(getIndexNameFromFile(file.name));
    }
  }, [file]);

  // Upload form handlers
  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
    }
  }, []);

  const handleUploadSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setUploadStatus({ type: 'error', message: 'Please select a file' });
      return;
    }
    if (!indexName.trim()) {
      setUploadStatus({ type: 'error', message: 'Please enter an index name' });
      return;
    }

    const form = e.currentTarget;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', indexName);
    formData.append('description', '');  // Auto-generate description
    formData.append('file_patterns', (form.elements.namedItem('file_patterns') as HTMLInputElement).value);
    formData.append('exclude_patterns', (form.elements.namedItem('exclude_patterns') as HTMLInputElement).value);
    formData.append('chunk_size', (form.elements.namedItem('chunk_size') as HTMLInputElement).value);
    formData.append('chunk_overlap', (form.elements.namedItem('chunk_overlap') as HTMLInputElement).value);

    setIsLoading(true);
    setUploadStatus({ type: 'info', message: 'Uploading and processing...' });
    setProgress(30);

    try {
      const job: IndexJob = await api.uploadAndIndex(formData);
      setProgress(100);
      setUploadStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      setFile(null);
      setIndexName('');
      setShowCreateWizard(false);
      onJobCreated?.();
    } catch (err) {
      setUploadStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Upload failed'}` });
    } finally {
      setIsLoading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  };

  const handleCancelWizard = () => {
    setShowCreateWizard(false);
    setFile(null);
    setIndexName('');
    setUploadStatus({ type: null, message: '' });
    setProgress(0);
  };

  const handleGitJobCreated = () => {
    setShowCreateWizard(false);
    onJobCreated?.();
  };

  const handleDelete = async (name: string) => {
    // If already showing confirmation for this index, cancel it
    if (deleteConfirmName === name) {
      setDeleteConfirmName(null);
      return;
    }

    // Show inline confirmation
    setDeleteConfirmName(name);
  };

  const confirmDelete = async (name: string) => {
    setDeleteConfirmName(null);
    setDeleting(name);
    try {
      await api.deleteIndex(name);
      onDelete();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Delete failed');
      setTimeout(() => setErrorMessage(null), 5000);
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
      setErrorMessage(err instanceof Error ? err.message : 'Toggle failed');
      setTimeout(() => setErrorMessage(null), 5000);
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
      setErrorMessage(err instanceof Error ? err.message : 'Save failed');
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setSavingDescription(false);
    }
  };

  const handleSaveWeight = async (name: string, weight: number) => {
    setSavingWeight(true);
    try {
      await api.updateIndexWeight(name, weight);
      setWeightEditIndex(null);
      onToggle?.();  // Refresh index list
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to update weight');
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setSavingWeight(false);
    }
  };

  const handleReindex = async () => {
    if (!reindexingIndex) return;

    setReindexing(true);
    try {
      await api.reindexFromGit(reindexingIndex.name, reindexToken || undefined);
      setReindexingIndex(null);
      setReindexToken('');
      onJobCreated?.();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Reindex failed';
      // Check if it's a token error
      if (message.includes('token') || message.includes('401') || message.includes('authentication')) {
        setErrorMessage('Authentication failed. Please check your token and try again.');
      } else {
        setErrorMessage(message);
      }
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setReindexing(false);
    }
  };

  return (
    <div className="card">
      <div className="section-header">
        <h2>Document Indexes</h2>
        <button
          type="button"
          className="btn"
          onClick={() => setShowCreateWizard(!showCreateWizard)}
        >
          {showCreateWizard ? 'Cancel' : 'Create Document Index'}
        </button>
      </div>

      <p className="section-description">
        FAISS-based indexes created from uploaded archives or Git repositories.
        Used for document search and RAG context retrieval.
      </p>

      {/* Create Wizard */}
      {showCreateWizard && (
        <div className="create-wizard">
          <div className="wizard-header">
            <div className="wizard-tabs">
              <button
                type="button"
                className={`wizard-tab ${activeSource === 'git' ? 'active' : ''}`}
                onClick={() => setActiveSource('git')}
              >
                Git Repository
              </button>
              <button
                type="button"
                className={`wizard-tab ${activeSource === 'upload' ? 'active' : ''}`}
                onClick={() => setActiveSource('upload')}
              >
                Upload Archive
              </button>
            </div>
          </div>

          {/* Help text explaining the difference */}
          <p className="field-help" style={{ marginTop: '0.5rem', marginBottom: '1rem', padding: '0.75rem', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '4px' }}>
            {activeSource === 'git' ? (
              <>
                <strong>Git Repository:</strong> Clone from GitHub or GitLab. Supports automatic updates via "Pull & Re-index" to keep your index current with the latest changes.
              </>
            ) : (
              <>
                <strong>Upload Archive:</strong> One-time indexing from a file. To update, you must delete and re-upload. Use Git indexing for content that changes frequently.
              </>
            )}
          </p>

          {activeSource === 'git' ? (
            <GitIndexWizard onJobCreated={handleGitJobCreated} onCancel={handleCancelWizard} />
          ) : (
            <form onSubmit={handleUploadSubmit}>
              {/* File drop area */}
              <div className="form-group">
                <div
                  className={`file-input-wrapper ${isDragOver ? 'dragover' : ''} ${file ? 'has-file' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <div className="icon">↑</div>
                  <div>Drag & drop an archive file here, or click to browse</div>
                  <div style={{ fontSize: '0.85rem', color: '#888', marginTop: 8 }}>
                    Supported: .zip, .tar, .tar.gz, .tar.bz2
                  </div>
                  {file && <div className="file-name">{file.name}</div>}
                  <input
                    type="file"
                    name="file"
                    accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2,.tbz2"
                    onChange={handleFileChange}
                  />
                </div>
              </div>

              {/* Show remaining fields only after file is selected */}
              {file && (
                <>
                  <div className="form-group">
                    <label>Index Name *</label>
                    <input
                      type="text"
                      name="name"
                      value={indexName}
                      onChange={(e) => setIndexName(e.target.value)}
                      placeholder="e.g., my-project, docs-v2"
                      required
                    />
                  </div>

                  <p className="field-help" style={{ marginBottom: '16px' }}>
                    A description will be auto-generated from the indexed content. You can edit it later.
                  </p>

                  <div className="row">
                    <div className="form-group">
                      <label>File Patterns (comma-separated)</label>
                      <input
                        type="text"
                        name="file_patterns"
                        defaultValue="**/*"
                        placeholder="e.g., **/*.py,**/*.md,**/*.xml"
                      />
                    </div>
                    <div className="form-group">
                      <label>Exclude Patterns</label>
                      <input
                        type="text"
                        name="exclude_patterns"
                        defaultValue=""
                        placeholder="e.g., **/node_modules/**,**/__pycache__/**"
                      />
                    </div>
                  </div>

                  <div className="row">
                    <div className="form-group">
                      <label>Chunk Size</label>
                      <input
                        type="number"
                        name="chunk_size"
                        defaultValue={1000}
                        min={100}
                        max={4000}
                      />
                    </div>
                    <div className="form-group">
                      <label>Chunk Overlap</label>
                      <input
                        type="number"
                        name="chunk_overlap"
                        defaultValue={200}
                        min={0}
                        max={1000}
                      />
                    </div>
                  </div>

                  <div className="wizard-actions">
                    <button type="button" className="btn btn-secondary" onClick={handleCancelWizard}>
                      Cancel
                    </button>
                    <button type="submit" className="btn" disabled={isLoading}>
                      {isLoading ? 'Creating...' : 'Create Index'}
                    </button>
                  </div>
                </>
              )}

              {progress > 0 && (
                <div className="progress-bar">
                  <div className="fill" style={{ width: `${progress}%` }} />
                </div>
              )}

              {uploadStatus.type && (
                <div className={`status-message ${uploadStatus.type}`}>{uploadStatus.message}</div>
              )}
            </form>
          )}
        </div>
      )}

      {/* Hide existing indexes when wizard is open */}
      {!showCreateWizard && (
        <>
          {errorMessage && (
            <div className="error-banner">
              {errorMessage}
              <button onClick={() => setErrorMessage(null)}>×</button>
            </div>
          )}

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
            <div className="index-meta-pills">
              <span className="meta-pill documents">{idx.document_count} documents</span>
              <span className="meta-pill size">{idx.size_mb} MB</span>
              {idx.source_type === 'git' && idx.source && (
                <span className="meta-pill git" title={`Git: ${idx.source}${idx.git_branch ? ` (${idx.git_branch})` : ''}`}>
                  Git
                </span>
              )}
              {idx.last_modified && (
                <span className="meta-pill date" title={`Last updated: ${new Date(idx.last_modified).toLocaleString()}`}>
                  {`Updated ${new Date(idx.last_modified).toLocaleString()}`}
                </span>
              )}
              <span
                className={`meta-pill weight ${idx.search_weight !== 1.0 ? 'weight-modified' : ''}`}
                title="Search weight: Higher values prioritize this index in results. Click to edit."
                style={{ cursor: 'pointer' }}
                onClick={() => setWeightEditIndex(idx)}
              >
                Weight: {idx.search_weight.toFixed(1)}
              </span>
              {!idx.enabled && <span className="meta-pill disabled">Excluded from RAG</span>}
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
            {idx.source_type === 'git' && idx.source && (
              <button
                className="btn btn-sm btn-primary"
                onClick={() => setReindexingIndex(idx)}
                title="Pull latest changes from git and re-index"
              >
                Pull &amp; Re-index
              </button>
            )}
            {deleteConfirmName === idx.name ? (
              <>
                <button
                  className="btn btn-sm btn-success"
                  onClick={() => confirmDelete(idx.name)}
                  title="Confirm delete"
                  disabled={deleting === idx.name}
                >
                  Confirm
                </button>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => setDeleteConfirmName(null)}
                  title="Cancel"
                >
                  Cancel
                </button>
              </>
            ) : (
              <button
                className="btn btn-sm btn-danger"
                onClick={() => handleDelete(idx.name)}
                disabled={deleting === idx.name}
              >
                {deleting === idx.name ? 'Deleting...' : 'Delete'}
              </button>
            )}
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

          {weightEditIndex && (
            <EditWeightModal
              index={weightEditIndex}
              onSave={handleSaveWeight}
              onClose={() => setWeightEditIndex(null)}
              saving={savingWeight}
            />
          )}

          {reindexingIndex && (
            <div className="modal-overlay">
              <div className="modal" style={{ maxWidth: '500px' }}>
                <h3>Pull &amp; Re-index</h3>
                <p style={{ marginBottom: '12px', color: 'var(--text-secondary)' }}>
                  Re-index <strong>{reindexingIndex.name}</strong> from Git repository.
                </p>
                <div style={{ marginBottom: '16px', padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', fontSize: '13px' }}>
                  <div><strong>Source:</strong> {reindexingIndex.source}</div>
                  {reindexingIndex.git_branch && (
                    <div><strong>Branch:</strong> {reindexingIndex.git_branch}</div>
                  )}
                </div>
                <div className="form-group">
                  <label htmlFor="reindex-token">
                    Git Token (optional)
                    <span style={{ color: 'var(--text-secondary)', fontWeight: 'normal', marginLeft: '8px' }}>
                      Required for private repos or if token expired
                    </span>
                  </label>
                  <input
                    id="reindex-token"
                    type="password"
                    className="form-input"
                    value={reindexToken}
                    onChange={(e) => setReindexToken(e.target.value)}
                    placeholder="Enter token if needed"
                  />
                </div>
                <div className="modal-actions">
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      setReindexingIndex(null);
                      setReindexToken('');
                    }}
                    disabled={reindexing}
                  >
                    Cancel
                  </button>
                  <button
                    className="btn btn-primary"
                    onClick={handleReindex}
                    disabled={reindexing}
                  >
                    {reindexing ? 'Starting...' : 'Re-index'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
