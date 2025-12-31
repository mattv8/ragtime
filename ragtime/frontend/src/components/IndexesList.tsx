import { useState, useCallback, useEffect, type DragEvent, type ChangeEvent } from 'react';
import { api } from '@/api';
import type { IndexInfo, IndexJob } from '@/types';

// Confirmation modal state
interface ConfirmationState {
  message: string;
  onConfirm: () => void;
  onCancel?: () => void;
}

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
  const [_confirmation, _setConfirmation] = useState<ConfirmationState | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [deleteConfirmName, setDeleteConfirmName] = useState<string | null>(null);

  // Create wizard state
  const [showCreateWizard, setShowCreateWizard] = useState(false);
  const [activeSource, setActiveSource] = useState<SourceType>('upload');

  // Upload form state
  const [file, setFile] = useState<File | null>(null);
  const [indexName, setIndexName] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<{ type: 'info' | 'success' | 'error' | null; message: string }>({
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
      setStatus({ type: 'error', message: 'Please select a file' });
      return;
    }
    if (!indexName.trim()) {
      setStatus({ type: 'error', message: 'Please enter an index name' });
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
    setStatus({ type: 'info', message: 'Uploading and processing...' });
    setProgress(30);

    try {
      const job: IndexJob = await api.uploadAndIndex(formData);
      setProgress(100);
      setStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      setFile(null);
      setIndexName('');
      setShowCreateWizard(false);
      onJobCreated?.();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Upload failed'}` });
    } finally {
      setIsLoading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  };

  const handleGitSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;

    const name = (form.elements.namedItem('name') as HTMLInputElement).value;
    const gitDescription = (form.elements.namedItem('description') as HTMLTextAreaElement).value;
    const gitUrl = (form.elements.namedItem('git_url') as HTMLInputElement).value;
    const gitBranch = (form.elements.namedItem('git_branch') as HTMLInputElement).value;
    const filePatterns = (form.elements.namedItem('file_patterns') as HTMLInputElement).value;
    const excludePatterns = (form.elements.namedItem('exclude_patterns') as HTMLInputElement).value;

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Starting git clone...' });

    try {
      const job: IndexJob = await api.indexFromGit({
        name,
        git_url: gitUrl,
        git_branch: gitBranch,
        config: {
          name,
          description: gitDescription,
          file_patterns: filePatterns.split(',').map((s) => s.trim()),
          exclude_patterns: excludePatterns.split(',').map((s) => s.trim()),
        },
      });
      setStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      setShowCreateWizard(false);
      onJobCreated?.();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Request failed'}` });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancelWizard = () => {
    setShowCreateWizard(false);
    setFile(null);
    setIndexName('');
    setStatus({ type: null, message: '' });
    setProgress(0);
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
                className={`wizard-tab ${activeSource === 'upload' ? 'active' : ''}`}
                onClick={() => setActiveSource('upload')}
              >
                Upload Archive
              </button>
              <button
                type="button"
                className={`wizard-tab ${activeSource === 'git' ? 'active' : ''}`}
                onClick={() => setActiveSource('git')}
              >
                Git Repository
              </button>
            </div>
          </div>

          {activeSource === 'upload' ? (
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
                      placeholder="e.g., odoo-17, my-codebase"
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
                        defaultValue="**/*.py,**/*.md,**/*.xml,**/*.rst"
                      />
                    </div>
                    <div className="form-group">
                      <label>Exclude Patterns</label>
                      <input
                        type="text"
                        name="exclude_patterns"
                        defaultValue="**/node_modules/**,**/__pycache__/**,**/.git/**"
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

              {status.type && (
                <div className={`status-message ${status.type}`}>{status.message}</div>
              )}
            </form>
          ) : (
            <form onSubmit={handleGitSubmit}>
              <div className="form-group">
                <label>Index Name *</label>
                <input
                  type="text"
                  name="name"
                  placeholder="e.g., odoo-17, my-codebase"
                  required
                />
              </div>

              <p className="field-help" style={{ marginBottom: '16px' }}>
                A description will be auto-generated from the indexed content. You can edit it later.
              </p>

              <div className="row">
                <div className="form-group">
                  <label>Git URL *</label>
                  <input
                    type="text"
                    name="git_url"
                    placeholder="https://github.com/user/repo.git"
                    required
                  />
                </div>
                <div className="form-group">
                  <label>Branch</label>
                  <input type="text" name="git_branch" defaultValue="main" />
                </div>
              </div>

              <div className="row">
                <div className="form-group">
                  <label>File Patterns (comma-separated)</label>
                  <input
                    type="text"
                    name="file_patterns"
                    defaultValue="**/*.py,**/*.md,**/*.xml"
                  />
                </div>
                <div className="form-group">
                  <label>Exclude Patterns</label>
                  <input
                    type="text"
                    name="exclude_patterns"
                    defaultValue="**/test/**,**/tests/**,**/__pycache__/**"
                  />
                </div>
              </div>

              <div className="wizard-actions">
                <button type="button" className="btn btn-secondary" onClick={handleCancelWizard}>
                  Cancel
                </button>
                <button type="submit" className="btn" disabled={isLoading}>
                  {isLoading ? 'Cloning...' : 'Clone & Index'}
                </button>
              </div>

              {status.type && (
                <div className={`status-message ${status.type}`}>{status.message}</div>
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
              {idx.last_modified && (
                <span className="meta-pill date" title={`Last updated: ${new Date(idx.last_modified).toLocaleString()}`}>
                  {`Updated ${new Date(idx.last_modified).toLocaleString()}`}
                </span>
              )}
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
        </>
      )}
    </div>
  );
}
