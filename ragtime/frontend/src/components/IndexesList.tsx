import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { api } from '@/api';
import type { IndexInfo, RepoVisibilityResponse } from '@/types';
import { GitIndexWizard } from './GitIndexWizard';
import { UploadForm } from './UploadForm';
import { DescriptionField } from './DescriptionField';

interface IndexesListProps {
  indexes: IndexInfo[];
  loading: boolean;
  error: string | null;
  onDelete: () => void;
  onToggle?: () => void;
  onDescriptionUpdate?: () => void;
  onJobCreated?: () => void;
  /** When true, hide search weight controls (not used in aggregate mode) */
  aggregateSearch?: boolean;
  /** Called when user wants to navigate to settings */
  onNavigateToSettings?: () => void;
}

type SourceType = 'upload' | 'git';

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
            <DescriptionField
              value={description}
              onChange={setDescription}
              disabled={saving}
              rows={4}
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

export function IndexesList({ indexes, loading, error, onDelete, onToggle, onDescriptionUpdate, onJobCreated, aggregateSearch = true, onNavigateToSettings }: IndexesListProps) {
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
  const [reindexVisibility, setReindexVisibility] = useState<RepoVisibilityResponse | null>(null);
  const [checkingReindexVisibility, setCheckingReindexVisibility] = useState(false);

  // Download state
  const [downloading, setDownloading] = useState<string | null>(null);

  // Create wizard state
  const [showCreateWizard, setShowCreateWizard] = useState(false);
  const [activeSource, setActiveSource] = useState<SourceType>('git');

  // Git index edit state
  const [editingGitIndex, setEditingGitIndex] = useState<IndexInfo | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Check repo visibility when reindex modal opens
  useEffect(() => {
    if (!reindexingIndex?.source) {
      setReindexVisibility(null);
      return;
    }

    const checkVisibility = async () => {
      setCheckingReindexVisibility(true);
      try {
        const result = await api.checkRepoVisibility({
          git_url: reindexingIndex.source!,
          index_name: reindexingIndex.name,
        });
        setReindexVisibility(result);
      } catch {
        // Fallback to has_stored_token if check fails
        setReindexVisibility({
          visibility: 'error',
          has_stored_token: reindexingIndex.has_stored_token,
          needs_token: !reindexingIndex.has_stored_token,
          message: 'Could not check repository visibility',
        });
      } finally {
        setCheckingReindexVisibility(false);
      }
    };

    checkVisibility();
  }, [reindexingIndex?.source, reindexingIndex?.name, reindexingIndex?.has_stored_token]);

  const handleCancelWizard = () => {
    setShowCreateWizard(false);
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

  const handleDownload = async (name: string) => {
    setDownloading(name);
    try {
      await api.downloadIndex(name);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Download failed');
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setDownloading(null);
    }
  };

  return (
    <div className="card">
      <div className="section-header">
        <h2>Document Indexes</h2>
        {showCreateWizard ? (
          <button
            type="button"
            className="close-btn"
            onClick={() => setShowCreateWizard(false)}
          >
            <X size={18} />
          </button>
        ) : (
          <button
            type="button"
            className="btn"
            onClick={() => setShowCreateWizard(true)}
          >
            Create Document Index
          </button>
        )}
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
                disabled={isAnalyzing}
              >
                Git Repository
              </button>
              <button
                type="button"
                className={`wizard-tab ${activeSource === 'upload' ? 'active' : ''}`}
                onClick={() => setActiveSource('upload')}
                disabled={isAnalyzing}
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
            <GitIndexWizard
              onJobCreated={handleGitJobCreated}
              onCancel={handleCancelWizard}
              onAnalysisStart={() => setIsAnalyzing(true)}
              onAnalysisComplete={() => setIsAnalyzing(false)}
              onNavigateToSettings={onNavigateToSettings}
            />
          ) : (
            <UploadForm
              onJobCreated={handleGitJobCreated}
              onCancel={handleCancelWizard}
              onAnalysisStart={() => setIsAnalyzing(true)}
              onAnalysisComplete={() => setIsAnalyzing(false)}
              onNavigateToSettings={onNavigateToSettings}
            />
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
            <h3>{idx.display_name || idx.name}</h3>
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
              {!aggregateSearch && (
                <span
                  className={`meta-pill weight ${idx.search_weight !== 1.0 ? 'weight-modified' : ''}`}
                  title="Search weight: Higher values prioritize this index in results. Click to edit."
                  style={{ cursor: 'pointer' }}
                  onClick={() => setWeightEditIndex(idx)}
                >
                  Weight: {idx.search_weight.toFixed(1)}
                </span>
              )}
              {!idx.enabled && <span className="meta-pill disabled">Excluded from RAG</span>}
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px' }}>
            {idx.source_type === 'git' && idx.source ? (
              <button
                className="btn btn-sm btn-secondary"
                onClick={() => setEditingGitIndex(idx)}
                title="Edit index configuration (description, patterns, branch, etc.)"
              >
                Edit
              </button>
            ) : (
              <button
                className="btn btn-sm btn-secondary"
                onClick={() => setEditingIndex(idx)}
                title="Edit description for AI context"
              >
                Edit
              </button>
            )}
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => handleDownload(idx.name)}
              disabled={downloading === idx.name}
              title="Download FAISS index files as zip"
            >
              {downloading === idx.name ? 'Downloading...' : 'Download'}
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
            <div className="modal-overlay" onClick={() => { setReindexingIndex(null); setReindexToken(''); setReindexVisibility(null); }}>
              <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '500px' }}>
                <div className="modal-header">
                  <h3>Pull &amp; Re-index</h3>
                  <button className="modal-close" onClick={() => { setReindexingIndex(null); setReindexToken(''); setReindexVisibility(null); }}>&times;</button>
                </div>
                <div className="modal-body">
                  <p style={{ marginBottom: '16px', color: 'var(--text-secondary)' }}>
                    Re-index <strong>{reindexingIndex.name}</strong> from Git repository.
                  </p>
                  <div style={{ marginBottom: '16px', padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', fontSize: '13px' }}>
                    <div style={{ marginBottom: '4px' }}><strong>Source:</strong> {reindexingIndex.source}</div>
                    {reindexingIndex.git_branch && (
                      <div><strong>Branch:</strong> {reindexingIndex.git_branch}</div>
                    )}
                  </div>

                  {/* Show loading state while checking visibility */}
                  {checkingReindexVisibility && (
                    <div style={{ marginBottom: '16px', padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', color: 'var(--text-secondary)' }}>
                      Checking repository access...
                    </div>
                  )}

                  {/* Private repo with valid stored token */}
                  {!checkingReindexVisibility && reindexVisibility?.visibility === 'private' && !reindexVisibility.needs_token && (
                    <div style={{ marginBottom: '16px', padding: '12px', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                      <span style={{ color: '#22c55e' }}>Private repository - will use stored token.</span>
                    </div>
                  )}

                  {/* Private repo needing token (no stored token or invalid) */}
                  {!checkingReindexVisibility && reindexVisibility?.needs_token && (
                    <div className="form-group">
                      {reindexVisibility.has_stored_token && (
                        <div style={{ marginBottom: '12px', padding: '12px', background: 'rgba(251, 191, 36, 0.1)', borderRadius: '8px', border: '1px solid rgba(251, 191, 36, 0.3)' }}>
                          <span style={{ color: '#fbbf24' }}>{reindexVisibility.message || 'Stored token is no longer valid.'}</span>
                        </div>
                      )}
                      <label htmlFor="reindex-token">
                        Git Token {reindexVisibility.visibility === 'private' ? '*' : '(optional)'}
                      </label>
                      <small style={{ display: 'block', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                        {reindexVisibility.visibility === 'private'
                          ? 'Required for this private repository. Token will be stored for future re-indexing.'
                          : 'Provide a token if the repository requires authentication.'}
                      </small>
                      <input
                        id="reindex-token"
                        type="password"
                        className="form-input"
                        value={reindexToken}
                        onChange={(e) => setReindexToken(e.target.value)}
                        placeholder="ghp_xxxx... or glpat-xxxx..."
                      />
                    </div>
                  )}

                  {/* Error checking visibility - show token input as fallback */}
                  {!checkingReindexVisibility && reindexVisibility?.visibility === 'error' && !reindexVisibility.has_stored_token && (
                    <div className="form-group">
                      <label htmlFor="reindex-token">
                        Git Token (optional)
                      </label>
                      <small style={{ display: 'block', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                        Provide a token if the repository requires authentication.
                      </small>
                      <input
                        id="reindex-token"
                        type="password"
                        className="form-input"
                        value={reindexToken}
                        onChange={(e) => setReindexToken(e.target.value)}
                        placeholder="ghp_xxxx... or glpat-xxxx..."
                      />
                    </div>
                  )}
                </div>
                <div className="modal-footer">
                  <button
                    className="btn btn-secondary"
                    onClick={() => {
                      setReindexingIndex(null);
                      setReindexToken('');
                      setReindexVisibility(null);
                    }}
                    disabled={reindexing || checkingReindexVisibility}
                  >
                    Cancel
                  </button>
                  <button
                    className="btn btn-primary"
                    onClick={handleReindex}
                    disabled={reindexing || checkingReindexVisibility || (reindexVisibility?.needs_token && !reindexToken)}
                  >
                    {checkingReindexVisibility ? 'Checking...' : reindexing ? 'Starting...' : 'Re-index'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {editingGitIndex && (
            <div className="modal-overlay" onClick={() => setEditingGitIndex(null)}>
              <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '700px' }}>
                <div className="modal-header">
                  <h3>Edit Index Configuration</h3>
                  <button className="modal-close" onClick={() => setEditingGitIndex(null)}>&times;</button>
                </div>
                <div className="modal-body">
                  <GitIndexWizard
                    key={editingGitIndex.name}
                    editIndex={editingGitIndex}
                    onCancel={() => setEditingGitIndex(null)}
                    onConfigSaved={() => {
                      setEditingGitIndex(null);
                      onToggle?.();  // Refresh the list to show updated config
                    }}
                    onNavigateToSettings={onNavigateToSettings}
                  />
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
