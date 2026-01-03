import { useState, useCallback, useEffect, type DragEvent, type ChangeEvent } from 'react';
import { api } from '@/api';
import type { IndexInfo, IndexJob } from '@/types';

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
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [deleteConfirmName, setDeleteConfirmName] = useState<string | null>(null);

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
  const [status, setStatus] = useState<{ type: 'info' | 'success' | 'error' | null; message: string }>({
    type: null,
    message: '',
  });

  // Git form state
  const [gitUrl, setGitUrl] = useState('');
  const [gitToken, setGitToken] = useState('');
  const [isPrivateRepo, setIsPrivateRepo] = useState(false);
  const [branches, setBranches] = useState<string[]>([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);

  /** Parse GitHub/GitLab URL to extract host, owner and repo */
  const parseGitUrl = useCallback((url: string): { host: 'github' | 'gitlab' | null; owner: string; repo: string } | null => {
    // GitHub HTTPS: https://github.com/owner/repo.git or https://github.com/owner/repo
    const githubHttpsMatch = url.match(/github\.com\/([^/]+)\/([^/.]+)/);
    if (githubHttpsMatch) {
      return { host: 'github', owner: githubHttpsMatch[1], repo: githubHttpsMatch[2] };
    }
    // GitHub SSH: git@github.com:owner/repo.git
    const githubSshMatch = url.match(/git@github\.com:([^/]+)\/([^/.]+)/);
    if (githubSshMatch) {
      return { host: 'github', owner: githubSshMatch[1], repo: githubSshMatch[2] };
    }
    // GitLab HTTPS: https://gitlab.com/owner/repo.git or https://gitlab.com/owner/repo
    const gitlabHttpsMatch = url.match(/gitlab\.com\/([^/]+)\/([^/.]+)/);
    if (gitlabHttpsMatch) {
      return { host: 'gitlab', owner: gitlabHttpsMatch[1], repo: gitlabHttpsMatch[2] };
    }
    // GitLab SSH: git@gitlab.com:owner/repo.git
    const gitlabSshMatch = url.match(/git@gitlab\.com:([^/]+)\/([^/.]+)/);
    if (gitlabSshMatch) {
      return { host: 'gitlab', owner: gitlabSshMatch[1], repo: gitlabSshMatch[2] };
    }
    return null;
  }, []);

  /** Fetch branches from GitHub or GitLab API */
  const fetchBranches = useCallback(async (url: string, token?: string, silent404 = false) => {
    const parsed = parseGitUrl(url);
    if (!parsed) {
      setBranches([]);
      setBranchError(null);
      return;
    }

    setLoadingBranches(true);
    setBranchError(null);

    try {
      let apiUrl: string;
      const headers: HeadersInit = {};

      if (parsed.host === 'github') {
        apiUrl = `https://api.github.com/repos/${parsed.owner}/${parsed.repo}/branches?per_page=100`;
        headers.Accept = 'application/vnd.github.v3+json';
        if (token) {
          headers.Authorization = `token ${token}`;
        }
      } else {
        // GitLab API
        const projectPath = encodeURIComponent(`${parsed.owner}/${parsed.repo}`);
        apiUrl = `https://gitlab.com/api/v4/projects/${projectPath}/repository/branches?per_page=100`;
        if (token) {
          headers['PRIVATE-TOKEN'] = token;
        }
      }

      const response = await fetch(apiUrl, { headers });

      if (!response.ok) {
        if (response.status === 404) {
          // For silent 404 (probing public repo), don't show error - repo might be private
          if (!silent404) {
            setBranchError(token ? 'Repository not found or token lacks access' : 'Repository not found or is private');
          }
        } else if (response.status === 401) {
          setBranchError('Invalid or expired token');
        } else {
          setBranchError(`API error: ${response.status}`);
        }
        setBranches([]);
        return;
      }

      const data = await response.json();
      const branchNames = data.map((b: { name: string }) => b.name);
      setBranches(branchNames);

      // Auto-select default branch if available
      if (branchNames.length > 0 && !selectedBranch) {
        const defaultBranch = branchNames.includes('main') ? 'main' : branchNames.includes('master') ? 'master' : branchNames[0];
        setSelectedBranch(defaultBranch);
      }
    } catch {
      if (!silent404) {
        setBranchError('Failed to fetch branches');
      }
      setBranches([]);
    } finally {
      setLoadingBranches(false);
    }
  }, [parseGitUrl, selectedBranch]);

  // Fetch branches when URL changes - only for private repos with a token
  // For public repos, we probe silently (don't show 404 errors since repo might be private)
  useEffect(() => {
    if (!gitUrl) {
      setBranches([]);
      setSelectedBranch('');
      setBranchError(null);
      return;
    }

    // Debounce the fetch
    const timer = setTimeout(() => {
      if (isPrivateRepo && gitToken && gitToken.length >= 10) {
        // For private repos with token, fetch branches and show any errors
        fetchBranches(gitUrl, gitToken, false);
      } else if (!isPrivateRepo) {
        // For public repos, try to fetch but silently handle 404
        // (repo might be private and user hasn't checked the box yet)
        fetchBranches(gitUrl, undefined, true);
      } else {
        // Private repo without token - clear branches
        setBranches([]);
        setBranchError(null);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [gitUrl, gitToken, isPrivateRepo, fetchBranches]);

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

    // Derive index name from repo URL
    const parsed = parseGitUrl(gitUrl);
    if (!parsed) {
      setStatus({ type: 'error', message: 'Invalid Git URL format' });
      return;
    }
    const name = parsed.repo.toLowerCase().replace(/[^a-z0-9_-]/g, '-');
    const filePatterns = (form.elements.namedItem('file_patterns') as HTMLInputElement).value || '**/*';
    const excludePatterns = (form.elements.namedItem('exclude_patterns') as HTMLInputElement).value || '';

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Starting git clone...' });

    try {
      const job: IndexJob = await api.indexFromGit({
        name,
        git_url: gitUrl,
        git_branch: selectedBranch || 'main',
        git_token: isPrivateRepo ? gitToken : undefined,
        config: {
          name,
          description: '',  // Auto-generated during indexing
          file_patterns: filePatterns.split(',').map((s) => s.trim()),
          exclude_patterns: excludePatterns.split(',').map((s) => s.trim()),
        },
      });
      setStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      // Reset git form state
      setGitUrl('');
      setGitToken('');
      setIsPrivateRepo(false);
      setBranches([]);
      setSelectedBranch('');
      setBranchError(null);
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
    // Reset git form state
    setGitUrl('');
    setGitToken('');
    setIsPrivateRepo(false);
    setBranches([]);
    setSelectedBranch('');
    setBranchError(null);
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
            <form onSubmit={handleGitSubmit}>
              <div className="row">
                <div className="form-group" style={{ flex: 2 }}>
                  <label>Git URL *</label>
                  <input
                    type="text"
                    value={gitUrl}
                    onChange={(e) => setGitUrl(e.target.value)}
                    placeholder="https://github.com/user/repo.git"
                    required
                  />
                </div>
                <div className="form-group" style={{ flex: 1 }}>
                  <label>
                    Branch
                    {loadingBranches && <span style={{ marginLeft: '0.5rem', color: '#888', fontSize: '0.85em' }}>(loading...)</span>}
                  </label>
                  {branches.length > 0 ? (
                    <select
                      value={selectedBranch}
                      onChange={(e) => setSelectedBranch(e.target.value)}
                      style={{ width: '100%' }}
                    >
                      {branches.map((branch) => (
                        <option key={branch} value={branch}>{branch}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      value={selectedBranch}
                      onChange={(e) => setSelectedBranch(e.target.value)}
                      placeholder={branchError ? 'Enter branch name' : 'main'}
                    />
                  )}
                  {branchError && (
                    <small style={{ color: '#f87171', fontSize: '0.85em', display: 'block', marginTop: '0.25rem' }}>
                      {branchError}
                    </small>
                  )}
                </div>
              </div>

              <p className="field-help" style={{ marginBottom: '16px' }}>
                Index name will be derived from the repository name. A description will be auto-generated from the indexed content. You can edit both later.
              </p>

              <div className="form-group" style={{ marginBottom: isPrivateRepo ? '0.5rem' : undefined }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={isPrivateRepo}
                    onChange={(e) => {
                      setIsPrivateRepo(e.target.checked);
                      if (!e.target.checked) {
                        setGitToken('');
                        // Re-fetch branches without token for public repos
                        if (gitUrl) {
                          fetchBranches(gitUrl);
                        }
                      }
                    }}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  Private repository (requires authentication)
                </label>
              </div>

              {isPrivateRepo && (
                <div className="form-group" style={{ marginLeft: '1.5rem', borderLeft: '2px solid #444', paddingLeft: '1rem', marginBottom: '1rem' }}>
                  <label>GitHub/GitLab Personal Access Token *</label>
                  <input
                    type="password"
                    value={gitToken}
                    onChange={(e) => setGitToken(e.target.value)}
                    placeholder="ghp_xxxx... or glpat-xxxx..."
                    autoComplete="off"
                    required={isPrivateRepo}
                  />
                  <small style={{ color: '#888', fontSize: '0.85em', display: 'block', marginTop: '0.25rem' }}>
                    Generate a token with repo access:{' '}
                    <a href="https://github.com/settings/tokens/new" target="_blank" rel="noopener noreferrer" style={{ color: '#60a5fa' }}>GitHub</a>
                    {' | '}
                    <a href="https://gitlab.com/-/user_settings/personal_access_tokens" target="_blank" rel="noopener noreferrer" style={{ color: '#60a5fa' }}>GitLab</a>
                    . Token is used once for cloning and never stored.
                  </small>
                </div>
              )}

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
                    placeholder="e.g., **/test/**,**/__pycache__/**"
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
