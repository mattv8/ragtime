import { useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, ArrowDownToLine, ArrowUpToLine, Check, Database, GitBranch, Link2, RefreshCw, RefreshCcw, Upload, X } from 'lucide-react';

import { api } from '@/api';
import type {
  RepoVisibilityResponse,
  SqliteImportResponse,
  UserSpaceWorkspace,
  UserSpaceWorkspaceScmExportRequest,
  UserSpaceWorkspaceScmImportRequest,
  UserSpaceWorkspaceScmPreviewResponse,
  UserSpaceWorkspaceScmSyncResponse,
} from '@/types';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';

type ModalTab = 'git-source' | 'sql-import';
type WizardMode = 'import' | 'export' | 'sql-import';
type WizardStep = 'input' | 'review' | 'result';
type StatusType = 'info' | 'success' | 'error' | null;

interface WorkspaceScmWizardProps {
  workspace: UserSpaceWorkspace;
  onClose: () => void;
  onSyncComplete: (response: UserSpaceWorkspaceScmSyncResponse) => Promise<void> | void;
  onAskAgent?: (prompt: string) => Promise<void> | void;
}

function getDefaultBranch(branches: string[], fallback: string): string {
  if (branches.includes(fallback)) return fallback;
  if (branches.includes('main')) return 'main';
  if (branches.includes('master')) return 'master';
  return branches[0] || fallback;
}

function formatSyncDirection(direction: 'import' | 'export' | null | undefined): string {
  if (direction === 'import') return 'Pull';
  if (direction === 'export') return 'Push';
  return 'Sync';
}

function formatSyncTimestamp(timestamp: string | null | undefined): string | null {
  if (!timestamp) return null;
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return null;
  return date.toLocaleString();
}

export function WorkspaceScmWizard({ workspace, onClose, onSyncComplete, onAskAgent }: WorkspaceScmWizardProps) {
  const initialScm = workspace.scm;
  const [activeTab, setActiveTab] = useState<ModalTab>('git-source');
  const [mode, setMode] = useState<WizardMode>('import');
  const [step, setStep] = useState<WizardStep>('input');
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({ type: null, message: '' });
  const [isLoading, setIsLoading] = useState(false);
  const [gitUrl, setGitUrl] = useState(initialScm?.git_url || '');
  const [gitBranch, setGitBranch] = useState(initialScm?.git_branch || 'main');
  const [gitToken, setGitToken] = useState('');
  const [repoVisibility, setRepoVisibility] = useState<RepoVisibilityResponse['visibility'] | null>(initialScm?.repo_visibility || null);
  const [hasStoredToken, setHasStoredToken] = useState(Boolean(initialScm?.has_stored_token));
  const [storedTokenValid, setStoredTokenValid] = useState(Boolean(initialScm?.has_stored_token));
  const [branches, setBranches] = useState<string[]>([]);
  const [branchError, setBranchError] = useState<string | null>(null);
  const [preview, setPreview] = useState<UserSpaceWorkspaceScmPreviewResponse | null>(null);
  const [result, setResult] = useState<UserSpaceWorkspaceScmSyncResponse | null>(null);
  const [createRepoIfMissing, setCreateRepoIfMissing] = useState(true);
  const [createRepoPrivate, setCreateRepoPrivate] = useState(true);
  const [createRepoDescription, setCreateRepoDescription] = useState(workspace.description || '');
  const [sqlFile, setSqlFile] = useState<File | null>(null);
  const [sqlImportResult, setSqlImportResult] = useState<SqliteImportResponse | null>(null);
  const [sqlDragOver, setSqlDragOver] = useState(false);
  const [loadingAction, setLoadingAction] = useState<'pull' | 'push' | 'overwrite' | 'sync' | 'preview' | 'execute' | null>(null);
  const [showOverwriteMenu, setShowOverwriteMenu] = useState(false);
  const sqlFileInputRef = useRef<HTMLInputElement>(null);
  const activeScm = result?.scm ?? initialScm ?? null;
  const hasConfiguredRemote = Boolean(activeScm?.connected || activeScm?.git_url);

  const tokenRequired = useMemo(() => {
    if (mode === 'export') {
      return !hasStoredToken;
    }
    return repoVisibility === 'private' && !hasStoredToken;
  }, [hasStoredToken, mode, repoVisibility]);

  const shouldShowStatus = useMemo(() => {
    if (!status.type || !status.message) return false;
    if (step === 'review' && preview && status.message === preview.summary) return false;
    if (step === 'result' && result && status.message === result.summary) return false;
    return true;
  }, [preview, result, status.message, status.type, step]);

  const syncStatusClassName = useMemo(() => {
    const syncStatus = activeScm?.last_sync_status?.toLowerCase();
    if (syncStatus === 'success') return 'userspace-status-pill userspace-status-pill-success';
    if (syncStatus === 'error' || syncStatus === 'failed' || syncStatus === 'failure') {
      return 'userspace-status-pill userspace-status-pill-danger';
    }
    if (activeScm?.connected || activeScm?.git_url) {
      return 'userspace-status-pill userspace-status-pill-info';
    }
    return 'userspace-status-pill userspace-status-pill-muted';
  }, [activeScm]);

  const setupModeLabel = mode === 'import' ? 'Import' : 'Export';

  useEffect(() => {
    if (hasConfiguredRemote) return;
    if (!gitUrl.trim()) {
      setRepoVisibility(null);
      setBranches([]);
      setBranchError(null);
      return;
    }

    let cancelled = false;
    const timer = window.setTimeout(async () => {
      try {
        const visibility = await api.checkUserSpaceWorkspaceScmRepoVisibility(workspace.id, {
          git_url: gitUrl.trim(),
        });
        if (cancelled) return;
        setRepoVisibility(visibility.visibility);
        setHasStoredToken(visibility.has_stored_token);
        setStoredTokenValid(visibility.has_stored_token && !visibility.needs_token);
        if (!hasConfiguredRemote) {
          setStatus({ type: 'info', message: visibility.message || '' });
        }

        if (mode === 'import' || visibility.visibility === 'public' || gitToken.trim() || visibility.has_stored_token) {
          const branchResult = await api.fetchUserSpaceWorkspaceScmBranches(workspace.id, {
            git_url: gitUrl.trim(),
            git_token: gitToken.trim() || undefined,
          });
          if (cancelled) return;
          setBranches(branchResult.branches || []);
          setBranchError(branchResult.error || null);
          if (branchResult.branches?.length) {
            setGitBranch((current) => getDefaultBranch(branchResult.branches, current || 'main'));
          }
        }
      } catch (error) {
        if (cancelled) return;
        setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Failed to inspect repository' });
      }
    }, 400);

    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [gitToken, gitUrl, mode, workspace.id]);

  async function handlePreview(explicitDirection?: 'import' | 'export', options?: { forceOverwrite?: boolean }): Promise<void> {
    if (!hasConfiguredRemote && !gitUrl.trim()) {
      setStatus({ type: 'error', message: 'Repository URL is required.' });
      return;
    }
    if (!hasConfiguredRemote && tokenRequired && !gitToken.trim() && !storedTokenValid) {
      setStatus({ type: 'error', message: 'A personal access token is required for this action.' });
      return;
    }

    const action = options?.forceOverwrite ? 'overwrite' : explicitDirection === 'export' ? 'push' : explicitDirection === 'import' ? 'pull' : hasConfiguredRemote ? 'sync' : 'preview';
    setIsLoading(true);
    setLoadingAction(action);
    setStatus({ type: null, message: '' });
    try {
      const payload = {
        git_url: gitUrl.trim() || undefined,
        git_branch: gitBranch.trim() || 'main',
        git_token: gitToken.trim() || undefined,
        create_repo_if_missing: createRepoIfMissing,
        create_repo_private: createRepoPrivate,
        create_repo_description: createRepoDescription.trim() || undefined,
        force_overwrite: options?.forceOverwrite || undefined,
      };
      let nextPreview: UserSpaceWorkspaceScmPreviewResponse;
      if (explicitDirection === 'import') {
        nextPreview = await api.previewUserSpaceWorkspaceScmImport(workspace.id, payload);
      } else if (explicitDirection === 'export') {
        nextPreview = await api.previewUserSpaceWorkspaceScmExport(workspace.id, payload);
      } else if (hasConfiguredRemote) {
        nextPreview = await api.previewUserSpaceWorkspaceScmSync(workspace.id, payload);
      } else if (mode === 'import') {
        nextPreview = await api.previewUserSpaceWorkspaceScmImport(workspace.id, payload);
      } else {
        nextPreview = await api.previewUserSpaceWorkspaceScmExport(workspace.id, payload);
      }
      setPreview(nextPreview);
      setStep('review');
      setStatus({ type: null, message: '' });
    } catch (error) {
      setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Failed to preview sync.' });
    } finally {
      setIsLoading(false);
      setLoadingAction(null);
    }
  }

  async function handleExecute(): Promise<void> {
    if (!preview) return;
    setIsLoading(true);
    setLoadingAction('execute');
    const direction = preview.direction;
    setStatus({ type: 'info', message: direction === 'import' ? 'Pulling from remote...' : 'Pushing to remote...' });
    try {
      const payload = {
        git_url: preview.git_url,
        git_branch: preview.git_branch,
        git_token: gitToken.trim() || undefined,
        create_repo_if_missing: createRepoIfMissing,
        create_repo_private: createRepoPrivate,
        create_repo_description: createRepoDescription.trim() || undefined,
        overwrite_preview_token: preview.preview_token || undefined,
      } satisfies UserSpaceWorkspaceScmImportRequest | UserSpaceWorkspaceScmExportRequest;
      const nextResult = direction === 'import'
        ? await api.importUserSpaceWorkspaceFromScm(workspace.id, payload as UserSpaceWorkspaceScmImportRequest)
        : await api.exportUserSpaceWorkspaceToScm(workspace.id, payload as UserSpaceWorkspaceScmExportRequest);
      setResult(nextResult);
      setStatus({ type: 'success', message: nextResult.summary });
      setStep('result');
      await onSyncComplete(nextResult);
    } catch (error) {
      setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Sync failed.' });
    } finally {
      setIsLoading(false);
      setLoadingAction(null);
    }
  }

  async function handleAskAgent(): Promise<void> {
    if (!result?.suggested_setup_prompt || !onAskAgent) return;
    onClose();
    await onAskAgent(result.suggested_setup_prompt);
  }

  async function handleSqlImport(): Promise<void> {
    if (!sqlFile) {
      setStatus({ type: 'error', message: 'Please select a SQL dump file.' });
      return;
    }
    setIsLoading(true);
    setStatus({ type: 'info', message: 'Importing SQL dump...' });
    try {
      const formData = new FormData();
      formData.append('file', sqlFile);
      const importResult = await api.importSqlToWorkspaceSqlite(workspace.id, formData);
      setSqlImportResult(importResult);
      setStatus({
        type: importResult.success ? 'success' : 'error',
        message: importResult.message,
      });
      setStep('result');
    } catch (error) {
      setStatus({ type: 'error', message: error instanceof Error ? error.message : 'SQL import failed.' });
    } finally {
      setIsLoading(false);
    }
  }

  const sqliteEnabled = workspace.sqlite_persistence_mode === 'include';
  const SQL_ACCEPT = '.sql,.dump,.pg,.pgsql,.mysql';

  function handleTabSwitch(tab: ModalTab) {
    if (isLoading) return;
    setActiveTab(tab);
    if (tab === 'sql-import') {
      setMode('sql-import');
      setStep('input');
      setSqlImportResult(null);
      setStatus({ type: null, message: '' });
    } else {
      setMode(hasConfiguredRemote ? 'import' : mode === 'sql-import' ? 'import' : mode);
      setStep('input');
      setStatus({ type: null, message: '' });
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-large" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header" style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
          <div style={{ display: 'flex', gap: 0, flex: 1 }}>
            <button
              type="button"
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'git-source' ? '2px solid var(--color-accent)' : '2px solid transparent',
                padding: '8px 16px',
                cursor: isLoading ? 'default' : 'pointer',
                color: activeTab === 'git-source' ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                fontSize: 14,
                fontWeight: 600,
                transition: 'color 0.15s, border-color 0.15s',
              }}
              onClick={() => handleTabSwitch('git-source')}
              disabled={isLoading}
            >
              Git Source
            </button>
            <button
              type="button"
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'sql-import' ? '2px solid var(--color-accent)' : '2px solid transparent',
                padding: '8px 16px',
                cursor: isLoading || !sqliteEnabled ? 'default' : 'pointer',
                color: activeTab === 'sql-import' ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                fontSize: 14,
                fontWeight: 600,
                opacity: sqliteEnabled ? 1 : 0.4,
                transition: 'color 0.15s, border-color 0.15s',
              }}
              onClick={() => handleTabSwitch('sql-import')}
              disabled={isLoading || !sqliteEnabled}
              title={sqliteEnabled ? 'Import a SQL dump into the workspace SQLite database' : 'Enable SQLite persistence mode on this workspace first'}
            >
              SQL Import
            </button>
          </div>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body" style={{ display: 'grid', gap: 16 }}>
          {activeTab === 'git-source' && !hasConfiguredRemote && (
            <div style={{ display: 'flex', gap: 8 }}>
              <button className={`btn btn-sm ${mode === 'import' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => { setMode('import'); setStep('input'); }} disabled={isLoading}>
                <ArrowDownToLine size={14} /> Import
              </button>
              <button className={`btn btn-sm ${mode === 'export' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => { setMode('export'); setStep('input'); }} disabled={isLoading}>
                <ArrowUpToLine size={14} /> Export
              </button>
            </div>
          )}

          {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && (
            <div style={{ display: 'grid', gap: 14 }}>
              {hasConfiguredRemote && activeScm && (
                <div style={{ display: 'grid', gap: 10, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <Link2 size={14} />
                    <strong style={{ flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {activeScm.git_url || gitUrl}
                    </strong>
                    <span className={syncStatusClassName}>
                      {activeScm.sync_paused
                        ? 'Paused'
                        : activeScm.last_sync_status === 'success'
                          ? `${formatSyncDirection(activeScm.last_sync_direction)} ready`
                          : activeScm.last_sync_status || 'Connected'}
                    </span>
                    {activeScm.remote_role === 'upstream' && (
                      <span className="userspace-status-pill userspace-status-pill-muted" style={{ fontSize: 11 }}>upstream</span>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', fontSize: 12 }} className="userspace-muted">
                    <span>Branch: {activeScm.git_branch || gitBranch || 'main'}</span>
                    {activeScm.last_sync_message && !activeScm.sync_paused && <span>{activeScm.last_sync_message}</span>}
                    {formatSyncTimestamp(activeScm.last_sync_at) && (
                      <span>Last {formatSyncDirection(activeScm.last_sync_direction).toLowerCase()}: {formatSyncTimestamp(activeScm.last_sync_at)}</span>
                    )}
                  </div>

                  {activeScm.sync_paused && activeScm.sync_paused_reason && (
                    <div style={{ fontSize: 12, padding: '8px 10px', borderRadius: 6, border: '1px solid var(--color-warning, #d69d2a)', background: 'rgba(214, 157, 42, 0.08)' }}>
                      <AlertCircle size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                      {activeScm.sync_paused_reason}
                    </div>
                  )}

                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', paddingTop: 8, borderTop: '1px solid var(--color-border-subtle)', flexWrap: 'wrap' }}>
                    <span className="userspace-muted" style={{ fontSize: 12, flex: 1 }}>
                      {activeScm.remote_role === 'upstream'
                        ? activeScm.auto_sync_policy === 'auto_push'
                          ? 'Snapshots are automatically pushed to the upstream remote.'
                          : 'Snapshots stay local. Use Pull to fetch upstream changes, Push to send local changes.'
                        : 'Snapshots from this workspace are automatically synced to the remote.'}
                    </span>
                    {activeScm.remote_role === 'upstream' && (
                      <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
                        <input
                          type="checkbox"
                          checked={activeScm.auto_sync_policy === 'auto_push'}
                          onChange={async (event) => {
                            try {
                              const nextPolicy = event.target.checked ? 'auto_push' : 'manual';
                              const resp = await api.updateUserSpaceWorkspaceScmSettings(workspace.id, {
                                auto_sync_policy: nextPolicy,
                                clear_sync_paused: nextPolicy === 'auto_push' ? true : undefined,
                              });
                              await onSyncComplete({ workspace_id: workspace.id, direction: 'export', state: 'settings_updated', summary: `Auto-push ${nextPolicy === 'auto_push' ? 'enabled' : 'disabled'}`, scm: resp.scm });
                            } catch (error) {
                              setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Failed to update setting' });
                            }
                          }}
                          disabled={isLoading}
                        />
                        Auto-push
                      </label>
                    )}
                    {activeScm.sync_paused && (
                      <button
                        className="btn btn-sm btn-secondary"
                        disabled={isLoading}
                        onClick={async () => {
                          try {
                            const resp = await api.updateUserSpaceWorkspaceScmSettings(workspace.id, { clear_sync_paused: true });
                            await onSyncComplete({ workspace_id: workspace.id, direction: 'export', state: 'resumed', summary: 'Sync resumed', scm: resp.scm });
                          } catch (error) {
                            setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Failed to resume sync' });
                          }
                        }}
                      >
                        <RefreshCcw size={12} /> Resume sync
                      </button>
                    )}
                  </div>
                </div>
              )}

              {!hasConfiguredRemote && (
                <label className="form-group" style={{ marginBottom: 0, paddingBottom: 0 }}>
                  <span>Repository URL</span>
                  <input type="text" value={gitUrl} onChange={(event) => setGitUrl(event.target.value)} placeholder="https://github.com/owner/repo.git" disabled={isLoading} />
                </label>
              )}

              {(!hasConfiguredRemote || !storedTokenValid) && repoVisibility && (
              <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <Link2 size={14} />
                  <strong>Personal Access Token</strong>
                </div>
                {hasStoredToken && storedTokenValid && (
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    A stored token is already available for this workspace connection.
                  </div>
                )}
                {(tokenRequired || !storedTokenValid || mode === 'export') && (
                  <label className="form-group" style={{ marginBottom: 0 }}>
                    <input type="password" value={gitToken} onChange={(event) => setGitToken(event.target.value)} placeholder={mode === 'export' ? 'Required for push or repo creation' : 'Only needed for private repos'} disabled={isLoading} autoComplete="off" />
                  </label>
                )}
              </div>
              )}

              {!hasConfiguredRemote && (
                <label className="form-group">
                  <span>Branch</span>
                  {branches.length > 0 ? (
                    <select value={gitBranch} onChange={(event) => setGitBranch(event.target.value)} disabled={isLoading}>
                      {branches.map((branch) => (
                        <option key={branch} value={branch}>{branch}</option>
                      ))}
                    </select>
                  ) : (
                    <input type="text" value={gitBranch} onChange={(event) => setGitBranch(event.target.value)} placeholder="main" disabled={isLoading} />
                  )}
                  {branchError && <span className="userspace-muted" style={{ fontSize: 12 }}>{branchError}</span>}
                </label>
              )}

              {mode === 'export' && !hasConfiguredRemote && (
                <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input type="checkbox" checked={createRepoIfMissing} onChange={(event) => setCreateRepoIfMissing(event.target.checked)} disabled={isLoading} />
                    Create the remote repository if it does not exist
                  </label>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input type="checkbox" checked={createRepoPrivate} onChange={(event) => setCreateRepoPrivate(event.target.checked)} disabled={isLoading} />
                    Create as private repository
                  </label>
                  <label className="form-group" style={{ marginBottom: 0 }}>
                    <span>Repository description</span>
                    <input type="text" value={createRepoDescription} onChange={(event) => setCreateRepoDescription(event.target.value)} placeholder="Optional repository description" disabled={isLoading} />
                  </label>
                </div>
              )}
            </div>
          )}

          {activeTab === 'sql-import' && step === 'input' && mode === 'sql-import' && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div className="userspace-muted" style={{ fontSize: 12 }}>
                Upload a PostgreSQL (<code>pg_dump --format=plain</code>), MySQL (<code>mysqldump</code>), or generic SQL text dump.
                It will be converted and imported into the workspace SQLite database at <code>.ragtime/db/app.sqlite3</code>.
              </div>

              <div
                style={{
                  padding: 24,
                  border: `2px dashed ${sqlDragOver ? 'var(--color-primary)' : 'var(--color-border)'}`,
                  borderRadius: 8,
                  textAlign: 'center',
                  cursor: isLoading ? 'default' : 'pointer',
                  background: sqlDragOver ? 'rgba(var(--color-primary-rgb, 59, 130, 246), 0.05)' : 'transparent',
                  transition: 'border-color 0.15s, background 0.15s',
                }}
                onClick={() => !isLoading && sqlFileInputRef.current?.click()}
                onDragOver={(event) => { event.preventDefault(); setSqlDragOver(true); }}
                onDragLeave={() => setSqlDragOver(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setSqlDragOver(false);
                  const droppedFile = event.dataTransfer.files[0];
                  if (droppedFile) {
                    setSqlFile(droppedFile);
                    setSqlImportResult(null);
                    setStatus({ type: null, message: '' });
                  }
                }}
              >
                <input
                  ref={sqlFileInputRef}
                  type="file"
                  accept={SQL_ACCEPT}
                  style={{ display: 'none' }}
                  onChange={(event) => {
                    const selectedFile = event.target.files?.[0] || null;
                    setSqlFile(selectedFile);
                    setSqlImportResult(null);
                    setStatus({ type: null, message: '' });
                  }}
                  disabled={isLoading}
                />
                {sqlFile ? (
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                    <Check size={16} style={{ color: 'var(--color-success, #2b7a2b)' }} />
                    <span>{sqlFile.name}</span>
                    <span className="userspace-muted" style={{ fontSize: 12 }}>
                      ({(sqlFile.size / 1024).toFixed(1)} KB)
                    </span>
                    <button
                      className="btn btn-sm btn-secondary"
                      style={{ marginLeft: 8, padding: '2px 6px' }}
                      onClick={(event) => { event.stopPropagation(); setSqlFile(null); setSqlImportResult(null); }}
                      disabled={isLoading}
                    >
                      <X size={12} />
                    </button>
                  </div>
                ) : (
                  <div>
                    <Upload size={24} style={{ marginBottom: 8, opacity: 0.5 }} />
                    <div>Drop a SQL dump file here or click to browse</div>
                    <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                      Accepts .sql, .dump, .pg, .pgsql, .mysql (text format only, max 100 MB)
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'sql-import' && step === 'result' && mode === 'sql-import' && sqlImportResult && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{
                display: 'flex', gap: 8, alignItems: 'center', padding: 12, borderRadius: 8,
                border: `1px solid ${sqlImportResult.success ? 'var(--color-success, #2b7a2b)' : 'var(--color-danger, #c53030)'}`,
                background: sqlImportResult.success ? 'rgba(43, 122, 43, 0.08)' : 'rgba(197, 48, 48, 0.08)',
              }}>
                {sqlImportResult.success ? <Check size={16} /> : <AlertCircle size={16} />}
                <div>
                  <strong>{sqlImportResult.message}</strong>
                </div>
              </div>

              <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 16px', fontSize: 13 }}>
                  <span className="userspace-muted">Dialect detected:</span>
                  <span>{sqlImportResult.dialect_detected}</span>
                  <span className="userspace-muted">Tables created:</span>
                  <span>{sqlImportResult.tables_created}</span>
                  <span className="userspace-muted">Rows inserted:</span>
                  <span>{sqlImportResult.rows_inserted}</span>
                  <span className="userspace-muted">Statements executed:</span>
                  <span>{sqlImportResult.statements_executed}</span>
                </div>
              </div>

              {sqlImportResult.warnings.length > 0 && (
                <div style={{ padding: 12, border: '1px solid var(--color-warning, #d69d2a)', borderRadius: 8 }}>
                  <strong style={{ fontSize: 13 }}>Warnings ({sqlImportResult.warnings.length})</strong>
                  <div style={{ maxHeight: 160, overflowY: 'auto', marginTop: 8 }}>
                    {sqlImportResult.warnings.map((warning, index) => (
                      <div key={index} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '2px 0' }}>{warning}</div>
                    ))}
                  </div>
                </div>
              )}

              {sqlImportResult.errors.length > 0 && (
                <div style={{ padding: 12, border: '1px solid var(--color-danger, #c53030)', borderRadius: 8 }}>
                  <strong style={{ fontSize: 13 }}>Errors ({sqlImportResult.errors.length})</strong>
                  <div style={{ maxHeight: 200, overflowY: 'auto', marginTop: 8 }}>
                    {sqlImportResult.errors.map((error, index) => (
                      <div key={index} style={{ fontFamily: 'var(--font-mono)', fontSize: 11, padding: '2px 0', color: 'var(--color-danger, #c53030)' }}>{error}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'git-source' && step === 'review' && preview && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GitBranch size={14} />
                  <strong>{preview.git_branch}</strong>
                  <span className="userspace-muted">{preview.git_url}</span>
                  <span className={`userspace-status-pill ${preview.direction === 'import' ? 'userspace-status-pill-info' : 'userspace-status-pill-success'}`}>
                    {preview.direction === 'import' ? 'Pull' : 'Push'}
                  </span>
                </div>
                <div>{preview.summary}</div>
                {preview.state_explanation && (
                  <div className="userspace-muted" style={{ fontSize: 12, marginTop: 8 }}>
                    {preview.state_explanation}
                  </div>
                )}
                <div className="userspace-muted" style={{ fontSize: 12, marginTop: 8 }}>
                  Workspace HEAD: {preview.workspace_head_commit_hash || preview.local_commit_hash || 'none'}
                </div>
                <div className="userspace-muted" style={{ fontSize: 12 }}>
                  Remote HEAD: {preview.remote_head_commit_hash || preview.remote_commit_hash || 'none'}
                </div>
                <div className="userspace-muted" style={{ fontSize: 12 }}>
                  Last synced remote: {preview.last_synced_remote_commit_hash || 'none'}
                </div>
              </div>

              {(preview.will_overwrite_local || preview.will_overwrite_remote) && (
                <div style={{ display: 'flex', gap: 8, padding: 12, borderRadius: 8, border: '1px solid var(--color-warning, #d69d2a)', background: 'rgba(214, 157, 42, 0.08)' }}>
                  <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                  <div>
                    <strong>Explicit overwrite required.</strong>
                    <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                      This action will replace existing {preview.will_overwrite_local ? 'workspace' : 'remote'} state. Review the sample paths below before continuing.
                    </div>
                  </div>
                </div>
              )}

              {preview.changed_files_sample.length > 0 && (
                <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <strong>Changed Paths</strong>
                  <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4, marginBottom: 8 }}>
                    Showing up to {preview.changed_files_sample.length} sampled paths.
                  </div>
                  <div style={{ display: 'grid', gap: 4, maxHeight: 240, overflowY: 'auto' }}>
                    {preview.changed_files_sample.map((path) => (
                      <div key={path} style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{path}</div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'git-source' && step === 'result' && mode !== 'sql-import' && result && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 12, border: '1px solid var(--color-success, #2b7a2b)', borderRadius: 8, background: 'rgba(43, 122, 43, 0.08)' }}>
                <Check size={16} />
                <div>
                  <strong>{result.summary}</strong>
                  <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                    Remote commit: {result.remote_commit_hash || 'unknown'}
                  </div>
                </div>
              </div>

              {result.direction === 'import' && result.suggested_setup_prompt && (
                <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <strong>Prepare the imported workspace</strong>
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    This keeps bring-up suggestion-only. The agent will inspect the imported repo first, then repair entrypoint and bootstrap configuration only if needed.
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button className="btn btn-primary btn-sm" onClick={() => void handleAskAgent()} disabled={!onAskAgent}>
                      <RefreshCw size={14} />
                      Ask Agent to Prepare
                    </button>
                    <button className="btn btn-secondary btn-sm" onClick={() => navigator.clipboard.writeText(result.suggested_setup_prompt || '')}>
                      Copy Prompt
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {shouldShowStatus && (
            <div className={`status-message ${status.type}`}>
              {status.message}
            </div>
          )}
        </div>
        <div className="modal-footer" style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
          <div>
            {activeTab === 'git-source' && step === 'review' && mode !== 'sql-import' && (
              <button className="btn btn-secondary" onClick={() => setStep('input')} disabled={isLoading}>
                Back
              </button>
            )}
            {activeTab === 'sql-import' && step === 'result' && mode === 'sql-import' && (
              <button className="btn btn-secondary" onClick={() => { setStep('input'); setSqlFile(null); setSqlImportResult(null); setStatus({ type: null, message: '' }); }} disabled={isLoading}>
                Import Another
              </button>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
              <X size={14} /> Close
            </button>
            {activeTab === 'sql-import' && step === 'input' && mode === 'sql-import' && (
              <button className="btn btn-primary" onClick={() => void handleSqlImport()} disabled={isLoading || !sqlFile}>
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <Database size={14} />}
                Import to SQLite
              </button>
            )}
            {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && !hasConfiguredRemote && (
              <button className="btn btn-primary" onClick={() => void handlePreview()} disabled={isLoading}>
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : mode === 'import' ? <ArrowDownToLine size={14} /> : <ArrowUpToLine size={14} />}
                {`Preview ${setupModeLabel}`}
              </button>
            )}
            {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && hasConfiguredRemote && activeScm?.remote_role === 'upstream' && (
              <>
                <button className="btn btn-primary" onClick={() => void handlePreview('import')} disabled={isLoading}>
                  {loadingAction === 'pull' ? <MiniLoadingSpinner variant="icon" size={14} /> : <ArrowDownToLine size={14} />}
                  Pull
                </button>
                <button className="btn btn-secondary" onClick={() => void handlePreview('export')} disabled={isLoading}>
                  {loadingAction === 'push' ? <MiniLoadingSpinner variant="icon" size={14} /> : <ArrowUpToLine size={14} />}
                  Push
                </button>
                <div style={{ position: 'relative', alignSelf: 'stretch' }}>
                  <button className="btn btn-secondary" onClick={() => setShowOverwriteMenu(prev => !prev)} disabled={isLoading}
                    title="More options" style={{ padding: '6px 8px', minWidth: 0, height: '100%' }}>
                    &#8230;
                  </button>
                  {showOverwriteMenu && (
                    <div style={{
                      position: 'absolute', bottom: 'calc(100% + 6px)', right: 0, minWidth: 200,
                      padding: '10px 12px', borderRadius: 8, border: '1px solid var(--color-border)',
                      background: 'var(--color-bg-secondary)', boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
                      display: 'grid', gap: 8, zIndex: 10,
                    }}>
                      <div style={{ fontSize: 11 }} className="userspace-muted">
                        Replaces all local files with the remote state. Local-only changes will be lost.
                      </div>
                      <DeleteConfirmButton
                        onDelete={() => { setShowOverwriteMenu(false); void handlePreview('import', { forceOverwrite: true }); }}
                        disabled={isLoading}
                        className="btn btn-sm btn-danger"
                        title="Overwrite local files with remote state"
                        buttonText="Overwrite local"
                      />
                    </div>
                  )}
                </div>
              </>
            )}
            {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && hasConfiguredRemote && activeScm?.remote_role !== 'upstream' && (
              <button className="btn btn-primary" onClick={() => void handlePreview()} disabled={isLoading}>
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <RefreshCcw size={14} />}
                Sync
              </button>
            )}
            {activeTab === 'git-source' && step === 'review' && preview && (
              <button
                className={`btn ${preview.will_overwrite_local || preview.will_overwrite_remote ? 'btn-danger' : 'btn-primary'}`}
                onClick={() => void handleExecute()}
                disabled={isLoading || (!preview.can_proceed_without_force && !preview.preview_token && preview.state !== 'up_to_date')}
              >
                {isLoading
                  ? <MiniLoadingSpinner variant="icon" size={14} />
                  : preview.direction === 'import' ? <ArrowDownToLine size={14} /> : <ArrowUpToLine size={14} />
                }
                {preview.will_overwrite_local
                  ? 'Overwrite Local'
                  : preview.will_overwrite_remote
                    ? 'Force Push'
                    : preview.direction === 'import' ? 'Pull' : 'Push'
                }
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
