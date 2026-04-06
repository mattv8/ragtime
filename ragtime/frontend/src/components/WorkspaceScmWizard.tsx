import { useEffect, useMemo, useState } from 'react';
import { AlertCircle, ArrowDownToLine, ArrowUpToLine, Check, GitBranch, Link2, RefreshCw, X } from 'lucide-react';

import { api } from '@/api';
import type {
  RepoVisibilityResponse,
  UserSpaceWorkspace,
  UserSpaceWorkspaceScmExportRequest,
  UserSpaceWorkspaceScmImportRequest,
  UserSpaceWorkspaceScmPreviewResponse,
  UserSpaceWorkspaceScmSyncResponse,
} from '@/types';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';

type WizardMode = 'import' | 'export';
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
  const [askedAgent, setAskedAgent] = useState(false);
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

  const modeActionLabel = hasConfiguredRemote
    ? (mode === 'import' ? 'Pull' : 'Push')
    : (mode === 'import' ? 'Import' : 'Export');

  useEffect(() => {
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
        setStatus({ type: 'info', message: visibility.message || '' });

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

  async function handlePreview(): Promise<void> {
    if (!gitUrl.trim()) {
      setStatus({ type: 'error', message: 'Repository URL is required.' });
      return;
    }
    if (tokenRequired && !gitToken.trim() && !storedTokenValid) {
      setStatus({ type: 'error', message: 'A personal access token is required for this action.' });
      return;
    }

    setIsLoading(true);
    setStatus({ type: null, message: '' });
    try {
      const payload = {
        git_url: gitUrl.trim(),
        git_branch: gitBranch.trim() || 'main',
        git_token: gitToken.trim() || undefined,
        create_repo_if_missing: createRepoIfMissing,
        create_repo_private: createRepoPrivate,
        create_repo_description: createRepoDescription.trim() || undefined,
      };
      const nextPreview = mode === 'import'
        ? await api.previewUserSpaceWorkspaceScmImport(workspace.id, payload)
        : await api.previewUserSpaceWorkspaceScmExport(workspace.id, payload);
      setPreview(nextPreview);
      setStep('review');
      setStatus({ type: null, message: '' });
    } catch (error) {
      setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Failed to preview sync.' });
    } finally {
      setIsLoading(false);
    }
  }

  async function handleExecute(): Promise<void> {
    if (!preview) return;
    setIsLoading(true);
    setStatus({ type: 'info', message: mode === 'import' ? 'Importing workspace from Git...' : 'Exporting workspace to Git...' });
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
      const nextResult = mode === 'import'
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
    }
  }

  async function handleAskAgent(): Promise<void> {
    if (!result?.suggested_setup_prompt || !onAskAgent) return;
    setIsLoading(true);
    try {
      await onAskAgent(result.suggested_setup_prompt);
      setAskedAgent(true);
      setStatus({ type: 'success', message: 'Asked the agent to inspect the imported workspace and prepare it for runtime.' });
    } catch (error) {
      setStatus({ type: 'error', message: error instanceof Error ? error.message : 'Failed to ask the agent.' });
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-large" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>Workspace SCM</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body" style={{ display: 'grid', gap: 16, maxHeight: '75vh', overflowY: 'auto' }}>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className={`btn btn-sm ${mode === 'import' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('import')} disabled={isLoading}>
              <ArrowDownToLine size={14} /> {hasConfiguredRemote ? 'Pull' : 'Import'}
            </button>
            <button className={`btn btn-sm ${mode === 'export' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setMode('export')} disabled={isLoading}>
              <ArrowUpToLine size={14} /> {hasConfiguredRemote ? 'Push' : 'Export'}
            </button>
          </div>

          {step === 'input' && (
            <div style={{ display: 'grid', gap: 14 }}>
              {hasConfiguredRemote && activeScm && (
                <div style={{ display: 'grid', gap: 10, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'flex-start', flexWrap: 'wrap' }}>
                    <div style={{ display: 'grid', gap: 6 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <Link2 size={14} />
                        <strong>Remote configured</strong>
                        <span className={syncStatusClassName}>
                          {activeScm.last_sync_status === 'success'
                            ? `${formatSyncDirection(activeScm.last_sync_direction)} ready`
                            : activeScm.last_sync_status || 'Connected'}
                        </span>
                      </div>
                      <div className="userspace-muted" style={{ fontSize: 12 }}>{activeScm.git_url || gitUrl}</div>
                      <div className="userspace-muted" style={{ fontSize: 12 }}>
                        Branch: {activeScm.git_branch || gitBranch || 'main'}
                      </div>
                      {activeScm.last_sync_message && (
                        <div className="userspace-muted" style={{ fontSize: 12 }}>
                          {activeScm.last_sync_message}
                        </div>
                      )}
                      {formatSyncTimestamp(activeScm.last_sync_at) && (
                        <div className="userspace-muted" style={{ fontSize: 12 }}>
                          Last {formatSyncDirection(activeScm.last_sync_direction).toLowerCase()}: {formatSyncTimestamp(activeScm.last_sync_at)}
                        </div>
                      )}
                    </div>
                    <div className="userspace-muted" style={{ fontSize: 12 }}>
                      Choose Pull or Push to sync with this remote.
                    </div>
                  </div>
                </div>
              )}

              <label className="form-group" style={{ marginBottom: 0, paddingBottom: 0 }}>
                <span>Repository URL</span>
                <input type="text" value={gitUrl} onChange={(event) => setGitUrl(event.target.value)} placeholder="https://github.com/owner/repo.git" disabled={isLoading} />
              </label>

              {repoVisibility && (
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

              {mode === 'export' && (
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

          {step === 'review' && preview && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <GitBranch size={14} />
                  <strong>{preview.git_branch}</strong>
                  <span className="userspace-muted">{preview.git_url}</span>
                </div>
                <div>{preview.summary}</div>
                <div className="userspace-muted" style={{ fontSize: 12, marginTop: 8 }}>
                  Local commit: {preview.local_commit_hash || 'none'}
                </div>
                <div className="userspace-muted" style={{ fontSize: 12 }}>
                  Remote commit: {preview.remote_commit_hash || 'none'}
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

          {step === 'result' && result && (
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

              {mode === 'import' && result.suggested_setup_prompt && (
                <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <strong>Prepare the imported workspace</strong>
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    This keeps bring-up suggestion-only. The agent will inspect the imported repo first, then repair entrypoint and bootstrap configuration only if needed.
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button className="btn btn-primary btn-sm" onClick={() => void handleAskAgent()} disabled={isLoading || askedAgent || !onAskAgent}>
                      {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <RefreshCw size={14} />}
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
        <div className="modal-footer" style={{ display: 'flex', justifyContent: 'space-between', gap: 8, padding: '0 20px 20px' }}>
          <div>
            {step !== 'input' && (
              <button className="btn btn-secondary" onClick={() => setStep(step === 'result' ? 'review' : 'input')} disabled={isLoading}>
                Back
              </button>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
              <X size={14} /> Close
            </button>
            {step === 'input' && (
              <button className="btn btn-primary" onClick={() => void handlePreview()} disabled={isLoading}>
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : mode === 'import' ? <ArrowDownToLine size={14} /> : <ArrowUpToLine size={14} />}
                Preview
              </button>
            )}
            {step === 'review' && preview && (
              <button
                className={`btn ${mode === 'import' && preview.will_overwrite_local ? 'btn-danger' : 'btn-primary'}`}
                onClick={() => void handleExecute()}
                disabled={isLoading || (!preview.can_proceed_without_force && !preview.preview_token && preview.state !== 'up_to_date')}
              >
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : mode === 'import' ? <ArrowDownToLine size={14} /> : <ArrowUpToLine size={14} />}
                {mode === 'import' ? (preview.will_overwrite_local ? 'Overwrite' : modeActionLabel) : modeActionLabel}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
