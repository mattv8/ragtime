import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { RotateCcw } from 'lucide-react';

import { api } from '@/api/client';
import type { UserSpaceCodeIndexAdminItem, UserSpaceCodeIndexJob } from '@/types';

import { DeleteConfirmButton } from '../DeleteConfirmButton';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import { ToastContainer, useToast } from './Toast';

const ACTIVE_INDEX_POLL_INTERVAL = 2000;

interface UserSpaceCodeIndexesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function formatDate(value: string | null): string {
  if (!value) return 'Never';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatCount(value: number): string {
  return value.toLocaleString();
}

function clampProgressPercent(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(100, Math.round(value)));
}

function getUserSpaceCodePhaseText(phase: string | null | undefined): string {
  switch (phase) {
    case 'collecting':
      return 'Collecting files';
    case 'loading_files':
      return 'Loading files';
    case 'chunking':
      return 'Chunking';
    case 'embedding':
      return 'Embedding';
    case 'indexing_symbols':
      return 'Indexing symbols';
    case 'finalizing':
      return 'Finalizing';
    default:
      return 'Indexing';
  }
}

export function UserSpaceCodeIndexesModal({ isOpen, onClose }: UserSpaceCodeIndexesModalProps) {
  const [toasts, toast] = useToast();
  const [items, setItems] = useState<UserSpaceCodeIndexAdminItem[]>([]);
  const [jobs, setJobs] = useState<UserSpaceCodeIndexJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [settingsLoading, setSettingsLoading] = useState(false);
  const [indexingEnabled, setIndexingEnabled] = useState(true);
  const [savingToggle, setSavingToggle] = useState(false);
  const [actionWorkspaceId, setActionWorkspaceId] = useState<string | null>(null);
  const [reconciling, setReconciling] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadItems = useCallback(async () => {
    setLoading(true);
    try {
      const [rows, jobRows] = await Promise.all([
        api.listUserSpaceCodeIndexes(),
        api.listUserSpaceCodeIndexJobs(),
      ]);
      setItems(rows);
      setJobs(jobRows);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load code indexes');
    } finally {
      setLoading(false);
    }
  }, [toast]);

  const loadIndexingSetting = useCallback(async () => {
    setSettingsLoading(true);
    try {
      const { settings } = await api.getSettings();
      setIndexingEnabled(settings.userspace_code_index_enabled ?? true);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to load code indexing setting');
    } finally {
      setSettingsLoading(false);
    }
  }, [toast]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    void loadItems();
    void loadIndexingSetting();
  }, [isOpen, loadIndexingSetting, loadItems]);

  const sortedItems = useMemo(() => {
    return [...items].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  }, [items]);

  const latestJobsByWorkspace = useMemo(() => {
    const byWorkspace = new Map<string, UserSpaceCodeIndexJob>();
    for (const job of jobs) {
      const existing = byWorkspace.get(job.workspace_id);
      if (!existing || job.created_at.localeCompare(existing.created_at) > 0) {
        byWorkspace.set(job.workspace_id, job);
      }
    }
    return byWorkspace;
  }, [jobs]);

  const hasActiveIndexing = useMemo(() => {
    return (
      sortedItems.some((item) => ['pending', 'stale', 'indexing'].includes(item.status)) ||
      jobs.some((job) => job.status === 'pending' || job.status === 'indexing')
    );
  }, [jobs, sortedItems]);

  useEffect(() => {
    if (!isOpen || !hasActiveIndexing) {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      return;
    }

    pollRef.current = setInterval(() => {
      void loadItems();
    }, ACTIVE_INDEX_POLL_INTERVAL);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [hasActiveIndexing, isOpen, loadItems]);

  if (!isOpen) {
    return null;
  }

  const handleReindex = async (workspaceId: string) => {
    if (!indexingEnabled) {
      toast.error('Enable User Space code indexing before scheduling reindex jobs');
      return;
    }
    setActionWorkspaceId(workspaceId);
    try {
      await api.reindexUserSpaceCodeIndex(workspaceId);
      toast.success('Workspace code index queued for reindexing');
      await loadItems();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to queue reindex');
    } finally {
      setActionWorkspaceId(null);
    }
  };

  const handleDelete = async (workspaceId: string) => {
    setActionWorkspaceId(workspaceId);
    try {
      await api.deleteUserSpaceCodeIndex(workspaceId);
      setItems((current) => current.filter((item) => item.workspace_id !== workspaceId));
      toast.success('Workspace code index deleted');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete code index');
    } finally {
      setActionWorkspaceId(null);
    }
  };

  const handleReconcile = async () => {
    if (!indexingEnabled) {
      toast.error('Enable User Space code indexing before indexing workspaces');
      return;
    }
    setReconciling(true);
    try {
      const result = await api.reconcileUserSpaceCodeIndexes();
      toast.success(
        `${result.scheduled_count.toLocaleString()} workspace index reconciliation job(s) scheduled`,
      );
      await loadItems();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to reconcile code indexes');
    } finally {
      setReconciling(false);
    }
  };

  const handleToggleIndexing = async (enabled: boolean) => {
    setSavingToggle(true);
    try {
      const updated = await api.updateSettings({ userspace_code_index_enabled: enabled });
      setIndexingEnabled(updated.userspace_code_index_enabled ?? enabled);
      toast.success(
        enabled ? 'User Space code indexing enabled' : 'User Space code indexing disabled',
      );
      if (enabled) {
        await loadItems();
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update code indexing setting');
    } finally {
      setSavingToggle(false);
    }
  };

  const reconcileLabel = sortedItems.length === 0 ? 'Index all workspaces' : 'Reconcile now';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-large" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>User Space Code Indexes</h3>
          <button className="modal-close" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          <p className="field-help" style={{ margin: '0 0 0.75rem 0' }}>
            Hidden per-workspace indexes power code search inside User Space conversations. They are
            separate from normal document indexes and tool selection.
          </p>

          <div className="form-group" style={{ marginBottom: 12 }}>
            <label
              className="chat-toggle-control"
              style={{ display: 'flex', alignItems: 'center', gap: 8 }}
            >
              <span className="toggle-switch">
                <input
                  type="checkbox"
                  checked={indexingEnabled}
                  disabled={settingsLoading || savingToggle}
                  onChange={(event) => void handleToggleIndexing(event.target.checked)}
                />
                <span className="toggle-slider"></span>
              </span>
              <span>
                Enable User Space code indexing
                {savingToggle && <span className="muted"> Saving...</span>}
              </span>
            </label>
            <p className="field-help">
              When disabled, Ragtime stops creating and updating User Space code indexes. Existing
              indexes remain available for deletion and are not removed automatically.
            </p>
          </div>

          {indexingEnabled && (
            <div className="form-actions" style={{ borderTop: 'none', paddingTop: 0, gap: 8 }}>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleReconcile}
                disabled={reconciling || loading}
              >
                {reconciling ? (
                  <MiniLoadingSpinner variant="icon" size={12} />
                ) : (
                  <RotateCcw size={13} />
                )}
                {reconcileLabel}
              </button>
            </div>
          )}

          {loading && sortedItems.length === 0 ? (
            <p className="userspace-muted">Loading code indexes...</p>
          ) : sortedItems.length === 0 ? (
            <p className="userspace-muted">No workspace code indexes have been created yet.</p>
          ) : (
            <div className="jobs-table-wrapper">
              <table className="jobs-table">
                <thead>
                  <tr>
                    <th>Workspace</th>
                    <th>Status</th>
                    <th>Files</th>
                    <th>Chunks</th>
                    <th>Symbols</th>
                    <th>Dirty</th>
                    <th>Progress</th>
                    <th>Last Indexed</th>
                    <th>Last Reconciled</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedItems.map((item) => {
                    const isBusy = actionWorkspaceId === item.workspace_id;
                    const latestJob = latestJobsByWorkspace.get(item.workspace_id) ?? null;
                    const activeStatus =
                      latestJob && ['pending', 'indexing'].includes(latestJob.status)
                        ? latestJob.status
                        : ['pending', 'stale', 'indexing'].includes(item.status)
                          ? item.status === 'indexing'
                            ? 'indexing'
                            : 'pending'
                          : null;
                    const activeProgressPercent = activeStatus
                      ? clampProgressPercent(latestJob?.progress_percent ?? 0)
                      : null;
                    return (
                      <tr key={item.workspace_id}>
                        <td>
                          <strong>{item.workspace_name}</strong>
                          {item.last_error && (
                            <div className="field-error" style={{ marginTop: 4 }}>
                              {item.last_error}
                            </div>
                          )}
                        </td>
                        <td>{item.status}</td>
                        <td>{formatCount(item.file_count)}</td>
                        <td>{formatCount(item.chunk_count)}</td>
                        <td>{formatCount(item.symbol_count)}</td>
                        <td>{formatCount(item.dirty_path_count)}</td>
                        <td className="progress-cell">
                          {activeStatus === 'pending' ? (
                            <span className="progress-pending">Waiting...</span>
                          ) : activeStatus === 'indexing' && activeProgressPercent !== null ? (
                            <div className="progress-container">
                              <div
                                className="progress-bar"
                                role="progressbar"
                                aria-label={`${item.workspace_name} indexing progress`}
                                aria-valuemin={0}
                                aria-valuemax={100}
                                aria-valuenow={activeProgressPercent}
                              >
                                <div
                                  className="progress-fill"
                                  style={{ width: `${activeProgressPercent}%` }}
                                />
                              </div>
                              <div className="progress-details">
                                <span className="progress-phase">
                                  {getUserSpaceCodePhaseText(latestJob?.phase)}{' '}
                                  {activeProgressPercent}%
                                </span>
                                {latestJob && (
                                  <span className="progress-stats">
                                    {['embedding', 'indexing_symbols', 'finalizing'].includes(
                                      latestJob.phase ?? '',
                                    ) && latestJob.total_chunks > 0
                                      ? `${formatCount(latestJob.processed_chunks)} / ${formatCount(latestJob.total_chunks)} chunks`
                                      : `${formatCount(latestJob.processed_files)} / ${formatCount(latestJob.total_files)} files`}
                                  </span>
                                )}
                              </div>
                            </div>
                          ) : latestJob?.status === 'completed' ? (
                            <span className="progress-complete">
                              {formatCount(latestJob.total_files)} files,{' '}
                              {formatCount(latestJob.total_chunks)} chunks
                            </span>
                          ) : latestJob?.status === 'failed' ? (
                            <span className="progress-failed">Failed</span>
                          ) : (
                            <span className="progress-failed">--</span>
                          )}
                          {latestJob && !activeStatus && latestJob.status !== 'completed' && (
                            <div className="muted" style={{ fontSize: '0.8em', marginTop: 2 }}>
                              {formatCount(latestJob.processed_files)} /{' '}
                              {formatCount(latestJob.total_files)} files
                            </div>
                          )}
                          {activeStatus === 'indexing' && latestJob?.current_file && (
                            <div className="muted" style={{ fontSize: '0.8em', marginTop: 2 }}>
                              {latestJob.current_file}
                            </div>
                          )}
                        </td>
                        <td>{formatDate(item.last_indexed_at)}</td>
                        <td>{formatDate(item.last_reconciled_at)}</td>
                        <td>
                          <div style={{ display: 'flex', gap: 6, flexWrap: 'nowrap' }}>
                            <button
                              type="button"
                              className="btn btn-secondary btn-sm"
                              onClick={() => void handleReindex(item.workspace_id)}
                              disabled={isBusy || !indexingEnabled}
                            >
                              {isBusy ? (
                                <MiniLoadingSpinner variant="icon" size={12} />
                              ) : (
                                <RotateCcw size={13} />
                              )}
                              Reindex
                            </button>
                            <DeleteConfirmButton
                              onDelete={() => void handleDelete(item.workspace_id)}
                              disabled={actionWorkspaceId !== null && !isBusy}
                              deleting={isBusy}
                              buttonText="Delete index"
                            />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
        <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
      </div>
    </div>
  );
}
