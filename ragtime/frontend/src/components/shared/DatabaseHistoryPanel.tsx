import { useCallback, useEffect, useRef, useState } from 'react';
import {
  AlertTriangle,
  DatabaseBackup,
  Download,
  Loader2,
  Plus,
  RotateCcw,
  Trash2,
  X,
} from 'lucide-react';

import { api, ApiError } from '@/api/client';
import type {
  SqliteHistoryBackup,
  SqliteHistoryConflictPolicy,
  SqliteHistoryListResponse,
  SqliteHistoryPreview,
  SqliteHistoryRestoreMode,
  SqliteHistoryBackupTrigger,
  SqliteBackupJob,
  SqliteBackupJobStatus,
} from '@/types';

interface DatabaseHistoryPanelProps {
  workspaceId: string;
  ownerOrAdmin: boolean;
  databaseName?: string;
  snapshotId?: string;
  triggerLabel?: string;
  hostId: string;
}

type Receipt = { safety_backup_id: string | null; operation_id: string };

const TRIGGER_GROUP = {
  snapshot: 'checkpoint',
  manual: 'checkpoint',
  pre_restore: 'safety',
  scheduled: 'hourly',
} as const satisfies Record<SqliteHistoryBackupTrigger, string>;

const TRIGGER_LABEL: Record<SqliteHistoryBackupTrigger, string> = {
  snapshot: 'Code snapshot',
  manual: 'Manual',
  pre_restore: 'Before restore',
  scheduled: 'Hourly',
};

const ACTIVITY_TITLE: Partial<Record<SqliteBackupJob['trigger'], string>> = {
  scheduled: 'Hourly backup check',
  snapshot: 'Code snapshot backup',
  manual: 'Manual backup',
};

const JOB_STATUS_LABEL: Record<SqliteBackupJobStatus, string> = {
  pending: 'Pending',
  running: 'Running',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled',
  interrupted: 'Interrupted',
};

const ACTIVE_JOB_STATUSES = new Set<SqliteBackupJobStatus>(['pending', 'running']);

function isActiveJob(job: SqliteBackupJob): boolean {
  return ACTIVE_JOB_STATUSES.has(job.status);
}

function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function groupBackupsByDatabase(
  backups: SqliteHistoryBackup[],
): Array<[string, SqliteHistoryBackup[]]> {
  const groups = new Map<string, SqliteHistoryBackup[]>();
  for (const backup of backups) {
    const group = groups.get(backup.database_name) ?? [];
    group.push(backup);
    groups.set(backup.database_name, group);
  }
  return [...groups.entries()]
    .sort(([nameA], [nameB]) => nameA.localeCompare(nameB))
    .map(([name, backupsForDatabase]) => [
      name,
      backupsForDatabase.sort(
        (a, b) => Date.parse(b.created_at) - Date.parse(a.created_at) || b.id.localeCompare(a.id),
      ),
    ]);
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function safeId(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]/g, '-');
}

function getFocusableElements(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return Array.from(
    container.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  );
}

export function DatabaseHistoryPanel({
  workspaceId,
  ownerOrAdmin,
  databaseName,
  snapshotId,
  triggerLabel = 'Database history',
  hostId,
}: DatabaseHistoryPanelProps) {
  const [open, setOpen] = useState(false);
  const [history, setHistory] = useState<SqliteHistoryListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SqliteHistoryBackup | null>(null);
  const [mode, setMode] = useState<SqliteHistoryRestoreMode>('merge');
  const [policy, setPolicy] = useState<SqliteHistoryConflictPolicy>('keep_current');
  const [tablePolicies, setTablePolicies] = useState<Record<string, SqliteHistoryConflictPolicy>>(
    {},
  );
  const [preview, setPreview] = useState<SqliteHistoryPreview | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [capturingBackupId, setCapturingBackupId] = useState<string | null>(null);
  const [captureJobs, setCaptureJobs] = useState<SqliteBackupJob[]>([]);
  const [, setLoadingJobs] = useState(false);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [cancellingJobIds, setCancellingJobIds] = useState<Set<string>>(() => new Set());
  const [deletingBackupIds, setDeletingBackupIds] = useState<Set<string>>(() => new Set());
  const [downloadingBackupIds, setDownloadingBackupIds] = useState<Set<string>>(() => new Set());
  const [expandedDatabases, setExpandedDatabases] = useState<Set<string>>(() => new Set());
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const generationRef = useRef(0);
  const captureJobsGenerationRef = useRef(0);
  const captureJobsRevisionRef = useRef(0);
  const historyRevisionRef = useRef(0);
  const historyLoadingRequestsRef = useRef(0);
  const jobsLoadingRequestsRef = useRef(0);
  const captureJobsRef = useRef<SqliteBackupJob[]>([]);
  const previousActiveJobIdsRef = useRef<Set<string>>(new Set());
  const busyRef = useRef(false);

  const key = `${safeId(hostId)}-${safeId(workspaceId)}-${safeId(snapshotId ?? 'workspace')}-${safeId(databaseName ?? 'all')}`;
  const maintenance = history?.interrupted_maintenance ?? null;
  const maintenanceActive = maintenance?.state === 'active';
  const busy = confirming || recovering || maintenanceActive;
  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);
  const invalidatePreview = useCallback(() => {
    generationRef.current += 1;
    setPreview(null);
    setPreparing(false);
    setConfirming(false);
  }, []);
  const close = useCallback(() => {
    invalidatePreview();
    captureJobsGenerationRef.current += 1;
    captureJobsRevisionRef.current += 1;
    historyLoadingRequestsRef.current = 0;
    jobsLoadingRequestsRef.current = 0;
    setOpen(false);
  }, [invalidatePreview]);

  const clearRevokedHistory = useCallback(() => {
    invalidatePreview();
    historyRevisionRef.current += 1;
    captureJobsGenerationRef.current += 1;
    captureJobsRevisionRef.current += 1;
    historyLoadingRequestsRef.current = 0;
    jobsLoadingRequestsRef.current = 0;
    captureJobsRef.current = [];
    previousActiveJobIdsRef.current = new Set();
    setHistory(null);
    setCaptureJobs([]);
    setLoading(false);
    setLoadingJobs(false);
    setSelected(null);
    setReceipt(null);
    setError(null);
    setJobsError(null);
    setOpen(false);
  }, [invalidatePreview]);

  const load = useCallback(
    async (options?: { preserveError?: boolean; silent?: boolean }) => {
      const generation = generationRef.current;
      const revision = ++historyRevisionRef.current;
      if (!options?.silent) {
        historyLoadingRequestsRef.current += 1;
        setLoading(true);
      }
      if (!options?.preserveError && !options?.silent) setError(null);
      try {
        const result = await api.listUserSpaceSqliteHistory(workspaceId, {
          databaseName,
          snapshotId,
        });
        if (generation === generationRef.current && revision === historyRevisionRef.current)
          setHistory(result);
      } catch (caught) {
        if (
          generation === generationRef.current &&
          revision === historyRevisionRef.current &&
          !options?.silent
        )
          setError(caught instanceof Error ? caught.message : 'Unable to load database history.');
      } finally {
        if (generation === generationRef.current && !options?.silent) {
          historyLoadingRequestsRef.current = Math.max(0, historyLoadingRequestsRef.current - 1);
          setLoading(historyLoadingRequestsRef.current > 0);
        }
      }
    },
    [workspaceId, databaseName, snapshotId],
  );

  const loadJobs = useCallback(
    async (options?: { silent?: boolean }) => {
      const generation = captureJobsGenerationRef.current;
      const revision = ++captureJobsRevisionRef.current;
      if (!options?.silent) {
        jobsLoadingRequestsRef.current += 1;
        setLoadingJobs(true);
        setJobsError(null);
      }
      try {
        const result = await api.listUserSpaceSqliteBackupJobs(workspaceId, {
          databaseName,
          snapshotId,
        });
        if (
          generation !== captureJobsGenerationRef.current ||
          revision !== captureJobsRevisionRef.current
        )
          return;
        const priorActiveIds = previousActiveJobIdsRef.current;
        const completedActiveJob = result.jobs.some(
          (job) => priorActiveIds.has(job.id) && !isActiveJob(job),
        );
        previousActiveJobIdsRef.current = new Set(
          result.jobs.filter(isActiveJob).map((job) => job.id),
        );
        captureJobsRef.current = result.jobs;
        setCaptureJobs(result.jobs);
        if (completedActiveJob && !options?.silent) void load({ preserveError: true });
      } catch (caught) {
        if (
          generation === captureJobsGenerationRef.current &&
          revision === captureJobsRevisionRef.current
        ) {
          if (caught instanceof ApiError && caught.status === 403) {
            captureJobsRevisionRef.current += 1;
            captureJobsRef.current = [];
            previousActiveJobIdsRef.current = new Set();
            setCaptureJobs([]);
          }
          if (!options?.silent)
            setJobsError(caught instanceof Error ? caught.message : 'Unable to load capture jobs.');
        }
      } finally {
        if (generation === captureJobsGenerationRef.current && !options?.silent) {
          jobsLoadingRequestsRef.current = Math.max(0, jobsLoadingRequestsRef.current - 1);
          setLoadingJobs(jobsLoadingRequestsRef.current > 0);
        }
      }
    },
    [workspaceId, databaseName, snapshotId, load],
  );

  useEffect(() => {
    invalidatePreview();
    historyRevisionRef.current += 1;
    setSelected(null);
    setReceipt(null);
    setHistory(null);
    setError(null);
    setRecovering(false);
    setCapturingBackupId(null);
    captureJobsGenerationRef.current += 1;
    captureJobsRevisionRef.current += 1;
    historyLoadingRequestsRef.current = 0;
    jobsLoadingRequestsRef.current = 0;
    previousActiveJobIdsRef.current = new Set();
    captureJobsRef.current = [];
    setCaptureJobs([]);
    setLoadingJobs(false);
    setJobsError(null);
    setCancellingJobIds(new Set());
    setDeletingBackupIds(new Set());
    setDownloadingBackupIds(new Set());
    setExpandedDatabases(new Set());
  }, [workspaceId, databaseName, snapshotId, invalidatePreview]);

  useEffect(
    () => () => {
      captureJobsGenerationRef.current += 1;
      captureJobsRevisionRef.current += 1;
    },
    [],
  );

  useEffect(() => {
    if (history && !history.can_manage) {
      captureJobsGenerationRef.current += 1;
      captureJobsRevisionRef.current += 1;
      captureJobsRef.current = [];
      previousActiveJobIdsRef.current = new Set();
      setCaptureJobs([]);
    }
  }, [history]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  useEffect(() => {
    if (!open || !ownerOrAdmin) return;
    let active = true;
    let refreshing = false;
    let refreshQueued = false;
    const events = api.subscribeUserSpaceSqliteHistoryEvents(workspaceId, {
      databaseName,
      snapshotId,
    });
    const refresh = async () => {
      if (!active) return;
      if (refreshing) {
        refreshQueued = true;
        return;
      }
      refreshing = true;
      do {
        refreshQueued = false;
        await Promise.all([
          load({ preserveError: true, silent: true }),
          loadJobs({ silent: true }),
        ]);
      } while (active && refreshQueued);
      refreshing = false;
    };
    const revoke = () => {
      if (!active) return;
      active = false;
      events.close();
      clearRevokedHistory();
    };
    const historyChanged = () => void refresh();
    events.addEventListener('history_changed', historyChanged);
    events.addEventListener('access_revoked', revoke);
    void loadJobs();
    return () => {
      active = false;
      events.removeEventListener('history_changed', historyChanged);
      events.removeEventListener('access_revoked', revoke);
      events.close();
    };
  }, [
    open,
    ownerOrAdmin,
    workspaceId,
    databaseName,
    snapshotId,
    load,
    loadJobs,
    clearRevokedHistory,
  ]);

  useEffect(() => {
    if (!open) return;
    returnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const timer = window.setTimeout(() => closeRef.current?.focus(), 0);
    const handleKeyDown = (event: KeyboardEvent) => {
      const dialog = dialogRef.current;
      if (!dialog) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        if (!busyRef.current) {
          close();
        }
        return;
      }
      if (event.key !== 'Tab') return;
      const focusable = getFocusableElements(dialog);
      if (!focusable.length) return;
      const index = focusable.indexOf(document.activeElement as HTMLElement);
      const next = event.shiftKey
        ? index <= 0
          ? focusable.length - 1
          : index - 1
        : index === -1 || index === focusable.length - 1
          ? 0
          : index + 1;
      event.preventDefault();
      focusable[next]?.focus();
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.clearTimeout(timer);
      document.removeEventListener('keydown', handleKeyDown);
      returnFocusRef.current?.focus();
    };
  }, [open, close]);

  if (!ownerOrAdmin) return null;
  const canManage = history?.can_manage === true;

  const prepare = async () => {
    if (!selected || preparing || busy) return;
    const generation = generationRef.current;
    const request = { mode, conflict_policy: policy, table_policies: tablePolicies };
    setPreparing(true);
    setError(null);
    try {
      const result = await api.previewUserSpaceSqliteHistory(workspaceId, selected.id, request);
      if (generation === generationRef.current) setPreview(result);
    } catch (caught) {
      if (generation === generationRef.current)
        setError(caught instanceof Error ? caught.message : 'Unable to prepare restore preview.');
    } finally {
      if (generation === generationRef.current) setPreparing(false);
    }
  };

  const restore = async () => {
    if (!preview?.preview_id || confirming || recovering) return;
    const generation = generationRef.current;
    const previewId = preview.preview_id;
    setConfirming(true);
    setError(null);
    try {
      const result = await api.restoreUserSpaceSqliteHistory(workspaceId, previewId);
      if (generation === generationRef.current)
        setReceipt({
          safety_backup_id: result.safety_backup_id,
          operation_id: result.operation_id,
        });
    } catch (caught) {
      if (generation !== generationRef.current) return;
      if (caught instanceof ApiError && caught.status === 409) {
        invalidatePreview();
        setError(
          'This preview is stale because the database or migrations changed. Regenerate the preview.',
        );
      } else {
        setError(
          caught instanceof Error
            ? caught.message
            : 'Restore failed. Recovery details remain available.',
        );
      }
    } finally {
      if (generation === generationRef.current) setConfirming(false);
    }
  };

  const selectBackup = (backup: SqliteHistoryBackup) => {
    if (busy) return;
    invalidatePreview();
    setSelected(backup);
    setMode('merge');
    setPolicy('keep_current');
    setTablePolicies({});
    setReceipt(null);
    setError(null);
  };

  const changeContext = (
    nextMode?: SqliteHistoryRestoreMode,
    nextPolicy?: SqliteHistoryConflictPolicy,
  ) => {
    if (busy) return;
    invalidatePreview();
    if (nextMode) setMode(nextMode);
    if (nextPolicy) setPolicy(nextPolicy);
    setTablePolicies({});
    setError(null);
  };

  const recover = async (operationId: string | null, action: 'complete' | 'abort') => {
    if (busy || !operationId) return;
    const generation = generationRef.current;
    setRecovering(true);
    setError(null);
    try {
      await api.recoverUserSpaceSqliteHistoryMaintenance(workspaceId, operationId, action);
      if (generation === generationRef.current) await load();
    } catch (caught) {
      if (generation === generationRef.current)
        setError(caught instanceof Error ? caught.message : 'Recovery failed.');
    } finally {
      if (generation === generationRef.current) setRecovering(false);
    }
  };

  const capture = async () => {
    if (!databaseName || capturingBackupId || busy) return;
    const generation = captureJobsGenerationRef.current;
    setCapturingBackupId(databaseName);
    setError(null);
    try {
      const requestId = crypto.randomUUID();
      const result = await api.enqueueUserSpaceSqliteBackup(workspaceId, databaseName, requestId);
      if (generation === captureJobsGenerationRef.current) {
        const nextJobs = [
          result.job,
          ...captureJobsRef.current.filter((job) => job.id !== result.job.id),
        ];
        captureJobsRevisionRef.current += 1;
        captureJobsRef.current = nextJobs;
        setCaptureJobs(nextJobs);
        previousActiveJobIdsRef.current = new Set([
          ...previousActiveJobIdsRef.current,
          result.job.id,
        ]);
      }
    } catch (caught) {
      if (generation === captureJobsGenerationRef.current)
        setError(caught instanceof Error ? caught.message : 'Capture failed.');
    } finally {
      if (generation === captureJobsGenerationRef.current) setCapturingBackupId(null);
    }
  };

  const cancelCaptureJob = async (jobId: string) => {
    if (cancellingJobIds.has(jobId)) return;
    const generation = captureJobsGenerationRef.current;
    captureJobsRevisionRef.current += 1;
    setCancellingJobIds((ids) => new Set(ids).add(jobId));
    setJobsError(null);
    try {
      const result = await api.cancelUserSpaceSqliteBackupJob(workspaceId, jobId);
      if (generation === captureJobsGenerationRef.current)
        setCaptureJobs((jobs) => {
          const nextJobs = jobs.map((job) => (job.id === jobId ? result.job : job));
          captureJobsRevisionRef.current += 1;
          captureJobsRef.current = nextJobs;
          return nextJobs;
        });
    } catch (caught) {
      if (generation === captureJobsGenerationRef.current)
        setJobsError(caught instanceof Error ? caught.message : 'Unable to cancel capture job.');
    } finally {
      if (generation === captureJobsGenerationRef.current)
        setCancellingJobIds((ids) => {
          const next = new Set(ids);
          next.delete(jobId);
          return next;
        });
    }
  };

  const activeCaptureJobs = captureJobs
    .filter(isActiveJob)
    .sort(
      (a, b) => Date.parse(a.created_at) - Date.parse(b.created_at) || a.id.localeCompare(b.id),
    );
  const terminalCaptureJobs = captureJobs
    .filter((job) => !isActiveJob(job))
    .sort(
      (a, b) =>
        Date.parse(b.finished_at ?? b.updated_at) - Date.parse(a.finished_at ?? a.updated_at) ||
        b.id.localeCompare(a.id),
    );
  const attentionCaptureJobs = terminalCaptureJobs.filter(
    (job) => job.status === 'failed' || job.status === 'interrupted',
  );
  const activityCaptureJobs = terminalCaptureJobs.filter(
    (job) => job.status === 'completed' || job.status === 'cancelled',
  );

  const renderCaptureJob = (job: SqliteBackupJob) => {
    const requestedDatabases = job.database_names.length
      ? job.database_names.join(', ')
      : 'All workspace databases';
    const hasProgress = job.total_databases > 0;
    const isCancelling = cancellingJobIds.has(job.id);
    return (
      <article
        key={job.id}
        className="database-history-backup database-history-capture-job"
        data-history-capture-job={job.id}
        data-history-capture-status={job.status}
      >
        <div>
          <strong>{requestedDatabases}</strong>
          <span>
            {TRIGGER_LABEL[job.trigger]} · Requested {new Date(job.created_at).toLocaleString()}
            {job.started_at && ` · Started ${new Date(job.started_at).toLocaleString()}`}
            {job.finished_at && ` · Finished ${new Date(job.finished_at).toLocaleString()}`}
          </span>
          {hasProgress && (
            <span>
              {job.completed_databases}/{job.total_databases} databases processed
            </span>
          )}
          {isActiveJob(job) && (
            <span className="userspace-muted">
              Captures current database when the worker executes, not when requested.
            </span>
          )}
          {job.backup_ids.length > 0 && job.status !== 'completed' && (
            <span className="userspace-muted">
              Partial result: {pluralize(job.backup_ids.length, 'backup record')} available in
              history.
            </span>
          )}
          {job.error_message && <span className="database-history-error">{job.error_message}</span>}
        </div>
        <div className="database-history-actions">
          <span
            className={`badge database-history-trigger-badge database-history-job-badge--${job.status}`}
          >
            {JOB_STATUS_LABEL[job.status]}
          </span>
          {isActiveJob(job) && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={isCancelling}
              data-history-cancel-job={job.id}
              onClick={() => void cancelCaptureJob(job.id)}
            >
              {isCancelling
                ? 'Cancelling…'
                : job.status === 'running'
                  ? 'Cancel (after current DB)'
                  : 'Cancel'}
            </button>
          )}
        </div>
      </article>
    );
  };

  const renderActivityRow = (job: SqliteBackupJob) => {
    const title = ACTIVITY_TITLE[job.trigger] ?? TRIGGER_LABEL[job.trigger];
    const scope = job.database_names.length
      ? job.database_names.join(', ')
      : 'All workspace databases';
    const hasRestorePoints = job.backup_ids.length > 0;
    const outcome =
      job.status === 'completed'
        ? hasRestorePoints
          ? `${pluralize(job.backup_ids.length, 'restore point')} created`
          : 'No new restore point created'
        : hasRestorePoints
          ? `${pluralize(job.backup_ids.length, 'restore point')} created before cancellation`
          : 'Cancelled before any restore point';
    return (
      <article
        key={job.id}
        className="database-history-activity database-history-backup"
        data-history-activity-job={job.id}
        data-history-activity-status={job.status}
      >
        <div>
          <strong>{title}</strong>
          <span>{scope}</span>
          <span>{outcome}</span>
          <span>{new Date(job.finished_at ?? job.updated_at).toLocaleString()}</span>
        </div>
        <div className="database-history-actions">
          <span
            className={`badge database-history-trigger-badge database-history-job-badge--${job.status}`}
          >
            {JOB_STATUS_LABEL[job.status]}
          </span>
        </div>
      </article>
    );
  };

  const download = async (backupId: string) => {
    if (busy || downloadingBackupIds.has(backupId)) return;
    const generation = generationRef.current;
    setDownloadingBackupIds((ids) => new Set(ids).add(backupId));
    setError(null);
    try {
      await api.downloadUserSpaceSqliteHistory(workspaceId, backupId);
    } catch (caught) {
      if (generation === generationRef.current)
        setError(caught instanceof Error ? caught.message : 'Database history download failed.');
    } finally {
      if (generation === generationRef.current)
        setDownloadingBackupIds((ids) => {
          const next = new Set(ids);
          next.delete(backupId);
          return next;
        });
    }
  };

  const remove = async (backupId: string) => {
    if (busy || deletingBackupIds.has(backupId)) return;
    const generation = generationRef.current;
    setDeletingBackupIds((ids) => new Set(ids).add(backupId));
    setError(null);
    try {
      await api.deleteUserSpaceSqliteHistory(workspaceId, backupId);
      if (generation !== generationRef.current) return;
      if (selected?.id === backupId) {
        invalidatePreview();
        setSelected(null);
        setReceipt(null);
      }
      await load();
    } catch (caught) {
      if (generation === generationRef.current)
        setError(caught instanceof Error ? caught.message : 'Delete failed.');
    } finally {
      if (generation === generationRef.current)
        setDeletingBackupIds((ids) => {
          const next = new Set(ids);
          next.delete(backupId);
          return next;
        });
    }
  };

  return (
    <>
      <button
        type="button"
        className="btn btn-secondary btn-sm database-history-trigger"
        data-history-workspace={workspaceId}
        data-history-snapshot={snapshotId ?? 'workspace'}
        data-history-database={databaseName ?? 'all'}
        data-history-host={hostId}
        data-history-panel={key}
        onClick={() => setOpen(true)}
      >
        <DatabaseBackup size={14} /> {triggerLabel}
      </button>
      {open && (
        <div
          id={`database-history-dialog-${key}`}
          className="modal-overlay database-history-overlay"
          role="dialog"
          aria-modal="true"
          aria-labelledby={`database-history-title-${key}`}
          ref={dialogRef}
          data-history-dialog={key}
        >
          <section
            className="modal modal-large database-history-panel"
            data-history-panel={key}
            data-history-host={hostId}
          >
            <header className="modal-header">
              <div>
                <h3 id={`database-history-title-${key}`}>Database history</h3>
                <p className="userspace-muted">
                  {snapshotId
                    ? `Exact snapshot ${snapshotId}`
                    : databaseName
                      ? `Restore points for ${databaseName}`
                      : 'Restore points for all workspace databases'}
                </p>
              </div>
              <button
                ref={closeRef}
                type="button"
                className="modal-close"
                onClick={close}
                disabled={busy}
                aria-label="Close database history"
              >
                <X size={18} />
              </button>
            </header>
            <div className="modal-body database-history-body" data-history-content={key}>
              {loading && (
                <p role="status">
                  <Loader2 size={14} /> Loading database history…
                </p>
              )}
              {busy && (
                <p
                  className="database-history-maintenance-busy"
                  role="status"
                  data-history-maintenance-status={maintenanceActive ? 'active' : 'busy'}
                >
                  {maintenanceActive
                    ? 'Active database maintenance is in progress. Restore, recovery, and destructive actions are unavailable.'
                    : 'Database maintenance is active. Restore, recovery, and destructive actions are temporarily unavailable.'}
                </p>
              )}
              {error && (
                <p className="database-history-error" role="alert">
                  {error}
                </p>
              )}
              {jobsError && (
                <p className="database-history-error" role="alert">
                  {jobsError}
                </p>
              )}
              {history && !canManage && (
                <p className="database-history-error" role="alert">
                  History is available only to the workspace owner or an administrator.
                </p>
              )}
              {((maintenance && !maintenanceActive && canManage) ||
                attentionCaptureJobs.length > 0) && (
                <section
                  className="database-history-band"
                  aria-labelledby={`db-hist-band-attention-${key}`}
                  data-history-band="attention"
                >
                  <h4
                    id={`db-hist-band-attention-${key}`}
                    className="database-history-band-heading"
                  >
                    Needs attention
                  </h4>
                  {maintenance && !maintenanceActive && canManage && (
                    <section
                      className="database-history-recovery"
                      aria-label="Interrupted database maintenance"
                      data-history-recovery
                      data-history-maintenance-status={maintenance.state}
                    >
                      <AlertTriangle size={16} />
                      <div>
                        <strong>Interrupted database maintenance</strong>
                        <p>
                          {maintenance.detail ??
                            'Runtime remains stopped until this operation is recovered.'}
                        </p>
                      </div>
                      {maintenance.operation_id && maintenance.can_complete && (
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          disabled={busy}
                          onClick={() => void recover(maintenance.operation_id, 'complete')}
                        >
                          Complete
                        </button>
                      )}
                      {maintenance.operation_id && maintenance.can_abort && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          disabled={busy}
                          onClick={() => void recover(maintenance.operation_id, 'abort')}
                        >
                          Abort
                        </button>
                      )}
                    </section>
                  )}
                  {attentionCaptureJobs.map(renderCaptureJob)}
                </section>
              )}
              {activeCaptureJobs.length > 0 && (
                <section
                  className="database-history-band database-history-capture-jobs"
                  aria-labelledby={`db-hist-band-in-progress-${key}`}
                  data-history-band="in-progress"
                  data-history-capture-jobs
                >
                  <h4
                    id={`db-hist-band-in-progress-${key}`}
                    className="database-history-band-heading"
                  >
                    In progress
                  </h4>
                  {activeCaptureJobs.map(renderCaptureJob)}
                </section>
              )}
              {canManage &&
                history &&
                (() => {
                  const databaseGroups = groupBackupsByDatabase(history.backups);
                  const readyBackups = history.backups.filter(
                    (backup) => backup.status === 'ready',
                  ).length;
                  return (
                    <section
                      className="database-history-band"
                      aria-labelledby={`db-hist-band-recoverable-${key}`}
                      data-history-band="recoverable"
                    >
                      <div className="database-history-band-header">
                        <h4
                          id={`db-hist-band-recoverable-${key}`}
                          className="database-history-band-heading"
                        >
                          Restore points
                        </h4>
                        <span className="database-history-band-count">
                          {pluralize(readyBackups, 'restore point')}
                        </span>
                        {databaseName && (
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            disabled={Boolean(capturingBackupId) || busy}
                            onClick={() => void capture()}
                          >
                            <Plus size={14} /> {capturingBackupId ? 'Capturing…' : 'Capture now'}
                          </button>
                        )}
                      </div>
                      {databaseGroups.map(([name, backups]) => {
                        const isExpanded = expandedDatabases.has(name);
                        const visibleBackups = isExpanded ? backups : backups.slice(0, 5);
                        const olderCount = Math.max(0, backups.length - 5);
                        const headingId = `db-hist-db-${safeId(name)}-${key}`;
                        return (
                          <section
                            key={name}
                            className="database-history-database-group"
                            aria-labelledby={headingId}
                            data-history-database-group={name}
                          >
                            <div className="database-history-database-header">
                              <h4 id={headingId} className="database-history-database-heading">
                                {name}
                              </h4>
                              <span className="database-history-database-count">
                                {pluralize(
                                  backups.filter((backup) => backup.status === 'ready').length,
                                  'restore point',
                                )}
                              </span>
                              <span className="database-history-database-latest">
                                Latest {new Date(backups[0].created_at).toLocaleString()}
                              </span>
                            </div>
                            {visibleBackups.map((backup) => (
                              <article
                                key={backup.id}
                                className="database-history-backup"
                                data-history-backup={backup.id}
                                data-history-trigger={backup.trigger}
                              >
                                <div>
                                  <strong>{new Date(backup.created_at).toLocaleString()}</strong>
                                  <span className="database-history-backup-name">
                                    {backup.database_name}
                                  </span>
                                  <span
                                    className={`badge database-history-trigger-badge database-history-trigger-badge--${TRIGGER_GROUP[backup.trigger]}`}
                                  >
                                    {TRIGGER_LABEL[backup.trigger]}
                                  </span>
                                  <span>{formatBytes(backup.size_bytes)}</span>
                                  {backup.snapshot_id && <span>Snapshot {backup.snapshot_id}</span>}
                                  {backup.status === 'failed' && (
                                    <span className="database-history-error">
                                      Capture failed:{' '}
                                      {backup.error ?? 'No recovery blob was captured.'}
                                    </span>
                                  )}
                                </div>
                                <div
                                  className="database-history-actions"
                                  data-history-actions={backup.id}
                                >
                                  <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    disabled={busy || downloadingBackupIds.has(backup.id)}
                                    onClick={() => void download(backup.id)}
                                    aria-label={`Download ${backup.database_name} backup`}
                                  >
                                    <Download size={14} />
                                  </button>
                                  {backup.can_restore && (
                                    <button
                                      type="button"
                                      className="btn btn-primary btn-sm"
                                      disabled={busy}
                                      onClick={() => selectBackup(backup)}
                                      aria-label={`Restore ${backup.database_name} backup from ${backup.trigger}${backup.snapshot_id ? ` snapshot ${backup.snapshot_id}` : ''}`}
                                    >
                                      Restore
                                    </button>
                                  )}
                                  {backup.can_delete && (
                                    <button
                                      type="button"
                                      className="btn btn-secondary btn-sm"
                                      disabled={busy || deletingBackupIds.has(backup.id)}
                                      onClick={() => void remove(backup.id)}
                                      aria-label={`Delete ${backup.database_name} backup`}
                                    >
                                      <Trash2 size={14} />
                                    </button>
                                  )}
                                </div>
                              </article>
                            ))}
                            {olderCount > 0 && (
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                data-history-show-older={name}
                                onClick={() =>
                                  setExpandedDatabases((names) => {
                                    const next = new Set(names);
                                    if (next.has(name)) next.delete(name);
                                    else next.add(name);
                                    return next;
                                  })
                                }
                              >
                                {isExpanded
                                  ? 'Show fewer restore points'
                                  : `Show ${pluralize(olderCount, 'older restore point')}`}
                              </button>
                            )}
                          </section>
                        );
                      })}
                    </section>
                  );
                })()}
              {history &&
                canManage &&
                history.backups.length === 0 &&
                (snapshotId && activeCaptureJobs.length > 0 ? (
                  <p className="userspace-muted" data-history-snapshot-queue-notice>
                    A capture for this snapshot is queued. Database state is captured when the
                    worker executes, not at enqueue time.
                  </p>
                ) : (
                  <p className="userspace-muted">
                    {snapshotId
                      ? 'Database history was not captured for this snapshot.'
                      : 'No captured database backups. Missing live databases can still be recovered here once a backup exists.'}
                  </p>
                ))}
              {activityCaptureJobs.length > 0 && (
                <section
                  className="database-history-band"
                  aria-labelledby={`db-hist-band-activity-${key}`}
                  data-history-band="activity"
                >
                  <h4 id={`db-hist-band-activity-${key}`} className="database-history-band-heading">
                    Activity
                  </h4>
                  <details>
                    <summary>
                      Capture activity ({pluralize(activityCaptureJobs.length, 'run')})
                    </summary>
                    <div className="database-history-activity-list">
                      {activityCaptureJobs.map(renderActivityRow)}
                    </div>
                  </details>
                </section>
              )}
              {selected && !receipt && (
                <section
                  className="database-history-wizard"
                  aria-label="Restore database backup"
                  data-history-restore-wizard
                >
                  <h4>Restore {selected.database_name}</h4>
                  <label>
                    Mode{' '}
                    <select
                      value={mode}
                      disabled={busy}
                      onChange={(event) =>
                        changeContext(event.target.value as SqliteHistoryRestoreMode)
                      }
                    >
                      <option value="merge">Merge</option>
                      <option value="overwrite">Overwrite</option>
                    </select>
                  </label>
                  {mode === 'merge' && (
                    <label>
                      Default conflict policy{' '}
                      <select
                        value={policy}
                        disabled={busy}
                        onChange={(event) =>
                          changeContext(
                            undefined,
                            event.target.value as SqliteHistoryConflictPolicy,
                          )
                        }
                      >
                        <option value="keep_current">Keep current</option>
                        <option value="use_backup">Use backup</option>
                      </select>
                    </label>
                  )}
                  {mode === 'merge' ? (
                    <p className="database-history-warning">
                      Merge can resurrect rows deliberately deleted from the current database.
                    </p>
                  ) : (
                    <p className="database-history-warning">Overwrite removes current-only data.</p>
                  )}
                  <button
                    type="button"
                    className="btn btn-primary"
                    disabled={preparing || busy}
                    onClick={() => void prepare()}
                  >
                    {preparing ? 'Preparing actual preview…' : 'Prepare preview'}
                  </button>
                  {preview && (
                    <div
                      className="database-history-preview"
                      data-history-preview={preview.preview_id ?? 'unavailable'}
                    >
                      <h4>Actual restore preview</h4>
                      <p>
                        Migrations:{' '}
                        {preview.migrations_applied.length
                          ? preview.migrations_applied.join(', ')
                          : 'none'}
                      </p>
                      {preview.warnings.map((warning) => (
                        <p key={warning} className="database-history-warning">
                          {warning}
                        </p>
                      ))}
                      {preview.blockers.map((blocker) => (
                        <p key={blocker} className="database-history-error">
                          {blocker}
                        </p>
                      ))}
                      {preview.tables.map((table) => (
                        <details key={table.name}>
                          <summary>
                            {table.name}: {table.inserted} inserted, {table.updated} updated,{' '}
                            {table.deleted} deleted, {table.conflicts} conflicts
                          </summary>
                          {mode === 'merge' && (
                            <label>
                              Table conflict policy ({table.name}){' '}
                              <select
                                value={tablePolicies[table.name] ?? policy}
                                disabled={busy}
                                onChange={(event) => {
                                  if (busy) return;
                                  invalidatePreview();
                                  setError(null);
                                  setTablePolicies((policies) => ({
                                    ...policies,
                                    [table.name]: event.target.value as SqliteHistoryConflictPolicy,
                                  }));
                                }}
                              >
                                <option value="keep_current">Keep current</option>
                                <option value="use_backup">Use backup</option>
                              </select>
                            </label>
                          )}
                          {table.conflict_samples.slice(0, 20).map((sample) => (
                            <pre key={JSON.stringify([table.name, sample.key])}>
                              {JSON.stringify(sample, null, 2)}
                            </pre>
                          ))}
                        </details>
                      ))}
                      {preview.can_apply && preview.preview_id && (
                        <button
                          type="button"
                          className="btn btn-primary"
                          disabled={busy}
                          onClick={() => void restore()}
                        >
                          <RotateCcw size={14} /> {confirming ? 'Restoring…' : 'Confirm restore'}
                        </button>
                      )}
                    </div>
                  )}
                </section>
              )}
              {receipt && (
                <section className="database-history-receipt" role="status" data-history-receipt>
                  <h4>Database restored</h4>
                  {receipt.safety_backup_id ? (
                    <p>
                      Safety backup: {receipt.safety_backup_id}{' '}
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        disabled={busy || downloadingBackupIds.has(receipt.safety_backup_id)}
                        onClick={() => void download(receipt.safety_backup_id!)}
                      >
                        Download safety backup
                      </button>
                    </p>
                  ) : (
                    <p>Safety backup: No safety backup created</p>
                  )}
                  <p>
                    The runtime is stopped. Start the preview when you are ready; no bootstrap or
                    migrations were run automatically.
                  </p>
                </section>
              )}
            </div>
          </section>
        </div>
      )}
    </>
  );
}
