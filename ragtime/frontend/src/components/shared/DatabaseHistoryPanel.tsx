import { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, DatabaseBackup, Download, Loader2, Plus, RotateCcw, Trash2, X } from 'lucide-react';

import { api, ApiError } from '@/api/client';
import type {
  SqliteHistoryBackup,
  SqliteHistoryConflictPolicy,
  SqliteHistoryListResponse,
  SqliteHistoryPreview,
  SqliteHistoryRestoreMode,
  SqliteHistoryBackupTrigger,
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

const GROUP_ORDER = ['checkpoint', 'safety', 'hourly'] as const;

type GroupKey = typeof GROUP_ORDER[number];

const GROUP_META: Record<GroupKey, { heading: string }> = {
  checkpoint: { heading: 'Snapshot & manual backups' },
  safety: { heading: 'Restore safety backups' },
  hourly: { heading: 'Hourly backups' },
};

const TRIGGER_LABEL: Record<SqliteHistoryBackupTrigger, string> = {
  snapshot: 'Code snapshot',
  manual: 'Manual backup',
  pre_restore: 'Before restore',
  scheduled: 'Hourly',
};

function groupBackups(backups: SqliteHistoryBackup[]): Partial<Record<GroupKey, SqliteHistoryBackup[]>> {
  const groups: Partial<Record<GroupKey, SqliteHistoryBackup[]>> = {};
  for (const backup of [...backups].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at) || b.id.localeCompare(a.id))) {
    const group = TRIGGER_GROUP[backup.trigger] as GroupKey;
    (groups[group] ??= []).push(backup);
  }
  return groups;
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
  return Array.from(container.querySelectorAll<HTMLElement>(
    'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
  ));
}

export function DatabaseHistoryPanel({
  workspaceId, ownerOrAdmin, databaseName, snapshotId, triggerLabel = 'Database history', hostId,
}: DatabaseHistoryPanelProps) {
  const [open, setOpen] = useState(false);
  const [history, setHistory] = useState<SqliteHistoryListResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<SqliteHistoryBackup | null>(null);
  const [mode, setMode] = useState<SqliteHistoryRestoreMode>('merge');
  const [policy, setPolicy] = useState<SqliteHistoryConflictPolicy>('keep_current');
  const [tablePolicies, setTablePolicies] = useState<Record<string, SqliteHistoryConflictPolicy>>({});
  const [preview, setPreview] = useState<SqliteHistoryPreview | null>(null);
  const [preparing, setPreparing] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [recovering, setRecovering] = useState(false);
  const [receipt, setReceipt] = useState<Receipt | null>(null);
  const [capturingBackupId, setCapturingBackupId] = useState<string | null>(null);
  const [deletingBackupIds, setDeletingBackupIds] = useState<Set<string>>(() => new Set());
  const [downloadingBackupIds, setDownloadingBackupIds] = useState<Set<string>>(() => new Set());
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const generationRef = useRef(0);
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
    setOpen(false);
  }, [invalidatePreview]);

  const load = useCallback(async () => {
    const generation = generationRef.current;
    setLoading(true);
    setError(null);
    try {
      const result = await api.listUserSpaceSqliteHistory(workspaceId, { databaseName, snapshotId });
      if (generation === generationRef.current) setHistory(result);
    } catch (caught) {
      if (generation === generationRef.current) setError(caught instanceof Error ? caught.message : 'Unable to load database history.');
    } finally {
      if (generation === generationRef.current) setLoading(false);
    }
  }, [workspaceId, databaseName, snapshotId]);

  useEffect(() => {
    invalidatePreview();
    setSelected(null);
    setReceipt(null);
    setHistory(null);
    setError(null);
    setRecovering(false);
    setCapturingBackupId(null);
    setDeletingBackupIds(new Set());
    setDownloadingBackupIds(new Set());
  }, [workspaceId, databaseName, snapshotId, invalidatePreview]);

  useEffect(() => {
    if (open) void load();
  }, [open, load]);

  useEffect(() => {
    if (!open) return;
    returnFocusRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
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
        ? index <= 0 ? focusable.length - 1 : index - 1
        : index === -1 || index === focusable.length - 1 ? 0 : index + 1;
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
      if (generation === generationRef.current) setError(caught instanceof Error ? caught.message : 'Unable to prepare restore preview.');
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
      if (generation === generationRef.current) setReceipt({ safety_backup_id: result.safety_backup_id, operation_id: result.operation_id });
    } catch (caught) {
      if (generation !== generationRef.current) return;
      if (caught instanceof ApiError && caught.status === 409) {
        invalidatePreview();
        setError('This preview is stale because the database or migrations changed. Regenerate the preview.');
      } else {
        setError(caught instanceof Error ? caught.message : 'Restore failed. Recovery details remain available.');
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

  const changeContext = (nextMode?: SqliteHistoryRestoreMode, nextPolicy?: SqliteHistoryConflictPolicy) => {
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
      if (generation === generationRef.current) setError(caught instanceof Error ? caught.message : 'Recovery failed.');
    } finally {
      if (generation === generationRef.current) setRecovering(false);
    }
  };

  const capture = async () => {
    if (!databaseName || capturingBackupId || busy) return;
    const generation = generationRef.current;
    setCapturingBackupId(databaseName);
    setError(null);
    try {
      await api.captureUserSpaceSqliteHistory(workspaceId, databaseName);
      if (generation === generationRef.current) await load();
    } catch (caught) {
      if (generation === generationRef.current) setError(caught instanceof Error ? caught.message : 'Capture failed.');
    } finally {
      if (generation === generationRef.current) setCapturingBackupId(null);
    }
  };

  const download = async (backupId: string) => {
    if (busy || downloadingBackupIds.has(backupId)) return;
    const generation = generationRef.current;
    setDownloadingBackupIds((ids) => new Set(ids).add(backupId));
    setError(null);
    try {
      await api.downloadUserSpaceSqliteHistory(workspaceId, backupId);
    } catch (caught) {
      if (generation === generationRef.current) setError(caught instanceof Error ? caught.message : 'Database history download failed.');
    } finally {
      if (generation === generationRef.current) setDownloadingBackupIds((ids) => { const next = new Set(ids); next.delete(backupId); return next; });
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
      if (generation === generationRef.current) setError(caught instanceof Error ? caught.message : 'Delete failed.');
    } finally {
      if (generation === generationRef.current) setDeletingBackupIds((ids) => { const next = new Set(ids); next.delete(backupId); return next; });
    }
  };

  return <>
    <button type="button" className="btn btn-secondary btn-sm database-history-trigger" data-history-workspace={workspaceId} data-history-snapshot={snapshotId ?? 'workspace'} data-history-database={databaseName ?? 'all'} data-history-host={hostId} data-history-panel={key} onClick={() => setOpen(true)}>
      <DatabaseBackup size={14} /> {triggerLabel}
    </button>
    {open && <div id={`database-history-dialog-${key}`} className="modal-overlay database-history-overlay" role="dialog" aria-modal="true" aria-labelledby={`database-history-title-${key}`} ref={dialogRef} data-history-dialog={key}>
      <section className="modal modal-large database-history-panel" data-history-panel={key} data-history-host={hostId}>
        <header className="modal-header">
          <div><h3 id={`database-history-title-${key}`}>Database history</h3><p className="userspace-muted">{snapshotId ? `Exact snapshot ${snapshotId}` : databaseName ? databaseName : 'All workspace databases'}</p></div>
          <button ref={closeRef} type="button" className="modal-close" onClick={close} disabled={busy} aria-label="Close database history"><X size={18} /></button>
        </header>
        <div className="modal-body database-history-body" data-history-content={key}>
          {loading && <p role="status"><Loader2 size={14} /> Loading database history…</p>}
          {busy && <p className="database-history-maintenance-busy" role="status" data-history-maintenance-status={maintenanceActive ? 'active' : 'busy'}>{maintenanceActive ? 'Active database maintenance is in progress. Restore, recovery, and destructive actions are unavailable.' : 'Database maintenance is active. Restore, recovery, and destructive actions are temporarily unavailable.'}</p>}
          {error && <p className="database-history-error" role="alert">{error}</p>}
          {history && !canManage && <p className="database-history-error" role="alert">History is available only to the workspace owner or an administrator.</p>}
          {maintenance && !maintenanceActive && canManage && <section className="database-history-recovery" aria-label="Interrupted database maintenance" data-history-recovery data-history-maintenance-status={maintenance.state}>
            <AlertTriangle size={16} /><div><strong>Interrupted database maintenance</strong><p>{maintenance.detail ?? 'Runtime remains stopped until this operation is recovered.'}</p></div>
            {maintenance.operation_id && maintenance.can_complete && <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => void recover(maintenance.operation_id, 'complete')}>Complete</button>}
            {maintenance.operation_id && maintenance.can_abort && <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => void recover(maintenance.operation_id, 'abort')}>Abort</button>}
          </section>}
          {canManage && databaseName && <button type="button" className="btn btn-secondary btn-sm" disabled={Boolean(capturingBackupId) || busy} onClick={() => void capture()}><Plus size={14} /> {capturingBackupId ? 'Capturing…' : 'Capture now'}</button>}
          {canManage && history && (() => {
            const groups = groupBackups(history.backups);
            return GROUP_ORDER.map((group) => {
              const backups = groups[group];
              if (!backups?.length) return null;
              const headingId = `db-hist-group-${group}-${key}`;
              return <section key={group} className={`database-history-group database-history-group--${group}`} aria-labelledby={headingId} data-history-group={group}>
                <h4 id={headingId} className="database-history-group-heading">{GROUP_META[group].heading}<span className={`badge database-history-trigger-badge database-history-trigger-badge--${group}`}>{backups.length}</span></h4>
                {backups.map((backup) => <article key={backup.id} className="database-history-backup" data-history-backup={backup.id} data-history-trigger={backup.trigger}>
                  <div><strong>{backup.database_name}</strong><span>{new Date(backup.created_at).toLocaleString()} · {TRIGGER_LABEL[backup.trigger]} · {formatBytes(backup.size_bytes)}</span>{backup.snapshot_id && <span>Snapshot {backup.snapshot_id}</span>}{backup.status === 'failed' && <span className="database-history-error">Capture failed: {backup.error ?? 'No recovery blob was captured.'}</span>}</div>
                  <div className="database-history-actions" data-history-actions={backup.id}>
                    <button type="button" className="btn btn-secondary btn-sm" disabled={busy || downloadingBackupIds.has(backup.id)} onClick={() => void download(backup.id)} aria-label={`Download ${backup.database_name} backup`}><Download size={14} /></button>
                    {backup.can_restore && <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => selectBackup(backup)} aria-label={`Restore ${backup.database_name} backup from ${backup.trigger}${backup.snapshot_id ? ` snapshot ${backup.snapshot_id}` : ''}`}>Restore</button>}
                    {backup.can_delete && <button type="button" className="btn btn-secondary btn-sm" disabled={busy || deletingBackupIds.has(backup.id)} onClick={() => void remove(backup.id)} aria-label={`Delete ${backup.database_name} backup`}><Trash2 size={14} /></button>}
                  </div>
                </article>)}
              </section>;
            });
          })()}
          {history && canManage && history.backups.length === 0 && <p className="userspace-muted">{snapshotId ? 'Database history was not captured for this snapshot.' : 'No captured database backups. Missing live databases can still be recovered here once a backup exists.'}</p>}
          {selected && !receipt && <section className="database-history-wizard" aria-label="Restore database backup" data-history-restore-wizard>
            <h4>Restore {selected.database_name}</h4>
            <label>Mode <select value={mode} disabled={busy} onChange={(event) => changeContext(event.target.value as SqliteHistoryRestoreMode)}><option value="merge">Merge</option><option value="overwrite">Overwrite</option></select></label>
            {mode === 'merge' && <label>Default conflict policy <select value={policy} disabled={busy} onChange={(event) => changeContext(undefined, event.target.value as SqliteHistoryConflictPolicy)}><option value="keep_current">Keep current</option><option value="use_backup">Use backup</option></select></label>}
            {mode === 'merge' ? <p className="database-history-warning">Merge can resurrect rows deliberately deleted from the current database.</p> : <p className="database-history-warning">Overwrite removes current-only data.</p>}
            <button type="button" className="btn btn-primary" disabled={preparing || busy} onClick={() => void prepare()}>{preparing ? 'Preparing actual preview…' : 'Prepare preview'}</button>
            {preview && <div className="database-history-preview" data-history-preview={preview.preview_id ?? 'unavailable'}><h4>Actual restore preview</h4><p>Migrations: {preview.migrations_applied.length ? preview.migrations_applied.join(', ') : 'none'}</p>{preview.warnings.map((warning) => <p key={warning} className="database-history-warning">{warning}</p>)}{preview.blockers.map((blocker) => <p key={blocker} className="database-history-error">{blocker}</p>)}{preview.tables.map((table) => <details key={table.name}><summary>{table.name}: {table.inserted} inserted, {table.updated} updated, {table.deleted} deleted, {table.conflicts} conflicts</summary>{mode === 'merge' && <label>Table conflict policy ({table.name}) <select value={tablePolicies[table.name] ?? policy} disabled={busy} onChange={(event) => { if (busy) return; invalidatePreview(); setError(null); setTablePolicies((policies) => ({ ...policies, [table.name]: event.target.value as SqliteHistoryConflictPolicy })); }}><option value="keep_current">Keep current</option><option value="use_backup">Use backup</option></select></label>}{table.conflict_samples.slice(0, 20).map((sample) => <pre key={JSON.stringify([table.name, sample.key])}>{JSON.stringify(sample, null, 2)}</pre>)}</details>)}{preview.can_apply && preview.preview_id && <button type="button" className="btn btn-primary" disabled={busy} onClick={() => void restore()}><RotateCcw size={14} /> {confirming ? 'Restoring…' : 'Confirm restore'}</button>}</div>}
          </section>}
          {receipt && <section className="database-history-receipt" role="status" data-history-receipt><h4>Database restored</h4>{receipt.safety_backup_id ? <p>Safety backup: {receipt.safety_backup_id} <button type="button" className="btn btn-secondary btn-sm" disabled={busy || downloadingBackupIds.has(receipt.safety_backup_id)} onClick={() => void download(receipt.safety_backup_id!)}>Download safety backup</button></p> : <p>Safety backup: No safety backup created</p>}<p>The runtime is stopped. Start the preview when you are ready; no bootstrap or migrations were run automatically.</p></section>}
        </div>
      </section>
    </div>}
  </>;
}
