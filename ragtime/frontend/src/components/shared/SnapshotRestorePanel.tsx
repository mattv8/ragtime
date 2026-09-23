import { useEffect, useId, useRef, useState } from 'react';
import { Loader2 } from 'lucide-react';

import { api, ApiError } from '@/api/client';
import type {
  SqliteHistoryBackup,
  SqliteHistoryConflictPolicy,
  SqliteHistoryMaintenanceStateResponse,
  SqliteHistoryPreview,
  SqliteHistoryRestoreMode,
} from '@/types';
import { subscribeHistoryEvents } from '@/utils/sqliteHistoryEventBus';

export interface SnapshotRestorePanelProps {
  workspaceId: string;
  snapshotId: string;
  initialBackupId?: string;
  defaultScope: 'code' | 'database';
  allowCodeRestore: boolean;
  allowDatabaseRestore: boolean;
  disabled?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onCodeRestored?: () => void | Promise<void>;
  onDatabaseRestored?: () => void | Promise<void>;
  onClose?: () => void;
}

type Scope = 'code' | 'combined' | 'database';
type Phase =
  | 'idle'
  | 'restoring-code'
  | 'code-restored'
  | 'preparing'
  | 'prepared'
  | 'applying'
  | 'completed'
  | 'uncertain-code'
  | 'uncertain-database'
  | 'recovering';
type Receipt = {
  backup: SqliteHistoryBackup;
  safetyBackupId: string | null;
  operationId: string;
};
type ApplyFailure = {
  backup: SqliteHistoryBackup;
  outcome: 'stale' | 'uncertain';
};

class PreflightError extends Error {}

function exactReadyBackups(
  backups: SqliteHistoryBackup[],
  snapshotId: string,
  initialBackupId?: string,
) {
  const ready = backups.filter(
    (backup) =>
      backup.snapshot_id === snapshotId && backup.status === 'ready' && backup.can_restore,
  );
  const latest = new Map<string, SqliteHistoryBackup>();
  for (const backup of ready) {
    const current = latest.get(backup.database_name);
    if (
      !current ||
      Date.parse(backup.created_at) > Date.parse(current.created_at) ||
      (backup.created_at === current.created_at && backup.id > current.id)
    ) {
      latest.set(backup.database_name, backup);
    }
  }
  const initial = ready.find((backup) => backup.id === initialBackupId);
  if (initial) latest.set(initial.database_name, initial);
  return [...latest.values()].sort((left, right) =>
    left.database_name.localeCompare(right.database_name),
  );
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function isForbidden(error: unknown) {
  return error instanceof ApiError && error.status === 403;
}

export function SnapshotRestorePanel({
  workspaceId,
  snapshotId,
  initialBackupId,
  defaultScope,
  allowCodeRestore,
  allowDatabaseRestore,
  disabled = false,
  onBusyChange,
  onCodeRestored,
  onDatabaseRestored,
  onClose,
}: SnapshotRestorePanelProps) {
  const radioGroupId = useId();
  const [scope, setScope] = useState<Scope>(defaultScope);
  const [phase, setPhase] = useState<Phase>('idle');
  const [codeAvailable, setCodeAvailable] = useState(false);
  const [codeSucceeded, setCodeSucceeded] = useState(false);
  const [backups, setBackups] = useState<SqliteHistoryBackup[]>([]);
  const [canManage, setCanManage] = useState(false);
  const [databaseSafetyKnown, setDatabaseSafetyKnown] = useState(!allowDatabaseRestore);
  const [maintenance, setMaintenance] = useState<SqliteHistoryMaintenanceStateResponse | null>(
    null,
  );
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set());
  const [previews, setPreviews] = useState<Record<string, SqliteHistoryPreview>>({});
  const [previewErrors, setPreviewErrors] = useState<Record<string, string>>({});
  const [receipts, setReceipts] = useState<Receipt[]>([]);
  const [applyFailure, setApplyFailure] = useState<ApplyFailure | null>(null);
  const [mode, setMode] = useState<SqliteHistoryRestoreMode>('merge');
  const [policy, setPolicy] = useState<SqliteHistoryConflictPolicy>('keep_current');
  const [tablePolicies, setTablePolicies] = useState<Record<string, SqliteHistoryConflictPolicy>>(
    {},
  );
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [downloadError, setDownloadError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const mountedRef = useRef(true);
  const contextVersionRef = useRef(0);
  const permissionVersionRef = useRef(0);
  const actionInFlightRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const canManageRef = useRef(false);
  const selectedIdsRef = useRef(selectedIds);
  const codeRestoredRef = useRef(false);
  const databaseSafetyKnownRef = useRef(!allowDatabaseRestore);
  const approvedAfterCodeRef = useRef<Set<string> | null>(null);
  const onBusyChangeRef = useRef(onBusyChange);

  onBusyChangeRef.current = onBusyChange;
  selectedIdsRef.current = selectedIds;
  canManageRef.current = canManage;
  busyRef.current = busy;

  const databaseAllowed = allowDatabaseRestore && canManage;
  const maintenanceBlocked = maintenance !== null;
  const terminal = phase === 'uncertain-code' || phase === 'uncertain-database';
  const codeRestored = codeSucceeded;
  codeRestoredRef.current = codeSucceeded;
  const selected = backups.filter((backup) => selectedIds.has(backup.id));
  const selectionPinned = receipts.length > 0 || phase === 'completed' || terminal;
  const restoreLocked =
    disabled ||
    busy ||
    maintenanceBlocked ||
    terminal ||
    (allowDatabaseRestore && !databaseSafetyKnown);
  const databaseScopeVisible = scope === 'database' || scope === 'combined';

  const isCurrent = (version: number) =>
    mountedRef.current && contextVersionRef.current === version;

  const clearPrivilegedState = () => {
    permissionVersionRef.current += 1;
    canManageRef.current = false;
    setCanManage(false);
    databaseSafetyKnownRef.current = false;
    setDatabaseSafetyKnown(false);
    setBackups([]);
    setSelectedIds(new Set());
    setPreviews({});
    setPreviewErrors({});
    setMaintenance(null);
  };

  const invalidatePreviews = () => {
    setPreviews({});
    setPreviewErrors({});
    setApplyFailure(null);
    if (codeRestoredRef.current) setPhase('code-restored');
    else setPhase('idle');
  };

  useEffect(() => {
    onBusyChangeRef.current?.(busy);
  }, [busy]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      contextVersionRef.current += 1;
      actionInFlightRef.current = null;
      if (busyRef.current) onBusyChangeRef.current?.(false);
    };
  }, []);

  useEffect(() => {
    if (!allowDatabaseRestore) return;
    return subscribeHistoryEvents(workspaceId, { onAccessRevoked: clearPrivilegedState });
  }, [allowDatabaseRestore, workspaceId]);

  useEffect(() => {
    const version = ++contextVersionRef.current;
    actionInFlightRef.current = null;
    busyRef.current = false;
    setBusy(false);
    setScope(defaultScope);
    setPhase('idle');
    setCodeAvailable(false);
    setCodeSucceeded(false);
    codeRestoredRef.current = false;
    setBackups([]);
    setCanManage(false);
    databaseSafetyKnownRef.current = !allowDatabaseRestore;
    setDatabaseSafetyKnown(!allowDatabaseRestore);
    canManageRef.current = false;
    setMaintenance(null);
    setSelectedIds(new Set());
    setPreviews({});
    setPreviewErrors({});
    setReceipts([]);
    setApplyFailure(null);
    setMode('merge');
    setPolicy('keep_current');
    setTablePolicies({});
    setError(null);
    setRefreshError(null);
    setDownloadError(null);
    setNotice(null);
    approvedAfterCodeRef.current = null;
    setLoading(true);

    void (async () => {
      try {
        const timeline = await api.getUserSpaceSnapshotTimeline(workspaceId);
        if (!isCurrent(version)) return;
        const hasCode =
          allowCodeRestore && timeline.snapshots.some((snapshot) => snapshot.id === snapshotId);
        let candidates: SqliteHistoryBackup[] = [];
        let manageable = false;
        let currentMaintenance: SqliteHistoryMaintenanceStateResponse | null = null;
        if (allowDatabaseRestore) {
          const permissionVersion = permissionVersionRef.current;
          const history = await api.listUserSpaceSqliteHistory(workspaceId, { snapshotId });
          if (!isCurrent(version) || permissionVersion !== permissionVersionRef.current) return;
          manageable = history.can_manage;
          currentMaintenance = history.interrupted_maintenance ?? null;
          if (manageable) {
            candidates = exactReadyBackups(history.backups, snapshotId, initialBackupId);
          }
        }
        if (!isCurrent(version)) return;
        setCodeAvailable(hasCode);
        databaseSafetyKnownRef.current = true;
        setDatabaseSafetyKnown(true);
        setCanManage(manageable);
        canManageRef.current = manageable;
        setMaintenance(currentMaintenance);
        setBackups(candidates);
        const initial = initialBackupId
          ? candidates.find((candidate) => candidate.id === initialBackupId)
          : null;
        setSelectedIds(
          new Set(initial ? [initial.id] : candidates.map((candidate) => candidate.id)),
        );
        if (defaultScope === 'database' && !candidates.length && hasCode) {
          setScope('code');
          setNotice(
            'No associated database restore point is available; code-only restore remains available.',
          );
        } else if (defaultScope === 'code' && !hasCode && candidates.length) {
          setScope('database');
          setNotice(
            'This code snapshot is no longer available; database-only restore remains available.',
          );
        } else if (!hasCode && !candidates.length) {
          setNotice('This restore point is no longer available.');
        }
      } catch (caught) {
        if (!isCurrent(version)) return;
        if (isForbidden(caught)) clearPrivilegedState();
        setError(errorMessage(caught, 'Unable to load restore options.'));
      } finally {
        if (isCurrent(version)) setLoading(false);
      }
    })();
  }, [
    workspaceId,
    snapshotId,
    initialBackupId,
    defaultScope,
    allowCodeRestore,
    allowDatabaseRestore,
  ]);

  const freshCodePreflight = async (version: number) => {
    const timeline = await api.getUserSpaceSnapshotTimeline(workspaceId);
    if (!isCurrent(version)) return false;
    const available =
      allowCodeRestore && timeline.snapshots.some((snapshot) => snapshot.id === snapshotId);
    setCodeAvailable(available);
    if (!available) {
      setError('This code snapshot is no longer available.');
      throw new PreflightError();
    }
    return true;
  };

  const freshDatabaseSafetyPreflight = async (version: number) => {
    if (!allowDatabaseRestore) return true;
    const permissionVersion = permissionVersionRef.current;
    try {
      const history = await api.listUserSpaceSqliteHistory(workspaceId, { snapshotId });
      if (!isCurrent(version) || permissionVersion !== permissionVersionRef.current) return false;
      databaseSafetyKnownRef.current = true;
      setDatabaseSafetyKnown(true);
      setMaintenance(history.interrupted_maintenance ?? null);
      if (history.interrupted_maintenance) {
        setError('Database maintenance must be recovered before restoring code.');
        throw new PreflightError();
      }
      return true;
    } catch (caught) {
      if (!isCurrent(version)) return false;
      if (caught instanceof PreflightError) throw caught;
      if (isForbidden(caught)) {
        clearPrivilegedState();
        setError(
          'Restore safety could not be verified because database history access was revoked.',
        );
        throw new PreflightError();
      }
      databaseSafetyKnownRef.current = false;
      setDatabaseSafetyKnown(false);
      setError('Restore safety could not be verified. Reload before restoring code.');
      throw new PreflightError(errorMessage(caught, 'Unable to verify restore safety.'));
    }
  };

  const freshDatabasePreflight = async (version: number) => {
    if (!allowDatabaseRestore || !canManageRef.current) {
      throw new PreflightError('Database restore permission is unavailable.');
    }
    const permissionVersion = permissionVersionRef.current;
    try {
      const history = await api.listUserSpaceSqliteHistory(workspaceId, { snapshotId });
      if (!isCurrent(version) || permissionVersion !== permissionVersionRef.current) return null;
      if (!history.can_manage) {
        clearPrivilegedState();
        setError('Database restore permission is no longer available.');
        throw new PreflightError();
      }
      databaseSafetyKnownRef.current = true;
      setDatabaseSafetyKnown(true);
      const candidates = exactReadyBackups(history.backups, snapshotId, initialBackupId);
      const approvedIds = new Set(candidates.map((candidate) => candidate.id));
      const narrowed = new Set(
        [...selectedIdsRef.current].filter((backupId) => approvedIds.has(backupId)),
      );
      setCanManage(true);
      canManageRef.current = true;
      setMaintenance(history.interrupted_maintenance ?? null);
      setBackups(candidates);
      setSelectedIds(narrowed);
      selectedIdsRef.current = narrowed;
      if (history.interrupted_maintenance) {
        setError('Database maintenance must be recovered before restoring.');
        throw new PreflightError();
      }
      if (!narrowed.size) {
        setError('Select at least one current associated database restore point.');
        throw new PreflightError();
      }
      return candidates.filter((candidate) => narrowed.has(candidate.id));
    } catch (caught) {
      if (!isCurrent(version)) return null;
      if (isForbidden(caught)) {
        clearPrivilegedState();
        setError('Database restore permission is no longer available.');
        throw new PreflightError();
      }
      throw caught;
    }
  };

  const prepareSelected = async (version: number, candidates: SqliteHistoryBackup[]) => {
    const next: Record<string, SqliteHistoryPreview> = {};
    const failures: Record<string, string> = {};
    for (const backup of candidates) {
      const perTable = Object.fromEntries(
        Object.entries(tablePolicies)
          .filter(([key]) => key.startsWith(`${backup.id}:`))
          .map(([key, value]) => [key.slice(backup.id.length + 1), value]),
      );
      try {
        const result = await api.previewUserSpaceSqliteHistory(workspaceId, backup.id, {
          mode,
          conflict_policy: policy,
          table_policies: perTable,
        });
        if (!isCurrent(version) || !canManageRef.current) return false;
        next[backup.id] = result;
      } catch (caught) {
        if (!isCurrent(version)) return false;
        if (isForbidden(caught)) {
          clearPrivilegedState();
          setError('Database restore permission is no longer available.');
          return false;
        }
        failures[backup.id] = errorMessage(caught, 'Preview failed.');
      }
    }
    if (!isCurrent(version)) return false;
    setPreviews(next);
    setPreviewErrors(failures);
    setPhase(
      Object.keys(failures).length
        ? codeRestoredRef.current
          ? 'code-restored'
          : 'idle'
        : 'prepared',
    );
    return Object.keys(failures).length === 0;
  };

  const beginAction = (version: number) => {
    if (actionInFlightRef.current !== null) return false;
    actionInFlightRef.current = version;
    setBusy(true);
    setError(null);
    setRefreshError(null);
    setDownloadError(null);
    return true;
  };

  const finishAction = (version: number) => {
    if (!isCurrent(version) || actionInFlightRef.current !== version) return;
    actionInFlightRef.current = null;
    setBusy(false);
  };

  const resetPreflightPhase = (version: number) => {
    if (!isCurrent(version)) return;
    setPhase(codeRestoredRef.current ? 'code-restored' : 'idle');
  };

  const restoreCode = async (combined: boolean) => {
    const version = contextVersionRef.current;
    if (!codeAvailable || restoreLocked || !beginAction(version)) return;
    setPhase('restoring-code');
    try {
      await freshCodePreflight(version);
      if (!isCurrent(version)) return;
      if (combined) {
        const approved = await freshDatabasePreflight(version);
        if (!isCurrent(version) || !approved || !canManageRef.current) {
          resetPreflightPhase(version);
          return;
        }
        approvedAfterCodeRef.current = new Set(approved.map((backup) => backup.id));
      } else {
        const safetyCurrent = await freshDatabaseSafetyPreflight(version);
        if (!isCurrent(version) || !safetyCurrent || !databaseSafetyKnownRef.current) {
          resetPreflightPhase(version);
          return;
        }
      }
      try {
        await api.restoreUserSpaceSnapshot(workspaceId, snapshotId);
      } catch (caught) {
        if (!isCurrent(version)) return;
        setPhase('uncertain-code');
        setError(
          `Code restore outcome is uncertain. Do not retry blindly; reload the timeline and verify the current snapshot. ${errorMessage(caught, '')}`,
        );
        return;
      }
      if (!isCurrent(version)) return;
      codeRestoredRef.current = true;
      setCodeSucceeded(true);
      setPhase('code-restored');
      setNotice(
        combined
          ? 'Code restored; databases not yet restored.'
          : 'Code restored. Existing database files were not changed.',
      );
      try {
        await onCodeRestored?.();
      } catch (caught) {
        if (!isCurrent(version)) return;
        setRefreshError(
          `Code restore succeeded, but refresh failed: ${errorMessage(caught, 'Unable to refresh.')}`,
        );
      }
      if (!isCurrent(version) || !combined) return;
      if (!canManageRef.current) {
        setError('Database restore permission was revoked after code restore.');
        return;
      }
      const candidates = await freshDatabasePreflight(version);
      if (!isCurrent(version) || !candidates) return;
      setPhase('preparing');
      await prepareSelected(version, candidates);
    } catch (caught) {
      if (!isCurrent(version)) return;
      if (caught instanceof PreflightError) {
        resetPreflightPhase(version);
        return;
      }
      setPhase(codeRestoredRef.current ? 'code-restored' : 'idle');
      setError(errorMessage(caught, 'Unable to prepare this restore.'));
    } finally {
      finishAction(version);
    }
  };

  const prepare = async () => {
    const version = contextVersionRef.current;
    if (
      !databaseAllowed ||
      !selected.length ||
      restoreLocked ||
      selectionPinned ||
      !beginAction(version)
    )
      return;
    setPhase('preparing');
    try {
      const candidates = await freshDatabasePreflight(version);
      if (!isCurrent(version) || !candidates || !canManageRef.current) {
        resetPreflightPhase(version);
        return;
      }
      await prepareSelected(version, candidates);
    } catch (caught) {
      if (!isCurrent(version)) return;
      if (caught instanceof PreflightError) {
        resetPreflightPhase(version);
        return;
      }
      setPhase(codeRestoredRef.current ? 'code-restored' : 'idle');
      setError(errorMessage(caught, 'Unable to prepare database previews.'));
    } finally {
      finishAction(version);
    }
  };

  const apply = async () => {
    const version = contextVersionRef.current;
    const applicable = selected.filter(
      (backup) => previews[backup.id]?.can_apply && previews[backup.id]?.preview_id,
    );
    if (
      !databaseAllowed ||
      applicable.length !== selected.length ||
      restoreLocked ||
      selectionPinned ||
      !beginAction(version)
    )
      return;
    setPhase('applying');
    setApplyFailure(null);
    try {
      const currentCandidates = await freshDatabasePreflight(version);
      if (!isCurrent(version) || !currentCandidates || !canManageRef.current) {
        resetPreflightPhase(version);
        return;
      }
      const currentIds = new Set(currentCandidates.map((candidate) => candidate.id));
      if (applicable.some((candidate) => !currentIds.has(candidate.id))) {
        setPreviews({});
        setPhase(codeRestoredRef.current ? 'code-restored' : 'idle');
        setError('A database restore point changed. Prepare all selected databases again.');
        return;
      }
      let completedThisRun = 0;
      for (const backup of applicable) {
        if (!canManageRef.current) {
          setPhase('completed');
          setError(
            'Database restore permission was revoked. Remaining databases were not attempted.',
          );
          return;
        }
        const currentPreview = previews[backup.id];
        try {
          const result = await api.restoreUserSpaceSqliteHistory(
            workspaceId,
            currentPreview.preview_id!,
          );
          if (!isCurrent(version)) return;
          setReceipts((prior) => [
            ...prior,
            {
              backup,
              safetyBackupId: result.safety_backup_id,
              operationId: result.operation_id,
            },
          ]);
          completedThisRun += 1;
          setNotice('Database restore completed. The workspace runtime was stopped for safety.');
          if (!canManageRef.current) {
            setPhase('completed');
            setError(
              'Database restore permission was revoked. Completed databases will not be replayed; remaining databases were not attempted.',
            );
            return;
          }
        } catch (caught) {
          if (!isCurrent(version)) return;
          if (isForbidden(caught)) clearPrivilegedState();
          if (caught instanceof ApiError && caught.status === 409) {
            setApplyFailure({ backup, outcome: 'stale' });
            setPreviews({});
            if (receipts.length + completedThisRun === 0) {
              setPhase(codeRestoredRef.current ? 'code-restored' : 'idle');
              setError('A database preview is stale. Prepare all selected databases again.');
            } else {
              setPhase('completed');
              setError(
                'A remaining database preview became stale after another database completed. Completed databases will not be replayed.',
              );
            }
          } else {
            setApplyFailure({ backup, outcome: 'uncertain' });
            setPhase('uncertain-database');
            setError(
              `${backup.database_name} restore outcome is uncertain. Do not retry; reload history and use maintenance recovery if offered.`,
            );
          }
          return;
        }
      }
      if (!isCurrent(version)) return;
      setPhase('completed');
      try {
        await onDatabaseRestored?.();
      } catch (caught) {
        if (!isCurrent(version)) return;
        setRefreshError(
          `Database restore succeeded, but refresh failed: ${errorMessage(caught, 'Unable to refresh.')}`,
        );
      }
    } catch (caught) {
      if (!isCurrent(version)) return;
      if (caught instanceof PreflightError) {
        resetPreflightPhase(version);
        return;
      }
      setPhase('prepared');
      setError(errorMessage(caught, 'Unable to verify database restore permission.'));
    } finally {
      finishAction(version);
    }
  };

  const recover = async (action: 'complete' | 'abort') => {
    const permitted = action === 'complete' ? maintenance?.can_complete : maintenance?.can_abort;
    if (
      !maintenance?.operation_id ||
      !permitted ||
      disabled ||
      busy ||
      actionInFlightRef.current !== null
    )
      return;
    const version = contextVersionRef.current;
    if (!beginAction(version)) return;
    const operationId = maintenance.operation_id;
    setPhase('recovering');
    try {
      await api.recoverUserSpaceSqliteHistoryMaintenance(workspaceId, operationId, action);
      if (!isCurrent(version)) return;
      setMaintenance(null);
      setPhase('idle');
      setNotice('Maintenance recovery completed. The workspace runtime remains stopped.');
      try {
        const refreshed = await api.listUserSpaceSqliteHistory(workspaceId, { snapshotId });
        if (!isCurrent(version)) return;
        if (!refreshed.can_manage) {
          clearPrivilegedState();
          return;
        }
        setMaintenance(refreshed.interrupted_maintenance ?? null);
        const candidates = exactReadyBackups(refreshed.backups, snapshotId, initialBackupId);
        setBackups(candidates);
        setSelectedIds(
          new Set(
            [...selectedIdsRef.current].filter((id) => candidates.some((item) => item.id === id)),
          ),
        );
      } catch (caught) {
        if (!isCurrent(version)) return;
        if (isForbidden(caught)) clearPrivilegedState();
        setRefreshError(
          `Maintenance recovery succeeded, but history refresh failed: ${errorMessage(caught, 'Unable to refresh.')}`,
        );
      }
    } catch (caught) {
      if (!isCurrent(version)) return;
      setPhase('idle');
      setError(errorMessage(caught, 'Maintenance recovery failed.'));
    } finally {
      finishAction(version);
    }
  };

  const downloadSafetyBackup = async (receipt: Receipt) => {
    const version = contextVersionRef.current;
    setDownloadError(null);
    try {
      await api.downloadUserSpaceSqliteHistory(workspaceId, receipt.safetyBackupId!);
    } catch (caught) {
      if (!isCurrent(version)) return;
      setDownloadError(errorMessage(caught, 'Safety backup download failed.'));
    }
  };

  const changeSelected = (backupId: string, checked: boolean) => {
    if (selectionPinned || busy) return;
    if (
      checked &&
      codeRestoredRef.current &&
      approvedAfterCodeRef.current &&
      !approvedAfterCodeRef.current.has(backupId)
    )
      return;
    setSelectedIds((current) => {
      const next = new Set(current);
      if (checked) next.add(backupId);
      else next.delete(backupId);
      selectedIdsRef.current = next;
      return next;
    });
    invalidatePreviews();
  };

  const completedIds = new Set(receipts.map((receipt) => receipt.backup.id));
  const notAttempted =
    receipts.length || applyFailure
      ? selected.filter(
          (backup) => !completedIds.has(backup.id) && backup.id !== applyFailure?.backup.id,
        )
      : [];
  const allPreviewsApply =
    selected.length > 0 &&
    selected.every((backup) => previews[backup.id]?.can_apply && previews[backup.id]?.preview_id);
  const canChangeScope = !loading && !busy && !codeRestored && !selectionPinned && !terminal;

  return (
    <section
      className="snapshot-restore-panel"
      data-snapshot-restore-panel={snapshotId}
      aria-busy={loading || busy}
    >
      {loading && (
        <p role="status" className="database-history-maintenance-busy">
          <Loader2 size={14} className="spinning" /> Loading restore options…
        </p>
      )}
      {!loading && busy && (
        <p
          role="status"
          className="database-history-maintenance-busy"
          data-snapshot-restore-phase={phase}
        >
          <Loader2 size={14} className="spinning" />
          {phase === 'restoring-code' && 'Restoring code snapshot…'}
          {phase === 'preparing' && 'Preparing database previews…'}
          {phase === 'applying' && 'Applying database restore…'}
          {phase === 'recovering' && 'Recovering database maintenance…'}
        </p>
      )}
      <fieldset className="snapshot-restore-scope-options" disabled={!canChangeScope}>
        <legend>Restore scope</legend>
        {codeAvailable && (
          <label>
            <input
              type="radio"
              name={`restore-scope-${radioGroupId}`}
              checked={scope === 'code'}
              onChange={() => {
                setScope('code');
                invalidatePreviews();
              }}
            />{' '}
            Code only
          </label>
        )}
        {codeAvailable && databaseAllowed && backups.length > 0 && (
          <label>
            <input
              type="radio"
              name={`restore-scope-${radioGroupId}`}
              checked={scope === 'combined'}
              onChange={() => {
                setScope('combined');
                invalidatePreviews();
              }}
            />{' '}
            Code and database
          </label>
        )}
        {databaseAllowed && backups.length > 0 && (
          <label>
            <input
              type="radio"
              name={`restore-scope-${radioGroupId}`}
              checked={scope === 'database'}
              onChange={() => {
                setScope('database');
                invalidatePreviews();
              }}
            />{' '}
            Database only
          </label>
        )}
      </fieldset>

      {maintenance && (
        <div role="alert" className="snapshot-restore-maintenance">
          <p>{maintenance.detail ?? 'Database maintenance must be recovered before restoring.'}</p>
          <div className="snapshot-restore-controls">
            {maintenance.operation_id && maintenance.can_complete && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={disabled || busy}
                onClick={() => void recover('complete')}
              >
                Complete maintenance
              </button>
            )}
            {maintenance.operation_id && maintenance.can_abort && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={disabled || busy}
                onClick={() => void recover('abort')}
              >
                Abort maintenance
              </button>
            )}
          </div>
        </div>
      )}

      {databaseAllowed && databaseScopeVisible && backups.length > 0 && (
        <fieldset
          className="snapshot-restore-databases"
          disabled={disabled || busy || selectionPinned}
        >
          <legend>Choose databases</legend>
          {backups.map((backup) => (
            <label className="snapshot-restore-database-option" key={backup.id}>
              <input
                type="checkbox"
                checked={selectedIds.has(backup.id)}
                onChange={(event) => changeSelected(backup.id, event.target.checked)}
              />{' '}
              {backup.database_name} — {new Date(backup.created_at).toLocaleString()} (
              {backup.trigger})
            </label>
          ))}
        </fieldset>
      )}

      {databaseAllowed && databaseScopeVisible && selected.length > 0 && (
        <div className="snapshot-restore-controls">
          <label>
            Restore mode{' '}
            <select
              value={mode}
              disabled={restoreLocked || selectionPinned}
              onChange={(event) => {
                setMode(event.target.value as SqliteHistoryRestoreMode);
                invalidatePreviews();
              }}
            >
              <option value="merge">Merge</option>
              <option value="overwrite">Overwrite</option>
            </select>
          </label>
          {mode === 'merge' && (
            <label>
              Conflict policy{' '}
              <select
                value={policy}
                disabled={restoreLocked || selectionPinned}
                onChange={(event) => {
                  setPolicy(event.target.value as SqliteHistoryConflictPolicy);
                  invalidatePreviews();
                }}
              >
                <option value="keep_current">Keep current</option>
                <option value="use_backup">Use backup</option>
              </select>
            </label>
          )}
        </div>
      )}

      {codeAvailable &&
        (scope === 'code' || scope === 'combined') &&
        !codeRestored &&
        !terminal && (
          <p className="database-history-warning" role="alert">
            Restoring code overwrites current workspace files. Database files are preserved until a
            database restore is explicitly confirmed.
          </p>
        )}
      {databaseAllowed && databaseScopeVisible && selected.length > 0 && !selectionPinned && (
        <p className="database-history-warning" role="alert">
          {mode === 'merge'
            ? 'Merge can restore rows that were deliberately deleted from the current database.'
            : 'Overwrite removes current-only data and replaces each selected database with its restore point.'}
        </p>
      )}

      {notice && <p role="status">{notice}</p>}
      {error && <p role="alert">{error}</p>}
      {refreshError && <p role="alert">{refreshError}</p>}
      {downloadError && <p role="alert">{downloadError}</p>}

      <div className="snapshot-restore-actions">
        {scope === 'code' && !codeRestored && !terminal && (
          <button
            type="button"
            className="btn btn-primary"
            disabled={!codeAvailable || restoreLocked}
            onClick={() => void restoreCode(false)}
          >
            Restore code
          </button>
        )}
        {scope === 'combined' && !codeRestored && !terminal && (
          <button
            type="button"
            className="btn btn-primary"
            disabled={!codeAvailable || !databaseAllowed || !selected.length || restoreLocked}
            onClick={() => void restoreCode(true)}
          >
            Restore code and prepare database previews
          </button>
        )}
        {databaseScopeVisible &&
          !selectionPinned &&
          !terminal &&
          (scope === 'database' || codeRestored) && (
            <button
              type="button"
              className="btn btn-primary"
              disabled={!databaseAllowed || !selected.length || restoreLocked}
              onClick={() => void prepare()}
            >
              Prepare database previews
            </button>
          )}
        {Object.keys(previews).length > 0 && !selectionPinned && !terminal && (
          <button
            type="button"
            className="btn btn-primary"
            disabled={restoreLocked || !allPreviewsApply}
            onClick={() => void apply()}
          >
            Confirm database restore
          </button>
        )}
        {onClose && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            disabled={busy}
            onClick={onClose}
          >
            Close
          </button>
        )}
      </div>

      {backups.map((backup) => {
        const currentPreview = previews[backup.id];
        const previewError = previewErrors[backup.id];
        if (!currentPreview && !previewError) return null;
        return (
          <section className="snapshot-restore-preview" key={backup.id}>
            <strong>{backup.database_name}</strong>
            {previewError && (
              <p role="alert">
                {backup.database_name}: {previewError}
              </p>
            )}
            {currentPreview && (
              <>
                <p>
                  Migrations:{' '}
                  {currentPreview.migrations_applied.length
                    ? currentPreview.migrations_applied.join(', ')
                    : 'none'}
                </p>
                {currentPreview.warnings.map((warning) => (
                  <p className="database-history-warning" key={warning}>
                    {warning}
                  </p>
                ))}
                {currentPreview.blockers.map((blocker) => (
                  <p role="alert" key={blocker}>
                    {blocker}
                  </p>
                ))}
                {currentPreview.tables.map((table) => (
                  <details key={table.name}>
                    <summary>
                      {table.name}: {table.inserted} inserted, {table.updated} updated,{' '}
                      {table.deleted} deleted, {table.conflicts} conflicts
                    </summary>
                    {mode === 'merge' && (
                      <label>
                        Table conflict policy ({table.name}){' '}
                        <select
                          value={tablePolicies[`${backup.id}:${table.name}`] ?? policy}
                          disabled={restoreLocked || selectionPinned}
                          onChange={(event) => {
                            setTablePolicies((current) => ({
                              ...current,
                              [`${backup.id}:${table.name}`]: event.target
                                .value as SqliteHistoryConflictPolicy,
                            }));
                            invalidatePreviews();
                          }}
                        >
                          <option value="keep_current">Keep current</option>
                          <option value="use_backup">Use backup</option>
                        </select>
                      </label>
                    )}
                  </details>
                ))}
              </>
            )}
          </section>
        );
      })}

      {(receipts.length > 0 || applyFailure) && (
        <section
          className="snapshot-restore-receipts"
          role="status"
          aria-label="Database restore results"
        >
          {receipts.map((receipt) => (
            <p key={receipt.operationId}>
              {receipt.backup.database_name}: restored; runtime stopped.
              {receipt.safetyBackupId && (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => void downloadSafetyBackup(receipt)}
                  aria-label={`Download ${receipt.backup.database_name} safety backup`}
                >
                  Download safety backup
                </button>
              )}
            </p>
          ))}
          {applyFailure && (
            <p>
              {applyFailure.backup.database_name}:{' '}
              {applyFailure.outcome === 'uncertain'
                ? 'outcome uncertain; do not retry.'
                : 'not restored because its preview became stale.'}
            </p>
          )}
          {notAttempted.map((backup) => (
            <p key={backup.id}>{backup.database_name}: not attempted.</p>
          ))}
        </section>
      )}
    </section>
  );
}
