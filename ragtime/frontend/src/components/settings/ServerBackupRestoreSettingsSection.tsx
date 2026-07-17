import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import { AlertTriangle, ArrowDownToLine } from 'lucide-react';
import { api } from '@/api';
import type {
  ServerBackupExportListItem,
  ServerBackupJob,
  ServerDeploymentEnvironmentRecovery,
  ServerBackupManifest,
  ServerBackupScope,
  ServerRestoreJob,
} from '@/types';
import { formatBytes } from '@/utils';
import {
  getExportPasswordPolicy,
  passwordMeetsRequirements,
  type ExportPasswordPolicy,
} from '@/utils/exportPasswordPolicy';
import { DeleteConfirmButton } from '../DeleteConfirmButton';
import { InlineCopyButton } from '../shared/InlineCopyButton';
import { PasswordRequirementsChecklist } from '../shared/PasswordRequirementsChecklist';
import { SERVER_BACKUP_RESTORE_HIGHLIGHT } from '../shared/securityWarnings';
import type { SettingsAccordionSectionId } from './settingsAccordionState';
import { SettingsAccordionSection } from './SettingsAccordionSection';

const POLL_INTERVAL_MS = 2000;
const BACKUP_CARD_CLEAR_DELAY_MS = 3000;
const SENSITIVE_DEPLOYMENT_ENVIRONMENT_VARIABLES = ['DATABASE_URL', 'POSTGRES_PASSWORD'] as const;

interface ServerBackupRestoreSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  settings?: Partial<ExportPasswordPolicy> | null;
  onEncryptedArtifactDelivered?: () => void;
  onServerBackupJobObserved?: (job: ServerBackupJob) => void;
  onServerRestoreJobObserved?: (job: ServerRestoreJob) => void;
  onServerOperationError?: (message: string) => void;
}

type BackupRestoreTab = 'backup' | 'restore';

const PHASE_LABELS: Record<string, string> = {
  queued: 'Queued',
  start: 'Starting',
  database_dump_start: 'Dumping database',
  database_dump_complete: 'Database dumped',
  files_collect_start: 'Copying data files',
  files_collect_complete: 'Data files copied',
  manifest_write: 'Writing manifest',
  archive_build_start: 'Building archive',
  archive_build_complete: 'Archive built',
  output_write_start: 'Writing backup artifact',
  output_write_complete: 'Artifact written',
  complete: 'Ready for download',
  delivered: 'Delivered',
  cancelled: 'Cancelled',
  failed: 'Failed',
  interrupted: 'Interrupted',
  archive_prepare: 'Preparing archive',
  archive_extract_start: 'Extracting archive',
  archive_extract_complete: 'Archive extracted',
  validation_complete: 'Validated — confirmation required',
  data_snapshot_start: 'Creating safety snapshot',
  data_snapshot_complete: 'Safety snapshot created',
  database_safety_dump_start: 'Creating database safety dump',
  database_safety_dump_complete: 'Database safety dump created',
  database_restore_start: 'Restoring database',
  database_restore_complete: 'Database restored',
  database_migrations_start: 'Applying migrations',
  database_migrations_complete: 'Migrations applied',
  files_restore_start: 'Restoring files',
  files_restore_complete: 'Files restored',
  acquiring_maintenance: 'Acquiring maintenance window',
};

function humanizeProgressLabel(value: string): string {
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function getProgressPhaseLabel(
  phase?: string | null,
  status?: string | null,
  kind: 'backup' | 'restore' = 'backup',
): string {
  if (phase === 'complete' && kind === 'restore') {
    return 'Restore completed';
  }
  if (phase && PHASE_LABELS[phase]) {
    return PHASE_LABELS[phase];
  }
  if (phase) {
    return humanizeProgressLabel(phase);
  }
  if (status) {
    return humanizeProgressLabel(status);
  }
  return 'In progress';
}

function getProgressCountsLabel(
  details?: ServerBackupJob['details'] | ServerRestoreJob['details'] | null,
): string | null {
  if (
    typeof details?.processed_items === 'number' &&
    typeof details?.total_items === 'number' &&
    details.total_items >= 0
  ) {
    return `${details.processed_items} / ${details.total_items} files`;
  }
  return null;
}

interface ProgressCardProps {
  job: Pick<ServerBackupJob, 'phase' | 'status' | 'progress' | 'message' | 'details'>;
  ariaLabel: string;
  kind: 'backup' | 'restore';
  className?: string;
  suppressedMessage?: string | null;
}

function ProgressStatusCard({
  job,
  ariaLabel,
  kind,
  className,
  suppressedMessage = null,
}: ProgressCardProps): JSX.Element {
  const progress = getProgressValue(job.progress);
  const countsLabel = getProgressCountsLabel(job.details);
  const progressMessage = job.message === suppressedMessage ? null : job.message;

  return (
    <div className={className ?? 'server-backup-status-card'}>
      <div className="server-backup-status-header">
        <strong>{getProgressPhaseLabel(job.phase, job.status, kind)}</strong>
        <span className="server-backup-progress-pct">{progress}%</span>
      </div>
      <div className="progress-container server-backup-progress-container">
        <div
          className="progress-bar"
          role="progressbar"
          aria-label={ariaLabel}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progress}
        >
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
        {progressMessage || countsLabel ? (
          <div className="progress-details server-backup-progress-details">
            <span className="server-backup-progress-message">{progressMessage}</span>
            {countsLabel ? (
              <span className="server-backup-progress-counts">{countsLabel}</span>
            ) : null}
          </div>
        ) : null}
        {job.details?.current_item ? (
          <div className="server-backup-current-item" title={job.details.current_item}>
            {job.details.current_item}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function isBackupJobTerminal(job: ServerBackupJob | null): boolean {
  if (!job) return true;
  return ['delivered', 'cancelled', 'failed', 'interrupted'].includes(job.status);
}

function isRestoreJobTerminal(job: ServerRestoreJob | null): boolean {
  if (!job) return true;
  return ['completed', 'cancelled', 'failed'].includes(job.status);
}

function isBackupReadyForDownload(job: ServerBackupJob | null): boolean {
  if (!job) return false;
  return job.status === 'completed';
}

function isBackupDelivered(job: ServerBackupJob | null): boolean {
  if (!job) return false;
  return job.status === 'delivered';
}

function requiresRestoreConfirmation(job: ServerRestoreJob | null): boolean {
  if (!job) return false;
  return job.status === 'ready_for_commit';
}

function getProgressValue(progress: number | null | undefined): number {
  if (typeof progress !== 'number' || Number.isNaN(progress)) {
    return 0;
  }
  return Math.max(0, Math.min(100, progress));
}

function getFailedJobError(
  job: Pick<ServerBackupJob, 'status' | 'error'> | null | undefined,
): string | null {
  if (job?.status !== 'failed') {
    return null;
  }
  return job.error ?? null;
}

function manifestValue(value: string | number | boolean | null | undefined): string {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (value == null || value === '') return '—';
  return String(value);
}

function formatExportDate(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return value;
  }
  return parsed.toLocaleString();
}

function renderManifestSummary(
  manifest: ServerBackupManifest | null | undefined,
): JSX.Element | null {
  if (!manifest) {
    return null;
  }

  return (
    <dl className="server-backup-manifest-grid">
      <div>
        <dt>Format</dt>
        <dd>{manifestValue(manifest.format)}</dd>
      </div>
      <div>
        <dt>Version</dt>
        <dd>{manifestValue(manifest.version)}</dd>
      </div>
      <div>
        <dt>Created</dt>
        <dd>{manifestValue(manifest.created_at)}</dd>
      </div>
      <div>
        <dt>Scope</dt>
        <dd>{manifestValue(manifest.scope)}</dd>
      </div>
      <div>
        <dt>Encrypted</dt>
        <dd>{manifestValue(manifest.encrypted)}</dd>
      </div>
      <div>
        <dt>Managed key included</dt>
        <dd>{manifestValue(manifest.includes_managed_key)}</dd>
      </div>
      <div>
        <dt>Legacy embedded key</dt>
        <dd>{manifestValue(manifest.legacy_embedded_key)}</dd>
      </div>
    </dl>
  );
}

function formatDotenvAssignment(name: string, value: string): string {
  return `${name}=${JSON.stringify(value)}`;
}

function hasExactNames(names: string[], expectedNames: string[]): boolean {
  if (names.length !== expectedNames.length) {
    return false;
  }
  const actualSorted = [...names].sort();
  const expectedSorted = [...expectedNames].sort();
  return actualSorted.every((name, index) => name === expectedSorted[index]);
}

export function ServerBackupRestoreSettingsSection(
  props: ServerBackupRestoreSettingsSectionProps,
): JSX.Element {
  const {
    open,
    onToggle,
    settings = null,
    onEncryptedArtifactDelivered,
    onServerBackupJobObserved,
    onServerRestoreJobObserved,
    onServerOperationError,
  } = props;
  const passwordPolicy = useMemo(() => getExportPasswordPolicy(settings), [settings]);
  const [activeTab, setActiveTab] = useState<BackupRestoreTab>('backup');
  const backupTabRef = useRef<HTMLButtonElement | null>(null);
  const restoreTabRef = useRef<HTMLButtonElement | null>(null);
  const [backupScope, setBackupScope] = useState<ServerBackupScope>('full');
  const [backupEncrypted, setBackupEncrypted] = useState(true);
  const [backupPassword, setBackupPassword] = useState('');
  const [backupPasswordConfirm, setBackupPasswordConfirm] = useState('');
  const [backupJob, setBackupJob] = useState<ServerBackupJob | null>(null);
  const [backupBusy, setBackupBusy] = useState(false);
  const [backupDownloadPending, setBackupDownloadPending] = useState(false);
  const [backupExports, setBackupExports] = useState<ServerBackupExportListItem[]>([]);
  const [downloadedBackupExportIds, setDownloadedBackupExportIds] = useState<Set<string>>(
    new Set(),
  );
  const [deletingBackupExportJobId, setDeletingBackupExportJobId] = useState<string | null>(null);
  const backupDeliveryNotifiedRef = useRef(false);
  const lastCompletedBackupHistoryRefreshRef = useRef<string | null>(null);
  const backupCleanupTimerRef = useRef<number | null>(null);

  const [restoreArchiveFile, setRestoreArchiveFile] = useState<File | null>(null);
  const [restorePassword, setRestorePassword] = useState('');
  const [restoreSkipMigrations, setRestoreSkipMigrations] = useState(false);
  const [restorePostgresDataOnly, setRestorePostgresDataOnly] = useState(false);
  const [restoreReplaceData, setRestoreReplaceData] = useState(false);
  const [restoreMirrorLocalAdmins, setRestoreMirrorLocalAdmins] = useState(false);
  const [restoreMirrorLocalAdminFrom, setRestoreMirrorLocalAdminFrom] = useState('auto');
  const [restoreLocalAdminUsername, setRestoreLocalAdminUsername] = useState('');
  const [restoreConfirmationText, setRestoreConfirmationText] = useState('');
  const [restoreLegacyKeyAcknowledged, setRestoreLegacyKeyAcknowledged] = useState(false);
  const [restoreJob, setRestoreJob] = useState<ServerRestoreJob | null>(null);
  const [restoreBusy, setRestoreBusy] = useState(false);
  const [recoveredEnvironmentVariables, setRecoveredEnvironmentVariables] = useState<Record<
    string,
    string
  > | null>(null);
  const [recoveryBusy, setRecoveryBusy] = useState(false);
  const pollingErrorRef = useRef<{ backup: string | null; restore: string | null }>({
    backup: null,
    restore: null,
  });

  const backupJobId = backupJob?.id ?? null;
  const backupJobStatus = backupJob?.status ?? null;
  const backupShouldPoll =
    backupJobId != null &&
    backupJobStatus != null &&
    !isBackupJobTerminal(backupJob) &&
    (backupJobStatus !== 'completed' || backupDownloadPending);

  const clearBackupCleanupTimer = useCallback(() => {
    if (backupCleanupTimerRef.current !== null) {
      window.clearTimeout(backupCleanupTimerRef.current);
      backupCleanupTimerRef.current = null;
    }
  }, []);

  const scheduleBackupCleanup = useCallback(
    (jobId: string) => {
      clearBackupCleanupTimer();
      backupCleanupTimerRef.current = window.setTimeout(() => {
        setBackupJob((current) => (current?.id === jobId ? null : current));
        backupCleanupTimerRef.current = null;
      }, BACKUP_CARD_CLEAR_DELAY_MS);
    },
    [clearBackupCleanupTimer],
  );

  const loadBackupExports = useCallback(async (): Promise<ServerBackupExportListItem[]> => {
    const response = await api.listServerBackupExports();
    pollingErrorRef.current.backup = null;
    setBackupExports(response.exports);
    return response.exports;
  }, []);

  const clearRecoveredEnvironmentState = useCallback(() => {
    setRecoveredEnvironmentVariables(null);
    setRecoveryBusy(false);
  }, []);

  const reportServerOperationError = useCallback(
    (
      message: string,
      options?: {
        dedupeKey?: 'backup' | 'restore';
      },
    ) => {
      const dedupeKey = options?.dedupeKey;
      if (dedupeKey) {
        if (pollingErrorRef.current[dedupeKey] === message) {
          return;
        }
        pollingErrorRef.current[dedupeKey] = message;
      }
      onServerOperationError?.(message);
    },
    [onServerOperationError],
  );

  const restoreJobId = restoreJob?.id ?? null;
  const restoreJobStatus = restoreJob?.status ?? null;
  const restoreShouldPoll =
    restoreJobId != null &&
    restoreJobStatus != null &&
    !isRestoreJobTerminal(restoreJob) &&
    !requiresRestoreConfirmation(restoreJob);
  const restoreManifestEnvironmentVariableNames =
    restoreJob?.manifest?.deployment_environment_variables ?? [];
  const hasRecoverableEnvironmentVariables = restoreManifestEnvironmentVariableNames.length > 0;
  const hasSensitiveRecoveredEnvironmentVariables = restoreManifestEnvironmentVariableNames.some(
    (name) =>
      SENSITIVE_DEPLOYMENT_ENVIRONMENT_VARIABLES.includes(
        name as (typeof SENSITIVE_DEPLOYMENT_ENVIRONMENT_VARIABLES)[number],
      ),
  );

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const activeJobs = await api.getActiveServerBackupJobs();
        if (cancelled) {
          return;
        }
        pollingErrorRef.current.backup = null;
        pollingErrorRef.current.restore = null;
        setBackupJob(activeJobs.backup_job);
        setRestoreJob(activeJobs.restore_job);
        if (activeJobs.restore_job && !activeJobs.backup_job) {
          setActiveTab('restore');
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        reportServerOperationError(
          error instanceof Error ? error.message : 'Failed to load active backup or restore job',
        );
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [reportServerOperationError]);

  useEffect(() => {
    return () => {
      clearBackupCleanupTimer();
      clearRecoveredEnvironmentState();
    };
  }, [clearBackupCleanupTimer, clearRecoveredEnvironmentState]);

  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        await loadBackupExports();
        if (cancelled) {
          return;
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        reportServerOperationError(
          error instanceof Error ? error.message : 'Failed to load previous backup exports',
        );
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [loadBackupExports, reportServerOperationError]);

  useEffect(() => {
    if (!backupShouldPoll || !backupJobId) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.getServerBackupJob(backupJobId);
        pollingErrorRef.current.backup = null;
        setBackupJob(nextJob);
      } catch (error) {
        reportServerOperationError(
          error instanceof Error ? error.message : 'Failed to refresh backup status',
          { dedupeKey: 'backup' },
        );
      }
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [backupJobId, backupShouldPoll, reportServerOperationError]);

  useEffect(() => {
    if (!restoreShouldPoll || !restoreJobId) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.getServerRestoreJob(restoreJobId);
        pollingErrorRef.current.restore = null;
        setRestoreJob(nextJob);
      } catch (error) {
        reportServerOperationError(
          error instanceof Error ? error.message : 'Failed to refresh restore status',
          { dedupeKey: 'restore' },
        );
      }
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [reportServerOperationError, restoreJobId, restoreShouldPoll]);

  useEffect(() => {
    clearRecoveredEnvironmentState();
  }, [clearRecoveredEnvironmentState, restoreJobId]);

  useEffect(() => {
    if (!hasRecoverableEnvironmentVariables) {
      clearRecoveredEnvironmentState();
    }
  }, [clearRecoveredEnvironmentState, hasRecoverableEnvironmentVariables]);

  useEffect(() => {
    if (backupDeliveryNotifiedRef.current || !backupJob?.encrypt || !isBackupDelivered(backupJob)) {
      return;
    }

    backupDeliveryNotifiedRef.current = true;
    setBackupDownloadPending(false);
    onEncryptedArtifactDelivered?.();
  }, [backupJob, onEncryptedArtifactDelivered]);

  useEffect(() => {
    if (backupJob && isBackupJobTerminal(backupJob) && backupJob.status !== 'delivered') {
      setBackupDownloadPending(false);
    }
  }, [backupJob]);

  useEffect(() => {
    if (!backupJob?.id || (backupJob.status !== 'completed' && backupJob.status !== 'delivered')) {
      return;
    }
    const refreshKey = `${backupJob.id}:${backupJob.status}`;
    if (lastCompletedBackupHistoryRefreshRef.current === refreshKey) {
      return;
    }
    lastCompletedBackupHistoryRefreshRef.current = refreshKey;
    void loadBackupExports().catch((error) => {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to refresh previous backup exports',
      );
    });
  }, [backupJob, loadBackupExports, reportServerOperationError]);

  useEffect(() => {
    if (!backupJob?.id || (backupJob.status !== 'completed' && backupJob.status !== 'delivered')) {
      return;
    }
    if (backupExports.some((item) => item.job_id === backupJob.id)) {
      scheduleBackupCleanup(backupJob.id);
      return;
    }
    clearBackupCleanupTimer();
  }, [backupExports, backupJob, clearBackupCleanupTimer, scheduleBackupCleanup]);

  const requiredConfirmationText = restoreJob?.required_confirmation || 'RESTORE';
  const restoreCanCommit =
    requiresRestoreConfirmation(restoreJob) &&
    restoreConfirmationText === requiredConfirmationText &&
    (!restoreJob?.requires_legacy_key_acknowledgement || restoreLegacyKeyAcknowledged);

  const restoreShowProgress = restoreJob != null;
  const backupJobError = getFailedJobError(backupJob);
  const restoreJobError = getFailedJobError(restoreJob);
  const setActiveTabWithOptionalFocus = (tab: BackupRestoreTab, focusTab: boolean) => {
    setActiveTab(tab);

    if (!focusTab) {
      return;
    }

    const target = tab === 'backup' ? backupTabRef.current : restoreTabRef.current;
    target?.focus();
  };

  const handleTabKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
      event.preventDefault();
      setActiveTabWithOptionalFocus(activeTab === 'backup' ? 'restore' : 'backup', true);
      return;
    }

    if (event.key === 'Home') {
      event.preventDefault();
      setActiveTabWithOptionalFocus('backup', true);
      return;
    }

    if (event.key === 'End') {
      event.preventDefault();
      setActiveTabWithOptionalFocus('restore', true);
    }
  };

  const handleStartBackup = async () => {
    if (backupEncrypted) {
      if (!backupPassword) {
        reportServerOperationError('Backup password is required when encryption is enabled.');
        return;
      }
      if (!passwordMeetsRequirements(backupPassword, passwordPolicy)) {
        reportServerOperationError(
          'Backup password does not meet the current export password policy.',
        );
        return;
      }
      if (backupPassword !== backupPasswordConfirm) {
        reportServerOperationError('Backup passwords do not match.');
        return;
      }
    }

    setBackupBusy(true);
    clearBackupCleanupTimer();
    backupDeliveryNotifiedRef.current = false;
    lastCompletedBackupHistoryRefreshRef.current = null;
    pollingErrorRef.current.backup = null;
    try {
      const created = await api.createServerBackupJob({
        scope: backupScope,
        encrypt: backupEncrypted,
        password: backupEncrypted ? backupPassword : undefined,
      });
      setBackupJob(created);
      setBackupDownloadPending(false);
      onServerBackupJobObserved?.(created);
    } catch (error) {
      reportServerOperationError(error instanceof Error ? error.message : 'Failed to start backup');
    } finally {
      setBackupBusy(false);
    }
  };

  const handleCancelBackup = async () => {
    if (!backupJob?.id) return;
    setBackupBusy(true);
    pollingErrorRef.current.backup = null;
    try {
      await api.cancelServerBackupJob(backupJob.id);
      setBackupJob((current) =>
        current ? { ...current, status: 'cancelled', message: 'Backup cancelled' } : current,
      );
    } catch (error) {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to cancel backup',
      );
    } finally {
      setBackupBusy(false);
    }
  };

  const handleDownloadBackup = async () => {
    if (!backupJob?.id) return;
    setBackupDownloadPending(true);
    pollingErrorRef.current.backup = null;
    try {
      await api.downloadServerBackup(backupJob.id);
      setDownloadedBackupExportIds((current) => new Set(current).add(backupJob.id));
      const exports = await loadBackupExports();
      if (exports.some((item) => item.job_id === backupJob.id)) {
        scheduleBackupCleanup(backupJob.id);
      }
    } catch (error) {
      setBackupDownloadPending(false);
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to download backup',
      );
    } finally {
      setBackupBusy(false);
    }
  };

  const handleDownloadBackupExport = async (jobId: string) => {
    try {
      await api.downloadServerBackup(jobId);
      setDownloadedBackupExportIds((current) => new Set(current).add(jobId));
      await loadBackupExports();
    } catch (error) {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to download backup',
      );
    }
  };

  const handleDeleteBackupExport = async (jobId: string) => {
    setDeletingBackupExportJobId(jobId);
    clearBackupCleanupTimer();
    pollingErrorRef.current.backup = null;
    try {
      await api.deleteServerBackupJob(jobId);
      setBackupExports((current) => current.filter((item) => item.job_id !== jobId));
    } catch (error) {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to delete backup export',
      );
    } finally {
      setDeletingBackupExportJobId(null);
    }
  };

  const handleRestoreValidate = async () => {
    if (!restoreArchiveFile) {
      reportServerOperationError('Choose a backup archive first.');
      return;
    }

    setRestoreBusy(true);
    clearRecoveredEnvironmentState();
    pollingErrorRef.current.restore = null;
    try {
      const upload = await api.uploadServerBackupArchive(restoreArchiveFile);
      const created = await api.createServerRestoreJob({
        upload_id: upload.upload_id,
        password: restorePassword || undefined,
        skip_migrations: restoreSkipMigrations,
        postgres_data_only: restorePostgresDataOnly,
        replace_data: restoreReplaceData,
        mirror_local_admin_access: restoreMirrorLocalAdmins,
        mirror_local_admin_from: restoreMirrorLocalAdminFrom || 'auto',
        local_admin_username: restoreLocalAdminUsername.trim() || undefined,
      });
      setRestoreConfirmationText('');
      setRestoreLegacyKeyAcknowledged(false);
      setRestoreJob(created);
      onServerRestoreJobObserved?.(created);
    } catch (error) {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to validate restore archive',
      );
    } finally {
      setRestoreBusy(false);
    }
  };

  const handleCommitRestore = async () => {
    if (!restoreJob?.id || !restoreCanCommit) {
      return;
    }

    setRestoreBusy(true);
    clearRecoveredEnvironmentState();
    pollingErrorRef.current.restore = null;
    try {
      const nextJob = await api.commitServerRestoreJob(restoreJob.id, {
        confirmation_text: restoreConfirmationText,
        acknowledge_legacy_key: restoreLegacyKeyAcknowledged,
      });
      setRestoreJob(nextJob);
      onServerRestoreJobObserved?.(nextJob);
    } catch (error) {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to start restore',
      );
    } finally {
      setRestoreBusy(false);
    }
  };

  const handleRevealRecoveredEnvironment = async () => {
    if (!restoreJob?.id) {
      return;
    }
    setRecoveryBusy(true);
    setRecoveredEnvironmentVariables(null);
    try {
      const recovery: ServerDeploymentEnvironmentRecovery =
        await api.recoverServerDeploymentEnvironment(restoreJob.id);
      const manifestNames = restoreManifestEnvironmentVariableNames;
      const recoveredNames = recovery.variable_names;
      const recoveredVariableNames = Object.keys(recovery.variables);
      if (
        !hasExactNames(recoveredNames, manifestNames) ||
        !hasExactNames(recoveredVariableNames, manifestNames)
      ) {
        throw new Error('Recovered environment variables did not match the validated manifest.');
      }
      setRecoveredEnvironmentVariables(recovery.variables);
    } catch (error) {
      reportServerOperationError(
        error instanceof Error ? error.message : 'Failed to recover deployment environment',
      );
    } finally {
      setRecoveryBusy(false);
    }
  };

  return (
    <SettingsAccordionSection
      id="server-backup-restore"
      title="Server Backup & Restore"
      open={open}
      onToggle={onToggle}
      status={backupJob && !isBackupJobTerminal(backupJob) ? backupJob.status : restoreJob?.status}
    >
      <fieldset id={`setting-${SERVER_BACKUP_RESTORE_HIGHLIGHT}`}>
        <legend>Server Backup & Restore</legend>

        <div className="server-backup-tabs" role="tablist" aria-label="Backup and restore modes">
          <button
            id="server-backup-tab"
            ref={backupTabRef}
            type="button"
            role="tab"
            aria-selected={activeTab === 'backup'}
            aria-controls="server-backup-panel"
            tabIndex={activeTab === 'backup' ? 0 : -1}
            className="server-backup-tab"
            onClick={() => setActiveTabWithOptionalFocus('backup', false)}
            onKeyDown={handleTabKeyDown}
          >
            Backup
          </button>
          <button
            id="server-restore-tab"
            ref={restoreTabRef}
            type="button"
            role="tab"
            aria-selected={activeTab === 'restore'}
            aria-controls="server-restore-panel"
            tabIndex={activeTab === 'restore' ? 0 : -1}
            className="server-backup-tab"
            onClick={() => setActiveTabWithOptionalFocus('restore', false)}
            onKeyDown={handleTabKeyDown}
          >
            Restore
          </button>
        </div>

        {activeTab === 'backup' ? (
          <section
            id="server-backup-panel"
            className="server-backup-panel server-backup-tab-panel"
            role="tabpanel"
            aria-labelledby="server-backup-tab"
          >
            <div className="server-backup-grid">
              <div className="server-backup-column server-backup-column--primary">
                <div className="form-group">
                  <span id="server-backup-scope-label" className="server-backup-field-label">
                    Backup scope
                  </span>
                  <div
                    className="server-backup-option-grid server-backup-option-grid--segmented"
                    role="radiogroup"
                    aria-labelledby="server-backup-scope-label"
                  >
                    {(['full', 'database', 'files'] as ServerBackupScope[]).map((scope) => (
                      <label key={scope} className="server-backup-choice-card">
                        <input
                          type="radio"
                          name="server-backup-scope"
                          checked={backupScope === scope}
                          onChange={() => setBackupScope(scope)}
                        />
                        <span>
                          {scope === 'full'
                            ? 'Full server'
                            : scope === 'database'
                              ? 'Database only'
                              : 'Files only'}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label className="server-backup-toggle-row">
                    <span className="toggle-switch">
                      <input
                        type="checkbox"
                        aria-label="Encrypt backup archive"
                        checked={backupEncrypted}
                        onChange={(event) => setBackupEncrypted(event.target.checked)}
                      />
                      <span className="toggle-slider" />
                    </span>
                    <span className="server-backup-toggle-text">Encrypt backup archive</span>
                    <span className="server-backup-toggle-state">
                      {backupEncrypted ? 'On' : 'Off'}
                    </span>
                  </label>
                  <p className="field-help">
                    Encryption is on by default and uses the same password policy as secure exports.
                  </p>
                </div>
              </div>

              <div className="server-backup-column server-backup-column--secondary">
                {backupEncrypted ? (
                  <>
                    <div className="form-group">
                      <label htmlFor="server-backup-password">Backup password</label>
                      <input
                        id="server-backup-password"
                        type="password"
                        value={backupPassword}
                        onChange={(event) => setBackupPassword(event.target.value)}
                        autoComplete="new-password"
                      />
                      <PasswordRequirementsChecklist
                        password={backupPassword}
                        policy={passwordPolicy}
                      />
                    </div>

                    <div className="form-group">
                      <label htmlFor="server-backup-password-confirm">
                        Confirm backup password
                      </label>
                      <input
                        id="server-backup-password-confirm"
                        type="password"
                        value={backupPasswordConfirm}
                        onChange={(event) => setBackupPasswordConfirm(event.target.value)}
                        autoComplete="new-password"
                      />
                    </div>
                  </>
                ) : null}
              </div>

              {backupJob ? (
                <div className="server-backup-status-row">
                  <ProgressStatusCard
                    job={backupJob}
                    ariaLabel="Backup progress"
                    kind="backup"
                    className="server-backup-status-card server-backup-status-card--full-width"
                    suppressedMessage={backupJobError}
                  />
                </div>
              ) : null}
            </div>

            <div className="server-backup-actions-row">
              <button
                type="button"
                className="btn"
                onClick={handleStartBackup}
                disabled={backupBusy}
              >
                {backupBusy ? 'Starting...' : 'Start Backup'}
              </button>
              {backupJob &&
              !isBackupReadyForDownload(backupJob) &&
              !isBackupJobTerminal(backupJob) ? (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleCancelBackup}
                  disabled={backupBusy}
                >
                  Cancel Backup
                </button>
              ) : null}
              {isBackupReadyForDownload(backupJob) ? (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={handleDownloadBackup}
                  disabled={backupDownloadPending}
                >
                  {backupDownloadPending ? 'Downloading...' : 'Download Backup'}
                </button>
              ) : null}
            </div>

            {backupExports.length > 0 ? (
              <div className="server-backup-exports-card">
                <div className="server-backup-exports-header">
                  <strong>Previous exports</strong>
                </div>
                <div className="server-backup-exports-list">
                  {backupExports.map((item) => {
                    const hasDownloaded = downloadedBackupExportIds.has(item.job_id);
                    const isDeleting = deletingBackupExportJobId === item.job_id;
                    return (
                      <div key={item.job_id} className="server-backup-export-item">
                        <div className="server-backup-export-main">
                          <div className="server-backup-export-name" title={item.file_name}>
                            {item.file_name}
                          </div>
                          <div className="server-backup-export-meta userspace-muted">
                            <span>{formatBytes(item.size_bytes)}</span>
                            <span>{formatExportDate(item.created_at)}</span>
                            <span>{`Scope: ${item.scope ?? '—'}`}</span>
                            <span>{item.encrypted ? 'Encrypted' : 'Unencrypted'}</span>
                            {item.delivered_at ? (
                              <span>{`Delivered: ${formatExportDate(item.delivered_at)}`}</span>
                            ) : null}
                          </div>
                        </div>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => void handleDownloadBackupExport(item.job_id)}
                          disabled={isDeleting}
                          title="Download backup"
                        >
                          <ArrowDownToLine size={12} /> {hasDownloaded ? 'Downloaded' : 'Download'}
                        </button>
                        <DeleteConfirmButton
                          onDelete={() => void handleDeleteBackupExport(item.job_id)}
                          disabled={isDeleting}
                          deleting={isDeleting}
                          className="btn btn-sm btn-danger"
                          title="Delete backup"
                          buttonText="Delete"
                        />
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : null}
          </section>
        ) : null}

        {activeTab === 'restore' ? (
          <section
            id="server-restore-panel"
            className="server-backup-panel server-backup-tab-panel"
            role="tabpanel"
            aria-labelledby="server-restore-tab"
          >
            <div className="server-backup-grid">
              <div className="server-backup-column server-backup-column--primary">
                <div className="form-group">
                  <label htmlFor="server-restore-archive">Restore archive</label>
                  <input
                    id="server-restore-archive"
                    type="file"
                    onChange={(event) => setRestoreArchiveFile(event.target.files?.[0] ?? null)}
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="server-restore-password">Restore password</label>
                  <input
                    id="server-restore-password"
                    type="password"
                    value={restorePassword}
                    onChange={(event) => setRestorePassword(event.target.value)}
                    autoComplete="off"
                  />
                  <p className="field-help">Leave blank only for plaintext legacy backups.</p>
                </div>
              </div>

              <div className="server-backup-column server-backup-column--secondary">
                <div className="server-backup-toggle-stack">
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={restoreSkipMigrations}
                      onChange={(event) => setRestoreSkipMigrations(event.target.checked)}
                    />
                    <span>Skip migrations after restore</span>
                  </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={restorePostgresDataOnly}
                      onChange={(event) => setRestorePostgresDataOnly(event.target.checked)}
                    />
                    <span>Restore PostgreSQL data only</span>
                  </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={restoreReplaceData}
                      onChange={(event) => setRestoreReplaceData(event.target.checked)}
                    />
                    <span>Replace existing server data</span>
                  </label>
                  <label className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={restoreMirrorLocalAdmins}
                      onChange={(event) => setRestoreMirrorLocalAdmins(event.target.checked)}
                    />
                    <span>Mirror local admin accounts after restore</span>
                  </label>
                </div>

                {restoreMirrorLocalAdmins ? (
                  <div className="server-backup-mirror-grid">
                    <div className="form-group">
                      <label htmlFor="server-restore-mirror-from">
                        Mirror local admin access from
                      </label>
                      <input
                        id="server-restore-mirror-from"
                        type="text"
                        value={restoreMirrorLocalAdminFrom}
                        onChange={(event) => setRestoreMirrorLocalAdminFrom(event.target.value)}
                        placeholder="auto"
                        autoComplete="off"
                      />
                      <p className="field-help">
                        Use <code>auto</code> to let the restore script choose the source account.
                      </p>
                    </div>

                    <div className="form-group">
                      <label htmlFor="server-restore-local-admin-username">
                        Target local admin username
                      </label>
                      <input
                        id="server-restore-local-admin-username"
                        type="text"
                        value={restoreLocalAdminUsername}
                        onChange={(event) => setRestoreLocalAdminUsername(event.target.value)}
                        placeholder="Optional local admin username"
                        autoComplete="off"
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              {restoreShowProgress ? (
                <div className="server-backup-status-row">
                  <ProgressStatusCard
                    job={restoreJob}
                    ariaLabel="Restore progress"
                    kind="restore"
                    suppressedMessage={restoreJobError}
                  />
                </div>
              ) : null}

              {restoreJob?.manifest ? (
                <div className="server-backup-status-row">
                  <div className="server-backup-manifest-card">
                    <div className="server-backup-status-header">
                      <strong>Manifest</strong>
                      <span>{restoreJob.status}</span>
                    </div>
                    {renderManifestSummary(restoreJob.manifest)}
                    {restoreJob.manifest.legacy_embedded_key ? (
                      <p className="server-backup-inline-warning">Legacy embedded key detected</p>
                    ) : null}
                    {restoreJob.restart_required ? (
                      <p className="server-backup-inline-warning">
                        Restart required:{' '}
                        {restoreJob.restart_state || 'Waiting for compose restart'}
                      </p>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {hasRecoverableEnvironmentVariables ? (
                <div className="server-backup-status-row">
                  <div
                    className={`server-backup-env-recovery${recoveredEnvironmentVariables ? ' server-backup-env-recovery--revealed' : ''}`}
                  >
                    <div className="server-backup-status-header">
                      <strong>Environment variables</strong>
                      {!recoveredEnvironmentVariables ? (
                        <span className="badge badge-muted">
                          {restoreManifestEnvironmentVariableNames.length} variables
                        </span>
                      ) : (
                        <div className="server-backup-env-recovery-actions">
                          <button
                            type="button"
                            className="btn btn-link"
                            onClick={clearRecoveredEnvironmentState}
                            aria-label="Clear loaded values"
                          >
                            Clear loaded values
                          </button>
                        </div>
                      )}
                    </div>

                    {hasSensitiveRecoveredEnvironmentVariables ? (
                      <p className="server-backup-inline-warning server-backup-env-recovery-warning">
                        This backup contains <code>DATABASE_URL</code> or{' '}
                        <code>POSTGRES_PASSWORD</code>. Do not apply recovered values to a different
                        deployment without updating them first.
                      </p>
                    ) : null}

                    <ul
                      className="server-backup-env-name-list"
                      aria-label="Captured environment variable names"
                    >
                      {restoreManifestEnvironmentVariableNames.map((name) => (
                        <li key={name} className="server-backup-env-name-item">
                          <div className="server-backup-env-name-row">
                            <code>{name}</code>
                            {SENSITIVE_DEPLOYMENT_ENVIRONMENT_VARIABLES.includes(
                              name as (typeof SENSITIVE_DEPLOYMENT_ENVIRONMENT_VARIABLES)[number],
                            ) ? (
                              <span
                                className="server-backup-env-sensitive-badge"
                                aria-label="Cross-deployment warning"
                              >
                                <AlertTriangle size={12} aria-hidden="true" />
                                <span>Cross-deployment</span>
                              </span>
                            ) : null}
                            {recoveredEnvironmentVariables ? (
                              <div className="server-backup-env-name-actions">
                                <InlineCopyButton
                                  copyText={() => {
                                    const value = recoveredEnvironmentVariables[name];
                                    return value == null
                                      ? null
                                      : formatDotenvAssignment(name, value);
                                  }}
                                  className="btn btn-sm btn-secondary"
                                  title={`Copy assignment for ${name}`}
                                  ariaLabel={`Copy assignment for ${name}`}
                                  label="Copy assignment"
                                  copiedLabel="Copied"
                                  allowDomFallback={false}
                                  onCopyError={(error) => reportServerOperationError(error.message)}
                                />
                                <InlineCopyButton
                                  copyText={() => recoveredEnvironmentVariables[name] ?? null}
                                  className="btn btn-sm btn-secondary"
                                  title={`Copy value for ${name}`}
                                  ariaLabel={`Copy value for ${name}`}
                                  label="Copy value"
                                  copiedLabel="Copied"
                                  allowDomFallback={false}
                                  onCopyError={(error) => reportServerOperationError(error.message)}
                                />
                              </div>
                            ) : null}
                          </div>
                        </li>
                      ))}
                    </ul>
                    {!recoveredEnvironmentVariables ? (
                      <button
                        type="button"
                        className="btn btn-secondary btn-sm"
                        onClick={() => {
                          void handleRevealRecoveredEnvironment();
                        }}
                        disabled={recoveryBusy}
                      >
                        {recoveryBusy ? 'Loading…' : 'Load values'}
                      </button>
                    ) : null}
                  </div>
                </div>
              ) : null}

              {requiresRestoreConfirmation(restoreJob) ? (
                <div className="server-backup-confirmation-block">
                  <div className="form-group">
                    <label htmlFor="server-restore-confirmation">
                      Type the confirmation phrase
                    </label>
                    <input
                      id="server-restore-confirmation"
                      type="text"
                      value={restoreConfirmationText}
                      onChange={(event) => setRestoreConfirmationText(event.target.value)}
                      placeholder={requiredConfirmationText}
                    />
                    <p className="field-help">
                      Required phrase: <code>{requiredConfirmationText}</code>
                    </p>
                  </div>

                  {restoreJob?.requires_legacy_key_acknowledgement ? (
                    <div className="form-group">
                      <label className="checkbox-label">
                        <input
                          type="checkbox"
                          checked={restoreLegacyKeyAcknowledged}
                          onChange={(event) =>
                            setRestoreLegacyKeyAcknowledged(event.target.checked)
                          }
                        />
                        <span>
                          I understand this backup contains a legacy embedded encryption key
                        </span>
                      </label>
                    </div>
                  ) : null}
                </div>
              ) : null}
            </div>

            <div className="server-backup-actions-row">
              <button
                type="button"
                className="btn"
                onClick={handleRestoreValidate}
                disabled={restoreBusy}
              >
                {restoreBusy ? 'Validating...' : 'Validate Restore Archive'}
              </button>
              {requiresRestoreConfirmation(restoreJob) ? (
                <button
                  type="button"
                  className="btn btn-danger"
                  onClick={handleCommitRestore}
                  disabled={!restoreCanCommit || restoreBusy}
                >
                  {restoreBusy ? 'Starting restore...' : 'Confirm Restore'}
                </button>
              ) : null}
            </div>
          </section>
        ) : null}
      </fieldset>
    </SettingsAccordionSection>
  );
}
