import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/api';
import type {
  ServerBackupJob,
  ServerBackupManifest,
  ServerBackupScope,
  ServerRestoreScopeSelection,
  ServerRestoreJob,
} from '@/types';
import {
  getExportPasswordPolicy,
  passwordMeetsRequirements,
  type ExportPasswordPolicy,
} from '@/utils/exportPasswordPolicy';
import { PasswordRequirementsChecklist } from '../shared/PasswordRequirementsChecklist';
import { SERVER_BACKUP_RESTORE_HIGHLIGHT } from '../shared/securityWarnings';
import type { SettingsAccordionSectionId } from './settingsAccordionState';
import { SettingsAccordionSection } from './SettingsAccordionSection';

const POLL_INTERVAL_MS = 2000;

interface ServerBackupRestoreSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  settings?: Partial<ExportPasswordPolicy> | null;
  onEncryptedArtifactDelivered?: () => void;
}

function getBackupJobMessage(job: ServerBackupJob | null): string | null {
  if (!job) return null;
  return job.message || job.error || null;
}

function getRestoreJobMessage(job: ServerRestoreJob | null): string | null {
  if (!job) return null;
  return job.message || job.error || null;
}

function isBackupJobTerminal(job: ServerBackupJob | null): boolean {
  if (!job) return true;
  return ['delivered', 'cancelled', 'failed'].includes(job.status);
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

function manifestValue(value: string | number | boolean | null | undefined): string {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (value == null || value === '') return '—';
  return String(value);
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

export function ServerBackupRestoreSettingsSection(
  props: ServerBackupRestoreSettingsSectionProps,
): JSX.Element {
  const { open, onToggle, settings = null, onEncryptedArtifactDelivered } = props;
  const passwordPolicy = useMemo(() => getExportPasswordPolicy(settings), [settings]);
  const [backupScope, setBackupScope] = useState<ServerBackupScope>('full');
  const [backupEncrypted, setBackupEncrypted] = useState(true);
  const [backupPassword, setBackupPassword] = useState('');
  const [backupPasswordConfirm, setBackupPasswordConfirm] = useState('');
  const [backupError, setBackupError] = useState<string | null>(null);
  const [backupJob, setBackupJob] = useState<ServerBackupJob | null>(null);
  const [backupBusy, setBackupBusy] = useState(false);
  const [backupDownloadPending, setBackupDownloadPending] = useState(false);
  const backupDeliveryNotifiedRef = useRef(false);

  const [restoreArchiveFile, setRestoreArchiveFile] = useState<File | null>(null);
  const [restoreScope, setRestoreScope] = useState<ServerRestoreScopeSelection>('archive');
  const [restorePassword, setRestorePassword] = useState('');
  const [restoreSkipMigrations, setRestoreSkipMigrations] = useState(false);
  const [restorePostgresDataOnly, setRestorePostgresDataOnly] = useState(false);
  const [restoreReplaceData, setRestoreReplaceData] = useState(false);
  const [restoreMirrorLocalAdmins, setRestoreMirrorLocalAdmins] = useState(false);
  const [restoreMirrorLocalAdminFrom, setRestoreMirrorLocalAdminFrom] = useState('auto');
  const [restoreLocalAdminUsername, setRestoreLocalAdminUsername] = useState('');
  const [restoreConfirmationText, setRestoreConfirmationText] = useState('');
  const [restoreLegacyKeyAcknowledged, setRestoreLegacyKeyAcknowledged] = useState(false);
  const [restoreError, setRestoreError] = useState<string | null>(null);
  const [restoreJob, setRestoreJob] = useState<ServerRestoreJob | null>(null);
  const [restoreBusy, setRestoreBusy] = useState(false);

  const backupJobId = backupJob?.id ?? null;
  const backupJobStatus = backupJob?.status ?? null;
  const backupShouldPoll =
    backupJobId != null &&
    backupJobStatus != null &&
    !isBackupJobTerminal(backupJob) &&
    (backupJobStatus !== 'completed' || backupDownloadPending);

  const restoreJobId = restoreJob?.id ?? null;
  const restoreJobStatus = restoreJob?.status ?? null;
  const restoreShouldPoll =
    restoreJobId != null &&
    restoreJobStatus != null &&
    !isRestoreJobTerminal(restoreJob) &&
    !requiresRestoreConfirmation(restoreJob);

  useEffect(() => {
    if (!backupShouldPoll || !backupJobId) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.getServerBackupJob(backupJobId);
        setBackupJob(nextJob);
      } catch (error) {
        setBackupError(error instanceof Error ? error.message : 'Failed to refresh backup status');
      }
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [backupJobId, backupShouldPoll]);

  useEffect(() => {
    if (!restoreShouldPoll || !restoreJobId) {
      return;
    }

    const timer = window.setInterval(async () => {
      try {
        const nextJob = await api.getServerRestoreJob(restoreJobId);
        setRestoreJob(nextJob);
      } catch (error) {
        setRestoreError(
          error instanceof Error ? error.message : 'Failed to refresh restore status',
        );
      }
    }, POLL_INTERVAL_MS);

    return () => {
      window.clearInterval(timer);
    };
  }, [restoreJobId, restoreShouldPoll]);

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

  const backupStatus = getBackupJobMessage(backupJob);
  const restoreStatus = getRestoreJobMessage(restoreJob);
  const requiredConfirmationText = restoreJob?.required_confirmation || 'RESTORE';
  const restoreCanCommit =
    requiresRestoreConfirmation(restoreJob) &&
    restoreConfirmationText === requiredConfirmationText &&
    (!restoreJob?.requires_legacy_key_acknowledgement || restoreLegacyKeyAcknowledged);

  const handleStartBackup = async () => {
    if (backupEncrypted) {
      if (!backupPassword) {
        setBackupError('Backup password is required when encryption is enabled.');
        return;
      }
      if (!passwordMeetsRequirements(backupPassword, passwordPolicy)) {
        setBackupError('Backup password does not meet the current export password policy.');
        return;
      }
      if (backupPassword !== backupPasswordConfirm) {
        setBackupError('Backup passwords do not match.');
        return;
      }
    }

    setBackupBusy(true);
    setBackupError(null);
    backupDeliveryNotifiedRef.current = false;
    try {
      const created = await api.createServerBackupJob({
        scope: backupScope,
        encrypt: backupEncrypted,
        password: backupEncrypted ? backupPassword : undefined,
      });
      setBackupJob(created);
      setBackupDownloadPending(false);
    } catch (error) {
      setBackupError(error instanceof Error ? error.message : 'Failed to start backup');
    } finally {
      setBackupBusy(false);
    }
  };

  const handleCancelBackup = async () => {
    if (!backupJob?.id) return;
    setBackupBusy(true);
    setBackupError(null);
    try {
      await api.cancelServerBackupJob(backupJob.id);
      setBackupJob((current) =>
        current ? { ...current, status: 'cancelled', message: 'Backup cancelled' } : current,
      );
    } catch (error) {
      setBackupError(error instanceof Error ? error.message : 'Failed to cancel backup');
    } finally {
      setBackupBusy(false);
    }
  };

  const handleDownloadBackup = async () => {
    if (!backupJob?.id) return;
    setBackupDownloadPending(true);
    setBackupError(null);
    try {
      await api.downloadServerBackup(backupJob.id);
    } catch (error) {
      setBackupDownloadPending(false);
      setBackupError(error instanceof Error ? error.message : 'Failed to download backup');
    } finally {
      setBackupBusy(false);
    }
  };

  const handleRestoreValidate = async () => {
    if (!restoreArchiveFile) {
      setRestoreError('Choose a backup archive first.');
      return;
    }

    setRestoreBusy(true);
    setRestoreError(null);
    try {
      const upload = await api.uploadServerBackupArchive(restoreArchiveFile);
      const created = await api.createServerRestoreJob({
        upload_id: upload.upload_id,
        password: restorePassword || undefined,
        scope_override: restoreScope === 'archive' ? undefined : restoreScope,
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
    } catch (error) {
      setRestoreError(
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
    setRestoreError(null);
    try {
      const nextJob = await api.commitServerRestoreJob(restoreJob.id, {
        confirmation_text: restoreConfirmationText,
        acknowledge_legacy_key: restoreLegacyKeyAcknowledged,
      });
      setRestoreJob(nextJob);
    } catch (error) {
      setRestoreError(error instanceof Error ? error.message : 'Failed to start restore');
    } finally {
      setRestoreBusy(false);
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
        <p className="fieldset-help">
          Create encrypted server backups, download them natively, inspect uploaded archives, and
          validate destructive restore options before commit.
        </p>

        <div className="server-backup-grid">
          <section className="server-backup-panel" aria-labelledby="server-backup-create-title">
            <div className="server-backup-panel-header">
              <h3 id="server-backup-create-title">Create backup</h3>
              <span className="server-backup-panel-kicker">Managed key exports</span>
            </div>

            <div className="form-group">
              <span id="server-backup-scope-label" className="server-backup-field-label">
                Backup scope
              </span>
              <div
                className="server-backup-option-grid"
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
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={backupEncrypted}
                  onChange={(event) => setBackupEncrypted(event.target.checked)}
                />
                <span>Encrypt backup archive</span>
              </label>
              <p className="field-help">
                Encryption is on by default and uses the same password policy as secure exports.
              </p>
            </div>

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
                  <label htmlFor="server-backup-password-confirm">Confirm backup password</label>
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

            {backupError ? <p className="status-message error">{backupError}</p> : null}
            {backupStatus ? <p className="status-message info">{backupStatus}</p> : null}

            {backupJob ? (
              <div className="server-backup-status-card">
                <div className="server-backup-status-row">
                  <strong>Status</strong>
                  <span>{backupJob.status}</span>
                </div>
                <div
                  className="server-backup-progress"
                  role="progressbar"
                  aria-label="Backup progress"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={getProgressValue(backupJob.progress)}
                >
                  <span
                    className="server-backup-progress-bar"
                    style={{ width: `${getProgressValue(backupJob.progress)}%` }}
                  />
                </div>
              </div>
            ) : (
              <div
                className="server-backup-progress server-backup-progress-idle"
                role="progressbar"
                aria-label="Backup progress"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={0}
              >
                <span className="server-backup-progress-bar" style={{ width: '0%' }} />
              </div>
            )}

            <div className="form-actions">
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
          </section>

          <section className="server-backup-panel" aria-labelledby="server-restore-title">
            <div className="server-backup-panel-header">
              <h3 id="server-restore-title">Restore backup</h3>
              <span className="server-backup-panel-kicker">Preflight before commit</span>
            </div>

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

            <div className="form-group">
              <span id="server-restore-scope-label" className="server-backup-field-label">
                Restore scope
              </span>
              <div
                className="server-backup-option-grid"
                role="radiogroup"
                aria-labelledby="server-restore-scope-label"
              >
                {(['archive', 'full', 'database', 'files'] as ServerRestoreScopeSelection[]).map(
                  (scope) => (
                    <label key={`restore-${scope}`} className="server-backup-choice-card">
                      <input
                        type="radio"
                        name="server-restore-scope"
                        checked={restoreScope === scope}
                        onChange={() => setRestoreScope(scope)}
                      />
                      <span>
                        {scope === 'archive'
                          ? 'Use archive scope'
                          : scope === 'full'
                            ? 'Restore full server'
                            : scope === 'database'
                              ? 'Restore database only'
                              : 'Restore files only'}
                      </span>
                    </label>
                  ),
                )}
              </div>
            </div>

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
                  <label htmlFor="server-restore-mirror-from">Mirror local admin access from</label>
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

            {restoreError ? <p className="status-message error">{restoreError}</p> : null}
            {restoreStatus ? <p className="status-message info">{restoreStatus}</p> : null}

            {restoreJob?.manifest ? (
              <div className="server-backup-manifest-card">
                <div className="server-backup-status-row">
                  <strong>Manifest</strong>
                  <span>{restoreJob.status}</span>
                </div>
                {renderManifestSummary(restoreJob.manifest)}
                {restoreJob.manifest.legacy_embedded_key ? (
                  <p className="status-message warning">Legacy embedded key detected</p>
                ) : null}
                {restoreJob.restart_required ? (
                  <p className="status-message warning">
                    Restart required: {restoreJob.restart_state || 'Waiting for compose restart'}
                  </p>
                ) : null}
              </div>
            ) : null}

            {requiresRestoreConfirmation(restoreJob) ? (
              <div className="server-backup-confirmation-block">
                <div className="form-group">
                  <label htmlFor="server-restore-confirmation">Type the confirmation phrase</label>
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
                        onChange={(event) => setRestoreLegacyKeyAcknowledged(event.target.checked)}
                      />
                      <span>
                        I understand this backup contains a legacy embedded encryption key
                      </span>
                    </label>
                  </div>
                ) : null}
              </div>
            ) : null}

            <div className="form-actions">
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
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
