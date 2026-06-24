import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, ArrowDownToLine, ArrowUpToLine, Check, Database, GitBranch, Link2, RefreshCw, RefreshCcw, Upload, X } from 'lucide-react';

import { api } from '@/api';
import { formatBytes } from '@/utils';
import { SQLITE_IMPORT_DEFAULT_MAX_BYTES } from '@/utils/sqliteImport';
import type {
  RepoVisibilityResponse,
  UserSpaceWorkspaceArchiveExportListItem,
  UserSpaceWorkspaceArchiveExportTask,
  UserSpaceWorkspaceArchiveFormat,
  UserSpaceWorkspaceArchiveImportTask,
  UserSpaceWorkspaceSqliteImportTask,
  UserSpaceWorkspace,
  UserSpaceWorkspaceScmExportRequest,
  UserSpaceWorkspaceScmImportRequest,
  UserSpaceWorkspaceScmImportTask,
  UserSpaceWorkspaceScmImportTaskPhase,
  UserSpaceWorkspaceScmPreviewResponse,
  UserSpaceWorkspaceScmSettingsRequest,
  UserSpaceWorkspaceScmStatus,
  UserSpaceWorkspaceScmSyncResponse,
} from '@/types';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { Popover } from './Popover';
import { defaultScheduleStartMinute, defaultScheduleTimezone, ScheduleStartTimeInput } from './ScheduleStartTimeInput';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { ToastContainer, useToast } from './shared/Toast';

type ModalTab = 'git-source' | 'archive' | 'sql-import';
type WizardMode = 'import' | 'export' | 'sql-import';
type WizardStep = 'input' | 'review' | 'result';
type StatusType = 'info' | 'success' | 'error' | null;
type ArchiveMode = 'export' | 'import';
type ArchiveStep = 'choose' | 'configure';

type WorkspaceScmWizardActivity =
  | {
    kind: 'preview';
    workspaceId: string;
    status: 'running' | 'ready' | 'failed';
    direction: 'import' | 'export';
    preview?: UserSpaceWorkspaceScmPreviewResponse;
    error?: string;
  }
  | {
    kind: 'import-task';
    workspaceId: string;
    status: 'running';
    taskId: string;
  };

const EMPTY_STATUS = { type: null, message: '' } as const;
const ARCHIVE_POLL_INTERVAL_MS = 1000;
const SQLITE_IMPORT_POLL_INTERVAL_MS = ARCHIVE_POLL_INTERVAL_MS;
const SCM_IMPORT_POLL_INTERVAL_MS = ARCHIVE_POLL_INTERVAL_MS;

const scmWizardActivityByWorkspace = new Map<string, WorkspaceScmWizardActivity>();
const scmWizardActivityListeners = new Set<() => void>();

function workspaceScmWizardActivityStorageKey(workspaceId: string): string {
  return `ragtime:workspace-scm-wizard-activity:${workspaceId}`;
}

function readStoredWorkspaceScmWizardActivity(workspaceId: string): WorkspaceScmWizardActivity | null {
  try {
    const raw = window.sessionStorage.getItem(workspaceScmWizardActivityStorageKey(workspaceId));
    if (!raw) return null;
    const activity = JSON.parse(raw) as WorkspaceScmWizardActivity;
    if (activity.workspaceId !== workspaceId) return null;
    return activity;
  } catch {
    return null;
  }
}

function setWorkspaceScmWizardActivity(workspaceId: string, activity: WorkspaceScmWizardActivity | null): void {
  if (activity) {
    scmWizardActivityByWorkspace.set(workspaceId, activity);
    if (activity.kind === 'preview' && activity.status === 'ready') {
      try {
        window.sessionStorage.setItem(workspaceScmWizardActivityStorageKey(workspaceId), JSON.stringify(activity));
      } catch {
        // Session persistence is best-effort; in-memory state still works.
      }
    } else {
      try {
        window.sessionStorage.removeItem(workspaceScmWizardActivityStorageKey(workspaceId));
      } catch {
        // Ignore storage failures.
      }
    }
  } else {
    scmWizardActivityByWorkspace.delete(workspaceId);
    try {
      window.sessionStorage.removeItem(workspaceScmWizardActivityStorageKey(workspaceId));
    } catch {
      // Ignore storage failures.
    }
  }
  scmWizardActivityListeners.forEach((listener) => listener());
}

export function useWorkspaceScmWizardActivity(workspace: UserSpaceWorkspace | null | undefined): WorkspaceScmWizardActivity | null {
  const [version, setVersion] = useState(0);
  useEffect(() => {
    const listener = () => setVersion((version) => version + 1);
    scmWizardActivityListeners.add(listener);
    return () => {
      scmWizardActivityListeners.delete(listener);
    };
  }, []);

  return useMemo(() => {
    if (!workspace) return null;
    if (workspace.scm_import_task_id && isScmImportPhaseInProgress(workspace.scm_import_task_phase)) {
      return {
        kind: 'import-task',
        workspaceId: workspace.id,
        status: 'running',
        taskId: workspace.scm_import_task_id,
      };
    }

    const localActivity = scmWizardActivityByWorkspace.get(workspace.id);
    if (localActivity) {
      if (localActivity.kind === 'import-task') {
        if (workspace.scm_import_task_id === localActivity.taskId && !isScmImportPhaseInProgress(workspace.scm_import_task_phase)) {
          scmWizardActivityByWorkspace.delete(workspace.id);
          return null;
        }
        return localActivity;
      }
      return localActivity;
    }

    const storedActivity = readStoredWorkspaceScmWizardActivity(workspace.id);
    if (storedActivity) {
      if (storedActivity.kind === 'import-task') {
        scmWizardActivityByWorkspace.delete(workspace.id);
        try {
          window.sessionStorage.removeItem(workspaceScmWizardActivityStorageKey(workspace.id));
        } catch {
          // Ignore storage failures.
        }
        return null;
      }
      scmWizardActivityByWorkspace.set(workspace.id, storedActivity);
      return storedActivity;
    }
    return null;
  }, [version, workspace?.id, workspace?.scm_import_task_id, workspace?.scm_import_task_phase]);
}

interface WorkspaceScmWizardProps {
  workspace: UserSpaceWorkspace;
  onClose: () => void;
  onSyncComplete: (response: UserSpaceWorkspaceScmSyncResponse) => Promise<void> | void;
  onAskAgent?: (prompt: string) => Promise<void> | void;
  onWorkspaceChanged?: () => Promise<void> | void;
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

const SCM_SYNC_INTERVAL_MIN = 300;
const SCM_SYNC_INTERVAL_MAX = 2592000;
const SCM_SYNC_INTERVAL_SCALE = Math.log(SCM_SYNC_INTERVAL_MAX / SCM_SYNC_INTERVAL_MIN);

function syncIntervalToSlider(seconds: number): number {
  if (seconds >= SCM_SYNC_INTERVAL_MAX) return 100;
  if (seconds <= SCM_SYNC_INTERVAL_MIN) return 0;
  return Math.max(0, Math.min(100, (Math.log(seconds / SCM_SYNC_INTERVAL_MIN) / SCM_SYNC_INTERVAL_SCALE) * 100));
}

function sliderToSyncInterval(slider: number): number {
  if (slider >= 100) return SCM_SYNC_INTERVAL_MAX;
  if (slider <= 0) return SCM_SYNC_INTERVAL_MIN;
  return Math.round(SCM_SYNC_INTERVAL_MIN * Math.exp((slider / 100) * SCM_SYNC_INTERVAL_SCALE));
}

function formatSyncInterval(seconds: number): string {
  if (seconds < 3600) {
    const minutes = Math.max(1, Math.round(seconds / 60));
    return `${minutes}m`;
  }
  if (seconds < 86400) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  }
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  if (days >= 7 && hours === 0) {
    const weeks = Math.floor(days / 7);
    const remDays = days % 7;
    return remDays > 0 ? `${weeks}w ${remDays}d` : `${weeks}w`;
  }
  return hours > 0 ? `${days}d ${hours}h` : `${days}d`;
}

function isArchiveExportTaskTerminal(phase: UserSpaceWorkspaceArchiveExportTask['phase']): boolean {
  return phase === 'completed' || phase === 'failed';
}

function isArchiveImportTaskTerminal(phase: UserSpaceWorkspaceArchiveImportTask['phase']): boolean {
  return phase === 'completed' || phase === 'failed';
}

function isArchiveExportPhaseInProgress(phase: UserSpaceWorkspaceArchiveExportTask['phase'] | null | undefined): boolean {
  return Boolean(phase && !isArchiveExportTaskTerminal(phase));
}

function isArchiveImportPhaseInProgress(phase: UserSpaceWorkspaceArchiveImportTask['phase'] | null | undefined): boolean {
  return Boolean(phase && !isArchiveImportTaskTerminal(phase));
}

function isSqliteImportTaskTerminal(phase: UserSpaceWorkspaceSqliteImportTask['phase']): boolean {
  return phase === 'completed' || phase === 'failed';
}

function isSqliteImportTaskActive(task: UserSpaceWorkspaceSqliteImportTask | null): boolean {
  return Boolean(task && !isSqliteImportTaskTerminal(task.phase));
}

function formatSqliteImportPhase(phase: UserSpaceWorkspaceSqliteImportTask['phase']): string {
  const labels: Record<UserSpaceWorkspaceSqliteImportTask['phase'], string> = {
    queued: 'Queued',
    staging_upload: 'Staging upload',
    waiting_for_slot: 'Waiting for slot',
    restoring_dump: 'Restoring PostgreSQL dump',
    transpiling_sql: 'Transpiling SQL',
    importing_sql: 'Importing SQL',
    finalizing_sqlite: 'Finalizing SQLite',
    completed: 'Completed',
    failed: 'Failed',
  };
  return labels[phase];
}

function getSqliteImportProgressPercent(task: UserSpaceWorkspaceSqliteImportTask): number {
  return Math.max(0, Math.min(100, Math.round(task.progress * 100)));
}

function getArchiveImportProgress(task: UserSpaceWorkspaceArchiveImportTask): { currentStep: number; totalSteps: number; percent: number } | null {
  if (isArchiveImportTaskTerminal(task.phase)) {
    return null;
  }
  const steps: UserSpaceWorkspaceArchiveImportTask['phase'][] = [
    'queued',
    'extracting_archive',
    'importing_files',
    'importing_metadata',
    'importing_snapshots',
    'importing_chats',
  ];
  const index = steps.indexOf(task.phase);
  const currentStep = index >= 0 ? index + 1 : 1;
  const totalSteps = steps.length;
  const percent = Math.min(100, Math.round((currentStep / totalSteps) * 100));
  return { currentStep, totalSteps, percent };
}

function formatArchiveTaskPhase(phase: string): string {
  return phase.replace(/_/g, ' ');
}

function formatArchiveSize(bytes: number | null | undefined): string | null {
  if (!bytes || bytes <= 0) return null;
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}

function isApiErrorWithStatus(error: unknown, status: number): error is { status: number } {
  return typeof error === 'object' && error !== null && 'status' in error && error.status === status;
}

function isScmImportTaskTerminal(phase: UserSpaceWorkspaceScmImportTaskPhase): boolean {
  return phase === 'preview_ready' || phase === 'completed' || phase === 'failed';
}

function shouldPollScmImportTask(task: UserSpaceWorkspaceScmImportTask | null): boolean {
  if (!task) return false;
  return !isScmImportTaskTerminal(task.phase) || (task.phase === 'preview_ready' && !task.preview);
}

function isScmImportPhaseInProgress(phase: UserSpaceWorkspaceScmImportTaskPhase | null | undefined): boolean {
  return Boolean(phase && ['queued', 'previewing', 'cloning', 'importing', 'backfilling'].includes(phase));
}

function formatScmImportTaskPhase(phase: UserSpaceWorkspaceScmImportTaskPhase): string {
  const labels: Record<UserSpaceWorkspaceScmImportTaskPhase, string> = {
    queued: 'Queued',
    previewing: 'Analyzing repository',
    preview_ready: 'Ready to review',
    cloning: 'Cloning repository',
    importing: 'Importing files',
    backfilling: 'Backfilling snapshot history',
    completed: 'Completed',
    failed: 'Failed',
  };
  return labels[phase];
}

function getScmImportTaskProgressPercent(phase: UserSpaceWorkspaceScmImportTaskPhase, progress?: number): number {
  if (phase === 'completed' || phase === 'preview_ready') return 100;
  if (phase === 'failed') return 0;
  const clamped = Math.max(0, Math.min(1, progress ?? 0));
  return Math.round(clamped * 100);
}

function buildArchiveExportTaskPlaceholder(workspace: UserSpaceWorkspace): UserSpaceWorkspaceArchiveExportTask | null {
  if (!workspace.archive_export_task_id) {
    return null;
  }
  return {
    task_id: workspace.archive_export_task_id,
    workspace_id: workspace.id,
    workspace_name: workspace.name,
    archive_format: 'zip',
    include_snapshots: false,
    include_chat_history: false,
    phase: workspace.archive_export_task_phase ?? 'queued',
    warnings: [],
    archive_file_name: null,
    archive_size_bytes: null,
    total_files: 0,
    processed_files: 0,
    total_bytes: 0,
    processed_bytes: 0,
    current_file_path: null,
    error: null,
    queued_at: workspace.updated_at,
    updated_at: workspace.updated_at,
  };
}

function buildArchiveImportTaskPlaceholder(workspace: UserSpaceWorkspace): UserSpaceWorkspaceArchiveImportTask | null {
  if (!workspace.archive_import_task_id) {
    return null;
  }
  return {
    task_id: workspace.archive_import_task_id,
    workspace_id: workspace.id,
    workspace_name: workspace.name,
    archive_format: null,
    include_snapshots: false,
    include_chat_history: false,
    phase: workspace.archive_import_task_phase ?? 'queued',
    warnings: [],
    imported_chat_count: 0,
    imported_snapshot_count: 0,
    error: null,
    queued_at: workspace.updated_at,
    updated_at: workspace.updated_at,
  };
}

function buildScmImportTaskPlaceholder(workspace: UserSpaceWorkspace): UserSpaceWorkspaceScmImportTask | null {
  if (!workspace.scm_import_task_id) {
    return null;
  }
  return {
    task_id: workspace.scm_import_task_id,
    workspace_id: workspace.id,
    workspace_name: workspace.name,
    git_url: workspace.scm?.git_url || '',
    git_branch: workspace.scm?.git_branch || 'main',
    phase: workspace.scm_import_task_phase ?? 'queued',
    progress: 0,
    error: null,
    scm: null,
    suggested_setup_prompt: null,
    remote_commit_hash: null,
    summary: null,
    queued_at: workspace.updated_at,
    updated_at: workspace.updated_at,
  };
}

export function WorkspaceScmWizard({ workspace, onClose, onSyncComplete, onAskAgent, onWorkspaceChanged }: WorkspaceScmWizardProps) {
  const initialScm = workspace.scm;
  const scmWizardActivity = useWorkspaceScmWizardActivity(workspace);
  const hasActiveArchiveExportTask = Boolean(workspace.archive_export_task_id) && isArchiveExportPhaseInProgress(workspace.archive_export_task_phase);
  const hasActiveArchiveImportTask = Boolean(workspace.archive_import_task_id) && isArchiveImportPhaseInProgress(workspace.archive_import_task_phase);
  const hasActiveScmImportTask = Boolean(
    workspace.scm_import_task_id && isScmImportPhaseInProgress(workspace.scm_import_task_phase)
    || workspace.scm_import_task_id && workspace.scm_import_task_phase === 'preview_ready'
    || scmWizardActivity?.kind === 'import-task'
    || scmWizardActivity?.kind === 'preview'
  );
  const shouldOpenArchiveByDefault = hasActiveArchiveExportTask || hasActiveArchiveImportTask;
  const defaultArchiveMode: ArchiveMode = hasActiveArchiveImportTask ? 'import' : 'export';
  const [toasts, toast] = useToast();
  const [activeTab, setActiveTab] = useState<ModalTab>(hasActiveScmImportTask ? 'git-source' : 'archive');
  const [mode, setMode] = useState<WizardMode>('import');
  const [step, setStep] = useState<WizardStep>(hasActiveScmImportTask ? 'result' : 'input');
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
  const [sqlImportResult, setSqlImportResult] = useState<UserSpaceWorkspaceSqliteImportTask | null>(null);
  const [lastSetupPrompt, setLastSetupPrompt] = useState<string | null>(null);
  const [sqlImportMaxBytes, setSqlImportMaxBytes] = useState(SQLITE_IMPORT_DEFAULT_MAX_BYTES);
  const [sqlImportLimitLoaded, setSqlImportLimitLoaded] = useState(false);
  const [sqlDragOver, setSqlDragOver] = useState(false);
  const [archiveMode, setArchiveMode] = useState<ArchiveMode>(defaultArchiveMode);
  const [archiveStep, setArchiveStep] = useState<ArchiveStep>(shouldOpenArchiveByDefault ? 'configure' : 'choose');
  const [archiveFormat, setArchiveFormat] = useState<UserSpaceWorkspaceArchiveFormat>('zip');
  const [archiveIncludeSnapshots, setArchiveIncludeSnapshots] = useState(false);
  const [archiveIncludeChatHistory, setArchiveIncludeChatHistory] = useState(false);
  const [archiveFile, setArchiveFile] = useState<File | null>(null);
  const [archiveDragOver, setArchiveDragOver] = useState(false);
  const [archiveExportTask, setArchiveExportTask] = useState<UserSpaceWorkspaceArchiveExportTask | null>(null);
  const [archiveImportTask, setArchiveImportTask] = useState<UserSpaceWorkspaceArchiveImportTask | null>(null);
  const [scmImportTask, setScmImportTask] = useState<UserSpaceWorkspaceScmImportTask | null>(() => buildScmImportTaskPlaceholder(workspace));
  const onWorkspaceChangedRef = useRef(onWorkspaceChanged);
  const lastNotifiedScmImportTaskIdRef = useRef<string | null>(null);
  const [archiveExports, setArchiveExports] = useState<UserSpaceWorkspaceArchiveExportListItem[]>([]);
  const [archiveExportsScanned, setArchiveExportsScanned] = useState(false);
  const [deletingExportTaskId, setDeletingExportTaskId] = useState<string | null>(null);
  const [downloadedExportTaskIds, setDownloadedExportTaskIds] = useState<Set<string>>(new Set());
  const [loadingAction, setLoadingAction] = useState<'pull' | 'push' | 'overwrite' | 'sync' | 'preview' | 'execute' | 'save-settings' | 'disconnect' | 'clear-fields' | null>(null);
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const [scmState, setScmState] = useState<UserSpaceWorkspaceScmStatus | null>(null);
  const [autoPushEnabled, setAutoPushEnabled] = useState(initialScm?.auto_sync_policy === 'auto_push');
  const [autoPullEnabled, setAutoPullEnabled] = useState(Boolean(initialScm?.auto_pull_enabled));
  const [autoPushIntervalSeconds, setAutoPushIntervalSeconds] = useState(initialScm?.auto_push_interval_seconds ?? 3600);
  const [autoPullIntervalSeconds, setAutoPullIntervalSeconds] = useState(initialScm?.auto_pull_interval_seconds ?? 3600);
  const [autoPushStartMinute, setAutoPushStartMinute] = useState<number | null>(initialScm?.auto_push_start_minute ?? null);
  const [autoPushTimezone, setAutoPushTimezone] = useState<string | null>(initialScm?.auto_push_timezone ?? null);
  const [autoPullStartMinute, setAutoPullStartMinute] = useState<number | null>(initialScm?.auto_pull_start_minute ?? null);
  const [autoPullTimezone, setAutoPullTimezone] = useState<string | null>(initialScm?.auto_pull_timezone ?? null);
  const sqlFileInputRef = useRef<HTMLInputElement>(null);
  const archiveFileInputRef = useRef<HTMLInputElement>(null);
  const lastDownloadedArchiveTaskIdRef = useRef<string | null>(null);
  const lastNotifiedArchiveExportTaskIdRef = useRef<string | null>(null);
  const lastNotifiedArchiveImportTaskIdRef = useRef<string | null>(null);
  const activeScm = scmState ?? result?.scm ?? initialScm ?? null;
  const hasConfiguredRemote = Boolean(activeScm?.connected || activeScm?.git_url);
  const setupPrompt = lastSetupPrompt ?? activeScm?.last_setup_prompt ?? null;
  const clearStatus = useCallback(() => setStatus(EMPTY_STATUS), []);

  const resetGitSourceState = useCallback((nextScm?: UserSpaceWorkspaceScmStatus | null) => {
    setPreview(null);
    setResult(null);
    setGitUrl(nextScm?.git_url || '');
    setGitBranch(nextScm?.git_branch || 'main');
    setGitToken('');
    setRepoVisibility(nextScm?.repo_visibility || null);
    setHasStoredToken(Boolean(nextScm?.has_stored_token));
    setStoredTokenValid(Boolean(nextScm?.has_stored_token));
    setBranches([]);
    setBranchError(null);
    setShowMoreMenu(false);
    setLoadingAction(null);
    setStep('input');
    setMode(nextScm?.connected || nextScm?.git_url ? 'import' : 'import');
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.getUserSpacePreviewSettings()
      .then((settings) => {
        if (cancelled) return;
        setSqlImportMaxBytes(settings.userspace_sqlite_import_max_bytes || SQLITE_IMPORT_DEFAULT_MAX_BYTES);
        setSqlImportLimitLoaded(true);
      })
      .catch(() => {
        if (!cancelled) setSqlImportLimitLoaded(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (archiveExportTask) {
      return;
    }
    const nextTask = buildArchiveExportTaskPlaceholder(workspace);
    if (nextTask) {
      setArchiveExportTask(nextTask);
    }
  }, [archiveExportTask, workspace.archive_export_task_id, workspace.archive_export_task_phase, workspace.id, workspace.name, workspace.updated_at]);

  useEffect(() => {
    if (archiveImportTask) {
      return;
    }
    const nextTask = buildArchiveImportTaskPlaceholder(workspace);
    if (nextTask) {
      setArchiveImportTask(nextTask);
    }
  }, [archiveImportTask, workspace.archive_import_task_id, workspace.archive_import_task_phase, workspace.id, workspace.name, workspace.updated_at]);

  useEffect(() => {
    if (scmImportTask) {
      return;
    }
    const nextTask = buildScmImportTaskPlaceholder(workspace);
    if (nextTask) {
      setScmImportTask(nextTask);
    }
  }, [scmImportTask, workspace.scm_import_task_id, workspace.scm_import_task_phase, workspace.id, workspace.name, workspace.updated_at]);

  useEffect(() => {
    setArchiveExports([]);
    setArchiveExportsScanned(false);
    setDownloadedExportTaskIds(new Set());
    setScmState(null);
    setPreview(null);
    setResult(null);
    setLastSetupPrompt(null);
    setScmImportTask(buildScmImportTaskPlaceholder(workspace));
    setGitUrl(workspace.scm?.git_url || '');
    setGitBranch(workspace.scm?.git_branch || 'main');
    setGitToken('');
    setRepoVisibility(workspace.scm?.repo_visibility || null);
    setHasStoredToken(Boolean(workspace.scm?.has_stored_token));
    setStoredTokenValid(Boolean(workspace.scm?.has_stored_token));
    setBranches([]);
    setBranchError(null);
    setShowMoreMenu(false);
  }, [workspace.id]);

  useEffect(() => {
    if (!hasActiveArchiveExportTask && !hasActiveArchiveImportTask) {
      return;
    }
    setActiveTab('archive');
    setArchiveStep('configure');
    setArchiveMode(hasActiveArchiveImportTask ? 'import' : 'export');
    setStep('input');
  }, [hasActiveArchiveExportTask, hasActiveArchiveImportTask, workspace.id]);

  useEffect(() => {
    if (!hasActiveScmImportTask) {
      return;
    }
    setActiveTab('git-source');
    if (scmWizardActivity?.kind === 'preview' && scmWizardActivity.status === 'ready' && scmWizardActivity.preview) {
      setPreview(scmWizardActivity.preview);
      setMode(scmWizardActivity.preview.direction === 'export' ? 'export' : 'import');
      setStep('review');
      setStatus({ type: null, message: '' });
      return;
    }
    if (scmWizardActivity?.kind === 'preview' && scmWizardActivity.status === 'failed') {
      setStep('input');
      setStatus({ type: 'error', message: scmWizardActivity.error || 'Failed to preview sync.' });
      return;
    }
    setStep('result');
  }, [hasActiveScmImportTask, scmWizardActivity, workspace.id]);

  const tokenRequired = useMemo(() => {
    if (mode === 'export') {
      return !hasStoredToken;
    }
    return repoVisibility === 'private' && !hasStoredToken;
  }, [hasStoredToken, mode, repoVisibility]);

  const shouldShowStatus = useMemo(() => {
    if (!status.type || !status.message) return false;
    if (activeTab === 'sql-import' && sqlImportResult && status.type !== 'error') return false;
    if (activeTab === 'git-source' && step === 'result' && scmImportTask && status.type !== 'error') return false;
    if (step === 'review' && preview && status.message === preview.summary) return false;
    if (step === 'result' && result && status.message === result.summary) return false;
    return true;
  }, [activeTab, preview, result, scmImportTask, sqlImportResult, status.message, status.type, step]);

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
    setAutoPushEnabled(activeScm?.auto_sync_policy === 'auto_push');
    setAutoPullEnabled(Boolean(activeScm?.auto_pull_enabled));
    setAutoPushIntervalSeconds(activeScm?.auto_push_interval_seconds ?? 3600);
    setAutoPullIntervalSeconds(activeScm?.auto_pull_interval_seconds ?? 3600);
    setAutoPushStartMinute(activeScm?.auto_push_start_minute ?? null);
    setAutoPushTimezone(activeScm?.auto_push_timezone ?? null);
    setAutoPullStartMinute(activeScm?.auto_pull_start_minute ?? null);
    setAutoPullTimezone(activeScm?.auto_pull_timezone ?? null);
  }, [activeScm?.auto_pull_enabled, activeScm?.auto_pull_interval_seconds, activeScm?.auto_pull_start_minute, activeScm?.auto_pull_timezone, activeScm?.auto_push_interval_seconds, activeScm?.auto_push_start_minute, activeScm?.auto_push_timezone, activeScm?.auto_sync_policy]);

  const hasPendingPatToken = useMemo(() => hasConfiguredRemote && gitToken.trim().length > 0, [gitToken, hasConfiguredRemote]);

  const connectedBranch = (activeScm?.git_branch || '').trim();
  const selectedBranchDiffers = hasConfiguredRemote
    && connectedBranch.length > 0
    && gitBranch.trim().length > 0
    && gitBranch.trim() !== connectedBranch;

  const hasRunningGitSourceTask = useMemo(() => {
    if (scmWizardActivity?.kind === 'preview' && scmWizardActivity.status === 'running') return true;
    if (scmWizardActivity?.kind === 'import-task') return true;
    return Boolean(scmImportTask && !isScmImportTaskTerminal(scmImportTask.phase));
  }, [scmImportTask, scmWizardActivity]);

  const hasDirtyUpstreamSyncSettings = useMemo(() => {
    if (!hasConfiguredRemote || activeScm?.remote_role !== 'upstream') {
      return false;
    }
    const savedAutoPushEnabled = activeScm.auto_sync_policy === 'auto_push';
    const savedAutoPullEnabled = Boolean(activeScm.auto_pull_enabled);
    const savedAutoPushInterval = activeScm.auto_push_interval_seconds ?? 3600;
    const savedAutoPullInterval = activeScm.auto_pull_interval_seconds ?? 3600;
    const savedAutoPushStartMinute = activeScm.auto_push_start_minute ?? null;
    const savedAutoPushTimezone = activeScm.auto_push_timezone ?? null;
    const savedAutoPullStartMinute = activeScm.auto_pull_start_minute ?? null;
    const savedAutoPullTimezone = activeScm.auto_pull_timezone ?? null;

    if (autoPushEnabled !== savedAutoPushEnabled) {
      return true;
    }
    if (autoPullEnabled !== savedAutoPullEnabled) {
      return true;
    }
    if (autoPushEnabled && autoPushIntervalSeconds !== savedAutoPushInterval) {
      return true;
    }
    if (autoPushEnabled && (autoPushStartMinute !== savedAutoPushStartMinute || autoPushTimezone !== savedAutoPushTimezone)) {
      return true;
    }
    if (autoPullEnabled && autoPullIntervalSeconds !== savedAutoPullInterval) {
      return true;
    }
    if (autoPullEnabled && (autoPullStartMinute !== savedAutoPullStartMinute || autoPullTimezone !== savedAutoPullTimezone)) {
      return true;
    }
    return false;
  }, [activeScm, autoPullEnabled, autoPullIntervalSeconds, autoPullStartMinute, autoPullTimezone, autoPushEnabled, autoPushIntervalSeconds, autoPushStartMinute, autoPushTimezone, hasConfiguredRemote]);

  const hasScmSettingsMutations = hasPendingPatToken || hasDirtyUpstreamSyncSettings;

  const applyScmSettingsPatch = useCallback(async (
    patch: UserSpaceWorkspaceScmSettingsRequest,
    summary: string,
    direction: 'import' | 'export',
  ) => {
    const resp = await api.updateUserSpaceWorkspaceScmSettings(workspace.id, patch);
    setScmState(resp.scm);
    await onSyncComplete({ workspace_id: workspace.id, direction, state: 'settings_updated', summary, scm: resp.scm });
  }, [onSyncComplete, workspace.id]);

  async function handleSaveScmSettings(): Promise<void> {
    if (!hasConfiguredRemote || !activeScm) {
      return;
    }

    const trimmedToken = gitToken.trim();
    if (!trimmedToken && !hasDirtyUpstreamSyncSettings) {
      return;
    }

    setIsLoading(true);
    setLoadingAction('save-settings');
    try {
      let savedTokenOnly = false;
      if (trimmedToken) {
        if (!activeScm.git_url) {
          throw new Error('Missing configured repository URL.');
        }
        const connResp = await api.updateUserSpaceWorkspaceScm(workspace.id, {
          git_url: activeScm.git_url,
          git_branch: activeScm.git_branch || gitBranch || 'main',
          git_token: trimmedToken,
          repo_visibility: activeScm.repo_visibility || undefined,
        });
        savedTokenOnly = true;
        setScmState(connResp.scm);
        setHasStoredToken(true);
        setStoredTokenValid(true);
        if (!hasDirtyUpstreamSyncSettings) {
          await onSyncComplete({
            workspace_id: workspace.id,
            direction: 'export',
            state: 'settings_updated',
            summary: 'Personal access token saved.',
            scm: connResp.scm,
          });
        }
      }

      if (hasDirtyUpstreamSyncSettings && activeScm.remote_role === 'upstream') {
        const nextAutoSyncPolicy = autoPushEnabled ? 'auto_push' : 'manual';
        const patch: UserSpaceWorkspaceScmSettingsRequest = {
          auto_sync_policy: nextAutoSyncPolicy,
          auto_pull_enabled: autoPullEnabled,
        };
        if (autoPushEnabled) {
          patch.auto_push_interval_seconds = autoPushIntervalSeconds;
          patch.auto_push_start_minute = autoPushStartMinute ?? defaultScheduleStartMinute();
          patch.auto_push_timezone = autoPushTimezone ?? defaultScheduleTimezone();
        } else {
          patch.auto_push_start_minute = null;
          patch.auto_push_timezone = null;
        }
        if (autoPullEnabled) {
          patch.auto_pull_interval_seconds = autoPullIntervalSeconds;
          patch.auto_pull_start_minute = autoPullStartMinute ?? defaultScheduleStartMinute();
          patch.auto_pull_timezone = autoPullTimezone ?? defaultScheduleTimezone();
        } else {
          patch.auto_pull_start_minute = null;
          patch.auto_pull_timezone = null;
        }
        if (nextAutoSyncPolicy === 'auto_push') {
          patch.clear_sync_paused = true;
        }
        await applyScmSettingsPatch(
          patch,
          savedTokenOnly ? 'SCM settings and token saved.' : 'SCM settings saved.',
          'export',
        );
      }

      setGitToken('');
      clearStatus();
      toast.success('SCM settings saved.');
    } catch (error) {
      clearStatus();
      toast.error(error instanceof Error ? error.message : 'Failed to save SCM settings.');
    } finally {
      setIsLoading(false);
      setLoadingAction(null);
    }
  }

  async function handleClearGitSourceFields(): Promise<void> {
    if (isLoading || hasRunningGitSourceTask) {
      return;
    }

    const shouldClearStoredToken = hasStoredToken
      || Boolean(preview && gitToken.trim())
      || Boolean(scmWizardActivity?.kind === 'preview' && scmWizardActivity.status === 'ready');

    setWorkspaceScmWizardActivity(workspace.id, null);
    setPreview(null);
    setResult(null);
    setScmImportTask(null);
    setGitUrl('');
    setGitBranch('main');
    setGitToken('');
    setRepoVisibility(null);
    setHasStoredToken(false);
    setStoredTokenValid(false);
    setBranches([]);
    setBranchError(null);
    setShowMoreMenu(false);
    setCreateRepoIfMissing(true);
    setCreateRepoPrivate(true);
    setCreateRepoDescription(workspace.description || '');
    setStep('input');
    setMode('import');
    clearStatus();

    if (hasConfiguredRemote) {
      toast.info('Use Disconnect Remote to clear the configured Git source.');
      return;
    }

    if (!shouldClearStoredToken) {
      toast.success('Git source fields cleared.');
      return;
    }

    setIsLoading(true);
    setLoadingAction('clear-fields');
    try {
      const resp = await api.updateUserSpaceWorkspaceScmSettings(workspace.id, { clear_git_token: true });
      setScmState(resp.scm);
      toast.success('Git source fields and stored token cleared.');
      await onWorkspaceChanged?.();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Fields cleared, but failed to clear stored token.');
    } finally {
      setIsLoading(false);
      setLoadingAction(null);
    }
  }

  useEffect(() => {
    if (hasConfiguredRemote) {
      const connectedUrl = (activeScm?.git_url || '').trim();
      if (!connectedUrl) {
        return;
      }

      let cancelled = false;
      const timer = window.setTimeout(async () => {
        try {
          const branchResult = await api.fetchUserSpaceWorkspaceScmBranches(workspace.id, {
            git_url: connectedUrl,
            git_token: gitToken.trim() || undefined,
          });
          if (cancelled) return;
          setBranches(branchResult.branches || []);
          setBranchError(branchResult.error || null);
        } catch (error) {
          if (cancelled) return;
          setBranchError(error instanceof Error ? error.message : 'Failed to load branches');
        }
      }, 400);

      return () => {
        cancelled = true;
        window.clearTimeout(timer);
      };
    }

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
  }, [activeScm?.git_url, gitToken, gitUrl, hasConfiguredRemote, mode, workspace.id]);

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
      const requestedDirection = explicitDirection ?? (hasConfiguredRemote ? 'import' : mode === 'export' ? 'export' : 'import');
      if (requestedDirection === 'import') {
        const task = await api.queueUserSpaceWorkspaceScmPreviewImport(workspace.id, payload);
        lastNotifiedScmImportTaskIdRef.current = null;
        setWorkspaceScmWizardActivity(workspace.id, null);
        setScmImportTask(task);
        setStep('result');
        await onWorkspaceChanged?.();
        return;
      }

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
      const message = error instanceof Error ? error.message : 'Failed to preview sync.';
      setWorkspaceScmWizardActivity(workspace.id, {
        kind: 'preview',
        workspaceId: workspace.id,
        status: 'failed',
        direction: explicitDirection ?? (mode === 'export' ? 'export' : 'import'),
        error: message,
      });
      setStatus({ type: 'error', message });
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
    setStatus({ type: null, message: '' });
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

      if (direction === 'import') {
        const task = await api.queueUserSpaceWorkspaceScmImport(workspace.id, payload as UserSpaceWorkspaceScmImportRequest);
        lastNotifiedScmImportTaskIdRef.current = null;
        setWorkspaceScmWizardActivity(workspace.id, {
          kind: 'import-task',
          workspaceId: workspace.id,
          status: 'running',
          taskId: task.task_id,
        });
        setScmImportTask(task);
        setStep('result');
        await onWorkspaceChanged?.();
      } else {
        const nextResult = await api.exportUserSpaceWorkspaceToScm(workspace.id, payload as UserSpaceWorkspaceScmExportRequest);
        setWorkspaceScmWizardActivity(workspace.id, null);
        setScmState(nextResult.scm);
        setResult(nextResult);
        setStatus({ type: 'success', message: nextResult.summary });
        setStep('result');
        await onSyncComplete(nextResult);
      }
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

  async function handleAskAgentFromScmTask(): Promise<void> {
    if (!scmImportTask?.suggested_setup_prompt || !onAskAgent) return;
    onClose();
    await onAskAgent(scmImportTask.suggested_setup_prompt);
  }

  async function handleDisconnectScm(): Promise<void> {
    if (!hasConfiguredRemote) {
      return;
    }

    setIsLoading(true);
    setLoadingAction('disconnect');
    setShowMoreMenu(false);
    clearStatus();
    try {
      const resp = await api.disconnectUserSpaceWorkspaceScm(workspace.id);
      setScmState(resp.scm);
      resetGitSourceState(resp.scm);
      toast.success('Remote disconnected.');
      await onWorkspaceChanged?.();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to disconnect remote.');
    } finally {
      setIsLoading(false);
      setLoadingAction(null);
    }
  }

  async function handleSqlImport(): Promise<void> {
    if (!sqlFile) {
      setStatus({ type: 'error', message: 'Please select a SQL dump file.' });
      return;
    }
    if (sqlImportLimitLoaded && sqlFile.size > sqlImportMaxBytes) {
      setStatus({
        type: 'error',
        message: `SQL dump exceeds the configured ${formatBytes(sqlImportMaxBytes)} size limit.`,
      });
      return;
    }
    setIsLoading(true);
    clearStatus();
    try {
      const formData = new FormData();
      formData.append('file', sqlFile);
      const importResult = await api.importSqlToWorkspaceSqlite(workspace.id, formData);
      setSqlImportResult(importResult);
      setStatus({
        type: 'info',
        message: 'SQL import queued.',
      });
      setStep('result');
    } catch (error) {
      setStatus({ type: 'error', message: error instanceof Error ? error.message : 'SQL import failed.' });
    } finally {
      setIsLoading(false);
    }
  }

  async function handleQueueArchiveExport(): Promise<void> {
    setIsLoading(true);
    try {
      const task = await api.queueUserSpaceWorkspaceArchiveExport(workspace.id, {
        archive_format: archiveFormat,
        include_snapshots: archiveIncludeSnapshots,
        include_chat_history: archiveIncludeChatHistory,
      });
      setArchiveExportTask(task);
      setArchiveImportTask(null);
      lastDownloadedArchiveTaskIdRef.current = null;
      lastNotifiedArchiveExportTaskIdRef.current = null;
      clearStatus();
      toast.success('Workspace archive export queued.');
      await onWorkspaceChanged?.();
    } catch (error) {
      clearStatus();
      toast.error(error instanceof Error ? error.message : 'Failed to queue workspace export.');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleQueueArchiveImport(): Promise<void> {
    if (!archiveFile) {
      clearStatus();
      toast.error('Select an archive file to import.');
      return;
    }
    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('archive_file', archiveFile);
      formData.append('include_snapshots', String(archiveIncludeSnapshots));
      formData.append('include_chat_history', String(archiveIncludeChatHistory));
      const task = await api.queueUserSpaceWorkspaceArchiveImport(workspace.id, formData);
      setArchiveImportTask(task);
      setArchiveExportTask(null);
      lastNotifiedArchiveImportTaskIdRef.current = null;
      clearStatus();
      toast.success('Workspace archive import queued.');
      await onWorkspaceChanged?.();
    } catch (error) {
      clearStatus();
      toast.error(error instanceof Error ? error.message : 'Failed to queue workspace import.');
    } finally {
      setIsLoading(false);
    }
  }

  async function handleDownloadArchive(taskId: string): Promise<void> {
    try {
      await api.downloadUserSpaceWorkspaceArchiveExportTask(taskId);
      lastDownloadedArchiveTaskIdRef.current = taskId;
      setDownloadedExportTaskIds(prev => {
        const next = new Set(prev);
        next.add(taskId);
        return next;
      });
      clearStatus();
      toast.success('Workspace archive downloaded.');
    } catch (error) {
      clearStatus();
      toast.error(error instanceof Error ? error.message : 'Failed to download workspace archive.');
    }
  }

  const loadArchiveExports = useCallback(async (options?: { force?: boolean }): Promise<void> => {
    if (archiveExportsScanned && !options?.force) {
      return;
    }
    try {
      const response = await api.listUserSpaceWorkspaceArchiveExports(workspace.id);
      setArchiveExports(response.exports);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to load workspace exports.');
    } finally {
      setArchiveExportsScanned(true);
    }
  }, [archiveExportsScanned, workspace.id, toast]);

  async function handleDeleteArchiveExport(taskId: string): Promise<void> {
    setDeletingExportTaskId(taskId);
    try {
      await api.deleteUserSpaceWorkspaceArchiveExportTask(taskId);
      setArchiveExports(prev => prev.filter(item => item.task_id !== taskId));
      if (archiveExportTask?.task_id === taskId) {
        setArchiveExportTask(null);
      }
      toast.success('Workspace archive deleted.');
      await onWorkspaceChanged?.();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to delete workspace archive.');
    } finally {
      setDeletingExportTaskId(null);
    }
  }

  useEffect(() => {
    if (activeTab !== 'archive' || archiveExportsScanned) return;
    void loadArchiveExports();
  }, [activeTab, archiveExportsScanned, loadArchiveExports]);

  useEffect(() => {
    if (!archiveExportTask || isArchiveExportTaskTerminal(archiveExportTask.phase)) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollTask = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const nextTask = await api.getUserSpaceWorkspaceArchiveExportTask(archiveExportTask.task_id);
        if (cancelled) return;
        setArchiveExportTask(nextTask);
        if (nextTask.phase === 'completed' && lastNotifiedArchiveExportTaskIdRef.current !== nextTask.task_id) {
          await loadArchiveExports({ force: true });
          toast.success(nextTask.warnings.length > 0 ? 'Archive export completed with warnings.' : 'Archive export completed.');
          lastNotifiedArchiveExportTaskIdRef.current = nextTask.task_id;
          if (onWorkspaceChanged) {
            await onWorkspaceChanged();
          }
        } else if (nextTask.phase === 'failed' && lastNotifiedArchiveExportTaskIdRef.current !== nextTask.task_id) {
          toast.error(nextTask.error?.trim() || 'Workspace archive export failed.');
          lastNotifiedArchiveExportTaskIdRef.current = nextTask.task_id;
          if (onWorkspaceChanged) {
            await onWorkspaceChanged();
          }
        }
      } catch (error) {
        if (!cancelled) {
          if (isApiErrorWithStatus(error, 404)) {
            setArchiveExportTask(null);
            await onWorkspaceChanged?.();
            return;
          }
          toast.error(error instanceof Error ? error.message : 'Failed to refresh export task.');
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollTask();
    const intervalId = window.setInterval(() => { void pollTask(); }, ARCHIVE_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [archiveExportTask, loadArchiveExports, onWorkspaceChanged, toast]);

  useEffect(() => {
    if (!archiveImportTask || isArchiveImportTaskTerminal(archiveImportTask.phase)) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollTask = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const nextTask = await api.getUserSpaceWorkspaceArchiveImportTask(archiveImportTask.task_id);
        if (cancelled) return;
        setArchiveImportTask(nextTask);
        if (nextTask.phase === 'completed' && lastNotifiedArchiveImportTaskIdRef.current !== nextTask.task_id) {
          toast.success(nextTask.warnings.length > 0 ? 'Archive import completed with warnings.' : 'Archive import completed.');
          lastNotifiedArchiveImportTaskIdRef.current = nextTask.task_id;
          if (onWorkspaceChanged) {
            await onWorkspaceChanged();
          }
        } else if (nextTask.phase === 'failed' && lastNotifiedArchiveImportTaskIdRef.current !== nextTask.task_id) {
          toast.error(nextTask.error?.trim() || 'Workspace archive import failed.');
          lastNotifiedArchiveImportTaskIdRef.current = nextTask.task_id;
        }
      } catch (error) {
        if (!cancelled) {
          if (isApiErrorWithStatus(error, 404)) {
            setArchiveImportTask(null);
            await onWorkspaceChanged?.();
            return;
          }
          toast.error(error instanceof Error ? error.message : 'Failed to refresh import task.');
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollTask();
    const intervalId = window.setInterval(() => { void pollTask(); }, ARCHIVE_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [archiveImportTask, onWorkspaceChanged, toast]);

  useEffect(() => {
    if (!scmImportTask || !shouldPollScmImportTask(scmImportTask)) {
      return;
    }
    const taskId = scmImportTask.task_id;

    let cancelled = false;
    let pollInFlight = false;

    const pollTask = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const nextTask = await api.getUserSpaceWorkspaceScmImportTask(taskId);
        if (cancelled) return;
        let displayTask = nextTask;
        if (nextTask.phase === 'preview_ready' && !nextTask.preview) {
          displayTask = await api.getUserSpaceWorkspaceScmImportTask(nextTask.task_id);
          if (cancelled) return;
        }
        setScmImportTask(displayTask);
        if (displayTask.phase === 'preview_ready' && displayTask.preview) {
          setWorkspaceScmWizardActivity(workspace.id, {
            kind: 'preview',
            workspaceId: workspace.id,
            status: 'ready',
            direction: displayTask.preview.direction === 'export' ? 'export' : 'import',
            preview: displayTask.preview,
          });
          setPreview(displayTask.preview);
          setMode(displayTask.preview.direction === 'export' ? 'export' : 'import');
          setStep('review');
          setStatus({ type: null, message: '' });
          await onWorkspaceChanged?.();
        } else if (displayTask.phase === 'completed' && lastNotifiedScmImportTaskIdRef.current !== displayTask.task_id) {
          setWorkspaceScmWizardActivity(workspace.id, null);
          lastNotifiedScmImportTaskIdRef.current = displayTask.task_id;
          if (displayTask.suggested_setup_prompt) {
            setLastSetupPrompt(displayTask.suggested_setup_prompt);
          }
          if (displayTask.scm) {
            setScmState(displayTask.scm);
          }
          const syncResponse: UserSpaceWorkspaceScmSyncResponse = {
            workspace_id: workspace.id,
            direction: 'import',
            state: 'success',
            summary: displayTask.summary || 'Import completed.',
            scm: displayTask.scm || workspace.scm || {} as UserSpaceWorkspaceScmStatus,
            remote_commit_hash: displayTask.remote_commit_hash || null,
            suggested_setup_prompt: displayTask.suggested_setup_prompt || null,
          };
          await onSyncComplete(syncResponse);
          await onWorkspaceChanged?.();
        } else if (displayTask.phase === 'failed' && lastNotifiedScmImportTaskIdRef.current !== displayTask.task_id) {
          setWorkspaceScmWizardActivity(workspace.id, null);
          toast.error(displayTask.error?.trim() || 'SCM import failed.');
          lastNotifiedScmImportTaskIdRef.current = displayTask.task_id;
          await onWorkspaceChanged?.();
        }
      } catch (error) {
        if (!cancelled) {
          if (isApiErrorWithStatus(error, 404)) {
            setScmImportTask(null);
            await onWorkspaceChanged?.();
            return;
          }
          toast.error(error instanceof Error ? error.message : 'Failed to refresh import task.');
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollTask();
    const intervalId = window.setInterval(() => { void pollTask(); }, SCM_IMPORT_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [scmImportTask, onSyncComplete, onWorkspaceChanged, toast, workspace.id, workspace.scm]);

  useEffect(() => {
    onWorkspaceChangedRef.current = onWorkspaceChanged;
  }, [onWorkspaceChanged]);

  useEffect(() => {
    if (activeTab !== 'sql-import' || sqlImportResult) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const task = await api.getLatestUserSpaceWorkspaceSqliteImportTask(workspace.id);
        if (cancelled || !task) return;
        setSqlImportResult(task);
        setMode('sql-import');
        setStep('result');
      } catch {
        // Best-effort recovery only.
      }
    })();
    return () => { cancelled = true; };
  }, [activeTab, sqlImportResult, workspace.id]);

  const activeSqlImportTaskId = sqlImportResult && !isSqliteImportTaskTerminal(sqlImportResult.phase)
    ? sqlImportResult.task_id
    : null;

  useEffect(() => {
    if (!activeSqlImportTaskId) {
      return;
    }
    const taskId = activeSqlImportTaskId;
    let cancelled = false;
    let pollInFlight = false;
    let intervalId: number | null = null;
    async function pollTask(): Promise<void> {
      if (pollInFlight) {
        return;
      }
      pollInFlight = true;
      try {
        const nextTask = await api.getUserSpaceWorkspaceSqliteImportTask(taskId);
        if (cancelled) return;
        setSqlImportResult(nextTask);
        if (nextTask.phase === 'completed') {
          if (intervalId !== null) window.clearInterval(intervalId);
          setStatus({ type: 'success', message: nextTask.message || 'SQL import completed.' });
          await onWorkspaceChangedRef.current?.();
        } else if (nextTask.phase === 'failed') {
          if (intervalId !== null) window.clearInterval(intervalId);
          setStatus({ type: 'error', message: nextTask.error || nextTask.message || 'SQL import failed.' });
        }
      } catch (error) {
        if (!cancelled) {
          setStatus({ type: 'error', message: error instanceof Error ? error.message : 'SQL import status unavailable.' });
        }
      } finally {
        pollInFlight = false;
      }
    }
    void pollTask();
    intervalId = window.setInterval(() => { void pollTask(); }, SQLITE_IMPORT_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      if (intervalId !== null) window.clearInterval(intervalId);
    };
  }, [activeSqlImportTaskId]);

  const sqliteEnabled = workspace.sqlite_persistence_mode === 'include';
  const SQL_ACCEPT = '.sql,.dump,.pg,.pgsql,.mysql';
  const ARCHIVE_ACCEPT = [
    '.zip',
    '.tar.gz',
    '.tgz',
    '.tar',
    '.gz',
    'application/zip',
    'application/x-zip-compressed',
    'application/gzip',
    'application/x-gzip',
  ].join(',');

  function handleTabSwitch(tab: ModalTab) {
    if (isLoading) return;
    setActiveTab(tab);
    if (tab === 'sql-import') {
      setMode('sql-import');
      setStep('input');
      setSqlImportResult(null);
      setStatus({ type: null, message: '' });
    } else if (tab === 'archive') {
      setStep('input');
      setArchiveStep('choose');
      setStatus({ type: null, message: '' });
    } else {
      if (scmWizardActivity?.kind === 'preview') {
        setMode(scmWizardActivity.direction === 'export' ? 'export' : 'import');
        if (scmWizardActivity.status === 'ready' && scmWizardActivity.preview) {
          setPreview(scmWizardActivity.preview);
          setStep('review');
          setStatus({ type: null, message: '' });
          return;
        }
        if (scmWizardActivity.status === 'running') {
          setStep('result');
          setStatus({ type: null, message: '' });
          return;
        }
      }
      if (scmImportTask && !isScmImportTaskTerminal(scmImportTask.phase)) {
        setMode('import');
        setStep('result');
        setStatus({ type: null, message: '' });
        return;
      }
      setMode(hasConfiguredRemote ? 'import' : mode === 'sql-import' ? 'import' : mode);
      setStep('input');
      setStatus({ type: null, message: '' });
    }
  }

  const scmSummaryItems = activeScm
    ? [
      {
        label: 'Remote',
        value: activeScm.remote_role === 'upstream' ? 'Upstream' : 'Connected',
      },
      {
        label: 'Status',
        value: activeScm.sync_paused
          ? 'Sync paused'
          : activeScm.last_sync_message || activeScm.last_sync_status || 'Connected',
      },
      ...(formatSyncTimestamp(activeScm.last_sync_at)
        ? [{
          label: `Last ${formatSyncDirection(activeScm.last_sync_direction).toLowerCase()}`,
          value: formatSyncTimestamp(activeScm.last_sync_at) || 'Unknown',
        }]
        : []),
    ]
    : [];

  const upstreamSyncRows = activeScm?.remote_role === 'upstream'
    ? [
      {
        key: 'push' as const,
        label: 'Auto-push',
        description: 'Commit and push local workspace changes on a schedule.',
        enabled: autoPushEnabled,
        setEnabled: setAutoPushEnabled,
        interval: autoPushIntervalSeconds,
        setInterval: setAutoPushIntervalSeconds,
        startMinute: autoPushStartMinute,
        setStartMinute: setAutoPushStartMinute,
        timezone: autoPushTimezone,
        setTimezone: setAutoPushTimezone,
      },
      {
        key: 'pull' as const,
        label: 'Auto-pull',
        description: 'Check the remote branch and import changes on a schedule.',
        enabled: autoPullEnabled,
        setEnabled: setAutoPullEnabled,
        interval: autoPullIntervalSeconds,
        setInterval: setAutoPullIntervalSeconds,
        startMinute: autoPullStartMinute,
        setStartMinute: setAutoPullStartMinute,
        timezone: autoPullTimezone,
        setTimezone: setAutoPullTimezone,
      },
    ]
    : [];
  const enabledUpstreamSyncCount = upstreamSyncRows.filter((row) => row.enabled).length;
  const useSingleEnabledSyncLayout = enabledUpstreamSyncCount === 1;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-large" onClick={(event) => event.stopPropagation()}>
        <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
        <div className="modal-header" style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
          <div style={{ display: 'flex', gap: 0, flex: 1 }}>
            <button
              type="button"
              style={{
                background: 'transparent',
                border: 'none',
                borderBottom: activeTab === 'archive' ? '2px solid var(--color-accent)' : '2px solid transparent',
                padding: '8px 16px',
                cursor: isLoading ? 'default' : 'pointer',
                color: activeTab === 'archive' ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                fontSize: 14,
                fontWeight: 600,
                transition: 'color 0.15s, border-color 0.15s',
              }}
              onClick={() => handleTabSwitch('archive')}
              disabled={isLoading}
            >
              Backup/Restore
            </button>
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

          {activeTab === 'archive' && archiveStep === 'choose' && (
            <div style={{ display: 'grid', gap: 20 }}>
              <p className="userspace-muted" style={{ fontSize: 13, margin: 0 }}>
                Export packages this workspace's files and metadata as a portable archive. Import replaces this workspace with an archive exported from another server.
              </p>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <button
                  type="button"
                  style={{
                    display: 'grid', gap: 10, padding: '24px 20px',
                    textAlign: 'center', cursor: 'pointer', borderRadius: 10,
                    border: '1px solid var(--color-border)', background: 'var(--color-bg-tertiary)',
                    color: 'inherit', transition: 'border-color 0.15s, background 0.15s',
                  }}
                  onClick={() => { setArchiveMode('export'); setArchiveStep('configure'); setStatus({ type: null, message: '' }); }}
                >
                  <ArrowUpToLine size={32} style={{ margin: '0 auto', opacity: 0.75 }} />
                  <div style={{ fontWeight: 700, fontSize: 15 }}>Export Workspace</div>
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    Download workspace files and metadata as a portable ZIP or tar.gz
                  </div>
                </button>
                <button
                  type="button"
                  style={{
                    display: 'grid', gap: 10, padding: '24px 20px',
                    textAlign: 'center', cursor: 'pointer', borderRadius: 10,
                    border: '1px solid var(--color-border)', background: 'var(--color-bg-tertiary)',
                    color: 'inherit', transition: 'border-color 0.15s, background 0.15s',
                  }}
                  onClick={() => { setArchiveMode('import'); setArchiveStep('configure'); setStatus({ type: null, message: '' }); }}
                >
                  <ArrowDownToLine size={32} style={{ margin: '0 auto', opacity: 0.75 }} />
                  <div style={{ fontWeight: 700, fontSize: 15 }}>Import Workspace</div>
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    Restore workspace files and metadata from an exported archive
                  </div>
                </button>
              </div>
            </div>
          )}

          {activeTab === 'archive' && archiveStep === 'configure' && (
            <div style={{ display: 'grid', gap: 16 }}>
              {archiveMode === 'export' && (
                <div style={{ display: 'grid', gap: 14 }}>
                  <div style={{ display: 'grid', gap: 8 }}>
                    <strong style={{ fontSize: 13 }}>Archive Format</strong>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button className={`btn btn-sm ${archiveFormat === 'zip' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setArchiveFormat('zip')} disabled={isLoading}>ZIP</button>
                      <button className={`btn btn-sm ${archiveFormat === 'tar.gz' ? 'btn-primary' : 'btn-secondary'}`} onClick={() => setArchiveFormat('tar.gz')} disabled={isLoading}>tar.gz</button>
                    </div>
                  </div>
                  <div style={{ display: 'grid', gap: 10, padding: 14, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                    <strong style={{ fontSize: 13 }}>Optional Content</strong>
                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                      <input type="checkbox" style={{ marginTop: 2 }} checked={archiveIncludeSnapshots} onChange={(event) => setArchiveIncludeSnapshots(event.target.checked)} disabled={isLoading} />
                      <div>
                        <div style={{ fontWeight: 500 }}>Snapshot history</div>
                        <div className="userspace-muted" style={{ fontSize: 12, marginTop: 2 }}>Included only when the workspace has a clean, committed snapshot timeline</div>
                      </div>
                    </label>
                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                      <input type="checkbox" style={{ marginTop: 2 }} checked={archiveIncludeChatHistory} onChange={(event) => setArchiveIncludeChatHistory(event.target.checked)} disabled={isLoading} />
                      <div style={{ fontWeight: 500 }}>Chat history</div>
                    </label>
                  </div>
                  <p className="userspace-muted" style={{ fontSize: 12, margin: 0 }}>
                    Secret values are not included. Env vars and mount configuration are carried as placeholders.
                  </p>
                </div>
              )}

              {archiveMode === 'import' && (
                <div style={{ display: 'grid', gap: 14 }}>
                  <div
                    style={{
                      padding: 28,
                      border: `2px dashed ${archiveDragOver ? 'var(--color-accent)' : 'var(--color-border)'}`,
                      borderRadius: 8,
                      textAlign: 'center',
                      cursor: isLoading ? 'default' : 'pointer',
                      background: archiveDragOver ? 'rgba(var(--color-primary-rgb, 59, 130, 246), 0.05)' : 'transparent',
                      transition: 'border-color 0.15s, background 0.15s',
                    }}
                    onClick={() => !isLoading && archiveFileInputRef.current?.click()}
                    onDragOver={(event) => { event.preventDefault(); setArchiveDragOver(true); }}
                    onDragLeave={() => setArchiveDragOver(false)}
                    onDrop={(event) => {
                      event.preventDefault();
                      setArchiveDragOver(false);
                      const droppedFile = event.dataTransfer.files[0];
                      if (droppedFile) {
                        setArchiveFile(droppedFile);
                        setStatus({ type: null, message: '' });
                      }
                    }}
                  >
                    <input
                      ref={archiveFileInputRef}
                      type="file"
                      accept={ARCHIVE_ACCEPT}
                      style={{ display: 'none' }}
                      onChange={(event) => {
                        const selectedFile = event.target.files?.[0] || null;
                        setArchiveFile(selectedFile);
                        setStatus({ type: null, message: '' });
                      }}
                      disabled={isLoading}
                    />
                    {archiveFile ? (
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                        <Check size={16} style={{ color: 'var(--color-success, #2b7a2b)' }} />
                        <span>{archiveFile.name}</span>
                        <span className="userspace-muted" style={{ fontSize: 12 }}>
                          ({(archiveFile.size / 1024).toFixed(1)} KB)
                        </span>
                        <button
                          className="btn btn-sm btn-secondary"
                          style={{ marginLeft: 8, padding: '2px 6px' }}
                          onClick={(event) => { event.stopPropagation(); setArchiveFile(null); }}
                          disabled={isLoading}
                        >
                          <X size={12} />
                        </button>
                      </div>
                    ) : (
                      <div>
                        <Upload size={28} style={{ marginBottom: 10, opacity: 0.45 }} />
                        <div style={{ fontWeight: 500 }}>Drop a workspace archive here or click to browse</div>
                        <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                          Accepts .zip, .tar.gz, .tgz
                        </div>
                      </div>
                    )}
                  </div>

                  <div style={{ display: 'grid', gap: 10, padding: 14, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                    <strong style={{ fontSize: 13 }}>Restore from Archive</strong>
                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                      <input type="checkbox" style={{ marginTop: 2 }} checked={archiveIncludeSnapshots} onChange={(event) => setArchiveIncludeSnapshots(event.target.checked)} disabled={isLoading} />
                      <div>
                        <div style={{ fontWeight: 500 }}>Snapshot history</div>
                        <div className="userspace-muted" style={{ fontSize: 12, marginTop: 2 }}>Restored only when the archive contains snapshot data and this workspace has no existing timeline</div>
                      </div>
                    </label>
                    <label style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
                      <input type="checkbox" style={{ marginTop: 2 }} checked={archiveIncludeChatHistory} onChange={(event) => setArchiveIncludeChatHistory(event.target.checked)} disabled={isLoading} />
                      <div style={{ fontWeight: 500 }}>Chat history</div>
                    </label>
                  </div>

                  <p className="userspace-muted" style={{ fontSize: 12, margin: 0 }}>
                    Import replaces workspace files and metadata. Existing env var secret values are preserved when keys match.
                  </p>
                </div>
              )}

              {(archiveExportTask || archiveImportTask) && (
                <div style={{ display: 'grid', gap: 10, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  {archiveExportTask && (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <strong>Export task</strong>
                        <span className="userspace-status-pill userspace-status-pill-info">{formatArchiveTaskPhase(archiveExportTask.phase)}</span>
                        {archiveExportTask.archive_size_bytes ? (
                          <span className="userspace-muted" style={{ fontSize: 12 }}>{formatArchiveSize(archiveExportTask.archive_size_bytes)}</span>
                        ) : null}
                      </div>
                      {!isArchiveExportTaskTerminal(archiveExportTask.phase) && archiveExportTask.total_files > 0 && (
                        <div style={{ display: 'grid', gap: 4 }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }} className="userspace-muted">
                            <span>
                              {archiveExportTask.processed_files} / {archiveExportTask.total_files} files
                              {archiveExportTask.total_bytes > 0 ? ` · ${formatArchiveSize(archiveExportTask.processed_bytes)} / ${formatArchiveSize(archiveExportTask.total_bytes)}` : ''}
                            </span>
                            <span>
                              {Math.min(100, Math.round((archiveExportTask.processed_files / Math.max(1, archiveExportTask.total_files)) * 100))}%
                            </span>
                          </div>
                          <div style={{ width: '100%', height: 6, background: 'var(--color-bg-tertiary)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{
                              width: `${Math.min(100, Math.round((archiveExportTask.processed_files / Math.max(1, archiveExportTask.total_files)) * 100))}%`,
                              height: '100%',
                              background: 'var(--color-accent, #3b82f6)',
                              transition: 'width 0.2s linear',
                            }} />
                          </div>
                          {archiveExportTask.current_file_path && (
                            <div className="userspace-muted" style={{ fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {archiveExportTask.current_file_path}
                            </div>
                          )}
                        </div>
                      )}
                      {archiveExportTask.warnings.length > 0 && (
                        <div style={{ display: 'grid', gap: 4 }}>
                          {archiveExportTask.warnings.map((warning) => (
                            <div key={warning} className="userspace-muted" style={{ fontSize: 12 }}>{warning}</div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                  {archiveImportTask && (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                        <strong>Import task</strong>
                        <span className="userspace-status-pill userspace-status-pill-info">{formatArchiveTaskPhase(archiveImportTask.phase)}</span>
                        {(archiveImportTask.imported_snapshot_count > 0 || archiveImportTask.imported_chat_count > 0) && (
                          <span className="userspace-muted" style={{ fontSize: 12 }}>
                            {archiveImportTask.imported_snapshot_count} snapshots, {archiveImportTask.imported_chat_count} chats
                          </span>
                        )}
                      </div>
                      {(() => {
                        const progress = getArchiveImportProgress(archiveImportTask);
                        if (!progress) {
                          return null;
                        }
                        return (
                          <div style={{ display: 'grid', gap: 4 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }} className="userspace-muted">
                              <span>
                                Step {progress.currentStep} / {progress.totalSteps}
                              </span>
                              <span>{progress.percent}%</span>
                            </div>
                            <div style={{ width: '100%', height: 6, background: 'var(--color-bg-tertiary)', borderRadius: 3, overflow: 'hidden' }}>
                              <div style={{
                                width: `${progress.percent}%`,
                                height: '100%',
                                background: 'var(--color-accent, #3b82f6)',
                                transition: 'width 0.2s linear',
                              }} />
                            </div>
                          </div>
                        );
                      })()}
                      {archiveImportTask.warnings.length > 0 && (
                        <div style={{ display: 'grid', gap: 4 }}>
                          {archiveImportTask.warnings.map((warning) => (
                            <div key={warning} className="userspace-muted" style={{ fontSize: 12 }}>{warning}</div>
                          ))}
                        </div>
                      )}
                    </>
                  )}
                </div>
              )}

              {archiveMode === 'export' && archiveExports.length > 0 && (
                <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                    <strong style={{ fontSize: 13 }}>Previous exports</strong>
                  </div>
                  <div style={{ display: 'grid', gap: 6 }}>
                    {archiveExports.map((item) => {
                      const hasDownloaded = downloadedExportTaskIds.has(item.task_id) || lastDownloadedArchiveTaskIdRef.current === item.task_id;
                      const isDeleting = deletingExportTaskId === item.task_id;
                      return (
                        <div
                          key={item.task_id}
                          style={{
                            display: 'grid',
                            gridTemplateColumns: '1fr auto auto',
                            gap: 8,
                            alignItems: 'center',
                            padding: '8px 10px',
                            border: '1px solid var(--color-border-subtle)',
                            borderRadius: 6,
                            background: 'var(--color-bg-tertiary)',
                          }}
                        >
                          <div style={{ display: 'grid', gap: 2, minWidth: 0 }}>
                            <div style={{ fontSize: 13, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{item.archive_file_name}</div>
                            <div className="userspace-muted" style={{ fontSize: 11, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                              <span>{formatArchiveSize(item.archive_size_bytes)}</span>
                              <span>{new Date(item.created_at).toLocaleString()}</span>
                              {item.include_snapshots && <span>+ snapshots</span>}
                              {item.include_chat_history && <span>+ chats</span>}
                            </div>
                          </div>
                          <button
                            className="btn btn-sm btn-secondary"
                            onClick={() => void handleDownloadArchive(item.task_id)}
                            disabled={isDeleting}
                            title="Download archive"
                          >
                            <ArrowDownToLine size={12} /> {hasDownloaded ? 'Downloaded' : 'Download'}
                          </button>
                          <DeleteConfirmButton
                            onDelete={() => handleDeleteArchiveExport(item.task_id)}
                            disabled={isDeleting}
                            deleting={isDeleting}
                            className="btn btn-sm btn-danger"
                            title="Delete archive"
                            buttonText="Delete"
                          />
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && (
            <div style={{ display: 'grid', gap: 14 }}>
              {hasConfiguredRemote && activeScm && (
                <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <Link2 size={14} style={{ flexShrink: 0 }} />
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

                  <div style={{ display: 'grid', gap: 12 }}>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 10 }}>
                      <div
                        style={{
                          display: 'grid',
                          gap: 4,
                          padding: '10px 12px',
                          border: selectedBranchDiffers ? '1px solid var(--color-accent)' : '1px solid var(--color-border-subtle)',
                          borderRadius: 8,
                          background: 'color-mix(in srgb, var(--color-bg-secondary) 84%, transparent)',
                          minWidth: 0,
                        }}
                      >
                        <span className="userspace-muted" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>Branch</span>
                        {branches.length > 0 ? (
                          <select
                            value={gitBranch}
                            onChange={(event) => setGitBranch(event.target.value)}
                            disabled={isLoading}
                            style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', width: '100%', minWidth: 0 }}
                            title="Switch the branch targeted by Pull, Push, and Sync"
                          >
                            {branches.map((branch) => (
                              <option key={branch} value={branch}>{branch}</option>
                            ))}
                          </select>
                        ) : (
                          <span
                            style={{ fontSize: 13, fontWeight: 600, fontFamily: 'var(--font-mono)', overflow: 'hidden', textOverflow: 'ellipsis' }}
                            title={gitBranch || connectedBranch || 'main'}
                          >
                            {gitBranch || connectedBranch || 'main'}
                          </span>
                        )}
                        {selectedBranchDiffers && (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--color-accent)' }}>
                            <GitBranch size={11} />
                            Will pull {gitBranch.trim()}
                          </span>
                        )}
                        {branchError && (
                          <span className="userspace-muted" style={{ fontSize: 11 }}>{branchError}</span>
                        )}
                      </div>
                      {scmSummaryItems.map((item) => (
                        <div
                          key={item.label}
                          style={{
                            display: 'grid',
                            gap: 4,
                            padding: '10px 12px',
                            border: '1px solid var(--color-border-subtle)',
                            borderRadius: 8,
                            background: 'color-mix(in srgb, var(--color-bg-secondary) 84%, transparent)',
                            minWidth: 0,
                          }}
                        >
                          <span className="userspace-muted" style={{ fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.04em' }}>{item.label}</span>
                          <span
                            style={{
                              fontSize: 13,
                              fontWeight: 600,
                              overflow: 'hidden',
                              textOverflow: 'ellipsis',
                            }}
                            title={item.value}
                          >
                            {item.value}
                          </span>
                        </div>
                      ))}
                    </div>

                    {(setupPrompt || activeScm.sync_paused) && (
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        {activeScm.sync_paused && (
                          <button
                            className="btn btn-sm btn-secondary"
                            disabled={isLoading}
                            onClick={async () => {
                              try {
                                const resp = await api.updateUserSpaceWorkspaceScmSettings(workspace.id, { clear_sync_paused: true });
                                setScmState(resp.scm);
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
                    )}

                    {activeScm.sync_paused && activeScm.sync_paused_reason && (
                      <div style={{ fontSize: 12, padding: '8px 10px', borderRadius: 6, border: '1px solid var(--color-warning, #d69d2a)', background: 'rgba(214, 157, 42, 0.08)' }}>
                        <AlertCircle size={12} style={{ verticalAlign: 'middle', marginRight: 4 }} />
                        {activeScm.sync_paused_reason}
                      </div>
                    )}

                    {activeScm.remote_role === 'upstream' && (
                      <div style={{ display: 'grid', gap: 10 }}>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12, alignItems: 'start' }}>
                          {upstreamSyncRows.map((row) => (
                            <button
                              key={row.key}
                              type="button"
                              aria-pressed={row.enabled}
                              disabled={isLoading}
                              title={row.enabled ? `Disable ${row.label.toLowerCase()}` : `Enable ${row.label.toLowerCase()}`}
                              onClick={() => row.setEnabled(!row.enabled)}
                              style={{
                                display: 'flex',
                                alignItems: 'center',
                                gap: 10,
                                width: '100%',
                                padding: '12px 14px',
                                borderRadius: 10,
                                border: row.enabled ? '1px solid var(--color-accent)' : '1px solid var(--color-border)',
                                background: row.enabled
                                  ? 'color-mix(in srgb, var(--color-accent) 9%, var(--color-bg-secondary))'
                                  : 'var(--color-bg-secondary)',
                                boxShadow: row.enabled
                                  ? '0 0 0 1px color-mix(in srgb, var(--color-accent) 20%, transparent)'
                                  : 'none',
                                color: 'inherit',
                                cursor: isLoading ? 'default' : 'pointer',
                                transition: 'border-color 0.15s ease, background 0.15s ease, box-shadow 0.15s ease',
                                textAlign: 'left',
                              }}
                            >
                              <span
                                style={{
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  width: 30,
                                  height: 30,
                                  borderRadius: 999,
                                  flexShrink: 0,
                                  background: row.enabled
                                    ? 'color-mix(in srgb, var(--color-accent) 18%, transparent)'
                                    : 'var(--color-bg-tertiary)',
                                  color: row.enabled ? 'var(--color-accent)' : 'var(--color-text-muted)',
                                }}
                              >
                                {row.key === 'push' ? <ArrowUpToLine size={14} /> : <ArrowDownToLine size={14} />}
                              </span>
                              <span style={{ display: 'grid', gap: 3, minWidth: 0, flex: 1 }}>
                                <span style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                                  <strong style={{ fontSize: 13 }}>{row.label}</strong>
                                  <span className={`userspace-status-pill ${row.enabled ? 'userspace-status-pill-success' : 'userspace-status-pill-muted'}`} style={{ fontSize: 11 }}>
                                    {row.enabled ? 'On' : 'Off'}
                                  </span>
                                  {row.enabled && (
                                    <span className="userspace-status-pill userspace-status-pill-warning" style={{ fontSize: 11 }}>
                                      {hasDirtyUpstreamSyncSettings ? 'Unsaved' : `Every ${formatSyncInterval(row.interval)}`}
                                    </span>
                                  )}
                                </span>
                                <span className="userspace-muted" style={{ fontSize: 12 }}>
                                  {row.enabled ? `Runs every ${formatSyncInterval(row.interval)}.` : row.description}
                                </span>
                              </span>
                            </button>
                          ))}
                        </div>

                        {enabledUpstreamSyncCount > 0 && (
                          <div style={{ display: 'grid', gridTemplateColumns: useSingleEnabledSyncLayout ? '1fr' : 'repeat(auto-fit, minmax(300px, 1fr))', gap: 12, alignItems: 'start' }}>
                            {upstreamSyncRows.filter((row) => row.enabled).map((row) => (
                              <div
                                key={row.key}
                                style={{
                                  display: 'grid',
                                  gap: 8,
                                  padding: '10px 12px 12px',
                                  border: '1px solid color-mix(in srgb, var(--color-accent) 35%, var(--color-border))',
                                  borderRadius: 8,
                                  background: 'color-mix(in srgb, var(--color-accent) 5%, transparent)',
                                }}
                              >
                                <div style={{ display: 'grid', gap: 10, gridTemplateColumns: useSingleEnabledSyncLayout ? 'repeat(auto-fit, minmax(260px, 1fr))' : '1fr', alignItems: 'start' }}>
                                  <div style={{ display: 'grid', gap: 8 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 8 }}>
                                      <span className="userspace-muted" style={{ fontSize: 12 }}>Interval</span>
                                      <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', fontWeight: 600 }}>
                                        {formatSyncInterval(row.interval)}
                                      </span>
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                                      <span className="userspace-muted" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>5m</span>
                                      <input
                                        type="range"
                                        min="0"
                                        max="100"
                                        step="1"
                                        value={syncIntervalToSlider(row.interval)}
                                        style={{ flex: 1 }}
                                        onChange={(event) => row.setInterval(sliderToSyncInterval(parseInt(event.target.value, 10)))}
                                        disabled={isLoading}
                                      />
                                      <span className="userspace-muted" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>30d</span>
                                    </div>
                                  </div>
                                  <ScheduleStartTimeInput
                                    enabled={row.enabled}
                                    startMinute={row.startMinute}
                                    timezone={row.timezone}
                                    onStartMinuteChange={row.setStartMinute}
                                    onTimezoneChange={row.setTimezone}
                                    disabled={isLoading}
                                    label="Start time"
                                    style={{ marginBottom: 0 }}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
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
                      ({formatBytes(sqlFile.size)})
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
                      Accepts .sql, .dump, .pg, .pgsql, .mysql (plain SQL or PostgreSQL custom dump, max {formatBytes(sqlImportMaxBytes)}{sqlImportLimitLoaded ? '' : ' default'})
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {activeTab === 'sql-import' && step === 'result' && mode === 'sql-import' && sqlImportResult && (
            <div style={{ display: 'grid', gap: 14 }}>
              {isSqliteImportTaskTerminal(sqlImportResult.phase) && (
                <div style={{
                  display: 'flex', gap: 8, alignItems: 'center', padding: 12, borderRadius: 8,
                  border: `1px solid ${sqlImportResult.phase === 'completed' ? 'var(--color-success, #2b7a2b)' : 'var(--color-danger, #c53030)'}`,
                  background: sqlImportResult.phase === 'completed' ? 'rgba(43, 122, 43, 0.08)' : 'rgba(197, 48, 48, 0.08)',
                }}>
                  {sqlImportResult.phase === 'completed' ? <Check size={16} /> : <AlertCircle size={16} />}
                  <div>
                    <strong>{sqlImportResult.message || formatSqliteImportPhase(sqlImportResult.phase)}</strong>
                  </div>
                </div>
              )}

              {isSqliteImportTaskActive(sqlImportResult) && (
                <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <MiniLoadingSpinner size={12} />
                      {formatSqliteImportPhase(sqlImportResult.phase)}
                    </span>
                    <span>{getSqliteImportProgressPercent(sqlImportResult)}%</span>
                  </div>
                  <div style={{ height: 8, borderRadius: 999, background: 'var(--color-bg-tertiary)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${getSqliteImportProgressPercent(sqlImportResult)}%`, background: 'var(--color-accent)', transition: 'width 160ms ease' }} />
                  </div>
                </div>
              )}

              <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '4px 16px', fontSize: 13 }}>
                  <span className="userspace-muted">Dialect detected:</span>
                  <span>{sqlImportResult.dialect_detected}</span>
                  <span className="userspace-muted">File:</span>
                  <span>{sqlImportResult.filename}</span>
                  <span className="userspace-muted">Tables created:</span>
                  <span>{sqlImportResult.tables_created}</span>
                  <span className="userspace-muted">Rows inserted:</span>
                  <span>{sqlImportResult.rows_inserted}</span>
                  <span className="userspace-muted">Statements executed:</span>
                  <span>{sqlImportResult.statements_executed}</span>
                  {sqlImportResult.total_statements > 0 && (
                    <>
                      <span className="userspace-muted">Total statements:</span>
                      <span>{sqlImportResult.total_statements}</span>
                    </>
                  )}
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

          {activeTab === 'git-source' && step === 'result' && mode !== 'sql-import' && scmWizardActivity?.kind === 'preview' && scmWizardActivity.status === 'running' && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <MiniLoadingSpinner size={12} />
                    Analyzing repository
                  </span>
                  <span>Preview</span>
                </div>
                <div style={{ height: 6, borderRadius: 999, background: 'var(--color-bg-tertiary)', overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: '35%', background: 'var(--color-accent)', transition: 'width 160ms ease' }} />
                </div>
                <div className="userspace-muted" style={{ fontSize: 12, marginTop: 8 }}>
                  Checking the remote branch and calculating the import plan. You can close this modal and return to this progress from the SCM button.
                </div>
              </div>
            </div>
          )}

          {activeTab === 'git-source' && step === 'result' && mode !== 'sql-import' && scmImportTask && scmWizardActivity?.kind !== 'preview' && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 10, border: '1px solid var(--color-border)', borderRadius: 8, background: 'var(--color-bg-tertiary)' }}>
                <GitBranch size={14} />
                <span style={{ fontWeight: 600, flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {scmImportTask.git_branch}
                </span>
                <span className="userspace-muted" style={{ fontSize: 12, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 200 }}>
                  {scmImportTask.git_url}
                </span>
              </div>

              {scmImportTask.phase === 'failed' ? (
                <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start', padding: 12, borderRadius: 8, border: '1px solid var(--color-danger, #c53030)', background: 'rgba(197, 48, 48, 0.08)' }}>
                  <AlertCircle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                  <div>
                    <strong>Import failed</strong>
                    {scmImportTask.error && (
                      <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>{scmImportTask.error}</div>
                    )}
                  </div>
                </div>
              ) : scmImportTask.phase === 'completed' ? (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 12, border: '1px solid var(--color-success, #2b7a2b)', borderRadius: 8, background: 'rgba(43, 122, 43, 0.08)' }}>
                  <Check size={16} style={{ flexShrink: 0 }} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <strong>{scmImportTask.summary || 'Import completed.'}</strong>
                    {scmImportTask.remote_commit_hash && (
                      <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                        Remote commit: {scmImportTask.remote_commit_hash}
                      </div>
                    )}
                  </div>
                </div>
              ) : scmImportTask.phase === 'preview_ready' ? (
                <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
                    <MiniLoadingSpinner size={12} />
                    Loading preview details...
                  </div>
                </div>
              ) : (
                <div style={{ padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <MiniLoadingSpinner size={12} />
                      {formatScmImportTaskPhase(scmImportTask.phase)}
                    </span>
                    <span>{getScmImportTaskProgressPercent(scmImportTask.phase, scmImportTask.progress)}%</span>
                  </div>
                  <div style={{ height: 6, borderRadius: 999, background: 'var(--color-bg-tertiary)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${getScmImportTaskProgressPercent(scmImportTask.phase, scmImportTask.progress)}%`, background: 'var(--color-accent)', transition: 'width 160ms ease' }} />
                  </div>
                </div>
              )}

              {scmImportTask.phase === 'completed' && scmImportTask.suggested_setup_prompt && (
                <div style={{ display: 'grid', gap: 8, padding: 12, border: '1px solid var(--color-border)', borderRadius: 8 }}>
                  <strong>Prepare the imported workspace</strong>
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    This keeps bring-up suggestion-only. The agent will inspect the imported repo first, then repair entrypoint and bootstrap configuration only if needed.
                  </div>
                  <div style={{ display: 'flex', gap: 8 }}>
                    <button className="btn btn-primary btn-sm" onClick={() => void handleAskAgentFromScmTask()} disabled={!onAskAgent}>
                      <RefreshCw size={14} />
                      Ask Agent to Prepare
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'git-source' && step === 'result' && mode !== 'sql-import' && !scmImportTask && result && (
            <div style={{ display: 'grid', gap: 14 }}>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 12, border: '1px solid var(--color-success, #2b7a2b)', borderRadius: 8, background: 'rgba(43, 122, 43, 0.08)' }}>
                <Check size={16} style={{ flexShrink: 0 }} />
                <div style={{ flex: 1, minWidth: 0 }}>
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
              <button className="btn btn-secondary" onClick={() => { setWorkspaceScmWizardActivity(workspace.id, null); setStep('input'); }} disabled={isLoading}>
                Back
              </button>
            )}
            {activeTab === 'git-source' && step === 'result' && mode !== 'sql-import' && scmImportTask && isScmImportTaskTerminal(scmImportTask.phase) && (
              <button className="btn btn-secondary" onClick={() => { setWorkspaceScmWizardActivity(workspace.id, null); setScmImportTask(null); setStep('input'); setStatus({ type: null, message: '' }); }} disabled={isLoading}>
                Back
              </button>
            )}
            {activeTab === 'sql-import' && step === 'result' && mode === 'sql-import' && (
              <button className="btn btn-secondary" onClick={() => { setStep('input'); setSqlFile(null); setSqlImportResult(null); setStatus({ type: null, message: '' }); }} disabled={isLoading || isSqliteImportTaskActive(sqlImportResult)}>
                Import Another
              </button>
            )}
            {activeTab === 'archive' && archiveStep === 'configure' && (
              <button className="btn btn-secondary" onClick={() => { setArchiveStep('choose'); setStatus({ type: null, message: '' }); }} disabled={isLoading}>
                Back
              </button>
            )}
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button className="btn btn-secondary" onClick={onClose} disabled={isLoading}>
              <X size={14} /> Close
            </button>
            {activeTab === 'git-source' && step !== 'result' && mode !== 'sql-import' && !hasConfiguredRemote && (
              <button className="btn btn-secondary" onClick={() => void handleClearGitSourceFields()} disabled={isLoading || hasRunningGitSourceTask}>
                {loadingAction === 'clear-fields' ? <MiniLoadingSpinner variant="icon" size={14} /> : <X size={14} />}
                Clear Fields
              </button>
            )}
            {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && hasConfiguredRemote && hasScmSettingsMutations && (
              <button className="btn btn-primary" onClick={() => void handleSaveScmSettings()} disabled={isLoading || loadingAction === 'save-settings'}>
                {loadingAction === 'save-settings' ? <MiniLoadingSpinner variant="icon" size={14} /> : <Check size={14} />}
                Save
              </button>
            )}

            {activeTab === 'sql-import' && step === 'input' && mode === 'sql-import' && (
              <button className="btn btn-primary" onClick={() => void handleSqlImport()} disabled={isLoading || !sqlFile || isSqliteImportTaskActive(sqlImportResult)}>
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <Database size={14} />}
                Import to SQLite
              </button>
            )}
            {activeTab === 'archive' && archiveStep === 'configure' && archiveMode === 'export' && (
              <>
                {archiveExportTask?.phase === 'completed' && archiveExportTask.archive_file_name && (
                  <button className="btn btn-secondary" onClick={() => void handleDownloadArchive(archiveExportTask.task_id)} disabled={isLoading || lastDownloadedArchiveTaskIdRef.current === archiveExportTask.task_id}>
                    <ArrowDownToLine size={14} />
                    {lastDownloadedArchiveTaskIdRef.current === archiveExportTask.task_id ? 'Downloaded' : 'Download Archive'}
                  </button>
                )}
                <button className="btn btn-primary" onClick={() => void handleQueueArchiveExport()} disabled={isLoading || Boolean(archiveExportTask && !isArchiveExportTaskTerminal(archiveExportTask.phase))}>
                  {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <ArrowUpToLine size={14} />}
                  Export
                </button>
              </>
            )}
            {activeTab === 'archive' && archiveStep === 'configure' && archiveMode === 'import' && (
              <button className="btn btn-primary" onClick={() => void handleQueueArchiveImport()} disabled={isLoading || !archiveFile || Boolean(archiveImportTask && !isArchiveImportTaskTerminal(archiveImportTask.phase))}>
                {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <ArrowDownToLine size={14} />}
                Import
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
                <button
                  className="btn btn-primary"
                  onClick={() => void handlePreview('import')}
                  disabled={isLoading}
                  title={selectedBranchDiffers ? `Pull branch ${gitBranch.trim()}` : 'Pull from the connected branch'}
                >
                  {loadingAction === 'pull' ? <MiniLoadingSpinner variant="icon" size={14} /> : <ArrowDownToLine size={14} />}
                  {selectedBranchDiffers ? `Pull ${gitBranch.trim()}` : 'Pull'}
                </button>
                <button className="btn btn-secondary" onClick={() => void handlePreview('export')} disabled={isLoading}>
                  {loadingAction === 'push' ? <MiniLoadingSpinner variant="icon" size={14} /> : <ArrowUpToLine size={14} />}
                  Push
                </button>
                <div style={{ position: 'relative', alignSelf: 'stretch' }}>
                  <button className="btn btn-secondary" onClick={() => setShowMoreMenu(prev => !prev)} disabled={isLoading}
                    title="More options" style={{ padding: '6px 8px', minWidth: 0, height: '100%' }}>
                    &#8230;
                  </button>
                  {showMoreMenu && (
                    <div style={{
                      position: 'absolute', bottom: 'calc(100% + 6px)', right: 0, minWidth: 200,
                      padding: '10px 12px', borderRadius: 8, border: '1px solid var(--color-border)',
                      background: 'var(--color-bg-secondary)', boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
                      display: 'grid', gap: 8, zIndex: 10,
                    }}>
                      {setupPrompt && (
                        <button
                          className="btn btn-sm btn-secondary"
                          style={{ width: '100%' }}
                          title="Copy the setup prompt from the last import to the clipboard"
                          onClick={() => { void navigator.clipboard.writeText(setupPrompt); setShowMoreMenu(false); }}
                        >
                          Copy Prompt
                        </button>
                      )}
                      <Popover
                        position="left"
                        trigger="hover"
                        style={{ width: '100%' }}
                        content={<span style={{ fontSize: 11 }}>Replaces all local files with the remote state. Local-only changes will be lost.</span>}
                      >
                        <DeleteConfirmButton
                          onDelete={() => { setShowMoreMenu(false); void handlePreview('import', { forceOverwrite: true }); }}
                          disabled={isLoading}
                          className="btn btn-sm btn-danger"
                          style={{ width: '100%' }}
                          title="Overwrite local files with remote state"
                          buttonText="Overwrite local"
                        />
                      </Popover>
                      <Popover
                        position="left"
                        trigger="hover"
                        style={{ width: '100%' }}
                        content={<span style={{ fontSize: 11 }}>Removes the configured Git remote and clears the stored token.</span>}
                      >
                        <DeleteConfirmButton
                          onDelete={() => { setShowMoreMenu(false); void handleDisconnectScm(); }}
                          disabled={isLoading}
                          deleting={loadingAction === 'disconnect'}
                          className="btn btn-sm btn-danger"
                          style={{ width: '100%' }}
                          title="Disconnect the configured remote and clear the stored token"
                          buttonText="Disconnect Remote"
                        />
                      </Popover>
                    </div>
                  )}
                </div>
              </>
            )}
            {activeTab === 'git-source' && step === 'input' && mode !== 'sql-import' && hasConfiguredRemote && activeScm?.remote_role !== 'upstream' && (
              <>
                <button
                  className="btn btn-primary"
                  onClick={() => void handlePreview()}
                  disabled={isLoading}
                  title={selectedBranchDiffers ? `Sync branch ${gitBranch.trim()}` : 'Sync with the connected branch'}
                >
                  {isLoading ? <MiniLoadingSpinner variant="icon" size={14} /> : <RefreshCcw size={14} />}
                  {selectedBranchDiffers ? `Sync ${gitBranch.trim()}` : 'Sync'}
                </button>
                <div style={{ position: 'relative', alignSelf: 'stretch' }}>
                  <button className="btn btn-secondary" onClick={() => setShowMoreMenu(prev => !prev)} disabled={isLoading}
                    title="More options" style={{ padding: '6px 8px', minWidth: 0, height: '100%' }}>
                    &#8230;
                  </button>
                  {showMoreMenu && (
                    <div style={{
                      position: 'absolute', bottom: 'calc(100% + 6px)', right: 0, minWidth: 200,
                      padding: '10px 12px', borderRadius: 8, border: '1px solid var(--color-border)',
                      background: 'var(--color-bg-secondary)', boxShadow: '0 4px 12px rgba(0,0,0,0.25)',
                      display: 'grid', gap: 8, zIndex: 10,
                    }}>
                      {setupPrompt && (
                        <button
                          className="btn btn-sm btn-secondary"
                          style={{ width: '100%' }}
                          title="Copy the setup prompt from the last import to the clipboard"
                          onClick={() => { void navigator.clipboard.writeText(setupPrompt); setShowMoreMenu(false); }}
                        >
                          Copy Prompt
                        </button>
                      )}
                      <Popover
                        position="left"
                        trigger="hover"
                        style={{ width: '100%' }}
                        content={<span style={{ fontSize: 11 }}>Removes the configured Git remote and clears the stored token.</span>}
                      >
                        <DeleteConfirmButton
                          onDelete={() => { setShowMoreMenu(false); void handleDisconnectScm(); }}
                          disabled={isLoading}
                          deleting={loadingAction === 'disconnect'}
                          className="btn btn-sm btn-danger"
                          style={{ width: '100%' }}
                          title="Disconnect the configured remote and clear the stored token"
                          buttonText="Disconnect Remote"
                        />
                      </Popover>
                    </div>
                  )}
                </div>
              </>
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
