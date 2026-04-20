import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { AlertCircle, ArrowLeft, ArrowLeftRight, ArrowRight, Check, ChevronDown, ChevronRight, Copy, CopyPlus, Crown, Database, ExternalLink, File, GitBranch, HardDrive, HardDriveDownload, HardDriveUpload, History, Info, KeyRound, Link2, Maximize2, Minimize2, Pencil, Play, Plus, RefreshCw, RotateCw, Save, Shield, Slash, Square, Terminal, Trash2, Users, X } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { keymap } from '@codemirror/view';
import { openSearchPanel } from '@codemirror/search';

import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';

import { api, ApiError } from '@/api';
import {
  clearInterruptDismiss,
  resolveWorkspaceInterruptStateFromSummary,
} from '@/utils';
import { useCodeMirrorLanguageExtension } from '@/utils/codemirrorLanguage';
import type { InterruptChatStateSnapshot } from '@/utils/cookies';
import AdminWorkspaceModal from './shared/AdminWorkspaceModal';
import { AgentAccessButton } from './shared/AgentAccessButton';
import { AgentAccessModal } from './shared/AgentAccessModal';
import { MemberManagementButton } from './shared/MemberManagementButton';
import { MemberManagementModal, type Member } from './shared/MemberManagementModal';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { ToolSelectorDropdown, type ToolGroupInfo } from './shared/ToolSelectorDropdown';
import type { BrowseResponse, DirectoryEntry, MountableSource, UpsertUserSpaceWorkspaceEnvVarRequest, UpsertWorkspaceAgentGrantRequest, User, UserSpaceArtifactType, UserSpaceAvailableTool, UserSpaceBrowserSurface, UserSpaceCollabMessage, UserSpaceFileInfo, UserSpaceLiveDataConnection, UserSpaceObjectStorageBucket, UserSpaceObjectStorageConfig, UserSpacePreviewWarning, UserSpaceRuntimeStatusResponse, UserSpaceShareAccessMode, UserSpaceSnapshot, UserSpaceSnapshotBranch, UserSpaceSnapshotDiffSummary, UserSpaceSnapshotFileDiff, UserSpaceSnapshotTimeline, UserSpaceWorkspace, UserSpaceWorkspaceCreateTask, UserSpaceWorkspaceCreateTaskPhase, UserSpaceWorkspaceDeleteTask, UserSpaceWorkspaceDeleteTaskPhase, UserSpaceWorkspaceDuplicateTask, UserSpaceWorkspaceDuplicateTaskPhase, UserSpaceWorkspaceEnvVar, UserSpaceWorkspaceMember, UserSpaceWorkspaceShareLinkStatus, UserSpaceWorkspaceScmStatus, UserSpaceWorkspaceScmSyncResponse, WorkspaceAgentGrant, WorkspaceChatStateResponse, WorkspaceMount, WorkspaceMountSyncMode, WorkspaceMountSyncPreviewResponse } from '@/types';
import { buildUserSpaceTree, collectFilePaths, getAncestorFolderPaths, listFolderPaths } from '@/utils/userspaceTree';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';
import { useDiffHoverTimers } from '@/utils/useDiffHoverTimers';
import { ChatPanel } from './ChatPanel';
import { LdapGroupSelect } from './LdapGroupSelect';
import { ResizeHandle } from './ResizeHandle';
import { UserSpaceArtifactPreview } from './UserSpaceArtifactPreview';
import { ConstrainedPathBrowser } from './ConstrainedPathBrowser';
import { FileDiffOverlay } from './shared/FileDiffOverlay';
import { UserSpaceEnvVarsModal } from './shared/UserSpaceEnvVarsModal';
import { WorkspaceObjectStorageWizard } from './MountSourceWizard';
import { WorkspaceScmWizard } from './WorkspaceScmWizard';

interface UserSpacePanelProps {
  currentUser: User;
  debugMode?: boolean;
  onFullscreenChange?: (fullscreen: boolean) => void;
  onNavigateToTools?: (section?: string) => void;
  onPreviewWarningChange?: (warning: UserSpacePreviewWarning | null) => void;
}

interface WorkspaceChatState {
  hasLive: boolean;
  hasInterrupted: boolean;
}

interface CachedUserSpaceFile {
  content: string;
  updatedAt: string;
  artifactType: UserSpaceArtifactType | null;
}

type ShareLinkType = 'named' | 'anonymous' | 'subdomain';

type MountSyncPreviewIntent = 'sync' | 'enable-auto' | 'update-auto-sync-mode';

const DEFAULT_WORKSPACE_CHAT_STATE: WorkspaceChatState = { hasLive: false, hasInterrupted: false };

function isWorkspaceCreateTaskTerminal(phase: UserSpaceWorkspaceCreateTaskPhase): boolean {
  return phase === 'completed' || phase === 'failed';
}

function formatWorkspaceCreateTaskStatus(task: UserSpaceWorkspaceCreateTask | null): string | null {
  if (!task) {
    return null;
  }

  const label = task.workspace_name?.trim() || 'workspace';
  switch (task.phase) {
    case 'queued':
      return `Preparing to create ${label}...`;
    case 'creating_workspace':
      return `Creating ${label}...`;
    case 'bootstrapping_files':
      return `Bootstrapping files for ${label}...`;
    case 'creating_conversation':
      return `Setting up conversation for ${label}...`;
    case 'failed':
      return task.error?.trim() || `Failed to create ${label}.`;
    default:
      return null;
  }
}

function formatWorkspaceCreateTasksStatus(tasks: UserSpaceWorkspaceCreateTask[]): string | null {
  if (tasks.length === 0) {
    return null;
  }
  if (tasks.length === 1) {
    return formatWorkspaceCreateTaskStatus(tasks[0]);
  }

  const queuedCount = tasks.filter((task) => task.phase === 'queued').length;
  return queuedCount > 0
    ? `Creating ${tasks.length} workspaces (${queuedCount} queued)...`
    : `Creating ${tasks.length} workspaces...`;
}

function isWorkspaceDuplicateTaskTerminal(phase: UserSpaceWorkspaceDuplicateTaskPhase): boolean {
  return phase === 'completed' || phase === 'failed';
}

function formatWorkspaceDuplicateTaskStatus(task: UserSpaceWorkspaceDuplicateTask | null): string | null {
  if (!task) {
    return null;
  }

  const label = task.workspace_name?.trim() || 'workspace';
  switch (task.phase) {
    case 'queued':
      return `Preparing to duplicate ${label}...`;
    case 'creating_workspace':
      return `Creating ${label}...`;
    case 'copying_files':
      return `Copying files into ${label}...`;
    case 'creating_conversation':
      return `Setting up conversation for ${label}...`;
    case 'failed':
      return task.error?.trim() || `Failed to duplicate ${label}.`;
    default:
      return null;
  }
}

function formatWorkspaceDuplicateTasksStatus(tasks: UserSpaceWorkspaceDuplicateTask[]): string | null {
  if (tasks.length === 0) {
    return null;
  }
  if (tasks.length === 1) {
    return formatWorkspaceDuplicateTaskStatus(tasks[0]);
  }

  const queuedCount = tasks.filter((task) => task.phase === 'queued').length;
  return queuedCount > 0
    ? `Duplicating ${tasks.length} workspaces (${queuedCount} queued)...`
    : `Duplicating ${tasks.length} workspaces...`;
}

function isWorkspaceDeleteTaskTerminal(phase: UserSpaceWorkspaceDeleteTaskPhase): boolean {
  return phase === 'completed' || phase === 'failed';
}

function formatWorkspaceDeleteTaskStatus(task: UserSpaceWorkspaceDeleteTask | null): string | null {
  if (!task) {
    return null;
  }

  const label = task.workspace_name?.trim() || 'workspace';
  switch (task.phase) {
    case 'queued':
      return `Preparing to delete ${label}...`;
    case 'stopping_runtime':
      return `Stopping runtime for ${label}...`;
    case 'deleting_conversations':
      return `Deleting conversations for ${label}...`;
    case 'deleting_workspace':
      return `Deleting ${label}...`;
    case 'failed':
      return task.error?.trim() || `Failed to delete ${label}.`;
    default:
      return null;
  }
}

function formatWorkspaceDeleteTasksStatus(tasks: UserSpaceWorkspaceDeleteTask[]): string | null {
  if (tasks.length === 0) {
    return null;
  }
  if (tasks.length === 1) {
    return formatWorkspaceDeleteTaskStatus(tasks[0]);
  }

  const queuedCount = tasks.filter((task) => task.phase === 'queued').length;
  return queuedCount > 0
    ? `Deleting ${tasks.length} workspaces (${queuedCount} queued)...`
    : `Deleting ${tasks.length} workspaces...`;
}

function normalizeWorkspacePath(value: string): string {
  return value.trim().replace(/^\/+|\/+$/g, '').replace(/\/+/g, '/');
}

function normalizeMountBrowserPath(value: string): string {
  const normalizedParts: string[] = [];
  for (const part of (value || '/').replace(/\\/g, '/').split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      normalizedParts.pop();
      continue;
    }
    normalizedParts.push(part);
  }
  return '/' + normalizedParts.join('/');
}

function sourcePathToBrowserPath(sourcePath: string): string {
  const normalized = sourcePath.trim();
  if (!normalized || normalized === '.') return '/';
  return normalizeMountBrowserPath(`/${normalized}`);
}

function browserPathToSourcePath(browserPath: string): string {
  const normalized = normalizeMountBrowserPath(browserPath);
  return normalized === '/' ? '.' : normalized.slice(1);
}

function browserPathToWorkspaceMountTargetPath(browserPath: string): string {
  const normalized = normalizeMountBrowserPath(browserPath);
  return normalized === '/' ? '/workspace' : `/workspace${normalized}`;
}

function getShareLinkTypeStorageKey(workspaceId: string): string {
  return `userspace-share-link-type:${workspaceId}`;
}

const WORKSPACE_MOUNT_SYNC_MODE_OPTIONS: Array<{
  value: WorkspaceMountSyncMode;
  label: string;
  icon: typeof ArrowRight;
  description: string;
}> = [
  {
    value: 'merge',
    label: 'Merge',
    icon: ArrowLeftRight,
    description: 'Bidirectional merge. Newer files win and nothing is deleted.',
  },
  {
    value: 'source_authoritative',
    label: 'Source',
    icon: ArrowRight,
    description: 'Remote source is authoritative. Target-only files are deleted on sync.',
  },
  {
    value: 'target_authoritative',
    label: 'Target',
    icon: ArrowLeft,
    description: 'Workspace target is authoritative. Source-only files are deleted on sync.',
  },
];

function isDestructiveMountSyncMode(syncMode: WorkspaceMountSyncMode): boolean {
  return syncMode !== 'merge';
}

function getMountSyncModeLabel(syncMode: WorkspaceMountSyncMode): string {
  return WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.find((option) => option.value === syncMode)?.label ?? 'Merge';
}

function getMountSyncModeIcon(syncMode: WorkspaceMountSyncMode): typeof ArrowRight {
  return WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.find((option) => option.value === syncMode)?.icon ?? ArrowLeftRight;
}

function getMountSyncModeDescription(syncMode: WorkspaceMountSyncMode): string {
  return WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.find((option) => option.value === syncMode)?.description
    ?? WORKSPACE_MOUNT_SYNC_MODE_OPTIONS[0].description;
}

function formatMountSyncPreviewPath(path: string): string {
  return path.startsWith('/') ? path : `/${path}`;
}

function isMountBrowserPathEqualOrDescendant(path: string, candidateAncestorPath: string): boolean {
  const normalizedPath = normalizeMountBrowserPath(path);
  const normalizedAncestorPath = normalizeMountBrowserPath(candidateAncestorPath);
  return normalizedPath === normalizedAncestorPath || normalizedPath.startsWith(`${normalizedAncestorPath}/`);
}

function resolveMountDirectoryToCreate(
  selectedBrowserPath: string,
  stagedBrowserPaths: string[],
  toRequestPath: (browserPath: string) => string,
): string | null {
  if (!selectedBrowserPath || stagedBrowserPaths.length === 0) {
    return null;
  }

  const normalizedSelectedPath = normalizeMountBrowserPath(selectedBrowserPath);
  const requiresCreation = stagedBrowserPaths.some((stagedPath) =>
    isMountBrowserPathEqualOrDescendant(normalizedSelectedPath, stagedPath)
  );
  return requiresCreation ? toRequestPath(normalizedSelectedPath) : null;
}

function getMountSourceBrowserStageKey(mountSourceId: string, rootSourcePath: string): string {
  return `${mountSourceId}::${rootSourcePath}`;
}

function getExpandedFoldersStorageKey(workspaceId: string): string {
  return `userspace:expanded-folders:${workspaceId}`;
}

function normalizeShareSlugInput(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, '_')
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[_-]+|[_-]+$/g, '')
    .slice(0, 80);
}

function getDefaultShareSlug(value: string | null | undefined): string {
  const normalized = normalizeShareSlugInput(value ?? '');
  if (normalized && !normalized.startsWith('workspace')) {
    return `share_${normalized.slice(0, 24)}`;
  }
  return 'share_workspace';
}

function normalizeUniqueStrings(values: string[]): string[] {
  return Array.from(new Set(values.map((value) => value.trim()).filter(Boolean))).sort();
}

function areSameNormalizedStringArrays(left: string[], right: string[]): boolean {
  const normalizedLeft = normalizeUniqueStrings(left);
  const normalizedRight = normalizeUniqueStrings(right);
  if (normalizedLeft.length !== normalizedRight.length) {
    return false;
  }
  return normalizedLeft.every((value, index) => value === normalizedRight[index]);
}

function fileEntriesFingerprint(entries: UserSpaceFileInfo[]): string {
  return entries.map((e) => `${e.path}:${e.updated_at ?? ''}`).join('\n');
}

/**
 * Compute an aligned side-by-side diff using LCS-based diffLines.
 * Both sides get the same number of lines with blank padding inserted
 * opposite added/deleted hunks so line numbers stay in sync.
 */
const LAST_WORKSPACE_COOKIE_PREFIX = 'userspace_last_workspace_id_';
const LAST_WORKSPACE_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
const USERSPACE_LAYOUT_COOKIE_PREFIX = 'userspace_layout_';
const USERSPACE_FULLSCREEN_COOKIE_PREFIX = 'userspace_fullscreen_';
const USERSPACE_CODEMIRROR_BASIC_SETUP = {
  lineNumbers: true,
  foldGutter: true,
  bracketMatching: true,
  closeBrackets: true,
  autocompletion: true,
  highlightActiveLine: true,
  indentOnInput: true,
  tabSize: 2,
};
const USERSPACE_CHANGED_FILE_STATE_MIN_INTERVAL_MS = 1000;
const USERSPACE_FILE_TREE_POLL_INTERVAL_MS = 5000;
const USERSPACE_FILE_TREE_IDLE_POLL_INTERVAL_MS = 10000;
const USERSPACE_FILE_TREE_MAX_IDLE_POLL_INTERVAL_MS = 15000;
const USERSPACE_FILE_TREE_BACKGROUND_POLL_INTERVAL_MS = 20000;
const USERSPACE_WORKSPACE_BADGE_BACKGROUND_POLL_INTERVAL_MS = 20000;
const USERSPACE_RUNTIME_BACKGROUND_POLL_INTERVAL_MS = 30000;
const USERSPACE_BROWSER_AUTH_REFRESH_LEAD_MS = 60_000;
const USERSPACE_PREVIEW_LAUNCH_REFRESH_LEAD_MS = 60_000;
const SNAPSHOT_FILE_DIFF_CACHE_MAX_ENTRIES = 20;

function getLastWorkspaceCookieName(userId: string): string {
  return `${LAST_WORKSPACE_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

function getUserSpaceLayoutCookieName(userId: string): string {
  return `${USERSPACE_LAYOUT_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

function getUserSpaceFullscreenCookieName(userId: string): string {
  return `${USERSPACE_FULLSCREEN_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

function getCookieValue(name: string): string | null {
  const entries = document.cookie ? document.cookie.split('; ') : [];
  for (const entry of entries) {
    const separatorIndex = entry.indexOf('=');
    if (separatorIndex < 0) continue;
    const key = entry.slice(0, separatorIndex);
    if (key !== name) continue;
    const value = entry.slice(separatorIndex + 1);
    try {
      return decodeURIComponent(value);
    } catch {
      return value;
    }
  }
  return null;
}

function setCookieValue(name: string, value: string): void {
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${LAST_WORKSPACE_COOKIE_MAX_AGE_SECONDS}; samesite=lax`;
}

function setSessionCookieValue(name: string, value: string): void {
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; samesite=lax`;
}

function clearCookieValue(name: string): void {
  document.cookie = `${name}=; path=/; max-age=0; samesite=lax`;
}

function parseUtcTimestamp(value: string): Date | null {
  const normalized = value.trim();
  if (!normalized) {
    return null;
  }

  const hasExplicitTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(normalized);
  const parsed = new Date(hasExplicitTimezone ? normalized : `${normalized}Z`);
  if (Number.isNaN(parsed.getTime())) {
    return null;
  }

  return parsed;
}

function parseUtcTimestampMs(value: string): number {
  return parseUtcTimestamp(value)?.getTime() ?? 0;
}

function formatSnapshotTimestamp(value: string): string {
  const date = parseUtcTimestamp(value);
  if (!date) {
    return value;
  }

  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function getSnapshotDiffFileKey(snapshotId: string, filePath: string): string {
  return `${snapshotId}:${filePath}`;
}

function getDeterministicBranchColor(seed: string): string {
  let hash = 0;
  for (let i = 0; i < seed.length; i += 1) {
    hash = ((hash << 5) - hash + seed.charCodeAt(i)) | 0;
  }

  const positiveHash = Math.abs(hash);
  const hue = positiveHash % 360;
  const saturation = 62 + ((positiveHash >> 8) % 14);
  const lightness = 50 + ((positiveHash >> 16) % 10);
  return `hsl(${hue} ${saturation}% ${lightness}%)`;
}

interface StoredUserSpaceLayout {
  sidebarWidth: number;
  sidebarCollapsed: boolean;
  leftPaneFraction: number;
  rightPaneCollapsed: boolean;
  editorFraction: number;
  editorChatCollapsedSide: 'before' | 'after' | null;
}

function readStoredUserSpaceLayout(cookieName: string): StoredUserSpaceLayout | null {
  const raw = getCookieValue(cookieName);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as Partial<StoredUserSpaceLayout>;
    const collapsedSide = parsed.editorChatCollapsedSide;
    return {
      sidebarWidth: typeof parsed.sidebarWidth === 'number' ? parsed.sidebarWidth : 180,
      sidebarCollapsed: Boolean(parsed.sidebarCollapsed),
      leftPaneFraction: typeof parsed.leftPaneFraction === 'number' ? parsed.leftPaneFraction : 0.5,
      rightPaneCollapsed: Boolean(parsed.rightPaneCollapsed),
      editorFraction: typeof parsed.editorFraction === 'number' ? parsed.editorFraction : 0.6,
      editorChatCollapsedSide: collapsedSide === 'before' || collapsedSide === 'after' ? collapsedSide : null,
    };
  } catch {
    return null;
  }
}

function readStoredUserSpaceFullscreen(cookieName: string): boolean {
  const raw = getCookieValue(cookieName);
  if (!raw) return false;

  const normalized = raw.trim().toLowerCase();
  return normalized === '1' || normalized === 'true';
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function extractApiErrorDetail(payload: string): string | null {
  const normalized = payload.trim();
  if (!normalized) return null;

  try {
    const parsed = JSON.parse(normalized) as { detail?: unknown };
    const detail = parsed?.detail;
    if (typeof detail === 'string' && detail.trim()) {
      return detail.trim();
    }
  } catch {
    // Ignore parse errors and try regex fallback.
  }

  const detailMatch = normalized.match(/"detail"\s*:\s*"((?:\\.|[^"\\])*)"/);
  if (detailMatch?.[1]) {
    try {
      return JSON.parse(`"${detailMatch[1]}"`).trim();
    } catch {
      return detailMatch[1].replace(/\\"/g, '"').trim();
    }
  }

  return null;
}

function formatUserSpaceErrorMessage(rawError: string | null): string | null {
  if (!rawError) return null;
  const normalized = rawError.trim();
  if (!normalized) return null;

  const payloadMatch = normalized.match(/^(.*?):\s*(\{[\s\S]*\})\s*$/);
  if (payloadMatch) {
    const prefix = payloadMatch[1].trim();
    const detail = extractApiErrorDetail(payloadMatch[2]);
    if (detail) {
      return `${prefix}: ${detail}`;
    }
  }

  const directDetail = extractApiErrorDetail(normalized);
  if (directDetail) {
    return directDetail;
  }

  return normalized;
}

function getUnsupportedEditorFileMessage(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== 415) {
    return null;
  }

  return formatUserSpaceErrorMessage(error.detail ?? error.message)
    ?? 'This file cannot be opened in the text editor.';
}

function getApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    return formatUserSpaceErrorMessage(error.detail ?? error.message) ?? fallback;
  }
  if (error instanceof Error) {
    return formatUserSpaceErrorMessage(error.message) ?? fallback;
  }
  return fallback;
}

function sortWorkspaceAgentGrants(grants: WorkspaceAgentGrant[]): WorkspaceAgentGrant[] {
  return [...grants].sort((left, right) => {
    const leftLabel = left.target_workspace_name?.trim() || left.target_workspace_id;
    const rightLabel = right.target_workspace_name?.trim() || right.target_workspace_id;
    return leftLabel.localeCompare(rightLabel);
  });
}

export function UserSpacePanel({ currentUser, debugMode = false, onFullscreenChange, onNavigateToTools, onPreviewWarningChange }: UserSpacePanelProps) {
  const previewEntryPath = 'dashboard/main.ts';
  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [workspacesTotal, setWorkspacesTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [_success, setSuccess] = useState<string | null>(null);
  const [statusOverlayVisible, setStatusOverlayVisible] = useState(false);
  const [statusOverlayFading, setStatusOverlayFading] = useState(false);
  const [statusOverlayPinned, setStatusOverlayPinned] = useState(false);
  const [statusOverlayInteracting, setStatusOverlayInteracting] = useState(false);

  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [activeWorkspaceConversationId, setActiveWorkspaceConversationId] = useState<string | null>(null);
  const [activeWorkspaceChatSnapshot, setActiveWorkspaceChatSnapshot] = useState<WorkspaceChatStateResponse | null>(null);
  const [workspaceChatStates, setWorkspaceChatStates] = useState<Record<string, WorkspaceChatState>>({});
  const [fileBrowserEntries, setFileBrowserEntries] = useState<UserSpaceFileInfo[]>([]);
  const [files, setFiles] = useState<UserSpaceFileInfo[]>([]);
  const [snapshots, setSnapshots] = useState<UserSpaceSnapshot[]>([]);
  const [snapshotBranches, setSnapshotBranches] = useState<UserSpaceSnapshotBranch[]>([]);
  const [currentSnapshotId, setCurrentSnapshotId] = useState<string | null>(null);
  const [currentSnapshotBranchId, setCurrentSnapshotBranchId] = useState<string | null>(null);
  const [renamingSnapshotId, setRenamingSnapshotId] = useState<string | null>(null);
  const [snapshotEditValue, setSnapshotEditValue] = useState('');
  const [savingSnapshotRename, setSavingSnapshotRename] = useState(false);
    const [deletingSnapshotId, setDeletingSnapshotId] = useState<string | null>(null);
    const [deleteConfirmSnapshotId, setDeleteConfirmSnapshotId] = useState<string | null>(null);
    const [navigatingSnapshots, setNavigatingSnapshots] = useState(false);
  const [restoringSnapshotId, setRestoringSnapshotId] = useState<string | null>(null);
  const [branchRestoreSnapshotId, setBranchRestoreSnapshotId] = useState<string | null>(null);
  const [availableTools, setAvailableTools] = useState<UserSpaceAvailableTool[]>([]);
  const [toolGroups, setToolGroups] = useState<ToolGroupInfo[]>([]);

  const [selectedFilePath, setSelectedFilePath] = useState<string>('dashboard/main.ts');
  const [fileContent, setFileContent] = useState<string>('');
  const [fileDirty, setFileDirty] = useState(false);
  const [selectedFileArtifactType, setSelectedFileArtifactType] = useState<UserSpaceArtifactType | null>(null);
  const [fileContentCache, setFileContentCache] = useState<Record<string, CachedUserSpaceFile>>({});
  const [selectedFileUnsupportedMessage, setSelectedFileUnsupportedMessage] = useState<string | null>(null);
  const [previewLiveDataConnections, setPreviewLiveDataConnections] = useState<UserSpaceLiveDataConnection[]>([]);
  const [previewExecuting, setPreviewExecuting] = useState(false);
  const [previewRefreshCounter, setPreviewRefreshCounter] = useState(0);
  const [previewFrameUrl, setPreviewFrameUrl] = useState<string | null>(null);
  const [previewOrigin, setPreviewOrigin] = useState<string | null>(null);
  const [previewAuthorizationPending, setPreviewAuthorizationPending] = useState(false);
  const [previewNotice, setPreviewNotice] = useState<{
    id: number;
    message: string;
    tone?: 'success' | 'error';
  } | null>(null);
  const [runtimeStatus, setRuntimeStatus] = useState<UserSpaceRuntimeStatusResponse | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [activeRightTab, setActiveRightTab] = useState<'preview' | 'console'>('preview');
  const [collabConnected, setCollabConnected] = useState(false);
  const [collabReadOnly, setCollabReadOnly] = useState(false);
  const [collabVersion, setCollabVersion] = useState(0);
  const [collabPresenceCount, setCollabPresenceCount] = useState(0);
  const [collabReconnectNonce, setCollabReconnectNonce] = useState(0);
  const [workspaceEventsReconnectNonce, setWorkspaceEventsReconnectNonce] = useState(0);
  const [terminalReadOnly, setTerminalReadOnly] = useState(false);
  const [terminalReconnectNonce, setTerminalReconnectNonce] = useState(0);
  const [isPageVisible, setIsPageVisible] = useState(
    typeof document === 'undefined' ? true : document.visibilityState !== 'hidden'
  );
  const [creatingWorkspaceTasks, setCreatingWorkspaceTasks] = useState<Record<string, UserSpaceWorkspaceCreateTask>>({});
  const [deletingWorkspaceTasks, setDeletingWorkspaceTasks] = useState<Record<string, UserSpaceWorkspaceDeleteTask>>({});
  const [duplicatingWorkspaceTasks, setDuplicatingWorkspaceTasks] = useState<Record<string, UserSpaceWorkspaceDuplicateTask>>({});
  const [sharingWorkspace, setSharingWorkspace] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const [shareLinkType, setShareLinkType] = useState<ShareLinkType>('anonymous');
  const [showShareModal, setShowShareModal] = useState(false);
  const [shareLinkStatus, setShareLinkStatus] = useState<UserSpaceWorkspaceShareLinkStatus | null>(null);
  const [loadingShareStatus, setLoadingShareStatus] = useState(false);
  const [rotatingShareLink, setRotatingShareLink] = useState(false);
  const [revokingShareLink, setRevokingShareLink] = useState(false);
  const [autoCreateShareLinkAttempted, setAutoCreateShareLinkAttempted] = useState(false);
  const [shareSlugDraft, setShareSlugDraft] = useState('');
  const [checkingShareSlug, setCheckingShareSlug] = useState(false);
  const [shareSlugAvailable, setShareSlugAvailable] = useState<boolean | null>(null);
  const [shareAccessMode, setShareAccessMode] = useState<UserSpaceShareAccessMode>('token');
  const [sharePasswordDraft, setSharePasswordDraft] = useState('');
  const [shareSelectedUserIdsDraft, setShareSelectedUserIdsDraft] = useState<string[]>([]);
  const [shareSelectedLdapGroupsDraft, setShareSelectedLdapGroupsDraft] = useState<string[]>([]);
  const [shareLdapGroupDraft, setShareLdapGroupDraft] = useState('');
  const [ldapDiscoveredGroups, setLdapDiscoveredGroups] = useState<{ dn: string; name: string }[]>([]);
  const [loadingLdapGroups, setLoadingLdapGroups] = useState(false);
  const [savingShareAccess, setSavingShareAccess] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = useCallback(() => {
    const next = !isFullscreen;
    setIsFullscreen(next);
    onFullscreenChange?.(next);
  }, [isFullscreen, onFullscreenChange]);
  const [savingTreeFile, setSavingTreeFile] = useState<string | null>(null);
  const [creatingSnapshot, setCreatingSnapshot] = useState(false);
  // Per-file changed tracking: files marked changed since last snapshot baseline.
  // A file is "changed" when the user edits it in the editor.
  // A file is "acknowledged" when the user clicks the per-file Save in the tree.
  // All markers reset on snapshot create/restore.
  const [changedFiles, setChangedFiles] = useState<Set<string>>(new Set());
  const [acknowledgedFiles, setAcknowledgedFiles] = useState<Set<string>>(new Set());
  const [savingWorkspaceTools, setSavingWorkspaceTools] = useState(false);
  const [deleteConfirmFileId, setDeleteConfirmFileId] = useState<string | null>(null);
  const [deleteConfirmFolderPath, setDeleteConfirmFolderPath] = useState<string | null>(null);
  const [newFileName, setNewFileName] = useState<string | null>(null);
  const [newFileParentPath, setNewFileParentPath] = useState<string>('');
  const [renamingFilePath, setRenamingFilePath] = useState<string | null>(null);
  const [renamingFolderPath, setRenamingFolderPath] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set());
  const [deleteConfirmWorkspaceId, setDeleteConfirmWorkspaceId] = useState<string | null>(null);
  const [isWorkspaceMenuOpen, setIsWorkspaceMenuOpen] = useState(false);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [showAgentAccessModal, setShowAgentAccessModal] = useState(false);
  const [showAdminWorkspacesModal, setShowAdminWorkspacesModal] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [pendingMembers, setPendingMembers] = useState<UserSpaceWorkspaceMember[]>([]);
  const [savingMembers, setSavingMembers] = useState(false);
  const [agentGrants, setAgentGrants] = useState<WorkspaceAgentGrant[]>([]);
  const [agentGrantWorkspaces, setAgentGrantWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [agentGrantsLoading, setAgentGrantsLoading] = useState(false);
  const [savingAgentGrantTargetId, setSavingAgentGrantTargetId] = useState<string | null>(null);
  const [revokingAgentGrantTargetId, setRevokingAgentGrantTargetId] = useState<string | null>(null);
  const [showEnvVarsModal, setShowEnvVarsModal] = useState(false);
  const [envVars, setEnvVars] = useState<UserSpaceWorkspaceEnvVar[]>([]);
  const [envVarsLoading, setEnvVarsLoading] = useState(false);
  const [objectStorageConfig, setObjectStorageConfig] = useState<UserSpaceObjectStorageConfig | null>(null);
  const [objectStorageLoading, setObjectStorageLoading] = useState(false);
  const [showObjectStorageWizard, setShowObjectStorageWizard] = useState(false);
  const [editingObjectStorageBucket, setEditingObjectStorageBucket] = useState<UserSpaceObjectStorageBucket | null>(null);
  const [deletingObjectStorageBucket, setDeletingObjectStorageBucket] = useState<string | null>(null);
  const [savingEnvVar, setSavingEnvVar] = useState(false);
  const [showMountsModal, setShowMountsModal] = useState(false);
  const [mountsModalTab, setMountsModalTab] = useState<'mounts' | 'object-storage'>('mounts');
  const [showScmWizard, setShowScmWizard] = useState(false);
  const [mounts, setMounts] = useState<WorkspaceMount[]>([]);
  const [mountsLoading, setMountsLoading] = useState(false);
  const [mountableSources, setMountableSources] = useState<MountableSource[]>([]);
  const [createMountSourceId, setCreateMountSourceId] = useState('');
  const [createMountSourcePath, setCreateMountSourcePath] = useState('');
  const [createMountRootSourcePath, setCreateMountRootSourcePath] = useState('');
  const [createMountBrowserPath, setCreateMountBrowserPath] = useState('');
  const [createMountTargetBrowserPath, setCreateMountTargetBrowserPath] = useState('');
  const [createMountTargetPath, setCreateMountTargetPath] = useState('');
  const [createMountStagedSourceDirectories, setCreateMountStagedSourceDirectories] = useState<Record<string, string[]>>({});
  const [createMountStagedTargetDirectories, setCreateMountStagedTargetDirectories] = useState<string[]>([]);
  const [createMountDescription, setCreateMountDescription] = useState('');
  const [createMountSyncMode, setCreateMountSyncMode] = useState<WorkspaceMountSyncMode>('merge');
  const [createMountActiveSourceTab, setCreateMountActiveSourceTab] = useState('');
  const [savingMount, setSavingMount] = useState(false);
  const [deletingMountId, setDeletingMountId] = useState<string | null>(null);

  const [syncingMountId, setSyncingMountId] = useState<string | null>(null);
  const [previewingMountId, setPreviewingMountId] = useState<string | null>(null);
  const [mountSyncPreview, setMountSyncPreview] = useState<WorkspaceMountSyncPreviewResponse | null>(null);
  const [mountSyncPreviewMount, setMountSyncPreviewMount] = useState<WorkspaceMount | null>(null);
  const [mountSyncPreviewIntent, setMountSyncPreviewIntent] = useState<MountSyncPreviewIntent | null>(null);
  const [mountSyncPreviewNextSyncMode, setMountSyncPreviewNextSyncMode] = useState<WorkspaceMountSyncMode | null>(null);
  const [expandedSyncModeInfo, setExpandedSyncModeInfo] = useState<false | 'hover' | 'pinned'>(false);
  const [savingMountWatchId, setSavingMountWatchId] = useState<string | null>(null);
  const [editingMountDescriptionId, setEditingMountDescriptionId] = useState<string | null>(null);
  const [editingMountDescriptionDraft, setEditingMountDescriptionDraft] = useState('');
  const [savingMountDescriptionId, setSavingMountDescriptionId] = useState<string | null>(null);
  const [showToolPicker, setShowToolPicker] = useState(false);
  const [showSnapshots, setShowSnapshots] = useState(false);
  const [showStaleBranches, setShowStaleBranches] = useState(false);
  const [snapshotsLoadedForWorkspace, setSnapshotsLoadedForWorkspace] = useState<string | null>(null);
  const [expandedSnapshotIds, setExpandedSnapshotIds] = useState<Set<string>>(new Set());
  const [snapshotDiffSummaries, setSnapshotDiffSummaries] = useState<Record<string, UserSpaceSnapshotDiffSummary>>({});
  const snapshotDiffSummariesRef = useRef(snapshotDiffSummaries);
  snapshotDiffSummariesRef.current = snapshotDiffSummaries;

  const {
    refresh: refreshAvailableModels,
    awaitReady: awaitAvailableModelsReady,
  } = useAvailableModels();
  const [loadingSnapshotDiffSummaryIds, setLoadingSnapshotDiffSummaryIds] = useState<Record<string, boolean>>({});
  const [snapshotDiffSummaryErrors, setSnapshotDiffSummaryErrors] = useState<Record<string, string>>({});
  const snapshotFileDiffCacheRef = useRef<Map<string, UserSpaceSnapshotFileDiff>>(new Map());
  const [activeSnapshotFileDiff, setActiveSnapshotFileDiff] = useState<UserSpaceSnapshotFileDiff | null>(null);
  const [activeSnapshotFileDiffKey, setActiveSnapshotFileDiffKey] = useState<string | null>(null);
  const [activeSnapshotFileDiffLoading, setActiveSnapshotFileDiffLoading] = useState(false);
  const [activeSnapshotFileDiffError, setActiveSnapshotFileDiffError] = useState<string | null>(null);
  const [activeSnapshotFileDiffTitle, setActiveSnapshotFileDiffTitle] = useState('Snapshot Diff');
  const [activeSnapshotFileDiffBeforeLabel, setActiveSnapshotFileDiffBeforeLabel] = useState('Snapshot');
  const [activeSnapshotFileDiffAfterLabel, setActiveSnapshotFileDiffAfterLabel] = useState('Current Workspace');

  const toolPickerRef = useRef<HTMLDivElement>(null);
  const workspaceDropdownRef = useRef<HTMLDivElement>(null);
  const selectedFilePathRef = useRef(selectedFilePath);
  const fileContentCacheRef = useRef(fileContentCache);
  const activeWorkspaceIdRef = useRef<string | null>(activeWorkspaceId);
  const activeWorkspaceConversationIdRef = useRef<string | null>(activeWorkspaceConversationId);
  const loadWorkspaceDataRequestIdRef = useRef(0);
  const loadChangedFileStateRequestIdRef = useRef(0);
  const loadRuntimeStatusRequestIdRef = useRef(0);
  const previewLaunchRequestIdRef = useRef(0);
  const snapshotDiffSummaryRequestIdsRef = useRef<Record<string, number>>({});
  const snapshotFileDiffRequestIdRef = useRef(0);
  const fileDirtyRef = useRef(false);
  const changedFileStateInFlightRef = useRef(false);
  const changedFileStateLastStartedAtRef = useRef(0);
  const changedFileStatePendingWorkspaceIdRef = useRef<string | null>(null);
  const changedFileStateGuardTimerRef = useRef<number | null>(null);
  const collabSocketRef = useRef<WebSocket | null>(null);
  const collabReconnectTimerRef = useRef<number | null>(null);
  const collabReconnectAttemptsRef = useRef(0);
  const collabSuppressNextSendRef = useRef(false);
  const workspaceEventsReconnectTimerRef = useRef<number | null>(null);
  const loadWorkspaceDataDebounceRef = useRef<number | null>(null);
  const loadChangedFileStateDebounceRef = useRef<number | null>(null);
  const refreshRuntimeStatusInflightRef = useRef(false);
  const fileBrowserEntriesRef = useRef<UserSpaceFileInfo[]>([]);
  const terminalSocketRef = useRef<WebSocket | null>(null);
  const terminalReconnectTimerRef = useRef<number | null>(null);
  const terminalReadOnlyRef = useRef(false);
  const terminalContainerRef = useRef<HTMLDivElement | null>(null);
  const terminalRef = useRef<XTerm | null>(null);
  const terminalFitRef = useRef<FitAddon | null>(null);
  const terminalResizeObserverRef = useRef<ResizeObserver | null>(null);
  const browserSurfaceAuthExpiryRef = useRef<Partial<Record<UserSpaceBrowserSurface, number>>>({});
  const previewLaunchExpiresAtMsRef = useRef(0);
  const previousRuntimeDisplayStateRef = useRef<string | null>(null);
  const statusOverlayDismissedSignatureRef = useRef<string | null>(null);
  const refreshRuntimeStatusPendingRef = useRef(false);
  const latestQueuedWorkspaceCreateTaskIdRef = useRef<string | null>(null);
  const workspacesRef = useRef(workspaces);
  const lastWorkspaceCookieNameRef = useRef('');
  const lastWorkspaceCookieName = useMemo(() => getLastWorkspaceCookieName(currentUser.id), [currentUser.id]);
  const userSpaceLayoutCookieName = useMemo(() => getUserSpaceLayoutCookieName(currentUser.id), [currentUser.id]);
  const userSpaceFullscreenCookieName = useMemo(() => getUserSpaceFullscreenCookieName(currentUser.id), [currentUser.id]);

  const authorizeBrowserSurfaces = useCallback(async (
    workspaceId: string,
    surfaces: UserSpaceBrowserSurface[],
    options?: { force?: boolean },
  ) => {
    const force = options?.force === true;
    const uniqueSurfaces = Array.from(new Set(surfaces));
    const now = Date.now();
    const surfacesToAuthorize = force
      ? uniqueSurfaces
      : uniqueSurfaces.filter((surface) => {
        const expiresAtMs = browserSurfaceAuthExpiryRef.current[surface] ?? 0;
        return expiresAtMs <= (now + USERSPACE_BROWSER_AUTH_REFRESH_LEAD_MS);
      });

    if (surfacesToAuthorize.length === 0) {
      return;
    }

    const response = await api.authorizeUserSpaceBrowserSurfaces(workspaceId, surfacesToAuthorize);
    for (const authorization of response.authorizations) {
      browserSurfaceAuthExpiryRef.current[authorization.surface] = parseUtcTimestampMs(authorization.expires_at);
    }
  }, []);

  const launchPreviewSurface = useCallback(async (
    workspaceId: string,
    options?: { clearOnError?: boolean; updateFrameUrl?: boolean },
  ): Promise<string> => {
    const requestId = ++previewLaunchRequestIdRef.current;
    const clearOnError = options?.clearOnError !== false;
    // updateFrameUrl=false is used by routine session-warming refreshes so we
    // mint a fresh bootstrap grant (preview cookie) without changing the
    // iframe `src`/key, which would remount the iframe and destroy any
    // in-flight live-data executions inside the running workspace app.
    const updateFrameUrl = options?.updateFrameUrl !== false;
    const isCurrentRequest = () => (
      requestId === previewLaunchRequestIdRef.current
      && activeWorkspaceIdRef.current === workspaceId
    );

    if (isCurrentRequest() && updateFrameUrl) {
      setPreviewAuthorizationPending(true);
    }

    try {
      const response = await api.launchUserSpacePreview(workspaceId, {
        path: '/',
        parent_origin: window.location.origin,
      });
      if (!isCurrentRequest()) {
        return '';
      }
      previewLaunchExpiresAtMsRef.current = parseUtcTimestampMs(response.expires_at);
      if (updateFrameUrl) {
        setPreviewFrameUrl(response.preview_url);
        setPreviewOrigin(response.preview_origin);
      }
      onPreviewWarningChange?.(response.preview_warning ?? null);
      return response.preview_url;
    } catch (err) {
      previewLaunchExpiresAtMsRef.current = 0;
      if (clearOnError && updateFrameUrl && isCurrentRequest()) {
        setPreviewFrameUrl(null);
        setPreviewOrigin(null);
      }
      throw err;
    } finally {
      if (isCurrentRequest() && updateFrameUrl) {
        setPreviewAuthorizationPending(false);
      }
    }
  }, [onPreviewWarningChange]);

  // Resize state
  const [sidebarWidth, setSidebarWidth] = useState(180);
  const [leftPaneFraction, setLeftPaneFraction] = useState(0.5);
  const [editorFraction, setEditorFraction] = useState(0.6);
  const contentRef = useRef<HTMLDivElement>(null);
  const leftPaneRef = useRef<HTMLDivElement>(null);
  const codeEditorRef = useRef<HTMLDivElement>(null);

  // Collapse state: track which panes are collapsed + their last size for restore
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rightPaneCollapsed, setRightPaneCollapsed] = useState(false);
  const [editorChatCollapsedSide, setEditorChatCollapsedSide] = useState<'before' | 'after' | null>(null);
  const prevSidebarWidth = useRef(180);
  const prevLeftPaneFraction = useRef(0.5);
  const prevEditorFraction = useRef(0.6);
  const skipNextLayoutPersistRef = useRef(true);
  const skipNextFullscreenPersistRef = useRef(true);

  const SIDEBAR_COLLAPSE_THRESHOLD = 60;
  const MAIN_COLLAPSE_LEFT_THRESHOLD = 0.08;
  const MAIN_COLLAPSE_RIGHT_THRESHOLD = 0.92;
  const EDITOR_COLLAPSE_TOP_THRESHOLD = 0.08;
  const EDITOR_COLLAPSE_BOTTOM_THRESHOLD = 0.92;

  const handleResizeSidebar = useCallback((delta: number) => {
    setSidebarWidth((prev) => {
      const next = prev + delta;
      if (next < SIDEBAR_COLLAPSE_THRESHOLD) {
        prevSidebarWidth.current = prev > SIDEBAR_COLLAPSE_THRESHOLD ? prev : prevSidebarWidth.current;
        setSidebarCollapsed(true);
        return 0;
      }
      setSidebarCollapsed(false);
      return Math.min(400, next);
    });
  }, []);

  const handleResizeMainSplit = useCallback((delta: number) => {
    const el = contentRef.current;
    if (!el) return;
    const totalWidth = el.offsetWidth;
    if (totalWidth === 0) return;
    setLeftPaneFraction((prev) => {
      const next = prev + delta / totalWidth;
      if (next > MAIN_COLLAPSE_RIGHT_THRESHOLD) {
        prevLeftPaneFraction.current = prev < MAIN_COLLAPSE_RIGHT_THRESHOLD ? prev : prevLeftPaneFraction.current;
        setRightPaneCollapsed(true);
        return 1;
      }
      if (next < MAIN_COLLAPSE_LEFT_THRESHOLD) {
        prevLeftPaneFraction.current = prev > MAIN_COLLAPSE_LEFT_THRESHOLD ? prev : prevLeftPaneFraction.current;
        // Left pane collapse not implemented (keep at min)
        return 0.05;
      }
      setRightPaneCollapsed(false);
      return next;
    });
  }, []);

  const handleResizeEditorChat = useCallback((delta: number) => {
    const el = leftPaneRef.current;
    if (!el) return;
    const totalHeight = el.offsetHeight;
    if (totalHeight === 0) return;
    setEditorFraction((prev) => {
      const next = prev + delta / totalHeight;
      if (next < EDITOR_COLLAPSE_TOP_THRESHOLD) {
        prevEditorFraction.current = prev > EDITOR_COLLAPSE_TOP_THRESHOLD ? prev : prevEditorFraction.current;
        setEditorChatCollapsedSide('before');
        return 0;
      }
      if (next > EDITOR_COLLAPSE_BOTTOM_THRESHOLD) {
        prevEditorFraction.current = prev < EDITOR_COLLAPSE_BOTTOM_THRESHOLD ? prev : prevEditorFraction.current;
        setEditorChatCollapsedSide('after');
        return 1;
      }
      setEditorChatCollapsedSide(null);
      return Math.min(0.9, Math.max(0.1, next));
    });
  }, []);

  const expandSidebar = useCallback(() => {
    setSidebarCollapsed(false);
    setSidebarWidth(prevSidebarWidth.current || 180);
  }, []);

  const expandRightPane = useCallback(() => {
    setRightPaneCollapsed(false);
    setLeftPaneFraction(prevLeftPaneFraction.current || 0.5);
  }, []);

  const expandChat = useCallback(() => {
    setEditorChatCollapsedSide(null);
    const restored = prevEditorFraction.current || 0.6;
    setEditorFraction(Math.min(0.9, Math.max(0.1, restored)));
  }, []);

  const isCodeEditorFocused = useCallback(() => {
    const activeElement = document.activeElement;
    if (!activeElement || !codeEditorRef.current) {
      return false;
    }
    return codeEditorRef.current.contains(activeElement);
  }, []);

  useEffect(() => {
    skipNextLayoutPersistRef.current = true;
    const stored = readStoredUserSpaceLayout(userSpaceLayoutCookieName);

    if (!stored) {
      setSidebarWidth(180);
      setSidebarCollapsed(false);
      setLeftPaneFraction(0.5);
      setRightPaneCollapsed(false);
      setEditorFraction(0.6);
      setEditorChatCollapsedSide(null);
      prevSidebarWidth.current = 180;
      prevLeftPaneFraction.current = 0.5;
      prevEditorFraction.current = 0.6;
      return;
    }

    const restoredSidebarWidth = clampNumber(stored.sidebarWidth, SIDEBAR_COLLAPSE_THRESHOLD, 400);
    const restoredLeftPaneFraction = clampNumber(stored.leftPaneFraction, 0.1, 0.9);
    const restoredEditorFraction = clampNumber(stored.editorFraction, 0.1, 0.9);

    setSidebarCollapsed(stored.sidebarCollapsed);
    setSidebarWidth(stored.sidebarCollapsed ? 0 : restoredSidebarWidth);
    setRightPaneCollapsed(stored.rightPaneCollapsed);
    setLeftPaneFraction(stored.rightPaneCollapsed ? 1 : restoredLeftPaneFraction);
    setEditorChatCollapsedSide(stored.editorChatCollapsedSide);

    if (stored.editorChatCollapsedSide === 'before') {
      setEditorFraction(0);
    } else if (stored.editorChatCollapsedSide === 'after') {
      setEditorFraction(1);
    } else {
      setEditorFraction(restoredEditorFraction);
    }

    prevSidebarWidth.current = restoredSidebarWidth;
    prevLeftPaneFraction.current = restoredLeftPaneFraction;
    prevEditorFraction.current = restoredEditorFraction;
  }, [SIDEBAR_COLLAPSE_THRESHOLD, userSpaceLayoutCookieName]);

  useEffect(() => {
    if (skipNextLayoutPersistRef.current) {
      skipNextLayoutPersistRef.current = false;
      return;
    }

    const payload: StoredUserSpaceLayout = {
      sidebarWidth: sidebarCollapsed
        ? clampNumber(prevSidebarWidth.current || 180, SIDEBAR_COLLAPSE_THRESHOLD, 400)
        : clampNumber(sidebarWidth, SIDEBAR_COLLAPSE_THRESHOLD, 400),
      sidebarCollapsed,
      leftPaneFraction: rightPaneCollapsed
        ? clampNumber(prevLeftPaneFraction.current || 0.5, 0.1, 0.9)
        : clampNumber(leftPaneFraction, 0.1, 0.9),
      rightPaneCollapsed,
      editorFraction: editorChatCollapsedSide
        ? clampNumber(prevEditorFraction.current || 0.6, 0.1, 0.9)
        : clampNumber(editorFraction, 0.1, 0.9),
      editorChatCollapsedSide,
    };

    setSessionCookieValue(userSpaceLayoutCookieName, JSON.stringify(payload));
  }, [
    SIDEBAR_COLLAPSE_THRESHOLD,
    editorChatCollapsedSide,
    editorFraction,
    leftPaneFraction,
    rightPaneCollapsed,
    sidebarCollapsed,
    sidebarWidth,
    userSpaceLayoutCookieName,
  ]);

  useEffect(() => {
    skipNextFullscreenPersistRef.current = true;
    const stored = readStoredUserSpaceFullscreen(userSpaceFullscreenCookieName);
    setIsFullscreen(stored);
    onFullscreenChange?.(stored);
  }, [onFullscreenChange, userSpaceFullscreenCookieName]);

  useEffect(() => {
    if (skipNextFullscreenPersistRef.current) {
      skipNextFullscreenPersistRef.current = false;
      return;
    }
    setSessionCookieValue(userSpaceFullscreenCookieName, isFullscreen ? '1' : '0');
  }, [isFullscreen, userSpaceFullscreenCookieName]);
  const [editingWorkspaceNameId, setEditingWorkspaceNameId] = useState<string | null>(null);
  const [workspaceNameDraft, setWorkspaceNameDraft] = useState('');

  const activeWorkspace = useMemo(
    () => workspaces.find((workspace) => workspace.id === activeWorkspaceId) ?? null,
    [workspaces, activeWorkspaceId]
  );
  const activeWorkspaceChatState = useMemo(
    () => (activeWorkspaceId ? (workspaceChatStates[activeWorkspaceId] ?? DEFAULT_WORKSPACE_CHAT_STATE) : DEFAULT_WORKSPACE_CHAT_STATE),
    [activeWorkspaceId, workspaceChatStates]
  );

  const previewWorkspaceFiles = useMemo(() => {
    const modules: Record<string, string> = {};
    for (const file of files) {
      const cached = fileContentCache[file.path];
      if (cached) {
        modules[file.path] = cached.content;
      }
    }
    if (selectedFilePath) {
      modules[selectedFilePath] = fileContent;
    }
    return modules;
  }, [fileContent, fileContentCache, files, selectedFilePath]);

  const resolvedSelectedToolIds = useMemo(
    () => activeWorkspace?.selected_tool_ids ?? [],
    [activeWorkspace?.selected_tool_ids]
  );
  const selectedToolIds = useMemo(
    () => new Set(resolvedSelectedToolIds),
    [resolvedSelectedToolIds]
  );
  const resolvedSelectedToolGroupIds = useMemo(
    () => activeWorkspace?.selected_tool_group_ids ?? [],
    [activeWorkspace?.selected_tool_group_ids]
  );
  const selectedToolGroupIds = useMemo(
    () => new Set(resolvedSelectedToolGroupIds),
    [resolvedSelectedToolGroupIds]
  );
  const fileTree = useMemo(() => buildUserSpaceTree(fileBrowserEntries), [fileBrowserEntries]);
  const folderPaths = useMemo(() => listFolderPaths(fileBrowserEntries), [fileBrowserEntries]);

  /** Repo-relative paths that are mount targets (e.g. "test" for target_path "/workspace/test"). */
  const mountTargetPaths = useMemo(() => {
    const paths = new Map<
      string,
      { id: string; enabled: boolean; sourceType: WorkspaceMount['source_type']; syncStatus: WorkspaceMount['sync_status']; lastSyncError: string | null; sourceAvailable: boolean }
    >();
    for (const mount of mounts) {
      const target = mount.target_path?.replace(/^\/workspace\//, '')?.replace(/^\/+|\/+$/g, '');
      if (target) {
        paths.set(target, {
          id: mount.id,
          enabled: mount.enabled,
          sourceType: mount.source_type,
          syncStatus: mount.sync_status,
          lastSyncError: mount.last_sync_error,
          sourceAvailable: mount.source_available,
        });
      }
    }
    return paths;
  }, [mounts]);
  const workspaceMountTargetBrowserCacheKey = useMemo(
    () => `${activeWorkspaceId ?? 'no-workspace'}:${fileEntriesFingerprint(fileBrowserEntries)}`,
    [activeWorkspaceId, fileBrowserEntries]
  );

  // Files that are changed but not yet acknowledged via per-file Save.
  const changedFilePaths = useMemo(() => {
    const paths = new Set<string>();
    for (const path of changedFiles) {
      if (!acknowledgedFiles.has(path)) {
        paths.add(path);
      }
    }
    return paths;
  }, [changedFiles, acknowledgedFiles]);

  const activeWorkspaceRole = useMemo(() => {
    if (!activeWorkspace) return 'viewer';
    if (activeWorkspace.owner_user_id === currentUser.id) return 'owner';
    if (currentUser.role === 'admin') return 'owner';
    return activeWorkspace.members.find((member) => member.user_id === currentUser.id)?.role ?? 'viewer';
  }, [activeWorkspace, currentUser.id, currentUser.role]);

  const canEditWorkspace = activeWorkspaceRole === 'owner' || activeWorkspaceRole === 'editor';
  const isOwner = activeWorkspaceRole === 'owner';
  const isAdminImpersonating = currentUser.role === 'admin' && activeWorkspace != null && activeWorkspace.owner_user_id !== currentUser.id;
  const workspaceChatShareableUserIds = useMemo(() => {
    if (!activeWorkspace) return [];
    return Array.from(new Set([
      activeWorkspace.owner_user_id,
      ...activeWorkspace.members.map((member) => member.user_id),
    ]));
  }, [activeWorkspace]);

  const snapshotsByBranch = useMemo(() => {
    const grouped = new Map<string, UserSpaceSnapshot[]>();
    for (const snapshot of snapshots) {
      const current = grouped.get(snapshot.branch_id) ?? [];
      current.push(snapshot);
      grouped.set(snapshot.branch_id, current);
    }
    for (const list of grouped.values()) {
      list.sort((a, b) => parseUtcTimestampMs(b.created_at) - parseUtcTimestampMs(a.created_at));
    }

    return snapshotBranches
      .map((branch) => ({
        branch,
        snapshots: grouped.get(branch.id) ?? [],
      }))
      .filter((group) => {
        // Always keep Main visible
        if (group.branch.name === 'Main') return true;
        // Always keep the current active branch
        if (group.branch.id === currentSnapshotBranchId) return true;
        // Hide empty branches
        if (group.snapshots.length === 0) return false;
        // Hide stale branches (shown separately in collapsed section)
        if (group.branch.is_stale) return false;
        return true;
      })
      .sort((a, b) => {
        // Main always first
        if (a.branch.name === 'Main') return -1;
        if (b.branch.name === 'Main') return 1;
        return 0;
      });
  }, [snapshotBranches, snapshots, currentSnapshotBranchId]);

  const staleBranches = useMemo(() => {
    const visibleIds = new Set(snapshotsByBranch.map(({ branch }) => branch.id));
    return snapshotBranches.filter(
      (branch) => branch.is_stale && !visibleIds.has(branch.id)
    );
  }, [snapshotBranches, snapshotsByBranch]);

  const snapshotTimelineRows = useMemo(() => {
    const sortedSnapshots = [...snapshots].sort((left, right) => {
      const timeDiff = parseUtcTimestampMs(right.created_at) - parseUtcTimestampMs(left.created_at);
      if (timeDiff !== 0) {
        return timeDiff;
      }
      return right.id.localeCompare(left.id);
    });

    const branchIndexById = new Map<string, number>();
    const branchById = new Map<string, UserSpaceSnapshotBranch>();
    const snapshotById = new Map<string, UserSpaceSnapshot>();
    const rowIndexBySnapshotId = new Map<string, number>();

    sortedSnapshots.forEach((snapshot, index) => {
      snapshotById.set(snapshot.id, snapshot);
      rowIndexBySnapshotId.set(snapshot.id, index);
    });

    snapshotsByBranch.forEach(({ branch }, index) => {
      branchIndexById.set(branch.id, index);
      branchById.set(branch.id, branch);
    });

    const laneMetaByBranchId = new Map<string, {
      laneIndex: number;
      startRow: number;
      endRow: number;
      forkRow: number | null;
      forkFromLaneIndex: number | null;
    }>();

    snapshotsByBranch.forEach(({ branch, snapshots: branchSnapshots }, index) => {
      const branchRows = branchSnapshots
        .map((snapshot) => rowIndexBySnapshotId.get(snapshot.id))
        .filter((rowIndex): rowIndex is number => typeof rowIndex === 'number')
        .sort((left, right) => left - right);

      if (branchRows.length === 0) {
        return;
      }

      let startRow = branchRows[0];
      let endRow = branchRows[branchRows.length - 1];
      let forkRow: number | null = null;
      let forkFromLaneIndex: number | null = null;

      if (branch.branched_from_snapshot_id) {
        const parentRow = rowIndexBySnapshotId.get(branch.branched_from_snapshot_id);
        const parentSnapshot = snapshotById.get(branch.branched_from_snapshot_id);
        const parentLaneIndex = parentSnapshot ? branchIndexById.get(parentSnapshot.branch_id) : undefined;
        if (typeof parentRow === 'number') {
          endRow = Math.max(endRow, parentRow);
          forkRow = parentRow;
        }
        if (typeof parentLaneIndex === 'number' && parentLaneIndex !== index) {
          forkFromLaneIndex = parentLaneIndex;
        }
      }

      laneMetaByBranchId.set(branch.id, {
        laneIndex: index,
        startRow,
        endRow,
        forkRow,
        forkFromLaneIndex,
      });
    });

    return sortedSnapshots.map((snapshot, rowIndex) => {
      const laneIndex = branchIndexById.get(snapshot.branch_id) ?? 0;
      const laneStates = snapshotsByBranch.map(({ branch }, branchIndex) => {
        const meta = laneMetaByBranchId.get(branch.id);
        const isActive = !!meta && rowIndex >= meta.startRow && rowIndex <= meta.endRow;
        return {
          branchId: branch.id,
          branchIndex,
          isActive,
          isStart: isActive && meta?.startRow === rowIndex,
          isEnd: isActive && meta?.endRow === rowIndex,
        };
      });

      const forkLinks = snapshotsByBranch
        .map(({ branch }) => {
          const meta = laneMetaByBranchId.get(branch.id);
          if (!meta || meta.forkRow !== rowIndex || meta.forkFromLaneIndex === null) {
            return null;
          }
          return {
            branchId: branch.id,
            fromLaneIndex: Math.min(meta.forkFromLaneIndex, meta.laneIndex),
            toLaneIndex: Math.max(meta.forkFromLaneIndex, meta.laneIndex),
          };
        })
        .filter((value): value is { branchId: string; fromLaneIndex: number; toLaneIndex: number } => value !== null);

      return {
        snapshot,
        laneIndex,
        laneStates,
        forkLinks,
      };
    });
  }, [snapshots, snapshotsByBranch]);

  const snapshotBranchColorById = useMemo(() => {
    const colors = new Map<string, string>();
    for (const { branch } of snapshotsByBranch) {
      colors.set(branch.id, getDeterministicBranchColor(branch.id));
    }
    return colors;
  }, [snapshotsByBranch]);

  const snapshotUiLocked = navigatingSnapshots || restoringSnapshotId !== null;
  const codeMirrorLanguageExtension = useCodeMirrorLanguageExtension(selectedFilePath);

  const codeMirrorExtensions = useMemo(
    () => {
      const extensions = [
        keymap.of([
          {
            key: 'Mod-f',
            run: openSearchPanel,
            preventDefault: true,
          },
        ]),
      ];
      if (codeMirrorLanguageExtension) {
        extensions.unshift(codeMirrorLanguageExtension);
      }
      return extensions;
    },
    [codeMirrorLanguageExtension]
  );

  const selectedFileDisplayName = useMemo(() => {
    const parts = selectedFilePath.split('/').filter(Boolean);
    return parts[parts.length - 1] ?? selectedFilePath;
  }, [selectedFilePath]);

  // Derive effective runtime display state from session_state + devserver_running
  const runtimeDisplayState = useMemo(() => {
    if (!runtimeStatus) return 'stopped';
    const { session_state, devserver_running, last_error } = runtimeStatus;
    if (session_state === 'running' && !devserver_running) {
      return last_error ? 'error' : 'starting';
    }
    return session_state;
  }, [runtimeStatus]);

  const runtimeCapSysAdminMissing = runtimeStatus?.runtime_has_cap_sys_admin === false;
  const runtimeOperationLabel = useMemo(() => {
    const phase = runtimeStatus?.runtime_operation_phase;
    if (!phase) return null;
    if (phase === 'provisioning') return 'preparing sandbox';
    if (phase === 'deps_install') return 'installing deps';
    return phase.replace(/_/g, ' ');
  }, [runtimeStatus?.runtime_operation_phase]);

  const runtimeOverlayStatus = useMemo(() => {
    if (runtimeDisplayState === 'starting') {
      return `Starting runtime${runtimeOperationLabel ? ` (${runtimeOperationLabel})` : ''}...`;
    }
    if (runtimeDisplayState === 'stopping') {
      return 'Stopping runtime...';
    }
    return null;
  }, [runtimeDisplayState, runtimeOperationLabel]);

  const activeWorkspaceCreateTasks = useMemo(
    () => Object.values(creatingWorkspaceTasks)
      .filter((task) => !isWorkspaceCreateTaskTerminal(task.phase))
      .sort((left, right) => Date.parse(left.queued_at) - Date.parse(right.queued_at)),
    [creatingWorkspaceTasks],
  );

  const creatingWorkspace = activeWorkspaceCreateTasks.length > 0;
  const creatingWorkspaceStatus = useMemo(
    () => formatWorkspaceCreateTasksStatus(activeWorkspaceCreateTasks),
    [activeWorkspaceCreateTasks],
  );

  const activeWorkspaceDeleteTasks = useMemo(
    () => Object.values(deletingWorkspaceTasks)
      .filter((task) => !isWorkspaceDeleteTaskTerminal(task.phase))
      .sort((left, right) => Date.parse(left.queued_at) - Date.parse(right.queued_at)),
    [deletingWorkspaceTasks],
  );

  const activeWorkspaceDuplicateTasks = useMemo(
    () => Object.values(duplicatingWorkspaceTasks)
      .filter((task) => !isWorkspaceDuplicateTaskTerminal(task.phase))
      .sort((left, right) => Date.parse(left.queued_at) - Date.parse(right.queued_at)),
    [duplicatingWorkspaceTasks],
  );

  const deletingWorkspaceId = activeWorkspaceDeleteTasks[0]?.workspace_id ?? null;
  const deletingWorkspaceStatus = useMemo(
    () => formatWorkspaceDeleteTasksStatus(activeWorkspaceDeleteTasks),
    [activeWorkspaceDeleteTasks],
  );
  const duplicatingWorkspaceSourceId = activeWorkspaceDuplicateTasks[0]?.source_workspace_id ?? null;
  const duplicatingWorkspaceStatus = useMemo(
    () => formatWorkspaceDuplicateTasksStatus(activeWorkspaceDuplicateTasks),
    [activeWorkspaceDuplicateTasks],
  );
  const activeWorkspaceDuplicateTaskBySourceId = useMemo(() => {
    const next: Record<string, UserSpaceWorkspaceDuplicateTask> = {};
    for (const task of activeWorkspaceDuplicateTasks) {
      if (!next[task.source_workspace_id]) {
        next[task.source_workspace_id] = task;
      }
    }
    return next;
  }, [activeWorkspaceDuplicateTasks]);

  const showStartRuntimeButton = runtimeDisplayState === 'stopped' || runtimeDisplayState === 'error';
  const showRestartRuntimeButton = runtimeDisplayState === 'running';
  const showStopRuntimeButton = runtimeDisplayState === 'running' || runtimeDisplayState === 'starting';

  const formatUserLabel = useCallback((user?: Pick<User, 'username' | 'display_name'> | null, fallbackId?: string) => {
    const username = user?.username?.trim() || fallbackId?.trim() || 'unknown';
    const displayName = user?.display_name?.trim();
    if (displayName && displayName !== username) {
      return `${displayName} (@${username})`;
    }
    return `@${username}`;
  }, []);

  const loadWorkspaces = useCallback(async (append = false) => {
    if (append) {
      setLoadingMore(true);
    } else {
      setLoading(true);
    }
    try {
      const offset = append ? workspaces.length : 0;
      const page = await api.listUserSpaceWorkspaces(offset, 50);
      if (append) {
        setWorkspaces((prev) => [
          ...prev,
          ...page.items.filter((workspace) => !prev.some((existing) => existing.id === workspace.id)),
        ]);
      } else {
        let nextItems = page.items;

        if (
          activeWorkspaceId
          && !nextItems.some((workspace) => workspace.id === activeWorkspaceId)
        ) {
          try {
            const activeWorkspace = await api.getUserSpaceWorkspace(activeWorkspaceId);
            nextItems = [activeWorkspace, ...nextItems.filter((workspace) => workspace.id !== activeWorkspace.id)];
          } catch {
            // Ignore fetch errors for active workspace backfill.
          }
        }

        setWorkspaces(nextItems);
        if (nextItems.length === 0) {
          setActiveWorkspaceId(null);
          clearCookieValue(lastWorkspaceCookieName);
        } else if (!activeWorkspaceId) {
          const lastWorkspaceId = getCookieValue(lastWorkspaceCookieName);
          const restorableWorkspaceId = lastWorkspaceId;
          const matchingWorkspace = restorableWorkspaceId
            ? nextItems.find((workspace) => workspace.id === restorableWorkspaceId)
            : null;

          if (matchingWorkspace) {
            setActiveWorkspaceId(matchingWorkspace.id);
          } else if (restorableWorkspaceId) {
            try {
              const workspace = await api.getUserSpaceWorkspace(restorableWorkspaceId);
              setWorkspaces((prev) => (
                prev.some((item) => item.id === workspace.id)
                  ? prev
                  : [workspace, ...prev]
              ));
              setActiveWorkspaceId(workspace.id);
            } catch {
              setActiveWorkspaceId(nextItems[0].id);
            }
          } else {
            setActiveWorkspaceId(nextItems[0].id);
          }
        }
      }
      setWorkspacesTotal(page.total);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load User Space workspaces');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [activeWorkspaceId, lastWorkspaceCookieName, workspaces.length]);

  useEffect(() => {
    if (!activeWorkspaceId) return;
    setCookieValue(lastWorkspaceCookieName, activeWorkspaceId);
  }, [activeWorkspaceId, lastWorkspaceCookieName]);

  useEffect(() => {
    setActiveWorkspaceConversationId(null);
    setActiveWorkspaceChatSnapshot(null);
  }, [activeWorkspaceId]);

  useEffect(() => {
    if (
      !branchRestoreSnapshotId
      || !activeWorkspaceId
      || snapshotsLoadedForWorkspace !== activeWorkspaceId
    ) {
      return;
    }
    if (!snapshots.some((snapshot) => snapshot.id === branchRestoreSnapshotId)) {
      setBranchRestoreSnapshotId(null);
    }
  }, [activeWorkspaceId, branchRestoreSnapshotId, snapshots, snapshotsLoadedForWorkspace]);

  const handleConversationStateChange = useCallback((hasLive: boolean, hasInterrupted: boolean) => {
    if (!activeWorkspaceId) return;
    // The active workspace should reflect the live chat state from ChatPanel
    // immediately. If a task is running, live state takes precedence over any
    // stale interrupted badge for this workspace.
    setWorkspaceChatStates((prev) => ({
      ...prev,
      [activeWorkspaceId]: {
        ...DEFAULT_WORKSPACE_CHAT_STATE,
        ...prev[activeWorkspaceId],
        hasInterrupted,
        hasLive,
      },
    }));
  }, [activeWorkspaceId]);

  useEffect(() => {
    if (workspaces.length === 0) {
      setWorkspaceChatStates({});
      return;
    }

    const validWorkspaceIds = new Set(workspaces.map((workspace) => workspace.id));
    setWorkspaceChatStates((prev) => {
      let changed = false;
      const next: Record<string, WorkspaceChatState> = {};
      for (const [workspaceId, state] of Object.entries(prev)) {
        if (validWorkspaceIds.has(workspaceId)) {
          next[workspaceId] = state;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [workspaces]);

  // Track previous raw interrupted state per workspace so we can detect
  // false -> true transitions and clear stale dismiss cookies.
  const prevChatStateRef = useRef<Record<string, InterruptChatStateSnapshot>>({});

  useEffect(() => {
    if (workspaces.length === 0) return;

    let cancelled = false;
    let isPolling = false;

    const pollWorkspaceConversationStates = async () => {
      if (isPolling) return;
      isPolling = true;
      try {
        // Skip the active workspace — ChatPanel's own poller handles it
        // and reports state via onConversationStateChange.
        const workspaceIds = workspaces
          .map((workspace) => workspace.id)
          .filter((id) => id !== activeWorkspaceId);

        if (workspaceIds.length === 0) {
          isPolling = false;
          return;
        }

        const summaries = await api.getWorkspacesConversationStateSummary(workspaceIds);

        const updates = summaries.map((summary) => {
          const resolved = resolveWorkspaceInterruptStateFromSummary(
            currentUser.id,
            summary,
            prevChatStateRef.current[summary.workspace_id],
          );

          if (resolved.transition.shouldClearDismiss) {
            clearInterruptDismiss(currentUser.id, resolved.workspaceId);
          }
          prevChatStateRef.current[resolved.workspaceId] = resolved.transition.nextState;

          return [resolved.workspaceId, resolved.indicator] as const;
        });

        if (cancelled) return;

        setWorkspaceChatStates((prev) => {
          const next = { ...prev };
          for (const [workspaceId, state] of updates) {
            next[workspaceId] = state;
          }
          return next;
        });
      } catch {
        // Keep existing state if a poll cycle fails; the next interval retries.
      } finally {
        isPolling = false;
      }
    };

    let timer: number | null = null;

    const scheduleNextPoll = () => {
      if (cancelled) {
        return;
      }
      timer = window.setTimeout(() => {
        void pollWorkspaceConversationStates();
      }, isPageVisible ? 5000 : USERSPACE_WORKSPACE_BADGE_BACKGROUND_POLL_INTERVAL_MS);
    };

    const runPollingLoop = async () => {
      await pollWorkspaceConversationStates();
      scheduleNextPoll();
    };

    void runPollingLoop();

    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [workspaces, activeWorkspaceId, currentUser.id, isPageVisible]);

  const reconcileWorkspaceFileTree = useCallback((nextEntries: UserSpaceFileInfo[]) => {
    const nextFiles = nextEntries.filter((entry) => entry.entry_type !== 'directory');
    const currentSelectedPath = selectedFilePathRef.current;
    const selectedExists = nextFiles.some((file) => file.path === currentSelectedPath);
    const preferredPath = selectedExists
      ? currentSelectedPath
      : nextFiles.some((file) => file.path === previewEntryPath)
        ? previewEntryPath
        : nextFiles[0]?.path ?? previewEntryPath;
    const treeChanged = fileEntriesFingerprint(nextEntries) !== fileEntriesFingerprint(fileBrowserEntriesRef.current);

    if (treeChanged) {
      setFileBrowserEntries(nextEntries);
      setFiles(nextFiles);

      const validPaths = new Set(nextFiles.map((file) => file.path));
      setFileContentCache((current) => {
        const next: Record<string, CachedUserSpaceFile> = {};
        for (const [path, value] of Object.entries(current)) {
          if (validPaths.has(path)) {
            next[path] = value;
          }
        }
        return next;
      });
    }

    if (selectedFilePathRef.current !== preferredPath) {
      selectedFilePathRef.current = preferredPath;
      setSelectedFilePath(preferredPath);
    }

    return {
      changed: treeChanged,
      nextFiles,
      preferredPath,
    };
  }, [previewEntryPath]);

  const warmWorkspaceFileCache = useCallback(async (
    workspaceId: string,
    nextFiles: UserSpaceFileInfo[],
    options?: {
      excludePaths?: string[];
    },
  ) => {
    const excludedPaths = new Set(options?.excludePaths ?? []);
    const staleFiles = nextFiles.filter((file) => {
      if (excludedPaths.has(file.path)) {
        return false;
      }
      const cached = fileContentCacheRef.current[file.path];
      return !cached || cached.updatedAt !== (file.updated_at ?? '');
    });

    if (staleFiles.length === 0) {
      return;
    }

    const fetched = await Promise.all(
      staleFiles.map(async (file) => {
        try {
          const loaded = await api.getUserSpaceFile(workspaceId, file.path);
          return {
            path: loaded.path,
            content: loaded.content,
            updatedAt: file.updated_at ?? '',
            artifactType: loaded.artifact_type ?? null,
          };
        } catch (err) {
          if (getUnsupportedEditorFileMessage(err)) {
            return null;
          }
          throw err;
        }
      })
    );

    setFileContentCache((current) => {
      const next: Record<string, CachedUserSpaceFile> = { ...current };
      for (const file of fetched) {
        if (!file) {
          continue;
        }
        next[file.path] = {
          content: file.content,
          updatedAt: file.updatedAt,
          artifactType: file.artifactType,
        };
      }
      return next;
    });
  }, []);

  const refreshWorkspaceFileTree = useCallback(async (
    workspaceId: string,
  ) => {
    const nextEntries = await api.listUserSpaceFiles(workspaceId, {
      includeDirs: true,
    });

    const { changed } = reconcileWorkspaceFileTree(nextEntries);
    setError(null);
    return changed;
  }, [reconcileWorkspaceFileTree]);

  const loadWorkspaceData = useCallback(async (
    workspaceId: string,
    options?: {
      skipSelectedFileReload?: boolean;
    },
  ) => {
    const requestId = ++loadWorkspaceDataRequestIdRef.current;

    try {
      const nextEntries = await api.listUserSpaceFiles(workspaceId, {
        includeDirs: true,
      });

      if (requestId !== loadWorkspaceDataRequestIdRef.current) {
        return;
      }

      const { changed, nextFiles, preferredPath } = reconcileWorkspaceFileTree(nextEntries);

      if (!changed) {
        return;
      }

      const selectedExists = nextFiles.some((file) => file.path === preferredPath);
      const skipSelectedFileReload = Boolean(
        options?.skipSelectedFileReload
        && selectedExists
        && selectedFilePathRef.current === preferredPath,
      );

      await warmWorkspaceFileCache(workspaceId, nextFiles, {
        excludePaths: selectedExists ? [preferredPath] : [],
      });

      if (requestId !== loadWorkspaceDataRequestIdRef.current) {
        return;
      }

      if (selectedExists) {
        if (skipSelectedFileReload) {
          setError(null);
          return;
        }

        const preferredMeta = nextFiles.find((file) => file.path === preferredPath);
        const preferredUpdatedAt = preferredMeta?.updated_at ?? '';
        const cached = fileContentCacheRef.current[preferredPath];

        if (cached && cached.updatedAt === preferredUpdatedAt) {
          if (requestId !== loadWorkspaceDataRequestIdRef.current || selectedFilePathRef.current !== preferredPath) {
            return;
          }
          setFileContent(cached.content);
          setSelectedFileArtifactType(cached.artifactType ?? null);
          setSelectedFileUnsupportedMessage(null);
        } else {
          try {
            const file = await api.getUserSpaceFile(workspaceId, preferredPath);

            if (requestId !== loadWorkspaceDataRequestIdRef.current || selectedFilePathRef.current !== preferredPath) {
              return;
            }

            setFileContent(file.content);
            setSelectedFileArtifactType(file.artifact_type ?? null);
            setFileContentCache((current) => ({
              ...current,
              [file.path]: {
                content: file.content,
                updatedAt: preferredUpdatedAt,
                artifactType: file.artifact_type ?? null,
              },
            }));
            setSelectedFileUnsupportedMessage(null);
          } catch (err) {
            if (requestId !== loadWorkspaceDataRequestIdRef.current || selectedFilePathRef.current !== preferredPath) {
              return;
            }

            const unsupportedMessage = getUnsupportedEditorFileMessage(err);
            if (!unsupportedMessage) {
              throw err;
            }

            setFileContent('');
            setSelectedFileArtifactType(null);
            setSelectedFileUnsupportedMessage(unsupportedMessage);
          }
        }
      } else {
        setFileContent('');
        setSelectedFileArtifactType(null);
        setSelectedFileUnsupportedMessage(null);
      }

      setFileDirty(false);
      setError(null);
    } catch (err) {
      if (requestId !== loadWorkspaceDataRequestIdRef.current) {
        return;
      }
      setError(err instanceof Error ? err.message : 'Failed to load workspace data');
    }
  }, [reconcileWorkspaceFileTree, warmWorkspaceFileCache]);

  const loadChangedFileState = useCallback(async (workspaceId: string) => {
    if (changedFileStateInFlightRef.current) {
      changedFileStatePendingWorkspaceIdRef.current = workspaceId;
      return;
    }

    const elapsedMs = Date.now() - changedFileStateLastStartedAtRef.current;
    if (elapsedMs < USERSPACE_CHANGED_FILE_STATE_MIN_INTERVAL_MS) {
      changedFileStatePendingWorkspaceIdRef.current = workspaceId;
      const waitMs = USERSPACE_CHANGED_FILE_STATE_MIN_INTERVAL_MS - elapsedMs;
      if (changedFileStateGuardTimerRef.current !== null) {
        window.clearTimeout(changedFileStateGuardTimerRef.current);
      }
      changedFileStateGuardTimerRef.current = window.setTimeout(() => {
        changedFileStateGuardTimerRef.current = null;
        const pendingWorkspaceId = changedFileStatePendingWorkspaceIdRef.current;
        changedFileStatePendingWorkspaceIdRef.current = null;
        if (pendingWorkspaceId) {
          void loadChangedFileState(pendingWorkspaceId);
        }
      }, waitMs);
      return;
    }

    changedFileStateInFlightRef.current = true;
    changedFileStateLastStartedAtRef.current = Date.now();
    const requestId = ++loadChangedFileStateRequestIdRef.current;

    try {
      const result = await api.getUserSpaceChangedFileState(workspaceId);
      if (requestId !== loadChangedFileStateRequestIdRef.current) {
        return;
      }

      setChangedFiles(new Set(result.changed_file_paths));
      setAcknowledgedFiles(new Set(result.acknowledged_changed_file_paths));
    } catch {
      // changedFile-state bootstrap failure should not break core editor UX.
    } finally {
      changedFileStateInFlightRef.current = false;

      const pendingWorkspaceId = changedFileStatePendingWorkspaceIdRef.current;
      if (pendingWorkspaceId) {
        changedFileStatePendingWorkspaceIdRef.current = null;
        const sinceStartMs = Date.now() - changedFileStateLastStartedAtRef.current;
        const waitMs = Math.max(0, USERSPACE_CHANGED_FILE_STATE_MIN_INTERVAL_MS - sinceStartMs);

        if (changedFileStateGuardTimerRef.current !== null) {
          window.clearTimeout(changedFileStateGuardTimerRef.current);
        }
        changedFileStateGuardTimerRef.current = window.setTimeout(() => {
          changedFileStateGuardTimerRef.current = null;
          void loadChangedFileState(pendingWorkspaceId);
        }, waitMs);
      }
    }
  }, []);

  const loadSnapshots = useCallback(async (workspaceId: string): Promise<UserSpaceSnapshotTimeline | null> => {
    try {
      const result = await api.getUserSpaceSnapshotTimeline(workspaceId);
      setSnapshots(result.snapshots);
      setSnapshotBranches(result.branches);
      setCurrentSnapshotId(result.current_snapshot_id ?? null);
      setCurrentSnapshotBranchId(result.current_branch_id ?? null);
      setSnapshotsLoadedForWorkspace(workspaceId);
      return result;
    } catch {
      // Snapshot list is non-critical; keep UI functional.
      return null;
    }
  }, []);

  const loadSnapshotDiffSummary = useCallback(async (workspaceId: string, snapshotId: string) => {
    if (snapshotDiffSummariesRef.current[snapshotId]) {
      return snapshotDiffSummariesRef.current[snapshotId];
    }

    const requestId = (snapshotDiffSummaryRequestIdsRef.current[snapshotId] ?? 0) + 1;
    snapshotDiffSummaryRequestIdsRef.current[snapshotId] = requestId;
    setLoadingSnapshotDiffSummaryIds((current) => ({ ...current, [snapshotId]: true }));
    setSnapshotDiffSummaryErrors((current) => {
      if (!(snapshotId in current)) {
        return current;
      }
      const next = { ...current };
      delete next[snapshotId];
      return next;
    });

    try {
      const summary = await api.getUserSpaceSnapshotDiffSummary(workspaceId, snapshotId);
      if (snapshotDiffSummaryRequestIdsRef.current[snapshotId] !== requestId) {
        return null;
      }
      setSnapshotDiffSummaries((current) => ({ ...current, [snapshotId]: summary }));
      return summary;
    } catch (err) {
      if (snapshotDiffSummaryRequestIdsRef.current[snapshotId] === requestId) {
        setSnapshotDiffSummaryErrors((current) => ({
          ...current,
          [snapshotId]: err instanceof Error ? err.message : 'Failed to load snapshot changes',
        }));
      }
      return null;
    } finally {
      if (snapshotDiffSummaryRequestIdsRef.current[snapshotId] === requestId) {
        setLoadingSnapshotDiffSummaryIds((current) => ({ ...current, [snapshotId]: false }));
      }
    }
  }, []);

  const clearFileDiffState = useCallback(() => {
    setActiveSnapshotFileDiffLoading(false);
    setActiveSnapshotFileDiffError(null);
    setActiveSnapshotFileDiff(null);
    setActiveSnapshotFileDiffKey(null);
    setActiveSnapshotFileDiffTitle('Snapshot Diff');
    setActiveSnapshotFileDiffBeforeLabel('Snapshot');
    setActiveSnapshotFileDiffAfterLabel('Current Workspace');
  }, []);

  const diffHover = useDiffHoverTimers({ onDismiss: clearFileDiffState });

  const loadSnapshotFileDiff = useCallback(async (workspaceId: string, snapshotId: string, filePath: string) => {
    const cacheKey = getSnapshotDiffFileKey(snapshotId, filePath);
    diffHover.cancelDismiss();

    const cached = snapshotFileDiffCacheRef.current.get(cacheKey);
    setActiveSnapshotFileDiffKey(cacheKey);
    setActiveSnapshotFileDiffError(null);
    if (cached) {
      setActiveSnapshotFileDiffLoading(false);
      setActiveSnapshotFileDiff(cached);
      return cached;
    }

    const requestId = ++snapshotFileDiffRequestIdRef.current;
    setActiveSnapshotFileDiffLoading(true);
    setActiveSnapshotFileDiff(null);

    try {
      const diff = await api.getUserSpaceSnapshotFileDiff(workspaceId, snapshotId, filePath);
      if (requestId !== snapshotFileDiffRequestIdRef.current) {
        return null;
      }
      const cache = snapshotFileDiffCacheRef.current;
      cache.delete(cacheKey);
      cache.set(cacheKey, diff);
      while (cache.size > SNAPSHOT_FILE_DIFF_CACHE_MAX_ENTRIES) {
        const oldest = cache.keys().next().value;
        if (oldest !== undefined) cache.delete(oldest);
        else break;
      }
      setActiveSnapshotFileDiff(diff);
      return diff;
    } catch (err) {
      if (requestId === snapshotFileDiffRequestIdRef.current) {
        setActiveSnapshotFileDiffError(err instanceof Error ? err.message : 'Failed to load file diff');
      }
      return null;
    } finally {
      if (requestId === snapshotFileDiffRequestIdRef.current) {
        setActiveSnapshotFileDiffLoading(false);
      }
    }
  }, [diffHover]);

  const handleToggleSnapshotExpanded = useCallback((snapshotId: string) => {
    setExpandedSnapshotIds((current) => {
      const next = new Set(current);
      if (next.has(snapshotId)) {
        next.delete(snapshotId);
      } else {
        next.add(snapshotId);
        if (activeWorkspaceId) {
          void loadSnapshotDiffSummary(activeWorkspaceId, snapshotId);
        }
      }
      return next;
    });
  }, [activeWorkspaceId, loadSnapshotDiffSummary]);

  const handleSnapshotFileHoverStart = useCallback((snapshotId: string, filePath: string) => {
    if (!activeWorkspaceId) return;
    diffHover.startHover(() => {
      setActiveSnapshotFileDiffTitle('Snapshot Diff');
      setActiveSnapshotFileDiffBeforeLabel('Snapshot');
      setActiveSnapshotFileDiffAfterLabel('Current Workspace');
      void loadSnapshotFileDiff(activeWorkspaceId, snapshotId, filePath);
    });
  }, [activeWorkspaceId, diffHover, loadSnapshotFileDiff]);

  const handleSnapshotFileHoverEnd = useCallback(() => {
    diffHover.endHover();
  }, [diffHover]);

  const handleTreeFileHoverStart = useCallback((filePath: string) => {
    if (!activeWorkspaceId) return;
    diffHover.startHover(() => {
      void (async () => {
        let snapshotId = currentSnapshotId;
        if (!snapshotId) {
          const timeline: UserSpaceSnapshotTimeline | null = await loadSnapshots(activeWorkspaceId);
          if (timeline) {
            snapshotId = timeline.current_snapshot_id ?? null;
          }
        }
        if (!snapshotId) {
          return;
        }
        setActiveSnapshotFileDiffTitle('Unsaved Changes');
        setActiveSnapshotFileDiffBeforeLabel('Last Snapshot');
        setActiveSnapshotFileDiffAfterLabel('Current');
        await loadSnapshotFileDiff(activeWorkspaceId, snapshotId, filePath);
      })();
    });
  }, [activeWorkspaceId, currentSnapshotId, diffHover, loadSnapshotFileDiff, loadSnapshots]);

  const handleTreeFileHoverEnd = useCallback(() => {
    diffHover.endHover();
  }, [diffHover]);

  const debouncedLoadWorkspaceData = useCallback((workspaceId: string) => {
    if (loadWorkspaceDataDebounceRef.current !== null) {
      window.clearTimeout(loadWorkspaceDataDebounceRef.current);
    }
    loadWorkspaceDataDebounceRef.current = window.setTimeout(() => {
      loadWorkspaceDataDebounceRef.current = null;
      void loadWorkspaceData(workspaceId);
    }, 300);
  }, [loadWorkspaceData]);

  const debouncedLoadChangedFileState = useCallback((workspaceId: string) => {
    if (loadChangedFileStateDebounceRef.current !== null) {
      window.clearTimeout(loadChangedFileStateDebounceRef.current);
    }
    loadChangedFileStateDebounceRef.current = window.setTimeout(() => {
      loadChangedFileStateDebounceRef.current = null;
      void loadChangedFileState(workspaceId);
    }, 300);
  }, [loadChangedFileState]);

  useEffect(() => {
    loadWorkspaces();
  }, [loadWorkspaces]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      setIsPageVisible(document.visibilityState !== 'hidden');
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  useEffect(() => {
    if (!activeWorkspaceId) {
      setPreviewLiveDataConnections([]);
      return;
    }

    if (!files.some((file) => file.path === previewEntryPath)) {
      setPreviewLiveDataConnections([]);
      return;
    }

    let cancelled = false;

    api.getUserSpaceFile(activeWorkspaceId, previewEntryPath)
      .then((file) => {
        if (cancelled) return;
        setPreviewLiveDataConnections(file.live_data_connections ?? []);
      })
      .catch(() => {
        if (cancelled) return;
        setPreviewLiveDataConnections([]);
      });

    return () => {
      cancelled = true;
    };
  }, [activeWorkspaceId, files, previewEntryPath]);

  useEffect(() => {
    const loadTools = async () => {
      try {
        const [tools, groups] = await Promise.all([
          api.listUserSpaceAvailableTools(),
          api.listUserSpaceToolGroups(),
        ]);
        setAvailableTools(tools);
        setToolGroups(groups.map((g) => ({ id: g.id, name: g.name })));
      } catch (err) {
        console.warn('Failed to load User Space tools', err);
      }
    };

    loadTools();
  }, []);

  useEffect(() => {
    if (!activeWorkspaceId) return;
    setSnapshots([]);
    setSnapshotBranches([]);
    setCurrentSnapshotId(null);
    setCurrentSnapshotBranchId(null);
    setSnapshotsLoadedForWorkspace(null);
    setShowSnapshots(false);
    setExpandedSnapshotIds(new Set());
    setSnapshotDiffSummaries({});
    setLoadingSnapshotDiffSummaryIds({});
    setSnapshotDiffSummaryErrors({});
    snapshotFileDiffCacheRef.current.clear();
    diffHover.dismiss();
    setChangedFiles(new Set());
    setAcknowledgedFiles(new Set());
    void Promise.all([
      loadWorkspaceData(activeWorkspaceId),
      loadChangedFileState(activeWorkspaceId),
      api.listWorkspaceMounts(activeWorkspaceId).then(setMounts).catch(() => setMounts([])),
    ]);
  }, [activeWorkspaceId, diffHover, loadChangedFileState, loadWorkspaceData]);

  useEffect(() => {
    if (!activeWorkspaceId) return;

    let cancelled = false;
    let inFlight = false;
    let idlePollCycles = 0;
    let timer: number | null = null;

    const getNextDelay = () => {
      if (!isPageVisible) {
        return USERSPACE_FILE_TREE_BACKGROUND_POLL_INTERVAL_MS;
      }
      if (idlePollCycles >= 3) {
        return USERSPACE_FILE_TREE_MAX_IDLE_POLL_INTERVAL_MS;
      }
      if (idlePollCycles >= 1) {
        return USERSPACE_FILE_TREE_IDLE_POLL_INTERVAL_MS;
      }
      return USERSPACE_FILE_TREE_POLL_INTERVAL_MS;
    };

    const scheduleNextPoll = () => {
      if (cancelled) {
        return;
      }
      timer = window.setTimeout(() => {
        void pollWorkspaceFiles();
      }, getNextDelay());
    };

    const pollWorkspaceFiles = async () => {
      if (inFlight) return;
      inFlight = true;
      try {
        const changed = await refreshWorkspaceFileTree(activeWorkspaceId);
        idlePollCycles = changed ? 0 : Math.min(idlePollCycles + 1, 4);
      } catch {
        // Ignore polling failures; the next interval will retry.
      } finally {
        if (!cancelled) {
          inFlight = false;
          scheduleNextPoll();
        }
      }
    };

    void pollWorkspaceFiles();

    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [activeWorkspaceId, isPageVisible, refreshWorkspaceFileTree]);

  useEffect(() => {
    fileContentCacheRef.current = fileContentCache;
  }, [fileContentCache]);

  useEffect(() => {
    fileDirtyRef.current = fileDirty;
  }, [fileDirty]);

  useEffect(() => {
    fileBrowserEntriesRef.current = fileBrowserEntries;
  }, [fileBrowserEntries]);

  useEffect(() => {
    activeWorkspaceIdRef.current = activeWorkspaceId;
  }, [activeWorkspaceId]);

  useEffect(() => {
    workspacesRef.current = workspaces;
  }, [workspaces]);

  useEffect(() => {
    lastWorkspaceCookieNameRef.current = lastWorkspaceCookieName;
  }, [lastWorkspaceCookieName]);

  useEffect(() => {
    activeWorkspaceConversationIdRef.current = activeWorkspaceConversationId;
  }, [activeWorkspaceConversationId]);

  const runRefreshActiveWorkspaceState = useCallback(async (
    workspaceId: string | null,
    conversationId: string | null,
  ) => {
    if (!workspaceId) {
      setRuntimeStatus(null);
      setActiveWorkspaceChatSnapshot(null);
      return;
    }

    refreshRuntimeStatusInflightRef.current = true;
    const requestId = ++loadRuntimeStatusRequestIdRef.current;
    const requestedConversationId = conversationId ?? null;

    try {
      const state = await api.getUserSpaceWorkspaceTabState(
        workspaceId,
        conversationId,
      );

      const isCurrentWorkspaceRequest = (
        requestId === loadRuntimeStatusRequestIdRef.current
        && activeWorkspaceIdRef.current === workspaceId
      );
      const isCurrentConversationRequest = (
        isCurrentWorkspaceRequest
        && activeWorkspaceConversationIdRef.current === requestedConversationId
      );

      if (isCurrentWorkspaceRequest) {
        setRuntimeStatus(state.runtime_status);
        // Ignore tab-state chat snapshots for a conversation that is no longer selected.
        if (isCurrentConversationRequest) {
          setActiveWorkspaceChatSnapshot(state.chat_state);
        }
      }
    } catch {
      const isCurrentWorkspaceRequest = (
        requestId === loadRuntimeStatusRequestIdRef.current
        && activeWorkspaceIdRef.current === workspaceId
      );
      const isCurrentConversationRequest = (
        isCurrentWorkspaceRequest
        && activeWorkspaceConversationIdRef.current === requestedConversationId
      );

      if (isCurrentWorkspaceRequest) {
        setRuntimeStatus(null);
        if (isCurrentConversationRequest) {
          setActiveWorkspaceChatSnapshot(null);
        }
      }
    } finally {
      refreshRuntimeStatusInflightRef.current = false;

      if (refreshRuntimeStatusPendingRef.current) {
        refreshRuntimeStatusPendingRef.current = false;
        void runRefreshActiveWorkspaceState(
          activeWorkspaceIdRef.current,
          activeWorkspaceConversationIdRef.current,
        );
      }
    }
  }, []);

  const refreshActiveWorkspaceState = useCallback(async () => {
    if (refreshRuntimeStatusInflightRef.current) {
      refreshRuntimeStatusPendingRef.current = true;
      return;
    }

    await runRefreshActiveWorkspaceState(
      activeWorkspaceIdRef.current,
      activeWorkspaceConversationIdRef.current,
    );
  }, [runRefreshActiveWorkspaceState]);

  // SSE subscription for workspace change events (file upsert/patch/delete, snapshots).
  // Bumps previewRefreshCounter to remount the preview iframe and reloads workspace data.
  useEffect(() => {
    if (!activeWorkspaceId) return;

    if (workspaceEventsReconnectTimerRef.current !== null) {
      window.clearTimeout(workspaceEventsReconnectTimerRef.current);
      workspaceEventsReconnectTimerRef.current = null;
    }

    const source = api.subscribeWorkspaceEvents(activeWorkspaceId, 0);
    let lastGeneration = 0;
    let consecutiveErrors = 0;
    let reconnectScheduled = false;

    source.onmessage = (event) => {
      consecutiveErrors = 0; // reset on successful message
      try {
        const data = JSON.parse(event.data) as { generation: number; event_type?: string; path?: string; old_path?: string; new_path?: string };
        if (data.generation > lastGeneration) {
          lastGeneration = data.generation;
          const eventType = data.event_type ?? 'update';
          if (eventType === 'runtime_phase') {
            void refreshActiveWorkspaceState();
            return;
          }

          // Mark files changed by agent tools (SSE events carry a path when originating from agent tools).
          if (data.path && (eventType === 'file_upsert' || eventType === 'file_patch')) {
            setChangedFiles((prev) => { const next = new Set(prev); next.add(data.path!); return next; });
            setAcknowledgedFiles((prev) => { const next = new Set(prev); next.delete(data.path!); return next; });
          }
          if (data.path && eventType === 'file_delete') {
            setChangedFiles((prev) => { const next = new Set(prev); next.delete(data.path!); return next; });
            setAcknowledgedFiles((prev) => { const next = new Set(prev); next.delete(data.path!); return next; });
          }
          if (eventType === 'file_move' && data.old_path && data.new_path) {
            setChangedFiles((prev) => {
              if (!prev.has(data.old_path!)) return prev;
              const next = new Set(prev); next.delete(data.old_path!); next.add(data.new_path!); return next;
            });
            setAcknowledgedFiles((prev) => {
              if (!prev.has(data.old_path!)) return prev;
              const next = new Set(prev); next.delete(data.old_path!); next.add(data.new_path!); return next;
            });
          }
          if (eventType === 'snapshot') {
            setChangedFiles(new Set());
            setAcknowledgedFiles(new Set());
          }

          // Avoid remounting preview on high-frequency collab doc updates.
          // Runtime HMR handles content refresh; remount only for structural events.
          const shouldRefreshPreview = eventType === 'file_upsert' || eventType === 'file_delete';
          if (shouldRefreshPreview) {
            setPreviewRefreshCounter((c) => c + 1);
          }

          // Skip full workspace reloads for collab/snapshot update events to reduce UI churn.
          const shouldReloadWorkspace = eventType !== 'update' && eventType !== 'snapshot';
          if (shouldReloadWorkspace && !fileDirtyRef.current && !isCodeEditorFocused()) {
            debouncedLoadWorkspaceData(activeWorkspaceId);
          }
          if (shouldReloadWorkspace || eventType === 'snapshot' || eventType === 'update') {
            debouncedLoadChangedFileState(activeWorkspaceId);
          }
        }
      } catch {
        // ignore malformed messages
      }
    };

    source.onerror = () => {
      consecutiveErrors++;
      // Close after repeated failures to avoid burning a connection slot
      // with an infinite auto-reconnect loop (e.g. 401/404 responses).
      if (consecutiveErrors >= 5) {
        source.close();
        if (!reconnectScheduled) {
          reconnectScheduled = true;
          workspaceEventsReconnectTimerRef.current = window.setTimeout(() => {
            workspaceEventsReconnectTimerRef.current = null;
            setWorkspaceEventsReconnectNonce((value) => value + 1);
          }, 5000);
        }
      }
    };

    return () => {
      source.close();
      if (loadWorkspaceDataDebounceRef.current !== null) {
        window.clearTimeout(loadWorkspaceDataDebounceRef.current);
        loadWorkspaceDataDebounceRef.current = null;
      }
      if (loadChangedFileStateDebounceRef.current !== null) {
        window.clearTimeout(loadChangedFileStateDebounceRef.current);
        loadChangedFileStateDebounceRef.current = null;
      }
      if (changedFileStateGuardTimerRef.current !== null) {
        window.clearTimeout(changedFileStateGuardTimerRef.current);
        changedFileStateGuardTimerRef.current = null;
      }
      if (workspaceEventsReconnectTimerRef.current !== null) {
        window.clearTimeout(workspaceEventsReconnectTimerRef.current);
        workspaceEventsReconnectTimerRef.current = null;
      }
    };
  }, [
    activeWorkspaceId,
    debouncedLoadChangedFileState,
    debouncedLoadWorkspaceData,
    isCodeEditorFocused,
    refreshActiveWorkspaceState,
    workspaceEventsReconnectNonce,
  ]);

  useEffect(() => {
    selectedFilePathRef.current = selectedFilePath;
  }, [selectedFilePath]);

  useEffect(() => {
    if (!showToolPicker) return;
    function handleClickOutside(event: MouseEvent) {
      if (toolPickerRef.current && !toolPickerRef.current.contains(event.target as Node)) {
        setShowToolPicker(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showToolPicker]);

  useEffect(() => {
    if (!isWorkspaceMenuOpen) return;

    function handleClickOutside(event: MouseEvent) {
      if (workspaceDropdownRef.current && !workspaceDropdownRef.current.contains(event.target as Node)) {
        setIsWorkspaceMenuOpen(false);
      }
    }

    function handleEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        if (editingWorkspaceNameId) {
          setEditingWorkspaceNameId(null);
          setWorkspaceNameDraft('');
          return;
        }
        setIsWorkspaceMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isWorkspaceMenuOpen, editingWorkspaceNameId]);

  useEffect(() => {
    if (!isWorkspaceMenuOpen) {
      setDeleteConfirmWorkspaceId(null);
      setEditingWorkspaceNameId(null);
      setWorkspaceNameDraft('');
    }
  }, [isWorkspaceMenuOpen]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      setExpandedFolders(new Set());
      return;
    }

    try {
      const raw = window.localStorage.getItem(getExpandedFoldersStorageKey(activeWorkspaceId));
      if (!raw) {
        setExpandedFolders(new Set());
        return;
      }

      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        setExpandedFolders(new Set());
        return;
      }

      setExpandedFolders(new Set(parsed.filter((value): value is string => typeof value === 'string')));
    } catch {
      setExpandedFolders(new Set());
    }
  }, [activeWorkspaceId]);

  useEffect(() => {
    setExpandedFolders((current) => {
      const next = new Set(Array.from(current).filter((path) => folderPaths.has(path)));
      if (next.size === current.size) {
        return current;
      }
      return next;
    });
  }, [folderPaths]);

  useEffect(() => {
    if (!activeWorkspaceId) return;
    window.localStorage.setItem(getExpandedFoldersStorageKey(activeWorkspaceId), JSON.stringify(Array.from(expandedFolders)));
  }, [activeWorkspaceId, expandedFolders]);

  useEffect(() => {
    if (!selectedFilePath) return;
    const ancestors = getAncestorFolderPaths(selectedFilePath);
    if (ancestors.length === 0) return;

    setExpandedFolders((current) => {
      const next = new Set(current);
      let changed = false;
      for (const folderPath of ancestors) {
        if (!next.has(folderPath)) {
          next.add(folderPath);
          changed = true;
        }
      }
      return changed ? next : current;
    });
  }, [selectedFilePath]);

  const handleCreateWorkspace = useCallback(async () => {
    setError(null);

    try {
      const task = await api.queueUserSpaceWorkspaceCreate({});
      latestQueuedWorkspaceCreateTaskIdRef.current = task.task_id;
      setCreatingWorkspaceTasks((current) => ({
        ...current,
        [task.task_id]: task,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue workspace creation');
    }
  }, []);

  const handleDuplicateWorkspace = useCallback(async (workspaceId: string) => {
    setError(null);

    try {
      const task = await api.queueUserSpaceWorkspaceDuplicate(workspaceId, {});
      setDuplicatingWorkspaceTasks((current) => ({
        ...current,
        [task.task_id]: task,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue workspace duplication');
    }
  }, []);

  useEffect(() => {
    const tasks = Object.values(creatingWorkspaceTasks);
    if (tasks.length === 0) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollCreateTasks = async () => {
      if (pollInFlight) {
        return;
      }
      pollInFlight = true;

      try {
        const results = await Promise.all(tasks.map(async (task) => {
          try {
            const status = await api.getUserSpaceWorkspaceCreateTask(task.task_id);
            return { task, status, error: null as Error | null };
          } catch (error) {
            return { task, status: null as UserSpaceWorkspaceCreateTask | null, error: error as Error };
          }
        }));

        if (cancelled) {
          return;
        }

        const terminalTaskIds = new Set<string>();
        const completedTasks: UserSpaceWorkspaceCreateTask[] = [];
        const updatedTasks: Record<string, UserSpaceWorkspaceCreateTask> = {};
        let nextError: string | null = null;

        for (const result of results) {
          if (result.status) {
            if (isWorkspaceCreateTaskTerminal(result.status.phase)) {
              terminalTaskIds.add(result.status.task_id);
              if (result.status.phase === 'completed' && result.status.workspace_id) {
                completedTasks.push(result.status);
              }
              if (result.status.phase === 'failed' && !nextError) {
                nextError = result.status.error?.trim() || `Failed to create ${result.status.workspace_name || 'workspace'}`;
              }
            } else {
              updatedTasks[result.status.task_id] = result.status;
            }
            continue;
          }

          if (result.error instanceof ApiError && result.error.status === 404) {
            terminalTaskIds.add(result.task.task_id);
            continue;
          }

          if (!nextError && result.error instanceof Error) {
            nextError = result.error.message;
          }
        }

        setCreatingWorkspaceTasks((current) => {
          const next = { ...current };
          for (const taskId of terminalTaskIds) {
            delete next[taskId];
          }
          for (const [taskId, task] of Object.entries(updatedTasks)) {
            next[taskId] = task;
          }
          return next;
        });

        if (nextError) {
          setError(nextError);
        }

        if (terminalTaskIds.size > 0) {
          try {
            await loadWorkspaces();
          } catch (error) {
            if (!nextError && error instanceof Error) {
              setError(error.message);
            }
          }
        }

        if (cancelled || completedTasks.length === 0) {
          return;
        }

        completedTasks.sort((left, right) => Date.parse(left.queued_at) - Date.parse(right.queued_at));
        const autoSelectTask = completedTasks.find((task) => task.task_id === latestQueuedWorkspaceCreateTaskIdRef.current)
          ?? (!activeWorkspaceId ? completedTasks[completedTasks.length - 1] : null);

        if (autoSelectTask?.workspace_id) {
          if (latestQueuedWorkspaceCreateTaskIdRef.current === autoSelectTask.task_id) {
            latestQueuedWorkspaceCreateTaskIdRef.current = null;
          }
          setRuntimeStatus(null);
          setPreviewLiveDataConnections([]);
          setActiveWorkspaceId(autoSelectTask.workspace_id);
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollCreateTasks();
    const intervalId = window.setInterval(() => {
      void pollCreateTasks();
    }, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [activeWorkspaceId, creatingWorkspaceTasks, loadWorkspaces]);

  useEffect(() => {
    const tasks = Object.values(duplicatingWorkspaceTasks);
    if (tasks.length === 0) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollDuplicateTasks = async () => {
      if (pollInFlight) {
        return;
      }
      pollInFlight = true;

      try {
        const results = await Promise.all(tasks.map(async (task) => {
          try {
            const status = await api.getUserSpaceWorkspaceDuplicateTask(task.task_id);
            return { task, status, error: null as Error | null };
          } catch (error) {
            return { task, status: null as UserSpaceWorkspaceDuplicateTask | null, error: error as Error };
          }
        }));

        if (cancelled) {
          return;
        }

        const terminalTaskIds = new Set<string>();
        const updatedTasks: Record<string, UserSpaceWorkspaceDuplicateTask> = {};
        let nextError: string | null = null;

        for (const result of results) {
          if (result.status) {
            if (isWorkspaceDuplicateTaskTerminal(result.status.phase)) {
              terminalTaskIds.add(result.status.task_id);
              if (result.status.phase === 'failed' && !nextError) {
                nextError = result.status.error?.trim() || `Failed to duplicate ${result.status.workspace_name || 'workspace'}`;
              }
            } else {
              updatedTasks[result.status.task_id] = result.status;
            }
            continue;
          }

          if (result.error instanceof ApiError && result.error.status === 404) {
            terminalTaskIds.add(result.task.task_id);
            continue;
          }

          if (!nextError && result.error instanceof Error) {
            nextError = result.error.message;
          }
        }

        setDuplicatingWorkspaceTasks((current) => {
          const next = { ...current };
          for (const taskId of terminalTaskIds) {
            delete next[taskId];
          }
          for (const [taskId, task] of Object.entries(updatedTasks)) {
            next[taskId] = task;
          }
          return next;
        });

        if (nextError) {
          setError(nextError);
        }

        if (terminalTaskIds.size > 0) {
          try {
            await loadWorkspaces();
          } catch {
            // Best-effort refresh after duplicate.
          }
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollDuplicateTasks();
    const intervalId = window.setInterval(() => {
      void pollDuplicateTasks();
    }, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [duplicatingWorkspaceTasks, loadWorkspaces]);

  const handleSelectFile = useCallback(async (path: string) => {
    if (!activeWorkspaceId) return;

    selectedFilePathRef.current = path;
    setSelectedFilePath(path);

    try {
      const selectedMeta = files.find((file) => file.path === path);
      const selectedUpdatedAt = selectedMeta?.updated_at ?? '';
      const cached = fileContentCacheRef.current[path];

      if (cached && cached.updatedAt === selectedUpdatedAt) {
        setFileContent(cached.content);
        setFileDirty(false);
        setSelectedFileArtifactType(cached.artifactType ?? null);
        setSelectedFileUnsupportedMessage(null);
        setError(null);
        return;
      }

      const file = await api.getUserSpaceFile(activeWorkspaceId, path);
      setFileContent(file.content);
      setSelectedFileArtifactType(file.artifact_type ?? null);
      setFileContentCache((current) => ({
        ...current,
        [file.path]: {
          content: file.content,
          updatedAt: selectedUpdatedAt,
          artifactType: file.artifact_type ?? null,
        },
      }));
      setFileDirty(false);
      setSelectedFileUnsupportedMessage(null);
      setError(null);
    } catch (err) {
      const unsupportedMessage = getUnsupportedEditorFileMessage(err);
      if (unsupportedMessage) {
        setFileContent('');
        setFileDirty(false);
        setSelectedFileArtifactType(null);
        setSelectedFileUnsupportedMessage(unsupportedMessage);
        setError(null);
        return;
      }

      setSelectedFileArtifactType(null);
      setSelectedFileUnsupportedMessage(null);
      setError(err instanceof Error ? err.message : 'Failed to open file');
    }
  }, [activeWorkspaceId, files]);

  const handleCreateSnapshot = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setCreatingSnapshot(true);
    try {
      await api.createUserSpaceSnapshot(activeWorkspaceId, {
        message: 'Manual snapshot',
        ...(activeWorkspaceConversationId ? { conversation_id: activeWorkspaceConversationId } : {}),
      });
      // Reset all per-file changed/acknowledged markers after snapshot baseline resets.
      setChangedFiles(new Set());
      setAcknowledgedFiles(new Set());
      setFileDirty(false);
      // Refresh snapshots list if panel is open.
      setSnapshotsLoadedForWorkspace(null);
      await refreshActiveWorkspaceState();
      await loadChangedFileState(activeWorkspaceId);
      if (showSnapshots) {
        await loadSnapshots(activeWorkspaceId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create snapshot');
    } finally {
      setCreatingSnapshot(false);
    }
  }, [activeWorkspaceId, activeWorkspaceConversationId, canEditWorkspace, loadChangedFileState, loadSnapshots, refreshActiveWorkspaceState, showSnapshots]);

  const handleSaveTreeFile = useCallback(async (filePath: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setSavingTreeFile(filePath);
    try {
      // Determine content: if the file is currently selected, use editor state;
      // otherwise fall back to the file content cache.
      const content = filePath === selectedFilePath
        ? fileContent
        : (fileContentCacheRef.current[filePath]?.content ?? '');
      const artifactType = filePath === selectedFilePath
        ? (selectedFileArtifactType ?? undefined)
        : (fileContentCacheRef.current[filePath]?.artifactType ?? undefined);
      await api.upsertUserSpaceFile(activeWorkspaceId, filePath, {
        content,
        artifact_type: artifactType,
      });
      const changedFileState = await api.acknowledgeUserSpaceChangedFilePath(activeWorkspaceId, { path: filePath });
      setChangedFiles(new Set(changedFileState.changed_file_paths));
      setAcknowledgedFiles(new Set(changedFileState.acknowledged_changed_file_paths));
      // If saving the currently selected file, clear its editor dirty flag too.
      if (filePath === selectedFilePath) {
        setFileDirty(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save file');
    } finally {
      setSavingTreeFile(null);
    }
  }, [activeWorkspaceId, canEditWorkspace, fileContent, selectedFileArtifactType, selectedFilePath]);

  const handleToggleWorkspaceTool = useCallback(async (toolId: string) => {
    if (!activeWorkspace || !canEditWorkspace) return;

    const targetTool = availableTools.find((tool) => tool.id === toolId);
    const currentGroupIds = new Set(activeWorkspace.selected_tool_group_ids ?? []);

    if (targetTool?.group_id && currentGroupIds.has(targetTool.group_id)) {
      const nextGroupIds = new Set(currentGroupIds);
      nextGroupIds.delete(targetTool.group_id);
      const nextSelected = new Set<string>();
      for (const tool of availableTools) {
        if (tool.group_id === targetTool.group_id && tool.id !== toolId) {
          nextSelected.add(tool.id);
        }
      }

      setSavingWorkspaceTools(true);
      try {
        const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, {
          selected_tool_ids: Array.from(nextSelected),
          selected_tool_group_ids: Array.from(nextGroupIds),
        });
        setWorkspaces((current) => current.map((workspace) => workspace.id === updated.id ? updated : workspace));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to update tool selection');
      } finally {
        setSavingWorkspaceTools(false);
      }
      return;
    }

    const nextSelected = new Set(activeWorkspace.selected_tool_ids ?? []);
    if (nextSelected.has(toolId)) {
      nextSelected.delete(toolId);
    } else {
      nextSelected.add(toolId);
    }

    setSavingWorkspaceTools(true);
    try {
      const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, {
        selected_tool_ids: Array.from(nextSelected),
        selected_tool_group_ids: Array.from(currentGroupIds),
      });
      setWorkspaces((current) => current.map((workspace) => workspace.id === updated.id ? updated : workspace));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update tool selection');
    } finally {
      setSavingWorkspaceTools(false);
    }
  }, [activeWorkspace, availableTools, canEditWorkspace]);

  const handleToggleWorkspaceToolGroup = useCallback(async (groupId: string) => {
    if (!activeWorkspace || !canEditWorkspace) return;

    const groupToolIds = availableTools
      .filter((tool) => tool.group_id === groupId)
      .map((tool) => tool.id);
    const currentGroupIds = new Set(activeWorkspace.selected_tool_group_ids ?? []);
    const nextGroupIds = new Set(currentGroupIds);
    const nextToolIds = new Set(activeWorkspace.selected_tool_ids ?? []);
    if (nextGroupIds.has(groupId)) {
      nextGroupIds.delete(groupId);
      for (const toolId of groupToolIds) {
        nextToolIds.delete(toolId);
      }
    } else {
      nextGroupIds.add(groupId);
    }

    setSavingWorkspaceTools(true);
    try {
      const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, {
        selected_tool_ids: Array.from(nextToolIds),
        selected_tool_group_ids: Array.from(nextGroupIds),
      });
      setWorkspaces((current) => current.map((workspace) => workspace.id === updated.id ? updated : workspace));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update tool group selection');
    } finally {
      setSavingWorkspaceTools(false);
    }
  }, [activeWorkspace, availableTools, canEditWorkspace]);

  const handleToggleSqlitePersistence = useCallback(async () => {
    if (!activeWorkspace || !canEditWorkspace) return;
    const nextMode = activeWorkspace.sqlite_persistence_mode === 'include' ? 'exclude' : 'include';
    try {
      const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, {
        sqlite_persistence_mode: nextMode,
      });
      setWorkspaces((current) => current.map((ws) => ws.id === updated.id ? updated : ws));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update SQLite persistence mode');
    }
  }, [activeWorkspace, canEditWorkspace]);

  const handleWorkspaceScmSyncComplete = useCallback(async (response: UserSpaceWorkspaceScmSyncResponse) => {
    setWorkspaces((current) => current.map((ws) => (
      ws.id === response.workspace_id
        ? { ...ws, scm: response.scm as UserSpaceWorkspaceScmStatus }
        : ws
    )));
    if (activeWorkspaceId === response.workspace_id) {
      await Promise.all([
        loadWorkspaceData(response.workspace_id),
        loadChangedFileState(response.workspace_id),
        loadSnapshots(response.workspace_id),
      ]);
    }
  }, [activeWorkspaceId, loadChangedFileState, loadSnapshots, loadWorkspaceData]);

  const handleAskAgentToPrepareWorkspace = useCallback(async (prompt: string) => {
    if (!activeWorkspaceId) return;
    expandChat();
    setError(null);

    // Serialize after available-models to avoid a false-positive "no models configured" error.
    refreshAvailableModels();
    const readyState = await awaitAvailableModelsReady();

    const hasAnyModel = readyState.models.length > 0;
    if (!hasAnyModel) {
      setError('No LLM configured. Please configure an LLM in Settings.');
      return;
    }

    try {
      const conversation = await api.createConversation(undefined, activeWorkspaceId);

      setActiveWorkspaceConversationId(conversation.id);
      setActiveWorkspaceChatSnapshot((current) => ({
        conversations: [
          conversation,
          ...(current?.conversations.filter((item) => item.id !== conversation.id) ?? []),
        ],
        interrupted_conversation_ids: current?.interrupted_conversation_ids ?? [],
        selected_conversation_id: conversation.id,
        active_task: null,
        interrupted_task: null,
      }));

      await api.sendMessageBackground(conversation.id, prompt, activeWorkspaceId);
    } catch (err) {
      setError(getApiErrorMessage(err, 'Failed to ask the agent to prepare the workspace'));
    }
  }, [activeWorkspaceId, awaitAvailableModelsReady, expandChat, refreshAvailableModels]);

  const handleUserMessageSubmitted = useCallback(async (_message: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    await loadWorkspaceData(activeWorkspaceId);
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData]);

  const handleRestoreSnapshot = useCallback(async (snapshotId: string, snapshotBranchId?: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

    const waitForRestoredCursor = async (workspaceId: string, expectedSnapshotId: string): Promise<void> => {
      const maxAttempts = 18;
      for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
        try {
          const timeline = await api.getUserSpaceSnapshotTimeline(workspaceId);
          setSnapshots(timeline.snapshots);
          setSnapshotBranches(timeline.branches);
          setCurrentSnapshotId(timeline.current_snapshot_id ?? null);
          setCurrentSnapshotBranchId(timeline.current_branch_id ?? null);
          setSnapshotsLoadedForWorkspace(workspaceId);
          if (timeline.current_snapshot_id === expectedSnapshotId) {
            return;
          }
        } catch {
          // Keep polling; final refresh happens below regardless.
        }
        await wait(250);
      }
    };

    setNavigatingSnapshots(true);
    setRestoringSnapshotId(snapshotId);
    try {
      if (snapshotBranchId && snapshotBranchId !== currentSnapshotBranchId) {
        const timeline = await api.switchUserSpaceSnapshotBranch(activeWorkspaceId, { branch_id: snapshotBranchId });
        setSnapshots(timeline.snapshots);
        setSnapshotBranches(timeline.branches);
        setCurrentSnapshotId(timeline.current_snapshot_id ?? null);
        setCurrentSnapshotBranchId(timeline.current_branch_id ?? null);
      }

      const restoreResult = await api.restoreUserSpaceSnapshot(activeWorkspaceId, snapshotId);
      await waitForRestoredCursor(activeWorkspaceId, restoreResult.restored_snapshot_id || snapshotId);
      setChangedFiles(new Set());
      setAcknowledgedFiles(new Set());
      setFileDirty(false);
      setSnapshotsLoadedForWorkspace(null);
      await Promise.all([
        loadWorkspaceData(activeWorkspaceId),
        loadChangedFileState(activeWorkspaceId),
        loadSnapshots(activeWorkspaceId),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restore snapshot');
    } finally {
      setRestoringSnapshotId(null);
      setNavigatingSnapshots(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, currentSnapshotBranchId, loadChangedFileState, loadSnapshots, loadWorkspaceData]);

  const handleMessageSnapshotRestored = useCallback(async (details?: {
    rolledBackSnapshot?: boolean;
    requiresRuntimeRestart?: boolean;
  }) => {
    if (!activeWorkspaceId) return;
    // The chat panel already updated the conversation. Refresh workspace data,
    // changed-file markers, and the snapshots timeline (cursor moves).
    setChangedFiles(new Set());
    setAcknowledgedFiles(new Set());
    setFileDirty(false);
    setSnapshotsLoadedForWorkspace(null);
    if (details?.rolledBackSnapshot && details.requiresRuntimeRestart) {
      setPreviewNotice({
        id: Date.now(),
        tone: 'success',
        message: 'Snapshot rolled back. Restart the runtime container to reload the preview.',
      });
    }
    try {
      await Promise.all([
        loadWorkspaceData(activeWorkspaceId),
        loadChangedFileState(activeWorkspaceId),
        loadSnapshots(activeWorkspaceId),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to refresh after snapshot restore');
    }
  }, [activeWorkspaceId, loadChangedFileState, loadSnapshots, loadWorkspaceData]);

  const handleBranchSwitch = useCallback((
    _branchId: string,
    associatedSnapshotId: string | null,
  ) => {
    // Switching branches MUST always prompt the user before restoring the
    // associated workspace snapshot, even when the active user can edit the
    // workspace. The snapshot restore is destructive (overwrites uncommitted
    // workspace state) so we never auto-apply it.
    if (!associatedSnapshotId) {
      setBranchRestoreSnapshotId(null);
      return;
    }
    setBranchRestoreSnapshotId(associatedSnapshotId);
  }, []);

  // Triggered by the chat panel whenever something happens that may have
  // produced or moved a userspace snapshot (chat-branch auto-snapshot,
  // walkback restore, live agent `create_userspace_snapshot` tool calls).
  // We refresh the snapshots timeline only when the snapshots panel is
  // currently open, to avoid pointless network chatter.
  const handleSnapshotsMaybeChanged = useCallback(() => {
    if (!activeWorkspaceId) return;
    if (!showSnapshots) return;
    void loadSnapshots(activeWorkspaceId);
  }, [activeWorkspaceId, loadSnapshots, showSnapshots]);

  const handleConfirmBranchRestore = useCallback(() => {
    if (branchRestoreSnapshotId) {
      void handleRestoreSnapshot(branchRestoreSnapshotId);
    }
    setBranchRestoreSnapshotId(null);
  }, [branchRestoreSnapshotId, handleRestoreSnapshot]);

  const handleDismissBranchRestore = useCallback(() => {
    setBranchRestoreSnapshotId(null);
  }, []);

  const handleStartSnapshotRename = useCallback((snapshot: UserSpaceSnapshot) => {
    if (!canEditWorkspace) return;
    setRenamingSnapshotId(snapshot.id);
    setSnapshotEditValue(snapshot.message ?? '');
  }, [canEditWorkspace]);

  const handleCancelSnapshotRename = useCallback(() => {
    setRenamingSnapshotId(null);
    setSnapshotEditValue('');
  }, []);

  const handleSaveSnapshotRename = useCallback(async (snapshotId: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    const next = snapshotEditValue.trim();
    if (!next) {
      handleCancelSnapshotRename();
      return;
    }
    setSavingSnapshotRename(true);
    try {
      await api.updateUserSpaceSnapshot(activeWorkspaceId, snapshotId, { message: next });
      await loadSnapshots(activeWorkspaceId);
      setRenamingSnapshotId(null);
      setSnapshotEditValue('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename snapshot');
    } finally {
      setSavingSnapshotRename(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, handleCancelSnapshotRename, loadSnapshots, snapshotEditValue]);

  const handleSwitchSnapshotBranch = useCallback(async (branchId: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setNavigatingSnapshots(true);
    try {
      const timeline = await api.switchUserSpaceSnapshotBranch(activeWorkspaceId, { branch_id: branchId });
      setSnapshots(timeline.snapshots);
      setSnapshotBranches(timeline.branches);
      setCurrentSnapshotId(timeline.current_snapshot_id ?? null);
      setCurrentSnapshotBranchId(timeline.current_branch_id ?? null);
      await Promise.all([
        loadWorkspaceData(activeWorkspaceId),
        loadChangedFileState(activeWorkspaceId),
      ]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch snapshot branch');
    } finally {
      setNavigatingSnapshots(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, loadChangedFileState, loadWorkspaceData]);

  const handleCreateSnapshotBranch = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setNavigatingSnapshots(true);
    try {
      const timeline = await api.createUserSpaceSnapshotBranch(activeWorkspaceId, {});
      setSnapshots(timeline.snapshots);
      setSnapshotBranches(timeline.branches);
      setCurrentSnapshotId(timeline.current_snapshot_id ?? null);
      setCurrentSnapshotBranchId(timeline.current_branch_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create branch');
    } finally {
      setNavigatingSnapshots(false);
    }
  }, [activeWorkspaceId, canEditWorkspace]);

  const handlePromoteBranchToMain = useCallback(async (branchId: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setNavigatingSnapshots(true);
    try {
      const timeline = await api.promoteBranchToMain(activeWorkspaceId, { branch_id: branchId });
      setSnapshots(timeline.snapshots);
      setSnapshotBranches(timeline.branches);
      setCurrentSnapshotId(timeline.current_snapshot_id ?? null);
      setCurrentSnapshotBranchId(timeline.current_branch_id ?? null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to promote branch to main');
    } finally {
      setNavigatingSnapshots(false);
    }
  }, [activeWorkspaceId, canEditWorkspace]);

  const handleDeleteSnapshot = useCallback(async (snapshotId: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setDeletingSnapshotId(snapshotId);
    try {
      const timeline = await api.deleteUserSpaceSnapshot(activeWorkspaceId, snapshotId);
      setSnapshots(timeline.snapshots);
      setSnapshotBranches(timeline.branches);
      setCurrentSnapshotId(timeline.current_snapshot_id ?? null);
      setCurrentSnapshotBranchId(timeline.current_branch_id ?? null);
      setSnapshotsLoadedForWorkspace(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete snapshot');
    } finally {
      setDeletingSnapshotId(null);
      setDeleteConfirmSnapshotId(null);
    }
  }, [activeWorkspaceId, canEditWorkspace]);

  const handleStartRuntime = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRuntimeBusy(true);
    setRuntimeStatus((prev) => prev ? { ...prev, session_state: 'starting', runtime_operation_phase: 'queued', last_error: null } : prev);
    try {
      await api.startUserSpaceRuntimeSession(activeWorkspaceId);
      await refreshActiveWorkspaceState();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start runtime');
    } finally {
      setRuntimeBusy(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, refreshActiveWorkspaceState]);

  const handleStopRuntime = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRuntimeBusy(true);
    setRuntimeStatus((prev) => prev ? { ...prev, session_state: 'stopping' } : prev);
    try {
      await api.stopUserSpaceRuntimeSession(activeWorkspaceId);
      await refreshActiveWorkspaceState();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop runtime');
    } finally {
      setRuntimeBusy(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, refreshActiveWorkspaceState]);

  const handleRestartRuntime = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRuntimeBusy(true);
    setRuntimeStatus((prev) => prev ? { ...prev, session_state: 'starting', runtime_operation_phase: 'queued', last_error: null } : prev);
    try {
      await api.restartUserSpaceRuntimeDevserver(activeWorkspaceId);
      await refreshActiveWorkspaceState();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restart runtime');
    } finally {
      setRuntimeBusy(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, refreshActiveWorkspaceState]);

  useEffect(() => {
    void refreshActiveWorkspaceState();
    if (!activeWorkspaceId) return;
    const isTransitional = runtimeDisplayState === 'starting' || runtimeDisplayState === 'stopping';
    let cancelled = false;
    let timer: number | null = null;
    let attempt = 0;

    const nextIntervalMs = () => {
      if (!isPageVisible) {
        return USERSPACE_RUNTIME_BACKGROUND_POLL_INTERVAL_MS;
      }
      if (!isTransitional) {
        return 10000;
      }
      const transitionalBackoff = [2500, 5000, 8000, 12000];
      return transitionalBackoff[Math.min(attempt, transitionalBackoff.length - 1)];
    };

    const scheduleNextPoll = () => {
      if (cancelled) {
        return;
      }
      timer = window.setTimeout(() => {
        void runPoll();
      }, nextIntervalMs());
    };

    const runPoll = async () => {
      await refreshActiveWorkspaceState();
      if (isTransitional && isPageVisible) {
        attempt += 1;
      } else {
        attempt = 0;
      }
      scheduleNextPoll();
    };

    scheduleNextPoll();

    return () => {
      cancelled = true;
      if (timer !== null) {
        window.clearTimeout(timer);
      }
    };
  }, [activeWorkspaceId, isPageVisible, refreshActiveWorkspaceState, runtimeDisplayState]);

  useEffect(() => {
    if (!activeWorkspaceId) return;
    void refreshActiveWorkspaceState();
  }, [activeWorkspaceConversationId, activeWorkspaceId, refreshActiveWorkspaceState]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      previewLaunchRequestIdRef.current += 1;
      previewLaunchExpiresAtMsRef.current = 0;
      browserSurfaceAuthExpiryRef.current = {};
      previousRuntimeDisplayStateRef.current = null;
      setError(null);
      setRuntimeStatus(null);
      setActiveWorkspaceChatSnapshot(null);
      setPreviewFrameUrl(null);
      setPreviewOrigin(null);
      setPreviewAuthorizationPending(false);
      return;
    }

    previewLaunchRequestIdRef.current += 1;
    previewLaunchExpiresAtMsRef.current = 0;
    browserSurfaceAuthExpiryRef.current = {};
  previousRuntimeDisplayStateRef.current = null;
    setError(null);
    setRuntimeStatus(null);
    setActiveWorkspaceChatSnapshot(null);
    setPreviewFrameUrl(null);
    setPreviewOrigin(null);
    setPreviewAuthorizationPending(true);
  }, [activeWorkspaceId]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      return;
    }

    let cancelled = false;
    const launchPreview = async () => {
      try {
        const previewUrl = await launchPreviewSurface(activeWorkspaceId);
        if (cancelled || !previewUrl) {
          return;
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to launch preview access');
        }
      }
    };

    void launchPreview();
    return () => {
      cancelled = true;
    };
  }, [activeWorkspaceId, launchPreviewSurface, previewRefreshCounter]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      previousRuntimeDisplayStateRef.current = null;
      return;
    }

    const previousState = previousRuntimeDisplayStateRef.current;
    const nextState = runtimeDisplayState;
    previousRuntimeDisplayStateRef.current = nextState;

    if (!previousState || previousState === nextState) {
      return;
    }

    // Runtime lifecycle transitions should invalidate stale preview bootstrap URLs.
    // If a frame is already mounted, refresh preview auth in place to avoid
    // tearing down iframe state during initial dashboard boot.
    if (nextState === 'running') {
      if (previewFrameUrl) {
        void launchPreviewSurface(activeWorkspaceId, {
          clearOnError: false,
          updateFrameUrl: false,
        });
      } else {
        setPreviewRefreshCounter((value) => value + 1);
      }
      return;
    }

    if (nextState === 'stopped' || nextState === 'error') {
      previewLaunchExpiresAtMsRef.current = 0;
      setPreviewAuthorizationPending(false);
      setPreviewFrameUrl(null);
      setPreviewOrigin(null);
    }
  }, [
    activeWorkspaceId,
    launchPreviewSurface,
    previewFrameUrl,
    runtimeDisplayState,
  ]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      return;
    }

    let cancelled = false;
    const refreshBrowserCapabilities = async () => {
      try {
        // Keep tab-specific browser surfaces warm to avoid reconnect failures from stale cookies.
        if (activeRightTab === 'console') {
          await authorizeBrowserSurfaces(activeWorkspaceId, ['runtime_pty']);
          return;
        }

        await authorizeBrowserSurfaces(activeWorkspaceId, ['collab']);
        if (!previewFrameUrl) {
          return;
        }

        const shouldRefreshPreviewLaunch = previewLaunchExpiresAtMsRef.current <= (Date.now() + USERSPACE_PREVIEW_LAUNCH_REFRESH_LEAD_MS);
        if (shouldRefreshPreviewLaunch) {
          // Session-warming only: keep the existing iframe alive instead of
          // remounting it, which would tear down the running workspace app
          // and any in-flight live-data executions.
          await launchPreviewSurface(activeWorkspaceId, {
            clearOnError: false,
            updateFrameUrl: false,
          });
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to refresh workspace browser authorization');
        }
      }
    };

    void refreshBrowserCapabilities();
    return () => {
      cancelled = true;
    };
  }, [activeRightTab, activeWorkspaceId, authorizeBrowserSurfaces, launchPreviewSurface, previewFrameUrl]);

  // Reset reconnect attempts when workspace or file changes (not on reconnect nonce)
  useEffect(() => {
    collabReconnectAttemptsRef.current = 0;
  }, [activeWorkspaceId, selectedFilePath]);

  useEffect(() => {
    if (collabReconnectTimerRef.current !== null) {
      window.clearTimeout(collabReconnectTimerRef.current);
      collabReconnectTimerRef.current = null;
    }

    collabSocketRef.current?.close();
    collabSocketRef.current = null;
    setCollabConnected(false);
    setCollabReadOnly(false);
    setCollabVersion(0);
    setCollabPresenceCount(0);

    if (!activeWorkspaceId || !selectedFilePath) return;

    let reconnectEnabled = true;
    const MAX_COLLAB_RECONNECT_ATTEMPTS = 8;
        const scheduleReconnect = () => {
      if (!reconnectEnabled || collabReconnectTimerRef.current !== null) {
        return;
      }
      if (collabReconnectAttemptsRef.current >= MAX_COLLAB_RECONNECT_ATTEMPTS) {
        return;
      }
      // Exponential backoff: 1.5s, 3s, 6s, 12s, capped at 15s
      const delay = Math.min(1500 * Math.pow(2, collabReconnectAttemptsRef.current), 15000);
      collabReconnectAttemptsRef.current++;
      collabReconnectTimerRef.current = window.setTimeout(() => {
        collabReconnectTimerRef.current = null;
        setCollabReconnectNonce((value) => value + 1);
      }, delay);
    };

    let socket: WebSocket | null = null;

    const connectCollab = async () => {
      try {
        await authorizeBrowserSurfaces(activeWorkspaceId, ['collab']);
        if (!reconnectEnabled) {
          return;
        }

        const socketUrl = api.getUserSpaceCollabWebSocketUrl(activeWorkspaceId, selectedFilePath);
        socket = new WebSocket(socketUrl);
        collabSocketRef.current = socket;

        socket.onopen = () => {
          setCollabConnected(true);
          collabReconnectAttemptsRef.current = 0;
          if (collabReconnectTimerRef.current !== null) {
            window.clearTimeout(collabReconnectTimerRef.current);
            collabReconnectTimerRef.current = null;
          }
          try {
            socket?.send(JSON.stringify({ type: 'presence' }));
          } catch {
            // ignore
          }
        };

        socket.onmessage = (event) => {
          let payload: UserSpaceCollabMessage | null = null;
          try {
            payload = JSON.parse(event.data) as UserSpaceCollabMessage;
          } catch {
            return;
          }

          if (!payload) {
            return;
          }

          if (payload.type === 'error') {
            setError(payload.message || 'Collaboration error');
            return;
          }

          if (payload.type === 'presence') {
            setCollabPresenceCount(payload.users.length);
            return;
          }

          if (payload.type === 'file_renamed') {
            if (selectedFilePath === payload.old_path) {
              setSelectedFilePath(payload.new_path);
            }
            return;
          }

          if (payload.type === 'file_created') {
            void loadWorkspaceData(activeWorkspaceId);
            return;
          }

          if (payload.file_path !== selectedFilePath) {
            return;
          }

          if (payload.type === 'snapshot' || payload.type === 'update') {
            collabSuppressNextSendRef.current = true;
            setCollabVersion(payload.version);
            setCollabReadOnly(payload.type === 'snapshot' ? payload.read_only : false);
            setFileContent(payload.content);
            setFileDirty(false);
            setFileContentCache((current) => ({
              ...current,
              [selectedFilePath]: {
                content: payload.content,
                updatedAt: current[selectedFilePath]?.updatedAt ?? '',
                artifactType: current[selectedFilePath]?.artifactType ?? selectedFileArtifactType,
              },
            }));
          }
          if (payload.type === 'ack') {
            setCollabVersion(payload.version);
          }
        };

        socket.onclose = (closeEvent) => {
          setCollabConnected(false);
          setCollabPresenceCount(0);

          // Do not retry for intentional/auth/path failures.
          if ([1000, 1001, 4401, 4403, 4404].includes(closeEvent.code)) {
            return;
          }
          scheduleReconnect();
        };

        socket.onerror = () => {
          setCollabConnected(false);
          setCollabPresenceCount(0);
          // onclose usually follows and controls reconnect behavior.
        };
      } catch (err) {
        if (!reconnectEnabled) {
          return;
        }
        setError(err instanceof Error ? err.message : 'Failed to authorize collaboration access');
        scheduleReconnect();
      }
    };

    void connectCollab();

    return () => {
      reconnectEnabled = false;
      if (collabReconnectTimerRef.current !== null) {
        window.clearTimeout(collabReconnectTimerRef.current);
        collabReconnectTimerRef.current = null;
      }
      socket?.close();
      if (socket && collabSocketRef.current === socket) {
        collabSocketRef.current = null;
      }
    };
  }, [activeWorkspaceId, authorizeBrowserSurfaces, loadWorkspaceData, selectedFilePath, collabReconnectNonce]);

  useEffect(() => {
    if (terminalReconnectTimerRef.current !== null) {
      window.clearTimeout(terminalReconnectTimerRef.current);
      terminalReconnectTimerRef.current = null;
    }

    terminalSocketRef.current?.close();
    terminalSocketRef.current = null;
    terminalResizeObserverRef.current?.disconnect();
    terminalResizeObserverRef.current = null;
    terminalFitRef.current = null;
    terminalRef.current?.dispose();
    terminalRef.current = null;
    setTerminalReadOnly(!canEditWorkspace);
    terminalReadOnlyRef.current = !canEditWorkspace;

    if (activeRightTab !== 'console' || !activeWorkspaceId) {
      return;
    }

    // The terminal should remain available whenever the runtime session exists,
    // even if preview-health checks currently mark the preview surface unhealthy.
    const runtimeSessionState = runtimeStatus?.session_state;
    if (runtimeSessionState !== 'running' && runtimeSessionState !== 'starting') {
      return;
    }

    let reconnectEnabled = true;
    let reconnectAttempts = 0;
    const scheduleReconnect = () => {
      if (!reconnectEnabled || terminalReconnectTimerRef.current !== null) {
        return;
      }
      // Exponential backoff: 1.5s, 3s, 6s, 12s, capped at 15s
      const delay = Math.min(1500 * Math.pow(2, reconnectAttempts), 15000);
      reconnectAttempts++;
      terminalReconnectTimerRef.current = window.setTimeout(() => {
        terminalReconnectTimerRef.current = null;
        setTerminalReconnectNonce((value) => value + 1);
      }, delay);
    };

    const terminalContainer = terminalContainerRef.current;
    if (!terminalContainer) {
      return;
    }

    const fitAddon = new FitAddon();
    const isReadOnlyInitial = !canEditWorkspace;
    const terminal = new XTerm({
      convertEol: true,
      cursorBlink: !isReadOnlyInitial,
      disableStdin: isReadOnlyInitial,
      fontFamily: 'var(--font-mono, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace)',
      fontSize: 12,
    });
    terminal.loadAddon(fitAddon);
    terminal.open(terminalContainer);
    fitAddon.fit();
    terminal.writeln('$ waiting for terminal output...');
    const focusTerminal = () => {
      terminal.focus();
    };
    terminalContainer.addEventListener('pointerdown', focusTerminal);

    terminalRef.current = terminal;
    terminalFitRef.current = fitAddon;

    const resizeObserver = new ResizeObserver(() => {
      try {
        fitAddon.fit();
      } catch {
        // ignore sizing errors during rapid layout changes
      }
      if (socket?.readyState === WebSocket.OPEN) {
        try {
          socket.send(JSON.stringify({ type: 'resize', cols: terminal.cols, rows: terminal.rows }));
        } catch {
          // ignore resize send failures during reconnects
        }
      }
    });
    resizeObserver.observe(terminalContainer);
    terminalResizeObserverRef.current = resizeObserver;

    let socket: WebSocket | null = null;

    const dataDisposable = terminal.onData((data) => {
      if (terminalReadOnlyRef.current) {
        return;
      }
      if (!socket || socket.readyState !== WebSocket.OPEN) {
        return;
      }
      try {
        socket.send(JSON.stringify({ type: 'input', data }));
      } catch {
        // ignore
      }
    });

    const connectTerminal = async () => {
      try {
        await authorizeBrowserSurfaces(activeWorkspaceId, ['runtime_pty']);
        if (!reconnectEnabled) {
          return;
        }

        const wsUrl = api.getUserSpaceRuntimePtyWebSocketUrl(activeWorkspaceId);
        socket = new WebSocket(wsUrl);
        terminalSocketRef.current = socket;

        socket.onopen = () => {
          terminal.clear();
          if (terminalReconnectTimerRef.current !== null) {
            window.clearTimeout(terminalReconnectTimerRef.current);
            terminalReconnectTimerRef.current = null;
          }
          try {
            fitAddon.fit();
          } catch {
            // ignore
          }
          try {
            socket?.send(JSON.stringify({ type: 'resize', cols: terminal.cols, rows: terminal.rows }));
          } catch {
            // ignore
          }
          terminal.focus();
        };

        socket.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as { type?: string; data?: string; read_only?: boolean; message?: string };
            if (payload.type === 'status') {
              const isReadOnly = Boolean(payload.read_only);
              terminalReadOnlyRef.current = isReadOnly;
              setTerminalReadOnly(isReadOnly);
              terminal.options.disableStdin = isReadOnly;
              terminal.options.cursorBlink = !isReadOnly;
              if (!isReadOnly) {
                setTimeout(() => terminal.focus(), 50);
              }
              const statusMessage = typeof payload.message === 'string' ? payload.message.trim() : '';
              if (statusMessage) {
                terminal.writeln(`[status] ${statusMessage}`);
              }
              return;
            }
            if (payload.type === 'output') {
              const workspaceRoot = `/data/_userspace/workspaces/${activeWorkspaceId}/files`;
              const output = (payload.data ?? '').split(workspaceRoot).join('<workspace>');
              terminal.write(output);
              return;
            }
          } catch {
            terminal.write(String(event.data ?? ''));
          }
        };

        socket.onclose = (closeEvent) => {
          if ([1000, 1001].includes(closeEvent.code)) {
            return;
          }
          if ([4401, 4403, 4404].includes(closeEvent.code)) {
            terminal.writeln('\r\n[status] Terminal access was rejected. Refresh runtime access and try again.');
            return;
          }
          if (reconnectEnabled) {
            terminal.writeln('\r\n[status] Terminal disconnected. Reconnecting...');
          }
          scheduleReconnect();
        };

        socket.onerror = () => {
          // onclose always fires after onerror; reconnect is handled there
        };
      } catch (err) {
        if (!reconnectEnabled) {
          return;
        }
        terminal.writeln('\r\n[status] Failed to authorize terminal access. Retrying...');
        setError(err instanceof Error ? err.message : 'Failed to authorize terminal access');
        scheduleReconnect();
      }
    };

    void connectTerminal();

    return () => {
      reconnectEnabled = false;
      if (terminalReconnectTimerRef.current !== null) {
        window.clearTimeout(terminalReconnectTimerRef.current);
        terminalReconnectTimerRef.current = null;
      }
      dataDisposable.dispose();
      socket?.close();
      if (socket && terminalSocketRef.current === socket) {
        terminalSocketRef.current = null;
      }
      if (terminalResizeObserverRef.current) {
        terminalResizeObserverRef.current.disconnect();
        terminalResizeObserverRef.current = null;
      }
      terminalContainer.removeEventListener('pointerdown', focusTerminal);
      terminalFitRef.current = null;
      terminalRef.current?.dispose();
      terminalRef.current = null;
    };
  }, [activeRightTab, activeWorkspaceId, authorizeBrowserSurfaces, canEditWorkspace, runtimeStatus?.session_state, terminalReconnectNonce]);

  useEffect(() => {
    const socket = collabSocketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    if (!activeWorkspaceId || !selectedFilePath) return;
    if (!canEditWorkspace || collabReadOnly) return;

    if (collabSuppressNextSendRef.current) {
      collabSuppressNextSendRef.current = false;
      return;
    }

    const timer = window.setTimeout(() => {
      try {
        socket.send(
          JSON.stringify({
            type: 'update',
            workspace_id: activeWorkspaceId,
            file_path: selectedFilePath,
            version: collabVersion,
            content: fileContent,
          })
        );
      } catch {
        // Ignore send errors; reconnect happens on workspace/file changes.
      }
    }, 250);

    return () => window.clearTimeout(timer);
  }, [activeWorkspaceId, canEditWorkspace, collabReadOnly, collabVersion, fileContent, selectedFilePath]);

  const handleStartCreateFile = useCallback((parentPath = '') => {
    const normalizedParent = normalizeWorkspacePath(parentPath);
    setNewFileParentPath(normalizedParent);
    setNewFileName(normalizedParent ? `${normalizedParent}/` : '');
    setDeleteConfirmFileId(null);
    setDeleteConfirmFolderPath(null);
    setRenamingFilePath(null);
    setRenamingFolderPath(null);
  }, []);

  const handleCreateNewFile = useCallback(async (path: string, parentPath: string = '') => {
    if (!activeWorkspaceId || !canEditWorkspace) return;

    const normalizedParent = normalizeWorkspacePath(parentPath);
    const normalizedInput = normalizeWorkspacePath(path);
    const nextPath = (() => {
      if (!normalizedInput) return '';
      if (!normalizedParent) return normalizedInput;
      if (normalizedInput === normalizedParent || normalizedInput.startsWith(`${normalizedParent}/`)) {
        return normalizedInput;
      }
      if (!normalizedInput.includes('/')) {
        return `${normalizedParent}/${normalizedInput}`;
      }
      return normalizedInput;
    })();

    if (!nextPath) {
      setError('File path is required');
      return;
    }

    try {
      await api.upsertUserSpaceFile(activeWorkspaceId, nextPath, {
        content: '',
        artifact_type: undefined,
      });
      setFileContentCache((current) => ({
        ...current,
        [nextPath]: {
          content: '',
          updatedAt: current[nextPath]?.updatedAt ?? '',
          artifactType: current[nextPath]?.artifactType ?? null,
        },
      }));
      setNewFileName(null);
      setNewFileParentPath('');
      await loadWorkspaceData(activeWorkspaceId);
      handleSelectFile(nextPath);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create file');
    }
  }, [activeWorkspaceId, canEditWorkspace, handleSelectFile, loadWorkspaceData]);

  const handleRenameFile = useCallback(async (oldPath: string, newPath: string) => {
    const normalizedNewPath = normalizeWorkspacePath(newPath);
    if (!activeWorkspaceId || !canEditWorkspace || !normalizedNewPath || normalizedNewPath === oldPath) {
      setRenamingFilePath(null);
      return;
    }
    try {
      const file = await api.getUserSpaceFile(activeWorkspaceId, oldPath);
      await api.upsertUserSpaceFile(activeWorkspaceId, normalizedNewPath, {
        content: file.content,
        artifact_type: file.artifact_type || undefined,
      });
      await api.deleteUserSpaceFile(activeWorkspaceId, oldPath);
      setRenamingFilePath(null);
      setFileContentCache((current) => {
        const next = { ...current };
        const currentValue = next[oldPath];
        if (currentValue) {
          next[normalizedNewPath] = currentValue;
        }
        delete next[oldPath];
        return next;
      });
      // Transfer changed/acknowledged markers from old path to new path.
      setChangedFiles((prev) => {
        if (!prev.has(oldPath)) return prev;
        const next = new Set(prev);
        next.delete(oldPath);
        next.add(normalizedNewPath);
        return next;
      });
      setAcknowledgedFiles((prev) => {
        if (!prev.has(oldPath)) return prev;
        const next = new Set(prev);
        next.delete(oldPath);
        next.add(normalizedNewPath);
        return next;
      });
      if (selectedFilePath === oldPath) {
        setSelectedFilePath(normalizedNewPath);
      }
      await loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename file');
    }
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData, selectedFilePath]);

  const handleRenameFolder = useCallback(async (oldFolderPath: string, newFolderPath: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) {
      setRenamingFolderPath(null);
      return;
    }

    const oldPrefix = normalizeWorkspacePath(oldFolderPath);
    const newPrefix = normalizeWorkspacePath(newFolderPath);

    if (!oldPrefix || !newPrefix || oldPrefix === newPrefix) {
      setRenamingFolderPath(null);
      return;
    }

    if (newPrefix.startsWith(`${oldPrefix}/`)) {
      setError('Cannot rename a folder into one of its own descendants');
      return;
    }

    const descendants = files.filter((file) => file.path.startsWith(`${oldPrefix}/`));
    if (descendants.length === 0) {
      setError('Folder is empty; no files to rename');
      setRenamingFolderPath(null);
      return;
    }

    const descendantPaths = new Set(descendants.map((file) => file.path));
    const existingPaths = new Set(files.map((file) => file.path));
    const moves = descendants.map((file) => ({
      oldPath: file.path,
      newPath: `${newPrefix}/${file.path.slice(oldPrefix.length + 1)}`,
    }));

    const conflictingPath = moves.find((move) => existingPaths.has(move.newPath) && !descendantPaths.has(move.newPath));
    if (conflictingPath) {
      setError(`Cannot rename folder because target file already exists: ${conflictingPath.newPath}`);
      return;
    }

    const createdPaths: string[] = [];
    try {
      for (const move of moves) {
        const sourceFile = await api.getUserSpaceFile(activeWorkspaceId, move.oldPath);
        await api.upsertUserSpaceFile(activeWorkspaceId, move.newPath, {
          content: sourceFile.content,
          artifact_type: sourceFile.artifact_type || undefined,
        });
        createdPaths.push(move.newPath);
      }
    } catch (err) {
      await Promise.allSettled(createdPaths.map((path) => api.deleteUserSpaceFile(activeWorkspaceId, path)));
      setError(err instanceof Error ? `Failed to rename folder: ${err.message}` : 'Failed to rename folder');
      return;
    }

    const deleteResults = await Promise.allSettled(moves.map((move) => api.deleteUserSpaceFile(activeWorkspaceId, move.oldPath)));
    const deleteFailures = deleteResults.filter((result) => result.status === 'rejected').length;

    setRenamingFolderPath(null);
    setRenameValue('');
    setExpandedFolders((current) => {
      const next = new Set<string>();
      for (const path of current) {
        if (path === oldPrefix || path.startsWith(`${oldPrefix}/`)) {
          const suffix = path.slice(oldPrefix.length);
          next.add(`${newPrefix}${suffix}`);
        } else {
          next.add(path);
        }
      }
      return next;
    });

    if (selectedFilePath.startsWith(`${oldPrefix}/`)) {
      setSelectedFilePath(`${newPrefix}/${selectedFilePath.slice(oldPrefix.length + 1)}`);
    }

    // Transfer changed/acknowledged markers from old paths to new paths.
    setChangedFiles((prev) => {
      const next = new Set(prev);
      for (const move of moves) {
        if (next.has(move.oldPath)) { next.delete(move.oldPath); next.add(move.newPath); }
      }
      return next;
    });
    setAcknowledgedFiles((prev) => {
      const next = new Set(prev);
      for (const move of moves) {
        if (next.has(move.oldPath)) { next.delete(move.oldPath); next.add(move.newPath); }
      }
      return next;
    });

    await loadWorkspaceData(activeWorkspaceId);

    if (deleteFailures > 0) {
      setError(`Folder renamed, but ${deleteFailures} source file(s) could not be removed`);
    }
  }, [activeWorkspaceId, canEditWorkspace, files, loadWorkspaceData, selectedFilePath]);

  const handleDeleteFile = useCallback(async (filePath: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    try {
      await api.deleteUserSpaceFile(activeWorkspaceId, filePath);
      setFileContentCache((current) => {
        const next = { ...current };
        delete next[filePath];
        return next;
      });
      setDeleteConfirmFileId(null);
      setChangedFiles((prev) => { const next = new Set(prev); next.delete(filePath); return next; });
      setAcknowledgedFiles((prev) => { const next = new Set(prev); next.delete(filePath); return next; });
      if (selectedFilePath === filePath) {
        setSelectedFilePath('');
        setFileContent('');
        setFileDirty(false);
      }
      await loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete file');
    }
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData, selectedFilePath]);

  const handleDeleteFolder = useCallback(async (folderPath: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;

    const normalizedFolderPath = normalizeWorkspacePath(folderPath);
    const descendants = files.filter((file) => file.path.startsWith(`${normalizedFolderPath}/`));
    if (descendants.length === 0) {
      setDeleteConfirmFolderPath(null);
      setError('Folder is empty; no files to delete');
      return;
    }

    const results = await Promise.allSettled(
      descendants.map((file) => api.deleteUserSpaceFile(activeWorkspaceId, file.path))
    );
    const failures = results.filter((result) => result.status === 'rejected').length;

    setDeleteConfirmFolderPath(null);
    // Clean up changed/acknowledged markers for all deleted descendants.
    const deletedPaths = new Set(descendants.map((file) => file.path));
    setChangedFiles((prev) => { const next = new Set(prev); for (const p of deletedPaths) next.delete(p); return next; });
    setAcknowledgedFiles((prev) => { const next = new Set(prev); for (const p of deletedPaths) next.delete(p); return next; });
    if (selectedFilePath.startsWith(`${normalizedFolderPath}/`)) {
      setSelectedFilePath('');
      setFileContent('');
      setFileDirty(false);
    }
    await loadWorkspaceData(activeWorkspaceId);

    if (failures > 0) {
      setError(`Deleted folder contents with ${failures} failure(s)`);
    }
  }, [activeWorkspaceId, canEditWorkspace, files, loadWorkspaceData, selectedFilePath]);

  const handleToggleFolder = useCallback((folderPath: string) => {
    setExpandedFolders((current) => {
      const next = new Set(current);
      if (next.has(folderPath)) {
        next.delete(folderPath);
      } else {
        next.add(folderPath);
      }
      return next;
    });
  }, []);

  const handleDeleteWorkspace = useCallback(async (workspaceId: string) => {
    setError(null);
    setDeleteConfirmWorkspaceId(null);

    try {
      const task = await api.queueUserSpaceWorkspaceDelete(workspaceId);
      setDeletingWorkspaceTasks((current) => ({
        ...current,
        [workspaceId]: task,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue workspace delete');
    }
  }, []);

  useEffect(() => {
    const tasks = Object.values(deletingWorkspaceTasks);
    if (tasks.length === 0) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollDeleteTasks = async () => {
      if (pollInFlight) {
        return;
      }
      pollInFlight = true;

      try {
        const results = await Promise.all(tasks.map(async (task) => {
          try {
            const status = await api.getUserSpaceWorkspaceDeleteTask(task.task_id);
            return { task, status, error: null as Error | null };
          } catch (error) {
            return { task, status: null as UserSpaceWorkspaceDeleteTask | null, error: error as Error };
          }
        }));

        if (cancelled) {
          return;
        }

        const terminalWorkspaceIds = new Set<string>();
        const updatedTasks: Record<string, UserSpaceWorkspaceDeleteTask> = {};
        let nextError: string | null = null;

        for (const result of results) {
          if (result.status) {
            if (isWorkspaceDeleteTaskTerminal(result.status.phase)) {
              terminalWorkspaceIds.add(result.status.workspace_id);
              if (result.status.phase === 'failed' && !nextError) {
                nextError = result.status.error?.trim() || `Failed to delete ${result.status.workspace_name}`;
              }
            } else {
              updatedTasks[result.status.workspace_id] = result.status;
            }
            continue;
          }

          if (result.error instanceof ApiError && result.error.status === 404) {
            terminalWorkspaceIds.add(result.task.workspace_id);
            continue;
          }

          if (!nextError && result.error instanceof Error) {
            nextError = result.error.message;
          }
        }

        setDeletingWorkspaceTasks((current) => {
          const next = { ...current };
          for (const workspaceId of terminalWorkspaceIds) {
            delete next[workspaceId];
          }
          for (const [workspaceId, task] of Object.entries(updatedTasks)) {
            next[workspaceId] = task;
          }
          return next;
        });

        if (nextError) {
          setError(nextError);
        }

        if (terminalWorkspaceIds.size > 0) {
          try {
            await loadWorkspaces();
          } catch {
            // Best-effort refresh after delete.
          }

          // Switch away from a successfully-deleted workspace only after
          // the backend confirms the delete completed (not at queue time).
          for (const deletedId of terminalWorkspaceIds) {
            if (deletedId !== activeWorkspaceIdRef.current) {
              continue;
            }
            // Check that this workspace was actually deleted (completed, not failed).
            const matchingResult = results.find(
              (r) => r.status?.workspace_id === deletedId || r.task.workspace_id === deletedId,
            );
            const wasDeleted =
              matchingResult?.status?.phase === 'completed'
              || (matchingResult?.error instanceof ApiError && matchingResult.error.status === 404);
            if (!wasDeleted) {
              continue;
            }

            setRuntimeStatus(null);
            setPreviewLiveDataConnections([]);
            const fallback = workspacesRef.current.find((ws) => ws.id !== deletedId)?.id ?? null;
            setActiveWorkspaceId(fallback);

            if (!fallback) {
              clearCookieValue(lastWorkspaceCookieNameRef.current);
              setFileBrowserEntries([]);
              setFiles([]);
              setSnapshots([]);
              setSnapshotBranches([]);
              setCurrentSnapshotId(null);
              setCurrentSnapshotBranchId(null);
              setFileContentCache({});
              setFileContent('');
              setSelectedFilePath('');
              setFileDirty(false);
            }
            break;
          }
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollDeleteTasks();
    const intervalId = window.setInterval(() => {
      void pollDeleteTasks();
    }, 1000);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [deletingWorkspaceTasks, loadWorkspaces]);

  const handleOpenMembersModal = useCallback(async () => {
    if (!activeWorkspace || !isOwner) return;
    setPendingMembers(activeWorkspace.members.filter((m) => m.user_id !== activeWorkspace.owner_user_id));
    try {
      const users = await api.listUsers();
      setAllUsers(users);
    } catch {
      setAllUsers([]);
    }
    setShowMembersModal(true);
  }, [activeWorkspace, isOwner]);

  const handleSaveMembers = useCallback(async (members: Member[]) => {
    if (!activeWorkspace || !isOwner) return;
    setSavingMembers(true);
    try {
      const updated = await api.updateUserSpaceWorkspaceMembers(activeWorkspace.id, { members });
      setWorkspaces((current) => current.map((ws) => ws.id === updated.id ? updated : ws));
      setShowMembersModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update members');
      throw err;
    } finally {
      setSavingMembers(false);
    }
  }, [activeWorkspace, isOwner]);

  const handleOpenAgentAccessModal = useCallback(async () => {
    if (!activeWorkspace || !isOwner) return;
    setShowAgentAccessModal(true);
    setAgentGrantsLoading(true);
    try {
      const limit = Math.max(workspacesTotal || workspaces.length || 50, 50);
      const [grants, workspacePage] = await Promise.all([
        api.listUserSpaceWorkspaceAgentGrants(activeWorkspace.id),
        api.listUserSpaceWorkspaces(0, limit),
      ]);
      setAgentGrants(sortWorkspaceAgentGrants(grants));
      setAgentGrantWorkspaces(workspacePage.items);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load agent access settings');
    } finally {
      setAgentGrantsLoading(false);
    }
  }, [activeWorkspace, isOwner, workspaces.length, workspacesTotal]);

  const handleUpsertAgentGrant = useCallback(async (request: UpsertWorkspaceAgentGrantRequest) => {
    if (!activeWorkspace || !isOwner) return;
    setSavingAgentGrantTargetId(request.target_workspace_id);
    try {
      const updated = await api.upsertUserSpaceWorkspaceAgentGrant(activeWorkspace.id, request);
      setAgentGrants((current) => sortWorkspaceAgentGrants([
        ...current.filter((grant) => grant.target_workspace_id !== updated.target_workspace_id),
        updated,
      ]));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save agent access');
      throw err;
    } finally {
      setSavingAgentGrantTargetId(null);
    }
  }, [activeWorkspace, isOwner]);

  const handleRevokeAgentGrant = useCallback(async (targetWorkspaceId: string) => {
    if (!activeWorkspace || !isOwner) return;
    setRevokingAgentGrantTargetId(targetWorkspaceId);
    try {
      await api.revokeUserSpaceWorkspaceAgentGrant(activeWorkspace.id, targetWorkspaceId);
      setAgentGrants((current) => current.filter((grant) => grant.target_workspace_id !== targetWorkspaceId));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke agent access');
      throw err;
    } finally {
      setRevokingAgentGrantTargetId(null);
    }
  }, [activeWorkspace, isOwner]);

  const loadWorkspaceEnvVars = useCallback(async (workspaceId: string) => {
    const vars = await api.listUserSpaceWorkspaceEnvVars(workspaceId);
    setEnvVars(vars);
    return vars;
  }, []);

  const mergeWorkspaceEnvVar = useCallback((envVar: UserSpaceWorkspaceEnvVar, previousKey?: string) => {
    setEnvVars((current) => {
      const keysToReplace = new Set([previousKey, envVar.key].filter(Boolean));
      const next = current.filter((item) => !keysToReplace.has(item.key));
      return [...next, envVar].sort((a, b) => a.key.localeCompare(b.key));
    });
  }, []);

  const handleOpenEnvVarsModal = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner) return;
    setShowEnvVarsModal(true);
    setEnvVarsLoading(true);
    try {
      await loadWorkspaceEnvVars(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load environment variables');
    } finally {
      setEnvVarsLoading(false);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceEnvVars]);

  const loadObjectStorageConfig = useCallback(async (workspaceId: string) => {
    const config = await api.getUserSpaceObjectStorageConfig(workspaceId);
    setObjectStorageConfig(config);
    return config;
  }, []);

  const handleObjectStorageWizardSaved = useCallback((config: UserSpaceObjectStorageConfig) => {
    setObjectStorageConfig(config);
    setShowObjectStorageWizard(false);
    setEditingObjectStorageBucket(null);
    setSuccess('Object storage updated.');
    setTimeout(() => setSuccess(null), 3000);
  }, []);

  const handleDeleteObjectStorageBucket = useCallback(async (bucketName: string) => {
    if (!activeWorkspaceId || !isOwner) return;
    setDeletingObjectStorageBucket(bucketName);
    try {
      await api.deleteUserSpaceObjectStorageBucket(activeWorkspaceId, bucketName);
      const next = await loadObjectStorageConfig(activeWorkspaceId);
      setObjectStorageConfig(next);
      setSuccess('Bucket deleted.');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete bucket');
    } finally {
      setDeletingObjectStorageBucket(null);
    }
  }, [activeWorkspaceId, isOwner, loadObjectStorageConfig]);

  const handleCreateEnvVar = useCallback(async (request: UpsertUserSpaceWorkspaceEnvVarRequest) => {
    if (!activeWorkspaceId || !isOwner) return;
    setSavingEnvVar(true);
    try {
      const upserted = await api.upsertUserSpaceWorkspaceEnvVar(activeWorkspaceId, request);
      try {
        await loadWorkspaceEnvVars(activeWorkspaceId);
      } catch {
        mergeWorkspaceEnvVar(upserted, request.key);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save environment variable');
    } finally {
      setSavingEnvVar(false);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceEnvVars, mergeWorkspaceEnvVar]);

  const handleUpdateEnvVar = useCallback(async (request: UpsertUserSpaceWorkspaceEnvVarRequest) => {
    if (!activeWorkspaceId || !isOwner) return;
    setSavingEnvVar(true);
    try {
      const upserted = await api.upsertUserSpaceWorkspaceEnvVar(activeWorkspaceId, request);
      try {
        await loadWorkspaceEnvVars(activeWorkspaceId);
      } catch {
        mergeWorkspaceEnvVar(upserted, request.key);
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update environment variable');
    } finally {
      setSavingEnvVar(false);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceEnvVars, mergeWorkspaceEnvVar]);

  const handleDeleteEnvVar = useCallback(async (key: string) => {
    if (!activeWorkspaceId || !isOwner) return;
    setSavingEnvVar(true);
    try {
      await api.deleteUserSpaceWorkspaceEnvVar(activeWorkspaceId, key);
      try {
        await loadWorkspaceEnvVars(activeWorkspaceId);
      } catch {
        setEnvVars((current) => current.filter((item) => item.key !== key));
      }
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete environment variable');
    } finally {
      setSavingEnvVar(false);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceEnvVars]);

  const handleOpenMountsModal = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner) return;
    setShowMountsModal(true);
    setMountsModalTab('mounts');
    setMountsLoading(true);
    setCreateMountSourceId('');
    setCreateMountSourcePath('');
    setCreateMountRootSourcePath('');
    setCreateMountBrowserPath('');
    setCreateMountTargetBrowserPath('');
    setCreateMountTargetPath('');
    setCreateMountStagedSourceDirectories({});
    setCreateMountStagedTargetDirectories([]);
    setCreateMountDescription('');
    setCreateMountSyncMode('merge');
    setCreateMountActiveSourceTab('');
    setMountSyncPreview(null);
    setMountSyncPreviewMount(null);
    setMountSyncPreviewIntent(null);
    setMountSyncPreviewNextSyncMode(null);
    setShowObjectStorageWizard(false);
    setEditingObjectStorageBucket(null);
    setObjectStorageLoading(true);
    try {
      const [mountList, sources] = await Promise.all([
        api.listWorkspaceMounts(activeWorkspaceId),
        api.listMountableSources(activeWorkspaceId),
        loadObjectStorageConfig(activeWorkspaceId).catch(() => null),
      ]);
      setMounts(mountList);
      setMountableSources(sources);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load mounts');
    } finally {
      setMountsLoading(false);
      setObjectStorageLoading(false);
    }
  }, [activeWorkspaceId, isOwner, loadObjectStorageConfig]);

  const handleCloseMountsModal = useCallback(() => {
    setShowMountsModal(false);
    setDeletingMountId(null);
    setMountSyncPreview(null);
    setMountSyncPreviewMount(null);
    setMountSyncPreviewIntent(null);
    setMountSyncPreviewNextSyncMode(null);
    setShowObjectStorageWizard(false);
    setEditingObjectStorageBucket(null);
  }, []);

  const createMountEffectiveSourcePath = useMemo(() => {
    if (createMountSourcePath) {
      return createMountSourcePath;
    }
    if (!createMountSourceId || !createMountBrowserPath) {
      return '';
    }
    return browserPathToSourcePath(createMountBrowserPath);
  }, [createMountBrowserPath, createMountSourceId, createMountSourcePath]);

  const createMountEffectiveTargetPath = useMemo(() => {
    const trimmedTargetPath = createMountTargetPath.trim();
    if (trimmedTargetPath) {
      return trimmedTargetPath;
    }
    if (!createMountTargetBrowserPath) {
      return '';
    }
    return browserPathToWorkspaceMountTargetPath(createMountTargetBrowserPath);
  }, [createMountTargetBrowserPath, createMountTargetPath]);

  const createMountSelectedSource = useMemo(() => (
    mountableSources.find((source) => (
      source.mount_source_id === createMountSourceId && source.source_path === createMountRootSourcePath
    )) ?? null
  ), [createMountRootSourcePath, createMountSourceId, mountableSources]);

  const isCreateMountDisabled = useMemo(() => (
    savingMount
    || !createMountSourceId
    || !createMountEffectiveSourcePath
    || !createMountEffectiveTargetPath
    || createMountEffectiveTargetPath === '/workspace'
  ), [createMountEffectiveSourcePath, createMountEffectiveTargetPath, createMountSourceId, savingMount]);

  const handleCreateMount = useCallback(async () => {
    if (
      !activeWorkspaceId
      || !isOwner
      || !createMountSourceId
      || !createMountEffectiveSourcePath
      || !createMountEffectiveTargetPath
    ) {
      return;
    }
    setSavingMount(true);
    const sourceStageKey = getMountSourceBrowserStageKey(createMountSourceId, createMountRootSourcePath);
    const sourceDirectoryToCreate = resolveMountDirectoryToCreate(
      createMountBrowserPath,
      createMountStagedSourceDirectories[sourceStageKey] ?? [],
      browserPathToSourcePath,
    );
    const targetDirectoryToCreate = resolveMountDirectoryToCreate(
      createMountTargetBrowserPath,
      createMountStagedTargetDirectories,
      browserPathToWorkspaceMountTargetPath,
    );
    try {
      const created = await api.createWorkspaceMount(activeWorkspaceId, {
        mount_source_id: createMountSourceId,
        source_path: createMountEffectiveSourcePath,
        target_path: createMountEffectiveTargetPath,
        source_directory_to_create: sourceDirectoryToCreate,
        target_directory_to_create: targetDirectoryToCreate,
        auto_sync_enabled: false,
        sync_mode: createMountSelectedSource?.source_type === 'ssh' ? createMountSyncMode : 'merge',
        description: createMountDescription.trim() || null,
      });
      setMounts((prev) => [...prev, created]);
      setCreateMountSourceId('');
      setCreateMountSourcePath('');
      setCreateMountRootSourcePath('');
      setCreateMountBrowserPath('');
      setCreateMountTargetBrowserPath('');
      setCreateMountTargetPath('');
      setCreateMountStagedSourceDirectories({});
      setCreateMountStagedTargetDirectories([]);
      setCreateMountDescription('');
      setCreateMountSyncMode('merge');
      setError(created.sync_notice || null);
      if (targetDirectoryToCreate) {
        await loadWorkspaceData(activeWorkspaceId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create mount');
    } finally {
      setSavingMount(false);
    }
  }, [activeWorkspaceId, createMountBrowserPath, createMountDescription, createMountEffectiveSourcePath, createMountEffectiveTargetPath, createMountRootSourcePath, createMountSelectedSource?.source_type, createMountSourceId, createMountStagedSourceDirectories, createMountStagedTargetDirectories, createMountSyncMode, createMountTargetBrowserPath, isOwner, loadWorkspaceData]);

  const handleSaveMountDescription = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner || !editingMountDescriptionId) return;
    setSavingMountDescriptionId(editingMountDescriptionId);
    try {
      const updated = await api.updateWorkspaceMount(activeWorkspaceId, editingMountDescriptionId, {
        description: editingMountDescriptionDraft.trim() || null,
      });
      setMounts((prev) => prev.map((mount) => mount.id === updated.id ? updated : mount));
      setEditingMountDescriptionId(null);
      setEditingMountDescriptionDraft('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update mount description');
    } finally {
      setSavingMountDescriptionId(null);
    }
  }, [activeWorkspaceId, editingMountDescriptionDraft, editingMountDescriptionId, isOwner]);

  const handleDeleteMount = useCallback(async (mountId: string) => {
    if (!activeWorkspaceId || !isOwner) return;
    setDeletingMountId(mountId);
    try {
      await api.deleteWorkspaceMount(activeWorkspaceId, mountId);
      setMounts((prev) => prev.filter((m) => m.id !== mountId));
      void loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete mount');
    } finally {
      setDeletingMountId(null);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceData]);

  const handleEjectMount = useCallback(async (mount: WorkspaceMount) => {
    if (!activeWorkspaceId || !isOwner) return;
    setSavingMountWatchId(mount.id);
    try {
      const updated = await api.updateWorkspaceMount(activeWorkspaceId, mount.id, {
        enabled: false,
      });
      setMounts((prev) => prev.map((m) => (m.id === mount.id ? updated : m)));
      setError(null);
      void loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to eject mount');
    } finally {
      setSavingMountWatchId(null);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceData]);

  const handleRemount = useCallback(async (mount: WorkspaceMount) => {
    if (!activeWorkspaceId || !isOwner) return;
    setSavingMountWatchId(mount.id);
    try {
      const updated = await api.updateWorkspaceMount(activeWorkspaceId, mount.id, {
        enabled: true,
      });
      setMounts((prev) => prev.map((m) => (m.id === mount.id ? updated : m)));
      setError(null);
      void loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to remount');
    } finally {
      setSavingMountWatchId(null);
    }
  }, [activeWorkspaceId, isOwner, loadWorkspaceData]);

  const applyMountSyncFailure = useCallback((mountId: string, err: unknown, fallback: string) => {
    const message = getApiErrorMessage(err, fallback);
    setMounts((prev) => prev.map((mount) => (
      mount.id === mountId
        ? {
            ...mount,
            sync_status: 'error',
            sync_notice: null,
            last_sync_error: message,
          }
        : mount
    )));
    setError(message);
  }, []);

  const handleSyncMount = useCallback(async (mount: WorkspaceMount) => {
    if (!activeWorkspaceId || !isOwner) return;
    if (isDestructiveMountSyncMode(mount.sync_mode)) {
      setMountSyncPreviewIntent('sync');
      setMountSyncPreviewNextSyncMode(mount.sync_mode);
      setPreviewingMountId(mount.id);
      try {
        const preview = await api.previewWorkspaceMountSync(activeWorkspaceId, mount.id);
        setMountSyncPreview(preview);
        setMountSyncPreviewMount(mount);
        setError(preview.sync_notice || null);
      } catch (err) {
        applyMountSyncFailure(mount.id, err, 'Failed to preview mount sync');
      } finally {
        setPreviewingMountId(null);
      }
      return;
    }

    setSyncingMountId(mount.id);
    try {
      const result = await api.syncWorkspaceMount(activeWorkspaceId, mount.id);
      setMounts((prev) => prev.map((m) => m.id === mount.id ? {
        ...m,
        sync_mode: result.sync_mode,
        sync_status: result.sync_status,
        sync_backend: result.sync_backend,
        sync_notice: result.sync_notice,
        last_sync_error: result.last_sync_error,
      } : m));
      setError(result.last_sync_error || result.sync_notice || null);
    } catch (err) {
      applyMountSyncFailure(mount.id, err, 'Failed to sync mount');
    } finally {
      setSyncingMountId(null);
    }
  }, [activeWorkspaceId, applyMountSyncFailure, isOwner]);

  const handleCloseMountSyncPreview = useCallback(() => {
    setMountSyncPreview(null);
    setMountSyncPreviewMount(null);
    setMountSyncPreviewIntent(null);
    setMountSyncPreviewNextSyncMode(null);
  }, []);

  const handleConfirmMountSyncPreview = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner || !mountSyncPreview || !mountSyncPreviewMount || !mountSyncPreviewIntent) return;
    if (mountSyncPreviewIntent === 'sync') {
      setSyncingMountId(mountSyncPreviewMount.id);
      try {
        const result = await api.syncWorkspaceMount(activeWorkspaceId, mountSyncPreviewMount.id, {
          preview_token: mountSyncPreview.preview_token,
        });
        setMounts((prev) => prev.map((m) => m.id === mountSyncPreviewMount.id ? {
          ...m,
          sync_mode: result.sync_mode,
          sync_status: result.sync_status,
          sync_backend: result.sync_backend,
          sync_notice: result.sync_notice,
          last_sync_error: result.last_sync_error,
        } : m));
        handleCloseMountSyncPreview();
        setError(result.last_sync_error || result.sync_notice || null);
      } catch (err) {
        handleCloseMountSyncPreview();
        applyMountSyncFailure(mountSyncPreviewMount.id, err, 'Failed to sync mount');
      } finally {
        setSyncingMountId(null);
      }
      return;
    }

    if (mountSyncPreviewIntent === 'enable-auto') {
      setSavingMountWatchId(mountSyncPreviewMount.id);
      try {
        const updated = await api.updateWorkspaceMount(activeWorkspaceId, mountSyncPreviewMount.id, {
          auto_sync_enabled: true,
          destructive_auto_sync_preview_token: mountSyncPreview.preview_token,
        });
        setMounts((prev) => prev.map((m) => (m.id === mountSyncPreviewMount.id ? updated : m)));
        handleCloseMountSyncPreview();
        setError(updated.last_sync_error || updated.sync_notice || null);
      } catch (err) {
        handleCloseMountSyncPreview();
        setError(err instanceof Error ? err.message : 'Failed to enable auto-sync');
      } finally {
        setSavingMountWatchId(null);
      }
      return;
    }

    if (!mountSyncPreviewNextSyncMode) return;
    setSavingMountWatchId(mountSyncPreviewMount.id);
    try {
      const updated = await api.updateWorkspaceMount(activeWorkspaceId, mountSyncPreviewMount.id, {
        sync_mode: mountSyncPreviewNextSyncMode,
        destructive_auto_sync_preview_token: mountSyncPreview.preview_token,
      });
      setMounts((prev) => prev.map((m) => (m.id === mountSyncPreviewMount.id ? updated : m)));
      handleCloseMountSyncPreview();
      setError(updated.last_sync_error || updated.sync_notice || null);
    } catch (err) {
      handleCloseMountSyncPreview();
      setError(err instanceof Error ? err.message : 'Failed to update mount sync mode');
    } finally {
      setSavingMountWatchId(null);
    }
  }, [activeWorkspaceId, applyMountSyncFailure, handleCloseMountSyncPreview, isOwner, mountSyncPreview, mountSyncPreviewIntent, mountSyncPreviewMount, mountSyncPreviewNextSyncMode]);

  const handleToggleMountAutoSync = useCallback(async (mount: WorkspaceMount, enabled: boolean) => {
    if (!activeWorkspaceId || !isOwner) return;
    if (enabled && isDestructiveMountSyncMode(mount.sync_mode)) {
      setMountSyncPreviewIntent('enable-auto');
      setMountSyncPreviewNextSyncMode(mount.sync_mode);
      setPreviewingMountId(mount.id);
      try {
        const preview = await api.previewWorkspaceMountSync(activeWorkspaceId, mount.id);
        setMountSyncPreview(preview);
        setMountSyncPreviewMount(mount);
        setError(preview.sync_notice || null);
      } catch (err) {
        setMountSyncPreviewIntent(null);
        setMountSyncPreviewNextSyncMode(null);
        applyMountSyncFailure(mount.id, err, 'Failed to preview auto-sync');
      } finally {
        setPreviewingMountId(null);
      }
      return;
    }

    setSavingMountWatchId(mount.id);
    try {
      const updated = await api.updateWorkspaceMount(activeWorkspaceId, mount.id, {
        auto_sync_enabled: enabled,
      });
      setMounts((prev) => prev.map((m) => (m.id === mount.id ? updated : m)));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update mount watch mode');
    } finally {
      setSavingMountWatchId(null);
    }
  }, [activeWorkspaceId, applyMountSyncFailure, isOwner]);

  const handleUpdateMountSyncMode = useCallback(async (mount: WorkspaceMount, syncMode: WorkspaceMountSyncMode) => {
    if (!activeWorkspaceId || !isOwner) return;
    if (mount.auto_sync_enabled && isDestructiveMountSyncMode(syncMode)) {
      setMountSyncPreviewIntent('update-auto-sync-mode');
      setMountSyncPreviewNextSyncMode(syncMode);
      setPreviewingMountId(mount.id);
      try {
        const preview = await api.previewWorkspaceMountSync(activeWorkspaceId, mount.id, {
          sync_mode: syncMode,
        });
        setMountSyncPreview(preview);
        setMountSyncPreviewMount(mount);
        setError(preview.sync_notice || null);
      } catch (err) {
        setMountSyncPreviewIntent(null);
        setMountSyncPreviewNextSyncMode(null);
        applyMountSyncFailure(mount.id, err, 'Failed to preview mount sync mode');
      } finally {
        setPreviewingMountId(null);
      }
      return;
    }

    setSavingMountWatchId(mount.id);
    try {
      const updated = await api.updateWorkspaceMount(activeWorkspaceId, mount.id, {
        sync_mode: syncMode,
      });
      setMounts((prev) => prev.map((m) => (m.id === mount.id ? updated : m)));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update mount sync mode');
    } finally {
      setSavingMountWatchId(null);
    }
  }, [activeWorkspaceId, applyMountSyncFailure, isOwner]);

  const browseWorkspaceMountTargetPath = useCallback(async (browserPath: string): Promise<BrowseResponse> => {
    const normalizedBrowserPath = normalizeMountBrowserPath(browserPath);
    const relativePath = normalizedBrowserPath === '/' ? '' : normalizedBrowserPath.slice(1);

    const entries: DirectoryEntry[] = fileBrowserEntries
      .filter((entry) => {
        const normalizedEntryPath = normalizeWorkspacePath(entry.path);
        const entryParentPath = normalizedEntryPath.includes('/')
          ? normalizedEntryPath.slice(0, normalizedEntryPath.lastIndexOf('/'))
          : '';
        return entryParentPath === relativePath;
      })
      .map((entry) => {
        const normalizedEntryPath = normalizeWorkspacePath(entry.path);
        const entryName = normalizedEntryPath.split('/').pop() || normalizedEntryPath;
        return {
          name: entryName,
          path: sourcePathToBrowserPath(normalizedEntryPath),
          is_dir: entry.entry_type === 'directory',
          size: entry.size_bytes,
        };
      })
      .sort((left, right) => {
        if (left.is_dir !== right.is_dir) {
          return left.is_dir ? -1 : 1;
        }
        return left.name.localeCompare(right.name);
      });

    return {
      path: normalizedBrowserPath,
      entries,
      error: undefined,
    };
  }, [fileBrowserEntries]);

  const handleStartWorkspaceRename = useCallback((workspace: UserSpaceWorkspace) => {
    setDeleteConfirmWorkspaceId(null);
    setEditingWorkspaceNameId(workspace.id);
    setWorkspaceNameDraft(workspace.name);
    setError(null);
  }, []);

  const handleCancelWorkspaceRename = useCallback(() => {
    setEditingWorkspaceNameId(null);
    setWorkspaceNameDraft('');
  }, []);

  const loadShareLinkStatus = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) {
      setShareLinkStatus(null);
      return;
    }

    setLoadingShareStatus(true);
    try {
      const status = await api.getUserSpaceWorkspaceShareLinkStatus(activeWorkspaceId);
      setShareLinkStatus(status);
      setShareSlugDraft(status.share_slug ?? getDefaultShareSlug(activeWorkspace?.name));
      setShareSlugAvailable(null);
      setShareAccessMode(status.share_access_mode ?? 'token');
      setShareSelectedUserIdsDraft(status.selected_user_ids ?? []);
      setShareSelectedLdapGroupsDraft(status.selected_ldap_groups ?? []);
      setSharePasswordDraft('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load share link state');
      setShareLinkStatus(null);
    } finally {
      setLoadingShareStatus(false);
    }
  }, [activeWorkspace?.name, activeWorkspaceId, canEditWorkspace]);

  useEffect(() => {
    if (!showShareModal) return;
    void loadShareLinkStatus();
  }, [loadShareLinkStatus, showShareModal]);

  const activeShareLinkStatus = useMemo(() => {
    if (!shareLinkStatus || !activeWorkspaceId) {
      return null;
    }
    return shareLinkStatus.workspace_id === activeWorkspaceId ? shareLinkStatus : null;
  }, [activeWorkspaceId, shareLinkStatus]);

  useEffect(() => {
    if (typeof window === 'undefined') return;
    if (!activeWorkspaceId) {
      setShareLinkType('anonymous');
      return;
    }
    const stored = window.localStorage.getItem(getShareLinkTypeStorageKey(activeWorkspaceId));
    if (stored === 'named' || stored === 'anonymous' || stored === 'subdomain') {
      setShareLinkType(stored);
      return;
    }
    setShareLinkType('anonymous');
  }, [activeWorkspaceId]);

  useEffect(() => {
    if (typeof window === 'undefined' || !activeWorkspaceId) return;
    window.localStorage.setItem(getShareLinkTypeStorageKey(activeWorkspaceId), shareLinkType);
  }, [activeWorkspaceId, shareLinkType]);

  const handleOpenShareModal = useCallback(() => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setShareSlugDraft(activeShareLinkStatus?.share_slug ?? getDefaultShareSlug(activeWorkspace?.name));
    setShareSlugAvailable(null);
    setShareAccessMode(activeShareLinkStatus?.share_access_mode ?? 'token');
    setShareSelectedUserIdsDraft(activeShareLinkStatus?.selected_user_ids ?? []);
    setShareSelectedLdapGroupsDraft(activeShareLinkStatus?.selected_ldap_groups ?? []);
    setSharePasswordDraft('');
    setShareLdapGroupDraft('');
    setAutoCreateShareLinkAttempted(false);
    void api.listUsers().then(setAllUsers).catch(() => setAllUsers([]));
    void (async () => {
      setLoadingLdapGroups(true);
      try {
        const ldapData = await api.getLdapConfig();
        let groups = Array.isArray(ldapData.discovered_groups)
          ? ldapData.discovered_groups
          : [];

        if (groups.length === 0) {
          const discovered = await api.discoverLdapWithStoredCredentials();
          if (discovered.success && Array.isArray(discovered.groups)) {
            groups = discovered.groups;
          }
        }

        const normalized = groups
          .filter((group): group is { dn: string; name: string } => Boolean(group?.dn))
          .filter(
            (group, index, all) =>
              all.findIndex((candidate) => candidate.dn === group.dn) === index,
          );
        setLdapDiscoveredGroups(normalized);
      } catch {
        setLdapDiscoveredGroups([]);
      } finally {
        setLoadingLdapGroups(false);
      }
    })();
    setShowShareModal(true);
  }, [
    activeWorkspace?.name,
    activeWorkspaceId,
    activeShareLinkStatus?.selected_ldap_groups,
    activeShareLinkStatus?.selected_user_ids,
    activeShareLinkStatus?.share_access_mode,
    activeShareLinkStatus?.share_slug,
    canEditWorkspace,
  ]);

  const handleEnsureShareLink = useCallback(async (rotateToken = false) => {
    if (!activeWorkspaceId || !canEditWorkspace) return null;

    setSharingWorkspace(true);
    if (rotateToken) {
      setRotatingShareLink(true);
    }

    try {
      const link = await api.createUserSpaceWorkspaceShareLink(activeWorkspaceId, rotateToken);
      const nextStatus: UserSpaceWorkspaceShareLinkStatus = {
        workspace_id: link.workspace_id,
        has_share_link: true,
        owner_username: link.owner_username,
        share_slug: link.share_slug,
        share_token: link.share_token,
        share_url: link.share_url,
        anonymous_share_url: link.anonymous_share_url,
        subdomain_share_url: link.subdomain_share_url,
        subdomain_share_enabled: link.subdomain_share_enabled,
        subdomain_share_disabled_reason: link.subdomain_share_disabled_reason,
        created_at: new Date().toISOString(),
        share_access_mode: activeShareLinkStatus?.share_access_mode ?? 'token',
        selected_user_ids: activeShareLinkStatus?.selected_user_ids ?? [],
        selected_ldap_groups: activeShareLinkStatus?.selected_ldap_groups ?? [],
        has_password: activeShareLinkStatus?.has_password ?? false,
      };
      setShareLinkStatus(nextStatus);
      setShareSlugDraft(link.share_slug);
      setShareSlugAvailable(true);
      setError(null);
      return nextStatus;
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create share link');
      return null;
    } finally {
      setSharingWorkspace(false);
      setRotatingShareLink(false);
    }
  }, [
    activeWorkspaceId,
    activeShareLinkStatus?.has_password,
    activeShareLinkStatus?.selected_ldap_groups,
    activeShareLinkStatus?.selected_user_ids,
    activeShareLinkStatus?.share_access_mode,
    canEditWorkspace,
  ]);

  useEffect(() => {
    if (!showShareModal || !activeWorkspaceId || !canEditWorkspace) return;
    if (loadingShareStatus || sharingWorkspace || autoCreateShareLinkAttempted) return;
    if (!shareLinkStatus || shareLinkStatus.has_share_link) return;

    setAutoCreateShareLinkAttempted(true);
    void handleEnsureShareLink(false);
  }, [
    showShareModal,
    activeWorkspaceId,
    canEditWorkspace,
    loadingShareStatus,
    sharingWorkspace,
    autoCreateShareLinkAttempted,
    activeShareLinkStatus,
    handleEnsureShareLink,
  ]);

  const resolveShareUrl = useCallback((status: UserSpaceWorkspaceShareLinkStatus | null | undefined): string | null => {
    if (!status?.has_share_link) return null;
    if (shareLinkType === 'subdomain' && status.subdomain_share_enabled && status.subdomain_share_url) {
      return status.subdomain_share_url;
    }
    if (shareLinkType === 'anonymous' && status.anonymous_share_url) {
      return status.anonymous_share_url;
    }
    return status.share_url;
  }, [shareLinkType]);

  const effectiveShareUrl = useMemo(() => resolveShareUrl(activeShareLinkStatus), [activeShareLinkStatus, resolveShareUrl]);
  const shareSubdomainEnabled = Boolean(activeShareLinkStatus?.subdomain_share_enabled);
  const shareSubdomainDisabledReason = activeShareLinkStatus?.subdomain_share_disabled_reason ?? null;
  const showProtectedSubdomainNotice = Boolean(
    activeShareLinkStatus?.has_share_link
      && shareLinkType === 'subdomain'
      && shareAccessMode !== 'token',
  );

  useEffect(() => {
    if (shareLinkType === 'subdomain' && !shareSubdomainEnabled) {
      setShareLinkType('anonymous');
    }
  }, [shareLinkType, shareSubdomainEnabled]);

  const handleCopyShareLink = useCallback(async () => {
    let url = effectiveShareUrl;
    if (!activeShareLinkStatus?.has_share_link) {
      const created = await handleEnsureShareLink(false);
      url = resolveShareUrl(created);
    }
    if (!url) return;

    try {
      await navigator.clipboard.writeText(url);
      setShareCopied(true);
      window.setTimeout(() => setShareCopied(false), 1500);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to copy share link');
    }
  }, [activeShareLinkStatus?.has_share_link, effectiveShareUrl, handleEnsureShareLink, resolveShareUrl]);

  const handleOpenFullPreview = useCallback(async () => {
    let url = resolveShareUrl(activeShareLinkStatus);
    if (!activeShareLinkStatus?.has_share_link) {
      const created = await handleEnsureShareLink(false);
      url = resolveShareUrl(created);
    }
    if (!url) return;
    window.open(url, '_blank', 'noopener,noreferrer');
  }, [activeShareLinkStatus, handleEnsureShareLink, resolveShareUrl]);

  const handleRotateShareLink = useCallback(async () => {
    await handleEnsureShareLink(true);
  }, [handleEnsureShareLink]);

  const handleRevokeShareLink = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRevokingShareLink(true);
    try {
      const status = await api.revokeUserSpaceWorkspaceShareLink(activeWorkspaceId);
      setShareLinkStatus(status);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke share link');
    } finally {
      setRevokingShareLink(false);
    }
  }, [activeWorkspaceId, canEditWorkspace]);

  useEffect(() => {
    if (!showShareModal || !activeWorkspaceId || !canEditWorkspace) return;
    const normalized = normalizeShareSlugInput(shareSlugDraft);
    if (!normalized) {
      setShareSlugAvailable(false);
      setCheckingShareSlug(false);
      return;
    }

    setCheckingShareSlug(true);
    const timer = window.setTimeout(async () => {
      try {
        const result = await api.checkUserSpaceWorkspaceShareSlugAvailability(activeWorkspaceId, normalized);
        setShareSlugAvailable(result.available);
      } catch {
        setShareSlugAvailable(false);
      } finally {
        setCheckingShareSlug(false);
      }
    }, 350);

    return () => {
      window.clearTimeout(timer);
      setCheckingShareSlug(false);
    };
  }, [showShareModal, activeWorkspaceId, canEditWorkspace, shareSlugDraft]);

  const handleToggleShareSelectedUser = useCallback((userId: string) => {
    setShareSelectedUserIdsDraft((current) => (
      current.includes(userId)
        ? current.filter((value) => value !== userId)
        : [...current, userId]
    ));
  }, []);

  const handleAddShareLdapGroup = useCallback(() => {
    const groupDn = shareLdapGroupDraft.trim();
    if (!groupDn) return;
    setShareSelectedLdapGroupsDraft((current) => (
      current.includes(groupDn) ? current : [...current, groupDn]
    ));
    setShareLdapGroupDraft('');
  }, [shareLdapGroupDraft]);

  const handleRemoveShareLdapGroup = useCallback((groupDn: string) => {
    setShareSelectedLdapGroupsDraft((current) => current.filter((value) => value !== groupDn));
  }, []);

  const handleSaveShareAccess = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;

    const normalizedSlug = normalizeShareSlugInput(shareSlugDraft);
    if (!normalizedSlug) {
      setError('Share slug is required');
      return;
    }

    let currentStatus = activeShareLinkStatus;
    if (!currentStatus?.has_share_link) {
      currentStatus = await handleEnsureShareLink(false);
    }
    if (!currentStatus) {
      return;
    }

    if (shareAccessMode === 'password' && !currentStatus.has_password && !sharePasswordDraft.trim()) {
      setError('Share password is required for password-protected access');
      return;
    }

    setSavingShareAccess(true);
    try {
      let status = currentStatus;

      if ((status.share_slug ?? '') !== normalizedSlug) {
        status = await api.updateUserSpaceWorkspaceShareSlug(activeWorkspaceId, normalizedSlug);
      }

      const selectedUserIds = normalizeUniqueStrings(shareSelectedUserIdsDraft);
      const selectedLdapGroups = normalizeUniqueStrings(shareSelectedLdapGroupsDraft);
      const hasAccessChanges =
        shareAccessMode !== status.share_access_mode
        || !areSameNormalizedStringArrays(selectedUserIds, status.selected_user_ids ?? [])
        || !areSameNormalizedStringArrays(selectedLdapGroups, status.selected_ldap_groups ?? [])
        || (shareAccessMode === 'password' && Boolean(sharePasswordDraft.trim()));

      if (hasAccessChanges) {
        status = await api.updateUserSpaceWorkspaceShareAccess(activeWorkspaceId, {
          share_access_mode: shareAccessMode,
          password: sharePasswordDraft.trim() || undefined,
          selected_user_ids: selectedUserIds,
          selected_ldap_groups: selectedLdapGroups,
        });
      }

      setShareLinkStatus(status);
      setShareSlugDraft(status.share_slug ?? normalizedSlug);
      setShareSlugAvailable(true);
      setShareAccessMode(status.share_access_mode);
      setShareSelectedUserIdsDraft(status.selected_user_ids ?? []);
      setShareSelectedLdapGroupsDraft(status.selected_ldap_groups ?? []);
      setSharePasswordDraft('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save share access settings');
    } finally {
      setSavingShareAccess(false);
    }
  }, [
    activeWorkspaceId,
    activeShareLinkStatus,
    canEditWorkspace,
    handleEnsureShareLink,
    shareAccessMode,
    sharePasswordDraft,
    shareSlugDraft,
    shareSelectedLdapGroupsDraft,
    shareSelectedUserIdsDraft,
  ]);

  const shareSelectableUsers = useMemo(() => {
    if (allUsers.length > 0) {
      return allUsers;
    }
    if (!activeWorkspace) {
      return [currentUser];
    }
    const workspaceMemberIds = new Set<string>([
      activeWorkspace.owner_user_id,
      ...activeWorkspace.members.map((member) => member.user_id),
    ]);
    const fallbackUsers: User[] = [currentUser, ...allUsers].filter((user, index, list) => (
      workspaceMemberIds.has(user.id) && list.findIndex((candidate) => candidate.id === user.id) === index
    ));
    return fallbackUsers;
  }, [activeWorkspace, allUsers, currentUser]);

  const shareHasUnsavedChanges = useMemo(() => {
    if (!showShareModal || !activeShareLinkStatus) {
      return false;
    }

    const slugChanged = normalizeShareSlugInput(shareSlugDraft) !== normalizeShareSlugInput(activeShareLinkStatus.share_slug ?? '');
    const accessModeChanged = shareAccessMode !== activeShareLinkStatus.share_access_mode;
    const selectedUsersChanged = !areSameNormalizedStringArrays(
      shareSelectedUserIdsDraft,
      activeShareLinkStatus.selected_user_ids ?? [],
    );
    const selectedLdapGroupsChanged = !areSameNormalizedStringArrays(
      shareSelectedLdapGroupsDraft,
      activeShareLinkStatus.selected_ldap_groups ?? [],
    );
    const pendingPasswordChange = shareAccessMode === 'password' && Boolean(sharePasswordDraft.trim());

    return slugChanged
      || accessModeChanged
      || selectedUsersChanged
      || selectedLdapGroupsChanged
      || pendingPasswordChange;
  }, [
    activeShareLinkStatus,
    shareAccessMode,
    sharePasswordDraft,
    shareSelectedLdapGroupsDraft,
    shareSelectedUserIdsDraft,
    shareSlugDraft,
    showShareModal,
  ]);

  const handleSaveWorkspaceRename = useCallback(async (workspace: UserSpaceWorkspace) => {
    const nextName = workspaceNameDraft.trim();
    if (!nextName) return;

    try {
      const updated = await api.updateUserSpaceWorkspace(workspace.id, { name: nextName });
      setWorkspaces((current) => current.map((ws) => ws.id === updated.id ? updated : ws));
      setEditingWorkspaceNameId(null);
      setWorkspaceNameDraft('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename workspace');
    }
  }, [workspaceNameDraft]);

  const renderTreeNodes = useCallback((nodes: ReturnType<typeof buildUserSpaceTree>, depth = 0) => {
    return nodes.flatMap((node) => {
      const indentStyle = { paddingLeft: `${depth * 14 + 6}px` };

      if (node.type === 'folder') {
        const isExpanded = expandedFolders.has(node.path);
        const isRenaming = renamingFolderPath === node.path;
        const isConfirmingDelete = deleteConfirmFolderPath === node.path;
        const rows: JSX.Element[] = [];

        if (isRenaming) {
          rows.push(
            <div key={`${node.path}-rename`} className="userspace-file-item userspace-tree-row active">
              <input
                className="userspace-file-rename-input"
                style={indentStyle}
                value={renameValue}
                onChange={(event) => setRenameValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') handleRenameFolder(node.path, renameValue);
                  if (event.key === 'Escape') setRenamingFolderPath(null);
                }}
                autoFocus
              />
              <div className="userspace-item-actions" style={{ opacity: 1 }}>
                <button className="chat-action-btn" onClick={() => handleRenameFolder(node.path, renameValue)} title="Confirm">
                  <Check size={12} />
                </button>
                <button className="chat-action-btn" onClick={() => setRenamingFolderPath(null)} title="Cancel">
                  <X size={12} />
                </button>
              </div>
            </div>
          );
        } else {
          const mountInfo = mountTargetPaths.get(node.path);
          const isMount = !!mountInfo;
          const isMountDisabled = isMount && !mountInfo.enabled;
          const isMountDisconnected = isMount && !mountInfo.sourceAvailable;
          const isSshMount = isMount && mountInfo.sourceType === 'ssh';
          const isMountSyncInProgress = isSshMount && !!mountInfo && (syncingMountId === mountInfo.id || previewingMountId === mountInfo.id);
          const mountStatusClass = isMountSyncInProgress
            ? 'userspace-status-pill-warning'
            : mountInfo?.syncStatus === 'synced'
            ? 'userspace-status-pill-success'
            : mountInfo?.syncStatus === 'pending'
              ? 'userspace-status-pill-info'
              : 'userspace-status-pill-danger';
          const mountBadgeLabel = isMountDisconnected
            ? 'disconnected'
            : isMountDisabled
              ? 'unmounted'
              : isSshMount
                ? isMountSyncInProgress
                  ? 'in progress'
                  : mountInfo?.syncStatus === 'synced'
                    ? 'synced'
                    : mountInfo?.syncStatus === 'pending'
                      ? 'pending'
                      : 'error'
                : 'live';
          const hasChangedFileDescendant = !isExpanded && collectFilePaths(node).some((p) => changedFilePaths.has(p));
          rows.push(
            <div key={node.path} className={`userspace-file-item userspace-tree-row userspace-tree-folder-row${isMount ? ' userspace-tree-mount-folder' : ''}${isMountDisabled || isMountDisconnected ? ' userspace-tree-mount-disabled' : ''}`}>
              <button className="userspace-item-content userspace-tree-content" onClick={() => handleToggleFolder(node.path)} style={indentStyle}>
                <span className="userspace-tree-chevron" aria-hidden="true">
                  {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </span>
                <span className={`userspace-folder-label${isMount ? ' userspace-tree-mount-label' : ''}`}>{node.name}</span>
                {hasChangedFileDescendant && <span className="userspace-tree-folder-changed-file-dot" title="Contains changed files" />}
                {isMount && (
                  <span
                    className={`userspace-tree-mount-badge${isMountDisconnected ? ' userspace-tree-mount-badge-disconnected' : isMountDisabled ? ' userspace-tree-mount-badge-disabled' : isSshMount && (isMountSyncInProgress || mountInfo?.syncStatus !== 'synced') ? ` userspace-tree-mount-badge-sync ${mountStatusClass}` : ''}`}
                    role="button"
                    tabIndex={0}
                    title={isMountDisconnected ? 'Mount source is no longer available' : isMountSyncInProgress ? 'Mount sync in progress' : isSshMount && mountInfo?.syncStatus === 'error' && mountInfo?.lastSyncError ? mountInfo.lastSyncError : 'Manage mounts'}
                    onClick={(e) => { e.stopPropagation(); void handleOpenMountsModal(); }}
                  >
                    {mountBadgeLabel}
                  </span>
                )}
              </button>
              {canEditWorkspace && !isMount && (
                <div className="userspace-item-actions">
                  {isConfirmingDelete ? (
                    <>
                      <button className="chat-action-btn confirm-delete" onClick={() => handleDeleteFolder(node.path)} title="Confirm">
                        <Check size={12} />
                      </button>
                      <button className="chat-action-btn cancel-delete" onClick={() => setDeleteConfirmFolderPath(null)} title="Cancel">
                        <X size={12} />
                      </button>
                    </>
                  ) : (
                    <>
                      <button className="chat-action-btn" onClick={() => handleStartCreateFile(node.path)} title="New file in folder">
                        <Plus size={12} />
                      </button>
                      <button className="chat-action-btn" onClick={() => { setRenamingFolderPath(node.path); setRenameValue(node.path); }} title="Rename folder">
                        <Pencil size={12} />
                      </button>
                      <button className="chat-action-btn" onClick={() => setDeleteConfirmFolderPath(node.path)} title="Delete folder">
                        <Trash2 size={12} />
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          );
        }

        if (isExpanded && node.children.length > 0) {
          rows.push(...renderTreeNodes(node.children, depth + 1));
        }

        return rows;
      }

      const isConfirmingDelete = deleteConfirmFileId === node.path;
      const isRenaming = renamingFilePath === node.path;

      if (isRenaming) {
        return [
          <div key={`${node.path}-rename`} className="userspace-file-item userspace-tree-row active">
            <input
              className="userspace-file-rename-input"
              style={indentStyle}
              value={renameValue}
              onChange={(event) => setRenameValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') handleRenameFile(node.path, renameValue);
                if (event.key === 'Escape') setRenamingFilePath(null);
              }}
              autoFocus
            />
            <div className="userspace-item-actions" style={{ opacity: 1 }}>
              <button className="chat-action-btn" onClick={() => handleRenameFile(node.path, renameValue)} title="Confirm">
                <Check size={12} />
              </button>
              <button className="chat-action-btn" onClick={() => setRenamingFilePath(null)} title="Cancel">
                <X size={12} />
              </button>
            </div>
          </div>,
        ];
      }

      const isFileChanged = changedFilePaths.has(node.path);

      return [
        <div
          key={node.path}
          className={`userspace-file-item userspace-tree-row ${node.path === selectedFilePath ? 'active' : ''}`}
        >
          <button
            className="userspace-item-content userspace-tree-content"
            onClick={() => handleSelectFile(node.path)}
            onMouseEnter={isFileChanged ? () => handleTreeFileHoverStart(node.path) : undefined}
            onMouseLeave={isFileChanged ? handleTreeFileHoverEnd : undefined}
            style={indentStyle}
          >
            <span className="userspace-tree-file-label">{node.name}</span>
            {isFileChanged && (
              <span
                className="userspace-tree-file-changed-dot"
                title="Changed since last snapshot"
              />
            )}
          </button>
          {canEditWorkspace && (
            <div className="userspace-item-actions">
              {isConfirmingDelete ? (
                <>
                  <button className="chat-action-btn confirm-delete" onClick={() => handleDeleteFile(node.path)} title="Confirm">
                    <Check size={12} />
                  </button>
                  <button className="chat-action-btn cancel-delete" onClick={() => setDeleteConfirmFileId(null)} title="Cancel">
                    <X size={12} />
                  </button>
                </>
              ) : (
                <>
                  {isFileChanged && (
                    <button
                      className="userspace-tree-save-btn"
                      onClick={(e) => { e.stopPropagation(); handleSaveTreeFile(node.path); }}
                      disabled={savingTreeFile === node.path}
                      title={savingTreeFile === node.path ? 'Saving...' : 'Save file'}
                    >
                      <Save size={12} className={savingTreeFile === node.path ? 'spinning' : undefined} />
                    </button>
                  )}
                  <button className="chat-action-btn" onClick={() => { setRenamingFilePath(node.path); setRenameValue(node.path); }} title="Rename">
                    <Pencil size={12} />
                  </button>
                  <button className="chat-action-btn" onClick={() => setDeleteConfirmFileId(node.path)} title="Delete">
                    <Trash2 size={12} />
                  </button>
                </>
              )}
            </div>
          )}
        </div>,
      ];
    });
  }, [canEditWorkspace, changedFilePaths, deleteConfirmFileId, deleteConfirmFolderPath, expandedFolders, handleDeleteFile, handleDeleteFolder, handleOpenMountsModal, handleRenameFile, handleRenameFolder, handleSaveTreeFile, handleSelectFile, handleStartCreateFile, handleToggleFolder, handleTreeFileHoverEnd, handleTreeFileHoverStart, mountTargetPaths, previewingMountId, renameValue, renamingFilePath, renamingFolderPath, savingTreeFile, selectedFilePath, syncingMountId]);

  const sqliteLiveDataOnlyMode = activeWorkspace?.sqlite_persistence_mode === 'exclude';
  const sqlitePersistenceModeTitle = sqliteLiveDataOnlyMode
    ? 'Live data only mode. SQLite local files are excluded from snapshots. Click to enable two-lane persistence (live data + SQLite local state with migrations).'
    : 'Two-lane persistence mode. Live data wiring is primary for dashboards; SQLite local state is persisted with snapshots. Click to switch to live data only mode.';
  const formattedError = useMemo(() => formatUserSpaceErrorMessage(error), [error]);
  const hasStatusOverlayContent = Boolean(
    loading || creatingWorkspace || duplicatingWorkspaceSourceId || deletingWorkspaceId || runtimeOverlayStatus || (formattedError && !creatingWorkspace && !duplicatingWorkspaceSourceId && !deletingWorkspaceId)
  );
  const statusOverlaySignature = useMemo(() => JSON.stringify({
    loading,
    creatingWorkspace,
    creatingWorkspaceStatus,
    duplicatingWorkspaceSourceId,
    duplicatingWorkspaceStatus,
    deletingWorkspaceId,
    deletingWorkspaceStatus,
    runtimeOverlayStatus,
    formattedError: formattedError && !creatingWorkspace && !duplicatingWorkspaceSourceId && !deletingWorkspaceId ? formattedError : null,
  }), [
    loading,
    creatingWorkspace,
    creatingWorkspaceStatus,
    duplicatingWorkspaceSourceId,
    duplicatingWorkspaceStatus,
    deletingWorkspaceId,
    deletingWorkspaceStatus,
    runtimeOverlayStatus,
    formattedError,
  ]);

  useEffect(() => {
    if (!hasStatusOverlayContent) {
      setStatusOverlayVisible(false);
      setStatusOverlayFading(false);
      setStatusOverlayPinned(false);
      setStatusOverlayInteracting(false);
      statusOverlayDismissedSignatureRef.current = null;
      return;
    }

    if (statusOverlayDismissedSignatureRef.current !== statusOverlaySignature) {
      setStatusOverlayVisible(true);
      setStatusOverlayFading(false);
      statusOverlayDismissedSignatureRef.current = null;
    }
  }, [hasStatusOverlayContent, statusOverlaySignature]);

  useEffect(() => {
    if (!hasStatusOverlayContent || !statusOverlayVisible || statusOverlayPinned || statusOverlayInteracting || runtimeOverlayStatus) {
      setStatusOverlayFading(false);
      return;
    }

    setStatusOverlayFading(true);
    const timer = window.setTimeout(() => {
      if (statusOverlayPinned || statusOverlayInteracting || !hasStatusOverlayContent) {
        return;
      }
      setStatusOverlayVisible(false);
      setStatusOverlayFading(false);
      statusOverlayDismissedSignatureRef.current = statusOverlaySignature;
    }, 2000);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    hasStatusOverlayContent,
    statusOverlayVisible,
    statusOverlayPinned,
    statusOverlayInteracting,
    runtimeOverlayStatus,
    statusOverlaySignature,
  ]);

  const handleStatusOverlayClick = useCallback(() => {
    setStatusOverlayPinned((current) => {
      const next = !current;
      if (!next) {
        setStatusOverlayVisible(true);
      }
      setStatusOverlayFading(false);
      statusOverlayDismissedSignatureRef.current = null;
      return next;
    });
  }, []);

  return (
    <div className={`userspace-layout${isFullscreen ? ' userspace-fullscreen' : ''}`}>
      {/* === Top toolbar === */}
      <div className="userspace-toolbar">
        <div className="userspace-toolbar-group">
          <div className="model-selector userspace-workspace-picker" ref={workspaceDropdownRef}>
            <button
              type="button"
              className="model-selector-trigger userspace-workspace-trigger"
              onClick={() => {
                if (workspaces.length > 0) {
                  setIsWorkspaceMenuOpen((open) => !open);
                }
              }}
              title="Select workspace"
              aria-haspopup="listbox"
              aria-expanded={isWorkspaceMenuOpen}
              disabled={workspaces.length === 0}
            >
              <span className="model-selector-text">{activeWorkspace?.name ?? 'No workspaces'}</span>
              {activeWorkspaceChatState.hasLive && (
                <MiniLoadingSpinner variant="icon" size={14} title="Chat in progress" />
              )}
              {!activeWorkspaceChatState.hasLive && activeWorkspaceChatState.hasInterrupted && (
                <span className="userspace-workspace-trigger-state is-interrupted" title="A conversation was interrupted">
                  <AlertCircle size={13} />
                </span>
              )}
              <span className="model-selector-arrow">▾</span>
            </button>

            {isWorkspaceMenuOpen && workspaces.length > 0 && (
              <div className="model-selector-dropdown userspace-workspace-dropdown">
                <div className="model-selector-dropdown-inner" role="listbox" aria-label="Workspace list">
                  {workspaces.map((ws) => {
                    const workspaceChatState = workspaceChatStates[ws.id] ?? DEFAULT_WORKSPACE_CHAT_STATE;
                    const canDeleteWorkspace =
                      ws.owner_user_id === currentUser.id ||
                      ws.members.some((member) => member.user_id === currentUser.id && member.role === 'owner');
                    const canRenameWorkspace = currentUser.role === 'admin'
                      || ws.owner_user_id === currentUser.id
                      || ws.members.some((member) => (
                        member.user_id === currentUser.id && (member.role === 'owner' || member.role === 'editor')
                      ));
                    const isConfirmingDelete = deleteConfirmWorkspaceId === ws.id;
                    const isDeletingWorkspace = Boolean(deletingWorkspaceTasks[ws.id]);
                    const duplicateTask = activeWorkspaceDuplicateTaskBySourceId[ws.id] ?? null;
                    const isDuplicatingWorkspace = Boolean(duplicateTask);
                    const workspaceActionBusy = isDeletingWorkspace || isDuplicatingWorkspace;
                    const isRenamingWorkspace = editingWorkspaceNameId === ws.id;
                    return (
                      <div
                        key={ws.id}
                        className={`model-selector-item userspace-workspace-item ${ws.id === activeWorkspaceId ? 'is-selected' : ''}${isDeletingWorkspace ? ' is-deleting' : ''}${isDuplicatingWorkspace ? ' is-duplicating' : ''} ${!canDeleteWorkspace ? 'is-shared' : ''}`}
                      >
                        {isRenamingWorkspace ? (
                          <div className="userspace-workspace-inline-edit">
                            <input
                              type="text"
                              className="userspace-workspace-rename-input"
                              value={workspaceNameDraft}
                              onChange={(event) => setWorkspaceNameDraft(event.target.value)}
                              onClick={(event) => event.stopPropagation()}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter') {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  void handleSaveWorkspaceRename(ws);
                                }
                                if (event.key === 'Escape') {
                                  event.preventDefault();
                                  event.stopPropagation();
                                  handleCancelWorkspaceRename();
                                }
                              }}
                              autoFocus
                            />
                          </div>
                        ) : (
                          <button
                            type="button"
                            role="option"
                            aria-selected={ws.id === activeWorkspaceId}
                            className="userspace-workspace-select-btn"
                            disabled={isDeletingWorkspace}
                            onClick={() => {
                              if (ws.id !== activeWorkspaceId) {
                                  setRuntimeStatus(null);
                              }
                              setActiveWorkspaceId(ws.id);
                              setIsWorkspaceMenuOpen(false);
                              setDeleteConfirmWorkspaceId(null);
                            }}
                          >
                            <span className="model-selector-item-name">{ws.name}</span>
                            {workspaceChatState.hasLive && (
                              <MiniLoadingSpinner variant="icon" size={14} title="Chat in progress" />
                            )}
                            {!workspaceChatState.hasLive && workspaceChatState.hasInterrupted && (
                              <span className="userspace-workspace-item-state is-interrupted" title="A conversation was interrupted">
                                <AlertCircle size={13} />
                              </span>
                            )}
                          </button>
                        )}

                        {(canDeleteWorkspace || canRenameWorkspace || isRenamingWorkspace) && (
                          <div className="userspace-workspace-item-actions">
                            {isRenamingWorkspace ? (
                              <>
                                <button
                                  type="button"
                                  className="chat-action-btn confirm-delete"
                                  disabled={workspaceActionBusy}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void handleSaveWorkspaceRename(ws);
                                  }}
                                  title="Save workspace name"
                                >
                                  <Check size={12} />
                                </button>
                                <button
                                  type="button"
                                  className="chat-action-btn cancel-delete"
                                  disabled={isDeletingWorkspace}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleCancelWorkspaceRename();
                                  }}
                                  title="Cancel rename"
                                >
                                  <X size={12} />
                                </button>
                              </>
                            ) : isConfirmingDelete ? (
                              <>
                                <button
                                  type="button"
                                  className="chat-action-btn confirm-delete"
                                  disabled={workspaceActionBusy}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void handleDeleteWorkspace(ws.id);
                                  }}
                                  title="Confirm delete workspace"
                                >
                                  {isDeletingWorkspace ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                </button>
                                <button
                                  type="button"
                                  className="chat-action-btn cancel-delete"
                                  disabled={isDeletingWorkspace}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setDeleteConfirmWorkspaceId(null);
                                  }}
                                  title="Cancel"
                                >
                                  <X size={12} />
                                </button>
                              </>
                            ) : (
                              <>
                                {canRenameWorkspace && (
                                  <button
                                    type="button"
                                    className="chat-action-btn"
                                    disabled={workspaceActionBusy}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      void handleDuplicateWorkspace(ws.id);
                                    }}
                                    title={isDuplicatingWorkspace ? (formatWorkspaceDuplicateTaskStatus(duplicateTask) || 'Duplicating workspace...') : 'Duplicate workspace'}
                                  >
                                    {isDuplicatingWorkspace ? <MiniLoadingSpinner variant="icon" size={12} /> : <CopyPlus size={12} />}
                                  </button>
                                )}
                                {canRenameWorkspace && (
                                  <button
                                    type="button"
                                    className="chat-action-btn"
                                    disabled={workspaceActionBusy}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      handleStartWorkspaceRename(ws);
                                    }}
                                    title="Rename workspace"
                                  >
                                    <Pencil size={12} />
                                  </button>
                                )}
                                {canDeleteWorkspace && (
                                  <button
                                    type="button"
                                    className="chat-action-btn"
                                    disabled={workspaceActionBusy}
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      setDeleteConfirmWorkspaceId(ws.id);
                                    }}
                                    title="Delete workspace"
                                  >
                                    {isDeletingWorkspace ? <MiniLoadingSpinner variant="icon" size={12} /> : <Trash2 size={12} />}
                                  </button>
                                )}
                              </>
                            )}
                          </div>
                        )}

                        {!canDeleteWorkspace && (
                          <span className="userspace-workspace-owner-hint" title="Only workspace owners can delete workspaces">
                            Shared
                          </span>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
          {workspaces.length < workspacesTotal && (
            <button className="btn btn-secondary btn-sm" onClick={() => loadWorkspaces(true)} disabled={loadingMore}>
              {loadingMore ? '...' : 'More'}
            </button>
          )}
          <button
            className="btn btn-primary btn-sm"
            onClick={handleCreateWorkspace}
            title={creatingWorkspace ? (creatingWorkspaceStatus || 'Creating workspaces...') : 'New workspace'}
          >
            {creatingWorkspace ? <MiniLoadingSpinner variant="icon" size={14} /> : <Plus size={14} />}
          </button>
          {canEditWorkspace && activeWorkspaceId && (
            <button className="btn btn-secondary btn-sm" onClick={() => setShowScmWizard(true)} title="Import or export this workspace with Git">
              <GitBranch size={14} />
            </button>
          )}
          {isOwner && (
            <>
              <MemberManagementButton onClick={handleOpenMembersModal} title="Manage members" />
              <AgentAccessButton onClick={handleOpenAgentAccessModal} title="Manage cross-workspace agent access" />
              <button className="btn btn-secondary btn-sm" onClick={handleOpenEnvVarsModal} title="Environment variables">
                <KeyRound size={14} />
              </button>
              <button className="btn btn-secondary btn-sm" onClick={handleOpenMountsModal} title="Filesystem mounts">
                <HardDrive size={14} />
              </button>
            </>
          )}
          {currentUser.role === 'admin' && (
            <button className="btn btn-secondary btn-sm" onClick={() => setShowAdminWorkspacesModal(true)} title="Manage all workspaces (Admin)">
              <Shield size={14} />
            </button>
          )}
        </div>

        <div className="userspace-toolbar-group userspace-toolbar-group-right">
          <div className="userspace-toolbar-status-group">
            {previewExecuting && (
              <span className="userspace-toolbar-live-status" title="Live data connection in progress">
                <MiniLoadingSpinner variant="icon" size={14} ariaHidden />
                Connecting data...
              </span>
            )}
            {activeWorkspace && !isAdminImpersonating && !isOwner && (
              <span className="userspace-status-pill userspace-status-pill-info">
                {activeWorkspaceRole}{!canEditWorkspace ? ' (read-only)' : ''}
              </span>
            )}
            {activeWorkspace?.scm?.connected && (
              <span
                className={`userspace-status-pill ${activeWorkspace.scm.sync_paused ? 'userspace-status-pill-warning' : 'userspace-status-pill-muted'}`}
                title={
                  activeWorkspace.scm.sync_paused
                    ? `Sync paused: ${activeWorkspace.scm.sync_paused_reason || 'conflict detected'}`
                    : activeWorkspace.scm.remote_role === 'upstream'
                      ? `Upstream: ${activeWorkspace.scm.git_url || ''} (${activeWorkspace.scm.auto_sync_policy === 'auto_push' ? 'auto-push' : 'manual push'})`
                      : activeWorkspace.scm.git_url || 'Workspace SCM connection'
                }
              >
                {activeWorkspace.scm.sync_paused
                  ? `git:${activeWorkspace.scm.git_branch || 'main'} (paused)`
                  : activeWorkspace.scm.remote_role === 'upstream'
                    ? `upstream:${activeWorkspace.scm.git_branch || 'main'}`
                    : `git:${activeWorkspace.scm.git_branch || 'main'}`}
              </span>
            )}
            {isAdminImpersonating && (
              <span
                className="userspace-status-pill userspace-status-pill-warning"
                title={`Viewing as admin (owner: ${activeWorkspace?.owner_display_name || activeWorkspace?.owner_username || 'unknown'})`}
              >
                {activeWorkspace?.owner_display_name || activeWorkspace?.owner_username || 'unknown'}
              </span>
            )}
            {activeWorkspaceId && collabPresenceCount > 1 && (
              <span
                className="userspace-collab-badge"
                title={`${collabPresenceCount} collaborators viewing this workspace`}
              >
                <Users size={14} />
                <span className="userspace-collab-badge-count">{collabPresenceCount}</span>
              </span>
            )}
            {runtimeStatus && (
              <span
                className={`userspace-status-pill ${
                  runtimeDisplayState === 'running'
                    ? 'userspace-status-pill-success'
                    : runtimeDisplayState === 'starting' || runtimeDisplayState === 'stopping'
                      ? 'userspace-status-pill-warning'
                      : runtimeDisplayState === 'error' || runtimeDisplayState === 'stopped'
                        ? 'userspace-status-pill-danger'
                        : 'userspace-status-pill-muted'
                }`}
                title={runtimeStatus.last_error || 'Workspace runtime session state'}
              >
                {runtimeDisplayState === 'starting'
                  ? `starting runtime${runtimeOperationLabel ? ` (${runtimeOperationLabel})` : ''}...`
                  : runtimeDisplayState === 'stopping'
                    ? 'stopping runtime…'
                    : runtimeDisplayState}
              </span>
            )}
          </div>

          {activeWorkspaceId && (
            <div className="userspace-toolbar-tabs" role="tablist" aria-label="Right pane tabs">
              <button
                className={`userspace-toolbar-tab ${activeRightTab === 'preview' ? 'active' : ''}`}
                onClick={() => setActiveRightTab('preview')}
                role="tab"
                aria-selected={activeRightTab === 'preview'}
              >
                Preview
              </button>
              <button
                className={`userspace-toolbar-tab ${activeRightTab === 'console' ? 'active' : ''}`}
                onClick={() => setActiveRightTab('console')}
                role="tab"
                aria-selected={activeRightTab === 'console'}
              >
                <Terminal size={13} /> Console
              </button>
            </div>
          )}

          {canEditWorkspace && activeWorkspaceId && (
            <div className="userspace-toolbar-runtime-controls">
              {showStartRuntimeButton && (
                <button className="btn btn-secondary btn-sm btn-icon" onClick={handleStartRuntime} disabled={runtimeBusy} title="Start runtime">
                  {runtimeBusy ? <MiniLoadingSpinner variant="icon" size={14} /> : <Play size={14} />}
                </button>
              )}
              {showRestartRuntimeButton && (
                <button className="btn btn-secondary btn-sm btn-icon" onClick={handleRestartRuntime} disabled={runtimeBusy} title="Restart runtime">
                  {runtimeBusy ? <MiniLoadingSpinner variant="icon" size={14} /> : <RotateCw size={14} />}
                </button>
              )}
              {showStopRuntimeButton && (
                <button className="btn btn-secondary btn-sm btn-icon" onClick={handleStopRuntime} disabled={runtimeBusy} title="Stop runtime">
                  {runtimeBusy ? <MiniLoadingSpinner variant="icon" size={14} /> : <Square size={14} />}
                </button>
              )}
            </div>
          )}

          <div className="userspace-toolbar-actions" aria-label="Workspace sharing actions">
            <button
              className="btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn"
              onClick={handleOpenShareModal}
              disabled={!activeWorkspaceId || !canEditWorkspace || sharingWorkspace}
              title="Manage share link"
            >
              <Link2 size={14} />
            </button>
            <button
              className="btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn"
              onClick={handleOpenFullPreview}
              disabled={!activeWorkspaceId || !canEditWorkspace || sharingWorkspace}
              title="Open shared preview"
            >
              <ExternalLink size={14} />
            </button>
          </div>

          <div className="userspace-toolbar-actions" aria-label="Workspace editor actions">
            <ToolSelectorDropdown
              availableTools={availableTools}
              selectedToolIds={selectedToolIds}
              onToggleTool={handleToggleWorkspaceTool}
              selectedToolGroupIds={selectedToolGroupIds}
              onToggleToolGroup={handleToggleWorkspaceToolGroup}
              toolGroups={toolGroups}
              disabled={savingWorkspaceTools}
              readOnly={!canEditWorkspace}
              saving={savingWorkspaceTools}
              title="Workspace Tools"
            />
            <button
              className={`btn btn-sm btn-icon userspace-toolbar-action-btn ${sqliteLiveDataOnlyMode ? 'btn-secondary userspace-sqlite-mode-excluded' : 'btn-primary'}`}
              onClick={handleToggleSqlitePersistence}
              disabled={!activeWorkspaceId || !canEditWorkspace}
              title={sqlitePersistenceModeTitle}
              aria-label={sqliteLiveDataOnlyMode ? 'Live data only mode' : 'Two-lane persistence mode'}
            >
              <span className="userspace-sqlite-mode-icon" aria-hidden="true">
                <Database size={14} />
                {sqliteLiveDataOnlyMode && <Slash size={12} className="userspace-sqlite-mode-slash" />}
              </span>
            </button>
            <button
              className="btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn"
              onClick={toggleFullscreen}
              title={isFullscreen ? 'Exit full screen' : 'Full screen'}
            >
              {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
            <button
              className="btn btn-primary btn-sm btn-icon userspace-toolbar-action-btn userspace-snapshot-btn"
              onClick={handleCreateSnapshot}
              disabled={!activeWorkspaceId || !canEditWorkspace || creatingSnapshot}
              title={creatingSnapshot ? 'Creating snapshot...' : 'Take snapshot'}
            >
              <Save size={14} className={creatingSnapshot ? 'spinning' : undefined} />
            </button>
          </div>
        </div>
      </div>

      {/* === Floating status overlay (non-layout shifting) === */}
      {hasStatusOverlayContent && statusOverlayVisible && (
        <div
          className={`userspace-status-overlay${statusOverlayFading ? ' is-fading' : ''}${statusOverlayPinned ? ' is-pinned' : ''}`}
          role="status"
          aria-live="polite"
          onMouseEnter={() => {
            setStatusOverlayInteracting(true);
            setStatusOverlayFading(false);
          }}
          onMouseLeave={() => setStatusOverlayInteracting(false)}
          onFocusCapture={() => {
            setStatusOverlayInteracting(true);
            setStatusOverlayFading(false);
          }}
          onBlurCapture={(event) => {
            if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
              return;
            }
            setStatusOverlayInteracting(false);
          }}
          onClick={handleStatusOverlayClick}
          title={statusOverlayPinned ? 'Pinned. Click to unpin and restore fade behavior.' : 'Click to pin this notification. Click again to unpin.'}
        >
          {loading && <p className="userspace-status userspace-status-overlay-item">Loading workspaces...</p>}
          {creatingWorkspace && (
            <p className="userspace-status userspace-status-overlay-item">
              <MiniLoadingSpinner variant="icon" size={14} /> {creatingWorkspaceStatus || 'Bootstrapping workspace...'}
            </p>
          )}
          {duplicatingWorkspaceSourceId && (
            <p className="userspace-status userspace-status-overlay-item">
              <MiniLoadingSpinner variant="icon" size={14} /> {duplicatingWorkspaceStatus || 'Duplicating workspace...'}
            </p>
          )}
          {deletingWorkspaceId && (
            <p className="userspace-status userspace-status-overlay-item">
              <MiniLoadingSpinner variant="icon" size={14} /> {deletingWorkspaceStatus || 'Deleting workspace...'}
            </p>
          )}
          {runtimeOverlayStatus && !creatingWorkspace && !duplicatingWorkspaceSourceId && !deletingWorkspaceId && (
            <p className="userspace-status userspace-status-overlay-item">
              <MiniLoadingSpinner variant="icon" size={14} /> {runtimeOverlayStatus}
            </p>
          )}
          {formattedError && !creatingWorkspace && !duplicatingWorkspaceSourceId && !deletingWorkspaceId && (
            <p className="userspace-error userspace-status userspace-status-overlay-item">{formattedError}</p>
          )}
        </div>
      )}

      {/* === Main content: left pane (editor+chat) | right pane (preview+snapshots) === */}
      <div className="userspace-content" ref={contentRef} style={{ gridTemplateColumns: rightPaneCollapsed ? '1fr 16px 0fr' : `${leftPaneFraction}fr 4px ${1 - leftPaneFraction}fr` }}>
        {/* Left pane */}
        <div className="userspace-left-pane" ref={leftPaneRef}>
          {editorChatCollapsedSide !== 'before' && (
          <div className="userspace-editor-section" style={{ flex: editorFraction }}>
            {/* File sidebar */}
            {!sidebarCollapsed && (
            <div className="userspace-file-sidebar" style={{ width: sidebarWidth }}>
              <div className="userspace-file-sidebar-header">
                <h4><File size={14} /> Files</h4>
              </div>
              <div className="userspace-file-list">
                {renderTreeNodes(fileTree)}
                {newFileName !== null ? (
                  <div className="userspace-file-item userspace-tree-row userspace-tree-new-file-row">
                    <input
                      className="userspace-file-rename-input"
                      style={{ paddingLeft: `${newFileParentPath ? 20 : 6}px` }}
                      value={newFileName}
                      onChange={(e) => setNewFileName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleCreateNewFile(newFileName, newFileParentPath);
                        if (e.key === 'Escape') {
                          setNewFileName(null);
                          setNewFileParentPath('');
                        }
                      }}
                      placeholder={newFileParentPath ? `${newFileParentPath}/file.ts` : 'path/to/file.ts'}
                      autoFocus
                    />
                    <div className="userspace-item-actions" style={{ opacity: 1 }}>
                      <button className="chat-action-btn" onClick={() => handleCreateNewFile(newFileName, newFileParentPath)} title="Create">
                        <Check size={12} />
                      </button>
                      <button className="chat-action-btn" onClick={() => { setNewFileName(null); setNewFileParentPath(''); }} title="Cancel">
                        <X size={12} />
                      </button>
                    </div>
                  </div>
                ) : canEditWorkspace ? (
                  <button className="userspace-new-file-btn" onClick={() => handleStartCreateFile('')} title="New file">
                    <Plus size={12} /> New file
                  </button>
                ) : fileTree.length === 0 ? (
                  <p className="userspace-muted" style={{ padding: '8px' }}>No files yet</p>
                ) : null}
              </div>
            </div>
            )}

            <ResizeHandle direction="horizontal" onResize={handleResizeSidebar} collapsed={sidebarCollapsed ? 'before' : undefined} onExpand={expandSidebar} />

            {/* Code editor */}
            <div className="userspace-code-editor" ref={codeEditorRef}>
              {!canEditWorkspace && <div className="userspace-readonly-badge">Read-only</div>}
              {selectedFileUnsupportedMessage ? (
                <div className="userspace-nontext-file-placeholder">
                  <File size={18} />
                  <div className="userspace-nontext-file-copy">
                    <strong>{selectedFileDisplayName || 'Selected file'}</strong>
                    <p className="userspace-muted" style={{ margin: 0, whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
                      {selectedFileUnsupportedMessage}
                    </p>
                  </div>
                </div>
              ) : (
                <CodeMirror
                  value={fileContent}
                  onChange={(value) => {
                    setFileContent(value);
                    setFileDirty(true);
                    setChangedFiles((prev) => {
                      if (!selectedFilePath) return prev;
                      if (prev.has(selectedFilePath)) return prev;
                      const next = new Set(prev);
                      next.add(selectedFilePath);
                      return next;
                    });
                    setAcknowledgedFiles((prev) => {
                      if (!selectedFilePath || !prev.has(selectedFilePath)) return prev;
                      const next = new Set(prev);
                      next.delete(selectedFilePath);
                      return next;
                    });

                    // Re-check git-backed changed-file state after local edits settle
                    // so undo/revert transitions are reflected in the tree markers.
                    if (activeWorkspaceId) {
                      debouncedLoadChangedFileState(activeWorkspaceId);
                    }
                  }}
                  extensions={codeMirrorExtensions}
                  editable={canEditWorkspace}
                  readOnly={!canEditWorkspace}
                  placeholder="Create dashboard/report/module source files here"
                  height="100%"
                  theme={(() => {
                    const stored = localStorage.getItem('ragtime-theme');
                    if (stored === 'light') return 'light';
                    if (stored === 'dark' || stored) return 'dark';
                    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
                  })()}
                  basicSetup={USERSPACE_CODEMIRROR_BASIC_SETUP}
                />
              )}
            </div>
          </div>
          )}

          <ResizeHandle
            direction="vertical"
            onResize={handleResizeEditorChat}
            collapsed={editorChatCollapsedSide ?? undefined}
            onExpand={expandChat}
          />

          {/* Chat section */}
          {editorChatCollapsedSide !== 'after' && (
          <div className="userspace-chat-section" style={{ flex: editorChatCollapsedSide === 'before' ? 1 : 1 - editorFraction }}>
            {activeWorkspaceId ? (
              <ChatPanel
                key={activeWorkspaceId}
                currentUser={currentUser}
                debugMode={debugMode}
                workspaceId={activeWorkspaceId}
                workspaceChatState={activeWorkspaceChatSnapshot}
                workspaceAvailableTools={availableTools}
                workspaceSelectedToolIds={resolvedSelectedToolIds}
                workspaceSelectedToolGroupIds={resolvedSelectedToolGroupIds}
                onToggleWorkspaceTool={handleToggleWorkspaceTool}
                onToggleWorkspaceToolGroup={handleToggleWorkspaceToolGroup}
                workspaceToolGroups={toolGroups}
                workspaceSavingTools={savingWorkspaceTools}
                conversationShareableUserIds={workspaceChatShareableUserIds}
                onUserMessageSubmitted={handleUserMessageSubmitted}
                onConversationStateChange={handleConversationStateChange}
                onActiveConversationChange={setActiveWorkspaceConversationId}
                onBranchSwitch={handleBranchSwitch}
                onOpenWorkspaceFile={handleSelectFile}
                onMessageSnapshotRestored={handleMessageSnapshotRestored}
                onSnapshotsMaybeChanged={handleSnapshotsMaybeChanged}
                embedded
                readOnly={false}
                allowAdminReadOnlyBypass={isAdminImpersonating}
                inputBanner={branchRestoreSnapshotId ? (
                  <div className="chat-branch-restore-banner">
                    <span>This branch has an associated code snapshot.</span>
                    {canEditWorkspace ? (
                      <button className="chat-branch-restore-btn confirm" onClick={handleConfirmBranchRestore}>Restore</button>
                    ) : (
                      <span className="chat-branch-restore-note">Only workspace owners and editors can restore files.</span>
                    )}
                    <button className="chat-branch-restore-btn dismiss" onClick={handleDismissBranchRestore}>Dismiss</button>
                  </div>
                ) : undefined}
              />
            ) : (
              <div className="userspace-chat-placeholder">
                <p className="userspace-muted">Select or create a workspace to start chatting</p>
              </div>
            )}
          </div>
          )}
        </div>

        <ResizeHandle direction="horizontal" onResize={handleResizeMainSplit} collapsed={rightPaneCollapsed ? 'after' : undefined} onExpand={expandRightPane} />

        {/* Right pane */}
        {!rightPaneCollapsed && (
        <div className="userspace-right-pane">
          {activeRightTab === 'preview' ? (
            <div className="userspace-preview-section">
              <UserSpaceArtifactPreview
                entryPath={previewEntryPath}
                workspaceFiles={previewWorkspaceFiles}
                liveDataConnections={previewLiveDataConnections}
                runtimePreviewUrl={previewFrameUrl ?? undefined}
                runtimePreviewOrigin={previewOrigin ?? undefined}
                runtimeAuthorizationPending={Boolean(activeWorkspaceId) && previewAuthorizationPending}
                runtimeAvailable={runtimeStatus?.devserver_running ?? false}
                runtimeError={runtimeStatus?.last_error ?? undefined}
                previewInstanceKey={`${activeWorkspaceId ?? ''}:${previewRefreshCounter}`}
                workspaceId={activeWorkspaceId ?? undefined}
                onExecutionStateChange={setPreviewExecuting}
                previewNotice={previewNotice}
              />
            </div>
          ) : (
            <div className="userspace-preview-section" style={{ padding: 12 }}>
              {runtimeCapSysAdminMissing && (
                <div className="userspace-snapshot-item" style={{ marginBottom: 8, alignItems: 'flex-start' }}>
                  <div
                    className="userspace-snapshot-info"
                    style={{ flexDirection: 'column', alignItems: 'flex-start', gap: 4, overflow: 'visible' }}
                  >
                    <strong>Runtime isolation notice</strong>
                    <span className="userspace-muted" style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>
                      Runtime is running without CAP_SYS_ADMIN. Console commands run with reduced isolation
                      (chroot fallback). Enable CAP_SYS_ADMIN on the runtime service for full namespace +
                      pivot_root isolation.
                    </span>
                  </div>
                </div>
              )}
              {runtimeStatus?.last_error && (
                <div className="userspace-snapshot-item" style={{ marginBottom: 8 }}>
                  <div className="userspace-snapshot-info">
                    <strong>Error</strong>
                    <span className="userspace-muted">{runtimeStatus.last_error}</span>
                  </div>
                </div>
              )}
              <div className="userspace-runtime-terminal-wrap">
                <div ref={terminalContainerRef} className="userspace-runtime-terminal" />
              </div>
              {terminalReadOnly && (
                <div className="userspace-toolbar-actions" style={{ marginTop: 8 }}>
                  <span className="userspace-muted">Terminal is read-only for your workspace role.</span>
                </div>
              )}
            </div>
          )}

          {/* Snapshots */}
          <div className="userspace-snapshots-section">
            <button className="userspace-snapshots-toggle" disabled={snapshotUiLocked} onClick={() => {
              const next = !showSnapshots;
              setShowSnapshots(next);
              if (next && activeWorkspaceId && snapshotsLoadedForWorkspace !== activeWorkspaceId) {
                void loadSnapshots(activeWorkspaceId);
              }
            }}>
              <History size={14} />
              <span>Snapshots{snapshotsLoadedForWorkspace === activeWorkspaceId ? ` (${snapshots.length})` : ''}</span>
              <ChevronDown size={14} className={showSnapshots ? '' : 'rotated'} />
            </button>
            {showSnapshots && (
              <div className="userspace-snapshots-list">
                {restoringSnapshotId && (
                  <div className="userspace-snapshot-busy-indicator" role="status" aria-live="polite">
                    Restoring snapshot {restoringSnapshotId.slice(0, 8)}...
                  </div>
                )}
                {snapshotsByBranch.length > 0 && snapshotTimelineRows.length > 0 ? (
                  <div className="userspace-snapshot-graph">
                    <div className="userspace-snapshot-graph-legend">
                      {snapshotsByBranch.map(({ branch, snapshots: branchSnapshots }) => {
                        const branchColor = snapshotBranchColorById.get(branch.id);
                        const isMainBranch = branch.name === 'Main';
                        return (
                          <button
                              key={`legend-${branch.id}`}
                              type="button"
                              className={`userspace-snapshot-branch-legend ${currentSnapshotBranchId === branch.id ? 'active' : ''}`}
                              onClick={() => handleSwitchSnapshotBranch(branch.id)}
                              disabled={!canEditWorkspace || snapshotUiLocked}
                              title={branch.git_ref_name}
                              style={{ '--userspace-branch-color': branchColor } as CSSProperties}
                            >
                              <span className="userspace-snapshot-branch-legend-name">{branch.name}</span>
                              <span className="userspace-snapshot-branch-legend-count">{branchSnapshots.length}</span>
                              {branch.branched_from_snapshot_id && (
                                <span className="userspace-snapshot-branch-legend-fork">
                                  from {branch.branched_from_snapshot_id.slice(0, 8)}
                                </span>
                              )}
                              {!isMainBranch && canEditWorkspace && (
                                <span
                                  role="button"
                                  tabIndex={0}
                                  className="userspace-snapshot-branch-promote"
                                  onClick={(e) => { e.stopPropagation(); handlePromoteBranchToMain(branch.id); }}
                                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); handlePromoteBranchToMain(branch.id); } }}
                                  title="Promote this branch to Main"
                                >
                                  <Crown size={10} />
                                </span>
                              )}
                            </button>
                        );
                      })}
                      <button
                        type="button"
                        className="userspace-snapshot-branch-legend"
                        onClick={() => void handleCreateSnapshotBranch()}
                        disabled={!canEditWorkspace || snapshotUiLocked || !currentSnapshotId}
                        title="Create a new branch from the current snapshot"
                      >
                        <Plus size={12} />
                      </button>
                      {staleBranches.length > 0 && (
                        <button
                          type="button"
                          className="userspace-snapshot-branch-legend stale-toggle"
                          onClick={() => setShowStaleBranches((prev) => !prev)}
                          title={showStaleBranches ? 'Hide stale branches' : 'Show stale branches'}
                        >
                          {showStaleBranches ? `Hide ${staleBranches.length} stale` : `${staleBranches.length} stale`}
                          <ChevronDown size={10} className={showStaleBranches ? '' : 'rotated'} />
                        </button>
                      )}
                      {showStaleBranches && staleBranches.map((branch) => (
                        <button
                            key={`stale-legend-${branch.id}`}
                            type="button"
                            className="userspace-snapshot-branch-legend stale"
                            onClick={() => handleSwitchSnapshotBranch(branch.id)}
                            disabled={!canEditWorkspace || snapshotUiLocked}
                            title={`${branch.git_ref_name} (${branch.commits_behind ?? 0} commits behind)`}
                          >
                            <span className="userspace-snapshot-branch-legend-name">{branch.name}</span>
                            <span className="userspace-snapshot-branch-legend-count">{branch.commits_behind ?? 0} behind</span>
                            {canEditWorkspace && (
                              <span
                                role="button"
                                tabIndex={0}
                                className="userspace-snapshot-branch-promote"
                                onClick={(e) => { e.stopPropagation(); handlePromoteBranchToMain(branch.id); }}
                                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); e.stopPropagation(); handlePromoteBranchToMain(branch.id); } }}
                                title="Promote this branch to Main"
                              >
                                <Crown size={10} />
                              </span>
                            )}
                          </button>
                      ))}
                    </div>

                    {snapshotTimelineRows.map(({ snapshot, laneIndex, laneStates, forkLinks }) => {
                      const isCurrentSnapshot = snapshot.is_current || currentSnapshotId === snapshot.id;
                      const branchColor = snapshotBranchColorById.get(snapshot.branch_id);
                      const isExpanded = expandedSnapshotIds.has(snapshot.id);
                      const diffSummary = snapshotDiffSummaries[snapshot.id];
                      const diffSummaryLoading = loadingSnapshotDiffSummaryIds[snapshot.id] === true;
                      const diffSummaryError = snapshotDiffSummaryErrors[snapshot.id];

                      return (
                        <div key={snapshot.id} className={`userspace-snapshot-row-group ${isExpanded ? 'expanded' : ''}`}>
                          <div
                            className={`userspace-snapshot-graph-row ${isCurrentSnapshot ? 'current' : ''}`}
                            style={{ '--userspace-branch-color': branchColor } as CSSProperties}
                            onClick={() => handleToggleSnapshotExpanded(snapshot.id)}
                            role="button"
                            tabIndex={0}
                            onKeyDown={(event) => {
                              if (event.target !== event.currentTarget) {
                                return;
                              }
                              if (event.key === 'Enter' || event.key === ' ') {
                                event.preventDefault();
                                handleToggleSnapshotExpanded(snapshot.id);
                              }
                            }}
                          >
                            <div
                              className="userspace-snapshot-graph-lanes"
                              style={{ gridTemplateColumns: `repeat(${snapshotsByBranch.length}, 18px)` }}
                              aria-hidden="true"
                            >
                              {snapshotsByBranch.map(({ branch }, branchIndex) => {
                                const laneState = laneStates[branchIndex];
                                const laneColor = snapshotBranchColorById.get(branch.id);
                                return (
                                <div key={`${snapshot.id}-${branch.id}`} className="userspace-snapshot-lane-cell">
                                  <span
                                    className={`userspace-snapshot-lane-line ${laneState?.isActive ? 'active' : ''} ${laneState?.isStart ? 'start' : ''} ${laneState?.isEnd ? 'end' : ''}`}
                                    style={{ '--userspace-branch-color': laneColor } as CSSProperties}
                                  />
                                  {branchIndex === laneIndex && (
                                    <span className={`userspace-snapshot-node-dot ${isCurrentSnapshot ? 'current' : ''}`} />
                                  )}
                                </div>
                                );
                              })}
                              {forkLinks.map((forkLink) => {
                                const forkWidth = forkLink.toLaneIndex - forkLink.fromLaneIndex;
                                if (forkWidth <= 0) {
                                  return null;
                                }
                                return (
                                <span
                                  key={`${snapshot.id}-${forkLink.branchId}`}
                                  className="userspace-snapshot-fork-link"
                                  style={{
                                    left: `${forkLink.fromLaneIndex * 18 + 9}px`,
                                    width: `${forkWidth * 18}px`,
                                    '--userspace-branch-color': snapshotBranchColorById.get(forkLink.branchId),
                                  } as CSSProperties}
                                />
                                );
                              })}
                            </div>

                            <div className="userspace-snapshot-row-main">
                              <ChevronRight size={12} className={`userspace-snapshot-expand-chevron ${isExpanded ? 'expanded' : ''}`} />
                              <code className="userspace-snapshot-hash">{snapshot.id.slice(0, 8)}</code>
                              {renamingSnapshotId === snapshot.id ? (
                                <input
                                  className="userspace-snapshot-rename-input"
                                  value={snapshotEditValue}
                                  onChange={(event) => setSnapshotEditValue(event.target.value)}
                                  onBlur={() => void handleSaveSnapshotRename(snapshot.id)}
                                  onClick={(event) => event.stopPropagation()}
                                  onKeyDown={(event) => {
                                    event.stopPropagation();
                                    if (event.key === 'Escape') {
                                      handleCancelSnapshotRename();
                                    }
                                    if (event.key === 'Enter') {
                                      event.preventDefault();
                                      void handleSaveSnapshotRename(snapshot.id);
                                    }
                                  }}
                                  disabled={savingSnapshotRename}
                                />
                              ) : (
                                <span className="userspace-snapshot-msg" title={snapshot.message || undefined}>
                                  {snapshot.message || 'No message'}
                                </span>
                              )}
                              {!renamingSnapshotId && canEditWorkspace && snapshot.can_rename && (
                                <button
                                  type="button"
                                  className="userspace-snapshot-rename-btn"
                                  title="Rename snapshot"
                                  disabled={snapshotUiLocked}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    handleStartSnapshotRename(snapshot);
                                  }}
                                >
                                  <Pencil size={10} />
                                </button>
                              )}
                              {canEditWorkspace && snapshot.can_delete && !renamingSnapshotId && (
                                <div className="userspace-snapshot-row-delete">
                                  {deleteConfirmSnapshotId === snapshot.id ? (
                                    <>
                                      <button
                                        type="button"
                                        className="userspace-snapshot-delete-confirm-btn"
                                        disabled={deletingSnapshotId === snapshot.id || snapshotUiLocked}
                                        onClick={(event) => {
                                          event.stopPropagation();
                                          void handleDeleteSnapshot(snapshot.id);
                                        }}
                                      >
                                        {deletingSnapshotId === snapshot.id ? <MiniLoadingSpinner variant="icon" size={10} /> : 'Confirm'}
                                      </button>
                                      <button
                                        type="button"
                                        className="userspace-snapshot-delete-cancel-btn"
                                        disabled={deletingSnapshotId === snapshot.id}
                                        onClick={(event) => {
                                          event.stopPropagation();
                                          setDeleteConfirmSnapshotId(null);
                                        }}
                                      >
                                        <X size={10} />
                                      </button>
                                    </>
                                  ) : (
                                    <button
                                      type="button"
                                      className="userspace-snapshot-delete-btn"
                                      title="Delete snapshot"
                                      disabled={snapshotUiLocked}
                                      onClick={(event) => {
                                        event.stopPropagation();
                                        setDeleteConfirmSnapshotId(snapshot.id);
                                      }}
                                    >
                                      <Trash2 size={10} />
                                    </button>
                                  )}
                                </div>
                              )}
                              <span className="userspace-snapshot-ts">
                                {formatSnapshotTimestamp(snapshot.created_at)}
                              </span>
                            </div>

                            <div className="userspace-snapshot-row-actions">
                              {isCurrentSnapshot ? (
                                <span className="userspace-snapshot-current-badge">You are here</span>
                              ) : (
                                <button
                                  type="button"
                                  className="userspace-snapshot-restore-btn"
                                  onClick={(event) => {
                                    event.preventDefault();
                                    event.stopPropagation();
                                    void handleRestoreSnapshot(snapshot.id, snapshot.branch_id);
                                  }}
                                  disabled={!canEditWorkspace || snapshotUiLocked}
                                >
                                  Restore
                                </button>
                              )}
                            </div>
                          </div>

                          {isExpanded && (
                            <>
                              {diffSummaryLoading ? (
                                <div className="userspace-snapshot-expanded-panel">
                                  <div className="userspace-snapshot-expanded-status">
                                    <MiniLoadingSpinner variant="icon" size={12} />
                                    <span>Loading changed files...</span>
                                  </div>
                                </div>
                              ) : diffSummaryError ? (
                                <div className="userspace-snapshot-expanded-panel">
                                  <p className="userspace-muted userspace-error">{formatUserSpaceErrorMessage(diffSummaryError)}</p>
                                </div>
                              ) : !diffSummary ? (
                                <div className="userspace-snapshot-expanded-panel">
                                  <div className="userspace-snapshot-expanded-status">
                                    <MiniLoadingSpinner variant="icon" size={12} />
                                    <span>Loading changed files...</span>
                                  </div>
                                </div>
                              ) : diffSummary.files.length > 0 ? (
                                <div className="userspace-snapshot-diff-file-list">
                                  {diffSummary.is_snapshot_own_diff && (
                                    <div className="userspace-snapshot-diff-own-label userspace-muted">Snapshot contents (no changes to current workspace)</div>
                                  )}
                                  {diffSummary.files.map((file) => {
                                    const fileKey = getSnapshotDiffFileKey(snapshot.id, file.path);
                                    return (
                                      <button
                                        key={fileKey}
                                        type="button"
                                        className={`userspace-snapshot-diff-file-row ${activeSnapshotFileDiffKey === fileKey ? 'active' : ''}`}
                                        onMouseEnter={() => handleSnapshotFileHoverStart(snapshot.id, file.path)}
                                        onMouseLeave={handleSnapshotFileHoverEnd}
                                        onFocus={() => handleSnapshotFileHoverStart(snapshot.id, file.path)}
                                        onBlur={handleSnapshotFileHoverEnd}
                                      >
                                        <span className={`userspace-snapshot-diff-status userspace-snapshot-diff-status-${file.status.toLowerCase()}`}>{file.status}</span>
                                        <span className="userspace-snapshot-diff-path" title={file.old_path ? `${file.old_path} -> ${file.path}` : file.path}>{file.path}</span>
                                        {file.old_path && <span className="userspace-snapshot-diff-old-path">from {file.old_path}</span>}
                                        <span className="userspace-snapshot-diff-counts">+{file.additions} -{file.deletions}</span>
                                      </button>
                                    );
                                  })}
                                </div>
                              ) : (
                                <div className="userspace-snapshot-expanded-panel">
                                  <p className="userspace-muted">No changes from this snapshot to the current workspace.</p>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : snapshotsLoadedForWorkspace !== activeWorkspaceId ? (
                  <p className="userspace-muted" style={{ padding: '8px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <MiniLoadingSpinner variant="icon" size={12} /> Loading snapshots
                  </p>
                ) : (
                  <p className="userspace-muted" style={{ padding: '8px' }}>No snapshots yet</p>
                )}
              </div>
            )}
          </div>
        </div>
        )}
      </div>

      <FileDiffOverlay
        diff={activeSnapshotFileDiff}
        loading={activeSnapshotFileDiffLoading}
        error={activeSnapshotFileDiffError}
        title={activeSnapshotFileDiffTitle}
        beforeLabel={activeSnapshotFileDiffBeforeLabel}
        afterLabel={activeSnapshotFileDiffAfterLabel}
        formatError={formatUserSpaceErrorMessage}
        onDismiss={diffHover.dismiss}
        onOverlayClick={diffHover.overlayClick}
        onOverlayMouseEnter={diffHover.overlayMouseEnter}
        onOverlayMouseLeave={diffHover.overlayMouseLeave}
      />

      <UserSpaceEnvVarsModal
        isOpen={showEnvVarsModal && Boolean(activeWorkspaceId)}
        onClose={() => setShowEnvVarsModal(false)}
        envVars={envVars}
        loading={envVarsLoading}
        saving={savingEnvVar}
        canManage={isOwner}
        showReadonlyAsCompact
        onCreateEnvVar={handleCreateEnvVar}
        onUpdateEnvVar={handleUpdateEnvVar}
        onDeleteEnvVar={handleDeleteEnvVar}
      />



      {showScmWizard && activeWorkspace && (
        <WorkspaceScmWizard
          workspace={activeWorkspace}
          onClose={() => setShowScmWizard(false)}
          onSyncComplete={handleWorkspaceScmSyncComplete}
          onAskAgent={handleAskAgentToPrepareWorkspace}
        />
      )}

      {/* === Mounts + Object Storage modal === */}
      {showMountsModal && activeWorkspaceId && (
        <div className="modal-overlay" onClick={handleCloseMountsModal}>
          <div className="modal-content modal-large" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header" style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
              <div style={{ display: 'flex', gap: 0, flex: 1 }}>
                <button
                  type="button"
                  style={{
                    background: 'transparent',
                    border: 'none',
                    borderBottom: mountsModalTab === 'mounts' ? '2px solid var(--color-accent)' : '2px solid transparent',
                    padding: '8px 16px',
                    cursor: 'pointer',
                    color: mountsModalTab === 'mounts' ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                    fontSize: 14,
                    fontWeight: 600,
                    transition: 'color 0.15s, border-color 0.15s',
                  }}
                  onClick={() => { setMountsModalTab('mounts'); setShowObjectStorageWizard(false); setEditingObjectStorageBucket(null); }}
                >
                  Filesystem Mounts
                </button>
                <button
                  type="button"
                  style={{
                    background: 'transparent',
                    border: 'none',
                    borderBottom: mountsModalTab === 'object-storage' ? '2px solid var(--color-accent)' : '2px solid transparent',
                    padding: '8px 16px',
                    cursor: 'pointer',
                    color: mountsModalTab === 'object-storage' ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                    fontSize: 14,
                    fontWeight: 600,
                    transition: 'color 0.15s, border-color 0.15s',
                  }}
                  onClick={() => setMountsModalTab('object-storage')}
                >
                  Object Storage
                </button>
              </div>
              <button className="modal-close" onClick={handleCloseMountsModal}>&times;</button>
            </div>
            <div className="modal-body">
            {mountsModalTab === 'object-storage' ? (
              showObjectStorageWizard ? (
                <WorkspaceObjectStorageWizard
                  workspaceId={activeWorkspaceId}
                  existingBucket={editingObjectStorageBucket}
                  existingBucketNames={objectStorageConfig?.buckets.map((bucket) => bucket.name) ?? []}
                  onClose={() => {
                    setShowObjectStorageWizard(false);
                    setEditingObjectStorageBucket(null);
                  }}
                  onSaved={handleObjectStorageWizardSaved}
                />
              ) : (
                <div style={{ display: 'grid', gap: 16 }}>
                  <p className="userspace-muted" style={{ marginBottom: 0 }}>
                    S3-compatible object storage with auto-injected credentials at runtime.
                  </p>
                  {objectStorageLoading ? (
                    <p className="userspace-muted">Loading...</p>
                  ) : objectStorageConfig ? (
                    <>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <strong>Buckets</strong>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          onClick={() => {
                            setEditingObjectStorageBucket(null);
                            setShowObjectStorageWizard(true);
                          }}
                        >
                          <Plus size={14} />
                          New Bucket
                        </button>
                      </div>

                      <div style={{ display: 'grid', gap: 10 }}>
                        {objectStorageConfig.buckets.map((bucket) => (
                          <div key={bucket.name} style={{ border: '1px solid var(--color-border)', borderRadius: 8, padding: 12, display: 'grid', gap: 8 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <strong>{bucket.name}</strong>
                              {bucket.is_default && (
                                <span className="userspace-muted" style={{ fontSize: 12, padding: '2px 6px', borderRadius: 999, background: 'var(--color-bg-tertiary)' }}>
                                  Default
                                </span>
                              )}
                              <span style={{ marginLeft: 'auto' }} />
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => {
                                  setEditingObjectStorageBucket(bucket);
                                  setShowObjectStorageWizard(true);
                                }}
                                title="Edit bucket"
                              >
                                <Pencil size={12} />
                              </button>
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                onClick={() => { void handleDeleteObjectStorageBucket(bucket.name); }}
                                disabled={objectStorageConfig.buckets.length <= 1 || deletingObjectStorageBucket === bucket.name}
                                title={objectStorageConfig.buckets.length <= 1 ? 'At least one bucket must remain' : 'Delete bucket'}
                              >
                                {deletingObjectStorageBucket === bucket.name ? <MiniLoadingSpinner variant="icon" size={12} /> : <Trash2 size={12} />}
                              </button>
                            </div>
                            <div className="userspace-muted" style={{ fontSize: 12 }}>
                              {bucket.description || 'No description'}
                            </div>
                            <div style={{ display: 'grid', gap: 4 }}>
                              <span className="userspace-muted" style={{ fontSize: 12 }}>
                                Public root: <code>/{bucket.name}/{bucket.public_prefix}</code>
                              </span>
                              <span className="userspace-muted" style={{ fontSize: 12 }}>
                                Private root: <code>/{bucket.name}/{bucket.private_prefix}</code>
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <p className="userspace-muted">Object storage is unavailable for this workspace.</p>
                  )}
                </div>
              )
            ) : (
              <>
              {!mountsLoading && mounts.length === 0 && (
                <p className="userspace-muted" style={{ marginBottom: 12 }}>
                    Attach folders from your connected tools so apps running in this workspace can read and write to them.
                </p>
              )}
              {mountsLoading ? (
                <p className="userspace-muted">Loading...</p>
              ) : (
                <>
                  {mounts.length > 0 && (
                    <div style={{ marginBottom: 16 }}>
                      <strong style={{ display: 'block', marginBottom: 8, fontSize: 13 }}>Active mounts</strong>
                      <div className="userspace-mount-list">
                      {mounts.map((mount) => {
                        const isEjected = !mount.enabled;
                        const SyncModeIcon = getMountSyncModeIcon(mount.sync_mode);
                        const displaySourcePath = mount.source_path === '.'
                          ? '/'
                          : (mount.source_path.startsWith('/') ? mount.source_path : `/${mount.source_path}`);
                        return (
                        <div key={mount.id} className="userspace-mount-row" style={isEjected ? { opacity: 0.45, filter: 'grayscale(0.6)' } : undefined}>
                          <div className="userspace-mount-primary-row">
                            <span className="userspace-mount-path-flow">
                              <HardDrive size={13} className="userspace-mount-target-icon" />
                              <span className="userspace-mount-source-path">{displaySourcePath}</span>
                              <ArrowRight size={12} className="userspace-mount-path-arrow" />
                              <span className="userspace-mount-target-path">{mount.target_path}</span>
                              <span className="userspace-mount-tool-label">({mount.source_name ?? 'Unknown source'})</span>
                            </span>
                            <div className="userspace-mount-controls">
                              <span className="userspace-mount-sync-status">
                                {!mount.source_available && <span className="userspace-status-pill userspace-status-pill-danger" style={{ fontSize: 11 }} title="Mount source is no longer available">Disconnected</span>}
                                {mount.source_available && mount.source_type === 'ssh' && (syncingMountId === mount.id || previewingMountId === mount.id) && <span className="userspace-status-pill userspace-status-pill-warning" style={{ fontSize: 11 }}>In Progress</span>}
                                {mount.source_available && mount.source_type === 'ssh' && syncingMountId !== mount.id && previewingMountId !== mount.id && mount.sync_status === 'synced' && <span className="userspace-status-pill userspace-status-pill-success" style={{ fontSize: 11 }}>Synced</span>}
                                {mount.source_available && mount.source_type === 'ssh' && syncingMountId !== mount.id && previewingMountId !== mount.id && mount.sync_status === 'pending' && <span className="userspace-status-pill userspace-status-pill-info" style={{ fontSize: 11 }}>Pending</span>}
                                {mount.source_available && mount.source_type === 'ssh' && syncingMountId !== mount.id && previewingMountId !== mount.id && mount.sync_status === 'error' && (
                                  <span className="userspace-status-pill userspace-status-pill-danger" style={{ fontSize: 11 }} title={mount.last_sync_error ?? undefined}>Error</span>
                                )}
                                {mount.source_available && mount.source_type !== 'ssh' && <span className="userspace-status-pill userspace-status-pill-success" style={{ fontSize: 11 }}>Live</span>}
                              </span>
                              <div className="userspace-mount-actions">
                                {mount.source_type === 'ssh' && (
                                  <>
                                    <button
                                      className="btn btn-sm btn-secondary"
                                      onClick={() => {
                                        const modes = WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.map((o) => o.value);
                                        const currentIndex = modes.indexOf(mount.sync_mode);
                                        const nextMode = modes[(currentIndex + 1) % modes.length];
                                        void handleUpdateMountSyncMode(mount, nextMode);
                                      }}
                                      disabled={savingMountWatchId === mount.id || isEjected}
                                      title={getMountSyncModeDescription(mount.sync_mode)}
                                      style={{
                                        minWidth: 100,
                                        padding: '4px 8px',
                                        fontSize: 12,
                                        display: 'inline-flex',
                                        alignItems: 'center',
                                        gap: 5,
                                      }}
                                    >
                                      {savingMountWatchId === mount.id
                                        ? <MiniLoadingSpinner variant="icon" size={12} />
                                        : <><SyncModeIcon size={12} /> {getMountSyncModeLabel(mount.sync_mode)}</>}
                                      <span
                                        role="button"
                                        onClick={(e) => { e.stopPropagation(); setExpandedSyncModeInfo((v) => v === 'pinned' ? false : 'pinned'); }}
                                        onMouseEnter={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : 'hover')}
                                        onMouseLeave={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : false)}
                                        title="About sync modes"
                                        style={{ display: 'inline-flex', alignItems: 'center', marginLeft: 2, cursor: 'pointer', position: 'relative' }}
                                      >
                                        <Info size={11} />
                                        {expandedSyncModeInfo && (
                                          <>
                                            {expandedSyncModeInfo === 'pinned' && (
                                              <div onClick={(e) => { e.stopPropagation(); setExpandedSyncModeInfo(false); }} style={{ position: 'fixed', inset: 0, zIndex: 999 }} />
                                            )}
                                            <div
                                              onMouseEnter={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : 'hover')}
                                              onMouseLeave={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : false)}
                                              style={{
                                              position: 'absolute',
                                              top: '100%',
                                              right: 0,
                                              marginTop: 8,
                                              padding: 12,
                                              background: 'var(--color-bg-primary, #fff)',
                                              border: '1px solid var(--color-border, #ddd)',
                                              borderRadius: 8,
                                              boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
                                              fontSize: 12,
                                              display: 'grid',
                                              gap: 8,
                                              minWidth: 280,
                                              zIndex: 1000,
                                              whiteSpace: 'normal',
                                              textAlign: 'left',
                                            }}>
                                              {WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.map((option) => {
                                                const OptionIcon = option.icon;
                                                return (
                                                  <div key={option.value} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                                                    <OptionIcon size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                                                    <span><strong>{option.label}</strong>: {option.description}</span>
                                                  </div>
                                                );
                                              })}
                                            </div>
                                          </>
                                        )}
                                      </span>
                                    </button>
                                    <button
                                      className={`btn btn-sm ${mount.auto_sync_enabled ? 'btn-primary' : 'btn-secondary userspace-mount-toggle-btn-inactive'}`}
                                      onClick={() => handleToggleMountAutoSync(mount, !mount.auto_sync_enabled)}
                                      disabled={savingMountWatchId === mount.id || isEjected}
                                      title={mount.auto_sync_enabled ? 'Disable auto-sync watch mode' : 'Enable auto-sync watch mode'}
                                    >
                                      {savingMountWatchId === mount.id ? <MiniLoadingSpinner variant="icon" size={12} /> : (
                                        <span className="userspace-mount-toggle-icon">
                                          Auto
                                        </span>
                                      )}
                                    </button>
                                    {!mount.auto_sync_enabled && (
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => handleSyncMount(mount)}
                                        disabled={syncingMountId === mount.id || previewingMountId === mount.id || isEjected}
                                        title="Sync mount now"
                                      >
                                        {syncingMountId === mount.id || previewingMountId === mount.id
                                          ? <MiniLoadingSpinner variant="icon" size={12} />
                                          : <RefreshCw size={12} />}
                                      </button>
                                    )}
                                  </>
                                )}
                                {isEjected ? (
                                  <>
                                    <button
                                      className="btn btn-secondary btn-sm"
                                      onClick={() => void handleRemount(mount)}
                                      disabled={deletingMountId === mount.id || savingMountWatchId === mount.id}
                                      title="Remount"
                                    >
                                      {savingMountWatchId === mount.id ? <MiniLoadingSpinner variant="icon" size={12} /> : <HardDriveDownload size={12} />}
                                    </button>
                                    <button
                                      className="btn btn-secondary btn-sm"
                                      onClick={() => void handleDeleteMount(mount.id)}
                                      disabled={deletingMountId === mount.id}
                                      title="Delete mount permanently"
                                    >
                                      {deletingMountId === mount.id ? <MiniLoadingSpinner variant="icon" size={12} /> : <Trash2 size={12} />}
                                    </button>
                                  </>
                                ) : (
                                  <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => void handleEjectMount(mount)}
                                    disabled={deletingMountId === mount.id || savingMountWatchId === mount.id}
                                    title="Unmount"
                                  >
                                    {savingMountWatchId === mount.id ? <MiniLoadingSpinner variant="icon" size={12} /> : <HardDriveUpload size={12} />}
                                  </button>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="userspace-mount-desc-row">
                            {editingMountDescriptionId === mount.id ? (
                              <div className="userspace-mount-desc-edit">
                                <input
                                  type="text"
                                  className="form-input userspace-mount-desc-input"
                                  placeholder="Description for agents (optional)"
                                  value={editingMountDescriptionDraft}
                                  onChange={(e) => setEditingMountDescriptionDraft(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') void handleSaveMountDescription();
                                    if (e.key === 'Escape') {
                                      setEditingMountDescriptionId(null);
                                      setEditingMountDescriptionDraft('');
                                    }
                                  }}
                                  autoFocus
                                />
                                <button
                                  className="btn btn-primary btn-sm"
                                  onClick={() => void handleSaveMountDescription()}
                                  disabled={savingMountDescriptionId === mount.id}
                                  title="Save description"
                                >
                                  {savingMountDescriptionId === mount.id ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                </button>
                                <button
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => {
                                    setEditingMountDescriptionId(null);
                                    setEditingMountDescriptionDraft('');
                                  }}
                                  title="Cancel"
                                >
                                  <X size={12} />
                                </button>
                              </div>
                            ) : (
                              <div
                                className="userspace-mount-desc-display"
                                onClick={() => {
                                  setEditingMountDescriptionId(mount.id);
                                  setEditingMountDescriptionDraft(mount.description ?? '');
                                }}
                              >
                                <span className="userspace-mount-desc-text">
                                  {mount.description || 'No description for agents'}
                                </span>
                                <button
                                  className="inline-edit-btn userspace-mount-desc-edit-btn"
                                  title="Edit description"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setEditingMountDescriptionId(mount.id);
                                    setEditingMountDescriptionDraft(mount.description ?? '');
                                  }}
                                >
                                  <Pencil size={11} />
                                </button>
                              </div>
                            )}
                          </div>
                          {mount.sync_status === 'error' && mount.last_sync_error && (
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 4, color: 'var(--color-error, #c0392b)', fontSize: 12 }}>
                              <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 2 }} />
                              <span>{mount.last_sync_error}</span>
                            </div>
                          )}
                          {mount.sync_notice && (
                            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 6, marginTop: 6, color: 'var(--color-warning, #b26a00)', fontSize: 12 }}>
                              <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 2 }} />
                              <span>{mount.sync_notice}</span>
                            </div>
                          )}
                        </div>
                      );
                    })}
                    </div>
                    </div>
                  )}

                  {mountableSources.length > 0 ? (
                    <div className="userspace-env-var-form" style={{ marginTop: 12 }}>
                      <strong className="userspace-env-var-form-title">Add mount</strong>
                      {/* Source + Target in same row */}
                      <div style={{ display: 'grid', gap: 12, gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', alignItems: 'start' }}>
                        <div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 0, marginBottom: 6 }}>
                            <div className="userspace-muted" style={{ fontSize: 12, marginRight: 8 }}>
                              <strong>Source</strong>
                            </div>
                            {mountableSources.map((src) => {
                              const tabKey = `${src.mount_source_id}::${src.source_path}`;
                              const isActive = createMountActiveSourceTab === tabKey
                                || (!createMountActiveSourceTab && mountableSources[0] && tabKey === `${mountableSources[0].mount_source_id}::${mountableSources[0].source_path}`);
                              return (
                                <button
                                  key={tabKey}
                                  type="button"
                                  style={{
                                    background: 'transparent',
                                    border: 'none',
                                    borderBottom: isActive ? '2px solid var(--color-accent)' : '2px solid transparent',
                                    padding: '4px 10px',
                                    cursor: 'pointer',
                                    color: isActive ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
                                    fontSize: 12,
                                    transition: 'color 0.15s, border-color 0.15s',
                                  }}
                                  onClick={() => setCreateMountActiveSourceTab(tabKey)}
                                >
                                  <span style={{ marginRight: 5, opacity: 0.5, display: 'inline-flex', verticalAlign: 'middle' }}>
                                    {src.source_type === 'ssh' ? <Terminal size={11} /> : <HardDrive size={11} />}
                                  </span>
                                  {src.source_name}
                                </button>
                              );
                            })}
                          </div>
                          {mountableSources.map((src) => {
                            const tabKey = `${src.mount_source_id}::${src.source_path}`;
                            const isActiveTab = createMountActiveSourceTab === tabKey
                              || (!createMountActiveSourceTab && mountableSources[0] && tabKey === `${mountableSources[0].mount_source_id}::${mountableSources[0].source_path}`);
                            if (!isActiveTab) return null;
                            const browserRootPath = sourcePathToBrowserPath(src.source_path);
                            const isSelectedSource = (
                              src.mount_source_id === createMountSourceId
                              && src.source_path === createMountRootSourcePath
                              && !!createMountSourcePath
                            );
                            return (
                              <div key={tabKey}>
                                <ConstrainedPathBrowser
                                  currentPath={isSelectedSource ? createMountBrowserPath : ''}
                                  rootPath={browserRootPath}
                                  rootLabel={src.source_path === '.' ? '/' : `/${src.source_path}`}
                                  defaultExpanded={isSelectedSource}
                                  cacheKey={`${src.mount_source_id}:${src.source_path}`}
                                  stagedDirectories={createMountStagedSourceDirectories[
                                    getMountSourceBrowserStageKey(src.mount_source_id, src.source_path)
                                  ] ?? []}
                                  onStageDirectory={(path) => {
                                    const stageKey = getMountSourceBrowserStageKey(src.mount_source_id, src.source_path);
                                    const normalizedPath = normalizeMountBrowserPath(path);
                                    setCreateMountStagedSourceDirectories((current) => {
                                      const existingPaths = current[stageKey] ?? [];
                                      if (existingPaths.includes(normalizedPath)) {
                                        return current;
                                      }
                                      return {
                                        ...current,
                                        [stageKey]: [...existingPaths, normalizedPath].sort((left, right) => left.localeCompare(right)),
                                      };
                                    });
                                  }}
                                  onSelectPath={(selectedPath) => {
                                    setCreateMountSourceId(src.mount_source_id);
                                    setCreateMountRootSourcePath(src.source_path);
                                    setCreateMountBrowserPath(normalizeMountBrowserPath(selectedPath));
                                    setCreateMountSourcePath(browserPathToSourcePath(selectedPath));
                                  }}
                                  onBrowsePath={(path) => api.browseWorkspaceMountSource(activeWorkspaceId, {
                                    mount_source_id: src.mount_source_id,
                                    root_source_path: src.source_path,
                                    path,
                                  })}
                                />
                              </div>
                            );
                          })}
                        </div>
                        <div>
                          <div className="userspace-muted" style={{ marginBottom: 11, fontSize: 12 }}>
                            <strong>Target in workspace</strong>
                          </div>
                          <ConstrainedPathBrowser
                            currentPath={createMountTargetBrowserPath}
                            rootPath="/"
                            rootLabel="/"
                            defaultExpanded={false}
                            cacheKey={workspaceMountTargetBrowserCacheKey}
                            emptyMessage="Workspace directory is empty"
                            canSelectPath={(path) => {
                              const normalized = normalizeMountBrowserPath(path);
                              if (normalized === '/') return false;
                              const asTargetPath = `/workspace${normalized}`;
                              return !mounts.some((m) => m.target_path === asTargetPath);
                            }}
                            cannotSelectPathMessage="This path is already mounted"
                            isPathDisabled={(path) => {
                              const asTargetPath = `/workspace${normalizeMountBrowserPath(path)}`;
                              return mounts.some((m) => m.target_path === asTargetPath) ? 'Mounted' : null;
                            }}
                            stagedDirectories={createMountStagedTargetDirectories}
                            onStageDirectory={(path) => {
                              const normalizedPath = normalizeMountBrowserPath(path);
                              setCreateMountStagedTargetDirectories((current) => (
                                current.includes(normalizedPath)
                                  ? current
                                  : [...current, normalizedPath].sort((left, right) => left.localeCompare(right))
                              ));
                            }}
                            onSelectPath={(selectedPath) => {
                              const normalizedPath = normalizeMountBrowserPath(selectedPath);
                              setCreateMountTargetBrowserPath(normalizedPath);
                              setCreateMountTargetPath(browserPathToWorkspaceMountTargetPath(normalizedPath));
                            }}
                            onBrowsePath={browseWorkspaceMountTargetPath}
                          />
                        </div>
                      </div>
                      <input
                        type="text"
                        className="form-input"
                        placeholder="Description for agents (optional)"
                        value={createMountDescription}
                        onChange={(e) => setCreateMountDescription(e.target.value)}
                      />
                      {createMountSelectedSource?.source_type === 'ssh' && (() => {
                        const CreateSyncModeIcon = getMountSyncModeIcon(createMountSyncMode);
                        return (
                        <div style={{ display: 'grid', gap: 6 }}>
                          <label className="userspace-muted" style={{ fontSize: 12 }}>
                            <strong>Sync mode</strong>
                          </label>
                          <button
                            className="btn btn-sm btn-secondary"
                            type="button"
                            onClick={() => {
                              const modes = WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.map((o) => o.value);
                              const currentIndex = modes.indexOf(createMountSyncMode);
                              setCreateMountSyncMode(modes[(currentIndex + 1) % modes.length]);
                            }}
                            title={getMountSyncModeDescription(createMountSyncMode)}
                            style={{
                              padding: '8px 10px',
                              borderRadius: 6,
                              textAlign: 'left',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: 6,
                            }}
                          >
                            <CreateSyncModeIcon size={13} /> {getMountSyncModeLabel(createMountSyncMode)}
                            <span
                              role="button"
                              onClick={(e) => { e.stopPropagation(); setExpandedSyncModeInfo((v) => v === 'pinned' ? false : 'pinned'); }}
                              onMouseEnter={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : 'hover')}
                              onMouseLeave={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : false)}
                              title="About sync modes"
                              style={{ display: 'inline-flex', alignItems: 'center', marginLeft: 2, cursor: 'pointer', position: 'relative' }}
                            >
                              <Info size={11} />
                              {expandedSyncModeInfo && (
                                <>
                                  {expandedSyncModeInfo === 'pinned' && (
                                    <div onClick={(e) => { e.stopPropagation(); setExpandedSyncModeInfo(false); }} style={{ position: 'fixed', inset: 0, zIndex: 999 }} />
                                  )}
                                  <div
                                    onMouseEnter={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : 'hover')}
                                    onMouseLeave={() => setExpandedSyncModeInfo((v) => v === 'pinned' ? v : false)}
                                    style={{
                                    position: 'absolute',
                                    top: '100%',
                                    left: 0,
                                    marginTop: 8,
                                    padding: 12,
                                    background: 'var(--color-bg-primary, #fff)',
                                    border: '1px solid var(--color-border, #ddd)',
                                    borderRadius: 8,
                                    boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
                                    fontSize: 12,
                                    display: 'grid',
                                    gap: 8,
                                    minWidth: 280,
                                    zIndex: 1000,
                                    whiteSpace: 'normal',
                                    textAlign: 'left',
                                  }}>
                                    {WORKSPACE_MOUNT_SYNC_MODE_OPTIONS.map((option) => {
                                      const OptionIcon = option.icon;
                                      return (
                                        <div key={option.value} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                                          <OptionIcon size={13} style={{ flexShrink: 0, marginTop: 1 }} />
                                          <span><strong>{option.label}</strong>: {option.description}</span>
                                        </div>
                                      );
                                    })}
                                  </div>
                                </>
                              )}
                            </span>
                          </button>
                        </div>
                        );
                      })()}
                      <div className="userspace-env-var-form-actions">
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={handleCreateMount}
                          disabled={isCreateMountDisabled}
                          title={createMountEffectiveTargetPath === '/workspace' ? 'Select a folder under /workspace before adding the mount' : undefined}
                        >
                          {savingMount ? <MiniLoadingSpinner variant="icon" size={14} /> : <Plus size={14} />}
                          Add
                        </button>
                      </div>
                    </div>
                  ) : (
                    <p className="userspace-muted" style={{ marginTop: 12 }}>
                      No mount sources are configured.{' '}
                      {onNavigateToTools ? (
                        <a
                          href="#"
                          onClick={(e) => { e.preventDefault(); handleCloseMountsModal(); onNavigateToTools('mount-sources'); }}
                          style={{ color: 'var(--color-accent)' }}
                        >
                          Add a source in Tools
                        </a>
                      ) : (
                        'Add a mount source in Tools'
                      )}, then attach it here.
                    </p>
                  )}
                </>
              )}
              </>
            )}
            </div>
          </div>
        </div>
      )}

      {mountSyncPreview && mountSyncPreviewMount && (
        <div className="modal-overlay" onClick={handleCloseMountSyncPreview}>
          <div className="modal-content modal-small" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>{mountSyncPreviewIntent === 'sync' ? 'Review Destructive Sync' : 'Confirm Destructive Auto-Sync'}</h3>
              <button className="modal-close" onClick={handleCloseMountSyncPreview}>&times;</button>
            </div>
            <div className="modal-body" style={{ display: 'grid', gap: 14, maxHeight: '70vh', overflowY: 'auto' }}>
              <div style={{ display: 'grid', gap: 6 }}>
                <div className="userspace-muted" style={{ fontSize: 12 }}>
                  <strong>{mountSyncPreviewMount.source_name ?? 'Mount source'}</strong>
                  {' '}
                  {mountSyncPreviewMount.source_path === '.' ? '/' : formatMountSyncPreviewPath(mountSyncPreviewMount.source_path)}
                  {' '}
                  <ArrowRight size={11} style={{ verticalAlign: 'middle' }} />
                  {' '}
                  {mountSyncPreviewMount.target_path}
                </div>
                <div className="userspace-muted" style={{ fontSize: 12 }}>
                  <strong>Mode:</strong> {getMountSyncModeLabel(mountSyncPreview.sync_mode)}
                </div>
                {mountSyncPreview.sync_backend && (
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    <strong>Backend:</strong> {mountSyncPreview.sync_backend}
                  </div>
                )}
                {mountSyncPreview.sync_notice && (
                  <div style={{ display: 'flex', gap: 6, color: 'var(--color-warning, #b26a00)', fontSize: 12 }}>
                    <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 2 }} />
                    <span>{mountSyncPreview.sync_notice}</span>
                  </div>
                )}
                {mountSyncPreviewIntent !== 'sync' && (
                  <div style={{ display: 'flex', gap: 6, color: 'var(--color-warning, #b26a00)', fontSize: 12 }}>
                    <AlertCircle size={12} style={{ flexShrink: 0, marginTop: 2 }} />
                    <span>Auto-sync will keep using this destructive mode until Auto is disabled or the sync mode changes.</span>
                  </div>
                )}
              </div>

              <div style={{ display: 'grid', gap: 10 }}>
                {/* Source deletions */}
                <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--color-border)', background: 'var(--color-bg-tertiary)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <strong>Deletes from source: {mountSyncPreview.delete_from_source_count}</strong>
                    {mountSyncPreview.delete_from_source_count > mountSyncPreview.delete_from_source_paths.length && (
                      <span className="userspace-muted" style={{ fontSize: 11 }}>
                        showing {mountSyncPreview.delete_from_source_paths.length} of {mountSyncPreview.delete_from_source_count}
                      </span>
                    )}
                  </div>
                  {mountSyncPreview.delete_from_source_paths.length > 0 ? (
                    <>
                      <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                        Files or folders that will be removed from the remote source.
                      </div>
                      <div style={{ marginTop: 8, maxHeight: 250, overflowY: 'auto', borderRadius: 4, border: '1px solid var(--color-border)' }}>
                        {mountSyncPreview.delete_from_source_paths.map((path, i) => (
                          <div
                            key={`source:${path}`}
                            title={formatMountSyncPreviewPath(path)}
                            style={{
                              fontSize: 12,
                              fontFamily: 'var(--font-mono)',
                              padding: '4px 8px',
                              overflowWrap: 'anywhere',
                              background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.03)',
                            }}
                          >
                            {formatMountSyncPreviewPath(path)}
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>No source deletions detected.</div>
                  )}
                </div>

                {/* Target deletions */}
                <div style={{ padding: 12, borderRadius: 8, border: '1px solid var(--color-border)', background: 'var(--color-bg-tertiary)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
                    <strong>Deletes from target: {mountSyncPreview.delete_from_target_count}</strong>
                    {mountSyncPreview.delete_from_target_count > mountSyncPreview.delete_from_target_paths.length && (
                      <span className="userspace-muted" style={{ fontSize: 11 }}>
                        showing {mountSyncPreview.delete_from_target_paths.length} of {mountSyncPreview.delete_from_target_count}
                      </span>
                    )}
                  </div>
                  {mountSyncPreview.delete_from_target_paths.length > 0 ? (
                    <>
                      <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>
                        Files or folders that will be removed from the workspace target.
                      </div>
                      <div style={{ marginTop: 8, maxHeight: 250, overflowY: 'auto', borderRadius: 4, border: '1px solid var(--color-border)' }}>
                        {mountSyncPreview.delete_from_target_paths.map((path, i) => (
                          <div
                            key={`target:${path}`}
                            title={formatMountSyncPreviewPath(path)}
                            style={{
                              fontSize: 12,
                              fontFamily: 'var(--font-mono)',
                              padding: '4px 8px',
                              overflowWrap: 'anywhere',
                              background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.03)',
                            }}
                          >
                            {formatMountSyncPreviewPath(path)}
                          </div>
                        ))}
                      </div>
                    </>
                  ) : (
                    <div className="userspace-muted" style={{ fontSize: 12, marginTop: 4 }}>No target deletions detected.</div>
                  )}
                </div>

                {(mountSyncPreview.delete_from_source_count > mountSyncPreview.delete_from_source_paths.length
                  || mountSyncPreview.delete_from_target_count > mountSyncPreview.delete_from_target_paths.length) && (
                  <div className="userspace-muted" style={{ fontSize: 12 }}>
                    Full list truncated. Run preview again right before syncing if the source or target changes.
                  </div>
                )}
              </div>
            </div>
            <div className="modal-footer" style={{ justifyContent: 'space-between' }}>
              <button className="btn btn-secondary" onClick={handleCloseMountSyncPreview}>
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={() => void handleConfirmMountSyncPreview()}
                disabled={syncingMountId === mountSyncPreviewMount.id || savingMountWatchId === mountSyncPreviewMount.id}
              >
                {mountSyncPreviewIntent === 'sync'
                  ? (syncingMountId === mountSyncPreviewMount.id ? 'Syncing...' : 'Confirm Sync')
                  : (savingMountWatchId === mountSyncPreviewMount.id ? 'Saving...' : mountSyncPreviewIntent === 'enable-auto' ? 'Enable Auto' : 'Confirm Mode Change')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* === Admin workspaces modal === */}
      {showAdminWorkspacesModal && (
        <AdminWorkspaceModal
          isOpen={showAdminWorkspacesModal}
          onClose={() => setShowAdminWorkspacesModal(false)}
          currentUser={currentUser}
          onSelectWorkspace={(ws) => {
            setWorkspaces((prev) => {
              if (prev.some((w) => w.id === ws.id)) return prev;
              return [...prev, ws];
            });
            setActiveWorkspaceId(ws.id);
            setRuntimeStatus(null);
          }}
          onWorkspaceDeleted={(wsId) => {
            setWorkspaces((prev) => prev.filter((w) => w.id !== wsId));
            setWorkspacesTotal((prev) => Math.max(0, prev - 1));
            if (activeWorkspaceId === wsId) {
              setActiveWorkspaceId(workspaces.find((w) => w.id !== wsId)?.id ?? null);
            }
          }}
        />
      )}

      {/* === Members modal === */}
      {showMembersModal && activeWorkspace && (
        <MemberManagementModal
          isOpen={showMembersModal}
          onClose={() => setShowMembersModal(false)}
          members={pendingMembers}
          onSave={handleSaveMembers}
          allUsers={allUsers}
          ownerId={activeWorkspace.owner_user_id}
          entityType="workspace"
          formatUserLabel={formatUserLabel}
          saving={savingMembers}
        />
      )}

      {showAgentAccessModal && activeWorkspace && (
        <AgentAccessModal
          isOpen={showAgentAccessModal}
          onClose={() => setShowAgentAccessModal(false)}
          sourceWorkspace={activeWorkspace}
          availableWorkspaces={agentGrantWorkspaces}
          grants={agentGrants}
          onUpsert={handleUpsertAgentGrant}
          onRevoke={handleRevokeAgentGrant}
          loading={agentGrantsLoading}
          savingTargetId={savingAgentGrantTargetId}
          revokingTargetId={revokingAgentGrantTargetId}
        />
      )}

      {/* === Share modal === */}
      {showShareModal && activeWorkspace && (
        <div className="modal-overlay" onClick={() => setShowShareModal(false)}>
          <div className="modal-content modal-small userspace-share-modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Share Workspace</h3>
              <button className="modal-close" onClick={() => setShowShareModal(false)}>&times;</button>
            </div>
            <div className="modal-body">
              {loadingShareStatus ? (
                <p className="userspace-muted">Loading share settings...</p>
              ) : (
                <>
                  <div className="userspace-share-link-pane">
                    {shareLinkType !== 'subdomain' && (
                      <>
                        <label htmlFor="userspace-share-slug" className="userspace-share-label">Custom slug</label>
                        <div className="userspace-share-slug-row">
                          <input
                            id="userspace-share-slug"
                            value={shareSlugDraft}
                            onChange={(event) => {
                              setShareSlugDraft(normalizeShareSlugInput(event.target.value));
                              setShareSlugAvailable(null);
                            }}
                            placeholder="custom_slug"
                            autoComplete="off"
                          />
                        </div>
                        {shareSlugAvailable !== null && (
                          <div className={`userspace-share-meta ${shareSlugAvailable ? '' : 'userspace-error'}`}>
                            {shareSlugAvailable ? 'Slug is available' : 'Slug is unavailable'}
                          </div>
                        )}
                      </>
                    )}

                    {activeShareLinkStatus?.has_share_link && (
                      <div className="userspace-share-link-type-toggle">
                        <label className="userspace-share-radio-option">
                          <input
                            type="radio"
                            name="shareLinkType"
                            value="named"
                            checked={shareLinkType === 'named'}
                            onChange={() => setShareLinkType('named')}
                          />
                          Named
                        </label>
                        <label className="userspace-share-radio-option">
                          <input
                            type="radio"
                            name="shareLinkType"
                            value="anonymous"
                            checked={shareLinkType === 'anonymous'}
                            onChange={() => setShareLinkType('anonymous')}
                          />
                          Anonymous
                        </label>
                        <label className="userspace-share-radio-option">
                          <input
                            type="radio"
                            name="shareLinkType"
                            value="subdomain"
                            checked={shareLinkType === 'subdomain'}
                            onChange={() => setShareLinkType('subdomain')}
                            disabled={!shareSubdomainEnabled}
                          />
                          Subdomain
                        </label>
                      </div>
                    )}

                    {activeShareLinkStatus?.has_share_link && !shareSubdomainEnabled && shareSubdomainDisabledReason && (
                      <div className="userspace-share-meta">
                        {shareSubdomainDisabledReason}
                      </div>
                    )}

                    {showProtectedSubdomainNotice && (
                      <div className="userspace-share-warning-banner" role="alert">
                        Warning: if this workspace has already been unlocked in this browser, opening the subdomain link again may not prompt you to login. Protection is still enforced for new sessions and other browsers.
                      </div>
                    )}

                    {activeShareLinkStatus?.has_share_link && effectiveShareUrl ? (
                      <>
                        <label htmlFor="userspace-share-url" className="userspace-share-label">Active share URL</label>
                        <div className="userspace-share-url-copy-wrap">
                          <input id="userspace-share-url" value={effectiveShareUrl} readOnly />
                          <button
                            type="button"
                            className="userspace-share-inline-copy"
                            onClick={handleCopyShareLink}
                            title="Copy share URL"
                            aria-label="Copy share URL"
                          >
                            {shareCopied ? <Check size={12} /> : <Copy size={12} />}
                          </button>
                        </div>
                        <div className="userspace-share-meta">
                          {activeShareLinkStatus.created_at ? `Created ${formatSnapshotTimestamp(activeShareLinkStatus.created_at)}` : 'Share link active'}
                        </div>
                      </>
                    ) : (
                      <p className="userspace-muted">No active share link for this workspace.</p>
                    )}
                  </div>

                  <div className="userspace-share-controls">
                    <h4>Access Controls</h4>
                    <div className="userspace-share-access-row">
                      <label htmlFor="userspace-share-access-mode" className="userspace-share-label">Access mode</label>
                      <select
                        id="userspace-share-access-mode"
                        value={shareAccessMode}
                        onChange={(event) => setShareAccessMode(event.target.value as UserSpaceShareAccessMode)}
                        disabled={savingShareAccess || sharingWorkspace || revokingShareLink}
                      >
                        <option value="token">Tokenized public link</option>
                        <option value="password">Password protected</option>
                        <option value="authenticated_users">Any authenticated user</option>
                        <option value="selected_users">Selected users only</option>
                        <option value="ldap_groups">Selected LDAP groups only</option>
                      </select>
                    </div>

                    {shareAccessMode === 'password' && (
                      <div className="userspace-share-access-row">
                        <label htmlFor="userspace-share-password" className="userspace-share-label">
                          Share password {activeShareLinkStatus?.has_password ? '(set)' : '(required)'}
                        </label>
                        <input
                          id="userspace-share-password"
                          type="password"
                          value={sharePasswordDraft}
                          onChange={(event) => setSharePasswordDraft(event.target.value)}
                          placeholder={activeShareLinkStatus?.has_password ? 'Enter new password to update' : 'Enter password'}
                          autoComplete="new-password"
                        />
                      </div>
                    )}

                    {shareAccessMode === 'selected_users' && (
                      <div className="userspace-share-access-row">
                        <label className="userspace-share-label">Allowed users</label>
                        <div className="userspace-share-user-grid">
                          {shareSelectableUsers.map((user) => (
                            <label key={user.id} className="userspace-share-user-option">
                              <input
                                type="checkbox"
                                checked={shareSelectedUserIdsDraft.includes(user.id)}
                                onChange={() => handleToggleShareSelectedUser(user.id)}
                              />
                              <span>{formatUserLabel(user, user.id)}</span>
                            </label>
                          ))}
                        </div>
                      </div>
                    )}

                    {shareAccessMode === 'ldap_groups' && (
                      <div className="userspace-share-access-row">
                        <label className="userspace-share-label">Allowed LDAP groups</label>
                        <div className="userspace-share-slug-row">
                          {ldapDiscoveredGroups.length > 0 ? (
                            <LdapGroupSelect
                              value={shareLdapGroupDraft}
                              onChange={setShareLdapGroupDraft}
                              groups={ldapDiscoveredGroups}
                              emptyOptionLabel="Select an LDAP group..."
                            />
                          ) : (
                            <input
                              value={shareLdapGroupDraft}
                              onChange={(event) => setShareLdapGroupDraft(event.target.value)}
                              placeholder="cn=group,ou=groups,dc=example,dc=com"
                              autoComplete="off"
                            />
                          )}
                          <button className="btn btn-secondary" onClick={handleAddShareLdapGroup} type="button">Add Group</button>
                        </div>
                        {loadingLdapGroups ? (
                          <p className="userspace-share-meta">Loading LDAP groups…</p>
                        ) : ldapDiscoveredGroups.length > 0 ? (
                          <p className="userspace-share-meta">Groups are discovered from the configured LDAP base domain.</p>
                        ) : (
                          <p className="userspace-share-meta">Could not auto-discover LDAP groups. Enter group DN manually.</p>
                        )}
                        {shareSelectedLdapGroupsDraft.length > 0 && (
                          <div className="userspace-share-group-list">
                            {shareSelectedLdapGroupsDraft.map((groupDn) => (
                              <div key={groupDn} className="userspace-share-group-item">
                                <span>{groupDn}</span>
                                <button
                                  className="btn btn-secondary"
                                  onClick={() => handleRemoveShareLdapGroup(groupDn)}
                                  type="button"
                                >
                                  Remove
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="userspace-share-actions userspace-share-actions-single">
                      <button
                        className="btn btn-secondary"
                        onClick={handleSaveShareAccess}
                        disabled={savingShareAccess || sharingWorkspace || revokingShareLink || checkingShareSlug}
                      >
                        {savingShareAccess ? 'Saving Access...' : 'Save Access'}
                      </button>
                    </div>

                    <h4>Sharing Controls</h4>
                    <div className="userspace-share-actions">
                      <button
                        className="btn btn-secondary"
                        onClick={handleCopyShareLink}
                        disabled={sharingWorkspace || revokingShareLink || checkingShareSlug || savingShareAccess || shareHasUnsavedChanges}
                      >
                        {shareCopied ? 'Copied' : 'Copy Link'}
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={handleOpenFullPreview}
                        disabled={sharingWorkspace || revokingShareLink || checkingShareSlug || savingShareAccess || shareHasUnsavedChanges}
                      >
                        Open Preview
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={handleRotateShareLink}
                        disabled={sharingWorkspace || revokingShareLink || checkingShareSlug || savingShareAccess}
                      >
                        {rotatingShareLink ? 'Rotating...' : 'Rotate Link'}
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={handleRevokeShareLink}
                        disabled={revokingShareLink || sharingWorkspace || checkingShareSlug || savingShareAccess || !activeShareLinkStatus?.has_share_link}
                      >
                        {revokingShareLink ? 'Revoking...' : 'Revoke Link'}
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
