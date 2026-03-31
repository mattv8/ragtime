import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react';
import { AlertCircle, Check, ChevronDown, ChevronRight, Copy, Database, ExternalLink, File, History, KeyRound, Link2, Maximize2, Minimize2, Pencil, Play, Plus, RotateCw, Save, Shield, Slash, Square, Terminal, Trash2, Users, X } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { json } from '@codemirror/lang-json';
import { css } from '@codemirror/lang-css';
import { html } from '@codemirror/lang-html';
import { markdown } from '@codemirror/lang-markdown';
import { xml } from '@codemirror/lang-xml';
import { yaml } from '@codemirror/lang-yaml';
import { sql } from '@codemirror/lang-sql';
import { keymap, Decoration, type DecorationSet, EditorView } from '@codemirror/view';
import { StateField, type Extension } from '@codemirror/state';
import { openSearchPanel } from '@codemirror/search';
import { diffLines } from 'diff';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';

import { api } from '@/api';
import {
  clearInterruptDismiss,
  resolveWorkspaceInterruptStateFromSummary,
} from '@/utils';
import type { InterruptChatStateSnapshot } from '@/utils/cookies';
import AdminWorkspaceModal from './shared/AdminWorkspaceModal';
import { MemberManagementModal, type Member } from './shared/MemberManagementModal';
import { MiniLoadingSpinner } from './shared/MiniLoadingSpinner';
import { ToolSelectorDropdown, type ToolGroupInfo } from './shared/ToolSelectorDropdown';
import type { User, UserSpaceAvailableTool, UserSpaceCollabMessage, UserSpaceFileInfo, UserSpaceLiveDataConnection, UserSpaceRuntimeStatusResponse, UserSpaceShareAccessMode, UserSpaceSnapshot, UserSpaceSnapshotBranch, UserSpaceSnapshotDiffSummary, UserSpaceSnapshotFileDiff, UserSpaceWorkspace, UserSpaceWorkspaceEnvVar, UserSpaceWorkspaceMember, UserSpaceWorkspaceShareLinkStatus } from '@/types';
import { buildUserSpaceTree, collectFilePaths, getAncestorFolderPaths, listFolderPaths } from '@/utils/userspaceTree';
import { ChatPanel } from './ChatPanel';
import { LdapGroupSelect } from './LdapGroupSelect';
import { ResizeHandle } from './ResizeHandle';
import { UserSpaceArtifactPreview } from './UserSpaceArtifactPreview';

interface UserSpacePanelProps {
  currentUser: User;
  debugMode?: boolean;
  onFullscreenChange?: (fullscreen: boolean) => void;
}

interface WorkspaceChatState {
  hasLive: boolean;
  hasInterrupted: boolean;
}

const DEFAULT_WORKSPACE_CHAT_STATE: WorkspaceChatState = { hasLive: false, hasInterrupted: false };

function normalizeWorkspacePath(value: string): string {
  return value.trim().replace(/^\/+|\/+$/g, '').replace(/\/+/g, '/');
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

/** Resolve a CodeMirror language extension based on file path. */
function getLanguageExtensionForPath(filePath: string) {
  const lower = filePath.toLowerCase();
  if (/\.[cm]?tsx?$/i.test(lower)) return javascript({ typescript: true, jsx: /x$/i.test(lower) });
  if (/\.[cm]?jsx?$/i.test(lower)) return javascript({ typescript: false, jsx: true });
  if (/\.py$/i.test(lower)) return python();
  if (/\.json[c5]?$/i.test(lower) || lower.endsWith('.jsonl')) return json();
  if (/\.css$/i.test(lower) || lower.endsWith('.scss') || lower.endsWith('.less')) return css();
  if (/\.html?$/i.test(lower) || lower.endsWith('.svelte') || lower.endsWith('.vue')) return html();
  if (/\.ya?ml$/i.test(lower)) return yaml();
  if (/\.xml$/i.test(lower) || lower.endsWith('.svg')) return xml();
  if (/\.sql$/i.test(lower)) return sql();
  if (/\.mdx?$/i.test(lower) || lower.endsWith('.markdown')) return markdown();
  return null;
}

/**
 * Compute an aligned side-by-side diff using LCS-based diffLines.
 * Both sides get the same number of lines with blank padding inserted
 * opposite added/deleted hunks so line numbers stay in sync.
 */
interface AlignedDiffResult {
  beforeText: string;
  afterText: string;
  beforeDeletedLines: Set<number>;
  afterAddedLines: Set<number>;
  beforePaddingLines: Set<number>;
  afterPaddingLines: Set<number>;
}

function computeAlignedDiff(before: string, after: string): AlignedDiffResult {
  const changes = diffLines(before, after);

  const beforeArr: string[] = [];
  const afterArr: string[] = [];
  const beforeDeletedLines = new Set<number>();
  const afterAddedLines = new Set<number>();
  const beforePaddingLines = new Set<number>();
  const afterPaddingLines = new Set<number>();

  for (const change of changes) {
    const raw = change.value;
    const lines = raw.endsWith('\n') ? raw.slice(0, -1).split('\n') : raw.split('\n');

    if (change.removed) {
      for (const line of lines) {
        beforeArr.push(line);
        afterArr.push('');
        beforeDeletedLines.add(beforeArr.length);
        afterPaddingLines.add(afterArr.length);
      }
    } else if (change.added) {
      for (const line of lines) {
        beforeArr.push('');
        afterArr.push(line);
        afterAddedLines.add(afterArr.length);
        beforePaddingLines.add(beforeArr.length);
      }
    } else {
      for (const line of lines) {
        beforeArr.push(line);
        afterArr.push(line);
      }
    }
  }

  return {
    beforeText: beforeArr.join('\n'),
    afterText: afterArr.join('\n'),
    beforeDeletedLines,
    afterAddedLines,
    beforePaddingLines,
    afterPaddingLines,
  };
}

const diffLineDeletedMark = Decoration.line({ class: 'cm-diff-line-deleted' });
const diffLineAddedMark = Decoration.line({ class: 'cm-diff-line-added' });
const diffLinePaddingMark = Decoration.line({ class: 'cm-diff-line-padding' });

/** Build a CM extension that decorates specific 1-indexed lines. */
function buildDiffHighlightExtension(lineNumbers: Set<number>, decoration: Decoration) {
  return StateField.define<DecorationSet>({
    create(state) {
      const builder: import('@codemirror/state').Range<Decoration>[] = [];
      for (let i = 1; i <= state.doc.lines; i++) {
        if (lineNumbers.has(i)) {
          builder.push(decoration.range(state.doc.line(i).from));
        }
      }
      return Decoration.set(builder);
    },
    update(value) {
      return value;
    },
    provide: (field) => EditorView.decorations.from(field),
  });
}

const LAST_WORKSPACE_COOKIE_PREFIX = 'userspace_last_workspace_id_';
const LAST_WORKSPACE_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
const USERSPACE_LAYOUT_COOKIE_PREFIX = 'userspace_layout_';
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
const SNAPSHOT_FILE_DIFF_CACHE_MAX_ENTRIES = 20;

function getLastWorkspaceCookieName(userId: string): string {
  return `${LAST_WORKSPACE_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

function getUserSpaceLayoutCookieName(userId: string): string {
  return `${USERSPACE_LAYOUT_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
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

function formatSnapshotDiffStatus(status: 'A' | 'D' | 'M' | 'R'): string {
  switch (status) {
    case 'A':
      return 'Added';
    case 'D':
      return 'Deleted';
    case 'R':
      return 'Renamed';
    default:
      return 'Modified';
  }
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

export function UserSpacePanel({ currentUser, debugMode = false, onFullscreenChange }: UserSpacePanelProps) {
  const previewEntryPath = 'dashboard/main.ts';
  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [workspacesTotal, setWorkspacesTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [statusOverlayVisible, setStatusOverlayVisible] = useState(false);
  const [statusOverlayFading, setStatusOverlayFading] = useState(false);
  const [statusOverlayPinned, setStatusOverlayPinned] = useState(false);
  const [statusOverlayInteracting, setStatusOverlayInteracting] = useState(false);

  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
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
  const [navigatingSnapshots, setNavigatingSnapshots] = useState(false);
  const [restoringSnapshotId, setRestoringSnapshotId] = useState<string | null>(null);
  const [availableTools, setAvailableTools] = useState<UserSpaceAvailableTool[]>([]);
  const [toolGroups, setToolGroups] = useState<ToolGroupInfo[]>([]);

  const [selectedFilePath, setSelectedFilePath] = useState<string>('dashboard/main.ts');
  const [fileContent, setFileContent] = useState<string>('');
  const [fileDirty, setFileDirty] = useState(false);
  const [fileContentCache, setFileContentCache] = useState<Record<string, { content: string; updatedAt: string }>>({});
  const [previewLiveDataConnections, setPreviewLiveDataConnections] = useState<UserSpaceLiveDataConnection[]>([]);
  const [previewExecuting, setPreviewExecuting] = useState(false);
  const [previewRefreshCounter, setPreviewRefreshCounter] = useState(0);
  const [previewCapabilityToken, setPreviewCapabilityToken] = useState<string | null>(null);
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
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [creatingWorkspaceStatus, setCreatingWorkspaceStatus] = useState<string | null>(null);
  const [deletingWorkspaceId, setDeletingWorkspaceId] = useState<string | null>(null);
  const [deletingWorkspaceStatus, setDeletingWorkspaceStatus] = useState<string | null>(null);
  const [sharingWorkspace, setSharingWorkspace] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const [shareLinkType, setShareLinkType] = useState<'named' | 'anonymous'>('named');
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
  const [showAdminWorkspacesModal, setShowAdminWorkspacesModal] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [pendingMembers, setPendingMembers] = useState<UserSpaceWorkspaceMember[]>([]);
  const [savingMembers, setSavingMembers] = useState(false);
  const [showEnvVarsModal, setShowEnvVarsModal] = useState(false);
  const [envVars, setEnvVars] = useState<UserSpaceWorkspaceEnvVar[]>([]);
  const [envVarsLoading, setEnvVarsLoading] = useState(false);
  const [draftEnvKey, setDraftEnvKey] = useState('');
  const [draftEnvValue, setDraftEnvValue] = useState('');
  const [draftEnvDescription, setDraftEnvDescription] = useState('');
  const [editingEnvKey, setEditingEnvKey] = useState<string | null>(null);
  const [editingEnvValueDraft, setEditingEnvValueDraft] = useState('');
  const [editingEnvDescKey, setEditingEnvDescKey] = useState<string | null>(null);
  const [editingEnvDescriptionDraft, setEditingEnvDescriptionDraft] = useState('');
  const [savingEnvVar, setSavingEnvVar] = useState(false);
  const [deletingEnvKey, setDeletingEnvKey] = useState<string | null>(null);
  const [confirmDeleteEnvKey, setConfirmDeleteEnvKey] = useState<string | null>(null);
  const [copiedEnvKey, setCopiedEnvKey] = useState<string | null>(null);
  const [showToolPicker, setShowToolPicker] = useState(false);
  const [showSnapshots, setShowSnapshots] = useState(false);
  const [snapshotsLoadedForWorkspace, setSnapshotsLoadedForWorkspace] = useState<string | null>(null);
  const [expandedSnapshotIds, setExpandedSnapshotIds] = useState<Set<string>>(new Set());
  const [snapshotDiffSummaries, setSnapshotDiffSummaries] = useState<Record<string, UserSpaceSnapshotDiffSummary>>({});
  const snapshotDiffSummariesRef = useRef(snapshotDiffSummaries);
  snapshotDiffSummariesRef.current = snapshotDiffSummaries;
  const [loadingSnapshotDiffSummaryIds, setLoadingSnapshotDiffSummaryIds] = useState<Record<string, boolean>>({});
  const [snapshotDiffSummaryErrors, setSnapshotDiffSummaryErrors] = useState<Record<string, string>>({});
  const snapshotFileDiffCacheRef = useRef<Map<string, UserSpaceSnapshotFileDiff>>(new Map());
  const [activeSnapshotFileDiff, setActiveSnapshotFileDiff] = useState<UserSpaceSnapshotFileDiff | null>(null);
  const [activeSnapshotFileDiffKey, setActiveSnapshotFileDiffKey] = useState<string | null>(null);
  const [activeSnapshotFileDiffLoading, setActiveSnapshotFileDiffLoading] = useState(false);
  const [activeSnapshotFileDiffError, setActiveSnapshotFileDiffError] = useState<string | null>(null);

  const issueWorkspaceCapabilityToken = useCallback(async (workspaceId: string, capabilities: string[]): Promise<string> => {
    const response = await api.issueUserSpaceCapabilityToken(workspaceId, capabilities);
    return response.token;
  }, []);
  const toolPickerRef = useRef<HTMLDivElement>(null);
  const workspaceDropdownRef = useRef<HTMLDivElement>(null);
  const selectedFilePathRef = useRef(selectedFilePath);
  const fileContentCacheRef = useRef(fileContentCache);
  const loadWorkspaceDataRequestIdRef = useRef(0);
  const loadChangedFileStateRequestIdRef = useRef(0);
  const loadRuntimeStatusRequestIdRef = useRef(0);
  const snapshotDiffSummaryRequestIdsRef = useRef<Record<string, number>>({});
  const snapshotFileDiffRequestIdRef = useRef(0);
  const snapshotFileDiffHoverTimerRef = useRef<number | null>(null);
  const snapshotFileDiffDismissTimerRef = useRef<number | null>(null);
  const snapshotFileDiffPinnedRef = useRef(false);
  const snapshotFileDiffEnteredOverlayRef = useRef(false);
  const snapshotDiffBeforeWrapRef = useRef<HTMLDivElement | null>(null);
  const snapshotDiffAfterWrapRef = useRef<HTMLDivElement | null>(null);
  const snapshotDiffScrollSyncingRef = useRef(false);
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
  const statusOverlayDismissedSignatureRef = useRef<string | null>(null);
  const lastWorkspaceCookieName = useMemo(() => getLastWorkspaceCookieName(currentUser.id), [currentUser.id]);
  const userSpaceLayoutCookieName = useMemo(() => getUserSpaceLayoutCookieName(currentUser.id), [currentUser.id]);

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

  const availableToolIds = useMemo(
    () => availableTools.map((tool) => tool.id),
    [availableTools]
  );
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
      .filter((group) => group.snapshots.length > 0);
  }, [snapshotBranches, snapshots]);

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

  const activeSnapshotDiffLanguageExtensions = useMemo(() => {
    const path = activeSnapshotFileDiff?.path ?? '';
    const ext = getLanguageExtensionForPath(path);
    return ext ? [ext] : [];
  }, [activeSnapshotFileDiff?.path]);

  const snapshotAlignedDiff = useMemo(() => {
    if (!activeSnapshotFileDiff || activeSnapshotFileDiff.is_binary) return null;
    return computeAlignedDiff(activeSnapshotFileDiff.before_content, activeSnapshotFileDiff.after_content);
  }, [activeSnapshotFileDiff]);

  const snapshotDiffBeforeExtensions = useMemo(() => {
    const exts: Extension[] = [...activeSnapshotDiffLanguageExtensions];
    if (snapshotAlignedDiff) {
      if (snapshotAlignedDiff.beforeDeletedLines.size > 0) {
        exts.push(buildDiffHighlightExtension(snapshotAlignedDiff.beforeDeletedLines, diffLineDeletedMark));
      }
      if (snapshotAlignedDiff.beforePaddingLines.size > 0) {
        exts.push(buildDiffHighlightExtension(snapshotAlignedDiff.beforePaddingLines, diffLinePaddingMark));
      }
    }
    return exts;
  }, [activeSnapshotDiffLanguageExtensions, snapshotAlignedDiff]);

  const snapshotDiffAfterExtensions = useMemo(() => {
    const exts: Extension[] = [...activeSnapshotDiffLanguageExtensions];
    if (snapshotAlignedDiff) {
      if (snapshotAlignedDiff.afterAddedLines.size > 0) {
        exts.push(buildDiffHighlightExtension(snapshotAlignedDiff.afterAddedLines, diffLineAddedMark));
      }
      if (snapshotAlignedDiff.afterPaddingLines.size > 0) {
        exts.push(buildDiffHighlightExtension(snapshotAlignedDiff.afterPaddingLines, diffLinePaddingMark));
      }
    }
    return exts;
  }, [activeSnapshotDiffLanguageExtensions, snapshotAlignedDiff]);

  const snapshotDiffCodeMirrorSetup = useMemo(() => ({
    ...USERSPACE_CODEMIRROR_BASIC_SETUP,
    autocompletion: false,
    closeBrackets: false,
    foldGutter: false,
    highlightActiveLine: false,
  }), []);

  const codeMirrorExtensions = useMemo(
    () => [
      javascript({ typescript: true, jsx: true }),
      keymap.of([
        {
          key: 'Mod-f',
          run: openSearchPanel,
          preventDefault: true,
        },
      ]),
    ],
    []
  );

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
    if (phase === 'deps_install') return 'installing deps';
    return phase.replace(/_/g, ' ');
  }, [runtimeStatus?.runtime_operation_phase]);

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
        setWorkspaces((prev) => [...prev, ...page.items]);
      } else {
        let nextItems = page.items;

        if (activeWorkspaceId && !nextItems.some((workspace) => workspace.id === activeWorkspaceId)) {
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
          const matchingWorkspace = lastWorkspaceId
            ? nextItems.find((workspace) => workspace.id === lastWorkspaceId)
            : null;

          if (matchingWorkspace) {
            setActiveWorkspaceId(matchingWorkspace.id);
          } else if (lastWorkspaceId) {
            try {
              const workspace = await api.getUserSpaceWorkspace(lastWorkspaceId);
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

    void pollWorkspaceConversationStates();
    // Non-active workspace badges are secondary indicators; poll at a relaxed cadence
    const timer = window.setInterval(() => {
      void pollWorkspaceConversationStates();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [workspaces, activeWorkspaceId, currentUser.id]);

  const loadWorkspaceData = useCallback(async (workspaceId: string) => {
    const requestId = ++loadWorkspaceDataRequestIdRef.current;

    try {
      const nextEntries = await api.listUserSpaceFiles(workspaceId, { includeDirs: true });

      const nextFiles = nextEntries.filter((entry) => entry.entry_type !== 'directory');

      if (requestId !== loadWorkspaceDataRequestIdRef.current) {
        return;
      }

      // Skip state updates if the file list is unchanged (same paths and timestamps).
      if (fileEntriesFingerprint(nextEntries) === fileEntriesFingerprint(fileBrowserEntriesRef.current)) {
        return;
      }

      setFileBrowserEntries(nextEntries);
      setFiles(nextFiles);

      const validPaths = new Set(nextFiles.map((file) => file.path));
      setFileContentCache((current) => {
        const next: Record<string, { content: string; updatedAt: string }> = {};
        for (const [path, value] of Object.entries(current)) {
          if (validPaths.has(path)) {
            next[path] = value;
          }
        }
        return next;
      });

      const currentSelectedPath = selectedFilePathRef.current;
      const selectedExists = nextFiles.some((file) => file.path === currentSelectedPath);
      const preferredPath = selectedExists
        ? currentSelectedPath
        : nextFiles.some((file) => file.path === previewEntryPath)
          ? previewEntryPath
          : nextFiles[0]?.path ?? previewEntryPath;

      if (selectedFilePathRef.current !== preferredPath) {
        selectedFilePathRef.current = preferredPath;
        setSelectedFilePath(preferredPath);
      }

      if (nextFiles.some((file) => file.path === preferredPath)) {
        const preferredMeta = nextFiles.find((file) => file.path === preferredPath);
        const preferredUpdatedAt = preferredMeta?.updated_at ?? '';
        const cached = fileContentCacheRef.current[preferredPath];

        if (cached && cached.updatedAt === preferredUpdatedAt) {
          if (requestId !== loadWorkspaceDataRequestIdRef.current || selectedFilePathRef.current !== preferredPath) {
            return;
          }
          setFileContent(cached.content);
        } else {
          const file = await api.getUserSpaceFile(workspaceId, preferredPath);

          if (requestId !== loadWorkspaceDataRequestIdRef.current || selectedFilePathRef.current !== preferredPath) {
            return;
          }

          setFileContent(file.content);
          setFileContentCache((current) => ({
            ...current,
            [file.path]: {
              content: file.content,
              updatedAt: preferredUpdatedAt,
            },
          }));
        }
      } else {
        setFileContent('');
      }

      setFileDirty(false);
      setError(null);
    } catch (err) {
      if (requestId !== loadWorkspaceDataRequestIdRef.current) {
        return;
      }
      setError(err instanceof Error ? err.message : 'Failed to load workspace data');
    }
  }, [previewEntryPath]);

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

  const loadSnapshots = useCallback(async (workspaceId: string) => {
    try {
      const result = await api.getUserSpaceSnapshotTimeline(workspaceId);
      setSnapshots(result.snapshots);
      setSnapshotBranches(result.branches);
      setCurrentSnapshotId(result.current_snapshot_id ?? null);
      setCurrentSnapshotBranchId(result.current_branch_id ?? null);
      setSnapshotsLoadedForWorkspace(workspaceId);
    } catch {
      // Snapshot list is non-critical; keep UI functional.
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

  const dismissSnapshotFileDiffOverlay = useCallback(() => {
    snapshotFileDiffPinnedRef.current = false;
    snapshotFileDiffEnteredOverlayRef.current = false;
    if (snapshotFileDiffHoverTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffHoverTimerRef.current);
      snapshotFileDiffHoverTimerRef.current = null;
    }
    if (snapshotFileDiffDismissTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffDismissTimerRef.current);
      snapshotFileDiffDismissTimerRef.current = null;
    }
    setActiveSnapshotFileDiffLoading(false);
    setActiveSnapshotFileDiffError(null);
    setActiveSnapshotFileDiff(null);
    setActiveSnapshotFileDiffKey(null);
  }, []);

  /** Wire up scroll sync between the two diff editors after CodeMirror mounts its scrollers. */
  useEffect(() => {
    if (!activeSnapshotFileDiff) return;

    let beforeScroller: HTMLElement | null = null;
    let afterScroller: HTMLElement | null = null;
    let rafId: number | null = null;
    let attempts = 0;
    let detachScrollListeners: (() => void) | null = null;

    const syncScroll = (src: HTMLElement, dst: HTMLElement) => {
      if (snapshotDiffScrollSyncingRef.current) return;
      snapshotDiffScrollSyncingRef.current = true;
      dst.scrollTop = src.scrollTop;
      dst.scrollLeft = src.scrollLeft;
      requestAnimationFrame(() => {
        snapshotDiffScrollSyncingRef.current = false;
      });
    };

    const attachScrollListeners = () => {
      const beforeWrap = snapshotDiffBeforeWrapRef.current;
      const afterWrap = snapshotDiffAfterWrapRef.current;
      if (!beforeWrap || !afterWrap) return false;

      beforeScroller = beforeWrap.querySelector<HTMLElement>('.cm-scroller');
      afterScroller = afterWrap.querySelector<HTMLElement>('.cm-scroller');
      if (!beforeScroller || !afterScroller) return false;

      const onBeforeScroll = () => syncScroll(beforeScroller!, afterScroller!);
      const onAfterScroll = () => syncScroll(afterScroller!, beforeScroller!);

      beforeScroller.addEventListener('scroll', onBeforeScroll, { passive: true });
      afterScroller.addEventListener('scroll', onAfterScroll, { passive: true });
      detachScrollListeners = () => {
        beforeScroller?.removeEventListener('scroll', onBeforeScroll);
        afterScroller?.removeEventListener('scroll', onAfterScroll);
      };
      return true;
    };

    const wireWhenReady = () => {
      if (attachScrollListeners()) return;
      attempts += 1;
      if (attempts >= 12) return;
      rafId = window.requestAnimationFrame(wireWhenReady);
    };

    wireWhenReady();

    return () => {
      if (rafId !== null) {
        window.cancelAnimationFrame(rafId);
      }
      detachScrollListeners?.();
      snapshotDiffScrollSyncingRef.current = false;
    };
  }, [activeSnapshotFileDiff, activeSnapshotFileDiffKey, snapshotAlignedDiff]);

  const scheduleSnapshotFileDiffDismiss = useCallback(() => {
    if (snapshotFileDiffPinnedRef.current) return;
    if (snapshotFileDiffDismissTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffDismissTimerRef.current);
    }
    snapshotFileDiffDismissTimerRef.current = window.setTimeout(() => {
      snapshotFileDiffDismissTimerRef.current = null;
      setActiveSnapshotFileDiffLoading(false);
      setActiveSnapshotFileDiffError(null);
      setActiveSnapshotFileDiff(null);
      setActiveSnapshotFileDiffKey(null);
    }, 500);
  }, []);

  const loadSnapshotFileDiff = useCallback(async (workspaceId: string, snapshotId: string, filePath: string) => {
    const cacheKey = getSnapshotDiffFileKey(snapshotId, filePath);
    if (snapshotFileDiffDismissTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffDismissTimerRef.current);
      snapshotFileDiffDismissTimerRef.current = null;
    }
    snapshotFileDiffEnteredOverlayRef.current = false;

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
  }, []);

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
    if (snapshotFileDiffDismissTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffDismissTimerRef.current);
      snapshotFileDiffDismissTimerRef.current = null;
    }
    if (snapshotFileDiffHoverTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffHoverTimerRef.current);
    }
    snapshotFileDiffHoverTimerRef.current = window.setTimeout(() => {
      snapshotFileDiffHoverTimerRef.current = null;
      void loadSnapshotFileDiff(activeWorkspaceId, snapshotId, filePath);
    }, 500);
  }, [activeWorkspaceId, loadSnapshotFileDiff]);

  const handleSnapshotFileHoverEnd = useCallback(() => {
    if (snapshotFileDiffHoverTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffHoverTimerRef.current);
      snapshotFileDiffHoverTimerRef.current = null;
    }
    if (snapshotFileDiffEnteredOverlayRef.current) {
      scheduleSnapshotFileDiffDismiss();
    }
  }, [scheduleSnapshotFileDiffDismiss]);

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
    dismissSnapshotFileDiffOverlay();
    setChangedFiles(new Set());
    setAcknowledgedFiles(new Set());
    void Promise.all([
      loadWorkspaceData(activeWorkspaceId),
      loadChangedFileState(activeWorkspaceId),
    ]);
  }, [activeWorkspaceId, dismissSnapshotFileDiffOverlay, loadChangedFileState, loadWorkspaceData]);

  useEffect(() => () => {
    if (snapshotFileDiffHoverTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffHoverTimerRef.current);
    }
    if (snapshotFileDiffDismissTimerRef.current !== null) {
      window.clearTimeout(snapshotFileDiffDismissTimerRef.current);
    }
  }, []);

  useEffect(() => {
    fileContentCacheRef.current = fileContentCache;
  }, [fileContentCache]);

  useEffect(() => {
    fileDirtyRef.current = fileDirty;
  }, [fileDirty]);

  useEffect(() => {
    fileBrowserEntriesRef.current = fileBrowserEntries;
  }, [fileBrowserEntries]);

  const refreshRuntimeStatus = useCallback(async () => {
    if (!activeWorkspaceId) {
      setRuntimeStatus(null);
      return;
    }
    if (refreshRuntimeStatusInflightRef.current) return;
    refreshRuntimeStatusInflightRef.current = true;
    const requestId = ++loadRuntimeStatusRequestIdRef.current;
    try {
      const status = await api.getUserSpaceRuntimeDevserverStatus(activeWorkspaceId);
      if (requestId === loadRuntimeStatusRequestIdRef.current) {
        setRuntimeStatus(status);
      }
    } catch {
      if (requestId === loadRuntimeStatusRequestIdRef.current) {
        setRuntimeStatus(null);
      }
    } finally {
      refreshRuntimeStatusInflightRef.current = false;
    }
  }, [activeWorkspaceId]);

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
            void refreshRuntimeStatus();
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
    refreshRuntimeStatus,
    workspaceEventsReconnectNonce,
  ]);

  useEffect(() => {
    selectedFilePathRef.current = selectedFilePath;
  }, [selectedFilePath]);

  useEffect(() => {
    if (!activeWorkspaceId) return;

    let cancelled = false;

    const syncWorkspaceFileCache = async () => {
      const staleFiles = files.filter((file) => {
        const cached = fileContentCacheRef.current[file.path];
        return !cached || cached.updatedAt !== (file.updated_at ?? '');
      });

      const validPaths = new Set(files.map((file) => file.path));

      if (staleFiles.length === 0) {
        setFileContentCache((current) => {
          const next: Record<string, { content: string; updatedAt: string }> = {};
          for (const [path, value] of Object.entries(current)) {
            if (validPaths.has(path)) {
              next[path] = value;
            }
          }
          return next;
        });
        return;
      }

      const fetched = await Promise.all(
        staleFiles.map(async (file) => {
          const loaded = await api.getUserSpaceFile(activeWorkspaceId, file.path);
          return {
            path: loaded.path,
            content: loaded.content,
            updatedAt: file.updated_at ?? '',
          };
        })
      );

      if (cancelled) return;

      setFileContentCache((current) => {
        const next: Record<string, { content: string; updatedAt: string }> = {};
        for (const [path, value] of Object.entries(current)) {
          if (validPaths.has(path)) {
            next[path] = value;
          }
        }

        for (const file of fetched) {
          next[file.path] = {
            content: file.content,
            updatedAt: file.updatedAt,
          };
        }

        return next;
      });
    };

    syncWorkspaceFileCache().catch((err) => {
      console.warn('Failed to sync userspace file cache', err);
    });

    return () => {
      cancelled = true;
    };
  }, [activeWorkspaceId, files]);

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
    setCreatingWorkspace(true);
    setCreatingWorkspaceStatus('Creating workspace...');
    setIsWorkspaceMenuOpen(false);
    setRuntimeStatus(null);
    setPreviewLiveDataConnections([]);
    setActiveWorkspaceId(null);
    setFileBrowserEntries([]);
    setFiles([]);
    setSnapshots([]);
    setSelectedFilePath(previewEntryPath);
    setFileContent('');
    setFileDirty(false);

    try {
      const created = await api.createUserSpaceWorkspace({
        selected_tool_ids: availableToolIds,
      });

      setWorkspaces((current) => (
        current.some((workspace) => workspace.id === created.id)
          ? current
          : [created, ...current]
      ));
      setActiveWorkspaceId(created.id);

      setCreatingWorkspaceStatus('Bootstrapping workspace files...');
      await api.upsertUserSpaceFile(created.id, 'dashboard/main.ts', {
        content: 'export function render(container: HTMLElement) {\n  container.innerHTML = `<h2>Interactive Report</h2><p>Ask chat to build your report and wire live data connections.</p>`;\n}\n',
        artifact_type: 'module_ts',
      });

      setCreatingWorkspaceStatus('Setting up workspace conversation...');
      await api.createConversation(undefined, created.id);

      setCreatingWorkspaceStatus('Loading workspace...');
      await loadWorkspaces();
      await loadWorkspaceData(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workspace');
    } finally {
      setCreatingWorkspaceStatus(null);
      setCreatingWorkspace(false);
    }
  }, [availableToolIds, loadWorkspaceData, loadWorkspaces, previewEntryPath]);

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
        return;
      }

      const file = await api.getUserSpaceFile(activeWorkspaceId, path);
      setFileContent(file.content);
      setFileContentCache((current) => ({
        ...current,
        [file.path]: {
          content: file.content,
          updatedAt: selectedUpdatedAt,
        },
      }));
      setFileDirty(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open file');
    }
  }, [activeWorkspaceId, files]);

  const handleCreateSnapshot = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setCreatingSnapshot(true);
    try {
      await api.createUserSpaceSnapshot(activeWorkspaceId, { message: 'Manual snapshot' });
      // Reset all per-file changed/acknowledged markers after snapshot baseline resets.
      setChangedFiles(new Set());
      setAcknowledgedFiles(new Set());
      setFileDirty(false);
      // Refresh snapshots list if panel is open.
      setSnapshotsLoadedForWorkspace(null);
      await loadChangedFileState(activeWorkspaceId);
      if (showSnapshots) {
        await loadSnapshots(activeWorkspaceId);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create snapshot');
    } finally {
      setCreatingSnapshot(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, loadChangedFileState, loadSnapshots, showSnapshots]);

  const handleSaveTreeFile = useCallback(async (filePath: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setSavingTreeFile(filePath);
    try {
      // Determine content: if the file is currently selected, use editor state;
      // otherwise fall back to the file content cache.
      const content = filePath === selectedFilePath
        ? fileContent
        : (fileContentCacheRef.current[filePath]?.content ?? '');
      await api.upsertUserSpaceFile(activeWorkspaceId, filePath, {
        content,
        artifact_type: 'module_ts',
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
  }, [activeWorkspaceId, canEditWorkspace, fileContent, selectedFilePath]);

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

  const handleStartRuntime = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRuntimeBusy(true);
    setRuntimeStatus((prev) => prev ? { ...prev, session_state: 'starting' } : prev);
    try {
      await api.startUserSpaceRuntimeSession(activeWorkspaceId);
      await refreshRuntimeStatus();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start runtime');
    } finally {
      setRuntimeBusy(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, refreshRuntimeStatus]);

  const handleStopRuntime = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRuntimeBusy(true);
    setRuntimeStatus((prev) => prev ? { ...prev, session_state: 'stopping' } : prev);
    try {
      await api.stopUserSpaceRuntimeSession(activeWorkspaceId);
      await refreshRuntimeStatus();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to stop runtime');
    } finally {
      setRuntimeBusy(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, refreshRuntimeStatus]);

  const handleRestartRuntime = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setRuntimeBusy(true);
    setRuntimeStatus((prev) => prev ? { ...prev, session_state: 'starting' } : prev);
    try {
      await api.restartUserSpaceRuntimeDevserver(activeWorkspaceId);
      await refreshRuntimeStatus();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restart runtime');
    } finally {
      setRuntimeBusy(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, refreshRuntimeStatus]);

  useEffect(() => {
    void refreshRuntimeStatus();
    if (!activeWorkspaceId) return;
    // Poll faster during transitional states (starting/stopping)
    const isTransitional = runtimeDisplayState === 'starting' || runtimeDisplayState === 'stopping';
    const interval = isTransitional ? 2000 : 10000;
    const timer = window.setInterval(() => {
      void refreshRuntimeStatus();
    }, interval);
    return () => window.clearInterval(timer);
  }, [activeWorkspaceId, refreshRuntimeStatus, runtimeDisplayState]);

  useEffect(() => {
    if (!activeWorkspaceId) {
      setPreviewCapabilityToken(null);
      return;
    }

    let cancelled = false;
    const loadPreviewCapability = async () => {
      try {
        const token = await issueWorkspaceCapabilityToken(activeWorkspaceId, ['userspace.preview_http', 'userspace.preview_ws']);
        if (!cancelled) {
          setPreviewCapabilityToken(token);
        }
      } catch (err) {
        if (!cancelled) {
          setPreviewCapabilityToken(null);
          setError(err instanceof Error ? err.message : 'Failed to authorize preview access');
        }
      }
    };

    void loadPreviewCapability();
    return () => {
      cancelled = true;
    };
  }, [activeWorkspaceId, issueWorkspaceCapabilityToken, previewRefreshCounter]);

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
        const token = await issueWorkspaceCapabilityToken(activeWorkspaceId, ['userspace.collab_connect']);
        if (!reconnectEnabled) {
          return;
        }

        const socketUrl = api.getUserSpaceCollabWebSocketUrl(activeWorkspaceId, selectedFilePath, token);
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
  }, [activeWorkspaceId, loadWorkspaceData, selectedFilePath, collabReconnectNonce, issueWorkspaceCapabilityToken]);

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

    // Allow terminal retries while runtime is starting to avoid a hard UI stall.
    if (runtimeDisplayState !== 'running' && runtimeDisplayState !== 'starting') {
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

    terminalRef.current = terminal;
    terminalFitRef.current = fitAddon;

    const resizeObserver = new ResizeObserver(() => {
      try {
        fitAddon.fit();
      } catch {
        // ignore sizing errors during rapid layout changes
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
        const token = await issueWorkspaceCapabilityToken(activeWorkspaceId, ['userspace.runtime_pty']);
        if (!reconnectEnabled) {
          return;
        }

        const wsUrl = api.getUserSpaceRuntimePtyWebSocketUrl(activeWorkspaceId, token);
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
        };

        socket.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data) as { type?: string; data?: string; read_only?: boolean; message?: string };
            if (payload.type === 'status') {
              const isReadOnly = Boolean(payload.read_only);
              terminalReadOnlyRef.current = isReadOnly;
              setTerminalReadOnly(isReadOnly);
              terminal.options.disableStdin = isReadOnly;
              const statusMessage = typeof payload.message === 'string' ? payload.message.trim() : '';
              if (statusMessage) {
                terminal.writeln(`\r\n[status] ${statusMessage}\r\n`);
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

        socket.onclose = () => {
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
      terminalFitRef.current = null;
      terminalRef.current?.dispose();
      terminalRef.current = null;
    };
  }, [activeRightTab, activeWorkspaceId, canEditWorkspace, runtimeDisplayState, terminalReconnectNonce, issueWorkspaceCapabilityToken]);

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
    setDeletingWorkspaceId(workspaceId);
    setDeletingWorkspaceStatus('Deleting workspace...');

    const deletingActiveWorkspace = activeWorkspaceId === workspaceId;
    const fallbackWorkspaceId = deletingActiveWorkspace
      ? workspaces.find((workspace) => workspace.id !== workspaceId)?.id ?? null
      : activeWorkspaceId;

    setDeleteConfirmWorkspaceId(null);
    setIsWorkspaceMenuOpen(false);
    setWorkspaces((current) => current.filter((workspace) => workspace.id !== workspaceId));
    setWorkspacesTotal((current) => Math.max(0, current - 1));

    if (deletingActiveWorkspace) {
      setRuntimeStatus(null);
      setPreviewLiveDataConnections([]);
      setActiveWorkspaceId(fallbackWorkspaceId);

      if (!fallbackWorkspaceId) {
        clearCookieValue(lastWorkspaceCookieName);
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
    }

    try {
      await api.deleteUserSpaceWorkspace(workspaceId);

      setDeletingWorkspaceStatus('Refreshing workspace list...');
      await loadWorkspaces();
    } catch (err) {
      try {
        await loadWorkspaces();
      } catch {
        // Best-effort refresh; continue with delete verification below.
      }

      try {
        await api.getUserSpaceWorkspace(workspaceId);
        setError(err instanceof Error ? err.message : 'Failed to delete workspace');
      } catch {
        setError(null);
      }
    } finally {
      setDeletingWorkspaceStatus(null);
      setDeletingWorkspaceId(null);
    }
  }, [activeWorkspaceId, lastWorkspaceCookieName, loadWorkspaces, workspaces]);

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

  const handleOpenEnvVarsModal = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner) return;
    setShowEnvVarsModal(true);
    setEnvVarsLoading(true);
    setDraftEnvKey('');
    setDraftEnvValue('');
    setDraftEnvDescription('');
    setEditingEnvKey(null);
    setEditingEnvValueDraft('');
    setEditingEnvDescKey(null);
    setEditingEnvDescriptionDraft('');
    setConfirmDeleteEnvKey(null);
    try {
      const vars = await api.listUserSpaceWorkspaceEnvVars(activeWorkspaceId);
      setEnvVars(vars);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load environment variables');
    } finally {
      setEnvVarsLoading(false);
    }
  }, [activeWorkspaceId, isOwner]);

  const handleSaveEnvVar = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner) return;
    const key = draftEnvKey.trim().toUpperCase();
    const value = draftEnvValue;
    const description = draftEnvDescription.trim();
    if (!key) return;
    setSavingEnvVar(true);
    try {
      const upserted = await api.upsertUserSpaceWorkspaceEnvVar(activeWorkspaceId, {
        key,
        value: value || undefined,
        description: description || undefined,
      });
      setEnvVars((current) => {
        const next = current.filter((v) => v.key !== key);
        return [...next, upserted].sort((a, b) => a.key.localeCompare(b.key));
      });
      setDraftEnvKey('');
      setDraftEnvValue('');
      setDraftEnvDescription('');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save environment variable');
    } finally {
      setSavingEnvVar(false);
    }
  }, [activeWorkspaceId, draftEnvDescription, draftEnvKey, draftEnvValue, isOwner]);

  const handleSaveEnvValue = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner || !editingEnvKey) return;
    setSavingEnvVar(true);
    try {
      const upserted = await api.upsertUserSpaceWorkspaceEnvVar(activeWorkspaceId, {
        key: editingEnvKey,
        value: editingEnvValueDraft || undefined,
      });
      setEnvVars((current) => {
        const next = current.filter((v) => v.key !== editingEnvKey);
        return [...next, upserted].sort((a, b) => a.key.localeCompare(b.key));
      });
      setEditingEnvKey(null);
      setEditingEnvValueDraft('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save environment variable value');
    } finally {
      setSavingEnvVar(false);
    }
  }, [activeWorkspaceId, editingEnvKey, editingEnvValueDraft, isOwner]);

  const handleSaveEnvDesc = useCallback(async () => {
    if (!activeWorkspaceId || !isOwner || !editingEnvDescKey) return;
    setSavingEnvVar(true);
    try {
      const upserted = await api.upsertUserSpaceWorkspaceEnvVar(activeWorkspaceId, {
        key: editingEnvDescKey,
        description: editingEnvDescriptionDraft.trim() || undefined,
      });
      setEnvVars((current) => {
        const next = current.filter((v) => v.key !== editingEnvDescKey);
        return [...next, upserted].sort((a, b) => a.key.localeCompare(b.key));
      });
      setEditingEnvDescKey(null);
      setEditingEnvDescriptionDraft('');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save description');
    } finally {
      setSavingEnvVar(false);
    }
  }, [activeWorkspaceId, editingEnvDescKey, editingEnvDescriptionDraft, isOwner]);

  const handleDeleteEnvVar = useCallback(async (key: string) => {
    if (!activeWorkspaceId || !isOwner) return;
    setDeletingEnvKey(key);
    try {
      await api.deleteUserSpaceWorkspaceEnvVar(activeWorkspaceId, key);
      setEnvVars((current) => current.filter((v) => v.key !== key));
      setConfirmDeleteEnvKey(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete environment variable');
    } finally {
      setDeletingEnvKey(null);
    }
  }, [activeWorkspaceId, isOwner]);

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

  const handleOpenShareModal = useCallback(() => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setShareSlugDraft(shareLinkStatus?.share_slug ?? getDefaultShareSlug(activeWorkspace?.name));
    setShareSlugAvailable(null);
    setShareAccessMode(shareLinkStatus?.share_access_mode ?? 'token');
    setShareSelectedUserIdsDraft(shareLinkStatus?.selected_user_ids ?? []);
    setShareSelectedLdapGroupsDraft(shareLinkStatus?.selected_ldap_groups ?? []);
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
    canEditWorkspace,
    shareLinkStatus?.share_access_mode,
    shareLinkStatus?.selected_ldap_groups,
    shareLinkStatus?.selected_user_ids,
    shareLinkStatus?.share_slug,
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
        created_at: new Date().toISOString(),
        share_access_mode: shareLinkStatus?.share_access_mode ?? 'token',
        selected_user_ids: shareLinkStatus?.selected_user_ids ?? [],
        selected_ldap_groups: shareLinkStatus?.selected_ldap_groups ?? [],
        has_password: shareLinkStatus?.has_password ?? false,
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
    canEditWorkspace,
    shareLinkStatus?.has_password,
    shareLinkStatus?.selected_ldap_groups,
    shareLinkStatus?.selected_user_ids,
    shareLinkStatus?.share_access_mode,
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
    shareLinkStatus,
    handleEnsureShareLink,
  ]);

  const resolveShareUrl = useCallback((status: UserSpaceWorkspaceShareLinkStatus | null | undefined): string | null => {
    if (!status?.has_share_link) return null;
    if (shareLinkType === 'anonymous' && status.share_token) {
      const base = status.share_url
        ? new URL(status.share_url, window.location.origin).origin
        : window.location.origin;
      return `${base}/shared/${encodeURIComponent(status.share_token)}`;
    }
    return status.share_url ?? null;
  }, [shareLinkType]);

  const effectiveShareUrl = useMemo(() => resolveShareUrl(shareLinkStatus), [resolveShareUrl, shareLinkStatus]);

  const handleCopyShareLink = useCallback(async () => {
    let url = effectiveShareUrl;
    if (!shareLinkStatus?.has_share_link) {
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
  }, [effectiveShareUrl, handleEnsureShareLink, resolveShareUrl, shareLinkStatus?.has_share_link]);

  const handleOpenFullPreview = useCallback(async () => {
    let url = effectiveShareUrl;
    if (!shareLinkStatus?.has_share_link) {
      const created = await handleEnsureShareLink(false);
      url = resolveShareUrl(created);
    }
    if (!url) return;
    window.open(url, '_blank', 'noopener,noreferrer');
  }, [effectiveShareUrl, handleEnsureShareLink, resolveShareUrl, shareLinkStatus?.has_share_link]);

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

    let currentStatus = shareLinkStatus;
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
    canEditWorkspace,
    handleEnsureShareLink,
    shareAccessMode,
    shareLinkStatus,
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
    if (!showShareModal || !shareLinkStatus) {
      return false;
    }

    const slugChanged = normalizeShareSlugInput(shareSlugDraft) !== normalizeShareSlugInput(shareLinkStatus.share_slug ?? '');
    const accessModeChanged = shareAccessMode !== shareLinkStatus.share_access_mode;
    const selectedUsersChanged = !areSameNormalizedStringArrays(
      shareSelectedUserIdsDraft,
      shareLinkStatus.selected_user_ids ?? [],
    );
    const selectedLdapGroupsChanged = !areSameNormalizedStringArrays(
      shareSelectedLdapGroupsDraft,
      shareLinkStatus.selected_ldap_groups ?? [],
    );
    const pendingPasswordChange = shareAccessMode === 'password' && Boolean(sharePasswordDraft.trim());

    return slugChanged
      || accessModeChanged
      || selectedUsersChanged
      || selectedLdapGroupsChanged
      || pendingPasswordChange;
  }, [
    shareAccessMode,
    shareLinkStatus,
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
          const hasChangedFileDescendant = !isExpanded && collectFilePaths(node).some((p) => changedFilePaths.has(p));
          rows.push(
            <div key={node.path} className="userspace-file-item userspace-tree-row userspace-tree-folder-row">
              <button className="userspace-item-content userspace-tree-content" onClick={() => handleToggleFolder(node.path)} style={indentStyle}>
                <span className="userspace-tree-chevron" aria-hidden="true">
                  {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </span>
                <span className="userspace-folder-label">{node.name}</span>
                {hasChangedFileDescendant && <span className="userspace-tree-folder-changed-file-dot" title="Contains changed files" />}
              </button>
              {canEditWorkspace && (
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
          <button className="userspace-item-content userspace-tree-content" onClick={() => handleSelectFile(node.path)} style={indentStyle}>
            <span className="userspace-tree-file-label">{node.name}</span>
            {isFileChanged && <span className="userspace-tree-file-changed-dot" title="Changed since last snapshot" />}
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
  }, [canEditWorkspace, changedFilePaths, deleteConfirmFileId, deleteConfirmFolderPath, expandedFolders, handleDeleteFile, handleDeleteFolder, handleRenameFile, handleRenameFolder, handleSaveTreeFile, handleSelectFile, handleStartCreateFile, handleToggleFolder, renameValue, renamingFilePath, renamingFolderPath, savingTreeFile, selectedFilePath]);

  const sqliteLiveDataOnlyMode = activeWorkspace?.sqlite_persistence_mode === 'exclude';
  const sqlitePersistenceModeTitle = sqliteLiveDataOnlyMode
    ? 'Live data only mode. SQLite local files are excluded from snapshots. Click to enable two-lane persistence (live data + SQLite local state with migrations).'
    : 'Two-lane persistence mode. Live data wiring is primary for dashboards; SQLite local state is persisted with snapshots. Click to switch to live data only mode.';
  const formattedError = useMemo(() => formatUserSpaceErrorMessage(error), [error]);
  const hasStatusOverlayContent = Boolean(
    loading || creatingWorkspace || deletingWorkspaceId || (formattedError && !creatingWorkspace && !deletingWorkspaceId)
  );
  const statusOverlaySignature = useMemo(() => JSON.stringify({
    loading,
    creatingWorkspace,
    creatingWorkspaceStatus,
    deletingWorkspaceId,
    deletingWorkspaceStatus,
    formattedError: formattedError && !creatingWorkspace && !deletingWorkspaceId ? formattedError : null,
  }), [
    loading,
    creatingWorkspace,
    creatingWorkspaceStatus,
    deletingWorkspaceId,
    deletingWorkspaceStatus,
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
    if (!hasStatusOverlayContent || !statusOverlayVisible || statusOverlayPinned || statusOverlayInteracting) {
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
                    const canDeleteWorkspace = ws.owner_user_id === currentUser.id;
                    const canRenameWorkspace = currentUser.role === 'admin'
                      || ws.owner_user_id === currentUser.id
                      || ws.members.some((member) => (
                        member.user_id === currentUser.id && (member.role === 'owner' || member.role === 'editor')
                      ));
                    const isConfirmingDelete = deleteConfirmWorkspaceId === ws.id;
                    const isDeletingWorkspace = deletingWorkspaceId === ws.id;
                    const isRenamingWorkspace = editingWorkspaceNameId === ws.id;
                    return (
                      <div
                        key={ws.id}
                        className={`model-selector-item userspace-workspace-item ${ws.id === activeWorkspaceId ? 'is-selected' : ''} ${!canDeleteWorkspace ? 'is-shared' : ''}`}
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
                            disabled={Boolean(deletingWorkspaceId)}
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
                                  disabled={Boolean(deletingWorkspaceId)}
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
                                  disabled={Boolean(deletingWorkspaceId)}
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
                                  disabled={Boolean(deletingWorkspaceId)}
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
                                  disabled={Boolean(deletingWorkspaceId)}
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
                                    disabled={Boolean(deletingWorkspaceId)}
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
                                    disabled={Boolean(deletingWorkspaceId)}
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
            disabled={creatingWorkspace}
            title={creatingWorkspace ? (creatingWorkspaceStatus || 'Bootstrapping workspace...') : 'New workspace'}
          >
            {creatingWorkspace ? <MiniLoadingSpinner variant="icon" size={14} /> : <Plus size={14} />}
          </button>
          {isOwner && (
            <>
              <button className="btn btn-secondary btn-sm" onClick={handleOpenMembersModal} title="Manage members">
                <Users size={14} />
              </button>
              <button className="btn btn-secondary btn-sm" onClick={handleOpenEnvVarsModal} title="Environment variables">
                <KeyRound size={14} />
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
            {activeWorkspace && !isAdminImpersonating && (
              <span className="userspace-status-pill userspace-status-pill-info">
                {activeWorkspaceRole}{!canEditWorkspace ? ' (read-only)' : ''}
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
            {activeWorkspaceId && (
              <span
                className={`userspace-status-pill ${collabConnected ? 'userspace-status-pill-success' : 'userspace-status-pill-muted'}`}
                title="Collaborative editor connection state"
              >
                {collabConnected ? `collab (${collabPresenceCount})` : 'offline'}
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
              {shareCopied ? <Check size={14} /> : <Link2 size={14} />}
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
          tabIndex={0}
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
          {deletingWorkspaceId && (
            <p className="userspace-status userspace-status-overlay-item">
              <MiniLoadingSpinner variant="icon" size={14} /> {deletingWorkspaceStatus || 'Deleting workspace...'}
            </p>
          )}
          {formattedError && !creatingWorkspace && !deletingWorkspaceId && (
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
                workspaceAvailableTools={availableTools}
                workspaceSelectedToolIds={resolvedSelectedToolIds}
                workspaceSelectedToolGroupIds={resolvedSelectedToolGroupIds}
                onToggleWorkspaceTool={handleToggleWorkspaceTool}
                onToggleWorkspaceToolGroup={handleToggleWorkspaceToolGroup}
                workspaceToolGroups={toolGroups}
                workspaceSavingTools={savingWorkspaceTools}
                onUserMessageSubmitted={canEditWorkspace ? handleUserMessageSubmitted : undefined}
                onConversationStateChange={handleConversationStateChange}
                embedded
                readOnly={!canEditWorkspace}
                readOnlyMessage="Workspace is read-only for viewers. You can review chat and files, but only owners/editors can send prompts."
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
                runtimePreviewUrl={
                  activeWorkspaceId && previewCapabilityToken
                    ? api.getUserSpaceRuntimePreviewUrl(activeWorkspaceId, '', previewCapabilityToken)
                    : undefined
                }
                runtimeAvailable={runtimeStatus?.devserver_running ?? false}
                runtimeError={runtimeStatus?.last_error ?? undefined}
                previewInstanceKey={`${activeWorkspaceId ?? ''}:${previewRefreshCounter}`}
                workspaceId={activeWorkspaceId ?? undefined}
                onExecutionStateChange={setPreviewExecuting}
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
                          </button>
                        );
                      })}
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
                ) : (
                  <p className="userspace-muted" style={{ padding: '8px' }}>No snapshots yet</p>
                )}
              </div>
            )}
          </div>
        </div>
        )}
      </div>

      {(activeSnapshotFileDiffLoading || activeSnapshotFileDiffError || activeSnapshotFileDiff) && (
        <div
          className="userspace-snapshot-diff-backdrop"
          onClick={dismissSnapshotFileDiffOverlay}
        >
          <div
            className="userspace-snapshot-diff-overlay"
            onClick={(event) => {
              event.stopPropagation();
              snapshotFileDiffPinnedRef.current = true;
            }}
            onMouseEnter={() => {
              snapshotFileDiffEnteredOverlayRef.current = true;
              if (snapshotFileDiffDismissTimerRef.current !== null) {
                window.clearTimeout(snapshotFileDiffDismissTimerRef.current);
                snapshotFileDiffDismissTimerRef.current = null;
              }
            }}
            onMouseLeave={scheduleSnapshotFileDiffDismiss}
          >
            <div className="userspace-snapshot-diff-overlay-header">
              <div>
                <div className="userspace-snapshot-diff-overlay-title">Snapshot Diff</div>
                {activeSnapshotFileDiff && (
                  <div className="userspace-snapshot-diff-overlay-subtitle">
                    <span>{activeSnapshotFileDiff.path}</span>
                    <span>{formatSnapshotDiffStatus(activeSnapshotFileDiff.status)}</span>
                    <span>+{activeSnapshotFileDiff.additions} -{activeSnapshotFileDiff.deletions}</span>
                  </div>
                )}
              </div>
              <button type="button" className="modal-close" onClick={dismissSnapshotFileDiffOverlay}>&times;</button>
            </div>

            {activeSnapshotFileDiffLoading ? (
              <div className="userspace-snapshot-diff-overlay-body">
                <div className="userspace-snapshot-expanded-status userspace-snapshot-diff-overlay-loading">
                  <MiniLoadingSpinner variant="icon" size={14} />
                  <span>Loading file diff...</span>
                </div>
              </div>
            ) : activeSnapshotFileDiffError ? (
              <div className="userspace-snapshot-diff-overlay-body">
                <p className="userspace-muted userspace-error">{formatUserSpaceErrorMessage(activeSnapshotFileDiffError)}</p>
              </div>
            ) : activeSnapshotFileDiff ? (
              activeSnapshotFileDiff.is_binary || activeSnapshotFileDiff.is_truncated ? (
                <div className="userspace-snapshot-diff-overlay-body">
                  <p className="userspace-muted">{activeSnapshotFileDiff.message ?? 'Content cannot be rendered.'}</p>
                </div>
              ) : (
                <div className="userspace-snapshot-diff-columns">
                  <div className="userspace-snapshot-diff-column">
                    <div className="userspace-snapshot-diff-column-header">
                      <span>{activeSnapshotFileDiff.is_snapshot_own_diff ? 'Previous' : 'Snapshot'}</span>
                      <code>{activeSnapshotFileDiff.before_path ?? activeSnapshotFileDiff.path}</code>
                    </div>
                    <div className="userspace-snapshot-diff-editor-wrap" ref={snapshotDiffBeforeWrapRef}>
                      <CodeMirror
                        value={snapshotAlignedDiff?.beforeText ?? activeSnapshotFileDiff.before_content}
                        basicSetup={snapshotDiffCodeMirrorSetup}
                        editable={false}
                        extensions={snapshotDiffBeforeExtensions}
                        height="100%"
                      />
                    </div>
                  </div>
                  <div className="userspace-snapshot-diff-column userspace-snapshot-diff-column-current">
                    <div className="userspace-snapshot-diff-column-header">
                      <span>{activeSnapshotFileDiff.is_snapshot_own_diff ? 'Snapshot' : 'Current Workspace'}</span>
                      <code>{activeSnapshotFileDiff.after_path ?? activeSnapshotFileDiff.path}</code>
                    </div>
                    <div className="userspace-snapshot-diff-editor-wrap" ref={snapshotDiffAfterWrapRef}>
                      <CodeMirror
                        value={snapshotAlignedDiff?.afterText ?? activeSnapshotFileDiff.after_content}
                        basicSetup={snapshotDiffCodeMirrorSetup}
                        editable={false}
                        extensions={snapshotDiffAfterExtensions}
                        height="100%"
                      />
                    </div>
                  </div>
                </div>
              )
            ) : null}
          </div>
        </div>
      )}

      {/* === Env vars modal === */}
      {showEnvVarsModal && activeWorkspaceId && (
        <div className="modal-overlay" onClick={() => setShowEnvVarsModal(false)}>
          <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Environment Variables</h3>
              <button className="modal-close" onClick={() => setShowEnvVarsModal(false)}>&times;</button>
            </div>
            <div className="modal-body">
              <p className="userspace-muted" style={{ marginBottom: 12 }}>
                Variables are encrypted at rest and injected into the devserver at runtime startup.
                Reference them as <code>process.env.KEY</code> (Node.js) or <code>os.environ[&quot;KEY&quot;]</code> (Python).
              </p>
              {envVarsLoading ? (
                <p className="userspace-muted">Loading...</p>
              ) : (
                <>
                  {envVars.length > 0 && (
                    <div className="userspace-env-var-list">
                      {envVars.map((envVar) => (
                        <div key={envVar.key} className="userspace-env-var-row">
                          {/* Primary row: key, value (or value input), actions */}
                          <div className="userspace-env-var-primary">
                            <span className="userspace-env-var-key">
                              {envVar.key}
                              <button
                                className="userspace-env-var-copy-btn"
                                title="Copy key"
                                onClick={async () => {
                                  await navigator.clipboard.writeText(envVar.key);
                                  setCopiedEnvKey(envVar.key);
                                  setTimeout(() => setCopiedEnvKey((c) => c === envVar.key ? null : c), 1500);
                                }}
                              >
                                {copiedEnvKey === envVar.key ? <Check size={13} /> : <Copy size={13} />}
                              </button>
                            </span>
                            {editingEnvKey === envVar.key ? (
                              <>
                                <input
                                  type="password"
                                  className="form-input userspace-env-var-value-input"
                                  placeholder="New value (leave blank to keep current)"
                                  value={editingEnvValueDraft}
                                  onChange={(e) => setEditingEnvValueDraft(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') handleSaveEnvValue();
                                    if (e.key === 'Escape') {
                                      setEditingEnvKey(null);
                                      setEditingEnvValueDraft('');
                                    }
                                  }}
                                  autoFocus
                                />
                                <div className="userspace-env-var-actions">
                                  <button
                                    className="btn btn-primary btn-sm"
                                    onClick={handleSaveEnvValue}
                                    disabled={savingEnvVar}
                                    title="Save"
                                  >
                                    {savingEnvVar && editingEnvKey === envVar.key ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                  </button>
                                  <button
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => {
                                      setEditingEnvKey(null);
                                      setEditingEnvValueDraft('');
                                    }}
                                    title="Cancel"
                                  >
                                    <X size={12} />
                                  </button>
                                </div>
                              </>
                            ) : (
                              <>
                                <span className="userspace-env-var-value">
                                  {envVar.has_value ? '••••••' : <em>not set</em>}
                                </span>
                                <div className="userspace-env-var-actions">
                                  {confirmDeleteEnvKey === envVar.key ? (
                                    <>
                                      <button
                                        className="btn btn-danger btn-sm"
                                        onClick={() => handleDeleteEnvVar(envVar.key)}
                                        disabled={deletingEnvKey === envVar.key}
                                        title="Confirm delete"
                                      >
                                        {deletingEnvKey === envVar.key ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                      </button>
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => setConfirmDeleteEnvKey(null)}
                                        title="Cancel"
                                      >
                                        <X size={12} />
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => {
                                          setEditingEnvKey(envVar.key);
                                          setEditingEnvValueDraft('');
                                        }}
                                        title="Edit value"
                                      >
                                        <Pencil size={12} />
                                      </button>
                                      <button
                                        className="btn btn-secondary btn-sm"
                                        onClick={() => setConfirmDeleteEnvKey(envVar.key)}
                                        title="Delete"
                                      >
                                        <Trash2 size={12} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              </>
                            )}
                          </div>
                          {/* Description sub-row with hover pencil */}
                          <div className="userspace-env-var-desc-row">
                            {editingEnvDescKey === envVar.key ? (
                              <div className="userspace-env-var-desc-edit">
                                <input
                                  type="text"
                                  className="form-input userspace-env-var-desc-input"
                                  placeholder="Description (optional)"
                                  value={editingEnvDescriptionDraft}
                                  onChange={(e) => setEditingEnvDescriptionDraft(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') handleSaveEnvDesc();
                                    if (e.key === 'Escape') {
                                      setEditingEnvDescKey(null);
                                      setEditingEnvDescriptionDraft('');
                                    }
                                  }}
                                  autoFocus
                                />
                                <button
                                  className="btn btn-primary btn-sm"
                                  onClick={handleSaveEnvDesc}
                                  disabled={savingEnvVar}
                                  title="Save description"
                                >
                                  {savingEnvVar && editingEnvDescKey === envVar.key ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                </button>
                                <button
                                  className="btn btn-secondary btn-sm"
                                  onClick={() => {
                                    setEditingEnvDescKey(null);
                                    setEditingEnvDescriptionDraft('');
                                  }}
                                  title="Cancel"
                                >
                                  <X size={12} />
                                </button>
                              </div>
                            ) : (
                              <div
                                className="userspace-env-var-desc-display"
                                onClick={() => {
                                  setEditingEnvDescKey(envVar.key);
                                  setEditingEnvDescriptionDraft(envVar.description ?? '');
                                }}
                              >
                                <span className="userspace-env-var-desc">
                                  {envVar.description || <em>No description</em>}
                                </span>
                                <button
                                  className="inline-edit-btn userspace-env-var-desc-edit-btn"
                                  title="Edit description"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setEditingEnvDescKey(envVar.key);
                                    setEditingEnvDescriptionDraft(envVar.description ?? '');
                                  }}
                                >
                                  <Pencil size={11} />
                                </button>
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="userspace-env-var-form">
                    <strong className="userspace-env-var-form-title">Add variable</strong>
                    <div className="userspace-env-var-primary-row">
                      <input
                        type="text"
                        className="form-input userspace-env-var-key-input"
                        placeholder="KEY_NAME"
                        value={draftEnvKey}
                        onChange={(e) => setDraftEnvKey(e.target.value.toUpperCase())}
                        onBlur={() => setDraftEnvKey((k) => k.replace(/[^A-Z0-9_]/g, '_').replace(/^[0-9]/, '_$&').replace(/_+/g, '_'))}
                      />
                      <input
                        className="form-input"
                        placeholder="Value (optional, leave blank for placeholder)"
                        type="password"
                        value={draftEnvValue}
                        onChange={(e) => setDraftEnvValue(e.target.value)}
                      />
                    </div>
                    <input
                      type="text"
                      className="form-input"
                      placeholder="Description (optional)"
                      value={draftEnvDescription}
                      onChange={(e) => setDraftEnvDescription(e.target.value)}
                    />
                    <div className="userspace-env-var-form-actions">
                      <button
                        className="btn btn-primary btn-sm"
                        onClick={handleSaveEnvVar}
                        disabled={savingEnvVar || !draftEnvKey.trim()}
                      >
                        {savingEnvVar && !editingEnvKey ? <MiniLoadingSpinner variant="icon" size={14} /> : <Check size={14} />}
                        Add
                      </button>
                    </div>
                  </div>
                </>
              )}
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

                    {shareLinkStatus?.has_share_link && (
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
                      </div>
                    )}

                    {shareLinkStatus?.has_share_link && effectiveShareUrl ? (
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
                          {shareLinkStatus.created_at ? `Created ${formatSnapshotTimestamp(shareLinkStatus.created_at)}` : 'Share link active'}
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
                          Share password {shareLinkStatus?.has_password ? '(set)' : '(required)'}
                        </label>
                        <input
                          id="userspace-share-password"
                          type="password"
                          value={sharePasswordDraft}
                          onChange={(event) => setSharePasswordDraft(event.target.value)}
                          placeholder={shareLinkStatus?.has_password ? 'Enter new password to update' : 'Enter password'}
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
                        disabled={revokingShareLink || sharingWorkspace || checkingShareSlug || savingShareAccess || !shareLinkStatus?.has_share_link}
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
