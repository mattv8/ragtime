import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, ChevronRight, ExternalLink, File, History, Link2, Maximize2, Minimize2, Pencil, Play, Plus, RotateCw, Save, Square, Terminal, Trash2, Users, X } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { Terminal as XTerm } from '@xterm/xterm';
import { FitAddon } from '@xterm/addon-fit';
import '@xterm/xterm/css/xterm.css';

import { api } from '@/api';
import { MemberManagementModal, type Member } from './shared/MemberManagementModal';
import { ToolSelectorDropdown } from './shared/ToolSelectorDropdown';
import type { User, UserSpaceAvailableTool, UserSpaceCollabMessage, UserSpaceFileInfo, UserSpaceLiveDataConnection, UserSpaceRuntimeStatusResponse, UserSpaceShareAccessMode, UserSpaceSnapshot, UserSpaceWorkspace, UserSpaceWorkspaceMember, UserSpaceWorkspaceShareLinkStatus } from '@/types';
import { buildUserSpaceTree, getAncestorFolderPaths, listFolderPaths } from '@/utils/userspaceTree';
import { ChatPanel } from './ChatPanel';
import { LdapGroupSelect } from './LdapGroupSelect';
import { ResizeHandle } from './ResizeHandle';
import { UserSpaceArtifactPreview } from './UserSpaceArtifactPreview';

interface UserSpacePanelProps {
  currentUser: User;
  onFullscreenChange?: (fullscreen: boolean) => void;
}

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

const LAST_WORKSPACE_COOKIE_PREFIX = 'userspace_last_workspace_id_';
const LAST_WORKSPACE_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;
const USERSPACE_LAYOUT_COOKIE_PREFIX = 'userspace_layout_';

function getLastWorkspaceCookieName(userId: string): string {
  return `${LAST_WORKSPACE_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

function getUserSpaceLayoutCookieName(userId: string, workspaceId: string | null): string {
  const scope = workspaceId ? encodeURIComponent(workspaceId) : 'global';
  return `${USERSPACE_LAYOUT_COOKIE_PREFIX}${encodeURIComponent(userId)}_${scope}`;
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

export function UserSpacePanel({ currentUser, onFullscreenChange }: UserSpacePanelProps) {
  const previewEntryPath = 'dashboard/main.ts';
  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [workspacesTotal, setWorkspacesTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [activeWorkspaceId, setActiveWorkspaceId] = useState<string | null>(null);
  const [files, setFiles] = useState<UserSpaceFileInfo[]>([]);
  const [snapshots, setSnapshots] = useState<UserSpaceSnapshot[]>([]);
  const [availableTools, setAvailableTools] = useState<UserSpaceAvailableTool[]>([]);

  const [selectedFilePath, setSelectedFilePath] = useState<string>('dashboard/main.ts');
  const [fileContent, setFileContent] = useState<string>('');
  const [fileDirty, setFileDirty] = useState(false);
  const [fileContentCache, setFileContentCache] = useState<Record<string, { content: string; updatedAt: string }>>({});
  const [previewLiveDataConnections, setPreviewLiveDataConnections] = useState<UserSpaceLiveDataConnection[]>([]);
  const [previewExecuting, setPreviewExecuting] = useState(false);
  const [previewRefreshCounter, setPreviewRefreshCounter] = useState(0);
  const [runtimeStatus, setRuntimeStatus] = useState<UserSpaceRuntimeStatusResponse | null>(null);
  const [runtimeBusy, setRuntimeBusy] = useState(false);
  const [activeRightTab, setActiveRightTab] = useState<'preview' | 'console'>('preview');
  const [collabConnected, setCollabConnected] = useState(false);
  const [collabReadOnly, setCollabReadOnly] = useState(false);
  const [collabVersion, setCollabVersion] = useState(0);
  const [collabPresenceCount, setCollabPresenceCount] = useState(0);
  const [collabReconnectNonce, setCollabReconnectNonce] = useState(0);
  const [terminalReadOnly, setTerminalReadOnly] = useState(false);
  const [terminalReconnectNonce, setTerminalReconnectNonce] = useState(0);
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [sharingWorkspace, setSharingWorkspace] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
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
  const [savingFile, setSavingFile] = useState(false);
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
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [pendingMembers, setPendingMembers] = useState<UserSpaceWorkspaceMember[]>([]);
  const [savingMembers, setSavingMembers] = useState(false);
  const [showToolPicker, setShowToolPicker] = useState(false);
  const [showSnapshots, setShowSnapshots] = useState(false);
  const toolPickerRef = useRef<HTMLDivElement>(null);
  const workspaceDropdownRef = useRef<HTMLDivElement>(null);
  const selectedFilePathRef = useRef(selectedFilePath);
  const fileContentCacheRef = useRef(fileContentCache);
  const loadWorkspaceDataRequestIdRef = useRef(0);
  const fileDirtyRef = useRef(false);
  const collabSocketRef = useRef<WebSocket | null>(null);
  const collabReconnectTimerRef = useRef<number | null>(null);
  const collabSuppressNextSendRef = useRef(false);
  const terminalSocketRef = useRef<WebSocket | null>(null);
  const terminalReconnectTimerRef = useRef<number | null>(null);
  const terminalReadOnlyRef = useRef(false);
  const terminalContainerRef = useRef<HTMLDivElement | null>(null);
  const terminalRef = useRef<XTerm | null>(null);
  const terminalFitRef = useRef<FitAddon | null>(null);
  const terminalResizeObserverRef = useRef<ResizeObserver | null>(null);
  const lastWorkspaceCookieName = useMemo(() => getLastWorkspaceCookieName(currentUser.id), [currentUser.id]);
  const userSpaceLayoutCookieName = useMemo(() => getUserSpaceLayoutCookieName(currentUser.id, activeWorkspaceId), [currentUser.id, activeWorkspaceId]);

  // Resize state
  const [sidebarWidth, setSidebarWidth] = useState(180);
  const [leftPaneFraction, setLeftPaneFraction] = useState(0.5);
  const [editorFraction, setEditorFraction] = useState(0.6);
  const contentRef = useRef<HTMLDivElement>(null);
  const leftPaneRef = useRef<HTMLDivElement>(null);

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
  const [editingName, setEditingName] = useState(false);
  const [draftName, setDraftName] = useState('');

  const activeWorkspace = useMemo(
    () => workspaces.find((workspace) => workspace.id === activeWorkspaceId) ?? null,
    [workspaces, activeWorkspaceId]
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

  const selectedToolIds = useMemo(() => new Set(activeWorkspace?.selected_tool_ids ?? []), [activeWorkspace?.selected_tool_ids]);
  const fileTree = useMemo(() => buildUserSpaceTree(files), [files]);
  const folderPaths = useMemo(() => listFolderPaths(files), [files]);

  const activeWorkspaceRole = useMemo(() => {
    if (!activeWorkspace) return 'viewer';
    if (activeWorkspace.owner_user_id === currentUser.id) return 'owner';
    return activeWorkspace.members.find((member) => member.user_id === currentUser.id)?.role ?? 'viewer';
  }, [activeWorkspace, currentUser.id]);

  const canEditWorkspace = activeWorkspaceRole === 'owner' || activeWorkspaceRole === 'editor';
  const isOwner = activeWorkspaceRole === 'owner';

  // Derive effective runtime display state from session_state + devserver_running
  const runtimeDisplayState = useMemo(() => {
    if (!runtimeStatus) return 'stopped';
    const { session_state, devserver_running, last_error } = runtimeStatus;
    if (session_state === 'running' && !devserver_running) {
      return last_error ? 'error' : 'starting';
    }
    return session_state;
  }, [runtimeStatus]);

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
        setWorkspaces(page.items);
        if (page.items.length === 0) {
          setActiveWorkspaceId(null);
          clearCookieValue(lastWorkspaceCookieName);
        } else if (!activeWorkspaceId) {
          const lastWorkspaceId = getCookieValue(lastWorkspaceCookieName);
          const matchingWorkspace = lastWorkspaceId
            ? page.items.find((workspace) => workspace.id === lastWorkspaceId)
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
              setActiveWorkspaceId(page.items[0].id);
            }
          } else {
            setActiveWorkspaceId(page.items[0].id);
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

  const loadWorkspaceData = useCallback(async (workspaceId: string) => {
    const requestId = ++loadWorkspaceDataRequestIdRef.current;

    try {
      const [nextFiles, nextSnapshots] = await Promise.all([
        api.listUserSpaceFiles(workspaceId),
        api.listUserSpaceSnapshots(workspaceId),
      ]);

      if (requestId !== loadWorkspaceDataRequestIdRef.current) {
        return;
      }

      setFiles(nextFiles);
      setSnapshots(nextSnapshots);

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
        const tools = await api.listUserSpaceAvailableTools();
        setAvailableTools(tools);
      } catch (err) {
        console.warn('Failed to load User Space tools', err);
      }
    };

    loadTools();
  }, []);

  useEffect(() => {
    if (!activeWorkspaceId) return;
    loadWorkspaceData(activeWorkspaceId);
  }, [activeWorkspaceId, loadWorkspaceData]);

  useEffect(() => {
    if (!activeWorkspaceId) return;

    const refreshInterval = window.setInterval(() => {
      if (fileDirty || savingFile) return;
      loadWorkspaceData(activeWorkspaceId);
    }, 4000);

    return () => {
      window.clearInterval(refreshInterval);
    };
  }, [activeWorkspaceId, fileDirty, loadWorkspaceData, savingFile]);

  useEffect(() => {
    fileContentCacheRef.current = fileContentCache;
  }, [fileContentCache]);

  useEffect(() => {
    fileDirtyRef.current = fileDirty;
  }, [fileDirty]);

  // SSE subscription for workspace change events (file upsert/patch/delete, snapshots).
  // Bumps previewRefreshCounter to remount the preview iframe and reloads workspace data.
  useEffect(() => {
    if (!activeWorkspaceId) return;

    const source = api.subscribeWorkspaceEvents(activeWorkspaceId, 0);
    let lastGeneration = 0;

    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as { generation: number };
        if (data.generation > lastGeneration) {
          lastGeneration = data.generation;
          setPreviewRefreshCounter((c) => c + 1);
          if (!fileDirtyRef.current) {
            void loadWorkspaceData(activeWorkspaceId);
          }
        }
      } catch {
        // ignore malformed messages
      }
    };

    source.onerror = () => {
      // EventSource auto-reconnects; nothing to do
    };

    return () => {
      source.close();
    };
  }, [activeWorkspaceId, loadWorkspaceData]);

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
        setIsWorkspaceMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleEscape);
    };
  }, [isWorkspaceMenuOpen]);

  useEffect(() => {
    if (!isWorkspaceMenuOpen) {
      setDeleteConfirmWorkspaceId(null);
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
    setCreatingWorkspace(true);
    try {
      const created = await api.createUserSpaceWorkspace({
        selected_tool_ids: [],
      });
      await api.upsertUserSpaceFile(created.id, 'dashboard/main.ts', {
        content: 'export function render(container: HTMLElement) {\n  container.innerHTML = `<h2>Interactive Report</h2><p>Ask chat to build your report and wire live data connections.</p>`;\n}\n',
        artifact_type: 'module_ts',
      });
      await api.createConversation(undefined, created.id);
      setActiveWorkspaceId(created.id);
      await loadWorkspaces();
      await loadWorkspaceData(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workspace');
    } finally {
      setCreatingWorkspace(false);
    }
  }, [loadWorkspaceData, loadWorkspaces]);

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

  const handleSaveFile = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setSavingFile(true);
    try {
      await api.upsertUserSpaceFile(activeWorkspaceId, selectedFilePath, {
        content: fileContent,
        artifact_type: 'module_ts',
      });

      const fileMeta = files.find((file) => file.path === selectedFilePath);
      setFileContentCache((current) => ({
        ...current,
        [selectedFilePath]: {
          content: fileContent,
          updatedAt: fileMeta?.updated_at ?? current[selectedFilePath]?.updatedAt ?? '',
        },
      }));

      setFileDirty(false);
      await loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save file');
    } finally {
      setSavingFile(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, fileContent, files, loadWorkspaceData, selectedFilePath]);

  const handleToggleWorkspaceTool = useCallback(async (toolId: string) => {
    if (!activeWorkspace || !canEditWorkspace) return;

    const nextSelected = new Set(activeWorkspace.selected_tool_ids);
    if (nextSelected.has(toolId)) {
      nextSelected.delete(toolId);
    } else {
      nextSelected.add(toolId);
    }

    setSavingWorkspaceTools(true);
    try {
      const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, {
        selected_tool_ids: Array.from(nextSelected),
      });
      setWorkspaces((current) => current.map((workspace) => workspace.id === updated.id ? updated : workspace));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update tool selection');
    } finally {
      setSavingWorkspaceTools(false);
    }
  }, [activeWorkspace, canEditWorkspace]);

  const handleUserMessageSubmitted = useCallback(async (message: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    const snapshot = await api.createUserSpaceSnapshot(activeWorkspaceId, { message });
    setSnapshots((current) => [snapshot, ...current]);
    await loadWorkspaceData(activeWorkspaceId);
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData]);

  const handleRestoreSnapshot = useCallback(async (snapshotId: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    try {
      await api.restoreUserSpaceSnapshot(activeWorkspaceId, snapshotId);
      await loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restore snapshot');
    }
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData]);

  const refreshRuntimeStatus = useCallback(async () => {
    if (!activeWorkspaceId) {
      setRuntimeStatus(null);
      return;
    }
    try {
      const status = await api.getUserSpaceRuntimeDevserverStatus(activeWorkspaceId);
      setRuntimeStatus(status);
    } catch {
      setRuntimeStatus(null);
    }
  }, [activeWorkspaceId]);

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
    const scheduleReconnect = () => {
      if (!reconnectEnabled || collabReconnectTimerRef.current !== null) {
        return;
      }
      collabReconnectTimerRef.current = window.setTimeout(() => {
        collabReconnectTimerRef.current = null;
        setCollabReconnectNonce((value) => value + 1);
      }, 1500);
    };

    const socketUrl = api.getUserSpaceCollabWebSocketUrl(activeWorkspaceId, selectedFilePath);
    const socket = new WebSocket(socketUrl);
    collabSocketRef.current = socket;

    socket.onopen = () => {
      setCollabConnected(true);
      if (collabReconnectTimerRef.current !== null) {
        window.clearTimeout(collabReconnectTimerRef.current);
        collabReconnectTimerRef.current = null;
      }
      try {
        socket.send(JSON.stringify({ type: 'presence' }));
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

    socket.onclose = () => {
      setCollabConnected(false);
      setCollabPresenceCount(0);
      scheduleReconnect();
    };

    socket.onerror = () => {
      setCollabConnected(false);
      setCollabPresenceCount(0);
      scheduleReconnect();
    };

    return () => {
      reconnectEnabled = false;
      if (collabReconnectTimerRef.current !== null) {
        window.clearTimeout(collabReconnectTimerRef.current);
        collabReconnectTimerRef.current = null;
      }
      socket.close();
      if (collabSocketRef.current === socket) {
        collabSocketRef.current = null;
      }
    };
  }, [activeWorkspaceId, loadWorkspaceData, selectedFilePath, collabReconnectNonce]);

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

    let reconnectEnabled = true;
    const scheduleReconnect = () => {
      if (!reconnectEnabled || terminalReconnectTimerRef.current !== null) {
        return;
      }
      terminalReconnectTimerRef.current = window.setTimeout(() => {
        terminalReconnectTimerRef.current = null;
        setTerminalReconnectNonce((value) => value + 1);
      }, 1500);
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

    const wsUrl = api.getUserSpaceRuntimePtyWebSocketUrl(activeWorkspaceId);
    const socket = new WebSocket(wsUrl);
    terminalSocketRef.current = socket;

    const dataDisposable = terminal.onData((data) => {
      if (terminalReadOnlyRef.current) {
        return;
      }
      if (socket.readyState !== WebSocket.OPEN) {
        return;
      }
      try {
        socket.send(JSON.stringify({ type: 'input', data }));
      } catch {
        // ignore
      }
    });

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
        const payload = JSON.parse(event.data) as { type?: string; data?: string; read_only?: boolean };
        if (payload.type === 'status') {
          const isReadOnly = Boolean(payload.read_only);
          terminalReadOnlyRef.current = isReadOnly;
          setTerminalReadOnly(isReadOnly);
          terminal.options.disableStdin = isReadOnly;
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
      scheduleReconnect();
    };

    socket.onerror = () => {
      scheduleReconnect();
    };

    return () => {
      reconnectEnabled = false;
      if (terminalReconnectTimerRef.current !== null) {
        window.clearTimeout(terminalReconnectTimerRef.current);
        terminalReconnectTimerRef.current = null;
      }
      dataDisposable.dispose();
      socket.close();
      if (terminalSocketRef.current === socket) {
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
  }, [activeRightTab, activeWorkspaceId, canEditWorkspace, terminalReconnectNonce]);

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
    try {
      await api.deleteUserSpaceWorkspace(workspaceId);
      setDeleteConfirmWorkspaceId(null);
      setIsWorkspaceMenuOpen(false);
      if (activeWorkspaceId === workspaceId) {
        setActiveWorkspaceId(null);
        clearCookieValue(lastWorkspaceCookieName);
        setFiles([]);
        setSnapshots([]);
        setFileContentCache({});
        setFileContent('');
        setSelectedFilePath('');
      }
      await loadWorkspaces();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete workspace');
    }
  }, [activeWorkspaceId, lastWorkspaceCookieName, loadWorkspaces]);

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

  const handleStartEditName = useCallback(() => {
    if (!activeWorkspace || !canEditWorkspace) return;
    setDraftName(activeWorkspace.name);
    setEditingName(true);
  }, [activeWorkspace, canEditWorkspace]);

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

  const handleCopyShareLink = useCallback(async () => {
    let url = shareLinkStatus?.share_url ?? null;
    if (!url) {
      const created = await handleEnsureShareLink(false);
      url = created?.share_url ?? null;
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
  }, [handleEnsureShareLink, shareLinkStatus?.share_url]);

  const handleOpenFullPreview = useCallback(async () => {
    let url = shareLinkStatus?.share_url ?? null;
    if (!url) {
      const created = await handleEnsureShareLink(false);
      url = created?.share_url ?? null;
    }

    if (!url) return;
    window.open(url, '_blank', 'noopener,noreferrer');
  }, [handleEnsureShareLink, shareLinkStatus?.share_url]);

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

  const handleSaveName = useCallback(async () => {
    if (!activeWorkspace || !canEditWorkspace || !draftName.trim()) return;
    try {
      const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, { name: draftName.trim() });
      setWorkspaces((current) => current.map((ws) => ws.id === updated.id ? updated : ws));
      setEditingName(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename workspace');
    }
  }, [activeWorkspace, canEditWorkspace, draftName]);

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
          rows.push(
            <div key={node.path} className="userspace-file-item userspace-tree-row userspace-tree-folder-row">
              <button className="userspace-item-content userspace-tree-content" onClick={() => handleToggleFolder(node.path)} style={indentStyle}>
                <span className="userspace-tree-chevron" aria-hidden="true">
                  {isExpanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                </span>
                <span className="userspace-folder-label">{node.name}</span>
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

      return [
        <div
          key={node.path}
          className={`userspace-file-item userspace-tree-row ${node.path === selectedFilePath ? 'active' : ''}`}
        >
          <button className="userspace-item-content userspace-tree-content" onClick={() => handleSelectFile(node.path)} style={indentStyle}>
            <span className="userspace-tree-file-label">{node.name}</span>
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
  }, [canEditWorkspace, deleteConfirmFileId, deleteConfirmFolderPath, expandedFolders, handleDeleteFile, handleDeleteFolder, handleRenameFile, handleRenameFolder, handleSelectFile, handleStartCreateFile, handleToggleFolder, renameValue, renamingFilePath, renamingFolderPath, selectedFilePath]);

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
              <span className="model-selector-arrow">▾</span>
            </button>

            {isWorkspaceMenuOpen && workspaces.length > 0 && (
              <div className="model-selector-dropdown userspace-workspace-dropdown">
                <div className="model-selector-dropdown-inner" role="listbox" aria-label="Workspace list">
                  {workspaces.map((ws) => {
                    const canDeleteWorkspace = ws.owner_user_id === currentUser.id;
                    const isConfirmingDelete = deleteConfirmWorkspaceId === ws.id;
                    return (
                      <div
                        key={ws.id}
                        className={`model-selector-item userspace-workspace-item ${ws.id === activeWorkspaceId ? 'is-selected' : ''} ${!canDeleteWorkspace ? 'is-shared' : ''}`}
                      >
                        <button
                          type="button"
                          role="option"
                          aria-selected={ws.id === activeWorkspaceId}
                          className="userspace-workspace-select-btn"
                          onClick={() => {
                            setActiveWorkspaceId(ws.id);
                            setIsWorkspaceMenuOpen(false);
                            setDeleteConfirmWorkspaceId(null);
                          }}
                        >
                          <span className="model-selector-item-name">{ws.name}</span>
                        </button>

                        {canDeleteWorkspace && (
                          <div className="userspace-workspace-item-actions">
                            {isConfirmingDelete ? (
                              <>
                                <button
                                  type="button"
                                  className="chat-action-btn confirm-delete"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    void handleDeleteWorkspace(ws.id);
                                  }}
                                  title="Confirm delete workspace"
                                >
                                  <Check size={12} />
                                </button>
                                <button
                                  type="button"
                                  className="chat-action-btn cancel-delete"
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
                              <button
                                type="button"
                                className="chat-action-btn"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  setDeleteConfirmWorkspaceId(ws.id);
                                }}
                                title="Delete workspace"
                              >
                                <Trash2 size={12} />
                              </button>
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
          <button className="btn btn-primary btn-sm" onClick={handleCreateWorkspace} disabled={creatingWorkspace} title="New workspace">
            <Plus size={14} />
          </button>
          {isOwner && (
            <>
              <button className="btn btn-secondary btn-sm" onClick={handleOpenMembersModal} title="Manage members">
                <Users size={14} />
              </button>
            </>
          )}
          {canEditWorkspace && activeWorkspace && (
            <>
              <button className="btn btn-secondary btn-sm" onClick={handleStartEditName} title="Rename workspace">
                <Pencil size={14} />
              </button>
            </>
          )}
        </div>

        <div className="userspace-toolbar-group userspace-toolbar-group-right">
          <div className="userspace-toolbar-status-group">
            {previewExecuting && (
              <span className="userspace-toolbar-live-status" title="Live data connection in progress">
                <span className="userspace-toolbar-live-spinner" aria-hidden="true" />
                Connecting data...
              </span>
            )}
            {activeWorkspace && (
              <span className="userspace-status-pill userspace-status-pill-info">
                {activeWorkspaceRole}{!canEditWorkspace ? ' (read-only)' : ''}
              </span>
            )}
            {activeWorkspaceId && (
              <span
                className={`userspace-status-pill ${collabConnected ? 'userspace-status-pill-success' : 'userspace-status-pill-muted'}`}
                title="Collaborative editor connection state"
              >
                {collabConnected ? `collab (${collabPresenceCount})` : 'collab offline'}
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
                {runtimeDisplayState}
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
                  <Play size={14} />
                </button>
              )}
              {showRestartRuntimeButton && (
                <button className="btn btn-secondary btn-sm btn-icon" onClick={handleRestartRuntime} disabled={runtimeBusy} title="Restart runtime">
                  <RotateCw size={14} />
                </button>
              )}
              {showStopRuntimeButton && (
                <button className="btn btn-secondary btn-sm btn-icon" onClick={handleStopRuntime} disabled={runtimeBusy} title="Stop runtime">
                  <Square size={14} />
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
              disabled={savingWorkspaceTools}
              readOnly={!canEditWorkspace}
              saving={savingWorkspaceTools}
              title="Workspace Tools"
            />
            <button
              className="btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn"
              onClick={toggleFullscreen}
              title={isFullscreen ? 'Exit full screen' : 'Full screen'}
            >
              {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
            </button>
            <button
              className="btn btn-primary btn-sm btn-icon userspace-toolbar-action-btn userspace-save-btn"
              onClick={handleSaveFile}
              disabled={!activeWorkspaceId || !canEditWorkspace || savingFile || !fileDirty}
              title={savingFile ? 'Saving file...' : 'Save file'}
            >
              <Save size={14} className={savingFile ? 'spinning' : undefined} />
            </button>
          </div>
        </div>
      </div>

      {/* === Status messages === */}
      {loading && <p className="userspace-status">Loading workspaces...</p>}
      {error && <p className="userspace-error userspace-status">{error}</p>}

      {/* === Rename inline editor === */}
      {editingName && activeWorkspace && (
        <div className="userspace-inline-edit">
          <label>Rename:</label>
          <input
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSaveName(); if (e.key === 'Escape') setEditingName(false); }}
            autoFocus
          />
          <button className="btn btn-primary btn-sm" onClick={handleSaveName}><Check size={14} /></button>
          <button className="btn btn-secondary btn-sm" onClick={() => setEditingName(false)}><X size={14} /></button>
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
            <div className="userspace-code-editor">
              {!canEditWorkspace && <div className="userspace-readonly-badge">Read-only</div>}
              <CodeMirror
                value={fileContent}
                onChange={(value) => {
                  setFileContent(value);
                  setFileDirty(true);
                }}
                extensions={[javascript({ typescript: true, jsx: true })]}
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
                basicSetup={{
                  lineNumbers: true,
                  foldGutter: true,
                  bracketMatching: true,
                  closeBrackets: true,
                  autocompletion: true,
                  highlightActiveLine: true,
                  indentOnInput: true,
                  tabSize: 2,
                }}
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
                workspaceId={activeWorkspaceId}
                workspaceAvailableTools={availableTools}
                workspaceSelectedToolIds={activeWorkspace?.selected_tool_ids ?? []}
                onToggleWorkspaceTool={handleToggleWorkspaceTool}
                workspaceSavingTools={savingWorkspaceTools}
                onUserMessageSubmitted={canEditWorkspace ? handleUserMessageSubmitted : undefined}
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
                  activeWorkspaceId
                    ? api.getUserSpaceRuntimePreviewUrl(activeWorkspaceId)
                    : undefined
                }
                runtimeAvailable={runtimeStatus?.devserver_running}
                runtimeError={runtimeStatus?.last_error ?? undefined}
                previewInstanceKey={`${activeWorkspaceId ?? ''}:${previewRefreshCounter}`}
                workspaceId={activeWorkspaceId ?? undefined}
                onExecutionStateChange={setPreviewExecuting}
              />
            </div>
          ) : (
            <div className="userspace-preview-section" style={{ padding: 12 }}>
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
            <button className="userspace-snapshots-toggle" onClick={() => setShowSnapshots(!showSnapshots)}>
              <History size={14} />
              <span>Snapshots ({snapshots.length})</span>
              <ChevronDown size={14} className={showSnapshots ? '' : 'rotated'} />
            </button>
            {showSnapshots && (
              <div className="userspace-snapshots-list">
                {snapshots.map((snapshot) => (
                  <div key={snapshot.id} className="userspace-snapshot-item">
                    <div className="userspace-snapshot-info">
                      <code>{snapshot.id.slice(0, 8)}</code>
                      <span className="userspace-muted">{snapshot.message || 'No message'}</span>
                    </div>
                    <button
                      className="btn btn-secondary btn-sm"
                      onClick={() => handleRestoreSnapshot(snapshot.id)}
                      disabled={!canEditWorkspace}
                    >
                      Restore
                    </button>
                  </div>
                ))}
                {snapshots.length === 0 && (
                  <p className="userspace-muted" style={{ padding: '8px' }}>No snapshots yet</p>
                )}
              </div>
            )}
          </div>
        </div>
        )}
      </div>

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

                    {shareLinkStatus?.has_share_link && shareLinkStatus.share_url ? (
                      <>
                        <label htmlFor="userspace-share-url" className="userspace-share-label">Active share URL</label>
                        <input id="userspace-share-url" value={shareLinkStatus.share_url} readOnly />
                        <div className="userspace-share-meta">
                          {shareLinkStatus.created_at ? `Created ${new Date(shareLinkStatus.created_at).toLocaleString()}` : 'Share link active'}
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
