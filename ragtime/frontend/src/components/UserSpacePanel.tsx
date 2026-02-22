import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, ChevronRight, ExternalLink, File, History, Link2, Maximize2, Minimize2, Pencil, Plus, Save, Settings, Trash2, Users, X } from 'lucide-react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';

import { api } from '@/api';
import type { User, UserSpaceAvailableTool, UserSpaceFileInfo, UserSpaceLiveDataConnection, UserSpaceSnapshot, UserSpaceWorkspace, UserSpaceWorkspaceMember, WorkspaceRole } from '@/types';
import { buildUserSpaceTree, getAncestorFolderPaths, listFolderPaths } from '@/utils/userspaceTree';
import { ChatPanel } from './ChatPanel';
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
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [sharingWorkspace, setSharingWorkspace] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
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
  const [showSnapshots, setShowSnapshots] = useState(true);
  const toolPickerRef = useRef<HTMLDivElement>(null);
  const workspaceDropdownRef = useRef<HTMLDivElement>(null);
  const fileContentCacheRef = useRef(fileContentCache);

  // Resize state
  const [sidebarWidth, setSidebarWidth] = useState(180);
  const [leftPaneFraction, setLeftPaneFraction] = useState(0.5);
  const [editorFraction, setEditorFraction] = useState(0.6);
  const contentRef = useRef<HTMLDivElement>(null);
  const leftPaneRef = useRef<HTMLDivElement>(null);

  // Collapse state: track which panes are collapsed + their last size for restore
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rightPaneCollapsed, setRightPaneCollapsed] = useState(false);
  const [chatCollapsed, setChatCollapsed] = useState(false);
  const prevSidebarWidth = useRef(180);
  const prevLeftPaneFraction = useRef(0.5);
  const prevEditorFraction = useRef(0.6);

  const SIDEBAR_COLLAPSE_THRESHOLD = 60;
  const MAIN_COLLAPSE_LEFT_THRESHOLD = 0.08;
  const MAIN_COLLAPSE_RIGHT_THRESHOLD = 0.92;
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
      if (next > EDITOR_COLLAPSE_BOTTOM_THRESHOLD) {
        prevEditorFraction.current = prev < EDITOR_COLLAPSE_BOTTOM_THRESHOLD ? prev : prevEditorFraction.current;
        setChatCollapsed(true);
        return 1;
      }
      setChatCollapsed(false);
      return Math.max(0.2, next);
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
    setChatCollapsed(false);
    setEditorFraction(prevEditorFraction.current || 0.6);
  }, []);
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
        if (!activeWorkspaceId && page.items.length > 0) {
          setActiveWorkspaceId(page.items[0].id);
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
  }, [activeWorkspaceId, workspaces.length]);

  const loadWorkspaceData = useCallback(async (workspaceId: string) => {
    try {
      const [nextFiles, nextSnapshots] = await Promise.all([
        api.listUserSpaceFiles(workspaceId),
        api.listUserSpaceSnapshots(workspaceId),
      ]);
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

      const selectedExists = nextFiles.some((file) => file.path === selectedFilePath);
      const preferredPath = selectedExists
        ? selectedFilePath
        : nextFiles.some((file) => file.path === previewEntryPath)
          ? previewEntryPath
          : nextFiles[0]?.path ?? previewEntryPath;

      setSelectedFilePath(preferredPath);

      if (nextFiles.some((file) => file.path === preferredPath)) {
        const preferredMeta = nextFiles.find((file) => file.path === preferredPath);
        const preferredUpdatedAt = preferredMeta?.updated_at ?? '';
        const cached = fileContentCacheRef.current[preferredPath];

        if (cached && cached.updatedAt === preferredUpdatedAt) {
          setFileContent(cached.content);
        } else {
          const file = await api.getUserSpaceFile(workspaceId, preferredPath);
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
      setError(err instanceof Error ? err.message : 'Failed to load workspace data');
    }
  }, [previewEntryPath, selectedFilePath]);

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
        name: `Workspace ${workspaces.length + 1}`,
        selected_tool_ids: availableTools.map((tool) => tool.id),
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
  }, [availableTools, loadWorkspaceData, loadWorkspaces, workspaces.length]);

  const handleSelectFile = useCallback(async (path: string) => {
    if (!activeWorkspaceId) return;

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
  }, [activeWorkspaceId, loadWorkspaces]);

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

  const handleAddMember = useCallback((userId: string) => {
    if (pendingMembers.some((m) => m.user_id === userId)) return;
    setPendingMembers((prev) => [...prev, { user_id: userId, role: 'viewer' as WorkspaceRole }]);
  }, [pendingMembers]);

  const handleRemoveMember = useCallback((userId: string) => {
    setPendingMembers((prev) => prev.filter((m) => m.user_id !== userId));
  }, []);

  const handleChangeMemberRole = useCallback((userId: string, role: WorkspaceRole) => {
    setPendingMembers((prev) => prev.map((m) => m.user_id === userId ? { ...m, role } : m));
  }, []);

  const handleSaveMembers = useCallback(async () => {
    if (!activeWorkspace || !isOwner) return;
    setSavingMembers(true);
    try {
      const updated = await api.updateUserSpaceWorkspaceMembers(activeWorkspace.id, { members: pendingMembers });
      setWorkspaces((current) => current.map((ws) => ws.id === updated.id ? updated : ws));
      setShowMembersModal(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update members');
    } finally {
      setSavingMembers(false);
    }
  }, [activeWorkspace, isOwner, pendingMembers]);

  const handleStartEditName = useCallback(() => {
    if (!activeWorkspace || !canEditWorkspace) return;
    setDraftName(activeWorkspace.name);
    setEditingName(true);
  }, [activeWorkspace, canEditWorkspace]);

  const requestWorkspaceShareUrl = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return null;
    const response = await api.createUserSpaceWorkspaceShareLink(activeWorkspaceId);
    return response.share_url;
  }, [activeWorkspaceId, canEditWorkspace]);

  const handleQuickShare = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setSharingWorkspace(true);
    try {
      const shareUrl = await requestWorkspaceShareUrl();
      if (!shareUrl) return;
      await navigator.clipboard.writeText(shareUrl);
      setShareCopied(true);
      window.setTimeout(() => setShareCopied(false), 1500);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to copy share link');
    } finally {
      setSharingWorkspace(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, requestWorkspaceShareUrl]);

  const handleOpenFullPreview = useCallback(async () => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    setSharingWorkspace(true);
    try {
      const shareUrl = await requestWorkspaceShareUrl();
      if (!shareUrl) return;
      window.open(shareUrl, '_blank', 'noopener,noreferrer');
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to open preview');
    } finally {
      setSharingWorkspace(false);
    }
  }, [activeWorkspaceId, canEditWorkspace, requestWorkspaceShareUrl]);

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
              <span className="userspace-role-badge">
                {activeWorkspaceRole}{!canEditWorkspace ? ' (read-only)' : ''}
              </span>
            )}
          </div>

          <div className="userspace-toolbar-actions" aria-label="Workspace sharing actions">
            <button
              className="btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn"
              onClick={handleQuickShare}
              disabled={!activeWorkspaceId || !canEditWorkspace || sharingWorkspace}
              title={shareCopied ? 'Share link copied' : 'Copy static share link'}
            >
              {shareCopied ? <Check size={14} /> : <Link2 size={14} />}
            </button>
            <button
              className="btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn"
              onClick={handleOpenFullPreview}
              disabled={!activeWorkspaceId || !canEditWorkspace || sharingWorkspace}
              title="Open shared full-screen preview"
            >
              <ExternalLink size={14} />
            </button>
          </div>

          <div className="userspace-toolbar-actions" aria-label="Workspace editor actions">
            <div className="userspace-tool-picker-wrap" ref={toolPickerRef}>
              <button
                className={`btn btn-secondary btn-sm btn-icon userspace-toolbar-action-btn ${showToolPicker ? 'active' : ''}`}
                onClick={() => setShowToolPicker(!showToolPicker)}
                title="Workspace tools"
              >
                <Settings size={14} />
              </button>
              {showToolPicker && activeWorkspace && (
                <div className="userspace-tool-dropdown">
                  <h4>Workspace Tools</h4>
                  {!canEditWorkspace && <p className="userspace-muted">Read-only access</p>}
                  <div className="userspace-tool-list">
                    {availableTools.map((tool) => (
                      <label key={tool.id} className="checkbox-label userspace-tool-item">
                        <input
                          type="checkbox"
                          checked={selectedToolIds.has(tool.id)}
                          onChange={() => handleToggleWorkspaceTool(tool.id)}
                          disabled={savingWorkspaceTools || !canEditWorkspace}
                        />
                        <span>
                          <strong>{tool.name}</strong>
                          <small className="userspace-muted">{tool.tool_type}</small>
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              )}
            </div>
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

          <ResizeHandle direction="vertical" onResize={handleResizeEditorChat} collapsed={chatCollapsed ? 'after' : undefined} onExpand={expandChat} />

          {/* Chat section */}
          {!chatCollapsed && (
          <div className="userspace-chat-section" style={{ flex: 1 - editorFraction }}>
            {activeWorkspaceId ? (
              <ChatPanel
                key={activeWorkspaceId}
                currentUser={currentUser}
                workspaceId={activeWorkspaceId}
                onUserMessageSubmitted={canEditWorkspace ? handleUserMessageSubmitted : undefined}
                embedded
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
          <div className="userspace-preview-section">
            <UserSpaceArtifactPreview
              entryPath={previewEntryPath}
              workspaceFiles={previewWorkspaceFiles}
              liveDataConnections={previewLiveDataConnections}
              previewInstanceKey={activeWorkspaceId ?? ''}
              workspaceId={activeWorkspaceId ?? undefined}
              onExecutionStateChange={setPreviewExecuting}
            />
          </div>

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
        <div className="modal-overlay" onClick={() => setShowMembersModal(false)}>
          <div className="modal-content modal-small" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Manage Members</h3>
              <button className="modal-close" onClick={() => setShowMembersModal(false)}>&times;</button>
            </div>
            <div className="modal-body">
              <div className="userspace-members-list">
                <div className="userspace-member-row userspace-member-owner">
                  <span>{formatUserLabel(currentUser)}</span>
                  <small className="userspace-muted">owner</small>
                </div>
                {pendingMembers.map((member) => {
                  const user = allUsers.find((u) => u.id === member.user_id);
                  return (
                    <div key={member.user_id} className="userspace-member-row">
                      <span>{formatUserLabel(user, member.user_id)}</span>
                      <select
                        value={member.role}
                        onChange={(e) => handleChangeMemberRole(member.user_id, e.target.value as WorkspaceRole)}
                      >
                        <option value="editor">editor</option>
                        <option value="viewer">viewer</option>
                      </select>
                      <button className="chat-action-btn" onClick={() => handleRemoveMember(member.user_id)} title="Remove member">
                        <X size={14} />
                      </button>
                    </div>
                  );
                })}
              </div>
              {allUsers.length > 0 && (
                <div className="userspace-add-member">
                  <select
                    id="userspace-add-member-select"
                    defaultValue=""
                    onChange={(e) => {
                      if (e.target.value) {
                        handleAddMember(e.target.value);
                        e.target.value = '';
                      }
                    }}
                  >
                    <option value="" disabled>Add a member...</option>
                    {allUsers
                      .filter((u) => u.id !== activeWorkspace.owner_user_id && !pendingMembers.some((m) => m.user_id === u.id))
                      .map((u) => (
                        <option key={u.id} value={u.id}>{formatUserLabel(u, u.id)}</option>
                      ))}
                  </select>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowMembersModal(false)}>Cancel</button>
              <button className="btn btn-primary" onClick={handleSaveMembers} disabled={savingMembers}>
                {savingMembers ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
