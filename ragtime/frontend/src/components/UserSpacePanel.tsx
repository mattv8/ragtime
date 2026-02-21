import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Check, ChevronDown, File, History, Maximize2, Minimize2, Pencil, Plus, Save, Settings, Trash2, Users, X } from 'lucide-react';

import { api } from '@/api';
import type { User, UserSpaceAvailableTool, UserSpaceFileInfo, UserSpaceSnapshot, UserSpaceWorkspace, UserSpaceWorkspaceMember, WorkspaceRole } from '@/types';
import { ChatPanel } from './ChatPanel';
import { ResizeHandle } from './ResizeHandle';
import { UserSpaceArtifactPreview } from './UserSpaceArtifactPreview';

interface UserSpacePanelProps {
  currentUser: User;
  onFullscreenChange?: (fullscreen: boolean) => void;
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
  const [creatingWorkspace, setCreatingWorkspace] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const toggleFullscreen = useCallback(() => {
    const next = !isFullscreen;
    setIsFullscreen(next);
    onFullscreenChange?.(next);
  }, [isFullscreen, onFullscreenChange]);
  const [savingFile, setSavingFile] = useState(false);
  const [savingWorkspaceTools, setSavingWorkspaceTools] = useState(false);
  const [deleteConfirmFileId, setDeleteConfirmFileId] = useState<string | null>(null);
  const [newFileName, setNewFileName] = useState<string | null>(null);
  const [renamingFilePath, setRenamingFilePath] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState('');
  const [deleteConfirmWorkspaceId, setDeleteConfirmWorkspaceId] = useState<string | null>(null);
  const [showMembersModal, setShowMembersModal] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [pendingMembers, setPendingMembers] = useState<UserSpaceWorkspaceMember[]>([]);
  const [savingMembers, setSavingMembers] = useState(false);
  const [showToolPicker, setShowToolPicker] = useState(false);
  const [showSnapshots, setShowSnapshots] = useState(true);
  const toolPickerRef = useRef<HTMLDivElement>(null);
  const fileContentCacheRef = useRef(fileContentCache);
  const creatingWorkspaceRef = useRef(false);

  // Resize state
  const [sidebarWidth, setSidebarWidth] = useState(180);
  const [leftPaneFraction, setLeftPaneFraction] = useState(0.5);
  const [editorFraction, setEditorFraction] = useState(0.6);
  const contentRef = useRef<HTMLDivElement>(null);
  const leftPaneRef = useRef<HTMLDivElement>(null);

  const handleResizeSidebar = useCallback((delta: number) => {
    setSidebarWidth((prev) => Math.max(100, Math.min(400, prev + delta)));
  }, []);

  const handleResizeMainSplit = useCallback((delta: number) => {
    const el = contentRef.current;
    if (!el) return;
    const totalWidth = el.offsetWidth;
    if (totalWidth === 0) return;
    setLeftPaneFraction((prev) => Math.max(0.25, Math.min(0.75, prev + delta / totalWidth)));
  }, []);

  const handleResizeEditorChat = useCallback((delta: number) => {
    const el = leftPaneRef.current;
    if (!el) return;
    const totalHeight = el.offsetHeight;
    if (totalHeight === 0) return;
    setEditorFraction((prev) => Math.max(0.2, Math.min(0.85, prev + delta / totalHeight)));
  }, []);
  const [editingName, setEditingName] = useState(false);
  const [editingDescription, setEditingDescription] = useState(false);
  const [draftName, setDraftName] = useState('');
  const [draftDescription, setDraftDescription] = useState('');

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

  const getNextWorkspaceName = useCallback(() => {
    const existingNames = new Set(workspaces.map((workspace) => workspace.name.trim().toLowerCase()));
    let index = 1;
    while (existingNames.has(`workspace ${index}`)) {
      index += 1;
    }
    return `Workspace ${index}`;
  }, [workspaces]);

  const activeWorkspaceRole = useMemo(() => {
    if (!activeWorkspace) return 'viewer';
    if (activeWorkspace.owner_user_id === currentUser.id) return 'owner';
    return activeWorkspace.members.find((member) => member.user_id === currentUser.id)?.role ?? 'viewer';
  }, [activeWorkspace, currentUser.id]);

  const canEditWorkspace = activeWorkspaceRole === 'owner' || activeWorkspaceRole === 'editor';
  const isOwner = activeWorkspaceRole === 'owner';

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

  const handleCreateWorkspace = useCallback(async () => {
    if (creatingWorkspaceRef.current) {
      return;
    }
    creatingWorkspaceRef.current = true;
    setCreatingWorkspace(true);
    try {
      const created = await api.createUserSpaceWorkspace({
        name: getNextWorkspaceName(),
        description: 'User Space dashboard workspace',
        selected_tool_ids: availableTools.map((tool) => tool.id),
      });
      await api.upsertUserSpaceFile(created.id, 'dashboard/main.ts', {
        content: 'export function render(container: HTMLElement) {\n  container.innerHTML = `\n    <div style="max-width: 800px; margin: 0 auto; padding: var(--space-lg, 24px);">\n      <h2 style="color: var(--color-text-primary, #f1f5f9); margin: 0 0 var(--space-sm, 8px) 0;">Interactive Report</h2>\n      <p style="color: var(--color-text-secondary, #94a3b8); margin: 0;">Ask chat to build your report and wire live data connections.</p>\n    </div>\n  `;\n}\n',
        artifact_type: 'module_ts',
      });
      await api.createConversation(undefined, created.id);
      setActiveWorkspaceId(created.id);
      await loadWorkspaces();
      await loadWorkspaceData(created.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workspace');
    } finally {
      creatingWorkspaceRef.current = false;
      setCreatingWorkspace(false);
    }
  }, [availableTools, getNextWorkspaceName, loadWorkspaceData, loadWorkspaces]);

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

  const handleTaskComplete = useCallback(() => {
    if (!activeWorkspaceId) return;
    loadWorkspaceData(activeWorkspaceId);
  }, [activeWorkspaceId, loadWorkspaceData]);

  const handleRestoreSnapshot = useCallback(async (snapshotId: string) => {
    if (!activeWorkspaceId || !canEditWorkspace) return;
    try {
      await api.restoreUserSpaceSnapshot(activeWorkspaceId, snapshotId);
      await loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to restore snapshot');
    }
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData]);

  const handleCreateNewFile = useCallback(async (path: string) => {
    if (!activeWorkspaceId || !canEditWorkspace || !path.trim()) return;
    try {
      const nextPath = path.trim();
      await api.upsertUserSpaceFile(activeWorkspaceId, path.trim(), {
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
      await loadWorkspaceData(activeWorkspaceId);
      handleSelectFile(nextPath);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create file');
    }
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData, handleSelectFile]);

  const handleRenameFile = useCallback(async (oldPath: string, newPath: string) => {
    if (!activeWorkspaceId || !canEditWorkspace || !newPath.trim() || newPath.trim() === oldPath) {
      setRenamingFilePath(null);
      return;
    }
    try {
      // Read old file content, save to new path, delete old
      const file = await api.getUserSpaceFile(activeWorkspaceId, oldPath);
      await api.upsertUserSpaceFile(activeWorkspaceId, newPath.trim(), {
        content: file.content,
        artifact_type: file.artifact_type || undefined,
      });
      await api.deleteUserSpaceFile(activeWorkspaceId, oldPath);
      setRenamingFilePath(null);
      setFileContentCache((current) => {
        const next = { ...current };
        const currentValue = next[oldPath];
        if (currentValue) {
          next[newPath.trim()] = currentValue;
        }
        delete next[oldPath];
        return next;
      });
      if (selectedFilePath === oldPath) {
        setSelectedFilePath(newPath.trim());
      }
      await loadWorkspaceData(activeWorkspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to rename file');
    }
  }, [activeWorkspaceId, canEditWorkspace, loadWorkspaceData, selectedFilePath]);

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

  const handleDeleteWorkspace = useCallback(async (workspaceId: string) => {
    try {
      await api.deleteUserSpaceWorkspace(workspaceId);
      setDeleteConfirmWorkspaceId(null);
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

  const handleStartEditDescription = useCallback(() => {
    if (!activeWorkspace || !canEditWorkspace) return;
    setDraftDescription(activeWorkspace.description ?? '');
    setEditingDescription(true);
  }, [activeWorkspace, canEditWorkspace]);

  const handleSaveDescription = useCallback(async () => {
    if (!activeWorkspace || !canEditWorkspace) return;
    try {
      const updated = await api.updateUserSpaceWorkspace(activeWorkspace.id, { description: draftDescription.trim() || undefined });
      setWorkspaces((current) => current.map((ws) => ws.id === updated.id ? updated : ws));
      setEditingDescription(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to update description');
    }
  }, [activeWorkspace, canEditWorkspace, draftDescription]);

  return (
    <div className={`userspace-layout${isFullscreen ? ' userspace-fullscreen' : ''}`}>
      {/* === Top toolbar === */}
      <div className="userspace-toolbar">
        <div className="userspace-toolbar-group">
          <select
            className="userspace-ws-select"
            value={activeWorkspaceId ?? ''}
            onChange={(e) => setActiveWorkspaceId(e.target.value || null)}
          >
            {workspaces.length === 0 && <option value="">No workspaces</option>}
            {workspaces.map((ws) => (
              <option key={ws.id} value={ws.id}>{ws.name}</option>
            ))}
          </select>
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
              <button className="btn btn-secondary btn-sm" onClick={() => setDeleteConfirmWorkspaceId(activeWorkspaceId)} title="Delete workspace">
                <Trash2 size={14} />
              </button>
            </>
          )}
          {canEditWorkspace && activeWorkspace && (
            <>
              <button className="btn btn-secondary btn-sm" onClick={handleStartEditName} title="Rename workspace">
                <Pencil size={14} />
              </button>
              <button className="btn btn-secondary btn-sm" onClick={handleStartEditDescription} title="Edit description">
                <File size={14} />
              </button>
            </>
          )}
        </div>

        <div className="userspace-toolbar-group userspace-toolbar-group-right">
          {activeWorkspace && (
            <span className="userspace-role-badge">
              {activeWorkspaceRole}{!canEditWorkspace ? ' (read-only)' : ''}
            </span>
          )}
          <div className="userspace-tool-picker-wrap" ref={toolPickerRef}>
            <button
              className={`btn btn-secondary btn-sm ${showToolPicker ? 'active' : ''}`}
              onClick={() => setShowToolPicker(!showToolPicker)}
              title="Workspace tools"
            >
              <Settings size={14} />
            </button>
            <button
              className="btn btn-secondary btn-sm"
              onClick={toggleFullscreen}
              title={isFullscreen ? 'Exit full screen' : 'Full screen'}
            >
              {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
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
            className="btn btn-primary btn-sm userspace-save-btn"
            onClick={handleSaveFile}
            disabled={!activeWorkspaceId || !canEditWorkspace || savingFile || !fileDirty}
            title="Save file"
          >
            <Save size={14} />
            {savingFile ? 'Saving...' : 'Save'}
          </button>
        </div>
      </div>

      {/* === Status messages === */}
      {loading && <p className="userspace-status">Loading workspaces...</p>}
      {error && <p className="userspace-error userspace-status">{error}</p>}

      {/* === Rename / description inline editors === */}
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
      {editingDescription && activeWorkspace && (
        <div className="userspace-inline-edit">
          <label>Description:</label>
          <input
            value={draftDescription}
            onChange={(e) => setDraftDescription(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSaveDescription(); if (e.key === 'Escape') setEditingDescription(false); }}
            placeholder="Workspace description"
            autoFocus
          />
          <button className="btn btn-primary btn-sm" onClick={handleSaveDescription}><Check size={14} /></button>
          <button className="btn btn-secondary btn-sm" onClick={() => setEditingDescription(false)}><X size={14} /></button>
        </div>
      )}

      {/* === Delete workspace confirmation === */}
      {deleteConfirmWorkspaceId && (
        <div className="userspace-confirm-bar">
          <span>Delete workspace &quot;{workspaces.find((w) => w.id === deleteConfirmWorkspaceId)?.name}&quot;?</span>
          <button className="btn btn-danger btn-sm" onClick={() => handleDeleteWorkspace(deleteConfirmWorkspaceId)}>
            <Check size={14} /> Delete
          </button>
          <button className="btn btn-secondary btn-sm" onClick={() => setDeleteConfirmWorkspaceId(null)}>
            <X size={14} /> Cancel
          </button>
        </div>
      )}

      {/* === Main content: left pane (editor+chat) | right pane (preview+snapshots) === */}
      <div className="userspace-content" ref={contentRef} style={{ gridTemplateColumns: `${leftPaneFraction}fr 0px ${1 - leftPaneFraction}fr` }}>
        {/* Left pane */}
        <div className="userspace-left-pane" ref={leftPaneRef}>
          <div className="userspace-editor-section" style={{ flex: editorFraction }}>
            {/* File sidebar */}
            <div className="userspace-file-sidebar" style={{ width: sidebarWidth }}>
              <div className="userspace-file-sidebar-header">
                <h4><File size={14} /> Files</h4>
              </div>
              <div className="userspace-file-list">
                {files.map((file) => {
                  const isConfirmingDelete = deleteConfirmFileId === file.path;
                  const isRenaming = renamingFilePath === file.path;
                  if (isRenaming) {
                    return (
                      <div key={file.path} className="userspace-file-item active">
                        <input
                          className="userspace-file-rename-input"
                          value={renameValue}
                          onChange={(e) => setRenameValue(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') handleRenameFile(file.path, renameValue);
                            if (e.key === 'Escape') setRenamingFilePath(null);
                          }}
                          autoFocus
                        />
                        <div className="userspace-item-actions" style={{ opacity: 1 }}>
                          <button className="chat-action-btn" onClick={() => handleRenameFile(file.path, renameValue)} title="Confirm">
                            <Check size={12} />
                          </button>
                          <button className="chat-action-btn" onClick={() => setRenamingFilePath(null)} title="Cancel">
                            <X size={12} />
                          </button>
                        </div>
                      </div>
                    );
                  }
                  return (
                    <div
                      key={file.path}
                      className={`userspace-file-item ${file.path === selectedFilePath ? 'active' : ''}`}
                    >
                      <button className="userspace-item-content" onClick={() => handleSelectFile(file.path)}>
                        <span>{file.path}</span>
                      </button>
                      {canEditWorkspace && (
                        <div className="userspace-item-actions">
                          {isConfirmingDelete ? (
                            <>
                              <button className="chat-action-btn confirm-delete" onClick={() => handleDeleteFile(file.path)} title="Confirm">
                                <Check size={12} />
                              </button>
                              <button className="chat-action-btn cancel-delete" onClick={() => setDeleteConfirmFileId(null)} title="Cancel">
                                <X size={12} />
                              </button>
                            </>
                          ) : (
                            <>
                              <button className="chat-action-btn" onClick={() => { setRenamingFilePath(file.path); setRenameValue(file.path); }} title="Rename">
                                <Pencil size={12} />
                              </button>
                              <button className="chat-action-btn" onClick={() => setDeleteConfirmFileId(file.path)} title="Delete">
                                <Trash2 size={12} />
                              </button>
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
                {/* New file input */}
                {newFileName !== null ? (
                  <div className="userspace-file-item">
                    <input
                      className="userspace-file-rename-input"
                      value={newFileName}
                      onChange={(e) => setNewFileName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleCreateNewFile(newFileName);
                        if (e.key === 'Escape') setNewFileName(null);
                      }}
                      placeholder="path/to/file.ts"
                      autoFocus
                    />
                    <div className="userspace-item-actions" style={{ opacity: 1 }}>
                      <button className="chat-action-btn" onClick={() => handleCreateNewFile(newFileName)} title="Create">
                        <Check size={12} />
                      </button>
                      <button className="chat-action-btn" onClick={() => setNewFileName(null)} title="Cancel">
                        <X size={12} />
                      </button>
                    </div>
                  </div>
                ) : canEditWorkspace ? (
                  <button className="userspace-new-file-btn" onClick={() => setNewFileName('')} title="New file">
                    <Plus size={12} /> New file
                  </button>
                ) : files.length === 0 ? (
                  <p className="userspace-muted" style={{ padding: '8px' }}>No files yet</p>
                ) : null}
              </div>
            </div>

            <ResizeHandle direction="horizontal" onResize={handleResizeSidebar} />

            {/* Code editor */}
            <div className="userspace-code-editor">
              {!canEditWorkspace && <div className="userspace-readonly-badge">Read-only</div>}
              <textarea
                value={fileContent}
                onChange={(e) => {
                  setFileContent(e.target.value);
                  setFileDirty(true);
                }}
                placeholder="Create dashboard/report/module source files here"
                disabled={!canEditWorkspace}
                spellCheck={false}
              />
            </div>
          </div>

          <ResizeHandle direction="vertical" onResize={handleResizeEditorChat} />

          {/* Chat section */}
          <div className="userspace-chat-section" style={{ flex: 1 - editorFraction }}>
            {activeWorkspaceId ? (
              <ChatPanel
                key={activeWorkspaceId}
                currentUser={currentUser}
                workspaceId={activeWorkspaceId}
                onUserMessageSubmitted={canEditWorkspace ? handleUserMessageSubmitted : undefined}
                onTaskComplete={handleTaskComplete}
                embedded
              />
            ) : (
              <div className="userspace-chat-placeholder">
                <p className="userspace-muted">Select or create a workspace to start chatting</p>
              </div>
            )}
          </div>
        </div>

        <ResizeHandle direction="horizontal" onResize={handleResizeMainSplit} />

        {/* Right pane */}
        <div className="userspace-right-pane">
          <div className="userspace-preview-section">
            <UserSpaceArtifactPreview
              entryPath={previewEntryPath}
              workspaceFiles={previewWorkspaceFiles}
              previewInstanceKey={activeWorkspaceId ?? ''}
            />
          </div>

          {/* Snapshots */}
          <div className="userspace-snapshots-section">
            <button className="userspace-snapshots-toggle" onClick={() => setShowSnapshots(!showSnapshots)}>
              <History size={14} />
              <span>Snapshots ({snapshots.length})</span>
              <ChevronDown size={14} className={showSnapshots ? 'rotated' : ''} />
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
                  <span>{currentUser.display_name || currentUser.username}</span>
                  <small className="userspace-muted">owner</small>
                </div>
                {pendingMembers.map((member) => {
                  const user = allUsers.find((u) => u.id === member.user_id);
                  return (
                    <div key={member.user_id} className="userspace-member-row">
                      <span>{user?.display_name || user?.username || member.user_id}</span>
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
                        <option key={u.id} value={u.id}>{u.display_name || u.username}</option>
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
