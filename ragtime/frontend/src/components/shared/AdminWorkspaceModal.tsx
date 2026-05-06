import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, ChevronDown, ChevronRight, X } from 'lucide-react';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import { WorkspaceRowList } from './WorkspaceRowList';
import { api, ApiError } from '@/api';
import type {
  User,
  UserSpaceWorkspace,
  UserSpaceWorkspaceDeleteTask,
  UserSpaceWorkspaceDeleteTaskPhase,
} from '@/types';
import {
  clearInterruptDismiss,
  resolveWorkspaceInterruptStateFromSummary,
} from '@/utils';
import type { InterruptChatStateSnapshot } from '@/utils/cookies';

const WORKSPACE_DELETE_TASK_POLL_INTERVAL_MS = 1000;

interface AdminWorkspaceModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: User;
  /** Called when admin selects a workspace to switch to */
  onSelectWorkspace: (workspace: UserSpaceWorkspace) => void;
  /** Called after admin deletes a workspace (so parent can refresh its own list) */
  onWorkspaceDeleted: (workspaceId: string) => void;
}

interface OwnerGroup {
  key: string;
  label: string;
  workspaces: UserSpaceWorkspace[];
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

export function AdminWorkspaceModal({
  isOpen,
  onClose,
  currentUser,
  onSelectWorkspace,
  onWorkspaceDeleted,
}: AdminWorkspaceModalProps) {
  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const [allUsers, setAllUsers] = useState<User[]>([]);
  const [workspaceChatStates, setWorkspaceChatStates] = useState<Record<string, { hasLive: boolean; hasInterrupted: boolean }>>({});
  const [deletingWorkspaceTasks, setDeletingWorkspaceTasks] = useState<Record<string, UserSpaceWorkspaceDeleteTask>>({});
  const deletingWorkspaceTasksRef = useRef<Record<string, UserSpaceWorkspaceDeleteTask>>({});
  // Track previous raw interrupted state per workspace so we can detect
  // false -> true transitions and clear stale dismiss cookies.
  const prevChatStateRef = useRef<Record<string, InterruptChatStateSnapshot>>({});

  useEffect(() => {
    if (!isOpen || workspaces.length === 0) return;

    let cancelled = false;
    let isPolling = false;

    const pollWorkspaceConversationStates = async () => {
      if (isPolling) return;
      isPolling = true;
      try {
        const workspaceIds = workspaces.map((workspace) => workspace.id);
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
    // Poll less frequently in admin modal due to the potentially high number of workspaces
    const timer = window.setInterval(() => {
      void pollWorkspaceConversationStates();
    }, 5000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [isOpen, workspaces, currentUser.id]);

  const loadWorkspaces = useCallback(async (append = false) => {
    if (append) {
      setLoadingMore(true);
    } else {
      setLoading(true);
    }
    try {
      const offset = append ? workspaces.length : 0;
      const page = await api.listUserSpaceWorkspaces(offset, 50, true);
      if (append) {
        setWorkspaces((prev) => [...prev, ...page.items]);
      } else {
        setWorkspaces(page.items);
      }
      setTotal(page.total);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workspaces');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [workspaces.length]);

  useEffect(() => {
    if (isOpen) {
      void loadWorkspaces();
      void api.listUsers().then(setAllUsers).catch(() => {});
    }
  }, [isOpen]); // eslint-disable-line react-hooks/exhaustive-deps

  const activeWorkspaceDeleteTasks = useMemo(
    () => Object.values(deletingWorkspaceTasks)
      .filter((task) => !isWorkspaceDeleteTaskTerminal(task.phase))
      .sort((left, right) => Date.parse(left.queued_at) - Date.parse(right.queued_at)),
    [deletingWorkspaceTasks],
  );

  const deletingWorkspaceIds = useMemo(
    () => new Set(activeWorkspaceDeleteTasks.map((task) => task.workspace_id)),
    [activeWorkspaceDeleteTasks],
  );

  const deletingWorkspaceStatus = useMemo(
    () => formatWorkspaceDeleteTasksStatus(activeWorkspaceDeleteTasks),
    [activeWorkspaceDeleteTasks],
  );
  const activeWorkspaceDeleteTaskCount = activeWorkspaceDeleteTasks.length;

  const groupedWorkspaces = useMemo<OwnerGroup[]>(() => {
    const groups = workspaces.reduce<Record<string, { label: string; workspaces: UserSpaceWorkspace[] }>>((acc, ws) => {
      const key = ws.owner_username || ws.owner_user_id;
      const label = ws.owner_display_name || ws.owner_username || ws.owner_user_id;
      const existing = acc[key];
      acc[key] = existing
        ? { label: existing.label || label, workspaces: [...existing.workspaces, ws] }
        : { label, workspaces: [ws] };
      return acc;
    }, {});

    return Object.entries(groups)
      .map(([key, value]) => ({ key, label: value.label, workspaces: value.workspaces }))
      .sort((a, b) => a.label.localeCompare(b.label));
  }, [workspaces]);

  // Auto-collapse newly discovered groups
  useEffect(() => {
    setCollapsedGroups((prev) => {
      const next = { ...prev };
      let changed = false;
      for (const group of groupedWorkspaces) {
        if (!(group.key in next)) {
          next[group.key] = true;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [groupedWorkspaces]);

  const toggleGroup = useCallback((key: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  const handleDelete = useCallback(async (workspaceId: string) => {
    setError(null);
    try {
      const task = await api.queueUserSpaceWorkspaceDelete(workspaceId);
      setDeletingWorkspaceTasks((current) => ({
        ...current,
        [workspaceId]: task,
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to queue workspace delete');
      throw err;
    }
  }, []);

  useEffect(() => {
    deletingWorkspaceTasksRef.current = deletingWorkspaceTasks;
  }, [deletingWorkspaceTasks]);

  useEffect(() => {
    if (activeWorkspaceDeleteTaskCount === 0) {
      return;
    }

    let cancelled = false;
    let pollInFlight = false;

    const pollDeleteTasks = async () => {
      if (pollInFlight) {
        return;
      }
      const tasks = Object.values(deletingWorkspaceTasksRef.current).filter(
        (task) => !isWorkspaceDeleteTaskTerminal(task.phase),
      );
      if (tasks.length === 0) {
        return;
      }
      pollInFlight = true;

      try {
        const results = await Promise.all(tasks.map(async (task) => {
          try {
            const status = await api.getUserSpaceWorkspaceDeleteTask(task.task_id);
            return { task, status, error: null as Error | null };
          } catch (pollError) {
            return { task, status: null as UserSpaceWorkspaceDeleteTask | null, error: pollError as Error };
          }
        }));

        if (cancelled) {
          return;
        }

        const completedWorkspaceIds = new Set<string>();
        const terminalWorkspaceIds = new Set<string>();
        const updatedTasks: Record<string, UserSpaceWorkspaceDeleteTask> = {};
        let nextError: string | null = null;

        for (const result of results) {
          if (result.status) {
            if (isWorkspaceDeleteTaskTerminal(result.status.phase)) {
              terminalWorkspaceIds.add(result.status.workspace_id);
              if (result.status.phase === 'completed') {
                completedWorkspaceIds.add(result.status.workspace_id);
              } else if (!nextError) {
                nextError = result.status.error?.trim() || `Failed to delete ${result.status.workspace_name}`;
              }
            } else {
              updatedTasks[result.status.workspace_id] = result.status;
            }
            continue;
          }

          if (result.error instanceof ApiError && result.error.status === 404) {
            terminalWorkspaceIds.add(result.task.workspace_id);
            completedWorkspaceIds.add(result.task.workspace_id);
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

        if (completedWorkspaceIds.size > 0) {
          setWorkspaces((current) => current.filter((workspace) => !completedWorkspaceIds.has(workspace.id)));
          setTotal((current) => Math.max(0, current - completedWorkspaceIds.size));
          setWorkspaceChatStates((current) => {
            const next = { ...current };
            for (const workspaceId of completedWorkspaceIds) {
              delete next[workspaceId];
            }
            return next;
          });
          for (const workspaceId of completedWorkspaceIds) {
            delete prevChatStateRef.current[workspaceId];
            onWorkspaceDeleted(workspaceId);
          }
        }

        if (nextError) {
          setError(nextError);
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollDeleteTasks();
    const intervalId = window.setInterval(() => {
      void pollDeleteTasks();
    }, WORKSPACE_DELETE_TASK_POLL_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [activeWorkspaceDeleteTaskCount, onWorkspaceDeleted]);

  const handleTransfer = useCallback(async (workspaceId: string, newOwnerId: string) => {
    try {
      const updated = await api.updateUserSpaceWorkspace(workspaceId, { owner_user_id: newOwnerId });
      setWorkspaces((prev) => prev.map((ws) => (ws.id === workspaceId ? updated : ws)));
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to transfer ownership');
      throw err;
    }
  }, []);

  const handleSelect = useCallback((ws: UserSpaceWorkspace) => {
    onSelectWorkspace(ws);
    onClose();
  }, [onSelectWorkspace, onClose]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>All Workspaces</h3>
          <button className="modal-close" onClick={onClose} title="Close">
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          {error && (
            <div className="admin-ws-error">{error}</div>
          )}
          {deletingWorkspaceStatus && (
            <div className="admin-ws-loading">
              <MiniLoadingSpinner variant="icon" size={16} />
              <span>{deletingWorkspaceStatus}</span>
            </div>
          )}

          {loading ? (
            <div className="admin-ws-loading">
              <MiniLoadingSpinner variant="icon" size={20} />
              <span>Loading workspaces...</span>
            </div>
          ) : workspaces.length === 0 ? (
            <div className="admin-ws-empty">No workspaces found.</div>
          ) : (
            <div className="admin-ws-groups">
              {groupedWorkspaces.map((group) => {
                const isCollapsed = collapsedGroups[group.key] ?? true;
                return (
                  <div key={group.key} className="admin-ws-group">
                    <button className="admin-ws-group-header" onClick={() => toggleGroup(group.key)}>
                      {isCollapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
                      <span className="admin-ws-group-name">{group.label}</span>
                      <span className="admin-ws-group-count">{group.workspaces.length}</span>
                    </button>
                    {!isCollapsed && (
                      <WorkspaceRowList
                        workspaces={group.workspaces}
                        users={allUsers}
                        deletingWorkspaceIds={deletingWorkspaceIds}
                        onTransfer={handleTransfer}
                        onDelete={handleDelete}
                        onSelect={handleSelect}
                        renderMeta={(ws) => (
                          <>
                            {ws.owner_user_id === currentUser.id && <span className="admin-ws-badge-own">You</span>}
                            {workspaceChatStates[ws.id]?.hasLive && (
                              <MiniLoadingSpinner variant="icon" size={14} title="Chat in progress" />
                            )}
                            {!workspaceChatStates[ws.id]?.hasLive && workspaceChatStates[ws.id]?.hasInterrupted && (
                              <span className="userspace-workspace-item-state is-interrupted" title="A conversation was interrupted">
                                <AlertCircle size={13} />
                              </span>
                            )}
                            <span className="admin-ws-item-date">
                              {new Date(ws.updated_at).toLocaleDateString()}
                            </span>
                          </>
                        )}
                      />
                    )}
                  </div>
                );
              })}

              {workspaces.length < total && (
                <div className="admin-ws-load-more">
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => void loadWorkspaces(true)}
                    disabled={loadingMore}
                  >
                    {loadingMore ? 'Loading...' : `Load more (${workspaces.length} of ${total})`}
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AdminWorkspaceModal;
