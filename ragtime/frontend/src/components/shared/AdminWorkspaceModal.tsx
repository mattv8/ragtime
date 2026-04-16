import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertCircle, ChevronDown, ChevronRight, X } from 'lucide-react';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import { WorkspaceRowList } from './WorkspaceRowList';
import { api } from '@/api';
import type { User, UserSpaceWorkspace } from '@/types';
import {
  clearInterruptDismiss,
  resolveWorkspaceInterruptStateFromSummary,
} from '@/utils';
import type { InterruptChatStateSnapshot } from '@/utils/cookies';

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
    try {
      await api.deleteUserSpaceWorkspace(workspaceId);
      setWorkspaces((prev) => prev.filter((ws) => ws.id !== workspaceId));
      setTotal((prev) => Math.max(0, prev - 1));
      onWorkspaceDeleted(workspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete workspace');
      throw err;
    }
  }, [onWorkspaceDeleted]);

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
