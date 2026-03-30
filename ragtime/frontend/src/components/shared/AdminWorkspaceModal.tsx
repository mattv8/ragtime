import { useCallback, useEffect, useMemo, useState } from 'react';
import { Check, ChevronDown, ChevronRight, Repeat, Trash2, X } from 'lucide-react';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import { api } from '@/api';
import type { User, UserSpaceWorkspace } from '@/types';

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
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [transferringId, setTransferringId] = useState<string | null>(null);
  const [transferTargetUserId, setTransferTargetUserId] = useState<string | null>(null);
  const [transferSaving, setTransferSaving] = useState(false);
  const [allUsers, setAllUsers] = useState<User[]>([]);

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
      setDeleteConfirmId(null);
      setDeletingId(null);
      setTransferringId(null);
      setTransferTargetUserId(null);
      setTransferSaving(false);
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
    setDeletingId(workspaceId);
    try {
      await api.deleteUserSpaceWorkspace(workspaceId);
      setWorkspaces((prev) => prev.filter((ws) => ws.id !== workspaceId));
      setTotal((prev) => Math.max(0, prev - 1));
      onWorkspaceDeleted(workspaceId);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete workspace');
    } finally {
      setDeletingId(null);
      setDeleteConfirmId(null);
    }
  }, [onWorkspaceDeleted]);

  const handleTransfer = useCallback(async (workspaceId: string, newOwnerId: string) => {
    setTransferSaving(true);
    try {
      const updated = await api.updateUserSpaceWorkspace(workspaceId, { owner_user_id: newOwnerId });
      setWorkspaces((prev) => prev.map((ws) => (ws.id === workspaceId ? updated : ws)));
      setTransferringId(null);
      setTransferTargetUserId(null);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to transfer ownership');
    } finally {
      setTransferSaving(false);
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
                      <div className="admin-ws-group-list">
                        {group.workspaces.map((ws) => {
                          const isOwn = ws.owner_user_id === currentUser.id;
                          const isConfirming = deleteConfirmId === ws.id;
                          const isDeleting = deletingId === ws.id;
                          const isTransferring = transferringId === ws.id;
                          return (
                            <div key={ws.id} className="admin-ws-item-wrapper">
                              <div className="admin-ws-item">
                                <button
                                  type="button"
                                  className="admin-ws-item-select"
                                  disabled={Boolean(deletingId) || transferSaving}
                                  onClick={() => handleSelect(ws)}
                                  title={`Open workspace: ${ws.name}`}
                                >
                                  <span className="admin-ws-item-name">{ws.name}</span>
                                  {isOwn && <span className="admin-ws-badge-own">You</span>}
                                  <span className="admin-ws-item-date">
                                    {new Date(ws.updated_at).toLocaleDateString()}
                                  </span>
                                </button>
                                <div className="admin-ws-item-actions">
                                  <button
                                    type="button"
                                    className="chat-action-btn"
                                    disabled={Boolean(deletingId) || transferSaving}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (isTransferring) {
                                        setTransferringId(null);
                                        setTransferTargetUserId(null);
                                      } else {
                                        setTransferringId(ws.id);
                                        setTransferTargetUserId(null);
                                        setDeleteConfirmId(null);
                                      }
                                    }}
                                    title="Transfer ownership"
                                  >
                                    <Repeat size={12} />
                                  </button>
                                  {isConfirming ? (
                                  <>
                                    <button
                                      type="button"
                                      className="chat-action-btn confirm-delete"
                                      disabled={Boolean(deletingId)}
                                      onClick={(e) => { e.stopPropagation(); void handleDelete(ws.id); }}
                                      title="Confirm delete"
                                    >
                                      {isDeleting ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                                    </button>
                                    <button
                                      type="button"
                                      className="chat-action-btn cancel-delete"
                                      disabled={Boolean(deletingId)}
                                      onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(null); }}
                                      title="Cancel"
                                    >
                                      <X size={12} />
                                    </button>
                                  </>
                                ) : (
                                  <button
                                    type="button"
                                    className="chat-action-btn"
                                    disabled={Boolean(deletingId)}
                                    onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(ws.id); }}
                                    title="Delete workspace"
                                  >
                                    <Trash2 size={12} />
                                  </button>
                                )}
                              </div>
                              </div>
                              {isTransferring && (
                                <div className="admin-ws-transfer-row">
                                  <select
                                    className="admin-ws-transfer-select"
                                    value={transferTargetUserId ?? ''}
                                    onChange={(e) => setTransferTargetUserId(e.target.value || null)}
                                    disabled={transferSaving}
                                  >
                                    <option value="">Select new owner...</option>
                                    {allUsers
                                      .filter((u) => u.id !== ws.owner_user_id)
                                      .map((u) => (
                                        <option key={u.id} value={u.id}>
                                          {u.display_name && u.display_name !== u.username
                                            ? `${u.display_name} (@${u.username})`
                                            : `@${u.username}`}
                                        </option>
                                      ))}
                                  </select>
                                  <button
                                    type="button"
                                    className="btn btn-primary btn-sm"
                                    disabled={!transferTargetUserId || transferSaving}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      if (transferTargetUserId) void handleTransfer(ws.id, transferTargetUserId);
                                    }}
                                  >
                                    {transferSaving ? <MiniLoadingSpinner variant="icon" size={12} /> : 'Transfer'}
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    disabled={transferSaving}
                                    onClick={(e) => { e.stopPropagation(); setTransferringId(null); setTransferTargetUserId(null); }}
                                  >
                                    Cancel
                                  </button>
                                </div>
                              )}
                            </div>
                          );
                        })}
                      </div>
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
