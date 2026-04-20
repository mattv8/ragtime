import { useState, useCallback, type ReactNode } from 'react';
import { Check, Repeat, Trash2, X } from 'lucide-react';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import type { User, UserSpaceWorkspace } from '@/types';

interface WorkspaceRowListProps {
  workspaces: UserSpaceWorkspace[];
  users: User[];
  disabled?: boolean;
  deletingWorkspaceIds?: ReadonlySet<string>;
  renderMeta?: (ws: UserSpaceWorkspace) => ReactNode;
  onTransfer: (workspaceId: string, newOwnerId: string) => Promise<void>;
  onDelete: (workspaceId: string) => Promise<void>;
  onSelect?: (ws: UserSpaceWorkspace) => void;
  emptyMessage?: string;
}

export function WorkspaceRowList({
  workspaces,
  users,
  disabled = false,
  deletingWorkspaceIds,
  renderMeta,
  onTransfer,
  onDelete,
  onSelect,
  emptyMessage = 'No workspaces.',
}: WorkspaceRowListProps) {
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [transferringId, setTransferringId] = useState<string | null>(null);
  const [transferTargetUserId, setTransferTargetUserId] = useState<string | null>(null);
  const [transferSaving, setTransferSaving] = useState(false);

  const handleDelete = useCallback(async (workspaceId: string) => {
    setDeletingId(workspaceId);
    try {
      await onDelete(workspaceId);
      setDeleteConfirmId(null);
    } catch {
      /* parent handles error display */
    } finally {
      setDeletingId(null);
    }
  }, [onDelete]);

  const handleTransfer = useCallback(async (workspaceId: string, newOwnerId: string) => {
    setTransferSaving(true);
    try {
      await onTransfer(workspaceId, newOwnerId);
      setTransferringId(null);
      setTransferTargetUserId(null);
    } catch {
      /* parent handles error display */
    } finally {
      setTransferSaving(false);
    }
  }, [onTransfer]);

  if (workspaces.length === 0) {
    return <p className="muted-text" style={{ margin: 0 }}>{emptyMessage}</p>;
  }

  const busy = disabled || Boolean(deletingId) || transferSaving;

  return (
    <div className="admin-ws-group-list">
      {workspaces.map((ws) => {
        const isConfirming = deleteConfirmId === ws.id;
        const isDeleting = deletingId === ws.id || deletingWorkspaceIds?.has(ws.id) === true;
        const isTransferring = transferringId === ws.id;
        const rowBusy = disabled || transferSaving || isDeleting;

        const nameContent = (
          <>
            <span className="admin-ws-item-name">{ws.name}</span>
            {renderMeta?.(ws)}
          </>
        );

        return (
          <div key={ws.id} className="admin-ws-item-wrapper">
            <div className="admin-ws-item">
              {onSelect ? (
                <button
                  type="button"
                  className="admin-ws-item-select"
                  disabled={rowBusy}
                  onClick={() => onSelect(ws)}
                  title={`Open workspace: ${ws.name}`}
                >
                  {nameContent}
                </button>
              ) : (
                <div className="admin-ws-item-select" style={{ cursor: 'default' }}>
                  {nameContent}
                </div>
              )}
              <div className="admin-ws-item-actions">
                <button
                  type="button"
                  className="chat-action-btn"
                  disabled={rowBusy}
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
                      disabled={busy}
                      onClick={(e) => { e.stopPropagation(); void handleDelete(ws.id); }}
                      title="Confirm delete"
                    >
                      {isDeleting ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
                    </button>
                    <button
                      type="button"
                      className="chat-action-btn cancel-delete"
                      disabled={busy}
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
                    disabled={busy || isDeleting}
                    onClick={(e) => { e.stopPropagation(); setDeleteConfirmId(ws.id); }}
                    title="Delete workspace"
                  >
                    {isDeleting ? <MiniLoadingSpinner variant="icon" size={12} /> : <Trash2 size={12} />}
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
                  {users
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
  );
}

export default WorkspaceRowList;
