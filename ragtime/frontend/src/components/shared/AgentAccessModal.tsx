import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import type { UserSpaceWorkspace, WorkspaceAgentGrant, WorkspaceAgentGrantMode } from '@/types';

interface AgentAccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  targetWorkspace: Pick<UserSpaceWorkspace, 'id' | 'name'>;
  availableSourceWorkspaces: Array<Pick<UserSpaceWorkspace, 'id' | 'name'>>;
  grants: WorkspaceAgentGrant[];
  onUpsert: (sourceWorkspaceId: string, accessMode: WorkspaceAgentGrantMode) => Promise<void>;
  onRevoke: (sourceWorkspaceId: string) => Promise<void>;
  canGrantReadWrite?: boolean;
  loading?: boolean;
  savingSourceId?: string | null;
  revokingSourceId?: string | null;
}

function sortWorkspacesByName(
  workspaces: Array<Pick<UserSpaceWorkspace, 'id' | 'name'>>,
): Array<Pick<UserSpaceWorkspace, 'id' | 'name'>> {
  return [...workspaces].sort((left, right) => left.name.localeCompare(right.name));
}

function grantModeLabel(mode: WorkspaceAgentGrantMode): string {
  return mode === 'read_write' ? 'Read / Write' : 'Read Only';
}

export function AgentAccessModal({
  isOpen,
  onClose,
  targetWorkspace,
  availableSourceWorkspaces,
  grants,
  onUpsert,
  onRevoke,
  canGrantReadWrite = false,
  loading = false,
  savingSourceId = null,
  revokingSourceId = null,
}: AgentAccessModalProps) {
  const [sourceWorkspaceId, setSourceWorkspaceId] = useState('');
  const [accessMode, setAccessMode] = useState<WorkspaceAgentGrantMode>('read');

  const workspaceOptions = useMemo(
    () => sortWorkspacesByName(availableSourceWorkspaces.filter((workspace) => workspace.id !== targetWorkspace.id)),
    [availableSourceWorkspaces, targetWorkspace.id],
  );

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const current = workspaceOptions[0]?.id ?? '';
    setSourceWorkspaceId((previous) => {
      if (previous && workspaceOptions.some((workspace) => workspace.id === previous)) {
        return previous;
      }
      return current;
    });
    setAccessMode('read');
  }, [isOpen, workspaceOptions]);

  useEffect(() => {
    if (!canGrantReadWrite && accessMode === 'read_write') {
      setAccessMode('read');
    }
  }, [accessMode, canGrantReadWrite]);

  const handleClose = () => {
    if (!loading && !savingSourceId && !revokingSourceId) {
      onClose();
    }
  };

  const handleAddGrant = async () => {
    if (!sourceWorkspaceId) {
      return;
    }
    await onUpsert(sourceWorkspaceId, accessMode);
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content modal-small userspace-agent-access-modal" onClick={(event) => event.stopPropagation()}>
        <div className="modal-header">
          <h3>Agent Access</h3>
          <button className="modal-close" onClick={handleClose} disabled={loading || Boolean(savingSourceId) || Boolean(revokingSourceId)}>&times;</button>
        </div>
        <div className="modal-body">
          <div className="userspace-agent-access-intro">
            <strong>{targetWorkspace.name}</strong>
            <small className="userspace-muted">Allow agents from your editable workspaces to use file/runtime tools in this workspace.</small>
          </div>

          {loading ? (
            <p className="userspace-muted">Loading agent grants...</p>
          ) : (
            <>
              <div className="userspace-members-list">
                {grants.length === 0 ? (
                  <div className="userspace-agent-access-empty">No cross-workspace agent grants configured.</div>
                ) : (
                  grants.map((grant) => {
                    const sourceLabel = grant.source_workspace_name?.trim() || grant.source_workspace_id;
                    const disabled = savingSourceId === grant.source_workspace_id || revokingSourceId === grant.source_workspace_id;
                    const canManageGrant = grant.access_mode === 'read' || canGrantReadWrite;
                    return (
                      <div key={grant.source_workspace_id} className="userspace-member-row userspace-agent-access-grant-row">
                        <div className="userspace-agent-access-grant-top">
                          <div className="userspace-agent-access-info">
                            <span>{sourceLabel}</span>
                          </div>
                          <div className="userspace-member-role-toggle" role="group" aria-label={`Access mode from ${sourceLabel}`}>
                            <button
                              type="button"
                              className={`userspace-member-role-option ${grant.access_mode === 'read' ? 'active' : ''}`}
                              onClick={() => void onUpsert(grant.source_workspace_id, 'read')}
                              disabled={disabled || !canManageGrant}
                            >
                              Read
                            </button>
                            <button
                              type="button"
                              className={`userspace-member-role-option ${grant.access_mode === 'read_write' ? 'active' : ''}`}
                              onClick={() => void onUpsert(grant.source_workspace_id, 'read_write')}
                              disabled={disabled || !canGrantReadWrite}
                            >
                              Read / Write
                            </button>
                          </div>
                          <button
                            className="chat-action-btn"
                            onClick={() => void onRevoke(grant.source_workspace_id)}
                            title={`Revoke ${grantModeLabel(grant.access_mode).toLowerCase()} access`}
                            disabled={disabled || !canManageGrant}
                          >
                            <X size={14} />
                          </button>
                        </div>
                        <div className="userspace-agent-access-workspace-id">
                          <small className="userspace-muted">source_workspace_id: {grant.source_workspace_id}</small>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              <div className="userspace-add-member userspace-agent-access-add-row">
                <label htmlFor="userspace-agent-source-select" className="userspace-share-label">Source workspace</label>
                <select
                  id="userspace-agent-source-select"
                  value={sourceWorkspaceId}
                  onChange={(event) => setSourceWorkspaceId(event.target.value)}
                  disabled={workspaceOptions.length === 0 || Boolean(savingSourceId) || Boolean(revokingSourceId)}
                >
                  {workspaceOptions.length === 0 ? (
                    <option value="">No editable source workspaces</option>
                  ) : (
                    workspaceOptions.map((workspace) => (
                      <option key={workspace.id} value={workspace.id}>{workspace.name}</option>
                    ))
                  )}
                </select>

                <div className="userspace-member-role-toggle userspace-agent-access-create-toggle" role="group" aria-label="New grant access mode">
                  <button
                    type="button"
                    className={`userspace-member-role-option ${accessMode === 'read' ? 'active' : ''}`}
                    onClick={() => setAccessMode('read')}
                    disabled={workspaceOptions.length === 0 || Boolean(savingSourceId) || Boolean(revokingSourceId)}
                  >
                    Read
                  </button>
                  <button
                    type="button"
                    className={`userspace-member-role-option ${accessMode === 'read_write' ? 'active' : ''}`}
                    onClick={() => setAccessMode('read_write')}
                    disabled={workspaceOptions.length === 0 || !canGrantReadWrite || Boolean(savingSourceId) || Boolean(revokingSourceId)}
                  >
                    Read / Write
                  </button>
                </div>
                <small className="userspace-muted">Read allows listing, reading, and screenshots. Read / Write also allows file edits and terminal commands.</small>
              </div>
            </>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={handleClose} disabled={loading || Boolean(savingSourceId) || Boolean(revokingSourceId)}>
            Close
          </button>
          <button
            className="btn btn-primary"
            onClick={() => void handleAddGrant()}
            disabled={loading || !sourceWorkspaceId || workspaceOptions.length === 0 || Boolean(savingSourceId) || Boolean(revokingSourceId)}
          >
            {savingSourceId === sourceWorkspaceId ? 'Saving...' : 'Add / Update Grant'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default AgentAccessModal;
