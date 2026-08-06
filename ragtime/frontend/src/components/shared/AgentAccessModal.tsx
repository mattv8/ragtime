import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import type {
  UpsertWorkspaceAgentGrantRequest,
  UserSpaceWorkspace,
  WorkspaceAgentGrant,
  WorkspaceAgentGrantMode,
  WorkspaceSqliteGrantMode,
} from '@/types';

type AgentAccessWorkspaceOption = Pick<UserSpaceWorkspace, 'id' | 'name'> & {
  canGrantReadWrite?: boolean;
  canGrantSqlite?: boolean;
};

interface AgentAccessModalProps {
  isOpen: boolean;
  onClose: () => void;
  sourceWorkspace: Pick<UserSpaceWorkspace, 'id' | 'name'>;
  availableWorkspaces: AgentAccessWorkspaceOption[];
  grants: WorkspaceAgentGrant[];
  onUpsert: (request: UpsertWorkspaceAgentGrantRequest) => Promise<void>;
  onRevoke: (targetWorkspaceId: string) => Promise<void>;
  loading?: boolean;
  savingTargetId?: string | null;
  revokingTargetId?: string | null;
}

function sortWorkspacesByName(
  workspaces: AgentAccessWorkspaceOption[],
): AgentAccessWorkspaceOption[] {
  return [...workspaces].sort((left, right) => left.name.localeCompare(right.name));
}

function grantModeLabel(mode: WorkspaceAgentGrantMode): string {
  return mode === 'read_write' ? 'Read / Write' : 'Read Only';
}

function sqliteModeLabel(mode: WorkspaceSqliteGrantMode): string {
  if (mode === 'read_write') return 'Read / Write';
  if (mode === 'read') return 'Read';
  return 'None';
}

export function AgentAccessModal({
  isOpen,
  onClose,
  sourceWorkspace,
  availableWorkspaces,
  grants,
  onUpsert,
  onRevoke,
  loading = false,
  savingTargetId = null,
  revokingTargetId = null,
}: AgentAccessModalProps) {
  const [targetWorkspaceId, setTargetWorkspaceId] = useState('');
  const [accessMode, setAccessMode] = useState<WorkspaceAgentGrantMode>('read');
  const [sqliteAccessMode, setSqliteAccessMode] = useState<WorkspaceSqliteGrantMode>('none');

  const workspaceOptions = useMemo(
    () =>
      sortWorkspacesByName(
        availableWorkspaces.filter((workspace) => workspace.id !== sourceWorkspace.id),
      ),
    [availableWorkspaces, sourceWorkspace.id],
  );
  const targetCanGrantReadWrite =
    workspaceOptions.find((workspace) => workspace.id === targetWorkspaceId)?.canGrantReadWrite ??
    false;
  const targetCanGrantSqlite =
    workspaceOptions.find((workspace) => workspace.id === targetWorkspaceId)?.canGrantSqlite ??
    false;

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const current = workspaceOptions[0]?.id ?? '';
    setTargetWorkspaceId((previous) => {
      if (previous && workspaceOptions.some((workspace) => workspace.id === previous)) {
        return previous;
      }
      return current;
    });
    setAccessMode('read');
    setSqliteAccessMode('none');
  }, [isOpen, workspaceOptions]);

  useEffect(() => {
    if (!targetCanGrantReadWrite && accessMode === 'read_write') {
      setAccessMode('read');
    }
  }, [accessMode, targetCanGrantReadWrite]);

  const handleClose = () => {
    if (!loading && !savingTargetId && !revokingTargetId) {
      onClose();
    }
  };

  const handleAddGrant = async () => {
    if (!targetWorkspaceId) {
      return;
    }
    const request: UpsertWorkspaceAgentGrantRequest = {
      target_workspace_id: targetWorkspaceId,
      access_mode: accessMode,
    };
    if (targetCanGrantSqlite) {
      request.sqlite_access_mode = sqliteAccessMode;
    }
    await onUpsert(request);
  };

  const handleTargetWorkspaceChange = (workspaceId: string) => {
    setTargetWorkspaceId(workspaceId);
    setSqliteAccessMode('none');
  };

  if (!isOpen) {
    return null;
  }

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div
        className="modal-content modal-small userspace-agent-access-modal"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3>Agent Access</h3>
          <button
            className="modal-close"
            onClick={handleClose}
            disabled={loading || Boolean(savingTargetId) || Boolean(revokingTargetId)}
          >
            &times;
          </button>
        </div>
        <div className="modal-body">
          <div className="userspace-agent-access-intro">
            <strong>{sourceWorkspace.name}</strong>
            <small className="userspace-muted">
              Allow this workspace&apos;s agent to use file/runtime tools in another workspace you
              can access.
            </small>
          </div>
          <p className="userspace-agent-access-note userspace-muted" role="note">
            A SQLite grant covers all data in app.sqlite3, including future data. Only the target
            workspace owner can change it.
          </p>

          {loading ? (
            <p className="userspace-muted">Loading agent grants...</p>
          ) : (
            <>
              <div className="userspace-members-list">
                {grants.length === 0 ? (
                  <div className="userspace-agent-access-empty">
                    No cross-workspace agent grants configured.
                  </div>
                ) : (
                  grants.map((grant) => {
                    const targetLabel =
                      grant.target_workspace_name?.trim() || grant.target_workspace_id;
                    const disabled =
                      savingTargetId === grant.target_workspace_id ||
                      revokingTargetId === grant.target_workspace_id;
                    const grantTargetCanReadWrite =
                      workspaceOptions.find(
                        (workspace) => workspace.id === grant.target_workspace_id,
                      )?.canGrantReadWrite ?? grant.access_mode === 'read_write';
                    const grantTargetCanGrantSqlite =
                      workspaceOptions.find(
                        (workspace) => workspace.id === grant.target_workspace_id,
                      )?.canGrantSqlite ?? false;
                    return (
                      <div
                        key={grant.target_workspace_id}
                        className="userspace-member-row userspace-agent-access-grant-row"
                      >
                        <div className="userspace-agent-access-grant-top">
                          <div className="userspace-agent-access-info">
                            <span>{targetLabel}</span>
                          </div>
                          <button
                            className="chat-action-btn"
                            onClick={() => void onRevoke(grant.target_workspace_id)}
                            title={`Revoke ${grantModeLabel(grant.access_mode).toLowerCase()} access`}
                            disabled={
                              disabled ||
                              (!grantTargetCanGrantSqlite && grant.sqlite_access_mode !== 'none')
                            }
                          >
                            <X size={14} />
                          </button>
                        </div>
                        <div className="userspace-agent-access-control-row">
                          <span className="userspace-agent-access-control-label">File/runtime</span>
                          <div
                            className="userspace-member-role-toggle userspace-agent-access-toggle"
                            role="group"
                            aria-label={`File/runtime access for ${targetLabel}`}
                          >
                            <button
                              type="button"
                              className={`userspace-member-role-option ${grant.access_mode === 'read' ? 'active' : ''}`}
                              aria-pressed={grant.access_mode === 'read'}
                              onClick={() =>
                                void onUpsert({
                                  target_workspace_id: grant.target_workspace_id,
                                  access_mode: 'read',
                                })
                              }
                              disabled={disabled}
                            >
                              Read
                            </button>
                            <button
                              type="button"
                              className={`userspace-member-role-option ${grant.access_mode === 'read_write' ? 'active' : ''}`}
                              aria-pressed={grant.access_mode === 'read_write'}
                              onClick={() =>
                                void onUpsert({
                                  target_workspace_id: grant.target_workspace_id,
                                  access_mode: 'read_write',
                                })
                              }
                              disabled={disabled || !grantTargetCanReadWrite}
                            >
                              Read / Write
                            </button>
                          </div>
                        </div>
                        <div className="userspace-agent-access-control-row">
                          <span className="userspace-agent-access-control-label">
                            Shared SQLite
                          </span>
                          <div
                            className="userspace-member-role-toggle userspace-agent-access-toggle"
                            role="group"
                            aria-label={`Shared SQLite access for ${targetLabel}`}
                          >
                            {(['none', 'read', 'read_write'] as const).map((mode) => (
                              <button
                                key={mode}
                                type="button"
                                className={`userspace-member-role-option ${grant.sqlite_access_mode === mode ? 'active' : ''}`}
                                aria-pressed={grant.sqlite_access_mode === mode}
                                onClick={() =>
                                  void onUpsert({
                                    target_workspace_id: grant.target_workspace_id,
                                    access_mode: grant.access_mode,
                                    sqlite_access_mode: mode,
                                  })
                                }
                                disabled={disabled || !grantTargetCanGrantSqlite}
                              >
                                {sqliteModeLabel(mode)}
                              </button>
                            ))}
                          </div>
                        </div>
                        <div className="userspace-agent-access-workspace-id">
                          <small className="userspace-muted">
                            workspace_id: {grant.target_workspace_id}
                          </small>
                        </div>
                      </div>
                    );
                  })
                )}
              </div>

              <div className="userspace-add-member userspace-agent-access-add-row">
                <label htmlFor="userspace-agent-target-select" className="userspace-share-label">
                  Target workspace
                </label>
                <select
                  id="userspace-agent-target-select"
                  value={targetWorkspaceId}
                  onChange={(event) => handleTargetWorkspaceChange(event.target.value)}
                  disabled={
                    workspaceOptions.length === 0 ||
                    Boolean(savingTargetId) ||
                    Boolean(revokingTargetId)
                  }
                >
                  {workspaceOptions.length === 0 ? (
                    <option value="">No other accessible workspaces</option>
                  ) : (
                    workspaceOptions.map((workspace) => (
                      <option key={workspace.id} value={workspace.id}>
                        {workspace.name}
                      </option>
                    ))
                  )}
                </select>

                <div
                  className="userspace-member-role-toggle userspace-agent-access-create-toggle"
                  role="group"
                  aria-label="New grant file/runtime access"
                >
                  <button
                    type="button"
                    className={`userspace-member-role-option ${accessMode === 'read' ? 'active' : ''}`}
                    aria-pressed={accessMode === 'read'}
                    onClick={() => setAccessMode('read')}
                    disabled={
                      workspaceOptions.length === 0 ||
                      Boolean(savingTargetId) ||
                      Boolean(revokingTargetId)
                    }
                  >
                    Read
                  </button>
                  <button
                    type="button"
                    className={`userspace-member-role-option ${accessMode === 'read_write' ? 'active' : ''}`}
                    aria-pressed={accessMode === 'read_write'}
                    onClick={() => setAccessMode('read_write')}
                    disabled={
                      workspaceOptions.length === 0 ||
                      !targetCanGrantReadWrite ||
                      Boolean(savingTargetId) ||
                      Boolean(revokingTargetId)
                    }
                  >
                    Read / Write
                  </button>
                </div>
                <div
                  className="userspace-member-role-toggle userspace-agent-access-create-toggle"
                  role="group"
                  aria-label="New grant Shared SQLite access"
                >
                  {(['none', 'read', 'read_write'] as const).map((mode) => (
                    <button
                      key={mode}
                      type="button"
                      className={`userspace-member-role-option ${sqliteAccessMode === mode ? 'active' : ''}`}
                      aria-pressed={sqliteAccessMode === mode}
                      onClick={() => setSqliteAccessMode(mode)}
                      disabled={
                        workspaceOptions.length === 0 ||
                        !targetCanGrantSqlite ||
                        Boolean(savingTargetId) ||
                        Boolean(revokingTargetId)
                      }
                    >
                      {sqliteModeLabel(mode)}
                    </button>
                  ))}
                </div>
                <small className="userspace-muted">
                  Read allows listing, reading, and screenshots. Read / Write also allows file edits
                  and terminal commands.
                </small>
              </div>
            </>
          )}
        </div>
        <div className="modal-footer">
          <button
            className="btn btn-secondary"
            onClick={handleClose}
            disabled={loading || Boolean(savingTargetId) || Boolean(revokingTargetId)}
          >
            Close
          </button>
          <button
            className="btn btn-primary"
            onClick={() => void handleAddGrant()}
            disabled={
              loading ||
              !targetWorkspaceId ||
              workspaceOptions.length === 0 ||
              Boolean(savingTargetId) ||
              Boolean(revokingTargetId)
            }
          >
            {savingTargetId === targetWorkspaceId ? 'Saving...' : 'Add / Update Grant'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default AgentAccessModal;
