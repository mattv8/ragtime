import { useEffect, useMemo, useRef, useState } from 'react';
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

type AgentAccessDraft = {
  accessMode: WorkspaceAgentGrantMode;
  sqliteAccessMode: WorkspaceSqliteGrantMode;
};

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

function buildGrantDraft(grant: WorkspaceAgentGrant): AgentAccessDraft {
  return {
    accessMode: grant.access_mode,
    sqliteAccessMode: grant.sqlite_access_mode,
  };
}

function grantDraftSignature(grant: WorkspaceAgentGrant): string {
  return `${grant.access_mode}:${grant.sqlite_access_mode}`;
}

function isDraftDirty(grant: WorkspaceAgentGrant, draft: AgentAccessDraft): boolean {
  return (
    grant.access_mode !== draft.accessMode || grant.sqlite_access_mode !== draft.sqliteAccessMode
  );
}

function buildUpsertRequest(
  targetWorkspaceId: string,
  draft: AgentAccessDraft,
  canGrantSqlite: boolean,
): UpsertWorkspaceAgentGrantRequest {
  const request: UpsertWorkspaceAgentGrantRequest = {
    target_workspace_id: targetWorkspaceId,
    access_mode: draft.accessMode,
  };
  if (canGrantSqlite) {
    request.sqlite_access_mode = draft.sqliteAccessMode;
  }
  return request;
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
  const [draftsByTargetId, setDraftsByTargetId] = useState<Record<string, AgentAccessDraft>>({});
  const [rowErrors, setRowErrors] = useState<Record<string, string>>({});
  const [createError, setCreateError] = useState<string | null>(null);
  const persistedGrantSignaturesRef = useRef<Record<string, string>>({});

  const workspaceOptions = useMemo(
    () =>
      sortWorkspacesByName(
        availableWorkspaces.filter((workspace) => workspace.id !== sourceWorkspace.id),
      ),
    [availableWorkspaces, sourceWorkspace.id],
  );
  const workspaceById = useMemo(
    () => new Map(workspaceOptions.map((workspace) => [workspace.id, workspace])),
    [workspaceOptions],
  );
  const grantedWorkspaceIds = useMemo(
    () => new Set(grants.map((grant) => grant.target_workspace_id)),
    [grants],
  );
  const isOperationLocked = loading || Boolean(savingTargetId) || Boolean(revokingTargetId);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setTargetWorkspaceId('');
    setCreateError(null);
  }, [isOpen]);

  useEffect(() => {
    const previousPersistedSignatures = persistedGrantSignaturesRef.current;
    const nextPersistedSignatures: Record<string, string> = {};

    for (const grant of grants) {
      nextPersistedSignatures[grant.target_workspace_id] = grantDraftSignature(grant);
    }

    setDraftsByTargetId((previous) => {
      const next: Record<string, AgentAccessDraft> = {};
      for (const grant of grants) {
        const targetId = grant.target_workspace_id;
        next[targetId] =
          previousPersistedSignatures[targetId] === nextPersistedSignatures[targetId]
            ? (previous[targetId] ?? buildGrantDraft(grant))
            : buildGrantDraft(grant);
      }
      return next;
    });
    persistedGrantSignaturesRef.current = nextPersistedSignatures;
    setRowErrors((previous) => {
      const next: Record<string, string> = {};
      for (const grant of grants) {
        const error = previous[grant.target_workspace_id];
        if (error) {
          next[grant.target_workspace_id] = error;
        }
      }
      return next;
    });
  }, [grants]);

  const handleClose = () => {
    if (!isOperationLocked) {
      onClose();
    }
  };

  const handleAddGrant = async () => {
    if (!targetWorkspaceId || grantedWorkspaceIds.has(targetWorkspaceId)) {
      return;
    }

    setCreateError(null);

    try {
      await onUpsert({
        target_workspace_id: targetWorkspaceId,
        access_mode: 'read',
      });
      setTargetWorkspaceId('');
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : 'Unable to add grant');
    }
  };

  const handleTargetWorkspaceChange = (workspaceId: string) => {
    setTargetWorkspaceId(workspaceId);
    setCreateError(null);
  };

  const handleDraftChange = (targetWorkspaceId: string, update: Partial<AgentAccessDraft>) => {
    setDraftsByTargetId((previous) => {
      const current = previous[targetWorkspaceId];
      if (!current) {
        return previous;
      }
      return {
        ...previous,
        [targetWorkspaceId]: {
          ...current,
          ...update,
        },
      };
    });
    setRowErrors((previous) => {
      if (!previous[targetWorkspaceId]) {
        return previous;
      }
      const next = { ...previous };
      delete next[targetWorkspaceId];
      return next;
    });
  };

  const handleSaveGrant = async (
    grant: WorkspaceAgentGrant,
    draft: AgentAccessDraft,
    canGrantSqlite: boolean,
  ) => {
    setRowErrors((previous) => {
      if (!previous[grant.target_workspace_id]) {
        return previous;
      }
      const next = { ...previous };
      delete next[grant.target_workspace_id];
      return next;
    });

    try {
      await onUpsert(buildUpsertRequest(grant.target_workspace_id, draft, canGrantSqlite));
    } catch (error) {
      setRowErrors((previous) => ({
        ...previous,
        [grant.target_workspace_id]:
          error instanceof Error ? error.message : 'Unable to save grant',
      }));
    }
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
          <button className="modal-close" onClick={handleClose} disabled={isOperationLocked}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          <div className="userspace-agent-access-intro">
            <strong>{sourceWorkspace.name}</strong>
            <small className="userspace-agent-access-note userspace-muted" role="note">
              Allow this workspace&apos;s agent to use file/runtime tools in another workspace you
              can access. A SQLite grant covers all data in app.sqlite3, including future data. Only
              the target workspace owner can change it.
            </small>
          </div>

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
                    const rowOperationActive =
                      savingTargetId === grant.target_workspace_id ||
                      revokingTargetId === grant.target_workspace_id;
                    const grantWorkspace = workspaceById.get(grant.target_workspace_id);
                    const canGrantReadWrite =
                      grantWorkspace?.canGrantReadWrite ?? grant.access_mode === 'read_write';
                    const canGrantSqlite = grantWorkspace?.canGrantSqlite ?? false;
                    const draft =
                      draftsByTargetId[grant.target_workspace_id] ?? buildGrantDraft(grant);
                    const isDirty = isDraftDirty(grant, draft);
                    const isInvalid =
                      (draft.accessMode === 'read_write' &&
                        !canGrantReadWrite &&
                        grant.access_mode !== 'read_write') ||
                      (draft.sqliteAccessMode !== grant.sqlite_access_mode && !canGrantSqlite);

                    return (
                      <div
                        key={grant.target_workspace_id}
                        className="userspace-member-row userspace-agent-access-grant-row"
                      >
                        <div className="userspace-agent-access-grant-top">
                          <div className="userspace-agent-access-info">
                            <span>{targetLabel}</span>
                          </div>
                          <div className="userspace-agent-access-grant-actions">
                            <button
                              type="button"
                              className="btn btn-secondary btn-sm"
                              aria-label={`Save grant for ${targetLabel}`}
                              onClick={() => void handleSaveGrant(grant, draft, canGrantSqlite)}
                              disabled={
                                isOperationLocked || rowOperationActive || !isDirty || isInvalid
                              }
                            >
                              {savingTargetId === grant.target_workspace_id ? 'Saving...' : 'Save'}
                            </button>
                            <button
                              className="chat-action-btn"
                              aria-label={`Revoke ${grantModeLabel(grant.access_mode).toLowerCase()} access`}
                              onClick={() => void onRevoke(grant.target_workspace_id)}
                              title={`Revoke ${grantModeLabel(grant.access_mode).toLowerCase()} access`}
                              disabled={
                                rowOperationActive ||
                                (!canGrantSqlite && grant.sqlite_access_mode !== 'none')
                              }
                            >
                              <X size={14} />
                            </button>
                          </div>
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
                              className={`userspace-member-role-option ${draft.accessMode === 'read' ? 'active' : ''}`}
                              aria-pressed={draft.accessMode === 'read'}
                              onClick={() =>
                                handleDraftChange(grant.target_workspace_id, {
                                  accessMode: 'read',
                                })
                              }
                              disabled={isOperationLocked || rowOperationActive}
                            >
                              Read
                            </button>
                            <button
                              type="button"
                              className={`userspace-member-role-option ${draft.accessMode === 'read_write' ? 'active' : ''}`}
                              aria-pressed={draft.accessMode === 'read_write'}
                              onClick={() =>
                                handleDraftChange(grant.target_workspace_id, {
                                  accessMode: 'read_write',
                                })
                              }
                              disabled={
                                isOperationLocked || rowOperationActive || !canGrantReadWrite
                              }
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
                                className={`userspace-member-role-option ${draft.sqliteAccessMode === mode ? 'active' : ''}`}
                                aria-pressed={draft.sqliteAccessMode === mode}
                                onClick={() =>
                                  handleDraftChange(grant.target_workspace_id, {
                                    sqliteAccessMode: mode,
                                  })
                                }
                                disabled={
                                  isOperationLocked || rowOperationActive || !canGrantSqlite
                                }
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
                        {rowErrors[grant.target_workspace_id] ? (
                          <div className="userspace-agent-access-row-error">
                            {rowErrors[grant.target_workspace_id]}
                          </div>
                        ) : null}
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
                  disabled={workspaceOptions.length === 0 || isOperationLocked}
                >
                  {workspaceOptions.length === 0 ? (
                    <option value="">No other accessible workspaces</option>
                  ) : (
                    <>
                      <option value="">Select a workspace...</option>
                      {workspaceOptions.map((workspace) => (
                        <option
                          key={workspace.id}
                          value={workspace.id}
                          disabled={grantedWorkspaceIds.has(workspace.id)}
                        >
                          {workspace.name}
                          {grantedWorkspaceIds.has(workspace.id) ? ' (already granted)' : ''}
                        </option>
                      ))}
                    </>
                  )}
                </select>
                {createError ? (
                  <div className="userspace-agent-access-row-error">{createError}</div>
                ) : null}
              </div>
            </>
          )}
        </div>
        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={handleClose} disabled={isOperationLocked}>
            Close
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void handleAddGrant()}
            disabled={
              isOperationLocked ||
              !targetWorkspaceId ||
              workspaceOptions.length === 0 ||
              grantedWorkspaceIds.has(targetWorkspaceId)
            }
          >
            {savingTargetId === targetWorkspaceId ? 'Adding...' : 'Add grant'}
          </button>
        </div>
      </div>
    </div>
  );
}

export default AgentAccessModal;
