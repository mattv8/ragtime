import { useEffect, useRef, useState } from 'react';

import { api } from '@/api';
import type { WorkspaceAgentAccessStatus } from '@/types';

import { InlineCopyButton } from './InlineCopyButton';

interface AgentAccessSectionProps {
  workspaceId: string;
}

function buildAgentInstructions(agentUrl: string): string {
  return [
    'Use this Ragtime workspace access to collaborate with me in this workspace.',
    '',
    'Open and follow the workspace agent manifest at:',
    agentUrl,
    '',
    "Read the /context endpoint first, inspect only the relevant workspace files, and use the documented task and reply endpoints to interact with Ragtime's builder chat.",
    '',
    'Treat this bearer URL as a secret. Do not repeat it in output, generated files, summaries, or external logs.',
  ].join('\n');
}

export function AgentAccessSection({ workspaceId }: AgentAccessSectionProps) {
  const [state, setState] = useState<{
    workspaceId: string;
    status: WorkspaceAgentAccessStatus | null;
    loading: boolean;
    saving: boolean;
    error: string | null;
  }>({ workspaceId, status: null, loading: true, saving: false, error: null });
  const currentWorkspaceIdRef = useRef(workspaceId);
  const loadGenerationRef = useRef(0);
  const actionGenerationRef = useRef(0);

  currentWorkspaceIdRef.current = workspaceId;

  const status = state.workspaceId === workspaceId ? state.status : null;
  const loading = state.workspaceId === workspaceId ? state.loading : true;
  const saving = state.workspaceId === workspaceId ? state.saving : false;
  const error = state.workspaceId === workspaceId ? state.error : null;

  useEffect(() => {
    currentWorkspaceIdRef.current = workspaceId;
    loadGenerationRef.current += 1;
    actionGenerationRef.current += 1;
    const loadGeneration = loadGenerationRef.current;

    setState({ workspaceId, status: null, loading: true, saving: false, error: null });

    void api
      .getWorkspaceAgentAccess(workspaceId)
      .then((nextStatus) => {
        if (
          currentWorkspaceIdRef.current === workspaceId &&
          loadGenerationRef.current === loadGeneration
        ) {
          setState((currentState) => ({
            ...currentState,
            workspaceId,
            status: nextStatus,
            error: null,
          }));
        }
      })
      .catch((loadError: unknown) => {
        if (
          currentWorkspaceIdRef.current === workspaceId &&
          loadGenerationRef.current === loadGeneration
        ) {
          setState((currentState) => ({
            ...currentState,
            workspaceId,
            error: loadError instanceof Error ? loadError.message : 'Failed to load agent access',
          }));
        }
      })
      .finally(() => {
        if (
          currentWorkspaceIdRef.current === workspaceId &&
          loadGenerationRef.current === loadGeneration
        ) {
          setState((currentState) => ({
            ...currentState,
            workspaceId,
            loading: false,
          }));
        }
      });
  }, [workspaceId]);

  const runAction = async (action: () => Promise<WorkspaceAgentAccessStatus>) => {
    const actionWorkspaceId = workspaceId;
    currentWorkspaceIdRef.current = actionWorkspaceId;
    actionGenerationRef.current += 1;
    const actionGeneration = actionGenerationRef.current;

    setState((currentState) => ({
      ...currentState,
      workspaceId: actionWorkspaceId,
      saving: true,
      error: null,
    }));

    try {
      const nextStatus = await action();
      if (
        currentWorkspaceIdRef.current === actionWorkspaceId &&
        actionGenerationRef.current === actionGeneration
      ) {
        setState((currentState) => ({
          ...currentState,
          workspaceId: actionWorkspaceId,
          status: nextStatus,
          error: null,
        }));
      }
    } catch (actionError) {
      if (
        currentWorkspaceIdRef.current === actionWorkspaceId &&
        actionGenerationRef.current === actionGeneration
      ) {
        setState((currentState) => ({
          ...currentState,
          workspaceId: actionWorkspaceId,
          error:
            actionError instanceof Error ? actionError.message : 'Failed to update agent access',
        }));
      }
    } finally {
      if (
        currentWorkspaceIdRef.current === actionWorkspaceId &&
        actionGenerationRef.current === actionGeneration
      ) {
        setState((currentState) => ({
          ...currentState,
          workspaceId: actionWorkspaceId,
          saving: false,
        }));
      }
    }
  };

  const enabled = status?.enabled === true && Boolean(status.agent_url);

  return (
    <section className="userspace-share-controls" aria-label="External agent access">
      <h4>External Agent Access</h4>
      <p className="userspace-muted">
        Allow a trusted external agent chat to collaborate in this workspace, review workspace
        context and files, and interact with Ragtime&apos;s builder chat on your behalf. Treat the
        access URL as a secret and revoke it when no longer needed.
      </p>

      {loading ? (
        <p className="userspace-muted">Loading agent access...</p>
      ) : error && !status ? (
        <>
          <p className="userspace-error" role="alert">
            {error}
          </p>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={saving}
            onClick={() => void runAction(() => api.getWorkspaceAgentAccess(workspaceId))}
          >
            Retry
          </button>
        </>
      ) : enabled && status?.agent_url ? (
        <>
          <div className="userspace-share-access-row userspace-share-link-pane">
            <label htmlFor="userspace-agent-url" className="userspace-share-label">
              Agent manifest URL
            </label>
            <div className="userspace-share-url-copy-wrap">
              <input id="userspace-agent-url" value={status.agent_url} readOnly />
              <InlineCopyButton
                copyText={status.agent_url}
                className="userspace-share-inline-copy"
                title="Copy agent manifest URL"
                ariaLabel="Copy agent manifest URL"
                copiedTitle="Agent manifest URL copied"
                copiedAriaLabel="Agent manifest URL copied"
                iconSize={12}
                disabled={saving}
              />
            </div>
          </div>

          <label className="chat-toggle-control">
            <span className="toggle-switch">
              <input
                type="checkbox"
                role="switch"
                aria-label="Allow external agents to submit build tasks"
                checked={status.allow_task_submission}
                disabled={saving}
                onChange={(event) => {
                  void runAction(() =>
                    api.enableWorkspaceAgentAccess(workspaceId, event.target.checked),
                  );
                }}
              />
              <span className="toggle-slider" />
            </span>
            <span>Allow external agents to submit build tasks</span>
          </label>

          <p className="userspace-muted">
            Copy the instructions below, then paste them into the trusted external agent chat you
            want to connect to this workspace.
          </p>

          {error && (
            <p className="userspace-error" role="alert">
              {error}
            </p>
          )}

          <div className="userspace-share-actions">
            <InlineCopyButton
              copyText={() => buildAgentInstructions(status.agent_url!)}
              className="btn btn-primary"
              title="Copy agent instructions"
              ariaLabel="Copy agent instructions"
              copiedTitle="Agent instructions copied"
              copiedAriaLabel="Agent instructions copied"
              label="Copy Instructions"
              copiedLabel="Copied"
              disabled={saving}
            />
            <button
              type="button"
              className="btn btn-secondary"
              disabled={saving}
              onClick={() => {
                void runAction(() => api.rotateWorkspaceAgentAccess(workspaceId));
              }}
            >
              Rotate Token
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={saving}
              onClick={() => {
                void runAction(() => api.disableWorkspaceAgentAccess(workspaceId));
              }}
            >
              Disable Agent Access
            </button>
          </div>
        </>
      ) : (
        <>
          {error && (
            <p className="userspace-error" role="alert">
              {error}
            </p>
          )}
          <button
            type="button"
            className="btn btn-secondary"
            disabled={saving}
            onClick={() => {
              void runAction(() => api.enableWorkspaceAgentAccess(workspaceId, true));
            }}
          >
            {saving ? 'Enabling...' : 'Enable Agent Access'}
          </button>
        </>
      )}
    </section>
  );
}
