import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import type { WorkspaceDevelopmentCredential } from '@/types';
import { InlineCopyButton } from './InlineCopyButton';

interface ConnectYourAgentPanelProps {
  workspaceId: string;
  canManage: boolean;
}

interface ExecJob {
  id: string;
  status: string;
  output?: string;
  exit_code?: number | null;
}

function getErrorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}

export function ConnectYourAgentPanel({ workspaceId, canManage }: ConnectYourAgentPanelProps) {
  const [open, setOpen] = useState(false);
  const [credentials, setCredentials] = useState<WorkspaceDevelopmentCredential[]>([]);
  const [token, setToken] = useState<string | null>(null);
  const [name, setName] = useState('External agent');
  const [command, setCommand] = useState('');
  const [jobs, setJobs] = useState<ExecJob[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const workspaceRef = useRef(workspaceId);
  workspaceRef.current = workspaceId;

  const isCurrentWorkspace = useCallback(
    (requestedWorkspaceId: string) => workspaceRef.current === requestedWorkspaceId,
    [],
  );

  const loadJobs = useCallback(
    async (requestedWorkspaceId: string) => {
      const result = await api.executeWorkspaceDevelopmentOperation<unknown>(
        requestedWorkspaceId,
        'exec_list',
      );
      if (!Array.isArray(result)) {
        throw new Error('Development activity returned an invalid job list.');
      }
      if (isCurrentWorkspace(requestedWorkspaceId)) {
        setJobs(result as ExecJob[]);
      }
    },
    [isCurrentWorkspace],
  );

  const runWorkspaceAction = useCallback(
    async (action: (requestedWorkspaceId: string) => Promise<void>, fallback: string) => {
      const requestedWorkspaceId = workspaceId;
      setLoading(true);
      setError(null);

      try {
        await action(requestedWorkspaceId);
      } catch (reason) {
        if (isCurrentWorkspace(requestedWorkspaceId)) {
          setError(getErrorMessage(reason, fallback));
        }
      } finally {
        if (isCurrentWorkspace(requestedWorkspaceId)) {
          setLoading(false);
        }
      }
    },
    [isCurrentWorkspace, workspaceId],
  );

  const refreshJobs = useCallback(() => {
    void runWorkspaceAction(loadJobs, 'Failed to load development activity');
  }, [loadJobs, runWorkspaceAction]);

  useEffect(() => {
    let cancelled = false;
    setCredentials([]);
    setToken(null);
    setJobs([]);
    setError(null);
    setOpen(false);

    if (!canManage) return;

    setLoading(true);
    void api
      .listWorkspaceDevelopmentCredentials(workspaceId)
      .then((items) => {
        if (!cancelled && isCurrentWorkspace(workspaceId)) {
          setCredentials(items);
        }
      })
      .catch((reason) => {
        if (!cancelled && isCurrentWorkspace(workspaceId)) {
          setError(getErrorMessage(reason, 'Failed to load credentials'));
        }
      })
      .finally(() => {
        if (!cancelled && isCurrentWorkspace(workspaceId)) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [canManage, isCurrentWorkspace, workspaceId]);

  const createCredential = () => {
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      const created = await api.createWorkspaceDevelopmentCredential(requestedWorkspaceId, {
        name: name.trim() || 'External agent',
      });
      if (isCurrentWorkspace(requestedWorkspaceId)) {
        setToken(created.token);
        setCredentials((items) => [...items, created]);
      }
    }, 'Failed to create credential');
  };

  const rotateCredential = (credentialId: string) => {
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      const rotated = await api.rotateWorkspaceDevelopmentCredential(
        requestedWorkspaceId,
        credentialId,
      );
      if (isCurrentWorkspace(requestedWorkspaceId)) {
        setToken(rotated.token);
        setCredentials((items) => items.map((item) => (item.id === rotated.id ? rotated : item)));
      }
    }, 'Failed to rotate credential');
  };

  const revokeCredential = (credentialId: string) => {
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      const revoked = await api.revokeWorkspaceDevelopmentCredential(
        requestedWorkspaceId,
        credentialId,
      );
      if (isCurrentWorkspace(requestedWorkspaceId)) {
        setCredentials((items) => items.map((item) => (item.id === revoked.id ? revoked : item)));
      }
    }, 'Failed to revoke credential');
  };

  const startJob = () => {
    const trimmedCommand = command.trim();
    if (!trimmedCommand) return;

    void runWorkspaceAction(async (requestedWorkspaceId) => {
      await api.executeWorkspaceDevelopmentOperation(requestedWorkspaceId, 'exec_start', {
        command: trimmedCommand,
      });
      if (isCurrentWorkspace(requestedWorkspaceId)) {
        setCommand('');
      }
      await loadJobs(requestedWorkspaceId);
    }, 'Failed to start command');
  };

  const cancelJob = (jobId: string) => {
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      await api.executeWorkspaceDevelopmentOperation(requestedWorkspaceId, 'exec_cancel', {
        job_id: jobId,
      });
      await loadJobs(requestedWorkspaceId);
    }, 'Failed to cancel command');
  };

  const contentId = `workspace-connect-agent-content-${workspaceId}`;
  const tokenId = `workspace-agent-token-${workspaceId}`;
  const mcpUrl = `${window.location.origin}/mcp`;
  const operationsUrl = `${window.location.origin}/indexes/userspace/development/workspaces/${workspaceId}/operations`;

  return (
    <section
      id={`workspace-connect-agent-${workspaceId}`}
      className="userspace-connect-agent"
      data-userspace-panel="connect-your-agent"
    >
      <button
        type="button"
        className="btn btn-secondary btn-sm"
        aria-expanded={open}
        aria-controls={contentId}
        onClick={() => setOpen((value) => !value)}
      >
        Connect your agent
      </button>

      {open && (
        <div id={contentId} className="card" data-userspace-panel="connect-your-agent-content">
          <h3>Connect your agent</h3>
          <p className="muted">
            Authenticate with <code>Authorization: Bearer &lt;credential&gt;</code>. Start with{' '}
            <code>workspace_development_context</code>, then use <code>workspace_development</code>{' '}
            for authorized operations.
          </p>
          <p className="muted">
            MCP endpoint: <code>{mcpUrl}</code>
          </p>
          <p className="muted">
            HTTP operations endpoint: <code>{operationsUrl}</code>
          </p>

          {!canManage ? (
            <p className="muted">
              Only workspace owners and admins can manage development credentials.
            </p>
          ) : (
            <>
              <label htmlFor={`workspace-agent-name-${workspaceId}`}>Credential name</label>
              <div className="form-row">
                <input
                  id={`workspace-agent-name-${workspaceId}`}
                  value={name}
                  onChange={(event) => setName(event.target.value)}
                />
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  disabled={loading}
                  onClick={createCredential}
                >
                  Create credential
                </button>
              </div>

              {token && (
                <div id={tokenId} className="api-key-display">
                  <code>{token}</code>
                  <InlineCopyButton
                    copyText={token}
                    className="btn btn-secondary btn-sm"
                    title="Copy development credential"
                    ariaLabel="Copy development credential"
                    label="Copy token"
                  />
                </div>
              )}

              {error && (
                <p role="alert" className="error-message">
                  {error}
                </p>
              )}

              <div data-userspace-list="development-credentials">
                {credentials.map((credential) => (
                  <div
                    key={credential.id}
                    id={`workspace-development-credential-${credential.id}`}
                    className="card"
                  >
                    <strong>{credential.name}</strong>{' '}
                    <span className="muted">
                      {credential.revoked_at ? 'revoked' : credential.scopes.join(', ')}
                    </span>
                    {!credential.revoked_at && (
                      <div className="form-row">
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          disabled={loading}
                          onClick={() => rotateCredential(credential.id)}
                        >
                          Rotate
                        </button>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          disabled={loading}
                          onClick={() => revokeCredential(credential.id)}
                        >
                          Revoke
                        </button>
                      </div>
                    )}
                  </div>
                ))}
              </div>

              <section
                id={`workspace-development-activity-${workspaceId}`}
                data-userspace-panel="development-activity"
              >
                <h4>Development activity</h4>
                <div className="form-row">
                  <input
                    aria-label="Sandbox command"
                    value={command}
                    onChange={(event) => setCommand(event.target.value)}
                    placeholder="Run a sandbox command"
                  />
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={loading || !command.trim()}
                    onClick={startJob}
                  >
                    Run
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={loading}
                    onClick={refreshJobs}
                  >
                    Refresh
                  </button>
                </div>
                <div data-userspace-list="development-jobs">
                  {jobs.map((job) => (
                    <div key={job.id} id={`workspace-development-job-${job.id}`} className="card">
                      <strong>{job.status}</strong>{' '}
                      {job.exit_code !== undefined && job.exit_code !== null && (
                        <span className="muted">exit {job.exit_code}</span>
                      )}
                      {job.output && <pre>{job.output}</pre>}
                      {!['completed', 'failed', 'cancelled'].includes(job.status) && (
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          disabled={loading}
                          onClick={() => cancelJob(job.id)}
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </section>
            </>
          )}
        </div>
      )}
    </section>
  );
}
