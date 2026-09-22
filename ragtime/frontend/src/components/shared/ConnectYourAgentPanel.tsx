import { useCallback, useEffect, useRef, useState } from 'react';
import { X } from 'lucide-react';
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

type SetupStep = 'create' | 'connect' | 'start';

function getErrorMessage(reason: unknown, fallback: string): string {
  return reason instanceof Error ? reason.message : fallback;
}

function buildSetupInstructions(
  bootstrapManifestUrl: string,
  workspaceId: string,
  token: string,
  credentialName: string,
): string {
  return [
    '# Set up native development client',
    '',
    `Authenticate with this workspace's development credential. It grants the agent read access to workspace context plus authorized edit and run operations. Use the manifest URL to discover and install required skills, configuration, and rules.`,
    '',
    '**Manifest URL:**',
    bootstrapManifestUrl,
    '',
    '**Workspace ID:**',
    workspaceId,
    '',
    `**Credential (${credentialName}):**`,
    token,
    '',
    '## Instructions for trusted receiving agent',
    '',
    '1. **Fetch the manifest** from the URL above with an Authorization: Bearer <credential> header. Choose the profile for your agent client.',
    '',
    '2. **Download and verify** all required skills, rules, and configuration files. Verify checksums as indicated in the manifest.',
    '',
    '3. **Install locally for your agent client** using that profile’s destinations and merge instructions. Preserve unrelated configuration, provider settings, and user rules; update managed entries without duplicating them. Keep setup files separate from the remote workspace’s application source.',
    '',
    '4. **Store the credential privately** outside version control. Arrange the environment variable(s) declared in the manifest when launching your client.',
    '',
    '5. **Load core guidance** and verify MCP connectivity and workspace context are accessible.',
    '',
    '6. **Report** whether your agent client needs a restart or new session to load the installed configuration.',
    '',
    'Keep the credential in the private store or launch environment from step 4. Do not repeat it in outputs, application files, managed configuration, summaries, or logs.',
    '',
    'If this credential has been revoked or rotated, create or rotate a new credential instead of re-using it.',
  ].join('\n');
}

function getCredentialStatus(
  credential: WorkspaceDevelopmentCredential,
): 'Active' | 'Expired' | 'Revoked' {
  if (credential.revoked_at) return 'Revoked';
  if (credential.expires_at && new Date(credential.expires_at).getTime() <= Date.now())
    return 'Expired';
  return 'Active';
}

function getScopeDescription(credential: WorkspaceDevelopmentCredential): string {
  return credential.scopes.length > 0
    ? `Authorized scopes: ${credential.scopes.join(', ')}`
    : 'Authorized workspace development access';
}

function SetupSteps({ step }: { step: SetupStep }) {
  const steps: Array<{ id: SetupStep; label: string }> = [
    { id: 'create', label: 'Create access' },
    { id: 'connect', label: 'Connect agent' },
    { id: 'start', label: 'Start working' },
  ];
  const currentIndex = steps.findIndex((item) => item.id === step);
  return (
    <ol
      className="wizard-progress userspace-connect-agent-steps"
      aria-label="Coding agent setup steps"
    >
      {steps.map((item, index) => (
        <li
          key={item.id}
          className={`wizard-step${index === currentIndex ? ' active' : ''}${index < currentIndex ? ' completed' : ''}`}
          aria-current={item.id === step ? 'step' : undefined}
        >
          <span className="step-number">{index + 1}</span>
          <span className="step-title">{item.label}</span>
        </li>
      ))}
    </ol>
  );
}

export function ConnectYourAgentPanel({ workspaceId, canManage }: ConnectYourAgentPanelProps) {
  const [credentials, setCredentials] = useState<WorkspaceDevelopmentCredential[]>([]);
  const [token, setToken] = useState<string | null>(null);
  const [tokenCredentialId, setTokenCredentialId] = useState<string | null>(null);
  const [activeCredentialId, setActiveCredentialId] = useState<string | null>(null);
  const [step, setStep] = useState<SetupStep>('create');
  const [showCreateForm, setShowCreateForm] = useState(true);
  const [pendingRotationId, setPendingRotationId] = useState<string | null>(null);
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [name, setName] = useState('External agent');
  const [command, setCommand] = useState('');
  const [jobs, setJobs] = useState<ExecJob[]>([]);
  const [activityOpen, setActivityOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const workspaceRef = useRef(workspaceId);
  const canManageRef = useRef(canManage);
  workspaceRef.current = workspaceId;
  canManageRef.current = canManage;

  const isCurrentContext = useCallback(
    (requestedWorkspaceId: string) =>
      workspaceRef.current === requestedWorkspaceId && canManageRef.current,
    [],
  );

  const runWorkspaceAction = useCallback(
    async (action: (requestedWorkspaceId: string) => Promise<void>, fallback: string) => {
      const requestedWorkspaceId = workspaceId;
      setLoading(true);
      setError(null);
      try {
        await action(requestedWorkspaceId);
      } catch (reason) {
        if (isCurrentContext(requestedWorkspaceId)) {
          setError(getErrorMessage(reason, fallback));
        }
      } finally {
        if (isCurrentContext(requestedWorkspaceId)) {
          setLoading(false);
        }
      }
    },
    [isCurrentContext, workspaceId],
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
      if (isCurrentContext(requestedWorkspaceId)) setJobs(result as ExecJob[]);
    },
    [isCurrentContext],
  );

  useEffect(() => {
    let cancelled = false;
    setCredentials([]);
    setToken(null);
    setTokenCredentialId(null);
    setActiveCredentialId(null);
    setStep('create');
    setShowCreateForm(true);
    setPendingRotationId(null);
    setPendingDeleteId(null);
    setJobs([]);
    setActivityOpen(false);
    setError(null);
    if (!canManage) return;

    setLoading(true);
    void api
      .listWorkspaceDevelopmentCredentials(workspaceId)
      .then((items) => {
        if (!cancelled && isCurrentContext(workspaceId)) {
          setCredentials(items);
          setShowCreateForm(items.length === 0);
        }
      })
      .catch((reason) => {
        if (!cancelled && isCurrentContext(workspaceId)) {
          setError(getErrorMessage(reason, 'Failed to load credentials'));
        }
      })
      .finally(() => {
        if (!cancelled && isCurrentContext(workspaceId)) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [canManage, isCurrentContext, workspaceId]);

  const createCredential = () => {
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      const created = await api.createWorkspaceDevelopmentCredential(requestedWorkspaceId, {
        name: name.trim() || 'External agent',
      });
      if (isCurrentContext(requestedWorkspaceId)) {
        setCredentials((items) => [...items, created]);
        setToken(created.token);
        setTokenCredentialId(created.id);
        setActiveCredentialId(created.id);
        setStep('connect');
        setShowCreateForm(false);
      }
    }, 'Failed to create credential');
  };

  const rotateCredential = () => {
    if (!pendingRotationId) return;
    const credentialId = pendingRotationId;
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      const rotated = await api.rotateWorkspaceDevelopmentCredential(
        requestedWorkspaceId,
        credentialId,
      );
      if (isCurrentContext(requestedWorkspaceId)) {
        setCredentials((items) => items.map((item) => (item.id === rotated.id ? rotated : item)));
        setToken(rotated.token);
        setTokenCredentialId(rotated.id);
        setActiveCredentialId(rotated.id);
        setStep('connect');
        setPendingRotationId(null);
      }
    }, 'Failed to rotate credential');
  };

  const revokeCredential = (credentialId: string) => {
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      const revoked = await api.revokeWorkspaceDevelopmentCredential(
        requestedWorkspaceId,
        credentialId,
      );
      if (isCurrentContext(requestedWorkspaceId)) {
        setCredentials((items) => items.map((item) => (item.id === revoked.id ? revoked : item)));
        if (tokenCredentialId === credentialId) {
          setToken(null);
          setTokenCredentialId(null);
        }
        if (activeCredentialId === credentialId) {
          setActiveCredentialId(null);
          setStep('create');
        }
      }
    }, 'Failed to revoke credential');
  };

  const deleteCredential = () => {
    if (!pendingDeleteId) return;
    const credentialId = pendingDeleteId;
    setPendingDeleteId(null); // close confirmation regardless of outcome
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      await api.deleteWorkspaceDevelopmentCredential(requestedWorkspaceId, credentialId);
      if (isCurrentContext(requestedWorkspaceId)) {
        setCredentials((items) => items.filter((item) => item.id !== credentialId));
      }
    }, 'Failed to delete credential');
  };

  const startJob = () => {
    const trimmedCommand = command.trim();
    if (!trimmedCommand) return;
    void runWorkspaceAction(async (requestedWorkspaceId) => {
      await api.executeWorkspaceDevelopmentOperation(requestedWorkspaceId, 'exec_start', {
        command: trimmedCommand,
      });
      if (isCurrentContext(requestedWorkspaceId)) setCommand('');
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

  const bootstrapManifestUrl = `${window.location.origin}/indexes/userspace/development/workspaces/${workspaceId}/bootstrap`;
  const mcpUrl = `${window.location.origin}/mcp`;
  const operationsUrl = `${window.location.origin}/indexes/userspace/development/workspaces/${workspaceId}/operations`;
  const wizardActive = showCreateForm || activeCredentialId !== null;

  return (
    <section
      id={`workspace-connect-agent-${workspaceId}`}
      className="userspace-connect-agent"
      data-userspace-panel="connect-your-agent"
      role="region"
      aria-label="Coding Agent Setup"
    >
      <div
        className="userspace-connect-agent-body"
        data-userspace-panel="connect-your-agent-content"
      >
        <header className="userspace-connect-agent-header">
          <div>
            <h3>Coding Agent Setup</h3>
            <p className="muted">
              Create access, connect your trusted coding agent, then start working.
            </p>
          </div>
        </header>

        {!canManage ? (
          <p className="muted">
            Only workspace owners and admins can manage development credentials.
          </p>
        ) : (
          <>
            <section
              className="userspace-connect-agent-section"
              aria-label="Development credentials"
            >
              <div className="userspace-connect-agent-section-header">
                <h4>Development credentials</h4>
                {credentials.length > 0 && !showCreateForm && (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={loading}
                    onClick={() => {
                      setError(null);
                      setShowCreateForm(true);
                      setActiveCredentialId(null);
                      setStep('create');
                    }}
                  >
                    New agent
                  </button>
                )}
              </div>

              {showCreateForm && (
                <div
                  id={`workspace-agent-create-${workspaceId}`}
                  className="userspace-connect-agent-create"
                  data-userspace-step="create"
                >
                  <div className="userspace-connect-agent-create-header">
                    <SetupSteps step="create" />
                    {credentials.length > 0 && (
                      <button
                        type="button"
                        className="close-btn"
                        aria-label="Dismiss new agent form"
                        onClick={() => setShowCreateForm(false)}
                      >
                        <X size={16} />
                      </button>
                    )}
                  </div>
                  <p className="muted">
                    Create a credential for one trusted agent. You will copy its setup instructions
                    next.
                  </p>
                  <label
                    className="userspace-connect-agent-field-label"
                    htmlFor={`workspace-agent-name-${workspaceId}`}
                  >
                    Credential name
                  </label>
                  <div className="userspace-connect-agent-input-row">
                    <input
                      id={`workspace-agent-name-${workspaceId}`}
                      type="text"
                      value={name}
                      onChange={(event) => setName(event.target.value)}
                    />
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={loading}
                      onClick={createCredential}
                    >
                      Create credential and continue
                    </button>
                  </div>
                </div>
              )}

              {!showCreateForm && credentials.length === 0 && !loading ? (
                <p className="userspace-muted">No development credentials yet.</p>
              ) : !showCreateForm ? (
                <div
                  className="userspace-connect-agent-card-list"
                  data-userspace-list="development-credentials"
                >
                  {credentials.map((credential) => {
                    const status = getCredentialStatus(credential);
                    const isActive = activeCredentialId === credential.id;
                    const hasSecret = tokenCredentialId === credential.id && token;
                    return (
                      <article
                        key={credential.id}
                        id={`workspace-development-credential-${credential.id}`}
                        className={`userspace-connect-agent-card${status === 'Revoked' ? ' is-revoked' : ''}`}
                        data-credential-id={credential.id}
                        aria-label={`${credential.name} credential`}
                      >
                        <div className="userspace-connect-agent-card-header">
                          <div className="userspace-connect-agent-card-copy">
                            <div className="userspace-connect-agent-card-title">
                              <strong>{credential.name}</strong>
                              <span
                                className={`userspace-external-api-status${status === 'Revoked' ? ' is-revoked' : ''}`}
                              >
                                {status}
                              </span>
                            </div>
                            <span className="userspace-connect-agent-card-meta muted">
                              {getScopeDescription(credential)}
                            </span>
                          </div>
                          {!isActive && status !== 'Revoked' && (
                            <div className="userspace-connect-agent-card-actions">
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                disabled={loading}
                                onClick={() => {
                                  setShowCreateForm(false);
                                  setActiveCredentialId(credential.id);
                                  setStep('connect');
                                }}
                              >
                                Setup instructions
                              </button>
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                disabled={loading}
                                onClick={() => setPendingRotationId(credential.id)}
                              >
                                Rotate credential
                              </button>
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                disabled={loading}
                                onClick={() => revokeCredential(credential.id)}
                              >
                                Revoke credential
                              </button>
                            </div>
                          )}
                          {!isActive && status === 'Revoked' && (
                            <div className="userspace-connect-agent-card-actions">
                              <button
                                type="button"
                                className="btn btn-danger btn-sm"
                                disabled={loading}
                                onClick={() => setPendingDeleteId(credential.id)}
                              >
                                Delete
                              </button>
                            </div>
                          )}
                        </div>

                        {isActive && status !== 'Revoked' && (
                          <div
                            id={`workspace-agent-setup-${credential.id}`}
                            className="userspace-connect-agent-setup"
                            data-userspace-setup={credential.id}
                          >
                            <SetupSteps step={step} />
                            {step === 'connect' && (
                              <section
                                id={`workspace-setup-instructions-${credential.id}`}
                                data-userspace-panel="setup-instructions"
                              >
                                <h5>Connect agent</h5>
                                {hasSecret ? (
                                  <>
                                    <p>
                                      Copy these instructions and paste them into a new conversation
                                      with your trusted coding agent. They include this credential
                                      and explain the read, edit, and run access it grants.
                                    </p>
                                    <InlineCopyButton
                                      copyText={() =>
                                        buildSetupInstructions(
                                          bootstrapManifestUrl,
                                          workspaceId,
                                          token,
                                          credential.name,
                                        )
                                      }
                                      className="btn btn-primary btn-sm"
                                      title="Copy setup instructions"
                                      ariaLabel="Copy setup instructions"
                                      label="Copy setup instructions"
                                    />
                                    <details className="userspace-connect-agent-preview">
                                      <summary>Preview setup instructions</summary>
                                      <pre>
                                        {buildSetupInstructions(
                                          bootstrapManifestUrl,
                                          workspaceId,
                                          token,
                                          credential.name,
                                        )}
                                      </pre>
                                    </details>
                                  </>
                                ) : (
                                  <div className="userspace-connect-agent-recovery">
                                    <p>
                                      Your secret is only shown when created or rotated. To recover
                                      setup instructions, rotate this credential or create a new
                                      one.
                                    </p>
                                    <button
                                      type="button"
                                      className="btn btn-secondary btn-sm"
                                      disabled={loading}
                                      aria-label="Rotate credential to recover setup"
                                      onClick={() => setPendingRotationId(credential.id)}
                                    >
                                      Rotate credential
                                    </button>
                                  </div>
                                )}
                                <details
                                  id={`workspace-agent-manual-details-${credential.id}`}
                                  className="userspace-connect-agent-manual-details"
                                  data-userspace-disclosure="manual-connection-details"
                                >
                                  <summary>Manual connection details</summary>
                                  <p>
                                    Authenticate with{' '}
                                    <code>Authorization: Bearer &lt;credential&gt;</code>.{' '}
                                    <code>/mcp</code> is the default endpoint for coding-agent
                                    workspace credentials; HTTP operations are an advanced
                                    alternative.
                                  </p>
                                  <div className="userspace-connect-agent-endpoint">
                                    <span className="userspace-connect-agent-endpoint-label">
                                      MCP endpoint
                                    </span>
                                    <code>{mcpUrl}</code>
                                    <InlineCopyButton
                                      copyText={mcpUrl}
                                      className="userspace-connect-agent-copy"
                                      title="Copy MCP endpoint"
                                      ariaLabel="Copy MCP endpoint"
                                      iconSize={13}
                                    />
                                  </div>
                                  <div className="userspace-connect-agent-endpoint">
                                    <span className="userspace-connect-agent-endpoint-label">
                                      HTTP operations endpoint
                                    </span>
                                    <code>{operationsUrl}</code>
                                    <InlineCopyButton
                                      copyText={operationsUrl}
                                      className="userspace-connect-agent-copy"
                                      title="Copy HTTP operations endpoint"
                                      ariaLabel="Copy HTTP operations endpoint"
                                      iconSize={13}
                                    />
                                  </div>
                                  {hasSecret && (
                                    <div className="api-key-display userspace-connect-agent-token">
                                      <span className="userspace-connect-agent-endpoint-label">
                                        Development credential
                                      </span>
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
                                </details>
                                <footer
                                  id={`workspace-agent-connect-footer-${credential.id}`}
                                  className="userspace-connect-agent-footer-actions"
                                  data-userspace-footer="connect"
                                >
                                  <button
                                    type="button"
                                    className="btn btn-primary btn-sm"
                                    onClick={() => setStep('start')}
                                  >
                                    Next: Start working
                                  </button>
                                </footer>
                              </section>
                            )}
                            {step === 'start' && (
                              <section
                                id={`workspace-agent-start-${credential.id}`}
                                className="userspace-connect-agent-start"
                                aria-label="Start working"
                                data-userspace-step="start"
                              >
                                <h5>Start working</h5>
                                <p>
                                  Ask your agent to load the workspace context. It may need a
                                  restart or a new session after setup.
                                </p>
                                <p>
                                  <strong>Suggested first request:</strong> “Inspect this workspace
                                  and summarize the current project state before making changes.”
                                </p>
                                <div
                                  id={`workspace-agent-start-footer-${credential.id}`}
                                  className="userspace-connect-agent-footer-actions"
                                  data-userspace-footer="start"
                                >
                                  <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => setStep('connect')}
                                  >
                                    Back to Connect agent
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-primary btn-sm"
                                    onClick={() => setActiveCredentialId(null)}
                                  >
                                    Done
                                  </button>
                                </div>
                              </section>
                            )}
                          </div>
                        )}
                      </article>
                    );
                  })}
                </div>
              ) : null}
            </section>

            {pendingRotationId && (
              <section className="userspace-connect-agent-rotation-warning" role="alert">
                <strong>Rotate this credential?</strong>
                <p>
                  The old token stops working immediately. Update the trusted agent with the new
                  setup instructions.
                </p>
                <div className="userspace-connect-agent-footer-actions">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={loading}
                    onClick={() => setPendingRotationId(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="btn btn-primary btn-sm"
                    disabled={loading}
                    onClick={rotateCredential}
                  >
                    Rotate and continue
                  </button>
                </div>
              </section>
            )}

            {pendingDeleteId && (
              <section className="userspace-connect-agent-rotation-warning" role="alert">
                <strong>Permanently delete this credential?</strong>
                <p>This cannot be undone. The credential record will be removed entirely.</p>
                <div className="userspace-connect-agent-footer-actions">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={loading}
                    onClick={() => setPendingDeleteId(null)}
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger btn-sm"
                    disabled={loading}
                    onClick={deleteCredential}
                  >
                    Delete permanently
                  </button>
                </div>
              </section>
            )}

            {error && (
              <p role="alert" className="error-message">
                {error}
              </p>
            )}

            {!wizardActive && (
              <details
                id={`workspace-development-activity-${workspaceId}`}
                className="userspace-connect-agent-activity"
                data-userspace-disclosure="development-activity"
                open={activityOpen}
                onToggle={(event) => setActivityOpen(event.currentTarget.open)}
              >
                <summary>Development activity</summary>
                <div className="userspace-connect-agent-activity-content">
                  <p className="muted userspace-connect-agent-activity-help">
                    Run a one-off command in this workspace's sandbox runtime and watch its status.
                    Useful for checking that a connected agent's environment works; not part of
                    setup.
                  </p>
                  <div className="userspace-connect-agent-input-row">
                    <input
                      aria-label="Sandbox command"
                      type="text"
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
                      onClick={() =>
                        void runWorkspaceAction(loadJobs, 'Failed to load development activity')
                      }
                    >
                      Refresh
                    </button>
                  </div>
                  {jobs.length > 0 && (
                    <ul
                      className="userspace-connect-agent-list"
                      data-userspace-list="development-jobs"
                    >
                      {jobs.map((job) => (
                        <li
                          key={job.id}
                          id={`workspace-development-job-${job.id}`}
                          className="userspace-connect-agent-row userspace-connect-agent-job"
                        >
                          <div className="userspace-connect-agent-row-main">
                            <strong>{job.status}</strong>
                            {job.exit_code !== undefined && job.exit_code !== null && (
                              <span className="muted">exit {job.exit_code}</span>
                            )}
                          </div>
                          {!['completed', 'failed', 'cancelled'].includes(job.status) && (
                            <div className="userspace-connect-agent-row-actions">
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm"
                                disabled={loading}
                                onClick={() => cancelJob(job.id)}
                              >
                                Cancel
                              </button>
                            </div>
                          )}
                          {job.output && (
                            <pre className="userspace-connect-agent-job-output">{job.output}</pre>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </details>
            )}
          </>
        )}
      </div>
    </section>
  );
}
