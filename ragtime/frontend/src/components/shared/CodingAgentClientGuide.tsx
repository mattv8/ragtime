import { getCodingAgentClientLabel, type CodingAgentClientId } from './codingAgentClients';
import {
  buildClientConfig,
  buildCursorInstallLink,
  WORKSPACE_CREDENTIAL_PLACEHOLDER,
  workspaceCredentialEnvName,
  workspaceServerName,
} from './codingAgentConfigs';
import { ConfigLocations, CopyableSnippet, GuideNote } from './CodingAgentSetupPrimitives';

interface Props {
  clientId: CodingAgentClientId;
  mcpUrl: string;
  workspaceId: string;
}

const nativeProfiles = new Set<CodingAgentClientId>(['opencode', 'claude-code', 'codex']);

function Steps({ children }: { children: React.ReactNode }) {
  return <ol className="coding-agent-guide-steps">{children}</ol>;
}

function Alternative({ summary, children }: { summary: string; children: React.ReactNode }) {
  return (
    <details className="coding-agent-guide-alternative">
      <summary>{summary}</summary>
      {children}
    </details>
  );
}

export function CodingAgentClientGuide({ clientId, mcpUrl, workspaceId }: Props) {
  const clientLabel = getCodingAgentClientLabel(clientId);
  const name = workspaceServerName(workspaceId);
  const env = workspaceCredentialEnvName(workspaceId);
  const config = buildClientConfig(clientId, mcpUrl, workspaceId);
  const native = nativeProfiles.has(clientId);
  const usesCredentialEnvironment = new Set<CodingAgentClientId>([
    'claude-code',
    'cursor',
    'continue',
    'opencode',
    'hermes',
    'codex',
  ]).has(clientId);
  const description =
    clientId === 'chatgpt'
      ? 'Use ChatGPT only with an administrator-provided OAuth-compatible MCP route. This workspace credential does not apply to ChatGPT.'
      : `Connect ${clientLabel} to this workspace MCP endpoint.`;

  return (
    <section
      className="coding-agent-client-guide"
      data-client-guide={clientId}
      aria-label={`${clientLabel} setup guide`}
    >
      <p className="coding-agent-guide-description">{description}</p>
      {clientId === 'claude-desktop' ? (
        <>
          <section className="coding-agent-guide-section">
            <h5>Claude Desktop bearer bridge</h5>
            <Steps>
              <li>Install Node.js and npm.</li>
              <li>
                Create a private headers file containing{' '}
                <code>Authorization: Bearer {WORKSPACE_CREDENTIAL_PLACEHOLDER}</code>.
              </li>
              <li>
                Edit the Claude Desktop config file and add the copied <code>mcpServers</code>{' '}
                entry.
              </li>
              <li>Restart Claude Desktop completely, then confirm its tools appear.</li>
            </Steps>
            <CopyableSnippet value={config!} label="Copy Claude Desktop configuration" />
            <ConfigLocations
              locations={[
                {
                  label: 'macOS',
                  path: '~/Library/Application Support/Claude/claude_desktop_config.json',
                },
                { label: 'Windows', path: '%APPDATA%\\Claude\\claude_desktop_config.json' },
              ]}
            />
            <GuideNote>
              mcp-remote requires HTTPS unless you explicitly trust a local HTTP endpoint with its
              documented <code>--allow-http</code> option.
            </GuideNote>
          </section>
          <Alternative summary="OAuth route only (not this workspace credential)">
            <Steps>
              <li>Open Settings → Connectors → Add custom connector.</li>
              <li>
                Enter the administrator-provided OAuth-compatible route URL, then select Connect and
                approve the route’s authorization.
              </li>
            </Steps>
            <GuideNote>
              This route has separate permissions and does not offer workspace selection or use the
              scoped <code>rtdev</code> credential.
            </GuideNote>
          </Alternative>
        </>
      ) : clientId === 'chatgpt' ? (
        <section className="coding-agent-guide-section">
          <h5>ChatGPT OAuth route only</h5>
          <Steps>
            <li>
              Open Settings → Apps → Advanced settings and enable Developer mode if your account and
              administrator allow it.
            </li>
            <li>
              Open Apps → Create, enter an administrator-provided OAuth-compatible MCP endpoint,
              select its authentication settings, then Scan Tools and Create.
            </li>
            <li>Confirm the app’s tools in a new chat.</li>
          </Steps>
          <GuideNote>
            Plan and admin availability vary. The scoped <code>rtdev</code> bearer credential cannot
            be used in this UI, so there is no local file, token copy prompt, or workspace selection
            flow here.
          </GuideNote>
        </section>
      ) : (
        <>
          <section className="coding-agent-guide-section">
            <h5>
              {clientId === 'claude-code'
                ? 'Claude Code CLI'
                : clientId === 'cursor'
                  ? 'Cursor install link'
                  : `${clientLabel} configuration`}
            </h5>
            {clientId === 'claude-code' ? (
              <>
                <CredentialEnvironment env={env} />
                <Steps>
                  <li>
                    Run this POSIX CLI one-liner. <code>--scope user</code> stores the connection in
                    your user configuration.
                  </li>
                </Steps>
                <CopyableSnippet
                  value={`claude mcp add --scope user --transport http ${name} ${JSON.stringify(mcpUrl)} --header 'Authorization: Bearer \${${env}}'`}
                  label="Copy Claude Code user-scope command"
                />
                <Alternative summary="Project JSON alternative (use instead of the CLI)">
                  <p>
                    Add this to the project <code>.mcp.json</code>; do not also add the same server
                    with the CLI. User/local settings are kept in <code>~/.claude.json</code>.
                  </p>
                  <CopyableSnippet value={config!} label="Copy Claude Code project configuration" />
                </Alternative>
              </>
            ) : clientId === 'cursor' ? (
              <>
                <CredentialEnvironment env={env} />
                <a
                  className="btn btn-primary btn-sm"
                  href={buildCursorInstallLink(mcpUrl, workspaceId)}
                >
                  Install in Cursor
                </a>
                <GuideNote>
                  Open the link with Cursor installed, review the generated server entry, and
                  approve installation. The link contains only the endpoint and environment-variable
                  template.
                </GuideNote>
                <Alternative summary="Manual JSON alternative">
                  <p>
                    Use <code>.cursor/mcp.json</code> for this project or{' '}
                    <code>~/.cursor/mcp.json</code> for your user configuration.
                  </p>
                  <CopyableSnippet value={config!} label="Copy Cursor configuration" />
                </Alternative>
              </>
            ) : (
              <>
                {usesCredentialEnvironment && <CredentialEnvironment env={env} />}
                <CopyableSnippet value={config!} label={`Copy ${clientLabel} configuration`} />
              </>
            )}
          </section>
          <section className="coding-agent-guide-section">
            <h5>{clientLabel} finish setup</h5>
            {clientId === 'cline' && (
              <GuideNote>
                Use <code>streamableHttp</code>; omitting the type uses Cline’s legacy SSE default.
                The VS Code MCP Servers → Configure → Configure MCP Servers opener is authoritative.
              </GuideNote>
            )}
            {clientId === 'continue' && (
              <GuideNote>
                The standalone YAML includes required metadata. For{' '}
                <code>~/.continue/config.yaml</code>, merge only the <code>mcpServers</code> list
                and preserve models and other root settings. Use Agent mode and reload.
              </GuideNote>
            )}
            {clientId === 'hermes' && (
              <GuideNote>
                Save the config, run <code>/reload-mcp</code>, then{' '}
                <code>hermes mcp test {name}</code>.
              </GuideNote>
            )}
            {clientId === 'opencode' && (
              <GuideNote>
                The <code>oauth: false</code> entry uses OpenCode’s <code>{'{env:NAME}'}</code>{' '}
                header syntax. Run <code>opencode mcp list</code> after saving.
              </GuideNote>
            )}
            <Steps>
              {clientId === 'cline' ? (
                <li>
                  Paste the JSON through the UI opener, replace the credential placeholder
                  privately, then restart Cline or open a new task.
                </li>
              ) : clientId === 'continue' ? (
                <li>Reload Continue in Agent mode and verify the MCP tools.</li>
              ) : clientId === 'hermes' ? (
                <li>Verify the named server after reload and test it from Hermes.</li>
              ) : clientId === 'opencode' ? (
                <li>
                  Save the config and verify the named server appears in{' '}
                  <code>opencode mcp list</code>.
                </li>
              ) : (
                <li>
                  Save the configuration, restart or open a new client session, and verify the MCP
                  tools.
                </li>
              )}
            </Steps>
            <ConfigLocations
              locations={
                clientId === 'cline'
                  ? [
                      {
                        label: 'macOS',
                        path: '~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json',
                      },
                      {
                        label: 'Windows',
                        path: '%APPDATA%\\Code\\User\\globalStorage\\saoudrizwan.claude-dev\\settings\\cline_mcp_settings.json',
                      },
                      {
                        label: 'Linux',
                        path: '~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json',
                      },
                      { label: 'CLI alternative', path: '~/.cline/mcp.json' },
                    ]
                  : clientId === 'continue'
                    ? [
                        { label: 'Project', path: '.continue/mcpServers/ragtime.yaml' },
                        { label: 'User', path: '~/.continue/config.yaml' },
                      ]
                    : clientId === 'opencode'
                      ? [{ label: 'User', path: 'opencode.json' }]
                      : clientId === 'hermes'
                        ? [
                            { label: 'Configuration', path: '~/.hermes/config.yaml' },
                            { label: 'Credential environment', path: '~/.hermes/.env' },
                          ]
                        : clientId === 'codex'
                          ? [{ label: 'User', path: '~/.codex/config.toml' }]
                          : [
                              { label: 'Project', path: '.mcp.json' },
                              { label: 'User', path: '~/.claude.json' },
                            ]
              }
            />
            {native && (
              <GuideNote>
                Automatic manifest-assisted skills and rules installation is available only for
                OpenCode, Claude Code, and Codex. It preserves unrelated local settings.
              </GuideNote>
            )}
          </section>
        </>
      )}
    </section>
  );
}

function CredentialEnvironment({ env }: { env: string }) {
  return (
    <details className="coding-agent-guide-credential-prerequisite">
      <summary>Private credential environment</summary>
      <p>
        Set <code>{env}</code> in this client’s private local secret store or launch environment. Do
        not add the credential to project files or source control.
      </p>
    </details>
  );
}
