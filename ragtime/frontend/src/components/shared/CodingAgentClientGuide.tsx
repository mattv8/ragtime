import { getCodingAgentClientLabel, type CodingAgentClientId } from './codingAgentClients';
import {
  buildClientConfig,
  buildCursorInstallLink,
  WORKSPACE_CREDENTIAL_PLACEHOLDER,
  workspaceCredentialEnvName,
  workspaceServerName,
} from './codingAgentConfigs';
import {
  claudeIllustratedExample,
  claudeSetupImages,
  codingAgentOfficialDocs,
} from './codingAgentGuideResources';
import {
  ConfigLocations,
  CopyableSnippet,
  GuideImage,
  GuideNote,
} from './CodingAgentSetupPrimitives';

interface Props {
  clientId: CodingAgentClientId;
  mcpUrl: string;
  workspaceId: string;
}

const nativeProfiles = new Set<CodingAgentClientId>(['opencode', 'claude-code', 'codex']);

function Steps({ children }: { children: React.ReactNode }) {
  return <ol className="coding-agent-guide-steps">{children}</ol>;
}

function Alternative({
  summary,
  disclosure,
  children,
}: {
  summary: string;
  disclosure: string;
  children: React.ReactNode;
}) {
  return (
    <details className="coding-agent-guide-alternative" data-guide-disclosure={disclosure}>
      <summary>{summary}</summary>
      {children}
    </details>
  );
}

function Docs({ href }: { href: string }) {
  return (
    <section className="coding-agent-guide-section" data-guide-section="official-docs">
      <h5>Official docs</h5>
      <a href={href} target="_blank" rel="noreferrer">
        Open the current client documentation
      </a>
    </section>
  );
}

function Verify({ workspaceId }: { workspaceId: string }) {
  const prompt = `Call workspace_development_context for workspace ${workspaceId}, then summarize the project and capabilities. Do not change files or run commands.`;
  return (
    <section className="coding-agent-guide-section" data-guide-section="verify">
      <h5>Verify</h5>
      <CopyableSnippet value={prompt} label="Copy safe verification prompt" />
    </section>
  );
}

function CredentialEnvironment({ env, client }: { env: string; client: string }) {
  return (
    <details
      className="coding-agent-guide-credential-prerequisite"
      data-guide-disclosure="credential-environment"
    >
      <summary>Before you start: private credential</summary>
      <p>
        Create or retrieve the credential in Development credentials below, then set{' '}
        <code>{env}</code> in {client}'s documented local environment mechanism before launching the
        client. Never put it in project files, URLs, screenshots, or source control.
      </p>
    </details>
  );
}

function ClaudeGuide({
  mcpUrl,
  workspaceId,
  config,
}: {
  mcpUrl: string;
  workspaceId: string;
  config: string;
}) {
  return (
    <>
      <section className="coding-agent-guide-section" data-guide-section="claude-remote">
        <h5>Before you start</h5>
        <p>
          The endpoint must be public HTTPS reachable from Anthropic's cloud. Your own VPN access is
          not enough.
        </p>
      </section>
      <section className="coding-agent-guide-section" data-guide-section="claude-connect">
        <h5>Connect Claude Cowork and Desktop</h5>
        <Steps>
          <li>
            Navigate to a connector: personal users choose Customize → Connectors → Add custom
            connector; organization owners choose Organization settings → Connectors → Add → Custom
            → Web, if asked.
          </li>
          <li>
            Give it a recognizable name and paste the OAuth MCP endpoint:
            <CopyableSnippet value={mcpUrl} label="Copy OAuth MCP endpoint" />
          </li>
          <li>
            Under Authentication choose <strong>Sign in now</strong>.
          </li>
          <li>
            Leave <strong>Use Claude's published identity</strong> selected.{' '}
            <strong>Register automatically</strong> also works if needed.
          </li>
          <li>
            Leave Request headers empty. Do not add a client ID, client secret, or workspace
            credential.
          </li>
          <li>
            Add the connector, then complete Ragtime sign-in and MFA if prompted (your existing
            session may finish without requiring another prompt). Enable the connector in + →
            Connectors.
          </li>
        </Steps>
        <GuideNote>
          <strong>No sign-in means no Claude OAuth,</strong> not disabling Ragtime authentication.
          OAuth uses your signed-in Ragtime user's existing workspace permissions. Workspace
          credential rotation or revocation in this panel does not revoke OAuth; manage that
          connection in Claude's Connectors settings.
        </GuideNote>
      </section>
      <Verify workspaceId={workspaceId} />
      <Alternative
        summary="Advanced: connect with a workspace credential"
        disclosure="claude-workspace-credential"
      >
        <GuideNote>
          <strong>Warning — shared credential:</strong> Organization request-header authentication
          is a limited-organization beta. Members using a connector share its credential; create one
          only for intended shared access. Rotating or revoking it disconnects everyone using it.
        </GuideNote>
        <Steps>
          <li>
            In the connector Authentication, choose <strong>No sign-in</strong> instead of Sign in
            now.
          </li>
          <li>
            Add request header name <code>Authorization</code> and value{' '}
            <code>Bearer {WORKSPACE_CREDENTIAL_PLACEHOLDER}</code> (including the space after
            Bearer); mark Required if shown. Keep default Streamable HTTP, not SSE.
          </li>
          <li>Add it, then enable the connector in a conversation.</li>
        </Steps>
        <p>
          Create or retrieve this credential in the disclosure below, then add request headers per
          those instructions. Never paste it into the URL or an image.
        </p>
      </Alternative>
      <Alternative summary="Troubleshooting and changes" disclosure="claude-troubleshooting">
        <p>
          After header or credential changes, remove and re-add the Claude connector and have
          members reconnect. If you see authentication errors, verify the Ragtime instance has OAuth
          enabled for the default <code>/mcp</code> route (Settings → Enable MCP Server, Require
          authentication for default /mcp route, Authentication Method: OAuth2).
        </p>
      </Alternative>
      <section
        className="coding-agent-guide-section"
        data-guide-section="claude-illustrated-example"
      >
        <h5>Illustrated example</h5>
        <a href={claudeIllustratedExample.href} target="_blank" rel="noreferrer">
          {claudeIllustratedExample.label}
        </a>
        <small>
          {claudeIllustratedExample.source}; follow the current dialog rather than older
          screenshots.
        </small>
        <details
          className="coding-agent-guide-alternative"
          data-guide-disclosure="claude-remote-image"
        >
          <summary>View older remote connector example</summary>
          <GuideImage {...claudeSetupImages.remote} />
        </details>
      </section>
      <Alternative
        summary="Desktop-only local bridge (not Cowork or web)"
        disclosure="claude-desktop-bridge"
      >
        <Steps>
          <li>
            Install Node.js and npm, then create a private headers file containing exactly{' '}
            <code>Authorization: Bearer {WORKSPACE_CREDENTIAL_PLACEHOLDER}</code>.
          </li>
          <li>
            In Claude Desktop choose Developer → Edit Config and merge this <code>mcpServers</code>{' '}
            entry into existing JSON; do not overwrite it.
          </li>
          <li>
            Replace the header-file path with an absolute private path (for example{' '}
            <code>/Users/me/.config/ragtime-headers.txt</code> on macOS or{' '}
            <code>C:\\Users\\me\\.config\\ragtime-headers.txt</code> on Windows; JSON requires
            escaped backslashes.
          </li>
          <li>Restart Desktop completely and check that tools appear.</li>
        </Steps>
        <CopyableSnippet value={config} label="Copy Claude Desktop bridge configuration" />
        <ConfigLocations
          locations={[
            {
              label: 'macOS',
              path: '~/Library/Application Support/Claude/claude_desktop_config.json',
            },
            { label: 'Windows', path: '%APPDATA%\\Claude\\claude_desktop_config.json' },
          ]}
        />
        <p>
          See the{' '}
          <a href={codingAgentOfficialDocs.mcpLocal} target="_blank" rel="noreferrer">
            local MCP documentation
          </a>{' '}
          and mcp-remote's{' '}
          <a href={codingAgentOfficialDocs.mcpRemoteHeaders} target="_blank" rel="noreferrer">
            custom header documentation
          </a>
          .
        </p>
        <GuideImage {...claudeSetupImages.desktop} />
      </Alternative>
      <Docs href={codingAgentOfficialDocs.claude} />
    </>
  );
}

export function CodingAgentClientGuide({ clientId, mcpUrl, workspaceId }: Props) {
  const clientLabel = getCodingAgentClientLabel(clientId);
  const name = workspaceServerName(workspaceId);
  const env = workspaceCredentialEnvName(workspaceId);
  const config = buildClientConfig(clientId, mcpUrl, workspaceId);
  const docs =
    clientId === 'claude-desktop'
      ? codingAgentOfficialDocs.claude
      : clientId === 'claude-code'
        ? codingAgentOfficialDocs.claudeCode
        : codingAgentOfficialDocs[clientId];

  return (
    <section
      className="coding-agent-client-guide"
      data-client-guide={clientId}
      aria-label={`${clientLabel} setup guide`}
    >
      <p className="coding-agent-guide-description">
        Connect {clientLabel} to this workspace MCP endpoint.
      </p>
      {clientId === 'claude-desktop' ? (
        <ClaudeGuide mcpUrl={mcpUrl} workspaceId={workspaceId} config={config!} />
      ) : clientId === 'chatgpt' ? (
        <>
          <section className="coding-agent-guide-section" data-guide-section="connect">
            <h5>Before you start</h5>
            <p>
              Hosted ChatGPT requires OAuth to access development tools. Developer mode is available
              on your plan and organization if applicable. No workspace credential is needed.
            </p>
            <h5>Connect</h5>
            <Steps>
              <li>
                Open Settings → Security and login → Developer mode, then enable it if allowed.
              </li>
              <li>Open ChatGPT Plugins → plus → create developer app.</li>
              <li>
                Choose Authentication: <strong>OAuth</strong>.
              </li>
              <li>
                Paste the MCP endpoint URL:
                <CopyableSnippet value={mcpUrl} label="Copy OAuth MCP endpoint" />
              </li>
              <li>
                Complete Ragtime sign-in and MFA if prompted, then authorize the app. The default{' '}
                <code>/mcp</code> route provides development tools; custom routes do not.
              </li>
              <li>Enable the app in a conversation and start using development tools.</li>
            </Steps>
          </section>
          <Verify workspaceId={workspaceId} />
          <Alternative summary="Troubleshooting" disclosure="chatgpt-troubleshooting">
            <p>
              If OAuth is unavailable, verify the Ragtime instance has it enabled for the default{' '}
              <code>/mcp</code> route (Settings → Enable MCP Server, Require authentication for
              default /mcp route, Authentication Method: OAuth2) and that your account has the
              required permissions.
            </p>
          </Alternative>
          <Docs href={docs} />
        </>
      ) : (
        <StandardGuide
          clientId={clientId}
          clientLabel={clientLabel}
          config={config!}
          env={env}
          name={name}
          mcpUrl={mcpUrl}
          workspaceId={workspaceId}
          docs={docs}
        />
      )}
    </section>
  );
}

function StandardGuide({
  clientId,
  clientLabel,
  config,
  env,
  name,
  mcpUrl,
  workspaceId,
  docs,
}: {
  clientId: Exclude<CodingAgentClientId, 'claude-desktop' | 'chatgpt'>;
  clientLabel: string;
  config: string;
  env: string;
  name: string;
  mcpUrl: string;
  workspaceId: string;
  docs: string;
}) {
  const isClaudeCode = clientId === 'claude-code';
  const isCursor = clientId === 'cursor';
  const locations =
    clientId === 'cline'
      ? [{ label: 'UI opener', path: 'MCP Servers → Configure → Configure MCP Servers' }]
      : clientId === 'continue'
        ? [
            { label: 'Standalone', path: '.continue/mcpServers/ragtime.yaml' },
            { label: 'User merge', path: '~/.continue/config.yaml' },
          ]
        : clientId === 'opencode'
          ? [
              { label: 'Global', path: '~/.config/opencode/opencode.json' },
              { label: 'Project', path: 'opencode.json' },
            ]
          : clientId === 'hermes'
            ? [
                { label: 'Configuration', path: '~/.hermes/config.yaml' },
                { label: 'Secrets', path: '~/.hermes/.env' },
              ]
            : clientId === 'codex'
              ? [
                  { label: 'User', path: '~/.codex/config.toml' },
                  { label: 'Trusted project', path: '.codex/config.toml' },
                ]
              : isCursor
                ? [
                    { label: 'Project', path: '.cursor/mcp.json' },
                    { label: 'User', path: '~/.cursor/mcp.json' },
                  ]
                : [
                    { label: 'Project', path: '.mcp.json' },
                    { label: 'User', path: '~/.claude.json' },
                  ];
  return (
    <>
      <section className="coding-agent-guide-section" data-guide-section="connect">
        <h5>Before you start</h5>
        {['claude-code', 'cursor', 'opencode', 'codex'].includes(clientId) && (
          <CredentialEnvironment env={env} client={clientLabel} />
        )}
        {clientId === 'cline' && (
          <p>
            Retrieve the credential in Development credentials below. In the copied Cline JSON,
            privately replace <code>{WORKSPACE_CREDENTIAL_PLACEHOLDER}</code>; Cline does not
            interpolate this environment variable.
          </p>
        )}
        {clientId === 'continue' && (
          <p>
            Create a private global <code>~/.continue/.env</code> file with{' '}
            <code>
              {env}={WORKSPACE_CREDENTIAL_PLACEHOLDER}
            </code>
            . Continue also loads workspace <code>.env</code> or <code>.continue/.env</code>; use
            those only when sharing access is intended. The copied <code>secrets.{env}</code> header
            resolves this same local secret. Use Agent mode.{' '}
            <a href={codingAgentOfficialDocs.continueSecrets} target="_blank" rel="noreferrer">
              Continue secrets documentation
            </a>
            .
          </p>
        )}
        {clientId === 'hermes' && (
          <p>
            In <code>~/.hermes/.env</code>, add{' '}
            <code>
              {env}={WORKSPACE_CREDENTIAL_PLACEHOLDER}
            </code>
            , then launch Hermes so it reads that file.
          </p>
        )}
        <h5>Connect</h5>
        {isClaudeCode ? (
          <>
            <Steps>
              <li>
                Use the current CLI for user scope, or use the project <code>.mcp.json</code>{' '}
                alternative.
              </li>
            </Steps>
            <CopyableSnippet
              value={`claude mcp add --scope user --transport http ${name} ${JSON.stringify(mcpUrl)} --header 'Authorization: Bearer \${${env}}'`}
              label="Copy Claude Code user-scope command"
            />
            <Alternative
              summary="Project configuration alternative"
              disclosure="claude-code-project-config"
            >
              <CopyableSnippet value={config} label="Copy Claude Code project configuration" />
            </Alternative>
          </>
        ) : isCursor ? (
          <>
            <a
              className="btn btn-primary btn-sm"
              href={buildCursorInstallLink(mcpUrl, workspaceId)}
            >
              Install in Cursor
            </a>
            <p>
              Review the install entry. Remote MCP does not support <code>envFile</code>; make this
              environment variable available to the running Cursor GUI.
            </p>
            <Alternative summary="Manual JSON alternative" disclosure="cursor-manual-config">
              <CopyableSnippet value={config} label="Copy Cursor configuration" />
            </Alternative>
          </>
        ) : (
          <CopyableSnippet value={config} label={`Copy ${clientLabel} configuration`} />
        )}
        <ConfigLocations locations={locations} />
      </section>
      <p className="coding-agent-guide-completion">
        {clientId === 'continue' ? (
          'Save the configuration, let Continue refresh, switch to Agent mode, and confirm the named MCP tools appear.'
        ) : clientId === 'cursor' ? (
          'Open Cursor Customize and MCP Logs, then confirm the named MCP server connects.'
        ) : clientId === 'cline' ? (
          'Save through the Configure MCP Servers opener, restart Cline or open a new task, and confirm the named tools appear.'
        ) : clientId === 'claude-code' ? (
          <>
            Run <code>claude mcp get {name}</code>, then confirm the named tools appear.
          </>
        ) : clientId === 'opencode' ? (
          <>
            Run <code>/mcps</code> or <code>opencode mcp list</code>, then confirm the named server
            appears.
          </>
        ) : clientId === 'hermes' ? (
          <>
            Run <code>/reload-mcp</code> then <code>hermes mcp test {name}</code>.
          </>
        ) : (
          <>
            Restart Codex, run <code>/mcp</code>, and confirm the named server appears.
          </>
        )}
      </p>
      <Verify workspaceId={workspaceId} />
      <Alternative summary="Troubleshooting" disclosure={`${clientId}-troubleshooting`}>
        <p>
          {clientId === 'cline' ? (
            <>
              Use <code>streamableHttp</code>, not Cline’s legacy SSE default; the UI Configure MCP
              Servers opener is authoritative.
            </>
          ) : clientId === 'continue' ? (
            <>
              Use Agent mode. The standalone YAML has metadata; when using the user config, merge
              only <code>mcpServers</code> and preserve other settings.
            </>
          ) : clientId === 'opencode' ? (
            <>
              Use V2 <code>mcp.servers</code>; legacy bootstrap format remains supported. Run{' '}
              <code>/mcps</code> or <code>opencode mcp list</code>.
            </>
          ) : clientId === 'hermes' ? (
            <>
              Use <code>{'${VAR}'}</code> in config, then run <code>/reload-mcp</code> and{' '}
              <code>hermes mcp test {name}</code>.
            </>
          ) : clientId === 'codex' ? (
            <>
              Ensure the launch environment contains the credential, trust the project config if
              used, then run <code>/mcp</code>.
            </>
          ) : isClaudeCode ? (
            <>
              Use <code>claude mcp get {name}</code> to inspect the configured server.
            </>
          ) : (
            <>Open MCP Logs after install to diagnose connection errors.</>
          )}
        </p>
      </Alternative>
      {nativeProfiles.has(clientId) && (
        <GuideNote>
          Native assisted setup is also available for {clientLabel}; it preserves unrelated local
          settings.
        </GuideNote>
      )}
      <Docs href={docs} />
    </>
  );
}
