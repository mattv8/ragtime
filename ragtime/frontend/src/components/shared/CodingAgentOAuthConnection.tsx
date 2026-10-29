import { CopyableSnippet } from './CodingAgentSetupPrimitives';

export type OAuthCodingAgentClientId = 'claude-desktop' | 'chatgpt' | 'opencode';

interface CodingAgentOAuthConnectionProps {
  clientId: OAuthCodingAgentClientId;
  mcpUrl: string;
}

function Steps({ children }: { children: React.ReactNode }) {
  return <ol className="coding-agent-guide-steps">{children}</ol>;
}

export function CodingAgentOAuthConnection({ clientId, mcpUrl }: CodingAgentOAuthConnectionProps) {
  if (clientId === 'claude-desktop') {
    return (
      <>
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
            Leave <strong>Use Claude&apos;s published identity</strong> selected.{' '}
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
      </>
    );
  }

  if (clientId === 'chatgpt') {
    return (
      <Steps>
        <li>Open Settings → Security and login → Developer mode, then enable it if available.</li>
        <li>Open ChatGPT Plugins, select plus, and create a developer app using OAuth.</li>
        <li>
          Paste the OAuth MCP endpoint:{' '}
          <CopyableSnippet value={mcpUrl} label="Copy OAuth MCP endpoint" />
        </li>
        <li>
          Complete Ragtime sign-in and MFA if prompted, then enable the app in a conversation.
        </li>
      </Steps>
    );
  }

  const command = `opencode mcp add ragtime --global --url ${JSON.stringify(mcpUrl)}`;
  return (
    <Steps>
      <li>
        Run this command in a local terminal:
        <CopyableSnippet value={command} label="Copy OpenCode OAuth connection command" />
      </li>
      <li>
        Open OpenCode and use <code>/mcps</code> to sign in to Ragtime.
      </li>
      <li>
        Use <code>/mcps</code> or <code>opencode mcp list</code> to verify the connection status.
      </li>
    </Steps>
  );
}
