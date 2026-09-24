import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { CodingAgentClientGuide } from './CodingAgentClientGuide';
import type { CodingAgentClientId } from './codingAgentClients';

const clients: Array<[CodingAgentClientId, RegExp]> = [
  ['claude-desktop', /mcp-remote/],
  ['claude-code', /claude mcp add/],
  ['cursor', /Install in Cursor/],
  ['chatgpt', /ChatGPT OAuth route only/],
  ['cline', /streamableHttp/],
  ['continue', /requestOptions/],
  ['opencode', /opencode mcp list/],
  ['hermes', /hermes mcp test/],
  ['codex', /bearer_token_env_var/],
];

describe('CodingAgentClientGuide', () => {
  afterEach(cleanup);

  it.each(clients)('renders the distinct %s guide', (clientId, expected) => {
    render(
      <CodingAgentClientGuide
        clientId={clientId}
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.getByLabelText(/setup guide/i).dataset.clientGuide).toBe(clientId);
    expect(screen.getAllByText(expected).length).toBeGreaterThan(0);
  });

  it('does not present a secret-bearing copy prompt for ChatGPT', () => {
    render(
      <CodingAgentClientGuide
        clientId="chatgpt"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.queryByRole('button', { name: /copy/i })).toBeNull();
    expect(screen.queryByText(/YOUR_WORKSPACE_CREDENTIAL/)).toBeNull();
  });

  it('provides Desktop JSON configuration and OAuth-only alternative', () => {
    render(
      <CodingAgentClientGuide
        clientId="claude-desktop"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.getByRole('button', { name: /copy claude desktop configuration/i })).toBeTruthy();
    expect(screen.getByText('Copy Claude Desktop configuration')).toBeTruthy();
    expect(screen.getByText('macOS')).toBeTruthy();
    expect(screen.getByText('Windows')).toBeTruthy();
    expect(screen.getByText(/OAuth route only/i)).toBeTruthy();
  });

  it('keeps credential environment guidance in a labelled disclosure when required', () => {
    render(
      <CodingAgentClientGuide
        clientId="codex"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.getByText('Private credential environment')).toBeTruthy();
    expect(screen.getByText('User')).toBeTruthy();
  });
});
