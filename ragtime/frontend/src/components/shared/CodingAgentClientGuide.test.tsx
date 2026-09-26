import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';
import { CodingAgentClientGuide } from './CodingAgentClientGuide';
import { WORKSPACE_CREDENTIAL_PLACEHOLDER } from './codingAgentConfigs';
import type { CodingAgentClientId } from './codingAgentClients';

const clients: Array<[CodingAgentClientId, RegExp]> = [
  ['claude-desktop', /Claude Cowork/],
  ['claude-code', /claude mcp add/],
  ['cursor', /Install in Cursor/],
  ['chatgpt', /ChatGPT Plugins/],
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

  it('provides OAuth MCP endpoint and safe verification for ChatGPT', () => {
    render(
      <CodingAgentClientGuide
        clientId="chatgpt"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.getByRole('button', { name: /copy oauth mcp endpoint/i })).toBeTruthy();
    expect(screen.getByRole('button', { name: /copy safe verification prompt/i })).toBeTruthy();
    expect(screen.getByText(/Choose Authentication:/)).toBeTruthy();
    expect(screen.getByText(/Complete Ragtime sign-in and MFA if prompted/)).toBeTruthy();
    expect(screen.getByText(/No workspace credential is needed/)).toBeTruthy();
    expect(screen.queryByText(/YOUR_WORKSPACE_CREDENTIAL/)).toBeNull();
  });

  it('provides OAuth MCP endpoint copy for Claude and keeps advanced alternatives collapsed', () => {
    render(
      <CodingAgentClientGuide
        clientId="claude-desktop"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.getByRole('button', { name: /copy oauth mcp endpoint/i })).toBeTruthy();
    expect(screen.getByText(/Use Claude's published identity/i)).toBeTruthy();
    // Advanced workspace credential warning should be in collapsed details
    const advancedDetails = screen
      .getByText(/Advanced: connect with a workspace credential/i)
      .closest('details');
    expect(advancedDetails?.open).toBe(false);
    expect(advancedDetails?.textContent).toContain('limited-organization beta');
    expect(screen.getByText(/Desktop-only local bridge/i).closest('details')?.open).toBe(false);
    expect(screen.getByRole('link', { name: /illustrated remote mcp/i })).toBeTruthy();
  });

  it('provides a source fallback when a Claude example image fails to load', () => {
    render(
      <CodingAgentClientGuide
        clientId="claude-desktop"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    const image = screen.getByAltText(/Older Claude Add custom connector/i);
    fireEvent.error(image);
    expect(screen.getByText(/example image could not load/i)).toBeTruthy();
    expect(screen.getByRole('link', { name: /open the illustrated source instead/i })).toBeTruthy();
  });

  it('uses a safe copyable verification prompt with the actual workspace', () => {
    render(
      <CodingAgentClientGuide
        clientId="cline"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(
      screen.getByText(/Call workspace_development_context for workspace workspace-1/),
    ).toBeTruthy();
    expect(screen.getByText(/Cline does not interpolate/i)).toBeTruthy();
  });

  it('keeps credential environment guidance in a labelled disclosure when required', () => {
    render(
      <CodingAgentClientGuide
        clientId="codex"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    expect(screen.getByText(/private credential/i)).toBeTruthy();
    expect(screen.getByText('User')).toBeTruthy();
  });

  it('keeps workspace credential instructions collapsed by default in Claude guide', () => {
    render(
      <CodingAgentClientGuide
        clientId="claude-desktop"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    const advancedDetails = screen
      .getByText(/Advanced: connect with a workspace credential/i)
      .closest('details');
    expect(advancedDetails?.open).toBe(false);
    // Bearer token instruction should not be visible until opened
    fireEvent.click(advancedDetails!.querySelector('summary')!);
    expect(advancedDetails?.open).toBe(true);
    // Verify the advanced section contains credential instructions
    expect(advancedDetails?.textContent).toContain('Add request header');
    expect(advancedDetails?.textContent).toContain(WORKSPACE_CREDENTIAL_PLACEHOLDER);
  });

  it('copies the correct OAuth MCP endpoint URL for Claude to clipboard', async () => {
    const user = userEvent.setup();
    render(
      <CodingAgentClientGuide
        clientId="claude-desktop"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    const copyButton = screen.getByRole('button', { name: /copy oauth mcp endpoint/i });
    await user.click(copyButton);
    expect(await navigator.clipboard.readText()).toBe('https://ragtime.example.test/mcp');
  });

  it('copies the correct OAuth MCP endpoint URL for ChatGPT to clipboard', async () => {
    const user = userEvent.setup();
    render(
      <CodingAgentClientGuide
        clientId="chatgpt"
        mcpUrl="https://ragtime.example.test/mcp"
        workspaceId="workspace-1"
      />,
    );
    const copyButton = screen.getByRole('button', { name: /copy oauth mcp endpoint/i });
    await user.click(copyButton);
    expect(await navigator.clipboard.readText()).toBe('https://ragtime.example.test/mcp');
  });
});
