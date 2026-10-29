import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import { CodingAgentOAuthConnection } from './CodingAgentOAuthConnection';

describe('CodingAgentOAuthConnection', () => {
  afterEach(cleanup);

  it('uses OAuth-native OpenCode setup without workspace credentials', () => {
    render(
      <CodingAgentOAuthConnection clientId="opencode" mcpUrl="https://ragtime.example.test/mcp" />,
    );

    expect(screen.getByText(/opencode mcp add ragtime --global --url/)).toBeTruthy();
    expect(screen.getAllByText('/mcps')).toHaveLength(2);
    expect(screen.queryByText(/oauth:false|WORKSPACE_CREDENTIAL|environment variable/i)).toBeNull();
  });

  it.each(['claude-desktop', 'chatgpt'] as const)(
    'keeps %s account setup free of credential placeholders',
    (clientId) => {
      render(
        <CodingAgentOAuthConnection
          clientId={clientId}
          mcpUrl="https://ragtime.example.test/mcp"
        />,
      );

      expect(screen.queryByRole('textbox')).toBeNull();
      expect(
        screen.queryByText(/WORKSPACE_CREDENTIAL|oauth:false|Authorization: Bearer/i),
      ).toBeNull();
    },
  );
});
