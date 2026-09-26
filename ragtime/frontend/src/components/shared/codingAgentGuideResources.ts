export const codingAgentOfficialDocs = {
  claude: 'https://claude.com/docs/connectors/custom/add-unlisted',
  claudeCode: 'https://code.claude.com/docs/en/mcp',
  cursor: 'https://cursor.com/docs/context/mcp',
  chatgpt: 'https://platform.openai.com/docs/guides/developer-mode',
  cline: 'https://docs.cline.bot/mcp/configuring-mcp-servers',
  continue: 'https://docs.continue.dev/customize/deep-dives/mcp',
  continueSecrets:
    'https://docs.continue.dev/guides/configuring-models-rules-tools#working-with-secrets',
  opencode: 'https://opencode.ai/v2/docs/mcp-servers',
  hermes: 'https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp',
  codex: 'https://developers.openai.com/codex/mcp/',
  mcpLocal: 'https://modelcontextprotocol.io/docs/develop/connect-local-servers',
  mcpRemoteHeaders: 'https://github.com/punkpeye/mcp-remote#custom-headers',
} as const;

export const claudeIllustratedExample = {
  href: 'https://modelcontextprotocol.io/docs/develop/connect-remote-servers',
  label: 'Illustrated remote MCP connector documentation',
  source: 'Model Context Protocol contributors, CC BY 4.0 (checked 2026-09-25)',
};

export const claudeSetupImages = {
  remote: {
    src: '/agent-setup/claude-remote-connector.png',
    alt: 'Older Claude Add custom connector dialog with an example remote server URL and connector trust warning.',
    caption:
      'Older Claude layout: an example only. Model Context Protocol contributors, CC BY 4.0, checked 2026-09-25. Use the current field values above for authentication.',
    href: claudeIllustratedExample.href,
  },
  desktop: {
    src: '/agent-setup/claude-desktop-edit-config.png',
    alt: 'Older Claude Desktop Developer settings panel with the Edit Config button for local MCP services.',
    caption:
      'Older Claude Desktop layout. Anthropic, PBC and contributors, MIT, checked 2026-09-25.',
    href: 'https://modelcontextprotocol.io/docs/develop/connect-local-servers',
  },
} as const;
