export type CodingAgentClientId =
  | 'claude-desktop'
  | 'claude-code'
  | 'cursor'
  | 'chatgpt'
  | 'cline'
  | 'continue'
  | 'opencode'
  | 'hermes'
  | 'codex';

export interface CodingAgentSelectionRequest {
  requestId: number;
  clientId: CodingAgentClientId;
  workspaceId: string;
}

export const CODING_AGENT_CLIENTS: readonly {
  id: CodingAgentClientId;
  label: string;
}[] = [
  { id: 'claude-desktop', label: 'Claude Cowork / Desktop' },
  { id: 'claude-code', label: 'Claude Code' },
  { id: 'cursor', label: 'Cursor' },
  { id: 'chatgpt', label: 'ChatGPT' },
  { id: 'cline', label: 'Cline' },
  { id: 'continue', label: 'Continue' },
  { id: 'opencode', label: 'OpenCode' },
  { id: 'hermes', label: 'Hermes' },
  { id: 'codex', label: 'Codex' },
];

export function getCodingAgentClientLabel(clientId: CodingAgentClientId): string {
  return CODING_AGENT_CLIENTS.find((client) => client.id === clientId)!.label;
}
