import { expect, test } from 'vitest';
import { readFileSync } from 'node:fs';
import path from 'node:path';

const chatPanelPath = path.resolve(import.meta.dirname, '../src/components/ChatPanel.tsx');
const chatPanelSource = readFileSync(chatPanelPath, 'utf8');

test('subagent handoff tool segments are collapsed before rendering', () => {
  expect(chatPanelSource).toMatch(
    /function pushSubagentToolSegment\(segments: StreamingSegment\[\], toolCall: ActiveToolCall\)/,
  );
  expect(chatPanelSource).toMatch(
    /toolCall\.tool === SUBAGENT_HANDOFF_TOOL_ID[\s\S]*?segments\.splice\(previousHandoffIndex, 1\)[\s\S]*?segments\.push\(\{ type: 'tool', toolCall \}\)/,
  );

  const helperUses = chatPanelSource.match(/pushSubagentToolSegment\(/g) ?? [];
  expect(helperUses.length).toBe(4);
});

test('the string-match handoff content dedupe apparatus is fully removed', () => {
  for (const deadSymbol of [
    'shouldSuppressDuplicateSubagentHandoffContent',
    'getLatestSubagentHandoffOutput',
    'SubagentHandoffLike',
    'latestSubagentHandoffOutput',
    'lastSubagentHandoffIndex',
  ]) {
    expect(chatPanelSource).not.toMatch(new RegExp(deadSymbol));
  }
});

test('the parent message never renders the subagent handoff as a standalone card', () => {
  // Parent streaming builder (consolidatedSegments) skips the handoff tool event;
  // it is shown inside the spawn_subagents subagent card instead.
  expect(chatPanelSource).toMatch(
    /if \(ev\.toolCall\.tool === SUBAGENT_HANDOFF_TOOL_ID\) \{\s*continue;\s*\}/,
  );
  // Parent saved-message renderer skips the handoff tool event too.
  expect(chatPanelSource).toMatch(
    /ev\.tool === SUBAGENT_HANDOFF_TOOL_ID\s*\)\s*\{\s*continue;\s*\}/,
  );
});

test('the handoff card is still rendered inside the subagent transcript', () => {
  expect(chatPanelSource).toMatch(
    /if \(segment\.type === 'tool' && segment\.toolCall\?\.tool === SUBAGENT_HANDOFF_TOOL_ID\) \{[\s\S]*?<SubAgentHandoffDisplay/,
  );
  expect(chatPanelSource).toMatch(/<div className="subagent-handoff-output">/);
});

test('parent final content is always rendered after a subagent task', () => {
  // No suppression gate on the parent saved-message final-content branch.
  expect(chatPanelSource).toMatch(
    /channel === 'final' &&\s*ev\.type === 'content'\s*\)\s*\{\s*result\.push\(/,
  );
});

test('subagent handoffs do not render as inline reasoning tools', () => {
  // Streaming builder skips the handoff before the reasoning-part branch, so the
  // reasoning guard only needs to exclude the spawn_subagents card + visualizations.
  expect(chatPanelSource).toMatch(
    /currentReasoning &&\s*ev\.toolCall\.tool !== WORKSPACE_SUBAGENTS_TOOL_ID &&\s*!isVisualizationToolCall\(ev\.toolCall\)/,
  );
  expect(chatPanelSource).toMatch(
    /pendingReasoning &&\s*ev\.tool !== SUBAGENT_HANDOFF_TOOL_ID &&\s*ev\.tool !== WORKSPACE_SUBAGENTS_TOOL_ID &&\s*!isVisualizationToolName\(ev\.tool\)/,
  );
});

test('active subagent runs render at the spawn_subagents stream position', () => {
  const streamingMessageMatch = chatPanelSource.match(
    /\{\/\* Streaming assistant message[\s\S]*?<div className="chat-message-streaming">/,
  );
  expect(streamingMessageMatch).toBeTruthy();

  const streamingMessageBlock = streamingMessageMatch[0];
  expect(streamingMessageBlock).not.toMatch(
    /segment\.toolCall\?\.tool === WORKSPACE_SUBAGENTS_TOOL_ID[\s\S]*?return null;/,
  );
  expect(streamingMessageBlock).toMatch(
    /segment\.type === 'tool'[\s\S]*?segment\.toolCall\?\.tool === WORKSPACE_SUBAGENTS_TOOL_ID[\s\S]*?<div[\s\S]*?className="chat-subagent-active-runs"/,
  );
});
