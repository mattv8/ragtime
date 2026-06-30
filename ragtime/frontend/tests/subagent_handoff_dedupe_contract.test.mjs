import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import test from 'node:test';

const chatPanelPath = path.resolve(import.meta.dirname, '../src/components/ChatPanel.tsx');
const chatPanelSource = readFileSync(chatPanelPath, 'utf8');

test('subagent handoff tool segments are collapsed before rendering', () => {
  assert.match(
    chatPanelSource,
    /function pushSubagentToolSegment\(segments: StreamingSegment\[\], toolCall: ActiveToolCall\)/,
    'ChatPanel should keep handoff segment dedupe in one helper',
  );
  assert.match(
    chatPanelSource,
    /toolCall\.tool === SUBAGENT_HANDOFF_TOOL_ID[\s\S]*?segments\.splice\(previousHandoffIndex, 1\)[\s\S]*?segments\.push\(\{ type: 'tool', toolCall \}\)/,
    'duplicate handoff segments should be replaced with the latest handoff',
  );

  const helperUses = chatPanelSource.match(/pushSubagentToolSegment\(/g) ?? [];
  assert.equal(
    helperUses.length,
    4,
    'helper should be defined once and used by all three segment builders',
  );
});

test('the string-match handoff content dedupe apparatus is fully removed', () => {
  for (const deadSymbol of [
    'shouldSuppressDuplicateSubagentHandoffContent',
    'getLatestSubagentHandoffOutput',
    'SubagentHandoffLike',
    'latestSubagentHandoffOutput',
    'lastSubagentHandoffIndex',
  ]) {
    assert.doesNotMatch(
      chatPanelSource,
      new RegExp(deadSymbol),
      `${deadSymbol} is unnecessary fluff and must not remain in ChatPanel`,
    );
  }
});

test('the parent message never renders the subagent handoff as a standalone card', () => {
  // Parent streaming builder (consolidatedSegments) skips the handoff tool event;
  // it is shown inside the spawn_subagents subagent card instead.
  assert.match(
    chatPanelSource,
    /if \(ev\.toolCall\.tool === SUBAGENT_HANDOFF_TOOL_ID\) \{\s*continue;\s*\}/,
    'parent streaming builder should skip the standalone subagent handoff segment',
  );
  // Parent saved-message renderer skips the handoff tool event too.
  assert.match(
    chatPanelSource,
    /ev\.tool === SUBAGENT_HANDOFF_TOOL_ID\s*\)\s*\{\s*continue;\s*\}/,
    'parent saved-message renderer should skip the standalone subagent handoff event',
  );
});

test('the handoff card is still rendered inside the subagent transcript', () => {
  assert.match(
    chatPanelSource,
    /if \(segment\.type === 'tool' && segment\.toolCall\?\.tool === SUBAGENT_HANDOFF_TOOL_ID\) \{[\s\S]*?<SubAgentHandoffDisplay/,
    'StreamingSegmentDisplay should keep rendering the handoff card (subagent-handoff-output container) for subagent transcripts',
  );
  assert.match(
    chatPanelSource,
    /<div className="subagent-handoff-output">/,
    'the subagent-handoff-output container should still exist',
  );
});

test('parent final content is always rendered after a subagent task', () => {
  // No suppression gate on the parent saved-message final-content branch.
  assert.match(
    chatPanelSource,
    /channel === 'final' &&\s*ev\.type === 'content'\s*\)\s*\{\s*result\.push\(/,
    'saved-message renderer should render parent final content unconditionally',
  );
});

test('subagent handoffs do not render as inline reasoning tools', () => {
  // Streaming builder skips the handoff before the reasoning-part branch, so the
  // reasoning guard only needs to exclude the spawn_subagents card + visualizations.
  assert.match(
    chatPanelSource,
    /currentReasoning &&\s*ev\.toolCall\.tool !== WORKSPACE_SUBAGENTS_TOOL_ID &&\s*!isVisualizationToolCall\(ev\.toolCall\)/,
    'active stream builder should keep subagent cards out of reasoning parts',
  );
  assert.match(
    chatPanelSource,
    /pendingReasoning &&\s*ev\.tool !== SUBAGENT_HANDOFF_TOOL_ID &&\s*ev\.tool !== WORKSPACE_SUBAGENTS_TOOL_ID &&\s*!isVisualizationToolName\(ev\.tool\)/,
    'saved-message renderer should force subagent events out of reasoning parts',
  );
});

test('active subagent runs render at the spawn_subagents stream position', () => {
  const streamingMessageMatch = chatPanelSource.match(
    /\{\/\* Streaming assistant message[\s\S]*?<div className="chat-message-streaming">/,
  );
  assert.ok(streamingMessageMatch, 'streaming assistant message renderer should be present');

  const streamingMessageBlock = streamingMessageMatch[0];
  assert.doesNotMatch(
    streamingMessageBlock,
    /segment\.toolCall\?\.tool === WORKSPACE_SUBAGENTS_TOOL_ID[\s\S]*?return null;/,
    'streaming renderer must not drop the spawn_subagents segment and append runs later',
  );
  assert.match(
    streamingMessageBlock,
    /segment\.type === 'tool'[\s\S]*?segment\.toolCall\?\.tool === WORKSPACE_SUBAGENTS_TOOL_ID[\s\S]*?<div[\s\S]*?className="chat-subagent-active-runs"/,
    'active subagent cards should be anchored inline at the spawn_subagents segment',
  );
});
