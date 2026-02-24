import type { ChatMessage } from '@/types';

const CHARS_PER_TOKEN = 4;

export function estimateTokens(text: string): number {
  return Math.ceil(text.length / CHARS_PER_TOKEN);
}

export function estimateTokensFromObject(value: unknown): number {
  if (value === undefined || value === null) return 0;
  try {
    return estimateTokens(JSON.stringify(value));
  } catch {
    return estimateTokens(String(value));
  }
}

export function calculateMessageTokens(msg: ChatMessage): number {
  if (msg.events?.length) {
    let tokens = 0;
    for (const event of msg.events) {
      if (event.type === 'content') {
        tokens += estimateTokens(event.content || '');
      } else if (event.type === 'tool') {
        tokens += estimateTokensFromObject(event.input);
        tokens += estimateTokens(event.output || '');
      }
    }
    return tokens;
  }

  const content = typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content);
  let tokens = estimateTokens(content || '');
  if (msg.tool_calls?.length) {
    for (const toolCall of msg.tool_calls) {
      tokens += estimateTokensFromObject(toolCall.input);
      tokens += estimateTokens(toolCall.output || '');
    }
  }
  return tokens;
}

export function calculateConversationTokens(messages: ChatMessage[]): number {
  return messages.reduce((total, message) => total + calculateMessageTokens(message), 0);
}

export type StreamingRenderEvent =
  | { type: 'content'; content: string }
  | { type: 'tool'; toolCall: { input?: Record<string, unknown>; output?: string } };

export function calculateStreamingTokens(events: StreamingRenderEvent[], streamingContent: string): number {
  if (events.length > 0) {
    let tokens = 0;
    for (const event of events) {
      if (event.type === 'content') {
        tokens += estimateTokens(event.content || '');
      } else {
        tokens += estimateTokensFromObject(event.toolCall.input);
        tokens += estimateTokens(event.toolCall.output || '');
      }
    }
    return tokens;
  }

  return estimateTokens(streamingContent || '');
}
