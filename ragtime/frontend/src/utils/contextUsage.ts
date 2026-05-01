import type { ChatMessage } from '@/types';
import { parseScopedModelIdentifier } from './modelProviders';

const CHARS_PER_TOKEN = 4;

export interface ParsedStoredModel {
  provider?: string;
  modelId: string;
}

export function parseStoredModelIdentifier(storedModel: string): ParsedStoredModel {
  const parsed = parseScopedModelIdentifier(storedModel);
  return parsed.provider
    ? { provider: parsed.provider, modelId: parsed.modelId }
    : { modelId: parsed.modelId };
}

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
  return estimateTokens(content || '');
}

export function calculateConversationTokens(messages: ChatMessage[]): number {
  return messages.reduce((total, message) => total + calculateMessageTokens(message), 0);
}

export type StreamingRenderEvent =
  | { type: 'content'; content: string }
  | { type: 'tool'; toolCall: { input?: Record<string, unknown>; output?: string } }
  | { type: 'reasoning'; content: string };

export interface ConversationContextUsage {
  currentTokens: number;
  totalTokens: number;
  contextLimit: number;
  contextUsagePercent: number;
  projectedInputPercent: number;
  hasHeadroom: boolean;
}

interface ConversationContextUsageParams {
  messages: ChatMessage[];
  persistedConversationTokens?: number | null;
  contextLimit: number;
  inputText?: string;
  isStreaming?: boolean;
  streamingEvents?: StreamingRenderEvent[];
  streamingContent?: string;
}

export function calculateConversationContextUsage({
  messages,
  persistedConversationTokens,
  contextLimit,
  inputText = '',
  isStreaming = false,
  streamingEvents = [],
  streamingContent = '',
}: ConversationContextUsageParams): ConversationContextUsage {
  const estimatedConversationTokens = calculateConversationTokens(messages);
  const persistedTokens = Math.max(0, persistedConversationTokens || 0);
  const currentTokens = persistedTokens > 0
    ? persistedTokens
    : estimatedConversationTokens;
  const streamingTokens = isStreaming
    ? calculateStreamingTokens(streamingEvents, streamingContent)
    : 0;
  const totalTokens = currentTokens + streamingTokens;
  const safeContextLimit = contextLimit > 0 ? contextLimit : 1;
  const contextUsagePercent = Math.round((totalTokens / safeContextLimit) * 100);
  const nextMessageTokens = estimateTokens(inputText.trim());
  const projectedInputPercent = Math.round(((totalTokens + nextMessageTokens) / safeContextLimit) * 100);
  const hasHeadroom = totalTokens + nextMessageTokens <= safeContextLimit * 0.9;

  return {
    currentTokens,
    totalTokens,
    contextLimit: safeContextLimit,
    contextUsagePercent,
    projectedInputPercent,
    hasHeadroom,
  };
}

export function calculateStreamingTokens(events: StreamingRenderEvent[], streamingContent: string): number {
  if (events.length > 0) {
    let tokens = 0;
    for (const event of events) {
      if (event.type === 'content' || event.type === 'reasoning') {
        tokens += estimateTokens(event.content || '');
      } else if (event.type === 'tool') {
        tokens += estimateTokensFromObject(event.toolCall.input);
        tokens += estimateTokens(event.toolCall.output || '');
      }
    }
    return tokens;
  }

  return estimateTokens(streamingContent || '');
}
