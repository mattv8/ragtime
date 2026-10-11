import type { ChatMessage, ContentProtectionDenialEvent } from '@/types';

const STORAGE_PREFIX = 'ragtime:denial-notices:v1:';
const MAX_SEEN_DENIALS = 200;

interface SeenDenialMap {
  version: 1;
  entries: Array<[conversationId: string, messageId: string]>;
}

const seenDenialsByViewer = new Map<string, SeenDenialMap>();

function storageKey(viewerId: string): string {
  return `${STORAGE_PREFIX}${encodeURIComponent(viewerId)}`;
}

function readSeenDenials(viewerId: string): SeenDenialMap {
  const cached = seenDenialsByViewer.get(viewerId);
  if (cached) return cached;
  const empty = { version: 1 as const, entries: [] as SeenDenialMap['entries'] };
  try {
    const stored = localStorage.getItem(storageKey(viewerId));
    if (!stored) {
      seenDenialsByViewer.set(viewerId, empty);
      return empty;
    }
    const parsed: unknown = JSON.parse(stored);
    if (
      !parsed ||
      typeof parsed !== 'object' ||
      !('version' in parsed) ||
      parsed.version !== 1 ||
      !('entries' in parsed) ||
      !Array.isArray(parsed.entries)
    ) {
      seenDenialsByViewer.set(viewerId, empty);
      return empty;
    }
    const seen: SeenDenialMap = {
      version: 1,
      entries: parsed.entries.filter(
        (entry): entry is [string, string] =>
          Array.isArray(entry) &&
          entry.length === 2 &&
          typeof entry[0] === 'string' &&
          typeof entry[1] === 'string',
      ),
    };
    seenDenialsByViewer.set(viewerId, seen);
    return seen;
  } catch {
    seenDenialsByViewer.set(viewerId, empty);
    return empty;
  }
}

function writeSeenDenials(viewerId: string, entries: SeenDenialMap['entries']): void {
  seenDenialsByViewer.set(viewerId, { version: 1, entries });
  try {
    localStorage.setItem(storageKey(viewerId), JSON.stringify({ version: 1, entries }));
  } catch {
    // Seen state is a local UX enhancement; unavailable storage must not affect chat.
  }
}

export function getDenialNoticeSeenId(viewerId: string, conversationId: string): string | null {
  return readSeenDenials(viewerId).entries.find(([id]) => id === conversationId)?.[1] ?? null;
}

export function markDenialNoticeSeen(
  viewerId: string,
  conversationId: string,
  messageId: string,
): void {
  const entries = readSeenDenials(viewerId).entries.filter(([id]) => id !== conversationId);
  entries.push([conversationId, messageId]);
  writeSeenDenials(viewerId, entries.slice(-MAX_SEEN_DENIALS));
}

export function clearDenialNoticeSeen(viewerId: string, conversationId: string): void {
  const entries = readSeenDenials(viewerId).entries.filter(([id]) => id !== conversationId);
  writeSeenDenials(viewerId, entries);
}

export function invalidateDenialNoticeSeen(viewerId?: string): void {
  if (viewerId) seenDenialsByViewer.delete(viewerId);
  else seenDenialsByViewer.clear();
}

export function getContentProtectionDenialEvent(
  message: ChatMessage,
): ContentProtectionDenialEvent | null {
  return (
    message.events?.find(
      (event): event is ContentProtectionDenialEvent => event.type === 'content_protection_denial',
    ) ?? null
  );
}

export function getLatestDenialMessageId(messages: ChatMessage[]): string | null {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.message_id && getContentProtectionDenialEvent(message)) return message.message_id;
  }
  return null;
}

export function hasDenialRequestId(messages: ChatMessage[], requestId: string): boolean {
  return messages.some(
    (message) => getContentProtectionDenialEvent(message)?.request_id === requestId,
  );
}

export function isDenialOnlyMessage(message: ChatMessage): boolean {
  return (
    message.role === 'assistant' &&
    message.events?.length === 1 &&
    message.events[0].type === 'content_protection_denial'
  );
}
