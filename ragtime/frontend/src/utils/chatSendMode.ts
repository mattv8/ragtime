export type ChatSendMode = 'button' | 'enter' | 'ctrl-enter';

export const DEFAULT_CHAT_SEND_MODE: ChatSendMode = 'ctrl-enter';

export function isChatSendMode(value: unknown): value is ChatSendMode {
  return value === 'button' || value === 'enter' || value === 'ctrl-enter';
}

export function shouldSubmitChatKey(
  event: Pick<
    KeyboardEvent,
    'key' | 'altKey' | 'ctrlKey' | 'isComposing' | 'keyCode' | 'metaKey' | 'repeat' | 'shiftKey'
  >,
  mode: ChatSendMode,
): boolean {
  if (
    event.key !== 'Enter' ||
    event.shiftKey ||
    event.altKey ||
    event.isComposing ||
    event.keyCode === 229 ||
    event.repeat
  ) {
    return false;
  }

  if (mode === 'enter') return !event.ctrlKey && !event.metaKey;
  if (mode === 'ctrl-enter') return Boolean(event.ctrlKey || event.metaKey);
  return false;
}
