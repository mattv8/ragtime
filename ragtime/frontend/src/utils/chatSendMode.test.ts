import { describe, expect, it } from 'vitest';
import { DEFAULT_CHAT_SEND_MODE, shouldSubmitChatKey } from './chatSendMode';

describe('chat send mode', () => {
  const key = (overrides: Partial<KeyboardEvent> = {}) =>
    ({ key: 'Enter', ...overrides }) as KeyboardEvent;
  it('defaults to Ctrl/Meta+Enter and recognizes only its configured shortcut', () => {
    expect(DEFAULT_CHAT_SEND_MODE).toBe('ctrl-enter');
    expect(shouldSubmitChatKey(key({ ctrlKey: true }), 'ctrl-enter')).toBe(true);
    expect(shouldSubmitChatKey(key({ metaKey: true }), 'ctrl-enter')).toBe(true);
    expect(shouldSubmitChatKey(key(), 'ctrl-enter')).toBe(false);
  });
  it('keeps button mode keyboard-only and Enter mode unmodified', () => {
    expect(shouldSubmitChatKey(key(), 'button')).toBe(false);
    expect(shouldSubmitChatKey(key(), 'enter')).toBe(true);
    expect(shouldSubmitChatKey(key({ ctrlKey: true }), 'enter')).toBe(false);
  });
  it('never submits Shift/Alt combinations, IME composition, or repeats', () => {
    for (const mode of ['button', 'enter', 'ctrl-enter'] as const) {
      expect(shouldSubmitChatKey(key({ shiftKey: true }), mode)).toBe(false);
      expect(shouldSubmitChatKey(key({ altKey: true, ctrlKey: true }), mode)).toBe(false);
      expect(shouldSubmitChatKey(key({ isComposing: true, ctrlKey: true }), mode)).toBe(false);
      expect(shouldSubmitChatKey(key({ keyCode: 229, ctrlKey: true }), mode)).toBe(false);
      expect(shouldSubmitChatKey(key({ repeat: true, ctrlKey: true }), mode)).toBe(false);
    }
  });
});
