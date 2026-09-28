import { useCallback, useEffect, useState } from 'react';
import { DEFAULT_CHAT_SEND_MODE, isChatSendMode, type ChatSendMode } from '@/utils/chatSendMode';

const CHANGE_EVENT = 'ragtime:composer:send-mode-change';
const storageKey = (userId: string) => `ragtime:composer:sendMode:${encodeURIComponent(userId)}`;

interface ChatSendModeState {
  key: string;
  mode: ChatSendMode;
}

function readMode(userId: string): ChatSendMode {
  try {
    const value = window.localStorage.getItem(storageKey(userId));
    return isChatSendMode(value) ? value : DEFAULT_CHAT_SEND_MODE;
  } catch {
    return DEFAULT_CHAT_SEND_MODE;
  }
}

export function useChatSendMode(
  userId: string,
): readonly [ChatSendMode, (mode: ChatSendMode) => void] {
  const key = storageKey(userId);
  const [state, setState] = useState<ChatSendModeState>(() => ({
    key,
    mode: readMode(userId),
  }));
  // Effects update state after paint. Read the new user's value during render so
  // a user switch can never momentarily use the previous user's shortcut.
  const mode = state.key === key ? state.mode : readMode(userId);

  useEffect(() => {
    setState({ key, mode: readMode(userId) });

    const onStorage = (event: StorageEvent) => {
      if (event.key === null) {
        setState({ key, mode: DEFAULT_CHAT_SEND_MODE });
      } else if (event.key === key) {
        setState({
          key,
          mode: isChatSendMode(event.newValue) ? event.newValue : DEFAULT_CHAT_SEND_MODE,
        });
      }
    };
    const onChange = (event: Event) => {
      const detail = (event as CustomEvent<{ key: string; mode: ChatSendMode }>).detail;
      if (detail?.key === key) {
        setState({
          key,
          mode: isChatSendMode(detail.mode) ? detail.mode : DEFAULT_CHAT_SEND_MODE,
        });
      }
    };

    window.addEventListener('storage', onStorage);
    window.addEventListener(CHANGE_EVENT, onChange);
    return () => {
      window.removeEventListener('storage', onStorage);
      window.removeEventListener(CHANGE_EVENT, onChange);
    };
  }, [key, userId]);

  const updateMode = useCallback(
    (nextMode: ChatSendMode) => {
      if (!isChatSendMode(nextMode)) return;
      const key = storageKey(userId);
      setState({ key, mode: nextMode });
      try {
        window.localStorage.setItem(key, nextMode);
      } catch {
        // Private browsing and denied storage still allow an in-memory preference.
      }
      window.dispatchEvent(new CustomEvent(CHANGE_EVENT, { detail: { key, mode: nextMode } }));
    },
    [userId],
  );

  return [mode, updateMode] as const;
}
