import type { ChangeEvent } from 'react';
import type { ChatSendMode } from '@/utils/chatSendMode';

interface ChatSendModeControlProps {
  mode: ChatSendMode;
  onChange: (mode: ChatSendMode) => void;
  disabled?: boolean;
}

export function ChatSendModeControl({
  mode,
  onChange,
  disabled = false,
}: ChatSendModeControlProps) {
  const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    onChange(event.target.value as ChatSendMode);
  };

  return (
    <div className="chat-composer-send-mode" data-chat-composer-send-mode>
      <select
        aria-label="Send behavior"
        className="chat-composer-send-mode-select"
        disabled={disabled}
        onChange={handleChange}
        value={mode}
      >
        <option value="button">Button only</option>
        <option value="enter">Enter</option>
        <option value="ctrl-enter">Ctrl/⌘+Enter</option>
      </select>
    </div>
  );
}
