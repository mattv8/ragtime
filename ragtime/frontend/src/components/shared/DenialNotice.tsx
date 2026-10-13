import { memo, useLayoutEffect, useRef } from 'react';
import { ShieldOff } from 'lucide-react';

import { formatChatTimestamp } from '@/utils';

interface DenialNoticeProps {
  reason: string;
  timestamp: string;
  requestId: string;
  messageId?: string;
  onVisible?: (messageId: string) => void;
}

export const DenialNotice = memo(function DenialNotice({
  reason,
  timestamp,
  requestId,
  messageId,
  onVisible,
}: DenialNoticeProps) {
  const noticeRef = useRef<HTMLDivElement | null>(null);

  useLayoutEffect(() => {
    const notice = noticeRef.current;
    if (!notice || !messageId || !onVisible || typeof IntersectionObserver === 'undefined') {
      return;
    }

    let disposed = false;
    const observer = new IntersectionObserver((entries) => {
      if (disposed) return;
      if (entries.some((entry) => entry.target === notice && entry.isIntersecting)) {
        onVisible(messageId);
      }
    });
    observer.observe(notice);

    return () => {
      disposed = true;
      observer.disconnect();
    };
  }, [messageId, onVisible]);

  return (
    <div
      ref={noticeRef}
      className="chat-denial-notice"
      data-denial-request-id={requestId}
      data-denial-message-id={messageId}
      role="note"
      aria-label="Request blocked"
    >
      <ShieldOff className="chat-denial-notice-icon" size={14} aria-hidden="true" />
      <div className="chat-denial-notice-body">
        <span className="chat-denial-notice-label">Request blocked</span>
        <span className="chat-denial-notice-reason">{reason}</span>
      </div>
      <span className="chat-denial-notice-time" title={timestamp}>
        {formatChatTimestamp(timestamp)}
      </span>
    </div>
  );
});
