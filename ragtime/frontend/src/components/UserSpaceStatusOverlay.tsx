import type { ReactNode } from 'react';

import { X } from 'lucide-react';

export type UserSpaceStatusOverlayTone = 'status' | 'success' | 'warning' | 'error';

export interface UserSpaceStatusOverlayItem {
  id: string;
  tone: UserSpaceStatusOverlayTone;
  content: ReactNode;
  dismissLabel?: string;
}

interface UserSpaceStatusOverlayProps {
  items: UserSpaceStatusOverlayItem[];
  extraClassName?: string;
  onDismiss?: (id: string) => void;
}

export function UserSpaceStatusOverlay({
  items,
  extraClassName = '',
  onDismiss,
}: UserSpaceStatusOverlayProps) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className={`userspace-status-overlay${extraClassName}`} role="status" aria-live="polite">
      {items.map((item) => {
        const toneClass = item.tone === 'status' ? '' : ` userspace-${item.tone}`;
        const dismissible = Boolean(item.dismissLabel);
        return (
          <p key={item.id} className={`userspace-status userspace-status-overlay-item${toneClass}`}>
            <span className="userspace-status-overlay-content">{item.content}</span>
            {dismissible ? (
              <button
                type="button"
                className="userspace-status-overlay-dismiss"
                aria-label={item.dismissLabel}
                onClick={() => onDismiss?.(item.id)}
              >
                <X size={14} aria-hidden="true" />
              </button>
            ) : null}
          </p>
        );
      })}
    </div>
  );
}
