import { useId, useRef, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

import { useDialogFocusTrap } from '@/hooks/useDialogFocusTrap';

interface NoticeDialogProps {
  title: string;
  message?: ReactNode;
  items?: string[];
  onClose: () => void;
  dialogKey: string;
  confirmLabel?: string;
}

export function NoticeDialog({
  title,
  message,
  items,
  onClose,
  dialogKey,
  confirmLabel = 'OK',
}: NoticeDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const confirmButtonRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const bodyId = useId();

  useDialogFocusTrap(dialogRef, confirmButtonRef, { onEscape: onClose });

  return createPortal(
    <div className="modal-overlay" data-notice-dialog={dialogKey} onClick={onClose}>
      <div
        ref={dialogRef}
        className="modal-content modal-small"
        role="alertdialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id={titleId}>{title}</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>
        <div id={bodyId} className="modal-body">
          {message ? <p>{message}</p> : null}
          {items?.length ? (
            <ul>
              {items.map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          ) : null}
        </div>
        <div className="modal-footer">
          <button
            ref={confirmButtonRef}
            type="button"
            className="btn btn-primary"
            onClick={onClose}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
