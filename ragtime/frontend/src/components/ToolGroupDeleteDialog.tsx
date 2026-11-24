import { useRef, useState } from 'react';
import { useDialogFocusTrap } from '../hooks/useDialogFocusTrap';

export interface ToolGroupDeleteDialogProps {
  groupName: string;
  toolCount: number;
  onConfirm: () => Promise<void>;
  onClose: () => void;
}

export function ToolGroupDeleteDialog({
  groupName,
  toolCount,
  onConfirm,
  onClose,
}: ToolGroupDeleteDialogProps) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  useDialogFocusTrap(dialogRef, cancelRef, { onEscape: onClose, disableEscape: busy });
  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      await onConfirm();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Failed to delete group');
      setBusy(false);
    }
  };
  return (
    <div className="modal-overlay">
      <div
        id="tools-delete-group-dialog"
        className="modal-content"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tools-delete-group-title"
        aria-busy={busy}
        ref={dialogRef}
      >
        <div className="modal-header">
          <h3 id="tools-delete-group-title">Delete group “{groupName}”?</h3>
        </div>
        <div className="modal-body">
          <p>
            {toolCount > 0
              ? `The ${toolCount} ${toolCount === 1 ? 'tool' : 'tools'} in this group move to Ungrouped. No tools are deleted.`
              : 'This group has no tools.'}
          </p>
          <p>Chats and workspaces that selected this group lose that selection.</p>
          {error && (
            <p className="tools-group-dialog-error" role="alert">
              {error}
            </p>
          )}
        </div>
        <div className="modal-footer">
          <button
            ref={cancelRef}
            type="button"
            className="btn btn-secondary"
            disabled={busy}
            onClick={onClose}
          >
            Cancel
          </button>
          <button type="button" className="btn btn-danger" disabled={busy} onClick={confirm}>
            {busy ? 'Deleting…' : 'Delete group'}
          </button>
        </div>
      </div>
    </div>
  );
}
