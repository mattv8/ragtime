import { Check, X } from 'lucide-react';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';

interface DeleteConfirmationActionsProps {
  disabled: boolean;
  deleting: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function DeleteConfirmationActions({
  disabled,
  deleting,
  onConfirm,
  onCancel,
}: DeleteConfirmationActionsProps) {
  return (
    <>
      <button
        type="button"
        className="chat-action-btn confirm-delete"
        disabled={disabled}
        onClick={(event) => {
          event.stopPropagation();
          onConfirm();
        }}
        title="Confirm delete"
      >
        {deleting ? <MiniLoadingSpinner variant="icon" size={12} /> : <Check size={12} />}
      </button>
      <button
        type="button"
        className="chat-action-btn cancel-delete"
        disabled={disabled}
        onClick={(event) => {
          event.stopPropagation();
          onCancel();
        }}
        title="Cancel"
      >
        <X size={12} />
      </button>
    </>
  );
}
