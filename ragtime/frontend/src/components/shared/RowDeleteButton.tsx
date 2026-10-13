import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { Trash2 } from 'lucide-react';
import { DeleteConfirmationActions } from './DeleteConfirmationActions';

interface RowDeleteButtonProps {
  label: string;
  blockedReason?: string;
  disabled: boolean;
  /** Ask for an inline check/cancel confirmation before deleting (for immediately persisted deletes). */
  confirm?: boolean;
  /** Resolves true when the row was removed, so focus can move to a neighboring row. */
  onDelete: () => boolean | Promise<boolean>;
}

function neighborRowFocusTarget(origin: HTMLElement): HTMLElement | null {
  const item = origin.closest('li');
  const neighbor = item?.nextElementSibling ?? item?.previousElementSibling;
  return (
    neighbor?.querySelector<HTMLElement>('.access-md-rail-row') ??
    origin
      .closest('.access-md-list')
      ?.querySelector<HTMLElement>('[data-new-category], [data-new-access-level]') ??
    null
  );
}

/** Blocked deletes stay clickable so the click surfaces the owner's explanation. */
export function RowDeleteButton({
  label,
  blockedReason,
  disabled,
  confirm = false,
  onDelete,
}: RowDeleteButtonProps) {
  const [confirming, setConfirming] = useState(false);
  const [pending, setPending] = useState(false);
  const trashRef = useRef<HTMLButtonElement>(null);
  const confirmRef = useRef<HTMLSpanElement>(null);
  const reasonId = useId();

  useEffect(() => {
    if (!confirming) return;
    const buttons = confirmRef.current?.querySelectorAll<HTMLButtonElement>('button');
    buttons?.[buttons.length - 1]?.focus();
  }, [confirming]);

  const focusTrash = () => requestAnimationFrame(() => trashRef.current?.focus());
  const cancelConfirm = () => {
    setConfirming(false);
    focusTrash();
  };
  const runDelete = async (origin: HTMLElement) => {
    const focusTarget = neighborRowFocusTarget(origin);
    setPending(true);
    const removed = await onDelete();
    setPending(false);
    setConfirming(false);
    if (removed) requestAnimationFrame(() => focusTarget?.focus());
    else focusTrash();
  };
  const handleTrashClick = () => {
    if (confirm && !blockedReason) setConfirming(true);
    else if (trashRef.current) void runDelete(trashRef.current);
  };
  const handleConfirmKeyDown = (event: KeyboardEvent<HTMLSpanElement>) => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    if (!pending) cancelConfirm();
  };

  if (confirming) {
    return (
      <span
        ref={confirmRef}
        className="access-md-rail-row-confirm"
        role="group"
        aria-label={`Confirm ${label}`}
        data-row-delete-confirm
        onKeyDown={handleConfirmKeyDown}
      >
        <DeleteConfirmationActions
          disabled={disabled || pending}
          deleting={pending}
          onConfirm={() => {
            if (confirmRef.current) void runDelete(confirmRef.current);
          }}
          onCancel={cancelConfirm}
        />
      </span>
    );
  }

  return (
    <>
      <button
        ref={trashRef}
        type="button"
        className="access-md-rail-row-delete"
        data-row-delete
        aria-label={label}
        aria-disabled={blockedReason ? 'true' : undefined}
        aria-describedby={blockedReason ? reasonId : undefined}
        title={blockedReason ?? label}
        disabled={disabled || pending}
        onClick={handleTrashClick}
      >
        <Trash2 size={14} aria-hidden="true" />
      </button>
      {blockedReason && (
        <span id={reasonId} hidden>
          {blockedReason}
        </span>
      )}
    </>
  );
}
