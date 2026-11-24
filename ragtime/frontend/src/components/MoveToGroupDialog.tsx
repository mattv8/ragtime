import { useRef, useState } from 'react';
import { useDialogFocusTrap } from '../hooks/useDialogFocusTrap';
import { validateToolGroupName } from './ToolGroupDialog';

export interface MoveToGroupOption {
  id: string;
  name: string;
  description: string;
  toolCount: number;
}
export type MoveToGroupTarget =
  | { kind: 'group'; groupId: string | null }
  | { kind: 'new'; name: string };
export interface MoveToGroupDialogProps {
  toolName: string;
  currentGroupId: string | null;
  groups: MoveToGroupOption[];
  onMove: (t: MoveToGroupTarget) => Promise<void>;
  onClose: () => void;
}

export function MoveToGroupDialog({
  toolName,
  currentGroupId,
  groups,
  onMove,
  onClose,
}: MoveToGroupDialogProps) {
  const effectiveCurrentGroupId = groups.some((group) => group.id === currentGroupId)
    ? currentGroupId
    : null;
  const [selection, setSelection] = useState<MoveToGroupTarget>({
    kind: 'group',
    groupId: effectiveCurrentGroupId,
  });
  const [attempted, setAttempted] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const initialRef = useRef<HTMLInputElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const pendingRef = useRef(false);
  useDialogFocusTrap(dialogRef, initialRef, { onEscape: onClose, disableEscape: busy });
  const validation =
    selection.kind === 'new' && attempted
      ? validateToolGroupName(
          selection.name,
          groups.map((group) => group.name),
        )
      : null;
  const move = async () => {
    setAttempted(true);
    if (
      pendingRef.current ||
      (selection.kind === 'group' && selection.groupId === effectiveCurrentGroupId)
    )
      return;
    if (
      selection.kind === 'new' &&
      validateToolGroupName(
        selection.name,
        groups.map((group) => group.name),
      )
    ) {
      nameRef.current?.focus();
      return;
    }
    pendingRef.current = true;
    setBusy(true);
    setError(null);
    try {
      await onMove(
        selection.kind === 'new' ? { kind: 'new', name: selection.name.trim() } : selection,
      );
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Failed to move tool');
      pendingRef.current = false;
      setBusy(false);
    }
  };
  return (
    <div className="modal-overlay">
      <div
        id="tools-move-to-group-dialog"
        className="modal-content"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tools-move-to-group-title"
        aria-busy={busy}
        ref={dialogRef}
      >
        <div className="modal-header">
          <h3 id="tools-move-to-group-title">Move “{toolName}” to group</h3>
        </div>
        <div className="modal-body">
          <div
            role="radiogroup"
            aria-labelledby="tools-move-to-group-title"
            className="tools-group-dialog-options"
          >
            {groups.map((group) => (
              <label
                key={group.id}
                className="tools-group-dialog-option"
                data-selected={selection.kind === 'group' && selection.groupId === group.id}
              >
                <input
                  type="radio"
                  name="tools-move-target"
                  value={group.id}
                  checked={selection.kind === 'group' && selection.groupId === group.id}
                  ref={effectiveCurrentGroupId === group.id ? initialRef : undefined}
                  onChange={() => setSelection({ kind: 'group', groupId: group.id })}
                />
                <span className="tools-group-dialog-option-details">
                  <span className="tools-group-dialog-option-summary">
                    <span className="tools-group-dialog-option-name" title={group.name}>
                      {group.name}
                    </span>{' '}
                    <span className="tools-group-dialog-count">
                      {group.toolCount} {group.toolCount === 1 ? 'tool' : 'tools'}
                    </span>{' '}
                    {effectiveCurrentGroupId === group.id && (
                      <span className="tool-badge">Current</span>
                    )}
                  </span>
                  {group.description && (
                    <span className="tools-group-dialog-description">{group.description}</span>
                  )}
                </span>
              </label>
            ))}
            <label
              className="tools-group-dialog-option"
              data-selected={selection.kind === 'group' && selection.groupId === null}
            >
              <input
                type="radio"
                name="tools-move-target"
                value=""
                checked={selection.kind === 'group' && selection.groupId === null}
                ref={effectiveCurrentGroupId === null ? initialRef : undefined}
                onChange={() => setSelection({ kind: 'group', groupId: null })}
              />{' '}
              No group (Ungrouped){' '}
              {effectiveCurrentGroupId === null && <span className="tool-badge">Current</span>}
            </label>
            <label className="tools-group-dialog-option" data-selected={selection.kind === 'new'}>
              <input
                type="radio"
                name="tools-move-target"
                value="new"
                checked={selection.kind === 'new'}
                onChange={() => setSelection({ kind: 'new', name: '' })}
              />{' '}
              New group…
            </label>
          </div>
          {selection.kind === 'new' && (
            <div className="form-group tools-group-dialog-new-name">
              <label htmlFor="tools-move-new-name">Group name</label>
              <input
                id="tools-move-new-name"
                ref={nameRef}
                value={selection.name}
                onChange={(event) => setSelection({ kind: 'new', name: event.target.value })}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    void move();
                  }
                }}
                aria-invalid={!!validation}
                aria-describedby={validation ? 'tools-move-name-error' : undefined}
              />
              {validation && (
                <p id="tools-move-name-error" className="tools-group-dialog-error" role="alert">
                  {validation}
                </p>
              )}
            </div>
          )}
          {error && (
            <p className="tools-group-dialog-error" role="alert">
              {error}
            </p>
          )}
        </div>
        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" disabled={busy} onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={
              busy ||
              (selection.kind === 'group'
                ? selection.groupId === effectiveCurrentGroupId
                : !selection.name.trim())
            }
            onClick={move}
          >
            {busy ? 'Moving…' : 'Move'}
          </button>
        </div>
      </div>
    </div>
  );
}
