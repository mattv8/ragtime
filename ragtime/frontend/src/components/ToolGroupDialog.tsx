import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';
import { useDialogFocusTrap } from '../hooks/useDialogFocusTrap';

export function validateToolGroupName(
  name: string,
  existingNames: string[],
  currentName?: string,
): string | null {
  const trimmed = name.trim();
  if (!trimmed) return 'Enter a group name.';
  if (trimmed.length > 120) return 'Use 120 characters or fewer.';
  if (trimmed.toLowerCase() === 'ungrouped') return '“Ungrouped” is reserved.';
  if (
    existingNames.some(
      (existing) =>
        existing.trim().toLowerCase() === trimmed.toLowerCase() &&
        existing.trim().toLowerCase() !== currentName?.trim().toLowerCase(),
    )
  )
    return `A group named “${trimmed}” already exists.`;
  return null;
}

export interface ToolGroupDialogProps {
  mode: 'create' | 'edit';
  initialName?: string;
  initialDescription?: string;
  existingNames: string[];
  onSubmit: (v: { name: string; description: string }) => Promise<void>;
  onClose: () => void;
}

export function ToolGroupDialog({
  mode,
  initialName = '',
  initialDescription = '',
  existingNames,
  onSubmit,
  onClose,
}: ToolGroupDialogProps) {
  const [name, setName] = useState(initialName);
  const [description, setDescription] = useState(initialDescription);
  const [attempted, setAttempted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const pendingRef = useRef(false);
  useDialogFocusTrap(dialogRef, nameRef, { onEscape: onClose, disableEscape: busy });
  useEffect(() => {
    if (mode === 'edit') nameRef.current?.select();
  }, [mode]);
  const validation = attempted
    ? validateToolGroupName(name, existingNames, mode === 'edit' ? initialName : undefined)
    : null;
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setAttempted(true);
    if (pendingRef.current) return;
    if (validateToolGroupName(name, existingNames, mode === 'edit' ? initialName : undefined)) {
      nameRef.current?.focus();
      return;
    }
    pendingRef.current = true;
    setBusy(true);
    setError(null);
    try {
      await onSubmit({ name: name.trim(), description: description.trim() });
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Failed to save group');
      pendingRef.current = false;
      setBusy(false);
    }
  };
  return (
    <div className="modal-overlay">
      <div
        id="tools-group-dialog"
        className="modal-content"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tools-group-dialog-title"
        aria-busy={busy}
        ref={dialogRef}
      >
        <form
          id="tools-group-dialog-form"
          className="tools-group-dialog-form"
          onSubmit={submit}
          noValidate
        >
          <div className="modal-header">
            <h3 id="tools-group-dialog-title">{mode === 'create' ? 'New group' : 'Edit group'}</h3>
          </div>
          <div className="modal-body">
            <div className="form-group">
              <label htmlFor="tools-group-name">Name</label>
              <input
                id="tools-group-name"
                ref={nameRef}
                required
                value={name}
                onChange={(event) => setName(event.target.value)}
                aria-invalid={!!validation}
                aria-describedby={validation ? 'tools-group-name-error' : undefined}
              />
            </div>
            {validation && (
              <p id="tools-group-name-error" className="tools-group-dialog-error" role="alert">
                {validation}
              </p>
            )}
            <div className="form-group">
              <label htmlFor="tools-group-description">Description</label>
              <textarea
                id="tools-group-description"
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
              />
            </div>
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
            <button type="submit" className="btn btn-primary" disabled={busy}>
              {busy ? 'Saving…' : mode === 'create' ? 'Create group' : 'Save'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
