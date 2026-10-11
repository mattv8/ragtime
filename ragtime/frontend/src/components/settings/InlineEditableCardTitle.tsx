import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { Icon } from '@/components/Icon';

/** Heading that switches to an input on click, matching the ToolCard name editor. */
export function InlineEditableCardTitle({
  value,
  ariaLabel,
  maxLength,
  disabled,
  startEditing: startInEditMode = false,
  headingId,
  headingLevel = 'h3',
  headingClassName,
  onCommit,
}: {
  value: string;
  ariaLabel: string;
  maxLength?: number;
  disabled?: boolean;
  /** Open the editor on mount, e.g. for a newly created unnamed item. */
  startEditing?: boolean;
  headingId?: string;
  headingLevel?: 'h3' | 'h4';
  headingClassName?: string;
  onCommit: (value: string) => void;
}): JSX.Element {
  const [editing, setEditing] = useState(startInEditMode && !disabled);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);
  const Heading = headingLevel;

  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  useEffect(() => {
    if (!editing) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [editing]);

  const startEditing = () => {
    if (!disabled) setEditing(true);
  };

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== value) onCommit(trimmed);
    else setDraft(value);
    setEditing(false);
  };

  const cancel = () => {
    setDraft(value);
    setEditing(false);
    if (headingId) requestAnimationFrame(() => document.getElementById(headingId)?.focus());
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      commit();
    } else if (event.key === 'Escape') {
      event.preventDefault();
      cancel();
    }
  };

  if (editing) {
    return (
      <div className="tool-card-title inline-edit-field">
        <input
          ref={inputRef}
          type="text"
          value={draft}
          maxLength={maxLength}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={commit}
          className="inline-edit-input"
          aria-label={ariaLabel}
        />
      </div>
    );
  }

  return (
    <div
      className={`tool-card-title name-wrapper${disabled ? '' : ' editable-field-wrapper'}`}
      onClick={startEditing}
    >
      <Heading
        id={headingId}
        tabIndex={headingId ? -1 : undefined}
        className={headingClassName}
        title={value || undefined}
      >
        {value || <em>Unnamed</em>}
      </Heading>
      {!disabled && (
        <button
          type="button"
          className="inline-edit-btn"
          onClick={(event) => {
            event.stopPropagation();
            startEditing();
          }}
          title="Edit name"
          aria-label={`Edit ${ariaLabel.toLowerCase()}`}
        >
          <Icon name="pencil" size={14} />
        </button>
      )}
    </div>
  );
}
