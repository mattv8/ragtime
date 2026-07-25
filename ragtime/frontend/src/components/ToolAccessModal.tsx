import { useEffect, useId } from 'react';
import { X } from 'lucide-react';

import {
  ToolAccessEditor,
  type ToolAccessGroupOption,
  type ToolAccessPolicy,
  type ToolAccessUserOption,
} from './ToolAccessEditor';

interface ToolAccessModalProps {
  open: boolean;
  toolName: string;
  policy: ToolAccessPolicy | null;
  userOptions: ToolAccessUserOption[];
  groupOptions: ToolAccessGroupOption[];
  loading?: boolean;
  saving?: boolean;
  disabled?: boolean;
  globalWriteEnabled?: boolean;
  onChange: (policy: ToolAccessPolicy) => void;
  onSave: (policy: ToolAccessPolicy) => void | Promise<void>;
  onClose: () => void;
}

export function ToolAccessModal({
  open,
  toolName,
  policy,
  userOptions,
  groupOptions,
  loading = false,
  saving = false,
  disabled = false,
  globalWriteEnabled = true,
  onChange,
  onSave,
  onClose,
}: ToolAccessModalProps) {
  const titleId = useId();

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose, open]);

  if (!open) {
    return null;
  }

  const editorDisabled = disabled || loading || saving || policy == null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content modal-large tool-access-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id={titleId}>Tool Access - {toolName}</h3>
          <button type="button" className="modal-close" aria-label="Close" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="modal-body">
          {loading || policy == null ? (
            <p className="field-help" style={{ margin: 0 }}>
              Loading access policy...
            </p>
          ) : (
            <ToolAccessEditor
              policy={policy}
              userOptions={userOptions}
              groupOptions={groupOptions}
              disabled={editorDisabled}
              globalWriteEnabled={globalWriteEnabled}
              autoFocusSearch
              onChange={onChange}
            />
          )}
        </div>
        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={policy == null || loading || saving || disabled}
            onClick={() => {
              if (policy) {
                void onSave(policy);
              }
            }}
          >
            {saving ? 'Saving...' : 'Save Access'}
          </button>
        </div>
      </div>
    </div>
  );
}
