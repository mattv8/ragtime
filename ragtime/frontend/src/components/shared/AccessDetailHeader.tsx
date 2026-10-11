import type { ReactNode } from 'react';
import { ChevronLeft } from 'lucide-react';
import { InlineEditableCardTitle } from '../settings/InlineEditableCardTitle';

interface EditableTitle {
  ariaLabel: string;
  maxLength?: number;
  disabled?: boolean;
  startEditing?: boolean;
  onCommit: (value: string) => void;
}

interface AccessDetailHeaderProps {
  headingId: string;
  title: string;
  meta?: ReactNode;
  /** When set, the title is renamed in place with the shared hover-click editor. */
  editableTitle?: EditableTitle;
  closeLabel?: string;
  onClose: () => void;
}

export function AccessDetailHeader({
  headingId,
  title,
  meta,
  editableTitle,
  closeLabel = 'Close access editor',
  onClose,
}: AccessDetailHeaderProps) {
  return (
    <header className="access-md-detail-header" data-access-detail-header>
      <button
        type="button"
        className="btn btn-icon btn-sm btn-secondary access-md-collapse"
        aria-label={closeLabel}
        data-access-detail-close
        onClick={onClose}
      >
        <ChevronLeft size={16} aria-hidden="true" />
      </button>
      <div className="access-md-detail-header-body">
        {editableTitle ? (
          <InlineEditableCardTitle
            value={title}
            ariaLabel={editableTitle.ariaLabel}
            maxLength={editableTitle.maxLength}
            disabled={editableTitle.disabled}
            startEditing={editableTitle.startEditing}
            headingId={headingId}
            headingLevel="h4"
            headingClassName="access-md-detail-title"
            onCommit={editableTitle.onCommit}
          />
        ) : (
          <h4 id={headingId} tabIndex={-1} className="access-md-detail-title" title={title}>
            {title}
          </h4>
        )}
        {meta && (
          <div className="access-md-detail-meta" data-access-detail-meta>
            {meta}
          </div>
        )}
      </div>
    </header>
  );
}
