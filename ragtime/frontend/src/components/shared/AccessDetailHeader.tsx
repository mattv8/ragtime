import type { ReactNode } from 'react';
import { ChevronLeft } from 'lucide-react';

interface AccessDetailHeaderProps {
  headingId: string;
  title: string;
  meta?: ReactNode;
  onClose: () => void;
}

export function AccessDetailHeader({ headingId, title, meta, onClose }: AccessDetailHeaderProps) {
  return (
    <header className="access-md-detail-header" data-access-detail-header>
      <button
        type="button"
        className="btn btn-icon btn-sm btn-secondary access-md-collapse"
        aria-label="Close access editor"
        data-access-detail-close
        onClick={onClose}
      >
        <ChevronLeft size={16} aria-hidden="true" />
      </button>
      <div className="access-md-detail-header-body">
        <h4 id={headingId} tabIndex={-1} className="access-md-detail-title" title={title}>
          {title}
        </h4>
        {meta && (
          <div className="access-md-detail-meta" data-access-detail-meta>
            {meta}
          </div>
        )}
      </div>
    </header>
  );
}
