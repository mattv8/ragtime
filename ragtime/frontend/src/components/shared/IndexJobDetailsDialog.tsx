import { useRef } from 'react';
import type { IndexingWarnings } from '@/types';
import { useDialogFocusTrap } from '@/hooks/useDialogFocusTrap';
import { InlineCopyButton } from './InlineCopyButton';
import { IndexJobWarnings, getIndexJobWarningCount } from './IndexJobWarnings';

interface IndexJobDetailsDialogProps {
  job: {
    id: string;
    name: string;
    type: string;
    status: string;
    errorMessage: string | null;
    warnings?: IndexingWarnings | null;
  };
  onClose: () => void;
  onCopySuccess: () => void;
  onCopyError: () => void;
}

export function IndexJobDetailsDialog({
  job,
  onClose,
  onCopySuccess,
  onCopyError,
}: IndexJobDetailsDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  useDialogFocusTrap(dialogRef, closeButtonRef, { onEscape: onClose });

  const warningCount = getIndexJobWarningCount(job.warnings);
  const isFatal = job.status === 'failed';
  const title = isFatal ? 'Job Error Details' : 'Job warnings';
  const status =
    job.status === 'completed' && warningCount > 0 ? 'Completed with warnings' : job.status;
  const jobIdentity = `${job.type}-${job.id}`;

  return (
    <div className="modal-overlay" onMouseDown={onClose}>
      <div
        id="index-job-details-dialog"
        ref={dialogRef}
        className="modal index-job-details"
        role="dialog"
        aria-modal="true"
        aria-labelledby="index-job-details-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id="index-job-details-title">{title}</h3>
          <button
            ref={closeButtonRef}
            type="button"
            className="modal-close"
            aria-label="Close job details"
            onClick={onClose}
          >
            &times;
          </button>
        </div>
        <div className="modal-body">
          <p>
            <strong>Job ID:</strong> <code>{job.id}</code>
          </p>
          <p>
            <strong>Name:</strong> {job.name}
          </p>
          <p>
            <strong>Type:</strong> <span className={`badge type-${job.type}`}>{job.type}</span>
          </p>
          <p>
            <strong>Status:</strong>{' '}
            <span
              className={`badge ${job.status === 'completed' && warningCount > 0 ? 'index-job-warning-badge' : job.status}`}
            >
              {status}
            </span>
          </p>
          {isFatal && (
            <section className="error-message-section" data-index-job-error>
              <div className="error-message-header">
                <h4>Error Message</h4>
                {job.errorMessage && (
                  <InlineCopyButton
                    copyText={job.errorMessage}
                    className="copy-error-btn"
                    title="Copy error message"
                    ariaLabel="Copy error message"
                    copiedTitle="Error message copied"
                    copiedAriaLabel="Error message copied"
                    iconSize={16}
                    feedbackMs={2000}
                    onCopySuccess={onCopySuccess}
                    onCopyError={onCopyError}
                  />
                )}
              </div>
              {job.errorMessage ? (
                <pre className="error-message-text">{job.errorMessage}</pre>
              ) : (
                <p>No error details are available for this job.</p>
              )}
            </section>
          )}
          {warningCount > 0 && (
            <IndexJobWarnings warnings={job.warnings} jobIdentity={jobIdentity} />
          )}
        </div>
      </div>
    </div>
  );
}
