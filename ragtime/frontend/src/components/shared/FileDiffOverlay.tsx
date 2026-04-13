import type { UserSpaceSnapshotFileDiff } from '@/types';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import { UserSpaceFileDiffView, formatDiffStatus } from './UserSpaceFileDiffView';

export interface FileDiffOverlayProps {
  diff: UserSpaceSnapshotFileDiff | null;
  loading: boolean;
  error: string | null;
  title: string;
  beforeLabel: string;
  afterLabel: string;
  formatError?: (error: string | null) => string | null;
  onDismiss: () => void;
  onOverlayClick: () => void;
  onOverlayMouseEnter: () => void;
  onOverlayMouseLeave: () => void;
}

export function FileDiffOverlay({
  diff,
  loading,
  error,
  title,
  beforeLabel,
  afterLabel,
  formatError,
  onDismiss,
  onOverlayClick,
  onOverlayMouseEnter,
  onOverlayMouseLeave,
}: FileDiffOverlayProps) {
  if (!loading && !error && !diff) return null;

  const displayError = formatError ? formatError(error) : error;
  const resolvedBeforeLabel = diff?.is_snapshot_own_diff ? 'Previous' : beforeLabel;
  const resolvedAfterLabel = diff?.is_snapshot_own_diff ? 'Snapshot' : afterLabel;

  return (
    <div
      className="userspace-snapshot-diff-backdrop"
      onClick={onDismiss}
    >
      <div
        className="userspace-snapshot-diff-overlay"
        onClick={(event) => {
          event.stopPropagation();
          onOverlayClick();
        }}
        onMouseEnter={onOverlayMouseEnter}
        onMouseLeave={onOverlayMouseLeave}
      >
        <div className="userspace-snapshot-diff-overlay-header">
          <div>
            <div className="userspace-snapshot-diff-overlay-title">{title}</div>
            {diff && (
              <div className="userspace-snapshot-diff-overlay-subtitle">
                <span>{diff.path}</span>
                <span>{formatDiffStatus(diff.status)}</span>
                <span>+{diff.additions} -{diff.deletions}</span>
              </div>
            )}
          </div>
          <button type="button" className="modal-close" onClick={onDismiss}>&times;</button>
        </div>

        {loading ? (
          <div className="userspace-snapshot-diff-overlay-body">
            <div className="userspace-snapshot-expanded-status userspace-snapshot-diff-overlay-loading">
              <MiniLoadingSpinner variant="icon" size={14} />
              <span>Loading file diff...</span>
            </div>
          </div>
        ) : displayError ? (
          <div className="userspace-snapshot-diff-overlay-body">
            <p className="userspace-muted userspace-error">{displayError}</p>
          </div>
        ) : diff ? (
          <div className="userspace-snapshot-diff-overlay-body">
            <UserSpaceFileDiffView
              diff={diff}
              beforeLabel={resolvedBeforeLabel}
              afterLabel={resolvedAfterLabel}
            />
          </div>
        ) : null}
      </div>
    </div>
  );
}
