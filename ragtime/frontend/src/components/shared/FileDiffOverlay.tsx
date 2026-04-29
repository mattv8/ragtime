import { useEffect, useState } from 'react';
import type { UserSpaceSnapshotFileDiff } from '@/types';
import { MiniLoadingSpinner } from './MiniLoadingSpinner';
import { UserSpaceFileDiffView, formatDiffStatus } from './UserSpaceFileDiffView';

export interface FileDiffOverlayEntry {
  key: string;
  path: string;
  op?: 'patch' | 'upsert' | 'move' | 'delete' | string;
  diff: UserSpaceSnapshotFileDiff | null;
  loading: boolean;
  error: string | null;
}

export interface FileDiffOverlayProps {
  diff?: UserSpaceSnapshotFileDiff | null;
  loading?: boolean;
  error?: string | null;
  // Batched mode: provide a list of entries and the initial active index.
  // When `entries` is provided, the overlay renders a sidebar nav and
  // `diff` / `loading` / `error` props are ignored in favor of per-entry
  // values. `activeIndex` is the initial selection.
  entries?: FileDiffOverlayEntry[];
  activeIndex?: number;
  title: string;
  beforeLabel: string;
  afterLabel: string;
  formatError?: (error: string | null) => string | null;
  onDismiss: () => void;
  onOverlayClick?: () => void;
  onOverlayMouseEnter?: () => void;
  onOverlayMouseLeave?: () => void;
}

export function FileDiffOverlay({
  diff,
  loading,
  error,
  entries,
  activeIndex: initialActiveIndex,
  title,
  beforeLabel,
  afterLabel,
  formatError,
  onDismiss,
  onOverlayClick,
  onOverlayMouseEnter,
  onOverlayMouseLeave,
}: FileDiffOverlayProps) {
  const isBatched = Array.isArray(entries) && entries.length > 0;
  const [selectedIndex, setSelectedIndex] = useState<number>(
    isBatched ? Math.min(Math.max(initialActiveIndex ?? 0, 0), (entries?.length ?? 1) - 1) : 0,
  );

  useEffect(() => {
    if (isBatched && initialActiveIndex != null) {
      setSelectedIndex(Math.min(Math.max(initialActiveIndex, 0), (entries?.length ?? 1) - 1));
    }
  }, [initialActiveIndex, isBatched, entries?.length]);

  const activeEntry = isBatched ? entries![selectedIndex] : null;
  const activeDiff = activeEntry ? activeEntry.diff : (diff ?? null);
  const activeLoading = activeEntry ? activeEntry.loading : Boolean(loading);
  const activeError = activeEntry ? activeEntry.error : (error ?? null);

  if (!isBatched && !activeLoading && !activeError && !activeDiff) return null;
  if (isBatched && entries!.length === 0) return null;

  const displayError = formatError ? formatError(activeError) : activeError;
  const resolvedBeforeLabel = activeDiff?.is_snapshot_own_diff ? 'Previous' : beforeLabel;
  const resolvedAfterLabel = activeDiff?.is_snapshot_own_diff ? 'Snapshot' : afterLabel;

  return (
    <div
      className="userspace-snapshot-diff-backdrop"
      onClick={onDismiss}
    >
      <div
        className={`userspace-snapshot-diff-overlay${isBatched ? ' userspace-snapshot-diff-overlay-batched' : ''}`}
        onClick={(event) => {
          event.stopPropagation();
          onOverlayClick?.();
        }}
        onMouseEnter={onOverlayMouseEnter}
        onMouseLeave={onOverlayMouseLeave}
      >
        <div className="userspace-snapshot-diff-overlay-header">
          <div>
            <div className="userspace-snapshot-diff-overlay-title">
              {title}
              {isBatched ? ` (${selectedIndex + 1} of ${entries!.length})` : ''}
            </div>
            {activeDiff && (
              <div className="userspace-snapshot-diff-overlay-subtitle">
                <span>{activeDiff.path}</span>
                <span>{formatDiffStatus(activeDiff.status)}</span>
                <span>+{activeDiff.additions} -{activeDiff.deletions}</span>
              </div>
            )}
            {!activeDiff && activeEntry && (
              <div className="userspace-snapshot-diff-overlay-subtitle">
                <span>{activeEntry.path}</span>
                {activeEntry.op ? <span>{activeEntry.op}</span> : null}
              </div>
            )}
          </div>
          <button type="button" className="modal-close" onClick={onDismiss}>&times;</button>
        </div>

        <div className="userspace-snapshot-diff-overlay-content">
          {isBatched && (
            <nav className="userspace-snapshot-diff-overlay-nav" aria-label="Files in this batch">
              {entries!.map((entry, index) => {
                const isActive = index === selectedIndex;
                const adds = entry.diff?.additions ?? 0;
                const dels = entry.diff?.deletions ?? 0;
                const status = entry.diff?.status ?? (entry.op === 'delete' ? 'D' : entry.op === 'move' ? 'R' : 'M');
                return (
                  <button
                    key={entry.key}
                    type="button"
                    className={`userspace-snapshot-diff-overlay-nav-item${isActive ? ' is-active' : ''}`}
                    onClick={() => setSelectedIndex(index)}
                    title={entry.path}
                  >
                    <span className={`userspace-snapshot-diff-status userspace-snapshot-diff-status-${String(status).toLowerCase()}`}>{status}</span>
                    <span className="userspace-snapshot-diff-overlay-nav-path">{entry.path}</span>
                    {(adds > 0 || dels > 0) && (
                      <span className="userspace-snapshot-diff-overlay-nav-meta">+{adds} -{dels}</span>
                    )}
                  </button>
                );
              })}
            </nav>
          )}

          {activeLoading ? (
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
          ) : activeDiff ? (
            <div className="userspace-snapshot-diff-overlay-body">
              <UserSpaceFileDiffView
                diff={activeDiff}
                beforeLabel={resolvedBeforeLabel}
                afterLabel={resolvedAfterLabel}
              />
            </div>
          ) : activeEntry ? (
            <div className="userspace-snapshot-diff-overlay-body">
              <p className="userspace-muted">
                {activeEntry.op === 'delete'
                  ? `${activeEntry.path} was deleted.`
                  : activeEntry.op === 'move'
                    ? `${activeEntry.path} was renamed.`
                    : 'No diff available for this entry.'}
              </p>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
