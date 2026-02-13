/**
 * Shared indexing status pill component for index cards.
 * Used by IndexesList (Document indexes) and FilesystemIndexPanel.
 */

interface IndexingPillProps {
  /** Whether this is an optimistic (newly created, 0 documents) index */
  isOptimistic?: boolean;
  /** Active job with status and progress */
  activeJob?: {
    status: string;
    progress_percent: number;
  } | null;
  /** Custom label for pending status (default: "Pending...") */
  pendingLabel?: string;
  /** Custom label prefix for progress (default: "Processing" or "Indexing") */
  progressLabelPrefix?: string;
}

/**
 * Renders an "Indexing..." pill when there's an active job.
 * Only shows during actual indexing activity, not for indexes that merely have 0 documents.
 * Returns null if no active indexing job.
 */
export function IndexingPill({
  activeJob,
  pendingLabel = 'Pending...',
  progressLabelPrefix = 'Indexing',
}: IndexingPillProps) {
  const isActive = activeJob && (activeJob.status === 'pending' || activeJob.status === 'processing' || activeJob.status === 'indexing');

  // Only show indexing pill when there's an active job.
  // An optimistic index (0 docs) without an active job is a failed/incomplete index,
  // not one that's currently indexing - that case is handled by the Incomplete badge.
  if (!isActive) {
    return null;
  }

  // Build title and label
  let title = pendingLabel;
  let label = 'Indexing...';

  if (activeJob && isActive) {
    if (activeJob.status === 'pending') {
      title = pendingLabel;
    } else {
      title = `${progressLabelPrefix}: ${activeJob.progress_percent.toFixed(0)}%`;
    }
    label = activeJob.progress_percent > 0
      ? `Indexing... ${Math.round(activeJob.progress_percent)}%`
      : 'Indexing...';
  }

  return (
    <span className="meta-pill indexing" title={title}>
      {label}
    </span>
  );
}
