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
 * Renders an "Indexing..." pill when there's an active job or the index is optimistic.
 * Returns null if no indexing activity detected.
 */
export function IndexingPill({
  isOptimistic = false,
  activeJob,
  pendingLabel = 'Pending...',
  progressLabelPrefix = 'Indexing',
}: IndexingPillProps) {
  const isActive = activeJob && (activeJob.status === 'pending' || activeJob.status === 'processing' || activeJob.status === 'indexing');

  // Show indexing pill if there's an active job OR if it's optimistic (newly created)
  // For optimistic indexes without a job yet, show a "Indexing..." placeholder
  if (!isActive && !isOptimistic) {
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
