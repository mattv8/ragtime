import { useState } from 'react';
import { api } from '@/api';
import type { IndexJob, FilesystemIndexJob } from '@/types';

interface JobsTableProps {
  jobs: IndexJob[];
  filesystemJobs?: FilesystemIndexJob[];
  loading: boolean;
  error: string | null;
  onJobsChanged?: () => void;
  onFilesystemJobsChanged?: () => void;
  onCancelFilesystemJob?: (toolId: string, jobId: string) => void;
}

// Unified job type for display
type UnifiedJob = {
  id: string;
  name: string;
  type: 'document' | 'filesystem';
  status: string;
  progress: number;
  totalFiles: number;
  processedFiles: number;
  skippedFiles: number;
  totalChunks: number;
  processedChunks: number;
  errorMessage: string | null;
  createdAt: string;
  phase: string;
  // For filesystem jobs
  toolConfigId?: string;
};

const RECENT_LIMIT = 5;

/**
 * Get a human-readable status message for a processing job
 */
function getProcessingPhase(job: IndexJob): string {
  if (job.status !== 'processing') return '';

  // If we have chunks and are embedding
  if (job.total_chunks > 0 && job.processed_chunks < job.total_chunks) {
    return 'Embedding';
  }
  // If we're still loading files
  if (job.processed_files < job.total_files) {
    return 'Loading files';
  }
  // If files are loaded but no chunks yet
  if (job.processed_files === job.total_files && job.total_chunks === 0) {
    return 'Chunking';
  }
  return 'Processing';
}

/**
 * Calculate overall progress percentage for a document job
 */
function calculateProgress(job: IndexJob): number {
  if (job.status === 'completed') return 100;
  if (job.status === 'pending') return 0;
  if (job.status === 'failed') return 0;

  // Processing: weight file loading at 30%, embedding at 70%
  const fileProgress = job.total_files > 0
    ? (job.processed_files / job.total_files) * 30
    : 0;

  const chunkProgress = job.total_chunks > 0
    ? (job.processed_chunks / job.total_chunks) * 70
    : 0;

  return Math.min(fileProgress + chunkProgress, 99); // Cap at 99 until completed
}

/**
 * Convert IndexJob to unified job format
 */
function toUnifiedJob(job: IndexJob): UnifiedJob {
  return {
    id: job.id,
    name: job.name,
    type: 'document',
    status: job.status,
    progress: calculateProgress(job),
    totalFiles: job.total_files,
    processedFiles: job.processed_files,
    skippedFiles: 0,
    totalChunks: job.total_chunks,
    processedChunks: job.processed_chunks,
    errorMessage: job.error_message,
    createdAt: job.created_at,
    phase: getProcessingPhase(job),
  };
}

/**
 * Convert FilesystemIndexJob to unified job format
 */
function toUnifiedFilesystemJob(job: FilesystemIndexJob): UnifiedJob {
  const progress = job.status === 'completed' ? 100
    : job.status === 'pending' ? 0
    : job.status === 'failed' || job.status === 'cancelled' ? 0
    : job.total_files > 0 ? Math.round((job.processed_files / job.total_files) * 100) : 0;

  return {
    id: job.id,
    name: job.index_name,
    type: 'filesystem',
    status: job.status,
    progress,
    totalFiles: job.total_files,
    processedFiles: job.processed_files,
    skippedFiles: job.skipped_files,
    totalChunks: job.total_chunks,
    processedChunks: job.processed_chunks,
    errorMessage: job.error_message,
    createdAt: job.created_at,
    phase: job.status === 'indexing' ? 'Indexing' : '',
    toolConfigId: job.tool_config_id,
  };
}

export function JobsTable({ jobs, filesystemJobs = [], loading, error, onJobsChanged, onFilesystemJobsChanged, onCancelFilesystemJob }: JobsTableProps) {
  const [showAll, setShowAll] = useState(false);
  const [selectedJob, setSelectedJob] = useState<UnifiedJob | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancelConfirmId, setCancelConfirmId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleString();
  };

  const handleCancel = async (jobId: string) => {
    // Show inline confirmation
    if (cancelConfirmId === jobId) {
      setCancelConfirmId(null);
      return;
    }
    setCancelConfirmId(jobId);
  };

  const confirmCancel = async (jobId: string, jobType: 'document' | 'filesystem', toolConfigId?: string) => {
    setCancelConfirmId(null);
    setActionLoading(jobId);
    try {
      if (jobType === 'filesystem' && toolConfigId) {
        await onCancelFilesystemJob?.(toolConfigId, jobId);
        onFilesystemJobsChanged?.();
      } else {
        await api.cancelJob(jobId);
        onJobsChanged?.();
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to cancel job');
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setActionLoading(null);
    }
  };

  const handleDelete = async (jobId: string) => {
    // Show inline confirmation
    if (deleteConfirmId === jobId) {
      setDeleteConfirmId(null);
      return;
    }
    setDeleteConfirmId(jobId);
  };

  const confirmDelete = async (jobId: string) => {
    setDeleteConfirmId(null);
    setActionLoading(jobId);
    try {
      await api.deleteJob(jobId);
      onJobsChanged?.();
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to delete job');
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setActionLoading(null);
    }
  };

  // Combine and sort all jobs by creation date
  const allJobs: UnifiedJob[] = [
    ...jobs.map(toUnifiedJob),
    ...filesystemJobs.map(toUnifiedFilesystemJob),
  ].sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime());

  const displayedJobs = showAll ? allJobs : allJobs.slice(0, RECENT_LIMIT);
  const hasMore = allJobs.length > RECENT_LIMIT;
  const hasActiveJobs = allJobs.some((j) =>
    j.status === 'pending' || j.status === 'processing' || j.status === 'indexing'
  );

  return (
    <div className="card">
      <div className="section-header">
        <h2>
          Indexing Jobs
          {hasActiveJobs && <span className="live-indicator" title="Auto-refreshing">LIVE</span>}
        </h2>
      </div>

      {errorMessage && (
        <div className="error-banner">
          {errorMessage}
          <button onClick={() => setErrorMessage(null)}>×</button>
        </div>
      )}

      {loading && allJobs.length === 0 && (
        <div className="empty-state">Loading...</div>
      )}

      {error && (
        <div className="empty-state" style={{ color: '#f87171' }}>
          Error loading jobs: {error}
        </div>
      )}

      {!loading && !error && allJobs.length === 0 && (
        <div className="empty-state">No indexing jobs yet</div>
      )}

      {allJobs.length > 0 && (
        <>
          <table className="jobs-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Type</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayedJobs.map((job) => {
                const isActive = job.status === 'pending' || job.status === 'processing' || job.status === 'indexing';

                return (
                  <tr key={`${job.type}-${job.id}`}>
                    <td>
                      <code>{job.id.slice(0, 8)}</code>
                    </td>
                    <td>{job.name}</td>
                    <td>
                      <span className={`badge type-${job.type}`}>
                        {job.type === 'document' ? 'Document' : 'Filesystem'}
                      </span>
                    </td>
                    <td>
                      {job.status === 'failed' && job.errorMessage ? (
                        <button
                          className="badge failed clickable"
                          onClick={() => setSelectedJob(job)}
                          title="Click to view error details"
                        >
                          failed <span className="info-icon">i</span>
                        </button>
                      ) : (
                        <span className={`badge ${job.status}`}>{job.status}</span>
                      )}
                    </td>
                    <td className="progress-cell">
                      {(job.status === 'processing' || job.status === 'indexing') ? (
                        <div className="progress-container">
                          <div className="progress-bar">
                            <div
                              className="progress-fill"
                              style={{ width: `${job.progress}%` }}
                            />
                          </div>
                          <div className="progress-details">
                            <span className="progress-phase">{job.phase}</span>
                            <span className="progress-stats">
                              {job.type === 'filesystem' ? (
                                // Filesystem jobs: show file progress (total is known upfront)
                                <>{job.processedFiles}/{job.totalFiles} files
                                  {job.totalChunks > 0 && ` (${job.totalChunks} chunks)`}
                                  {job.skippedFiles > 0 && ` (${job.skippedFiles} unchanged)`}
                                </>
                              ) : job.totalChunks > 0 ? (
                                // Document jobs: show chunk progress
                                <>{job.processedChunks}/{job.totalChunks} chunks</>
                              ) : (
                                <>{job.processedFiles}/{job.totalFiles} files
                                  {job.skippedFiles > 0 && ` (${job.skippedFiles} skipped)`}
                                </>
                              )}
                            </span>
                          </div>
                        </div>
                      ) : job.status === 'completed' ? (
                        <span className="progress-complete">
                          {job.type === 'filesystem' ? (
                            <>
                              {job.processedFiles > 0 ? (
                                <>
                                  {job.processedFiles} indexed, {job.totalChunks} chunks
                                  {job.skippedFiles > 0 && ` (${job.skippedFiles} unchanged)`}
                                </>
                              ) : job.skippedFiles > 0 ? (
                                <>All {job.skippedFiles} files unchanged</>
                              ) : (
                                <>No files to index</>
                              )}
                            </>
                          ) : (
                            <>
                              {job.totalFiles} files, {job.totalChunks} chunks
                              {job.skippedFiles > 0 && ` (${job.skippedFiles} skipped)`}
                            </>
                          )}
                        </span>
                      ) : job.status === 'pending' ? (
                        <span className="progress-pending">Waiting...</span>
                      ) : job.status === 'cancelled' ? (
                        <span className="progress-cancelled">Cancelled</span>
                      ) : (
                        <span className="progress-failed">--</span>
                      )}
                    </td>
                    <td>{formatDate(job.createdAt)}</td>
                    <td className="actions-cell">
                      {actionLoading === job.id ? (
                        <span className="action-loading">...</span>
                      ) : (
                        <>
                          {isActive && (
                            cancelConfirmId === job.id ? (
                              <div style={{ display: 'flex', gap: '4px' }}>
                                <button
                                  className="action-btn action-btn-confirm"
                                  onClick={() => confirmCancel(job.id, job.type, job.toolConfigId)}
                                  title="Confirm cancel"
                                >
                                  Confirm
                                </button>
                                <button
                                  className="action-btn action-btn-secondary"
                                  onClick={() => setCancelConfirmId(null)}
                                  title="Cancel"
                                >
                                  Back
                                </button>
                              </div>
                            ) : (
                              <button
                                className="action-btn action-btn-cancel"
                                onClick={() => handleCancel(job.id)}
                                title="Cancel this job"
                              >
                                Cancel
                              </button>
                            )
                          )}
                          {job.type === 'document' && (job.status === 'completed' || job.status === 'failed') && (
                            deleteConfirmId === job.id ? (
                              <div style={{ display: 'flex', gap: '4px' }}>
                                <button
                                  className="action-btn action-btn-confirm"
                                  onClick={() => confirmDelete(job.id)}
                                  title="Confirm delete"
                                >
                                  Confirm
                                </button>
                                <button
                                  className="action-btn action-btn-secondary"
                                  onClick={() => setDeleteConfirmId(null)}
                                  title="Cancel"
                                >
                                  Back
                                </button>
                              </div>
                            ) : (
                              <button
                                className="action-btn action-btn-delete"
                                onClick={() => handleDelete(job.id)}
                                title="Delete this job record"
                              >
                                Delete
                              </button>
                            )
                          )}
                        </>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          {hasMore && (
            <div style={{ textAlign: 'center', marginTop: '12px' }}>
              <button
                className="link-btn"
                onClick={() => setShowAll(!showAll)}
              >
                {showAll ? `Show Recent (${RECENT_LIMIT})` : `Show All (${allJobs.length})`}
              </button>
            </div>
          )}
        </>
      )}

      {/* Error Details Modal */}
      {selectedJob && (
        <div className="modal-overlay" onClick={() => setSelectedJob(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Job Error Details</h3>
              <button
                className="modal-close"
                onClick={() => setSelectedJob(null)}
              >
                &times;
              </button>
            </div>
            <div className="modal-body">
              <p><strong>Job ID:</strong> <code>{selectedJob.id}</code></p>
              <p><strong>Name:</strong> {selectedJob.name}</p>
              <p><strong>Type:</strong> <span className={`badge type-${selectedJob.type}`}>{selectedJob.type}</span></p>
              <p><strong>Status:</strong> <span className={`badge ${selectedJob.status}`}>{selectedJob.status}</span></p>
              <div className="error-details">
                <strong>Error Message:</strong>
                <pre>{selectedJob.errorMessage}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
