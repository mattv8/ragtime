import { useState } from 'react';
import { api } from '@/api';
import type { IndexJob } from '@/types';

interface JobsTableProps {
  jobs: IndexJob[];
  loading: boolean;
  error: string | null;
  onJobsChanged?: () => void;
}

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
 * Calculate overall progress percentage for a job
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

export function JobsTable({ jobs, loading, error, onJobsChanged }: JobsTableProps) {
  const [showAll, setShowAll] = useState(false);
  const [selectedJob, setSelectedJob] = useState<IndexJob | null>(null);
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

  const confirmCancel = async (jobId: string) => {
    setCancelConfirmId(null);
    setActionLoading(jobId);
    try {
      await api.cancelJob(jobId);
      onJobsChanged?.();
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

  const displayedJobs = showAll ? jobs : jobs.slice(0, RECENT_LIMIT);
  const hasMore = jobs.length > RECENT_LIMIT;
  const hasActiveJobs = jobs.some((j) => j.status === 'pending' || j.status === 'processing');

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

      {loading && jobs.length === 0 && (
        <div className="empty-state">Loading...</div>
      )}

      {error && (
        <div className="empty-state" style={{ color: '#f87171' }}>
          Error loading jobs: {error}
        </div>
      )}

      {!loading && !error && jobs.length === 0 && (
        <div className="empty-state">No indexing jobs yet</div>
      )}

      {jobs.length > 0 && (
        <>
          <table className="jobs-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Name</th>
                <th>Status</th>
                <th>Progress</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {displayedJobs.map((job) => {
                const progress = calculateProgress(job);
                const phase = getProcessingPhase(job);

                return (
                  <tr key={job.id}>
                    <td>
                      <code>{job.id}</code>
                    </td>
                    <td>{job.name}</td>
                    <td>
                      {job.status === 'failed' && job.error_message ? (
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
                      {job.status === 'processing' ? (
                        <div className="progress-container">
                          <div className="progress-bar">
                            <div
                              className="progress-fill"
                              style={{ width: `${progress}%` }}
                            />
                          </div>
                          <div className="progress-details">
                            <span className="progress-phase">{phase}</span>
                            <span className="progress-stats">
                              {job.total_chunks > 0 ? (
                                <>
                                  {job.processed_chunks}/{job.total_chunks} chunks
                                </>
                              ) : (
                                <>
                                  {job.processed_files}/{job.total_files} files
                                </>
                              )}
                            </span>
                          </div>
                        </div>
                      ) : job.status === 'completed' ? (
                        <span className="progress-complete">
                          {job.total_files} files, {job.total_chunks} chunks
                        </span>
                      ) : job.status === 'pending' ? (
                        <span className="progress-pending">Waiting...</span>
                      ) : (
                        <span className="progress-failed">--</span>
                      )}
                    </td>
                    <td>{formatDate(job.created_at)}</td>
                    <td className="actions-cell">
                      {actionLoading === job.id ? (
                        <span className="action-loading">...</span>
                      ) : (
                        <>
                          {(job.status === 'pending' || job.status === 'processing') && (
                            cancelConfirmId === job.id ? (
                              <div style={{ display: 'flex', gap: '4px' }}>
                                <button
                                  className="action-btn action-btn-confirm"
                                  onClick={() => confirmCancel(job.id)}
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
                          {(job.status === 'completed' || job.status === 'failed') && (
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
                {showAll ? `Show Recent (${RECENT_LIMIT})` : `Show All (${jobs.length})`}
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
              <p><strong>Status:</strong> <span className={`badge ${selectedJob.status}`}>{selectedJob.status}</span></p>
              <div className="error-details">
                <strong>Error Message:</strong>
                <pre>{selectedJob.error_message}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
