import { useState } from 'react';
import { api } from '@/api';
import type { IndexJob, FilesystemIndexJob, SchemaIndexJob, PdmIndexJob } from '@/types';

interface JobsTableProps {
  jobs: IndexJob[];
  filesystemJobs?: FilesystemIndexJob[];
  schemaJobs?: SchemaIndexJob[];
  pdmJobs?: PdmIndexJob[];
  loading: boolean;
  error: string | null;
  onJobsChanged?: () => void;
  onFilesystemJobsChanged?: () => void;
  onSchemaJobsChanged?: () => void;
  onPdmJobsChanged?: () => void;
  onCancelFilesystemJob?: (toolId: string, jobId: string) => void;
  onCancelSchemaJob?: (toolId: string, jobId: string) => void;
  onCancelPdmJob?: (toolId: string, jobId: string) => void;
}

// Unified job type for display
type UnifiedJob = {
  id: string;
  name: string;
  type: 'document' | 'filesystem' | 'schema' | 'pdm';
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
  cancelRequested?: boolean;
  // Collection phase info (filesystem jobs)
  filesScanned?: number;
  currentDirectory?: string | null;
  // For filesystem/schema/pdm jobs
  toolConfigId?: string;
  // Schema-specific fields
  totalTables?: number;
  processedTables?: number;
  // PDM-specific fields
  totalDocuments?: number;
  processedDocuments?: number;
  skippedDocuments?: number;
};

const RECENT_LIMIT = 5;

/**
 * Get a human-readable status message for a processing job
 */
function getProcessingPhase(job: IndexJob): string {
  if (job.status !== 'processing') return '';

  // Check for cloning phase (error_message used as status hint during clone)
  if (job.error_message?.includes('Cloning')) {
    return 'Cloning repository';
  }
  // If no files found yet, we're still scanning
  if (job.total_files === 0) {
    return 'Scanning files';
  }
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
  // Determine the phase and progress based on job state
  let phase = '';
  let progress = 0;

  if (job.status === 'completed') {
    phase = 'Complete';
    progress = 100;
  } else if (job.status === 'pending') {
    phase = 'Queued';
    progress = 0;
  } else if (job.status === 'failed' || job.status === 'cancelled') {
    phase = job.status === 'failed' ? 'Failed' : 'Cancelled';
    progress = 0;
  } else if (job.status === 'indexing') {
    if (job.cancel_requested) {
      phase = 'Cancelling...';
      progress = Math.max(progress, 5);
    }
    // Check if still in collection phase (total_files is 0 but files_scanned > 0)
    if (job.total_files === 0 && job.files_scanned > 0) {
      phase = `Scanning: ${job.files_scanned} files found`;
      if (job.current_directory) {
        // Truncate long paths
        const dir = job.current_directory.length > 30
          ? '...' + job.current_directory.slice(-30)
          : job.current_directory;
        phase += ` (${dir})`;
      }
      // Show indeterminate-ish progress during scan (pulse between 5-15%)
      progress = 5 + (job.files_scanned % 10);
    } else if (job.total_files > 0) {
      // File processing phase
      const processedTotal = job.processed_files + job.skipped_files;
      progress = Math.round((processedTotal / job.total_files) * 100);
      phase = job.cancel_requested
        ? 'Cancelling...'
        : `Indexing: ${processedTotal}/${job.total_files}`;
      if (job.skipped_files > 0) {
        phase += ` (${job.skipped_files} unchanged)`;
      }
    } else {
      phase = 'Starting...';
      progress = 0;
    }
  }

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
      phase,
    filesScanned: job.files_scanned,
    currentDirectory: job.current_directory,
    toolConfigId: job.tool_config_id,
    cancelRequested: job.cancel_requested,
  };
}

/**
 * Convert SchemaIndexJob to unified job format
 */
function toUnifiedSchemaJob(job: SchemaIndexJob): UnifiedJob {
  // Determine the phase and progress based on job state
  let phase = '';
  let progress = 0;

  if (job.status === 'completed') {
    phase = 'Complete';
    progress = 100;
  } else if (job.status === 'pending') {
    phase = 'Queued';
    progress = 0;
  } else if (job.status === 'failed' || job.status === 'cancelled') {
    phase = job.status === 'failed' ? 'Failed' : 'Cancelled';
    progress = 0;
  } else if (job.status === 'indexing') {
    if (job.cancel_requested) {
      phase = 'Cancelling...';
    } else if (job.total_tables === 0) {
      phase = 'Introspecting schema...';
      progress = 5;
    } else {
      progress = Math.round((job.processed_tables / job.total_tables) * 100);
      phase = `Indexing: ${job.processed_tables}/${job.total_tables} tables`;
    }
  }

  return {
    id: job.id,
    name: job.index_name,
    type: 'schema',
    status: job.status,
    progress,
    totalFiles: 0,
    processedFiles: 0,
    skippedFiles: 0,
    totalChunks: job.total_chunks,
    processedChunks: job.processed_chunks,
    errorMessage: job.error_message,
    createdAt: job.created_at,
    phase,
    toolConfigId: job.tool_config_id,
    cancelRequested: job.cancel_requested,
    totalTables: job.total_tables,
    processedTables: job.processed_tables,
  };
}

/**
 * Convert PdmIndexJob to unified job format
 */
function toUnifiedPdmJob(job: PdmIndexJob): UnifiedJob {
  // Determine the phase and progress based on job state
  let phase = '';
  let progress = 0;

  if (job.status === 'completed') {
    phase = 'Complete';
    progress = 100;
  } else if (job.status === 'pending') {
    phase = 'Queued';
    progress = 0;
  } else if (job.status === 'failed' || job.status === 'cancelled') {
    phase = job.status === 'failed' ? 'Failed' : 'Cancelled';
    progress = 0;
  } else if (job.status === 'indexing') {
    if (job.cancel_requested) {
      phase = 'Cancelling...';
    } else if (job.total_documents === 0) {
      phase = 'Counting documents...';
      progress = 5;
    } else {
      const processed = job.processed_documents + job.skipped_documents;
      progress = Math.round((processed / job.total_documents) * 100);
      phase = `Indexing: ${processed}/${job.total_documents} documents`;
      if (job.skipped_documents > 0) {
        phase += ` (${job.skipped_documents} unchanged)`;
      }
    }
  }

  return {
    id: job.id,
    name: job.index_name,
    type: 'pdm',
    status: job.status,
    progress,
    totalFiles: job.total_documents,
    processedFiles: job.processed_documents,
    skippedFiles: job.skipped_documents,
    totalChunks: job.total_chunks,
    processedChunks: job.processed_chunks,
    errorMessage: job.error_message,
    createdAt: job.created_at,
    phase,
    toolConfigId: job.tool_config_id,
    cancelRequested: job.cancel_requested,
    totalDocuments: job.total_documents,
    processedDocuments: job.processed_documents,
    skippedDocuments: job.skipped_documents,
  };
}

export function JobsTable({ jobs, filesystemJobs = [], schemaJobs = [], pdmJobs = [], loading, error, onJobsChanged, onFilesystemJobsChanged, onSchemaJobsChanged, onPdmJobsChanged, onCancelFilesystemJob, onCancelSchemaJob, onCancelPdmJob }: JobsTableProps) {
  const [showAll, setShowAll] = useState(false);
  const [selectedJob, setSelectedJob] = useState<UnifiedJob | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancelConfirmId, setCancelConfirmId] = useState<string | null>(null);
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [retryConfirmId, setRetryConfirmId] = useState<string | null>(null);

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

  const confirmCancel = async (jobId: string, jobType: 'document' | 'filesystem' | 'schema' | 'pdm', toolConfigId?: string) => {
    setCancelConfirmId(null);
    setActionLoading(jobId);
    try {
      if (jobType === 'filesystem' && toolConfigId) {
        await onCancelFilesystemJob?.(toolConfigId, jobId);
        onFilesystemJobsChanged?.();
      } else if (jobType === 'schema' && toolConfigId) {
        await onCancelSchemaJob?.(toolConfigId, jobId);
        onSchemaJobsChanged?.();
      } else if (jobType === 'pdm' && toolConfigId) {
        await onCancelPdmJob?.(toolConfigId, jobId);
        onPdmJobsChanged?.();
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

  const handleRetry = async (jobId: string) => {
    // Show inline confirmation
    if (retryConfirmId === jobId) {
      setRetryConfirmId(null);
      return;
    }
    setRetryConfirmId(jobId);
  };

  const confirmRetry = async (jobId: string, jobType: 'document' | 'filesystem' | 'schema' | 'pdm', toolConfigId?: string) => {
    setRetryConfirmId(null);
    setActionLoading(jobId);
    try {
      if (jobType === 'filesystem' && toolConfigId) {
        await api.retryFilesystemJob(toolConfigId, jobId);
        onFilesystemJobsChanged?.();
      } else if (jobType === 'schema' && toolConfigId) {
        await api.retrySchemaJob(toolConfigId, jobId);
        onSchemaJobsChanged?.();
      } else if (jobType === 'pdm' && toolConfigId) {
        await api.retryPdmJob(toolConfigId, jobId);
        onPdmJobsChanged?.();
      } else {
        await api.retryJob(jobId);
        onJobsChanged?.();
      }
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Failed to retry job');
      setTimeout(() => setErrorMessage(null), 5000);
    } finally {
      setActionLoading(null);
    }
  };

  // Combine and sort all jobs by creation date
  const allJobs: UnifiedJob[] = [
    ...jobs.map(toUnifiedJob),
    ...filesystemJobs.map(toUnifiedFilesystemJob),
    ...schemaJobs.map(toUnifiedSchemaJob),
    ...pdmJobs.map(toUnifiedPdmJob),
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
          <div className="jobs-table-wrapper">
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
                    <td data-label="ID">
                      <code>{job.id.slice(0, 8)}</code>
                    </td>
                    <td data-label="Name">{job.name}</td>
                    <td data-label="Type">
                      <span className={`badge type-${job.type}`}>
                        {job.type === 'document' ? 'Document' : job.type === 'filesystem' ? 'Filesystem' : 'Schema'}
                      </span>
                    </td>
                    <td data-label="Status">
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
                    <td data-label="Progress" className="progress-cell">
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
                              {job.type === 'schema' ? (
                                // Schema jobs: show table progress
                                <>{job.processedTables}/{job.totalTables} tables
                                  {job.totalChunks > 0 && ` (${job.totalChunks} chunks)`}
                                </>
                              ) : job.type === 'filesystem' ? (
                                // Filesystem jobs: show file progress (total is known upfront)
                                <>{job.processedFiles}/{job.totalFiles} files
                                  {job.totalChunks > 0 && ` (${job.totalChunks} chunks)`}
                                  {job.skippedFiles > 0 && ` (${job.skippedFiles} unchanged)`}
                                </>
                              ) : job.type === 'pdm' ? (
                                // PDM jobs: phase already shows doc progress, just show chunk count
                                <>{job.totalChunks.toLocaleString()} chunks</>
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
                          {job.type === 'schema' ? (
                            <>
                              {job.totalTables} tables, {job.totalChunks} chunks
                            </>
                          ) : job.type === 'filesystem' ? (
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
                    <td data-label="Created">{formatDate(job.createdAt)}</td>
                    <td data-label="Actions" className="actions-cell">
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
                          {(job.status === 'failed' || job.status === 'cancelled') && (
                            retryConfirmId === job.id ? (
                              <div style={{ display: 'flex', gap: '4px' }}>
                                <button
                                  className="action-btn action-btn-confirm"
                                  onClick={() => confirmRetry(job.id, job.type, job.toolConfigId)}
                                  title="Confirm retry"
                                >
                                  Confirm
                                </button>
                                <button
                                  className="action-btn action-btn-secondary"
                                  onClick={() => setRetryConfirmId(null)}
                                  title="Cancel"
                                >
                                  Back
                                </button>
                              </div>
                            ) : (
                              <button
                                className="action-btn action-btn-retry"
                                onClick={() => handleRetry(job.id)}
                                title="Retry this failed job"
                              >
                                Retry
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
          </div>

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
