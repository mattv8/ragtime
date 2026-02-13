import { useState } from 'react';
import { Copy, Check, ArrowUp, ArrowDown, Search } from 'lucide-react';
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
  completedAt: string | null;
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
  // Check for scanning phase
  if (job.error_message?.includes('Scanning')) {
    return 'Scanning files';
  }
  // If no files found yet, we're still scanning
  if (job.total_files === 0) {
    return 'Scanning files';
  }
  // If we're still loading files
  if (job.processed_files < job.total_files) {
    return 'Loading files';
  }
  // If files are loaded but no chunks yet, we're chunking
  if (job.total_chunks === 0) {
    return 'Chunking';
  }
  // If we have chunks but haven't started embedding yet
  if (job.processed_chunks === 0) {
    return 'Preparing embeddings';
  }
  // Actively embedding
  if (job.processed_chunks < job.total_chunks) {
    return 'Embedding';
  }
  return 'Finalizing';
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
    completedAt: job.completed_at,
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
    } else if (job.total_files === 0 && job.files_scanned === 0) {
      // Haven't started scanning yet
      phase = 'Starting...';
      progress = 0;
    } else if (job.total_files === 0 && job.files_scanned > 0) {
      // Still scanning for files
      phase = `Scanning: ${job.files_scanned.toLocaleString()} files found`;
      if (job.current_directory) {
        const dir = job.current_directory.length > 30
          ? '...' + job.current_directory.slice(-30)
          : job.current_directory;
        phase += ` (${dir})`;
      }
      progress = 5 + (job.files_scanned % 10);
    } else if (job.total_files > 0) {
      const processedTotal = job.processed_files + job.skipped_files;
      // Check what sub-phase we're in
      if (processedTotal < job.total_files) {
        // Still processing files
        progress = Math.round((processedTotal / job.total_files) * 70); // Files = 0-70%
        phase = 'Loading files';
      } else if (job.total_chunks === 0) {
        // Files done, chunking
        phase = 'Chunking documents';
        progress = 75;
      } else if (job.processed_chunks === 0) {
        // Chunks created, preparing to embed
        phase = 'Preparing embeddings';
        progress = 80;
      } else if (job.processed_chunks < job.total_chunks) {
        // Embedding in progress
        progress = 80 + Math.round((job.processed_chunks / job.total_chunks) * 19);
        phase = `Embedding: ${job.processed_chunks.toLocaleString()}/${job.total_chunks.toLocaleString()}`;
      } else {
        // All done, finalizing
        phase = 'Finalizing index';
        progress = 99;
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
    completedAt: job.completed_at,
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
    } else if (job.total_tables === 0 && !job.status_detail) {
      // Very early: haven't started introspection yet
      phase = 'Connecting to database...';
      progress = 2;
    } else if (job.introspected_tables < job.total_tables || (job.total_tables > 0 && job.processed_tables === 0 && job.introspected_tables > 0 && job.introspected_tables < job.total_tables)) {
      // Introspection phase: tables discovered but not all introspected yet
      // Use status_detail for rich info, fallback to generic
      if (job.status_detail) {
        phase = job.status_detail;
      } else {
        phase = `Introspecting: ${job.introspected_tables}/${job.total_tables}`;
      }
      // Introspection is 0-30% of total progress
      progress = job.total_tables > 0
        ? Math.round((job.introspected_tables / job.total_tables) * 30)
        : 5;
    } else if (job.processed_tables < job.total_tables) {
      // Embedding phase: all tables introspected, now generating embeddings
      progress = 30 + Math.round((job.processed_tables / job.total_tables) * 65);
      phase = `Embedding: ${job.processed_tables}/${job.total_tables}`;
    } else {
      // All done, finalizing
      phase = 'Finalizing index';
      progress = 99;
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
    completedAt: job.completed_at,
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
    completedAt: job.completed_at,
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
  const [searchText, setSearchText] = useState('');
  const [sortConfig, setSortConfig] = useState<{ key: keyof UnifiedJob; direction: 'asc' | 'desc' } | null>(null);
  const [selectedJob, setSelectedJob] = useState<UnifiedJob | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [cancelConfirmId, setCancelConfirmId] = useState<string | null>(null);
  const [retryConfirmId, setRetryConfirmId] = useState<string | null>(null);
  const [copiedErrorId, setCopiedErrorId] = useState<string | null>(null);

  const handleSort = (key: keyof UnifiedJob) => {
    if (sortConfig && sortConfig.key === key) {
      if (sortConfig.direction === 'asc') {
        setSortConfig({ key, direction: 'desc' });
      } else {
        setSortConfig(null);
      }
    } else {
      setSortConfig({ key, direction: 'asc' });
    }
  };

  const copyErrorMessage = async (text: string, jobId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedErrorId(jobId);
      setTimeout(() => setCopiedErrorId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

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

  // Combine all jobs
  const combinedJobs: UnifiedJob[] = [
    ...jobs.map(toUnifiedJob),
    ...filesystemJobs.map(toUnifiedFilesystemJob),
    ...schemaJobs.map(toUnifiedSchemaJob),
    ...pdmJobs.map(toUnifiedPdmJob),
  ];

  // Filter and sort
  const allJobs = combinedJobs
    .filter((job) => {
      if (!searchText) return true;
      const search = searchText.toLowerCase();
      return (
        job.name.toLowerCase().includes(search) ||
        job.id.toLowerCase().includes(search) ||
        job.status.toLowerCase().includes(search) ||
        job.type.toLowerCase().includes(search)
      );
    })
    .sort((a, b) => {
      if (sortConfig) {
        const { key, direction } = sortConfig;

        // Handle null values
        if (a[key] === null && b[key] === null) return 0;
        if (a[key] === null) return 1;
        if (b[key] === null) return -1;

        if (a[key]! < b[key]!) {
          return direction === 'asc' ? -1 : 1;
        }
        if (a[key]! > b[key]!) {
          return direction === 'asc' ? 1 : -1;
        }
        return 0;
      }
      // Default sort by created date desc
      return new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime();
    });

  const displayedJobs = showAll ? allJobs : allJobs.slice(0, RECENT_LIMIT);
  const hasMore = allJobs.length > RECENT_LIMIT;
  const hasActiveJobs = combinedJobs.some((j) =>
    j.status === 'pending' || j.status === 'processing' || j.status === 'indexing'
  );

  return (
    <div className="card">
      <div className="section-header">
        <h2>
          Indexing Jobs
          {hasActiveJobs && <span className="live-indicator" title="Auto-refreshing">LIVE</span>}
        </h2>
        <div className="search-input-wrapper" style={{ position: 'relative' }}>
          <Search size={16} className="search-icon" style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-secondary)' }} />
          <input
            type="text"
            placeholder="Filter jobs..."
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="form-input"
            style={{ paddingLeft: '32px', width: '250px' }}
          />
        </div>
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
                  <th onClick={() => handleSort('type')} style={{ cursor: 'pointer' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      Type
                      {sortConfig?.key === 'type' && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </div>
                  </th>
                  <th onClick={() => handleSort('name')} style={{ cursor: 'pointer' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      Name
                      {sortConfig?.key === 'name' && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </div>
                  </th>
                  <th onClick={() => handleSort('createdAt')} style={{ cursor: 'pointer' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      Created
                      {sortConfig?.key === 'createdAt' && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </div>
                  </th>
                  <th onClick={() => handleSort('completedAt')} style={{ cursor: 'pointer' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      Completed
                      {sortConfig?.key === 'completedAt' && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </div>
                  </th>
                  <th onClick={() => handleSort('progress')} style={{ cursor: 'pointer' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      Progress
                      {sortConfig?.key === 'progress' && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </div>
                  </th>
                  <th onClick={() => handleSort('id')} style={{ cursor: 'pointer' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                      ID
                      {sortConfig?.key === 'id' && (sortConfig.direction === 'asc' ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </div>
                  </th>
                  <th className="sticky-action-header">Actions</th>
                </tr>
              </thead>
              <tbody>
              {displayedJobs.map((job) => {
                const isActive = job.status === 'pending' || job.status === 'processing' || job.status === 'indexing';

                return (
                  <tr key={`${job.type}-${job.id}`}>
                    <td data-label="Type">
                      <span className={`badge type-${job.type}`}>
                        {job.type === 'document' ? 'Document' : job.type === 'filesystem' ? 'Filesystem' : job.type === 'pdm' ? 'PDM' : 'Schema'}
                      </span>
                    </td>
                    <td data-label="Name" title={job.name}>{job.name}</td>
                    <td data-label="Created">{formatDate(job.createdAt)}</td>
                    <td data-label="Completed">{job.completedAt ? formatDate(job.completedAt) : '-'}</td>
                    <td data-label="Progress" className="progress-cell">
                      {job.status === 'failed' ? (
                        job.errorMessage ? (
                          <button
                            className="badge failed clickable"
                            onClick={() => setSelectedJob(job)}
                            title="Click to view error details"
                          >
                            failed <span className="info-icon">i</span>
                          </button>
                        ) : (
                          <span className="badge failed">failed</span>
                        )
                      ) : (job.status === 'processing' || job.status === 'indexing') ? (
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
                                // Schema jobs: show appropriate progress based on phase
                                job.phase.startsWith('Embedding') ? (
                                  <>{job.processedTables}/{job.totalTables} tables</>
                                ) : job.phase.startsWith('Introspecting') || job.phase.startsWith('Discovering') || job.phase.startsWith('Connecting') ? (
                                  job.totalTables > 0
                                    ? <>{job.totalTables} tables found</>
                                    : null
                                ) : job.totalTables > 0 ? (
                                  <>{job.processedTables}/{job.totalTables} tables</>
                                ) : null
                              ) : job.type === 'filesystem' ? (
                                // Filesystem jobs: show appropriate progress based on phase
                                job.phase.startsWith('Embedding') ? (
                                  <>{job.processedChunks.toLocaleString()}/{job.totalChunks.toLocaleString()} chunks</>
                                ) : job.phase.startsWith('Loading') || job.phase.startsWith('Scanning') ? (
                                  <>{(job.processedFiles + job.skippedFiles).toLocaleString()}/{job.totalFiles.toLocaleString()} files
                                    {job.skippedFiles > 0 && ` (${job.skippedFiles.toLocaleString()} unchanged)`}
                                  </>
                                ) : job.totalChunks > 0 ? (
                                  <>{job.totalChunks.toLocaleString()} chunks</>
                                ) : (
                                  <>{job.processedFiles.toLocaleString()}/{job.totalFiles.toLocaleString()} files</>
                                )
                              ) : job.type === 'pdm' ? (
                                // PDM jobs: phase already shows doc progress, just show chunk count
                                <>{job.totalChunks.toLocaleString()} chunks</>
                              ) : (
                                // Document jobs: show appropriate progress based on phase
                                job.phase === 'Embedding' ? (
                                  <>{job.processedChunks.toLocaleString()}/{job.totalChunks.toLocaleString()} chunks</>
                                ) : job.phase === 'Loading files' ? (
                                  <>{job.processedFiles.toLocaleString()}/{job.totalFiles.toLocaleString()} files</>
                                ) : job.totalChunks > 0 ? (
                                  <>{job.totalChunks.toLocaleString()} chunks</>
                                ) : job.totalFiles > 0 ? (
                                  <>{job.processedFiles.toLocaleString()}/{job.totalFiles.toLocaleString()} files</>
                                ) : null
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
                    <td data-label="ID">
                      <code>{job.id.slice(0, 8)}</code>
                    </td>
                    <td data-label="Actions" className="sticky-action-cell">
                      <div className="actions-cell">
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

                        </>
                      )}
                      </div>
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

              <div className="error-message-section">
                <div className="error-message-header">
                  <h4>Error Message</h4>
                  <button
                    className="copy-error-btn"
                    onClick={() => copyErrorMessage(selectedJob.errorMessage ?? '', selectedJob.id)}
                    title="Copy error message"
                  >
                    {copiedErrorId === selectedJob.id ? (
                      <Check size={16} />
                    ) : (
                      <Copy size={16} />
                    )}
                  </button>
                </div>
                <pre className="error-message-text">{selectedJob.errorMessage}</pre>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
