import { useState, useEffect, useCallback, useRef } from 'react';
import { MemoryStick, HardDrive } from 'lucide-react';
import { api } from '@/api';
import { formatSizeMB } from '@/utils';
import type { ToolConfig, FilesystemIndexJob, FilesystemIndexStats } from '@/types';
import { ToolWizard } from './ToolWizard';
import { IndexCard } from './IndexCard';
import { DeleteConfirmButton } from './DeleteConfirmButton';
import { AnimatedCreateButton } from './AnimatedCreateButton';
import { IndexingPill } from './IndexingPill';

// Polling interval for active jobs (2 seconds)
const ACTIVE_JOB_POLL_INTERVAL = 2000;

interface FilesystemIndexPanelProps {
  onToolsChanged?: () => void;
  onJobsChanged?: () => void;
  /** Embedding dimensions for memory calculation (from app settings) */
  embeddingDimensions?: number | null;
}

interface FilesystemIndexCardProps {
  tool: ToolConfig;
  activeJob: FilesystemIndexJob | null;
  stats: FilesystemIndexStats | null;
  onStartIndex: (toolId: string, fullReindex: boolean) => void;
  onDeleteIndex: (toolId: string) => void;
  onEdit: (tool: ToolConfig) => void;
  onDelete: (toolId: string) => void;
  onToggle: (toolId: string, enabled: boolean) => void;
  onRename: (toolName: string, newName: string) => Promise<void>;
  onDescriptionUpdate: (toolId: string, newDesc: string) => Promise<void>;
  indexing: boolean;
  embeddingDimensions?: number | null;
}

function FilesystemIndexCard({
  tool,
  activeJob,
  stats,
  onStartIndex,
  onDeleteIndex: _onDeleteIndex,
  onEdit,
  onDelete,
  onToggle,
  onRename,
  onDescriptionUpdate,
  indexing,
  embeddingDimensions,
}: FilesystemIndexCardProps) {
  const config = tool.connection_config as { base_path?: string; index_name?: string; vector_store_type?: string };

  const isActive = activeJob ? (activeJob.status === 'pending' || activeJob.status === 'indexing') : false;
  // Check both stats (active index) and config (intended index) for FAISS type
  const isFaiss = stats?.vector_store_type === 'faiss' || config.vector_store_type === 'faiss';
  const [downloading, setDownloading] = useState(false);

  const handleDownload = async () => {
    const indexName = config.index_name || tool.id;
    setDownloading(true);
    try {
      await api.downloadFilesystemFaissIndex(indexName);
    } catch (e) {
      console.error('Failed to download FAISS index:', e);
    } finally {
      setDownloading(false);
    }
  };

  const metaPills = (
    <>
      <span className="meta-pill path" title={config.base_path}>
        {config.base_path || 'Path not configured'}
      </span>
      {stats && stats.embedding_count > 0 && (
        <>
          <span className="meta-pill files">{stats.file_count} files</span>
          {(() => {
            const isPgvector = stats.vector_store_type === 'pgvector';

            // For pgvector, show actual disk storage size; for FAISS, show RAM
            if (isPgvector) {
              // pgvector: Show disk storage size
              const storageMb = stats.pgvector_size_mb;
              if (storageMb === undefined || storageMb === null) {
                return null;
              }
              return (
                <span
                  className="meta-pill size"
                  title={`pgvector storage: ${formatSizeMB(storageMb)} on disk (${stats.embedding_count.toLocaleString()} embeddings)`}
                >
                  <HardDrive size={12} />
                  {formatSizeMB(storageMb)}
                </span>
              );
            } else {
              // FAISS: Show RAM usage
              let ramMb = 0;

              // Use server-side estimate if available, otherwise calculate fallback
              if (stats.estimated_memory_mb !== undefined && stats.estimated_memory_mb !== null) {
                ramMb = stats.estimated_memory_mb;
              } else if (embeddingDimensions && embeddingDimensions > 0) {
                // Memory formula: embeddings * dimensions * 4 bytes (float32)
                const bytesPerEmbedding = embeddingDimensions * 4 * 1.15;
                ramMb = (stats.embedding_count * bytesPerEmbedding) / (1024 * 1024);
              } else {
                return null;
              }

              return (
                <span
                  className={`meta-pill ram ${tool.enabled ? 'ram-loaded' : 'ram-unloaded'}`}
                  title={`${tool.enabled ? 'Loaded in RAM' : 'Not loaded (disabled)'}: ${formatSizeMB(ramMb)} (${stats.embedding_count.toLocaleString()} embeddings)`}
                >
                  <MemoryStick size={12} />
                  {formatSizeMB(ramMb)}
                </span>
              );
            }
          })()}
        </>
      )}
      {stats?.last_indexed && (
        <span className="meta-pill date" title={`Last indexed: ${new Date(stats.last_indexed).toLocaleString()}`}>
          {`Updated ${new Date(stats.last_indexed).toLocaleString()}`}
        </span>
      )}
      <IndexingPill
        isOptimistic={!stats || stats.embedding_count === 0}
        activeJob={activeJob}
        progressLabelPrefix="Indexing"
      />
      {!tool.enabled && <span className="meta-pill disabled">Excluded from RAG</span>}
    </>
  );

  const actions = (
    <>
      {/* Management controls */}
      <button
        type="button"
        className="btn btn-sm btn-secondary"
        onClick={() => onEdit(tool)}
        title="Edit configuration"
      >
        Edit
      </button>

      {/* Download FAISS index */}
      {isFaiss && stats && stats.embedding_count > 0 && (
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          onClick={handleDownload}
          disabled={downloading}
          title="Download FAISS index as zip"
        >
          {downloading ? 'Downloading...' : 'Download'}
        </button>
      )}

      {/* Indexing controls */}
      {!isFaiss && (
        <button
          type="button"
          className="btn btn-sm btn-primary"
          onClick={() => onStartIndex(tool.id, false)}
          disabled={indexing || isActive}
          title="Start incremental indexing (skip unchanged files)"
        >
          {indexing ? 'Starting...' : 'Re-Index'}
        </button>
      )}
      <button
        type="button"
        className="btn btn-sm"
        onClick={() => onStartIndex(tool.id, true)}
        disabled={indexing || isActive}
        title={isFaiss ? "Re-index all files from scratch" : "Force full re-index of all files"}
      >
        Full Re-Index
      </button>

      <DeleteConfirmButton
        onDelete={() => onDelete(tool.id)}
        title="Delete this filesystem index configuration"
      />
    </>
  );

  return (
    <IndexCard
      title={tool.name}
      description={tool.description}
      enabled={tool.enabled ?? false}
      onToggle={(checked) => onToggle(tool.id, checked)}
      onEditTitle={(newName) => onRename(tool.id, newName)}
      onEditDescription={(newDesc) => onDescriptionUpdate(tool.id, newDesc)}
      className="filesystem-index-item"
      metaPills={metaPills}
      actions={actions}
      toggleTitle={
        tool.disabled_reason
          ? `Disabled: ${tool.disabled_reason}`
          : tool.enabled
            ? "Enabled for RAG"
            : "Disabled from RAG"
      }
    />
  );
}

export function FilesystemIndexPanel({ onToolsChanged, onJobsChanged, embeddingDimensions }: FilesystemIndexPanelProps) {
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Wizard state
  const [showWizard, setShowWizard] = useState(false);
  const [editingTool, setEditingTool] = useState<ToolConfig | null>(null);

  // Job state
  const [filesystemJobs, setFilesystemJobs] = useState<Record<string, FilesystemIndexJob | null>>({});
  const [filesystemStats, setFilesystemStats] = useState<Record<string, FilesystemIndexStats | null>>({});
  const [indexingToolId, setIndexingToolId] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Confirmation state
  const [confirmation, setConfirmation] = useState<{
    message: string;
    onConfirm: () => void;
  } | null>(null);

  // Calculate total storage/memory for filesystem indexes
  // For pgvector: show disk storage; for FAISS: show RAM estimate
  const calculateTotalStorage = (): { totalStorage: number; enabledStorage: number; hasPgvector: boolean; hasFaiss: boolean } | null => {
    let totalMb = 0;
    let enabledMb = 0;
    let hasData = false;
    let hasPgvector = false;
    let hasFaiss = false;

    for (const tool of tools) {
      const stats = filesystemStats[tool.id];
      if (stats && stats.embedding_count > 0) {
        const isPgvector = stats.vector_store_type === 'pgvector';

        if (isPgvector) {
          // pgvector: Use actual disk storage
          hasPgvector = true;
          if (stats.pgvector_size_mb !== undefined && stats.pgvector_size_mb !== null) {
            totalMb += stats.pgvector_size_mb;
            if (tool.enabled) {
              enabledMb += stats.pgvector_size_mb;
            }
            hasData = true;
          }
        } else {
          // FAISS: Use memory estimate
          hasFaiss = true;
          if (stats.estimated_memory_mb !== undefined && stats.estimated_memory_mb !== null) {
            totalMb += stats.estimated_memory_mb;
            if (tool.enabled) {
              enabledMb += stats.estimated_memory_mb;
            }
            hasData = true;
          } else if (embeddingDimensions && embeddingDimensions > 0) {
            // Fallback memory calculation for FAISS
            const bytesPerEmbedding = embeddingDimensions * 4 * 1.15;
            const memory = (stats.embedding_count * bytesPerEmbedding) / (1024 * 1024);
            totalMb += memory;
            if (tool.enabled) {
              enabledMb += memory;
            }
            hasData = true;
          }
        }
      }
    }

    if (!hasData) return null;

    return { totalStorage: totalMb, enabledStorage: enabledMb, hasPgvector, hasFaiss };
  };

  const storageEstimate = calculateTotalStorage();

  // Load filesystem indexer tools
  const loadTools = useCallback(async () => {
    try {
      setLoading(true);
      const allTools = await api.listToolConfigs();
      // Filter to only filesystem indexers
      const fsTools = allTools.filter(t => t.tool_type === 'filesystem_indexer');
      setTools(fsTools);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load filesystem indexes');
    } finally {
      setLoading(false);
    }
  }, []);

  // Fetch job status and stats for all filesystem tools
  const fetchStatus = useCallback(async () => {
    if (tools.length === 0) return;

    const jobUpdates: Record<string, FilesystemIndexJob | null> = {};
    const statsUpdates: Record<string, FilesystemIndexStats | null> = {};

    await Promise.all(tools.map(async (tool) => {
      try {
        const [jobs, stats] = await Promise.all([
          api.getFilesystemJobs(tool.id),
          api.getFilesystemStats(tool.id).catch(() => null),
        ]);
        // Get the most recent active job, or most recent completed job
        const activeJob = jobs.find((j: typeof jobs[0]) => j.status === 'pending' || j.status === 'indexing');
        const recentJob = activeJob || jobs[0] || null;
        jobUpdates[tool.id] = recentJob;
        statsUpdates[tool.id] = stats;
      } catch (err) {
        console.warn(`Failed to fetch status for ${tool.id}:`, err);
      }
    }));

    setFilesystemJobs(prev => ({ ...prev, ...jobUpdates }));
    setFilesystemStats(prev => ({ ...prev, ...statsUpdates }));
  }, [tools]);

  // Initial load
  useEffect(() => {
    loadTools();
  }, [loadTools]);

  // Fetch status when tools change
  useEffect(() => {
    if (tools.length > 0) {
      fetchStatus();
    }
  }, [tools, fetchStatus]);

  // Fast polling when jobs are active
  useEffect(() => {
    const hasActiveJob = Object.values(filesystemJobs).some(
      job => job && (job.status === 'pending' || job.status === 'indexing')
    );

    if (hasActiveJob) {
      pollRef.current = setInterval(fetchStatus, ACTIVE_JOB_POLL_INTERVAL);
    } else {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
      }
    };
  }, [filesystemJobs, fetchStatus]);

  // Handlers
  const handleStartIndex = async (toolId: string, fullReindex: boolean) => {
    try {
      setIndexingToolId(toolId);
      await api.triggerFilesystemIndex(toolId, fullReindex);
      setSuccess(fullReindex ? 'Full reindex started' : 'Indexing started');
      setTimeout(() => setSuccess(null), 3000);
      await fetchStatus();
      onJobsChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start indexing');
    } finally {
      setIndexingToolId(null);
    }
  };

  const handleDeleteIndex = async (toolId: string) => {
    setConfirmation({
      message: 'Are you sure you want to delete all indexed embeddings? This cannot be undone.',
      onConfirm: async () => {
        setConfirmation(null);
        try {
          const result = await api.deleteFilesystemIndex(toolId);
          setSuccess(`Deleted ${result.deleted_count} embeddings`);
          setTimeout(() => setSuccess(null), 3000);
          await fetchStatus();
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to delete index');
        }
      },
    });
  };

  const handleDeleteTool = async (toolId: string) => {
    // No confirmation modal - DeleteConfirmButton already has countdown confirmation
    try {
      await api.deleteToolConfig(toolId);
      await loadTools();
      onToolsChanged?.();
      setSuccess('Filesystem index deleted');
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete');
    }
  };

  const handleToggleTool = async (toolId: string, enabled: boolean) => {
    try {
      await api.toggleToolConfig(toolId, enabled);
      await loadTools();
      onToolsChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to toggle');
    }
  };

  const handleRename = async (toolId: string, newName: string) => {
    try {
      await api.updateToolConfig(toolId, { name: newName });
      await loadTools();
      onToolsChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rename failed');
      setTimeout(() => setError(null), 3000);
      throw err;
    }
  };

  const handleDescriptionUpdate = async (toolId: string, newDesc: string) => {
    try {
      await api.updateToolConfig(toolId, { description: newDesc });
      await loadTools();
      onToolsChanged?.();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Update failed');
      setTimeout(() => setError(null), 3000);
      throw err;
    }
  };

  const handleEdit = (tool: ToolConfig) => {
    setEditingTool(tool);
    setShowWizard(true);
  };

  const handleAddNew = () => {
    setEditingTool(null);
    setShowWizard(true);
  };

  const handleWizardClose = () => {
    setShowWizard(false);
    setEditingTool(null);
  };

  const handleWizardSave = async () => {
    setShowWizard(false);
    setEditingTool(null);
    await loadTools();
    onToolsChanged?.();
    setSuccess('Filesystem index configuration saved');
    setTimeout(() => setSuccess(null), 3000);
  };

  return (
    <div className="card filesystem-index-panel">
      {/* Confirmation Modal */}
      {confirmation && (
        <div className="modal-overlay" onClick={() => setConfirmation(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm Action</h3>
              <button className="modal-close" onClick={() => setConfirmation(null)}>x</button>
            </div>
            <div className="modal-body">
              <p>{confirmation.message}</p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setConfirmation(null)}>
                Cancel
              </button>
              <button className="btn btn-danger" onClick={confirmation.onConfirm}>
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="section-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <h2>Filesystem Indexes</h2>
          {storageEstimate && tools.length > 0 && (
            <span
              className="memory-badge"
              title={`${storageEstimate.enabledStorage.toFixed(1)} MB active / ${storageEstimate.totalStorage.toFixed(1)} MB total${storageEstimate.hasPgvector && !storageEstimate.hasFaiss ? ' (pgvector disk)' : storageEstimate.hasFaiss && !storageEstimate.hasPgvector ? ' (FAISS RAM)' : ' (mixed)'}`}
            >
              <HardDrive size={12} />
              {storageEstimate.enabledStorage.toFixed(0)} MB
              {storageEstimate.enabledStorage !== storageEstimate.totalStorage && (
                <span className="memory-total"> / {storageEstimate.totalStorage.toFixed(0)}</span>
              )}
            </span>
          )}
        </div>
        <AnimatedCreateButton
          isExpanded={showWizard}
          onClick={() => showWizard ? handleWizardClose() : handleAddNew()}
          label="Add Filesystem Index"
        />
      </div>

      {showWizard ? (
        <ToolWizard
          existingTool={editingTool}
          onClose={handleWizardClose}
          onSave={handleWizardSave}
          defaultToolType="filesystem_indexer"
          embedded={true}
        />
      ) : (
        <>
      <p className="section-description">
        Index files from mounted volumes (Docker mounts, SMB shares, NFS) for RAG queries.
        Uses pgvector for efficient similarity search.
      </p>

      {error && (
        <div className="error-banner">
          {error}
          <button onClick={() => setError(null)}>x</button>
        </div>
      )}

      {success && (
        <div className="success-banner">
          {success}
        </div>
      )}

      {loading && tools.length === 0 && (
        <div className="empty-state">Loading...</div>
      )}

      {!loading && tools.length === 0 && (
        <div className="empty-state">
          <p>No filesystem indexes configured yet.</p>
          <p className="muted">
            Click "Add Filesystem Index" to set up indexing for files in mounted volumes.
          </p>
        </div>
      )}

      {tools.map((tool) => (
        <FilesystemIndexCard
          key={tool.id}
          tool={tool}
          activeJob={filesystemJobs[tool.id] || null}
          stats={filesystemStats[tool.id] || null}
          onStartIndex={handleStartIndex}
          onDeleteIndex={handleDeleteIndex}
          onEdit={handleEdit}
          onDelete={handleDeleteTool}
          onToggle={handleToggleTool}
          onRename={handleRename}
          onDescriptionUpdate={handleDescriptionUpdate}
          indexing={indexingToolId === tool.id}
          embeddingDimensions={embeddingDimensions}
        />
      ))}
        </>
      )}
    </div>
  );
}
