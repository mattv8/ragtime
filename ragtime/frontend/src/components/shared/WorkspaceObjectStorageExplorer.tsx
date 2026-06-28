import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ArrowLeft,
  ChevronRight,
  Download,
  File as FileIcon,
  Folder,
  HardDrive,
  Loader2,
  Pencil,
  Plus,
  RefreshCw,
  Upload,
} from 'lucide-react';

import { api } from '@/api/client';
import type {
  UserSpaceObjectStorageBucket,
  UserSpaceObjectStorageConfig,
  UserSpaceObjectStorageEntry,
  UserSpaceObjectStorageListResponse,
} from '@/types';

import { DeleteConfirmButton } from '../DeleteConfirmButton';
import { WorkspaceObjectStorageWizard } from '../MountSourceWizard';

interface WorkspaceObjectStorageExplorerProps {
  workspaceId: string;
  config: UserSpaceObjectStorageConfig | null;
  loading: boolean;
  canManage: boolean;
  onConfigChange: (config: UserSpaceObjectStorageConfig) => void;
  onNotify?: (message: string, tone: 'success' | 'error') => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '';
  try {
    return new Date(value).toLocaleString();
  } catch {
    return '';
  }
}

export function WorkspaceObjectStorageExplorer({
  workspaceId,
  config,
  loading,
  canManage,
  onConfigChange,
  onNotify,
}: WorkspaceObjectStorageExplorerProps) {
  const [selectedBucket, setSelectedBucket] = useState<UserSpaceObjectStorageBucket | null>(null);
  const [prefix, setPrefix] = useState('');
  const [listing, setListing] = useState<UserSpaceObjectStorageListResponse | null>(null);
  const [listingLoading, setListingLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [showBucketWizard, setShowBucketWizard] = useState(false);
  const [editingBucket, setEditingBucket] = useState<UserSpaceObjectStorageBucket | null>(null);
  const [deletingBucket, setDeletingBucket] = useState<string | null>(null);
  const [renamingBucketName, setRenamingBucketName] = useState<string | null>(null);
  const [bucketRenameDraft, setBucketRenameDraft] = useState('');
  const [savingBucketRename, setSavingBucketRename] = useState(false);

  const [renamingKey, setRenamingKey] = useState<string | null>(null);
  const [objectRenameDraft, setObjectRenameDraft] = useState('');
  const [savingObjectRename, setSavingObjectRename] = useState(false);

  const uploadInputRef = useRef<HTMLInputElement | null>(null);

  const notify = useCallback(
    (message: string, tone: 'success' | 'error') => {
      if (onNotify) onNotify(message, tone);
      if (tone === 'error') setError(message);
    },
    [onNotify],
  );

  const loadObjects = useCallback(
    async (bucketName: string, nextPrefix: string) => {
      setListingLoading(true);
      setError(null);
      try {
        const result = await api.listUserSpaceObjectStorageObjects(
          workspaceId,
          bucketName,
          nextPrefix,
        );
        setListing(result);
        setPrefix(result.prefix);
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to load objects', 'error');
      } finally {
        setListingLoading(false);
      }
    },
    [notify, workspaceId],
  );

  const openBucket = useCallback(
    async (bucket: UserSpaceObjectStorageBucket) => {
      setSelectedBucket(bucket);
      setPrefix('');
      setListing(null);
      await loadObjects(bucket.name, '');
    },
    [loadObjects],
  );

  const closeBucket = useCallback(() => {
    setSelectedBucket(null);
    setListing(null);
    setPrefix('');
    setRenamingKey(null);
  }, []);

  // If the bucket disappears after a config refresh (deleted/renamed), reset.
  useEffect(() => {
    if (!selectedBucket || !config) return;
    if (!config.buckets.some((bucket) => bucket.name === selectedBucket.name)) {
      closeBucket();
    }
  }, [closeBucket, config, selectedBucket]);

  const handleUploadFiles = useCallback(
    async (files: FileList | null) => {
      if (!selectedBucket || !files || files.length === 0) return;
      setUploading(true);
      setError(null);
      try {
        for (const file of Array.from(files)) {
          await api.uploadUserSpaceObjectStorageObject(
            workspaceId,
            selectedBucket.name,
            file,
            prefix,
          );
        }
        notify(`Uploaded ${files.length} file${files.length === 1 ? '' : 's'}`, 'success');
        await loadObjects(selectedBucket.name, prefix);
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to upload', 'error');
      } finally {
        setUploading(false);
        if (uploadInputRef.current) uploadInputRef.current.value = '';
      }
    },
    [loadObjects, notify, prefix, selectedBucket, workspaceId],
  );

  const handleDownloadObject = useCallback(
    async (entry: UserSpaceObjectStorageEntry) => {
      if (!selectedBucket) return;
      try {
        await api.downloadUserSpaceObjectStorageObject(workspaceId, selectedBucket.name, entry.key);
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to download', 'error');
      }
    },
    [notify, selectedBucket, workspaceId],
  );

  const handleDeleteObject = useCallback(
    async (entry: UserSpaceObjectStorageEntry) => {
      if (!selectedBucket) return;
      try {
        await api.deleteUserSpaceObjectStorageObject(workspaceId, selectedBucket.name, entry.key);
        notify(`Deleted ${entry.name}`, 'success');
        await loadObjects(selectedBucket.name, prefix);
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to delete object', 'error');
      }
    },
    [loadObjects, notify, prefix, selectedBucket, workspaceId],
  );

  const handleSaveObjectRename = useCallback(
    async (entry: UserSpaceObjectStorageEntry) => {
      if (!selectedBucket) return;
      const trimmed = objectRenameDraft.trim();
      if (!trimmed || trimmed === entry.name) {
        setRenamingKey(null);
        return;
      }
      const parentSegment = prefix ? `${prefix}/` : '';
      const newKey = `${parentSegment}${trimmed.replace(/^\/+/, '')}`;
      setSavingObjectRename(true);
      try {
        await api.renameUserSpaceObjectStorageObject(workspaceId, selectedBucket.name, entry.key, {
          new_key: newKey,
        });
        notify(`Renamed to ${trimmed}`, 'success');
        setRenamingKey(null);
        await loadObjects(selectedBucket.name, prefix);
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to rename object', 'error');
      } finally {
        setSavingObjectRename(false);
      }
    },
    [loadObjects, notify, objectRenameDraft, prefix, selectedBucket, workspaceId],
  );

  const handleDeleteBucket = useCallback(
    async (bucketName: string) => {
      setDeletingBucket(bucketName);
      try {
        await api.deleteUserSpaceObjectStorageBucket(workspaceId, bucketName);
        const next = await api.getUserSpaceObjectStorageConfig(workspaceId);
        onConfigChange(next);
        notify(`Deleted bucket ${bucketName}`, 'success');
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to delete bucket', 'error');
      } finally {
        setDeletingBucket(null);
      }
    },
    [notify, onConfigChange, workspaceId],
  );

  const handleSaveBucketRename = useCallback(
    async (bucket: UserSpaceObjectStorageBucket) => {
      const trimmed = bucketRenameDraft.trim().toLowerCase();
      if (!trimmed || trimmed === bucket.name) {
        setRenamingBucketName(null);
        return;
      }
      setSavingBucketRename(true);
      try {
        const next = await api.updateUserSpaceObjectStorageBucket(workspaceId, bucket.name, {
          new_name: trimmed,
        });
        onConfigChange(next);
        notify(`Renamed bucket to ${trimmed}`, 'success');
        setRenamingBucketName(null);
      } catch (err) {
        notify(err instanceof Error ? err.message : 'Failed to rename bucket', 'error');
      } finally {
        setSavingBucketRename(false);
      }
    },
    [bucketRenameDraft, notify, onConfigChange, workspaceId],
  );

  const handleWizardSaved = useCallback(
    (next: UserSpaceObjectStorageConfig) => {
      onConfigChange(next);
      setShowBucketWizard(false);
      setEditingBucket(null);
    },
    [onConfigChange],
  );

  // Breadcrumb segments for prefix navigation
  const prefixSegments = prefix ? prefix.split('/') : [];

  if (showBucketWizard) {
    return (
      <WorkspaceObjectStorageWizard
        workspaceId={workspaceId}
        existingBucket={editingBucket}
        existingBucketNames={config?.buckets.map((bucket) => bucket.name) ?? []}
        onClose={() => {
          setShowBucketWizard(false);
          setEditingBucket(null);
        }}
        onSaved={handleWizardSaved}
      />
    );
  }

  // ── Bucket browse view ──────────────────────────────────────────────
  if (selectedBucket) {
    const entries = listing?.entries ?? [];
    return (
      <div className="userspace-object-explorer">
        <input
          ref={uploadInputRef}
          type="file"
          multiple
          aria-hidden="true"
          tabIndex={-1}
          style={{ display: 'none' }}
          onChange={(event) => void handleUploadFiles(event.target.files)}
        />
        <div className="userspace-object-toolbar">
          <nav className="userspace-sqlite-breadcrumb" aria-label="Object storage breadcrumb">
            <span className="userspace-sqlite-breadcrumb-item">
              <button
                type="button"
                className="userspace-sqlite-breadcrumb-link"
                onClick={closeBucket}
              >
                Buckets
              </button>
            </span>
            <span className="userspace-sqlite-breadcrumb-item">
              <ChevronRight size={14} className="userspace-sqlite-breadcrumb-sep" />
              <button
                type="button"
                className="userspace-sqlite-breadcrumb-link"
                onClick={() => void loadObjects(selectedBucket.name, '')}
              >
                {selectedBucket.name}
              </button>
            </span>
            {prefixSegments.map((segment, idx) => {
              const segPrefix = prefixSegments.slice(0, idx + 1).join('/');
              const isLast = idx === prefixSegments.length - 1;
              return (
                <span key={segPrefix} className="userspace-sqlite-breadcrumb-item">
                  <ChevronRight size={14} className="userspace-sqlite-breadcrumb-sep" />
                  {isLast ? (
                    <span className="userspace-sqlite-breadcrumb-current">{segment}</span>
                  ) : (
                    <button
                      type="button"
                      className="userspace-sqlite-breadcrumb-link"
                      onClick={() => void loadObjects(selectedBucket.name, segPrefix)}
                    >
                      {segment}
                    </button>
                  )}
                </span>
              );
            })}
          </nav>
          <div className="userspace-object-toolbar-actions">
            <button
              type="button"
              className="btn btn-secondary btn-sm btn-icon"
              onClick={() => void loadObjects(selectedBucket.name, prefix)}
              disabled={listingLoading}
              title="Refresh"
            >
              <RefreshCw size={14} className={listingLoading ? 'spinning' : undefined} />
            </button>
            {listing?.parent_prefix !== null && listing?.parent_prefix !== undefined && (
              <button
                type="button"
                className="btn btn-secondary btn-sm btn-icon"
                onClick={() => void loadObjects(selectedBucket.name, listing.parent_prefix ?? '')}
                title="Up one level"
              >
                <ArrowLeft size={14} />
              </button>
            )}
            {canManage && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={() => uploadInputRef.current?.click()}
                disabled={uploading}
              >
                {uploading ? <Loader2 size={14} className="spinning" /> : <Upload size={14} />}{' '}
                Upload
              </button>
            )}
          </div>
        </div>

        <div className="userspace-object-meta">
          <span>
            {listing?.total_objects ?? 0} object{(listing?.total_objects ?? 0) === 1 ? '' : 's'}
          </span>
          <span>{formatBytes(listing?.total_bytes ?? 0)}</span>
        </div>

        {error && <div className="userspace-object-error">{error}</div>}

        <div className="userspace-sqlite-table-wrapper">
          {listingLoading && !listing ? (
            <div className="userspace-sqlite-empty">
              <Loader2 size={16} className="spinning" /> Loading…
            </div>
          ) : entries.length === 0 ? (
            <div className="userspace-sqlite-empty">
              <FileIcon size={24} />
              <p>This {prefix ? 'folder' : 'bucket'} is empty.</p>
              {canManage && (
                <button
                  type="button"
                  className="btn btn-primary btn-sm"
                  onClick={() => uploadInputRef.current?.click()}
                >
                  <Upload size={14} /> Upload a file
                </button>
              )}
            </div>
          ) : (
            <table className="userspace-sqlite-table userspace-object-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Size</th>
                  <th>Type</th>
                  <th>Modified</th>
                  <th className="userspace-sqlite-actions-col" />
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  if (entry.entry_type === 'prefix') {
                    return (
                      <tr
                        key={`prefix:${entry.key}`}
                        className="userspace-object-row userspace-object-row-prefix"
                        onClick={() => void loadObjects(selectedBucket.name, entry.key)}
                      >
                        <td>
                          <span className="userspace-object-name">
                            <Folder size={14} className="userspace-object-icon" />
                            <span>{entry.name}</span>
                          </span>
                        </td>
                        <td>—</td>
                        <td>Folder</td>
                        <td>
                          {entry.object_count} item{entry.object_count === 1 ? '' : 's'}
                        </td>
                        <td className="userspace-sqlite-actions-col" />
                      </tr>
                    );
                  }
                  const isRenaming = renamingKey === entry.key;
                  return (
                    <tr key={`object:${entry.key}`} className="userspace-object-row">
                      <td>
                        {isRenaming ? (
                          <input
                            type="text"
                            className="userspace-sqlite-cell-input"
                            autoFocus
                            value={objectRenameDraft}
                            onChange={(event) => setObjectRenameDraft(event.target.value)}
                            onClick={(event) => event.stopPropagation()}
                            onBlur={() => void handleSaveObjectRename(entry)}
                            onKeyDown={(event) => {
                              if (event.key === 'Enter') {
                                event.preventDefault();
                                event.currentTarget.blur();
                              }
                              if (event.key === 'Escape') {
                                event.preventDefault();
                                setRenamingKey(null);
                              }
                            }}
                            disabled={savingObjectRename}
                          />
                        ) : (
                          <span className="userspace-object-name">
                            <FileIcon size={14} className="userspace-object-icon" />
                            <span>{entry.name}</span>
                          </span>
                        )}
                      </td>
                      <td>{formatBytes(entry.size_bytes)}</td>
                      <td className="userspace-object-type" title={entry.content_type ?? undefined}>
                        {entry.content_type ?? '—'}
                      </td>
                      <td>{formatTimestamp(entry.updated_at)}</td>
                      <td className="userspace-sqlite-actions-col">
                        <div className="userspace-object-row-actions">
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm btn-icon"
                            onClick={() => void handleDownloadObject(entry)}
                            title="Download"
                          >
                            <Download size={14} />
                          </button>
                          {canManage && (
                            <>
                              <button
                                type="button"
                                className="btn btn-secondary btn-sm btn-icon"
                                onClick={() => {
                                  setRenamingKey(entry.key);
                                  setObjectRenameDraft(entry.name);
                                }}
                                title="Rename"
                              >
                                <Pencil size={14} />
                              </button>
                              <DeleteConfirmButton
                                onDelete={() => handleDeleteObject(entry)}
                                className="btn btn-danger btn-sm"
                                title="Delete object"
                                buttonText="Delete"
                              />
                            </>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </div>
    );
  }

  // ── Buckets grid view ───────────────────────────────────────────────
  return (
    <div className="userspace-object-explorer">
      <div className="userspace-object-header">
        <p className="userspace-muted" style={{ margin: 0 }}>
          S3-compatible object storage with auto-injected credentials at runtime. Click a bucket to
          browse its files.
        </p>
        {canManage && (
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => {
              setEditingBucket(null);
              setShowBucketWizard(true);
            }}
          >
            <Plus size={14} /> New Bucket
          </button>
        )}
      </div>

      {error && <div className="userspace-object-error">{error}</div>}

      {loading ? (
        <div className="userspace-sqlite-empty">
          <Loader2 size={16} className="spinning" /> Loading…
        </div>
      ) : !config ? (
        <div className="userspace-sqlite-empty">
          <HardDrive size={24} />
          <p>Object storage is unavailable for this workspace.</p>
        </div>
      ) : config.buckets.length === 0 ? (
        <div className="userspace-sqlite-empty">
          <HardDrive size={24} />
          <p>No buckets yet.</p>
          {canManage && (
            <button
              type="button"
              className="btn btn-primary btn-sm"
              onClick={() => setShowBucketWizard(true)}
            >
              <Plus size={14} /> New Bucket
            </button>
          )}
        </div>
      ) : (
        <div className="userspace-sqlite-card-grid">
          {config.buckets.map((bucket) => {
            const isRenaming = renamingBucketName === bucket.name;
            return (
              <div key={bucket.name} className="userspace-sqlite-card">
                {isRenaming ? (
                  <div className="userspace-sqlite-card-main" style={{ cursor: 'default' }}>
                    <div className="userspace-sqlite-card-title">
                      <HardDrive size={16} />
                      <input
                        type="text"
                        className="userspace-sqlite-cell-input"
                        autoFocus
                        value={bucketRenameDraft}
                        onChange={(event) =>
                          setBucketRenameDraft(
                            event.target.value.replace(/[^a-zA-Z0-9-]/g, '-').toLowerCase(),
                          )
                        }
                        onBlur={() => void handleSaveBucketRename(bucket)}
                        onKeyDown={(event) => {
                          if (event.key === 'Enter') {
                            event.preventDefault();
                            event.currentTarget.blur();
                          }
                          if (event.key === 'Escape') {
                            event.preventDefault();
                            setRenamingBucketName(null);
                          }
                        }}
                        disabled={savingBucketRename}
                      />
                    </div>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="userspace-sqlite-card-main"
                    onClick={() => void openBucket(bucket)}
                  >
                    <div className="userspace-sqlite-card-title">
                      <HardDrive size={16} />
                      <span title={bucket.name}>{bucket.name}</span>
                      {bucket.is_default && (
                        <span className="userspace-sqlite-card-badge">default</span>
                      )}
                    </div>
                    <div className="userspace-sqlite-card-meta">
                      <span>{bucket.description || 'No description'}</span>
                    </div>
                    <div className="userspace-sqlite-card-meta">
                      <span>public: {bucket.public_prefix}</span>
                      <span>private: {bucket.private_prefix}</span>
                    </div>
                  </button>
                )}
                {canManage && !isRenaming && (
                  <div className="userspace-sqlite-card-actions">
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm btn-icon"
                      onClick={() => {
                        setEditingBucket(bucket);
                        setShowBucketWizard(true);
                      }}
                      title="Edit bucket"
                    >
                      <Pencil size={14} />
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm btn-icon"
                      onClick={() => {
                        setRenamingBucketName(bucket.name);
                        setBucketRenameDraft(bucket.name);
                      }}
                      title="Rename bucket"
                    >
                      <HardDrive size={14} />
                    </button>
                    <DeleteConfirmButton
                      onDelete={() => handleDeleteBucket(bucket.name)}
                      className="btn btn-danger btn-sm"
                      title={
                        config.buckets.length <= 1
                          ? 'At least one bucket must remain'
                          : 'Delete bucket'
                      }
                      buttonText="Delete"
                      disabled={config.buckets.length <= 1 || deletingBucket === bucket.name}
                    />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
