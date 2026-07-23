import { useCallback, useState } from 'react';
import { Upload } from 'lucide-react';
import { api } from '@/api';
import type { ImportFaissIndexResponse } from '@/types';
import { FileDropZone } from './shared/FileDropZone';

interface ImportFaissFormProps {
  onImported?: (result: ImportFaissIndexResponse) => void;
  onCancel?: () => void;
}

/**
 * Import a previously-exported FAISS index zip and re-create the index in place.
 *
 * The zip is expected to have been produced by the "Download" button on the
 * document index list (it contains index.faiss, index.pkl, and an optional
 * metadata.json that restores the original description, source, branch, and
 * config_snapshot).
 */
export function ImportFaissForm({ onImported, onCancel }: ImportFaissFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [overwrite, setOverwrite] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showRiskConfirm, setShowRiskConfirm] = useState(false);
  const [status, setStatus] = useState<{
    type: 'info' | 'success' | 'error';
    message: string;
  } | null>(null);
  const [result, setResult] = useState<ImportFaissIndexResponse | null>(null);

  const reset = useCallback(() => {
    setFile(null);
    setName('');
    setDescription('');
    setOverwrite(false);
    setShowRiskConfirm(false);
    setStatus(null);
    setResult(null);
  }, []);

  const handleFile = useCallback((selected: File) => {
    setFile(selected);
    setStatus(null);
    setResult(null);
    // Default the index name to the zip filename (without extension)
    setName((current) => current || selected.name.replace(/\.zip$/i, ''));
  }, []);

  const handleImport = async () => {
    if (!file) {
      setStatus({ type: 'error', message: 'Please select a FAISS zip file to import.' });
      return;
    }
    const trimmedName = name.trim();
    if (!trimmedName) {
      setStatus({ type: 'error', message: 'Please provide a name for the imported index.' });
      return;
    }

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Uploading and restoring FAISS index...' });

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', trimmedName);
      if (description) {
        formData.append('description', description);
      }
      formData.append('overwrite', overwrite ? 'true' : 'false');

      const response = await api.importFaissIndex(formData);
      setResult(response);
      setStatus({ type: 'success', message: response.message });
      onImported?.(response);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Import failed';
      setStatus({ type: 'error', message });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div>
      <div className="form-group">
        <FileDropZone
          file={file}
          accept=".zip,application/zip"
          disabled={isLoading}
          icon={<Upload size={28} />}
          prompt={<>Drag &amp; drop an exported FAISS zip here, or click to browse</>}
          helpText={<>Produced by the &quot;Download&quot; button on an existing document index</>}
          onFileSelected={handleFile}
        />
      </div>

      {file && (
        <>
          <p className="field-help" style={{ marginBottom: '16px' }}>
            <strong>Import a FAISS index</strong> to instantly re-create an index that was
            previously exported from this server (or another server). The original description,
            source, and configuration are restored from <code>metadata.json</code> inside the zip.
          </p>

          <div className="form-group">
            <label>Index Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., odev_proj"
              disabled={isLoading}
            />
            <small style={{ color: 'var(--color-text-muted)', fontSize: '0.8rem' }}>
              Defaults to the zip filename. The server will use the name embedded in{' '}
              <code>metadata.json</code> if you leave this blank or clear it (the name is restored
              from the original export).
            </small>
          </div>

          <div className="form-group">
            <label>Description (optional override)</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Leave blank to use the description from metadata.json"
              rows={3}
              style={{ width: '100%', resize: 'vertical', minHeight: '60px' }}
              disabled={isLoading}
            />
            <small style={{ color: 'var(--color-text-muted)', fontSize: '0.8rem' }}>
              Helps the AI understand when to search this index. Overrides the description from the
              zip when set.
            </small>
          </div>

          <div className="form-group">
            <label
              style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}
            >
              <input
                type="checkbox"
                checked={overwrite}
                onChange={(e) => setOverwrite(e.target.checked)}
                style={{ width: 'auto', margin: 0 }}
                disabled={isLoading}
              />
              Overwrite existing index with the same name
            </label>
            <small
              style={{
                color: 'var(--color-text-muted)',
                fontSize: '0.8rem',
                display: 'block',
                marginTop: '0.25rem',
              }}
            >
              When unchecked, the import fails if an index with the same name already exists.
            </small>
          </div>

          <div className="wizard-actions">
            {onCancel && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  reset();
                  onCancel();
                }}
                disabled={isLoading}
              >
                {result ? 'Close' : 'Cancel'}
              </button>
            )}
            {!result && (
              <button
                type="button"
                className="btn"
                onClick={() => setShowRiskConfirm(true)}
                disabled={isLoading || !file || !name.trim()}
              >
                {isLoading ? 'Importing...' : 'Import FAISS Index'}
              </button>
            )}
          </div>
        </>
      )}

      {result && (
        <div
          style={{
            marginTop: '16px',
            padding: '12px',
            background: 'var(--color-success-light)',
            border: '1px solid var(--color-success-border)',
            borderRadius: '8px',
            fontSize: '0.9rem',
          }}
        >
          <div style={{ marginBottom: '6px' }}>
            <strong>Imported</strong> &quot;{result.display_name || result.name}&quot;
          </div>
          <div>Chunks: {result.chunk_count.toLocaleString()}</div>
          <div>Source type: {result.source_type}</div>
          <div>Vector store: {result.vector_store_type}</div>
          {result.description && (
            <div style={{ marginTop: '6px', color: 'var(--color-text-secondary)' }}>
              Description: {result.description}
            </div>
          )}
        </div>
      )}

      {status && <div className={`status-message ${status.type}`}>{status.message}</div>}

      {showRiskConfirm && (
        <div className="modal-overlay" onClick={() => setShowRiskConfirm(false)}>
          <div className="modal-content modal-small" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Confirm FAISS Import</h3>
              <button className="modal-close" onClick={() => setShowRiskConfirm(false)}>
                x
              </button>
            </div>
            <div className="modal-body">
              <p>
                FAISS imports restore server-side index files, including serialized index metadata.
                Only import archives that were exported from a Ragtime instance you trust.
              </p>
              <p>
                This action requires an authenticated admin session and can overwrite an existing
                index if that option is selected.
              </p>
            </div>
            <div className="modal-footer">
              <button className="btn btn-secondary" onClick={() => setShowRiskConfirm(false)}>
                Cancel
              </button>
              <button
                className="btn btn-danger"
                onClick={() => {
                  setShowRiskConfirm(false);
                  void handleImport();
                }}
              >
                I Understand, Import
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
