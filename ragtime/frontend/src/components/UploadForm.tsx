import { useState, useCallback, useEffect, type DragEvent, type ChangeEvent } from 'react';
import { api } from '@/api';
import type { IndexJob } from '@/types';

interface UploadFormProps {
  onJobCreated: () => void;
}

type StatusType = 'info' | 'success' | 'error' | null;

/** Extract index name from archive filename (strip extension) */
function getIndexNameFromFile(filename: string): string {
  return filename
    .replace(/\.(zip|tar|tar\.gz|tgz|tar\.bz2|tbz2)$/i, '')
    .replace(/[^a-zA-Z0-9_-]/g, '-')
    .toLowerCase();
}

export function UploadForm({ onJobCreated }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const [indexName, setIndexName] = useState('');
  const [description, setDescription] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: null,
    message: '',
  });

  // Auto-fill index name when file is selected
  useEffect(() => {
    if (file) {
      setIndexName(getIndexNameFromFile(file.name));
    }
  }, [file]);

  const handleDragOver = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files.length) {
      setFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      setFile(e.target.files[0]);
    }
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!file) {
      setStatus({ type: 'error', message: 'Please select a file' });
      return;
    }
    if (!indexName.trim()) {
      setStatus({ type: 'error', message: 'Please enter an index name' });
      return;
    }

    const form = e.currentTarget;
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', indexName);
    formData.append('description', description);
    formData.append('file_patterns', (form.elements.namedItem('file_patterns') as HTMLInputElement).value);
    formData.append('exclude_patterns', (form.elements.namedItem('exclude_patterns') as HTMLInputElement).value);
    formData.append('chunk_size', (form.elements.namedItem('chunk_size') as HTMLInputElement).value);
    formData.append('chunk_overlap', (form.elements.namedItem('chunk_overlap') as HTMLInputElement).value);

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Uploading and processing...' });
    setProgress(30);

    try {
      const job: IndexJob = await api.uploadAndIndex(formData);
      setProgress(100);
      setStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      setFile(null);
      setIndexName('');
      setDescription('');
      onJobCreated();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Upload failed'}` });
    } finally {
      setIsLoading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {/* File drop area - always visible */}
      <div className="form-group">
        <div
          className={`file-input-wrapper ${isDragOver ? 'dragover' : ''} ${file ? 'has-file' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <div className="icon">↑</div>
          <div>Drag & drop an archive file here, or click to browse</div>
          <div style={{ fontSize: '0.85rem', color: '#888', marginTop: 8 }}>
            Supported: .zip, .tar, .tar.gz, .tar.bz2
          </div>
          {file && <div className="file-name">{file.name}</div>}
          <input
            type="file"
            name="file"
            accept=".zip,.tar,.tar.gz,.tgz,.tar.bz2,.tbz2"
            onChange={handleFileChange}
          />
        </div>
      </div>

      {/* Show remaining fields only after file is selected */}
      {file && (
        <>
          <div className="form-group">
            <label>Index Name *</label>
            <input
              type="text"
              name="name"
              value={indexName}
              onChange={(e) => setIndexName(e.target.value)}
              placeholder="e.g., odoo-17, my-codebase"
              required
            />
          </div>

          <div className="form-group">
            <label>Description (for AI context)</label>
            <textarea
              name="description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Describe what this index contains so the AI knows when to use it, e.g., 'Odoo 17 modules source code including accounting, inventory, and sales apps'"
              rows={2}
              style={{ resize: 'vertical', minHeight: '60px' }}
            />
          </div>

          <div className="row">
            <div className="form-group">
              <label>File Patterns (comma-separated)</label>
              <input
                type="text"
                name="file_patterns"
                defaultValue="**/*.py,**/*.md,**/*.xml,**/*.rst"
              />
            </div>
            <div className="form-group">
              <label>Exclude Patterns</label>
              <input
                type="text"
                name="exclude_patterns"
                defaultValue="**/node_modules/**,**/__pycache__/**,**/.git/**"
              />
            </div>
          </div>

          <div className="row">
            <div className="form-group">
              <label>Chunk Size</label>
              <input
                type="number"
                name="chunk_size"
                defaultValue={1000}
                min={100}
                max={4000}
              />
            </div>
            <div className="form-group">
              <label>Chunk Overlap</label>
              <input
                type="number"
                name="chunk_overlap"
                defaultValue={200}
                min={0}
                max={1000}
              />
            </div>
          </div>

          <button type="submit" className="btn" disabled={isLoading}>
            Create Index
          </button>
        </>
      )}

      {progress > 0 && (
        <div className="progress-bar">
          <div className="fill" style={{ width: `${progress}%` }} />
        </div>
      )}

      {status.type && (
        <div className={`status-message ${status.type}`}>{status.message}</div>
      )}
    </form>
  );
}
