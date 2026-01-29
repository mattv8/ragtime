import { useState, useCallback, type DragEvent, type ChangeEvent } from 'react';
import { api } from '@/api';
import type { IndexJob, IndexAnalysisResult, OcrMode } from '@/types';
import { DescriptionField } from './DescriptionField';
import { AnalysisStats } from './AnalysisStats';
import { IndexConfigFields } from './IndexConfigFields';

interface UploadFormProps {
  onJobCreated: () => void;
  onCancel?: () => void;
  onAnalysisStart?: () => void;
  onAnalysisComplete?: () => void;
  /** Called when user wants to navigate to settings */
  onNavigateToSettings?: () => void;
}

type StatusType = 'info' | 'success' | 'error' | null;
type WizardStep = 'upload' | 'analyzing' | 'review' | 'indexing';

// Default file patterns to include all files
const DEFAULT_FILE_PATTERNS = '**/*';

/** Extract index name from archive filename (strip extension) */
function getIndexNameFromFile(filename: string): string {
  return filename
    .replace(/\.(zip|tar|tar\.gz|tgz|tar\.bz2|tbz2)$/i, '')
    .replace(/[^a-zA-Z0-9_-]/g, '-')
    .toLowerCase();
}

export function UploadForm({ onJobCreated, onCancel, onAnalysisStart, onAnalysisComplete, onNavigateToSettings }: UploadFormProps) {
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

  const [wizardStep, setWizardStep] = useState<WizardStep>('upload');
  const [analysisResult, setAnalysisResult] = useState<IndexAnalysisResult | null>(null);

  const [filePatterns, setFilePatterns] = useState(DEFAULT_FILE_PATTERNS);
  const [excludePatterns, setExcludePatterns] = useState('');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [maxFileSizeKb, setMaxFileSizeKb] = useState(500);
  const [ocrMode, setOcrMode] = useState<OcrMode>('disabled');
  const [ocrVisionModel, setOcrVisionModel] = useState('');
  const [exclusionsApplied, setExclusionsApplied] = useState(false);
  const [patternsExpanded, setPatternsExpanded] = useState(false);

  const resetState = useCallback(() => {
    setFile(null);
    setIndexName('');
    setDescription('');
    setIsLoading(false);
    setProgress(0);
    setStatus({ type: null, message: '' });
    setWizardStep('upload');
    setAnalysisResult(null);
    setFilePatterns(DEFAULT_FILE_PATTERNS);
    setExcludePatterns('');
    setChunkSize(1000);
    setChunkOverlap(200);
    setMaxFileSizeKb(500);
    setOcrMode('disabled');
    setOcrVisionModel('');
    setExclusionsApplied(false);
    setPatternsExpanded(false);
  }, []);

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
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      setIndexName(getIndexNameFromFile(droppedFile.name));
    }
  }, []);

  const handleFileChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setIndexName(getIndexNameFromFile(selectedFile.name));
    }
  }, []);

  const handleAnalyze = async () => {
    if (!file) {
      setStatus({ type: 'error', message: 'Please select a file first' });
      return;
    }

    setWizardStep('analyzing');
    setIsLoading(true);
    setStatus({ type: 'info', message: 'Uploading and analyzing archive...' });
    onAnalysisStart?.();

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('file_patterns', filePatterns);
      formData.append('exclude_patterns', excludePatterns);
      formData.append('chunk_size', String(chunkSize));
      formData.append('chunk_overlap', String(chunkOverlap));
      formData.append('max_file_size_kb', String(maxFileSizeKb));
      formData.append('ocr_mode', ocrMode);
      if (ocrMode === 'ollama' && ocrVisionModel) {
        formData.append('ocr_vision_model', ocrVisionModel);
      }

      const result = await api.analyzeUpload(formData);
      setAnalysisResult(result);
      setWizardStep('review');
      setStatus({ type: null, message: '' });
    } catch (err) {
      setStatus({ type: 'error', message: `Analysis failed: ${err instanceof Error ? err.message : 'Request failed'}` });
      setWizardStep('upload');
    } finally {
      setIsLoading(false);
      onAnalysisComplete?.();
    }
  };

  const applySuggestedExclusions = () => {
    if (!analysisResult?.suggested_exclusions.length) {
      return;
    }

    const currentExcludes = excludePatterns.split(',').map((s) => s.trim()).filter(Boolean);
    const newExcludes = [...new Set([...currentExcludes, ...analysisResult.suggested_exclusions])];
    setExcludePatterns(newExcludes.join(','));
    setExclusionsApplied(true);
    setPatternsExpanded(true);
  };

  const handleReanalyze = async () => {
    setExclusionsApplied(false);
    setWizardStep('analyzing');
    await handleAnalyze();
  };

  const handleStartIndexing = async () => {
    if (!file) {
      setStatus({ type: 'error', message: 'No file selected' });
      return;
    }
    if (!indexName.trim()) {
      setStatus({ type: 'error', message: 'Please enter an index name' });
      return;
    }

    setWizardStep('indexing');
    setIsLoading(true);
    setStatus({ type: 'info', message: 'Uploading and starting indexing job...' });
    setProgress(30);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', indexName);
      formData.append('description', description);
      formData.append('file_patterns', filePatterns);
      formData.append('exclude_patterns', excludePatterns);
      formData.append('chunk_size', String(chunkSize));
      formData.append('chunk_overlap', String(chunkOverlap));
      formData.append('ocr_mode', ocrMode);
      if (ocrMode === 'ollama' && ocrVisionModel) {
        formData.append('ocr_vision_model', ocrVisionModel);
      }

      const job: IndexJob = await api.uploadAndIndex(formData);
      setProgress(100);
      const successMessage = `Job started - ID: ${job.id} - Status: ${job.status}`;
      resetState();
      setStatus({ type: 'success', message: successMessage });
      onJobCreated();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Upload failed'}` });
      setWizardStep('review');
    } finally {
      setIsLoading(false);
      setTimeout(() => setProgress(0), 2000);
    }
  };

  const handleBack = () => {
    setWizardStep('upload');
    setAnalysisResult(null);
    setStatus({ type: null, message: '' });
    setExclusionsApplied(false);
  };

  const handleCancel = () => {
    resetState();
    onCancel?.();
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Upload step - file selection
  if (wizardStep === 'upload' || wizardStep === 'analyzing') {
    return (
      <div>
        {/* File drop area */}
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
              disabled={isLoading}
            />
          </div>
        </div>

        {/* Show config options after file is selected */}
        {file && (
          <>
            <p className="field-help" style={{ marginBottom: '16px' }}>
              Index name will be derived from the archive filename. Click "Analyze" to preview the index before creating.
            </p>

            <details style={{ marginBottom: '16px' }}>
              <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Options</summary>
              <IndexConfigFields
                isLoading={isLoading}
                filePatterns={filePatterns}
                setFilePatterns={setFilePatterns}
                excludePatterns={excludePatterns}
                setExcludePatterns={setExcludePatterns}
                chunkSize={chunkSize}
                setChunkSize={setChunkSize}
                chunkOverlap={chunkOverlap}
                setChunkOverlap={setChunkOverlap}
                maxFileSizeKb={maxFileSizeKb}
                setMaxFileSizeKb={setMaxFileSizeKb}
                ocrMode={ocrMode as any}
                setOcrMode={setOcrMode as any}
                ocrVisionModel={ocrVisionModel}
                setOcrVisionModel={setOcrVisionModel}
              />
            </details>

            <div className="wizard-actions">
              {onCancel && (
                <button type="button" className="btn btn-secondary" onClick={handleCancel} disabled={isLoading}>
                  Cancel
                </button>
              )}
              <button type="button" className="btn" onClick={handleAnalyze} disabled={isLoading || !file}>
                {isLoading ? 'Analyzing...' : 'Analyze Archive'}
              </button>
            </div>
          </>
        )}

        {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
      </div>
    );
  }

  // Review step - show analysis results
  if (wizardStep === 'review' && analysisResult) {
    return (
      <div>
        <h4 style={{ marginBottom: '16px' }}>
          Analysis Results for: {file?.name || 'Unknown'}
        </h4>

        {analysisResult.warnings.length > 0 && (
          <div
            style={{
              background: 'rgba(251, 191, 36, 0.1)',
              border: '1px solid rgba(251, 191, 36, 0.3)',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '16px',
            }}
          >
            <strong style={{ color: '#fbbf24' }}>Warnings:</strong>
            <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
              {analysisResult.warnings.map((warning, i) => (
                <li key={i} style={{ color: '#fbbf24', fontSize: '0.9rem' }}>
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}

        <AnalysisStats result={analysisResult} onNavigateToSettings={onNavigateToSettings} />

        {/* Suggested exclusions */}
        {analysisResult.suggested_exclusions.length > 0 && !exclusionsApplied && (
          <div
            style={{
              background: 'rgba(96, 165, 250, 0.1)',
              border: '1px solid rgba(96, 165, 250, 0.3)',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '16px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '8px' }}>
              <strong style={{ color: '#60a5fa' }}>Suggested Exclusions:</strong>
              <button
                type="button"
                className="btn btn-sm"
                onClick={applySuggestedExclusions}
                style={{ fontSize: '0.8rem', padding: '4px 12px' }}
              >
                Apply All
              </button>
            </div>
            <code style={{ fontSize: '0.85rem', color: '#94a3b8', wordBreak: 'break-word' }}>
              {analysisResult.suggested_exclusions.join(', ')}
            </code>
          </div>
        )}

        {exclusionsApplied && (
          <div
            style={{
              background: 'rgba(34, 197, 94, 0.1)',
              border: '1px solid rgba(34, 197, 94, 0.3)',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '16px',
            }}
          >
            <strong style={{ color: '#22c55e' }}>Exclusions applied!</strong>
            <span style={{ marginLeft: '8px', color: '#94a3b8', fontSize: '0.9rem' }}>
              Click "Re-analyze" to see the updated estimates.
            </span>
          </div>
        )}

        {/* File type breakdown */}
        {analysisResult.file_type_stats.length > 0 && (
          <details style={{ marginBottom: '16px' }}>
            <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>
              File Type Breakdown ({analysisResult.file_type_stats.length} types)
            </summary>
            <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
              <table style={{ width: '100%', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ textAlign: 'left', borderBottom: '1px solid #444' }}>
                    <th style={{ padding: '4px 8px' }}>Extension</th>
                    <th style={{ padding: '4px 8px' }}>Files</th>
                    <th style={{ padding: '4px 8px' }}>Size</th>
                    <th style={{ padding: '4px 8px' }}>Est. Chunks</th>
                  </tr>
                </thead>
                <tbody>
                  {analysisResult.file_type_stats.slice(0, 15).map((stat) => (
                    <tr key={stat.extension} style={{ borderBottom: '1px solid #333' }}>
                      <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{stat.extension}</td>
                      <td style={{ padding: '4px 8px' }}>{stat.file_count}</td>
                      <td style={{ padding: '4px 8px' }}>{formatBytes(stat.total_size_bytes)}</td>
                      <td style={{ padding: '4px 8px' }}>{stat.estimated_chunks}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {analysisResult.file_type_stats.length > 15 && (
                <div style={{ padding: '8px', color: '#888', fontSize: '0.85rem' }}>
                  ... and {analysisResult.file_type_stats.length - 15} more types
                </div>
              )}
            </div>
          </details>
        )}

        {/* Advanced options with current patterns */}
        <details open={patternsExpanded} style={{ marginBottom: '16px' }}>
          <summary
            style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}
            onClick={(e) => {
              e.preventDefault();
              setPatternsExpanded(!patternsExpanded);
            }}
          >
            Indexing Configuration
          </summary>

          <div className="form-group">
            <label>Index Display Name *</label>
            <input
              type="text"
              value={indexName}
              onChange={(e) => setIndexName(e.target.value)}
              placeholder="e.g., odoo-17, my-codebase"
            />
          </div>

          <DescriptionField
            value={description}
            onChange={setDescription}
            rows={2}
            compact
          />

          <IndexConfigFields
            isLoading={isLoading}
            filePatterns={filePatterns}
            setFilePatterns={setFilePatterns}
            excludePatterns={excludePatterns}
            setExcludePatterns={setExcludePatterns}
            chunkSize={chunkSize}
            setChunkSize={setChunkSize}
            chunkOverlap={chunkOverlap}
            setChunkOverlap={setChunkOverlap}
            maxFileSizeKb={maxFileSizeKb}
            setMaxFileSizeKb={setMaxFileSizeKb}
            ocrMode={ocrMode as any}
            setOcrMode={setOcrMode as any}
            ocrVisionModel={ocrVisionModel}
            setOcrVisionModel={setOcrVisionModel}
          />
        </details>

        {/* Wizard actions */}
        <div className="wizard-actions">
          <button type="button" className="btn btn-secondary" onClick={handleBack} disabled={isLoading}>
            Back
          </button>
          {exclusionsApplied && (
            <button type="button" className="btn btn-secondary" onClick={handleReanalyze} disabled={isLoading}>
              Re-analyze
            </button>
          )}
          <button type="button" className="btn" onClick={handleStartIndexing} disabled={isLoading || !indexName.trim()}>
            {isLoading ? 'Starting...' : 'Create Index'}
          </button>
        </div>

        {progress > 0 && (
          <div className="progress-bar">
            <div className="fill" style={{ width: `${progress}%` }} />
          </div>
        )}

        {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
      </div>
    );
  }

  // Fallback
  return (
    <div>
      {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
    </div>
  );
}
