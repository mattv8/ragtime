import type { ChangeEvent } from 'react';
import type { OcrMode, VectorStoreType } from '@/types';

interface OcrVectorStoreFieldsProps {
  isLoading: boolean;
  ocrMode: OcrMode;
  setOcrMode: (val: OcrMode) => void;
  ollamaAvailable?: boolean;
  vectorStoreType: VectorStoreType;
  setVectorStoreType: (val: VectorStoreType) => void;
  /** If true, vector store selection is disabled (locked after first index) */
  vectorStoreDisabled?: boolean;
}

/**
 * Reusable component for OCR Mode and Vector Store selection.
 * Displayed outside Advanced Options for prominent visibility.
 */
export function OcrVectorStoreFields({
  isLoading,
  ocrMode,
  setOcrMode,
  ollamaAvailable = false,
  vectorStoreType,
  setVectorStoreType,
  vectorStoreDisabled = false,
}: OcrVectorStoreFieldsProps) {
  const handleOcrModeChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setOcrMode(e.target.value as OcrMode);
  };

  const handleVectorStoreChange = (e: ChangeEvent<HTMLSelectElement>) => {
    setVectorStoreType(e.target.value as VectorStoreType);
  };

  return (
    <div className="form-row" style={{ marginBottom: '16px' }}>
      <div className="form-group" style={{ flex: 1 }}>
        <label>OCR Mode</label>
        <select
          value={ocrMode}
          onChange={handleOcrModeChange}
          disabled={isLoading}
        >
          <option value="disabled">Disabled - Skip image files</option>
          <option value="tesseract">Tesseract - Fast traditional OCR</option>
          {ollamaAvailable && (
            <option value="ollama">Ollama Vision - Semantic OCR (uses global settings)</option>
          )}
        </select>
        <p className="field-help">
          {ocrMode === 'ollama'
            ? 'Uses global OCR vision model setting for semantic text extraction.'
            : ocrMode === 'tesseract'
            ? 'Uses Tesseract for fast basic text extraction from images.'
            : 'Image files (PNG, JPG, etc.) will be skipped during indexing.'}
        </p>
      </div>

      <div className="form-group" style={{ flex: 1 }}>
        <label>Vector Store</label>
        <select
          value={vectorStoreType}
          onChange={handleVectorStoreChange}
          disabled={isLoading || vectorStoreDisabled}
        >
          <option value="faiss">FAISS (In-memory)</option>
          <option value="pgvector">pgvector (PostgreSQL)</option>
        </select>
        <p className="field-help">
          {vectorStoreType === 'pgvector'
            ? 'pgvector stores embeddings in PostgreSQL. Persistent and scalable. Recommended for larger indexes.'
            : 'FAISS stores embeddings in memory with disk persistence. Faster searches but uses more RAM.'}
          {vectorStoreDisabled && ' (Locked to match existing indexes)'}
        </p>
      </div>
    </div>
  );
}
