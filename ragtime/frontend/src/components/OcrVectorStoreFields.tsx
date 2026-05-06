import type { ChangeEvent } from 'react';
import type { OcrMode, OcrProvider, VectorStoreType, VisionModel } from '@/types';

export const OCR_PROVIDER_LABELS: Record<OcrProvider, string> = {
  ollama: 'Ollama',
  openai: 'OpenAI',
  omlx: 'oMLX',
  lmstudio: 'LM Studio',
  llama_cpp: 'llama.cpp',
};

const OCR_PROVIDERS: OcrProvider[] = ['ollama', 'openai', 'omlx', 'lmstudio', 'llama_cpp'];

interface OcrVectorStoreFieldsProps {
  isLoading: boolean;
  ocrMode: OcrMode;
  setOcrMode: (val: OcrMode) => void;
  ocrProvider?: OcrProvider | null;
  setOcrProvider?: (val: OcrProvider | null) => void;
  ocrVisionModel?: string;
  setOcrVisionModel?: (val: string) => void;
  visionModels?: VisionModel[];
  visionModelsLoading?: boolean;
  visionModelsError?: string | null;
  onRefreshVisionModels?: () => void;
  visionOcrAvailable?: boolean;
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
  ocrProvider = null,
  setOcrProvider,
  ocrVisionModel = '',
  setOcrVisionModel,
  visionModels = [],
  visionModelsLoading = false,
  visionModelsError = null,
  onRefreshVisionModels,
  visionOcrAvailable,
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

  const handleProviderChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    if (!val) {
      // Revert to global default: clear both provider and model
      setOcrProvider?.(null);
      setOcrVisionModel?.('');
    } else {
      setOcrProvider?.(val as OcrProvider);
    }
  };

  const isVisionMode = ocrMode === 'vision';
  const canUseVision = visionOcrAvailable ?? ollamaAvailable;

  return (
    <>
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
          {canUseVision && (
            <option value="vision">Vision Model - Semantic OCR</option>
          )}
        </select>
        <p className="field-help">
          {isVisionMode
            ? 'Uses a vision-capable model for semantic text extraction from images.'
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
    {isVisionMode && (
      <div className="form-row" style={{ marginBottom: '16px' }}>
        <div className="form-group" style={{ flex: 1 }}>
          <label>Vision Provider</label>
          <select
            value={ocrProvider || ''}
            onChange={handleProviderChange}
            disabled={isLoading || !setOcrProvider}
          >
            <option value="">Use global default</option>
            {OCR_PROVIDERS.map((provider) => (
              <option key={provider} value={provider}>{OCR_PROVIDER_LABELS[provider]}</option>
            ))}
          </select>
          <p className="field-help">
            {ocrProvider
              ? 'Override the vision provider for this index only.'
              : 'Uses the provider and model configured in global OCR settings.'}
          </p>
        </div>
        {ocrProvider && (
          <div className="form-group" style={{ flex: 1 }}>
            <label>Vision Model</label>
            <div style={{ display: 'flex', gap: '8px' }}>
              <select
                value={ocrVisionModel}
                onChange={(e) => setOcrVisionModel?.(e.target.value)}
                disabled={isLoading || !setOcrVisionModel}
                style={{ flex: 1 }}
              >
                <option value="">Use global default</option>
                {ocrVisionModel && !visionModels.some((model) => model.name === ocrVisionModel) && (
                  <option value={ocrVisionModel}>{ocrVisionModel}</option>
                )}
                {visionModels.map((model) => (
                  <option key={`${model.provider || ocrProvider}:${model.name}`} value={model.name}>
                    {model.name}
                  </option>
                ))}
              </select>
              {onRefreshVisionModels && (
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  onClick={onRefreshVisionModels}
                  disabled={isLoading || visionModelsLoading}
                >
                  {visionModelsLoading ? 'Loading...' : 'Load Models'}
                </button>
              )}
            </div>
            <p className="field-help">
              {visionModelsError || 'Leave blank to use the global OCR model for this provider.'}
            </p>
          </div>
        )}
      </div>
    )}
    </>
  );
}
