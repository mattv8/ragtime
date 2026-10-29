import { useCallback, useEffect, useId, useRef, useState, type ChangeEvent } from 'react';
import { api } from '@/api';
import type { OcrMode, OcrProvider, VectorStoreType, VisionModel } from '@/types';

export const OCR_PROVIDER_LABELS: Record<OcrProvider, string> = {
  ollama: 'Ollama',
  anthropic: 'Anthropic API',
  claude_code: 'Claude Code subscription',
  openai: 'OpenAI',
  openai_codex: 'OpenAI Codex subscription',
  openrouter: 'OpenRouter',
  omlx: 'oMLX',
  lmstudio: 'LM Studio',
  llama_cpp: 'llama.cpp',
};

const OCR_PROVIDERS: OcrProvider[] = [
  'ollama',
  'anthropic',
  'claude_code',
  'openai',
  'openai_codex',
  'openrouter',
  'omlx',
  'lmstudio',
  'llama_cpp',
];

const SUBSCRIPTION_OCR_PROVIDERS: ReadonlySet<OcrProvider> = new Set<OcrProvider>([
  'claude_code',
  'openai_codex',
]);

export function isOcrTargetComplete(
  mode: OcrMode,
  provider: OcrProvider | null,
  model: string,
  defaultProvider: OcrProvider | null,
): boolean {
  return mode !== 'vision' || !provider || provider === defaultProvider || Boolean(model.trim());
}

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
  /** Global default OCR provider label to show in helptext (e.g. "OpenAI") */
  defaultOcrProviderLabel?: string;
  /** Global OCR provider ID, used to determine whether an override may inherit its model. */
  defaultOcrProvider?: OcrProvider | null;
  /** Global default OCR vision model to show in helptext (e.g. "gpt-4o-mini") */
  defaultOcrVisionModelLabel?: string;
  /** Stable prefix for form control IDs when a caller renders multiple instances. */
  idPrefix?: string;
  /** Opens Settings without discarding the active index draft. */
  onOpenOcrProviderSetup?: () => void;
}

type OcrProviderConnectionState =
  | 'checking'
  | 'connected'
  | 'not-configured'
  | 'cli-missing'
  | 'check-failed';

function isSavedProviderConfigured(
  provider: OcrProvider,
  settings: Record<string, unknown>,
): boolean {
  switch (provider) {
    case 'openai':
      return Boolean(settings.openai_api_key);
    case 'anthropic':
      return Boolean(settings.anthropic_api_key);
    case 'openrouter':
      return Boolean(settings.openrouter_api_key);
    case 'ollama':
      return Boolean(
        (settings.llm_ollama_protocol && settings.llm_ollama_host && settings.llm_ollama_port) ||
        (settings.ollama_protocol && settings.ollama_host && settings.ollama_port),
      );
    case 'llama_cpp':
      return Boolean(
        settings.llm_llama_cpp_protocol &&
        settings.llm_llama_cpp_host &&
        settings.llm_llama_cpp_port,
      );
    case 'lmstudio':
      return Boolean(
        settings.llm_lmstudio_protocol && settings.llm_lmstudio_host && settings.llm_lmstudio_port,
      );
    case 'omlx':
      return Boolean(
        settings.llm_omlx_protocol && settings.llm_omlx_host && settings.llm_omlx_port,
      );
    default:
      return false;
  }
}

function useOcrProviderConnectionStatus(provider: OcrProvider | null) {
  const [state, setState] = useState<OcrProviderConnectionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const generation = useRef(0);

  useEffect(() => {
    const requestGeneration = generation.current + 1;
    generation.current = requestGeneration;
    if (!provider) {
      setState(null);
      setError(null);
      return;
    }
    setState('checking');
    setError(null);
    const update = (nextState: OcrProviderConnectionState, nextError: string | null = null) => {
      if (generation.current === requestGeneration) {
        setState(nextState);
        setError(nextError);
      }
    };
    void (async () => {
      try {
        if (provider === 'claude_code') {
          const status = await api.getClaudeCodeAuthStatus();
          update(
            !status.installed ? 'cli-missing' : status.connected ? 'connected' : 'not-configured',
            status.error ?? null,
          );
          return;
        }
        if (provider === 'openai_codex') {
          const status = await api.getOpenAICodexAuthStatus();
          update(status.connected ? 'connected' : 'not-configured');
          return;
        }
        const response = await api.getSettings();
        update(
          isSavedProviderConfigured(
            provider,
            response.settings as unknown as Record<string, unknown>,
          )
            ? 'connected'
            : 'not-configured',
        );
      } catch (statusError) {
        update(
          'check-failed',
          statusError instanceof Error
            ? statusError.message
            : 'Unable to check saved provider setup',
        );
      }
    })();
  }, [provider]);

  return { state, error };
}

function useVisionModels(provider: OcrProvider | null): {
  models: VisionModel[];
  isLoading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
} {
  const [models, setModels] = useState<VisionModel[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestGeneration = useRef(0);

  useEffect(() => {
    requestGeneration.current += 1;
    setModels([]);
    setIsLoading(false);
    setError(null);
  }, [provider]);

  const refresh = useCallback(async () => {
    if (!provider) return;
    const generation = requestGeneration.current + 1;
    requestGeneration.current = generation;
    setIsLoading(true);
    setError(null);

    try {
      // Omitting credentials intentionally uses the server's stored provider setup.
      const response = await api.getVisionModels({ provider });
      if (generation !== requestGeneration.current) return;
      if (response.success) {
        setModels(response.models);
        if (response.models.length === 0) {
          setError(`No vision-capable models found for ${OCR_PROVIDER_LABELS[provider]}.`);
        }
      } else {
        setError(response.message);
      }
    } catch (requestError) {
      if (generation === requestGeneration.current) {
        setError(
          requestError instanceof Error ? requestError.message : 'Failed to fetch vision models',
        );
      }
    } finally {
      if (generation === requestGeneration.current) setIsLoading(false);
    }
  }, [provider]);

  return { models, isLoading, error, refresh };
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
  visionModels: suppliedVisionModels,
  visionModelsLoading: suppliedVisionModelsLoading,
  visionModelsError: suppliedVisionModelsError,
  onRefreshVisionModels,
  visionOcrAvailable,
  ollamaAvailable = false,
  vectorStoreType,
  setVectorStoreType,
  vectorStoreDisabled = false,
  defaultOcrProviderLabel,
  defaultOcrProvider,
  defaultOcrVisionModelLabel,
  idPrefix = 'ocr-vector-store',
  onOpenOcrProviderSetup,
}: OcrVectorStoreFieldsProps) {
  const instanceId = useId().replace(/:/g, '');
  const fieldIdPrefix = `${idPrefix}-${instanceId}`;
  const {
    models,
    isLoading: modelsLoading,
    error: modelsError,
    refresh,
  } = useVisionModels(ocrProvider);
  const visionModels = suppliedVisionModels ?? models;
  const visionModelsLoading = suppliedVisionModelsLoading ?? modelsLoading;
  const visionModelsError = suppliedVisionModelsError ?? modelsError;
  const handleOcrModeChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const mode = e.target.value as OcrMode;
    setOcrMode(mode);
    if (mode !== 'vision') {
      setOcrProvider?.(null);
      setOcrVisionModel?.('');
    }
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
      setOcrVisionModel?.('');
    }
  };

  const isVisionMode = ocrMode === 'vision';
  const canUseVision = visionOcrAvailable ?? ollamaAvailable;
  const requiresExplicitModel = Boolean(ocrProvider && ocrProvider !== defaultOcrProvider);
  const effectiveProvider = ocrProvider ?? defaultOcrProvider ?? null;
  const connectionStatus = useOcrProviderConnectionStatus(isVisionMode ? effectiveProvider : null);
  const openSetup = () => {
    if (onOpenOcrProviderSetup) {
      onOpenOcrProviderSetup();
      return;
    }
    window.open('?view=settings&highlight=llm_provider', '_blank', 'noopener,noreferrer');
  };

  return (
    <>
      <div
        id={`${fieldIdPrefix}-configuration`}
        className="form-row"
        style={{ marginBottom: '16px' }}
      >
        <div className="form-group" style={{ flex: 1 }}>
          <label htmlFor={`${fieldIdPrefix}-mode`}>OCR Mode</label>
          <select
            id={`${fieldIdPrefix}-mode`}
            value={ocrMode}
            onChange={handleOcrModeChange}
            disabled={isLoading}
          >
            <option value="disabled">Disabled - Skip image files</option>
            <option value="tesseract">Tesseract - Fast traditional OCR</option>
            {canUseVision && <option value="vision">Vision Model - Semantic OCR</option>}
          </select>
          <p className="field-help">
            {isVisionMode
              ? 'Processes image files with a vision model. PDFs continue through the separate AnyDoc extraction path; OCR and embeddings are configured independently.'
              : ocrMode === 'tesseract'
                ? 'Uses Tesseract for fast basic text extraction from images.'
                : 'Image files (PNG, JPG, etc.) will be skipped during indexing.'}
          </p>
        </div>

        <div className="form-group" style={{ flex: 1 }}>
          <label htmlFor={`${fieldIdPrefix}-vector-store`}>Vector Store</label>
          <select
            id={`${fieldIdPrefix}-vector-store`}
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
        <div
          id={`${fieldIdPrefix}-vision-target`}
          className="form-row"
          style={{ marginBottom: '16px' }}
        >
          <div className="form-group" style={{ flex: 1 }}>
            <label htmlFor={`${fieldIdPrefix}-provider`}>Vision Provider</label>
            <select
              id={`${fieldIdPrefix}-provider`}
              value={ocrProvider || ''}
              onChange={handleProviderChange}
              disabled={isLoading || !setOcrProvider}
            >
              <option value="">Use global default</option>
              {OCR_PROVIDERS.map((provider) => (
                <option key={provider} value={provider}>
                  {OCR_PROVIDER_LABELS[provider]}
                </option>
              ))}
            </select>
            <p className="field-help">
              {ocrProvider
                ? 'Override the vision provider for this index only.'
                : defaultOcrProviderLabel
                  ? `Uses the global default (${defaultOcrProviderLabel}${defaultOcrVisionModelLabel ? ` / ${defaultOcrVisionModelLabel}` : ''}).`
                  : 'Uses the provider and model configured in global OCR settings.'}
            </p>
            {connectionStatus.state && (
              <p
                className={connectionStatus.state === 'check-failed' ? 'field-error' : 'field-help'}
                role={connectionStatus.state === 'check-failed' ? 'alert' : undefined}
                aria-live="polite"
              >
                {connectionStatus.state === 'checking' && 'Checking saved provider setup…'}
                {connectionStatus.state === 'connected' &&
                  (SUBSCRIPTION_OCR_PROVIDERS.has(effectiveProvider!)
                    ? `${OCR_PROVIDER_LABELS[effectiveProvider!]} is connected.`
                    : `Saved ${OCR_PROVIDER_LABELS[effectiveProvider!]} setup found.`)}
                {connectionStatus.state === 'not-configured' &&
                  `${OCR_PROVIDER_LABELS[effectiveProvider!]} is not configured.${connectionStatus.error ? ` ${connectionStatus.error}` : ''}`}
                {connectionStatus.state === 'cli-missing' &&
                  `Claude Code CLI is missing.${connectionStatus.error ? ` ${connectionStatus.error}` : ''}`}
                {connectionStatus.state === 'check-failed' &&
                  `Provider setup check failed: ${connectionStatus.error}`}{' '}
                {connectionStatus.state !== 'connected' && (
                  <button type="button" className="btn-link" onClick={openSetup}>
                    Open provider setup
                  </button>
                )}
              </p>
            )}
            {isVisionMode && (ocrProvider || defaultOcrProvider) && (
              <p className="field-help">
                {['openai', 'anthropic', 'openrouter'].includes(
                  ocrProvider || defaultOcrProvider || '',
                )
                  ? 'Cloud OCR transfers images to the selected API provider and may incur API charges.'
                  : SUBSCRIPTION_OCR_PROVIDERS.has(
                        (ocrProvider || defaultOcrProvider) as OcrProvider,
                      )
                    ? 'Subscription OCR transfers images through the selected provider and uses its plan quota and rate limits.'
                    : 'Self-hosted OCR sends images to the configured server, which may be remote.'}
              </p>
            )}
          </div>
          {ocrProvider && (
            <div className="form-group" style={{ flex: 1 }}>
              <label htmlFor={`${fieldIdPrefix}-model`}>Vision Model</label>
              <div style={{ display: 'flex', gap: '8px' }}>
                <select
                  id={`${fieldIdPrefix}-model`}
                  value={ocrVisionModel}
                  onChange={(e) => setOcrVisionModel?.(e.target.value)}
                  disabled={isLoading || !setOcrVisionModel}
                  aria-describedby={`${fieldIdPrefix}-model-help`}
                  aria-invalid={requiresExplicitModel && !ocrVisionModel}
                  required={requiresExplicitModel}
                  style={{ flex: 1 }}
                >
                  <option value="">
                    {requiresExplicitModel ? 'Select a model' : 'Use global default'}
                  </option>
                  {ocrVisionModel &&
                    !visionModels.some((model) => model.name === ocrVisionModel) && (
                      <option value={ocrVisionModel}>{ocrVisionModel}</option>
                    )}
                  {visionModels.map((model) => (
                    <option
                      key={`${model.provider || ocrProvider}:${model.name}`}
                      value={model.name}
                    >
                      {model.name}
                    </option>
                  ))}
                </select>
                {ocrProvider && (
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    onClick={() => void (onRefreshVisionModels ?? refresh)()}
                    disabled={isLoading || visionModelsLoading}
                  >
                    {visionModelsLoading ? 'Loading...' : 'Load Models'}
                  </button>
                )}
              </div>
              <p id={`${fieldIdPrefix}-model-help`} className="field-help" aria-live="polite">
                {visionModelsError
                  ? visionModelsError
                  : ocrVisionModel
                    ? 'Custom vision model for this index.'
                    : requiresExplicitModel
                      ? 'Select a model for this provider. Provider overrides cannot inherit a model from the global provider.'
                      : defaultOcrVisionModelLabel
                        ? `Uses the global default model (${defaultOcrVisionModelLabel}).`
                        : 'Leave blank to use the global OCR model.'}
              </p>
            </div>
          )}
        </div>
      )}
    </>
  );
}
