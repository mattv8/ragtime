import { createContext, useContext, useState, useCallback, useRef, useEffect, type ReactNode } from 'react';
import type { AvailableModel, AvailableModelsResponse } from '@/types';

/** How long to wait before aborting a model-fetch request. */
const MODEL_FETCH_TIMEOUT_MS = 20_000;

interface AvailableModelsContextValue {
  /** Filtered models for chat (respects allowed_models). */
  models: AvailableModel[];
  /** Whether a fetch is currently in flight. */
  loading: boolean;
  /** Last fetch error message, when model discovery fails. */
  error: string | null;
  /** Readiness details for provider/model availability checks. */
  readiness: Pick<AvailableModelsResponse, 'models_loading' | 'copilot_refresh_in_progress' | 'provider_states'> | null;
  /** Metadata returned alongside models (default/current model, allowed list). */
  meta: Pick<AvailableModelsResponse, 'default_model' | 'current_model' | 'allowed_models'> | null;
  /** Trigger a (re)fetch. Safe to call multiple times; concurrent calls are coalesced. */
  refresh: () => void;
}

const AvailableModelsContext = createContext<AvailableModelsContextValue | null>(null);

export function AvailableModelsProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<AvailableModelsContextValue['readiness']>(null);
  const [meta, setMeta] = useState<AvailableModelsContextValue['meta']>(null);

  // AbortController for the in-flight request so we can cancel on unmount or re-fetch.
  const abortRef = useRef<AbortController | null>(null);
  // Track whether a fetch is already in flight to coalesce concurrent refresh() calls.
  const inflightRef = useRef(false);
  // Flag any refresh() calls that arrived while a fetch was running.
  const pendingRefreshRef = useRef(false);

  const doFetch = useCallback(async () => {
    if (inflightRef.current) {
      pendingRefreshRef.current = true;
      return;
    }
    inflightRef.current = true;
    pendingRefreshRef.current = false;

    // Cancel any previous in-flight request.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    const timeoutId = setTimeout(() => controller.abort(), MODEL_FETCH_TIMEOUT_MS);

    try {
      const response = await fetch('/indexes/chat/available-models', {
        credentials: 'include',
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`Model fetch failed: ${response.status}`);
      }
      const data: AvailableModelsResponse = await response.json();
      setModels(data.models);
      setError(null);
      setReadiness({
        models_loading: data.models_loading ?? false,
        copilot_refresh_in_progress: data.copilot_refresh_in_progress ?? false,
        provider_states: data.provider_states ?? [],
      });
      setMeta({
        default_model: data.default_model,
        current_model: data.current_model,
        allowed_models: data.allowed_models,
      });
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load available models';
        setError(errorMessage);
        console.error('Failed to load available models:', err);
      }
    } finally {
      clearTimeout(timeoutId);
      setLoading(false);
      inflightRef.current = false;

      // If another refresh was requested while we were fetching, run it now.
      if (pendingRefreshRef.current) {
        pendingRefreshRef.current = false;
        void doFetch();
      }
    }
  }, []);

  const refresh = useCallback(() => {
    void doFetch();
  }, [doFetch]);

  // Abort on unmount.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  return (
    <AvailableModelsContext.Provider value={{ models, loading, error, readiness, meta, refresh }}>
      {children}
    </AvailableModelsContext.Provider>
  );
}

export function useAvailableModels(): AvailableModelsContextValue {
  const ctx = useContext(AvailableModelsContext);
  if (!ctx) {
    throw new Error('useAvailableModels must be used within AvailableModelsProvider');
  }
  return ctx;
}
