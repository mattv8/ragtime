import { createContext, useContext, useState, useCallback, useRef, useEffect, type ReactNode } from 'react';
import type { AvailableModel, AvailableModelsResponse } from '@/types';

/** How long to wait before aborting a model-fetch request. */
const MODEL_FETCH_TIMEOUT_MS = 20_000;
/** Poll cadence while backend reports model discovery/refresh still in progress. */
const MODEL_REFRESH_POLL_MS = 1_500;
/** Minimum gap between manual refresh requests to avoid back-to-back duplicates. */
const MODEL_MANUAL_REFRESH_COOLDOWN_MS = 1_000;

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
  /**
   * Await a settled available-models cycle and return the latest state snapshot.
   * This allows callers to serialize work after model discovery completes.
   */
  awaitReady: () => Promise<{
    models: AvailableModel[];
    error: string | null;
    readiness: AvailableModelsContextValue['readiness'];
    meta: AvailableModelsContextValue['meta'];
  }>;
}

const AvailableModelsContext = createContext<AvailableModelsContextValue | null>(null);

export function AvailableModelsProvider({ children }: { children: ReactNode }) {
  const [models, setModels] = useState<AvailableModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<AvailableModelsContextValue['readiness']>(null);
  const [meta, setMeta] = useState<AvailableModelsContextValue['meta']>(null);

  // AbortController for the in-flight request so we can cancel on unmount or re-fetch.
  const abortRef = useRef<AbortController | null>(null);
  // Track whether a fetch is already in flight to coalesce concurrent refresh() calls.
  const inflightRef = useRef(false);
  // Flag any refresh() calls that arrived while a fetch was running.
  const pendingRefreshRef = useRef(false);
  // Timer for follow-up polls while backend refresh/model discovery is in progress.
  const pollTimeoutRef = useRef<number | null>(null);
  // Last successful fetch timestamp (used to coalesce rapid manual refresh calls).
  const lastSuccessfulFetchAtRef = useRef<number>(0);
  const waitersRef = useRef<
    Array<
      (value: {
        models: AvailableModel[];
        error: string | null;
        readiness: AvailableModelsContextValue['readiness'];
        meta: AvailableModelsContextValue['meta'];
      }) => void
    >
  >([]);

  const isSettled = useCallback(() => {
    return !inflightRef.current
      && !readiness?.models_loading
      && !readiness?.copilot_refresh_in_progress;
  }, [readiness?.copilot_refresh_in_progress, readiness?.models_loading]);

  const resolveWaiters = useCallback(
    (value: {
      models: AvailableModel[];
      error: string | null;
      readiness: AvailableModelsContextValue['readiness'];
      meta: AvailableModelsContextValue['meta'];
    }) => {
      if (!waitersRef.current.length) {
        return;
      }
      const pending = [...waitersRef.current];
      waitersRef.current = [];
      for (const resolve of pending) {
        resolve(value);
      }
    },
    [],
  );

  const clearPollTimer = useCallback(() => {
    if (pollTimeoutRef.current !== null) {
      window.clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
  }, []);

  const doFetch = useCallback(async (reason: 'manual' | 'poll' = 'manual') => {
    if (inflightRef.current) {
      pendingRefreshRef.current = true;
      return;
    }

    // Coalesce rapid manual refresh calls (e.g. StrictMode mount + immediate UI-triggered refresh).
    if (
      reason === 'manual'
      && models.length > 0
      && Date.now() - lastSuccessfulFetchAtRef.current < MODEL_MANUAL_REFRESH_COOLDOWN_MS
    ) {
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
        cache: 'no-store',
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
      lastSuccessfulFetchAtRef.current = Date.now();
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
        void doFetch('manual');
      }
    }
  }, [models.length]);

  const refresh = useCallback(() => {
    void doFetch('manual');
  }, [doFetch]);

  const awaitReady = useCallback(() => {
    // Fast path: if already settled, resolve immediately.
    if (isSettled()) {
      return Promise.resolve({
        models,
        error,
        readiness,
        meta,
      });
    }

    // Otherwise resolve as soon as the in-flight cycle settles.
    return new Promise<{
      models: AvailableModel[];
      error: string | null;
      readiness: AvailableModelsContextValue['readiness'];
      meta: AvailableModelsContextValue['meta'];
    }>((resolve) => {
      waitersRef.current.push(resolve);
    });
  }, [error, isSettled, meta, models, readiness]);

  useEffect(() => {
    if (!isSettled()) {
      return;
    }
    resolveWaiters({ models, error, readiness, meta });
  }, [error, isSettled, meta, models, readiness, resolveWaiters]);

  useEffect(() => {
    clearPollTimer();
    const shouldPoll = Boolean(
      readiness?.models_loading || readiness?.copilot_refresh_in_progress,
    );
    if (!shouldPoll) {
      return;
    }

    pollTimeoutRef.current = window.setTimeout(() => {
      void doFetch('poll');
    }, MODEL_REFRESH_POLL_MS);

    return clearPollTimer;
  }, [clearPollTimer, doFetch, readiness?.copilot_refresh_in_progress, readiness?.models_loading]);

  // Abort on unmount.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      clearPollTimer();
    };
  }, [clearPollTimer]);

  return (
    <AvailableModelsContext.Provider value={{ models, loading, error, readiness, meta, refresh, awaitReady }}>
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
