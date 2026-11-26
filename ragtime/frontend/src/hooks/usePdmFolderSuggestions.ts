import { useCallback, useEffect, useRef, useState } from 'react';
import type { PdmFolderChild } from '../api/solidworks';
import { listAllPdmFolderChildren } from '../utils/pdmFolderListing';

const DEBOUNCE_MS = 150;
const CACHE_TTL_MS = 60_000;

interface Input {
  toolId: string | null;
  pdmToolId: string | null;
  parent: string | null;
  enabled: boolean;
}

interface Entry {
  items: PdmFolderChild[];
  total: number;
  savedAt: number;
}

interface Result {
  status: 'idle' | 'loading' | 'ready' | 'error';
  items: PdmFolderChild[];
  total: number;
  error: { message: string; code: string } | null;
  retry: () => void;
}

export function usePdmFolderSuggestions({ toolId, pdmToolId, parent, enabled }: Input): Result {
  const identity = JSON.stringify([toolId, pdmToolId]);
  const cache = useRef<{ identity: string; entries: Map<string, Entry> }>({
    identity,
    entries: new Map(),
  });
  if (cache.current.identity !== identity) {
    cache.current = { identity, entries: new Map() };
  }
  const key = parent?.toLowerCase() ?? null;
  const active = Boolean(enabled && toolId && pdmToolId && parent !== null);
  const cached = active && key !== null ? cache.current.entries.get(key) : undefined;
  const requestScope = JSON.stringify([identity, key, active]);
  const scope = useRef({ key: requestScope, generation: 0 });
  if (scope.current.key !== requestScope) {
    scope.current = { key: requestScope, generation: scope.current.generation + 1 };
  }
  const generation = scope.current.generation;
  const [result, setResult] = useState<{
    generation: number;
    status: 'loading' | 'ready' | 'error';
    entry?: Entry;
    error?: { message: string; code: string };
  } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const lastAttempt = useRef(0);
  const retry = useCallback(() => setAttempt((value) => value + 1), []);

  useEffect(() => {
    if (!active || !toolId || !pdmToolId || parent === null || key === null) return;
    const entry = cache.current.entries.get(key);
    const force = lastAttempt.current !== attempt;
    lastAttempt.current = attempt;
    if (entry && Date.now() - entry.savedAt < CACHE_TTL_MS && !force) return;
    const controller = new AbortController();
    const timer = setTimeout(() => {
      if (!entry) setResult({ generation, status: 'loading' });
      void listAllPdmFolderChildren(toolId, pdmToolId, parent, controller.signal)
        .then((response) => {
          if (controller.signal.aborted) return;
          const next = { ...response, savedAt: Date.now() };
          cache.current.entries.set(key, next);
          setResult({ generation, status: 'ready', entry: next });
        })
        .catch((error: unknown) => {
          if (controller.signal.aborted) return;
          const message = error instanceof Error ? error.message : String(error);
          const code =
            typeof error === 'object' &&
            error !== null &&
            'code' in error &&
            typeof error.code === 'string'
              ? error.code
              : '';
          setResult({ generation, status: 'error', error: { message, code } });
        });
    }, DEBOUNCE_MS);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [active, toolId, pdmToolId, parent, key, generation, attempt]);

  const current = active && result?.generation === generation ? result : null;
  const entry = current?.entry ?? cached;
  return {
    status: !active ? 'idle' : current?.status === 'error' ? 'error' : entry ? 'ready' : 'loading',
    items: entry?.items ?? [],
    total: entry?.total ?? 0,
    error: current?.error ?? null,
    retry,
  };
}
