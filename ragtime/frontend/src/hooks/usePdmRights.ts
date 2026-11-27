import { useCallback, useEffect, useRef, useState } from 'react';
import { solidworksApi, type PdmRightsResponse } from '../api/solidworks';
import { vaultFolderKey } from '../utils/solidworksPdmLink';
import { resolveRightsState, type RightsState } from '../utils/solidworksPdmRights';

const DEBOUNCE_MS = 400;

interface UsePdmRightsInput {
  toolId: string | null;
  pdmToolId: string | null;
  folders: string[];
  allowChanges: boolean;
  enabled: boolean;
}

interface UsePdmRightsResult {
  state: RightsState;
  loading: boolean;
  error: string | null;
  recheck: () => void;
}

export function usePdmRights(input: UsePdmRightsInput): UsePdmRightsResult {
  const { toolId, pdmToolId, folders, allowChanges, enabled } = input;
  const active = Boolean(enabled && toolId && pdmToolId);
  const key = JSON.stringify([toolId, pdmToolId, allowChanges, folders.map(vaultFolderKey).sort()]);
  const current = useRef({ input, key, active });
  current.current = { input, key, active };
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const controller = useRef<AbortController | null>(null);
  const identity = active ? JSON.stringify([toolId, pdmToolId]) : null;
  const carried = useRef<{
    identity: string | null;
    response: PdmRightsResponse | null;
    generation: number;
  }>({
    identity,
    response: null,
    generation: 0,
  });
  if (carried.current.identity !== identity) {
    carried.current = { identity, response: null, generation: carried.current.generation + 1 };
  }
  const [result, setResult] = useState<{
    key: string;
    generation: number;
    response: PdmRightsResponse | null;
    error: string | null;
    loading: boolean;
  } | null>(null);

  const recheck = useCallback(() => {
    if (timer.current !== null) clearTimeout(timer.current);
    timer.current = null;
    controller.current?.abort();
    const { input: options, key: requestKey, active: canRequest } = current.current;
    if (!canRequest || !options.toolId || !options.pdmToolId) return;
    const abort = new AbortController();
    const generation = carried.current.generation;
    controller.current = abort;
    setResult({ key: requestKey, generation, response: null, error: null, loading: true });
    void solidworksApi
      .checkPdmRights(
        options.toolId,
        { pdm_tool_id: options.pdmToolId, folders: options.allowChanges ? options.folders : [] },
        abort.signal,
      )
      .then((response) => {
        if (
          !abort.signal.aborted &&
          current.current.active &&
          current.current.key === requestKey &&
          carried.current.generation === generation
        ) {
          if (!response.error) carried.current.response = response;
          setResult({
            key: requestKey,
            generation,
            response,
            error: response.error?.message ?? null,
            loading: false,
          });
        }
      })
      .catch((error: unknown) => {
        if (abort.signal.aborted || (error instanceof Error && error.name === 'AbortError')) return;
        if (
          current.current.active &&
          current.current.key === requestKey &&
          carried.current.generation === generation
        ) {
          setResult({
            key: requestKey,
            generation,
            response: null,
            error: error instanceof Error ? error.message : String(error),
            loading: false,
          });
        }
      });
  }, []);

  useEffect(() => {
    if (!active) {
      setResult(null);
      return;
    }
    timer.current = setTimeout(recheck, DEBOUNCE_MS);
    return () => {
      if (timer.current !== null) clearTimeout(timer.current);
      timer.current = null;
      controller.current?.abort();
      controller.current = null;
    };
  }, [active, key, recheck]);

  const fresh =
    active && result?.key === key && result.generation === carried.current.generation
      ? result
      : null;
  return {
    state: resolveRightsState({
      response:
        fresh?.response ?? (active && (!fresh || fresh.loading) ? carried.current.response : null),
      allowChanges,
      folders,
    }),
    loading: active && (fresh?.loading ?? true),
    error: fresh?.error ?? null,
    recheck,
  };
}
