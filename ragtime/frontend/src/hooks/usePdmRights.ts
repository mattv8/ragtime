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
  const [result, setResult] = useState<{
    key: string;
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
    controller.current = abort;
    setResult({ key: requestKey, response: null, error: null, loading: true });
    void solidworksApi
      .checkPdmRights(
        options.toolId,
        { pdm_tool_id: options.pdmToolId, folders: options.allowChanges ? options.folders : [] },
        abort.signal,
      )
      .then((response) => {
        if (!abort.signal.aborted && current.current.active && current.current.key === requestKey) {
          setResult({
            key: requestKey,
            response,
            error: response.error?.message ?? null,
            loading: false,
          });
        }
      })
      .catch((error: unknown) => {
        if (abort.signal.aborted || (error instanceof Error && error.name === 'AbortError')) return;
        if (current.current.active && current.current.key === requestKey) {
          setResult({
            key: requestKey,
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

  const fresh = active && result?.key === key ? result : null;
  return {
    state: resolveRightsState({ response: fresh?.response ?? null, allowChanges, folders }),
    loading: active && (fresh?.loading ?? true),
    error: fresh?.error ?? null,
    recheck,
  };
}
