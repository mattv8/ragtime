import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { PdmIndexStatusResponse, ToolConfig } from '../types/api';

export interface UsePdmToolsResult {
  tools: ToolConfig[];
  loading: boolean;
  loadError: string | null;
  indexStatus: PdmIndexStatusResponse | null;
  statusError: string | null;
  refresh: () => Promise<void>;
  refreshStatus: () => Promise<void>;
}

const message = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function usePdmTools(linkedToolId: string | null): UsePdmToolsResult {
  const [tools, setTools] = useState<ToolConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [indexStatus, setIndexStatus] = useState<PdmIndexStatusResponse | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [hidden, setHidden] = useState(document.hidden);
  const [statusSettled, setStatusSettled] = useState(0);
  const listGeneration = useRef(0);
  const statusGeneration = useRef(0);
  const inFlight = useRef<{ id: string; promise: Promise<void> } | null>(null);

  const refreshStatus = useCallback((): Promise<void> => {
    if (linkedToolId && inFlight.current?.id === linkedToolId) return inFlight.current.promise;
    const generation = ++statusGeneration.current;
    if (!linkedToolId) {
      setIndexStatus(null);
      setStatusError(null);
      return Promise.resolve();
    }
    const promise = (async () => {
      try {
        const next = await api.getPdmIndexStatus(linkedToolId);
        if (generation === statusGeneration.current) {
          setIndexStatus(next);
          setStatusError(null);
        }
      } catch (error) {
        if (generation === statusGeneration.current) setStatusError(message(error));
      } finally {
        if (generation === statusGeneration.current) {
          inFlight.current = null;
          setStatusSettled((previous) => previous + 1);
        }
      }
    })();
    inFlight.current = { id: linkedToolId, promise };
    return promise;
  }, [linkedToolId]);

  const refreshList = useCallback(async () => {
    const generation = ++listGeneration.current;
    setLoading(true);
    try {
      const next = await api.listToolConfigs();
      if (generation === listGeneration.current) {
        setTools(next.filter((tool) => tool.tool_type === 'solidworks_pdm'));
        setLoadError(null);
      }
    } catch (error) {
      if (generation === listGeneration.current) setLoadError(message(error));
    } finally {
      if (generation === listGeneration.current) setLoading(false);
    }
  }, []);

  const refresh = useCallback(async () => {
    await Promise.all([refreshList(), refreshStatus()]);
  }, [refreshList, refreshStatus]);

  useEffect(() => {
    void refreshList();
    return () => {
      listGeneration.current += 1;
    };
  }, [refreshList]);
  useEffect(() => {
    setIndexStatus(null);
    setStatusError(null);
    void refreshStatus();
    return () => {
      statusGeneration.current += 1;
      inFlight.current = null;
    };
  }, [refreshStatus]);
  useEffect(() => {
    const onVisibilityChange = () => setHidden(document.hidden);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => document.removeEventListener('visibilitychange', onVisibilityChange);
  }, []);
  useEffect(() => {
    const job = indexStatus?.current_job?.status;
    if (hidden || inFlight.current || (job !== 'pending' && job !== 'indexing')) return;
    const timer = window.setTimeout(() => void refreshStatus(), 3000);
    return () => window.clearTimeout(timer);
  }, [hidden, indexStatus, refreshStatus, statusSettled]);

  return { tools, loading, loadError, indexStatus, statusError, refresh, refreshStatus };
}
