import { useEffect, useMemo, useRef, useState } from 'react';

import { api } from '@/api';
import type { WorkspaceConversationSearchMatch } from '@/types';

interface UseWorkspaceChatSearchArgs {
  workspaceIds: string[];
  query: string;
  enabled: boolean;
  debounceMs?: number;
}

interface UseWorkspaceChatSearchResult {
  loading: boolean;
  matchedWorkspaceIds: ReadonlySet<string>;
  snippetsByWorkspaceId: Record<string, string>;
  matchesByWorkspaceId: Record<string, WorkspaceConversationSearchMatch[]>;
}

const EMPTY_MATCHES: WorkspaceConversationSearchMatch[] = [];

export function useWorkspaceChatSearch({
  workspaceIds,
  query,
  enabled,
  debounceMs = 180,
}: UseWorkspaceChatSearchArgs): UseWorkspaceChatSearchResult {
  const normalizedQuery = query.trim();
  const workspaceIdsKey = useMemo(() => Array.from(new Set(workspaceIds)).join('\u0000'), [workspaceIds]);
  const requestIdRef = useRef(0);
  const cacheRef = useRef<Map<string, WorkspaceConversationSearchMatch[]>>(new Map());
  const [loading, setLoading] = useState(false);
  const [matches, setMatches] = useState<WorkspaceConversationSearchMatch[]>(EMPTY_MATCHES);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    if (!enabled || !normalizedQuery || !workspaceIdsKey) {
      setLoading(false);
      setMatches(EMPTY_MATCHES);
      return;
    }

    const dedupedWorkspaceIds = workspaceIdsKey.split('\u0000').filter(Boolean);
    const cacheKey = `${normalizedQuery.toLowerCase()}\u0001${workspaceIdsKey}`;
    const cached = cacheRef.current.get(cacheKey);
    if (cached) {
      setLoading(false);
      setMatches(cached);
      return;
    }

    setMatches(EMPTY_MATCHES);
    const timer = window.setTimeout(() => {
      setLoading(true);
      void api.searchWorkspaceConversations(dedupedWorkspaceIds, normalizedQuery)
        .then((response) => {
          if (requestIdRef.current !== requestId) return;
          cacheRef.current.set(cacheKey, response.matches);
          setMatches(response.matches);
        })
        .catch(() => {
          if (requestIdRef.current !== requestId) return;
          setMatches(EMPTY_MATCHES);
        })
        .finally(() => {
          if (requestIdRef.current === requestId) {
            setLoading(false);
          }
        });
    }, debounceMs);

    return () => {
      window.clearTimeout(timer);
    };
  }, [debounceMs, enabled, normalizedQuery, workspaceIdsKey]);

  return useMemo(() => {
    const matchedWorkspaceIds = new Set<string>();
    const snippetsByWorkspaceId: Record<string, string> = {};
    const matchesByWorkspaceId: Record<string, WorkspaceConversationSearchMatch[]> = {};

    for (const match of matches) {
      matchedWorkspaceIds.add(match.workspace_id);
      matchesByWorkspaceId[match.workspace_id] = [...(matchesByWorkspaceId[match.workspace_id] ?? []), match];
      if (!snippetsByWorkspaceId[match.workspace_id] && match.snippet) {
        snippetsByWorkspaceId[match.workspace_id] = match.snippet;
      }
    }

    return {
      loading,
      matchedWorkspaceIds,
      snippetsByWorkspaceId,
      matchesByWorkspaceId,
    };
  }, [loading, matches]);
}