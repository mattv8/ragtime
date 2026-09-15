export type ConversationCursor = {
  cursorUpdatedAt: string;
  cursorId: string;
};

export function cursorFromLastRow(
  row: { id: string; updated_at: string } | undefined,
): ConversationCursor | null {
  if (!row) return null;
  return {
    cursorUpdatedAt: row.updated_at,
    cursorId: row.id,
  };
}

export function hasMorePages(rows: readonly unknown[], requestedLimit: number): boolean {
  return rows.length === requestedLimit;
}

export function archiveCutoffIso(daysAgo: number): string | null {
  if (!Number.isFinite(daysAgo) || daysAgo <= 0) return null;
  return new Date(Date.now() - daysAgo * 24 * 60 * 60 * 1000).toISOString();
}

export function isWithinWindowIso(updatedAtIso: string, cutoffIso: string | null): boolean {
  if (!cutoffIso) return true;
  const updatedAtMs = Date.parse(updatedAtIso);
  const cutoffMs = Date.parse(cutoffIso);
  if (!Number.isFinite(updatedAtMs) || !Number.isFinite(cutoffMs)) return true;
  return updatedAtMs >= cutoffMs;
}

export function mergeSummaryPages<T extends { id: string }>(
  current: T[],
  incoming: T[],
  opts?: { deletedIds?: ReadonlySet<string> },
): T[] {
  const deletedIds = opts?.deletedIds;
  const seen = new Set<string>();
  const merged: T[] = [];

  for (const row of [...current, ...incoming]) {
    if (deletedIds?.has(row.id) || seen.has(row.id)) continue;
    seen.add(row.id);
    merged.push(row);
  }

  return merged;
}

export function orderInitialCandidates(o: {
  initialConversationId?: string | null;
  currentConversationId?: string | null;
  topSummaryId?: string | null;
}): string[] {
  const seen = new Set<string>();
  const candidates: string[] = [];

  for (const id of [o.initialConversationId, o.currentConversationId, o.topSummaryId]) {
    const normalized = id?.trim();
    if (!normalized || seen.has(normalized)) continue;
    seen.add(normalized);
    candidates.push(normalized);
  }

  return candidates;
}

export function isEligibleStandaloneConversation(
  c: {
    workspace_id?: string | null;
    workspaceId?: string | null;
    parent_conversation_id?: string | null;
    updated_at?: string;
  },
  cutoffIso: string | null,
): boolean {
  if (c.workspace_id || c.workspaceId || c.parent_conversation_id) return false;
  return isWithinWindowIso(c.updated_at || '', cutoffIso);
}

export function shouldHydrateHistoryForSearch(query: string): boolean {
  return query.trim().length > 0;
}
