import { describe, expect, it, vi } from 'vitest';

import {
  archiveCutoffIso,
  cursorFromLastRow,
  hasMorePages,
  isEligibleStandaloneConversation,
  isWithinWindowIso,
  mergeSummaryPages,
  orderInitialCandidates,
  shouldHydrateHistoryForSearch,
} from './conversationLoading';

describe('conversation loading helpers', () => {
  it('returns the final row cursor without changing timestamp precision', () => {
    expect(
      cursorFromLastRow({ id: 'last', updated_at: '2026-01-02T03:04:05.123456+00:00' }),
    ).toEqual({
      cursorUpdatedAt: '2026-01-02T03:04:05.123456+00:00',
      cursorId: 'last',
    });
    expect(cursorFromLastRow(undefined)).toBeNull();
  });

  it('recognizes exactly full pages', () => {
    expect(hasMorePages([1, 2], 2)).toBe(true);
    expect(hasMorePages([1], 2)).toBe(false);
  });

  it('makes a cutoff only for positive finite ages', () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2026-01-31T12:00:00.000Z'));

    expect(archiveCutoffIso(2)).toBe('2026-01-29T12:00:00.000Z');
    expect(archiveCutoffIso(0)).toBeNull();
    expect(archiveCutoffIso(Number.NaN)).toBeNull();

    vi.useRealTimers();
  });

  it('includes equality at the cutoff and tolerates invalid dates', () => {
    const cutoff = '2026-01-30T00:00:00.000Z';

    expect(isWithinWindowIso(cutoff, cutoff)).toBe(true);
    expect(isWithinWindowIso('2026-01-29T23:59:59.999Z', cutoff)).toBe(false);
    expect(isWithinWindowIso('not-a-date', cutoff)).toBe(true);
    expect(isWithinWindowIso('2026-01-01T00:00:00.000Z', 'not-a-date')).toBe(true);
    expect(isWithinWindowIso('2026-01-01T00:00:00.000Z', null)).toBe(true);
  });

  it('merges pages in stable order with existing records winning and tombstones removed', () => {
    expect(
      mergeSummaryPages(
        [
          { id: 'a', title: 'existing' },
          { id: 'a', title: 'duplicate existing' },
          { id: 'deleted', title: 'old' },
        ],
        [
          { id: 'a', title: 'incoming replacement' },
          { id: 'b', title: 'new' },
          { id: 'b', title: 'duplicate incoming' },
          { id: 'deleted', title: 'deleted incoming' },
        ],
        { deletedIds: new Set(['deleted']) },
      ),
    ).toEqual([
      { id: 'a', title: 'existing' },
      { id: 'b', title: 'new' },
    ]);
  });

  it('only accepts recent top-level conversations outside either workspace alias', () => {
    const cutoff = '2026-01-30T00:00:00.000Z';
    expect(
      isEligibleStandaloneConversation(
        { updated_at: cutoff, parent_conversation_id: null },
        cutoff,
      ),
    ).toBe(true);
    expect(
      isEligibleStandaloneConversation({ updated_at: cutoff, workspace_id: 'ws' }, cutoff),
    ).toBe(false);
    expect(
      isEligibleStandaloneConversation({ updated_at: cutoff, workspaceId: 'ws' }, cutoff),
    ).toBe(false);
    expect(
      isEligibleStandaloneConversation(
        { updated_at: cutoff, parent_conversation_id: 'parent' },
        cutoff,
      ),
    ).toBe(false);
    expect(isEligibleStandaloneConversation({ updated_at: 'invalid' }, cutoff)).toBe(true);
  });

  it('orders trimmed candidate IDs by priority without duplicates', () => {
    expect(
      orderInitialCandidates({
        initialConversationId: ' initial ',
        currentConversationId: 'initial',
        topSummaryId: ' summary ',
      }),
    ).toEqual(['initial', 'summary']);
  });

  it('hydrates history only for nonblank search queries', () => {
    expect(shouldHydrateHistoryForSearch(' query ')).toBe(true);
    expect(shouldHydrateHistoryForSearch(' \n\t ')).toBe(false);
  });
});
