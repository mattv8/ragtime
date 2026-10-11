import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  clearDenialNoticeSeen,
  getDenialNoticeSeenId,
  markDenialNoticeSeen,
} from './denialNotices';

const viewerId = 'viewer-a';
const conversationId = 'conversation-a';
let values = new Map<string, string>();

beforeEach(() => {
  values = new Map();
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
  });
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe('denial notice seen state', () => {
  it('keeps opaque per-viewer notice IDs across reload-like reads', () => {
    markDenialNoticeSeen(viewerId, conversationId, 'notice-1');

    expect(getDenialNoticeSeenId(viewerId, conversationId)).toBe('notice-1');
    expect(getDenialNoticeSeenId('viewer-b', conversationId)).toBeNull();
  });

  it('removes a seen marker when the summary no longer has a notice', () => {
    markDenialNoticeSeen(viewerId, conversationId, 'notice-1');
    clearDenialNoticeSeen(viewerId, conversationId);

    expect(getDenialNoticeSeenId(viewerId, conversationId)).toBeNull();
  });

  it('evicts the oldest marker when more than 200 conversations are recorded', () => {
    for (let index = 0; index <= 200; index += 1) {
      markDenialNoticeSeen(viewerId, `conversation-${index}`, `notice-${index}`);
    }

    expect(getDenialNoticeSeenId(viewerId, 'conversation-0')).toBeNull();
    expect(getDenialNoticeSeenId(viewerId, 'conversation-200')).toBe('notice-200');
  });
});
