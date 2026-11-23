import { describe, expect, it } from 'vitest';
import type {
  SolidworksHostAutoUpdate,
  SolidworksHostAutoUpdateReason,
  SolidworksHostAutoUpdateState,
} from '../types/api';
import {
  formatAttemptTime,
  hostUpdateBadgeKind,
  hostUpdateFailureParts,
  hostUpdateFailureText,
  hostUpdateWaitingText,
} from './solidworksHostUpdate';

const DAY_MS = 24 * 60 * 60 * 1000;
const NOW = new Date(Date.UTC(2026, 10, 23, 12, 0));
const LAST_ISO = new Date(Date.UTC(2026, 10, 23, 9, 30)).toISOString();

function update(overrides: Partial<SolidworksHostAutoUpdate> = {}): SolidworksHostAutoUpdate {
  return {
    enabled: true,
    state: 'pending',
    installed_version: '0.3.0',
    target_version: '0.4.0',
    reason: null,
    last_attempt_at: null,
    next_attempt_at: null,
    message: null,
    ...overrides,
  };
}

describe('hostUpdateWaitingText', () => {
  it.each<[SolidworksHostAutoUpdateReason | null, string]>([
    ['in_use', 'The host is in use. Ragtime updates it after 5 minutes without requests.'],
    [
      'documents_open',
      'A document is open in a SolidWorks the host started. Ragtime updates the host after that instance closes.',
    ],
    [
      'logged_off',
      'No one is logged in as the host account. Ragtime updates the host once the host is running again.',
    ],
    ['not_running', 'The host is not running. Ragtime updates it once it is running.'],
    ['busy', 'The host was busy when the update started. Ragtime tries again shortly.'],
    [null, 'Ragtime updates the host when it is idle.'],
  ])('reason %s', (reason, sentence) => {
    expect(hostUpdateWaitingText(update({ reason }))).toBe(sentence);
  });
});

describe('hostUpdateFailureText', () => {
  const last = formatAttemptTime(LAST_ISO, NOW);
  const futureNext = new Date(NOW.getTime() + 2 * 60 * 60 * 1000).toISOString();
  const futureNextText = formatAttemptTime(futureNext, NOW);

  it('uses every part when all are usable', () => {
    expect(
      hostUpdateFailureText(
        update({
          state: 'failed',
          last_attempt_at: LAST_ISO,
          next_attempt_at: futureNext,
          message: 'The installer could not replace the host.',
        }),
        NOW,
      ),
    ).toBe(
      `Automatic update failed at ${last}. The installer could not replace the host. Ragtime tries again at ${futureNextText}.`,
    );
  });

  it.each<[string, Partial<SolidworksHostAutoUpdate>, string]>([
    [
      'drops the last attempt when unusable',
      { last_attempt_at: 'not a time', next_attempt_at: futureNext, message: 'Failed.' },
      `Automatic update failed. Failed. Ragtime tries again at ${futureNextText}.`,
    ],
    [
      'drops the last attempt when null',
      { last_attempt_at: null, next_attempt_at: futureNext, message: 'Failed.' },
      `Automatic update failed. Failed. Ragtime tries again at ${futureNextText}.`,
    ],
    [
      'drops the message when null',
      { last_attempt_at: LAST_ISO, next_attempt_at: futureNext, message: null },
      `Automatic update failed at ${last}. Ragtime tries again at ${futureNextText}.`,
    ],
    [
      'says soon when next is null',
      { last_attempt_at: LAST_ISO, next_attempt_at: null, message: 'Failed.' },
      `Automatic update failed at ${last}. Failed. Ragtime tries again soon.`,
    ],
    [
      'says soon when next is in the past',
      {
        last_attempt_at: LAST_ISO,
        next_attempt_at: new Date(NOW.getTime() - 60_000).toISOString(),
        message: 'Failed.',
      },
      `Automatic update failed at ${last}. Failed. Ragtime tries again soon.`,
    ],
    [
      'says soon when next equals now',
      { last_attempt_at: LAST_ISO, next_attempt_at: NOW.toISOString(), message: 'Failed.' },
      `Automatic update failed at ${last}. Failed. Ragtime tries again soon.`,
    ],
    [
      'says soon when next is unusable',
      { last_attempt_at: LAST_ISO, next_attempt_at: 'soon-ish', message: 'Failed.' },
      `Automatic update failed at ${last}. Failed. Ragtime tries again soon.`,
    ],
    [
      'adds punctuation to the message',
      { last_attempt_at: null, next_attempt_at: null, message: 'Disk <full> & "locked"' },
      'Automatic update failed. Disk <full> & "locked". Ragtime tries again soon.',
    ],
  ])('%s', (_name, fields, sentence) => {
    expect(hostUpdateFailureText(update({ state: 'failed', ...fields }), NOW)).toBe(sentence);
  });
});

describe('hostUpdateFailureParts', () => {
  it('returns usable ISO values and normalizes the message', () => {
    const next = new Date(NOW.getTime() + 1000).toISOString();
    expect(
      hostUpdateFailureParts(
        update({ last_attempt_at: LAST_ISO, next_attempt_at: next, message: '  Disk full  ' }),
        NOW,
      ),
    ).toEqual({ last: LAST_ISO, next, message: 'Disk full.' });
  });

  it.each(['Failed.', 'Failed!', 'Failed?'])('preserves terminal punctuation in %s', (message) => {
    expect(hostUpdateFailureParts(update({ message }), NOW).message).toBe(message);
  });

  it('drops invalid last, non-future next and whitespace-only messages', () => {
    expect(
      hostUpdateFailureParts(
        update({ last_attempt_at: 'invalid', next_attempt_at: NOW.toISOString(), message: '  ' }),
        NOW,
      ),
    ).toEqual({ last: null, next: null, message: null });
  });
});

describe('hostUpdateBadgeKind', () => {
  it.each<[SolidworksHostAutoUpdateState, 'pending' | 'updating' | 'failed' | null]>([
    ['current', null],
    ['newer', null],
    ['off', null],
    ['pending', 'pending'],
    ['updating', 'updating'],
    ['failed', 'failed'],
  ])('state %s', (state, kind) => {
    expect(hostUpdateBadgeKind(update({ state }))).toBe(kind);
  });

  it.each([
    ['null', null],
    ['undefined', undefined],
  ])('%s shows no badge', (_name, value) => {
    expect(hostUpdateBadgeKind(value)).toBeNull();
  });
});

describe('formatAttemptTime', () => {
  const sameDayIso = new Date(Date.UTC(2026, 10, 23, 9, 30)).toISOString();

  it('shows local time only on the same local day', () => {
    const date = new Date(sameDayIso);
    expect(formatAttemptTime(sameDayIso, new Date(date.getTime() + 60_000))).toBe(
      date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }),
    );
  });

  it('shows the date and time on another local day', () => {
    const date = new Date(sameDayIso);
    expect(formatAttemptTime(sameDayIso, new Date(date.getTime() + 3 * DAY_MS))).toBe(
      date.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
      }),
    );
  });

  it.each(['not a time', '', '2026-13-45T99:99:99Z'])('returns null for %j', (value) => {
    expect(formatAttemptTime(value, NOW)).toBeNull();
  });

  it('reads a timestamp without an offset as UTC', () => {
    const withoutOffset = '2026-11-23T09:30:00';
    expect(formatAttemptTime(withoutOffset, NOW)).not.toBeNull();
    expect(formatAttemptTime(withoutOffset, NOW)).toBe(formatAttemptTime(`${withoutOffset}Z`, NOW));
  });

  it('keeps an explicit offset', () => {
    expect(formatAttemptTime('2026-11-23T09:30:00+00:00', NOW)).toBe(
      formatAttemptTime('2026-11-23T09:30:00Z', NOW),
    );
  });
});
