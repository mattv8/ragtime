import { describe, expect, it } from 'vitest';

import { isInDatabaseCaptureWindow } from './databaseCaptureWindow';

describe('isInDatabaseCaptureWindow', () => {
  const window = { start: '2026-09-17T10:00:00Z', end: '2026-09-17T11:00:00Z' };

  it('uses an inclusive start and exclusive end', () => {
    expect(isInDatabaseCaptureWindow('2026-09-17T10:00:00Z', window)).toBe(true);
    expect(isInDatabaseCaptureWindow('2026-09-17T10:59:59Z', window)).toBe(true);
    expect(isInDatabaseCaptureWindow('2026-09-17T09:59:59Z', window)).toBe(false);
    expect(isInDatabaseCaptureWindow('2026-09-17T11:00:00Z', window)).toBe(false);
  });

  it('keeps a null-snapshot backup eligible by capture time', () => {
    expect(isInDatabaseCaptureWindow('2026-09-17T10:30:00Z', window)).toBe(true);
  });

  it('keeps the newest open-ended interval open', () => {
    expect(
      isInDatabaseCaptureWindow('2026-09-18T10:00:00Z', { start: '2026-09-17T10:00:00Z' }),
    ).toBe(true);
  });

  it('fails closed for invalid timestamp or bounds', () => {
    expect(isInDatabaseCaptureWindow('2026-09-17T10:30:00Z', { start: 'not-a-time' })).toBe(false);
    expect(
      isInDatabaseCaptureWindow('2026-09-17T10:30:00Z', {
        start: '2026-09-17T11:00:00Z',
        end: '2026-09-17T10:00:00Z',
      }),
    ).toBe(false);
    expect(isInDatabaseCaptureWindow('not-a-time', window)).toBe(false);
  });

  it('leaves header history unfiltered', () => {
    expect(isInDatabaseCaptureWindow('not-a-time', undefined)).toBe(true);
  });
});
