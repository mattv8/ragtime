export interface DatabaseCaptureWindow {
  start: string;
  end?: string;
}

/** Parse timestamps without a timezone as UTC, matching User Space snapshot timestamps. */
export function parseUtcTimestampMs(value: string): number {
  const normalized = value.trim();
  if (!normalized) return Number.NaN;

  const hasExplicitTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(normalized);
  const parsed = new Date(hasExplicitTimezone ? normalized : `${normalized}Z`);
  return parsed.getTime();
}

/**
 * A defined capture window fails closed: malformed or inverted bounds never expose backups.
 * The start is inclusive and the optional end is exclusive.
 */
export function isInDatabaseCaptureWindow(
  createdAt: string,
  window: DatabaseCaptureWindow | undefined,
): boolean {
  if (!window) return true;

  const start = parseUtcTimestampMs(window.start);
  const end = window.end === undefined ? undefined : parseUtcTimestampMs(window.end);
  const created = parseUtcTimestampMs(createdAt);
  if (!Number.isFinite(start) || !Number.isFinite(created)) return false;
  if (end !== undefined && (!Number.isFinite(end) || end <= start)) return false;

  return created >= start && (end === undefined || created < end);
}
