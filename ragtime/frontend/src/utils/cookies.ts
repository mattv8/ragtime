/**
 * Shared browser-cookie utilities used across multiple components.
 */

const COOKIE_PERSISTENT_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;

export const INTERRUPT_DISMISS_COOKIE_PREFIX = 'userspace_interrupt_dismissed_';

export function getInterruptDismissCookieName(userId: string, workspaceId: string): string {
  return `${INTERRUPT_DISMISS_COOKIE_PREFIX}${encodeURIComponent(userId)}_${encodeURIComponent(workspaceId)}`;
}

export function getCookieValue(name: string): string | null {
  if (typeof document === 'undefined') return null;
  const entries = document.cookie ? document.cookie.split('; ') : [];
  for (const entry of entries) {
    const separatorIndex = entry.indexOf('=');
    if (separatorIndex < 0) continue;
    const key = entry.slice(0, separatorIndex);
    if (key !== name) continue;
    const value = entry.slice(separatorIndex + 1);
    try {
      return decodeURIComponent(value);
    } catch {
      return value;
    }
  }
  return null;
}

export function setSessionCookieValue(name: string, value: string): void {
  if (typeof document === 'undefined') return;
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; samesite=lax`;
}

export function setPersistentCookieValue(name: string, value: string): void {
  if (typeof document === 'undefined') return;
  document.cookie = `${name}=${encodeURIComponent(value)}; path=/; max-age=${COOKIE_PERSISTENT_MAX_AGE_SECONDS}; samesite=lax`;
}

export function clearCookieValue(name: string): void {
  if (typeof document === 'undefined') return;
  document.cookie = `${name}=; path=/; max-age=0; samesite=lax`;
}

export function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
