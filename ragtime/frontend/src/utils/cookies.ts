/**
 * Shared browser-cookie utilities used across multiple components.
 */

import type { WorkspaceConversationStateSummaryItem } from '@/types';

const COOKIE_PERSISTENT_MAX_AGE_SECONDS = 60 * 60 * 24 * 365;

export const INTERRUPT_DISMISS_COOKIE_PREFIX = 'userspace_interrupt_dismissed_';

export interface InterruptChatStateSnapshot {
  rawInterrupted: boolean;
  hasLive: boolean;
}

export interface InterruptDismissTransitionResult {
  effectiveInterrupted: boolean;
  nextState: InterruptChatStateSnapshot;
  shouldClearDismiss: boolean;
}

export interface WorkspaceInterruptIndicatorState {
  hasLive: boolean;
  hasInterrupted: boolean;
}

export interface WorkspaceInterruptResolutionResult {
  workspaceId: string;
  indicator: WorkspaceInterruptIndicatorState;
  transition: InterruptDismissTransitionResult;
}

export function getInterruptDismissCookieName(userId: string, workspaceId: string): string {
  return `${INTERRUPT_DISMISS_COOKIE_PREFIX}${encodeURIComponent(userId)}_${encodeURIComponent(workspaceId)}`;
}

export function isInterruptDismissed(userId: string, workspaceId: string): boolean {
  return getCookieValue(getInterruptDismissCookieName(userId, workspaceId)) === '1';
}

export function dismissInterruptAlert(userId: string, workspaceId: string): void {
  setPersistentCookieValue(getInterruptDismissCookieName(userId, workspaceId), '1');
}

export function clearInterruptDismiss(userId: string, workspaceId: string): void {
  clearCookieValue(getInterruptDismissCookieName(userId, workspaceId));
}

export function resolveInterruptDismissTransition(
  previousState: InterruptChatStateSnapshot | undefined,
  rawInterrupted: boolean,
  hasLive: boolean,
  dismissed: boolean,
): InterruptDismissTransitionResult {
  if (!previousState) {
    return {
      effectiveInterrupted: rawInterrupted && !dismissed,
      nextState: { rawInterrupted, hasLive },
      shouldClearDismiss: false,
    };
  }

  // Re-open a dismissed interruption after we've observed a running state
  // and then returned to interrupted (active -> interrupted).
  const shouldClearDismiss = dismissed
    && rawInterrupted
    && !hasLive
    && previousState.hasLive;
  const dismissedAfterTransition = shouldClearDismiss ? false : dismissed;

  return {
    effectiveInterrupted: rawInterrupted && !dismissedAfterTransition,
    nextState: { rawInterrupted, hasLive },
    shouldClearDismiss,
  };
}

export function resolveWorkspaceInterruptStateFromSummary(
  userId: string,
  summary: WorkspaceConversationStateSummaryItem,
  previousState: InterruptChatStateSnapshot | undefined,
): WorkspaceInterruptResolutionResult {
  const workspaceId = summary.workspace_id;
  const rawInterrupted = Boolean(summary.has_interrupted_task);
  const hasLive = Boolean(summary.has_live_task);
  const dismissed = isInterruptDismissed(userId, workspaceId);
  const transition = resolveInterruptDismissTransition(previousState, rawInterrupted, hasLive, dismissed);

  return {
    workspaceId,
    indicator: {
      hasLive,
      hasInterrupted: transition.effectiveInterrupted,
    },
    transition,
  };
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
