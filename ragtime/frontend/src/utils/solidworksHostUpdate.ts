import type {
  SolidworksHostAutoUpdate,
  SolidworksHostAutoUpdateReason,
  SolidworksHostAutoUpdateState,
} from '../types/api';

export type HostUpdateBadgeKind = 'pending' | 'updating' | 'failed';

const WAITING_TEXT_BY_REASON: Record<SolidworksHostAutoUpdateReason, string> = {
  in_use: 'The agent is in use. Ragtime updates it after 5 minutes without requests.',
  documents_open:
    'A document is open in a SolidWorks instance the agent started. Ragtime updates the agent after that instance closes.',
  logged_off:
    'No one is logged in to the Windows account the agent runs as. Ragtime updates the agent once it is running again.',
  not_running: 'The agent is not running. Ragtime updates it once it is running.',
  busy: 'The agent was busy when the update started. Ragtime tries again shortly.',
};
const WAITING_TEXT_WITHOUT_REASON = 'Ragtime updates the agent when it is idle.';

const BADGE_KIND_BY_STATE: Partial<Record<SolidworksHostAutoUpdateState, HostUpdateBadgeKind>> = {
  pending: 'pending',
  updating: 'updating',
  failed: 'failed',
};

/** Parses an API timestamp; a value without an offset is UTC, as in `formatChatTimestamp`. */
function parseIsoTime(iso: string): Date | null {
  const normalized = iso.trim();
  const hasExplicitTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(normalized);
  const date = new Date(hasExplicitTimezone ? normalized : `${normalized}Z`);
  return Number.isNaN(date.getTime()) ? null : date;
}

/** Absolute local time: time only on the same local day, otherwise date and time. Null when unusable. */
export function formatAttemptTime(iso: string, now = new Date()): string | null {
  const date = parseIsoTime(iso);
  if (!date) return null;
  if (date.toDateString() === now.toDateString()) {
    return date.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' });
  }
  return date.toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

export function hostUpdateWaitingText(update: SolidworksHostAutoUpdate): string {
  return update.reason ? WAITING_TEXT_BY_REASON[update.reason] : WAITING_TEXT_WITHOUT_REASON;
}

export const HOST_UPDATE_OFF_TEXT =
  'Automatic updates are off for this tool. Choose Update agent to update now.';

export function hostUpdateUpdatingText(update: SolidworksHostAutoUpdate): string {
  const version = update.target_version ? ` to version ${update.target_version}` : '';
  return `Updating the agent${version}… Requests wait until it finishes, usually a minute or two.`;
}

export function hostUpdateFailureParts(
  update: SolidworksHostAutoUpdate,
  now = new Date(),
): { last: string | null; next: string | null; message: string | null } {
  const last =
    update.last_attempt_at && parseIsoTime(update.last_attempt_at) ? update.last_attempt_at : null;
  const nextTime = update.next_attempt_at ? parseIsoTime(update.next_attempt_at) : null;
  const next = nextTime && nextTime.getTime() > now.getTime() ? update.next_attempt_at : null;
  const trimmed = update.message?.trim();
  const message = trimmed ? (/[.!?]$/.test(trimmed) ? trimmed : `${trimmed}.`) : null;
  return { last, next, message };
}

export function hostUpdateFailureText(update: SolidworksHostAutoUpdate, now = new Date()): string {
  const { last, next, message } = hostUpdateFailureParts(update, now);
  const lastClause = last ? ` at ${formatAttemptTime(last, now)}` : '';
  const messageClause = message ? ` ${message}` : '';
  const retryClause = next
    ? `Ragtime tries again at ${formatAttemptTime(next, now)}.`
    : 'Ragtime tries again soon.';
  return `Automatic update failed${lastClause}.${messageClause} ${retryClause}`;
}

export function hostUpdateBadgeKind(
  update: SolidworksHostAutoUpdate | null | undefined,
): HostUpdateBadgeKind | null {
  if (!update) return null;
  return BADGE_KIND_BY_STATE[update.state] ?? null;
}
