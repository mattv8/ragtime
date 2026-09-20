import type { PublicErrorDetail } from '@/types';

function asRecord(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

/** Recognize public refusals without treating validation details as refusals. */
export function getPublicErrorDetail(value: unknown): PublicErrorDetail | null {
  const record = asRecord(value);
  if (!record) return null;

  const code = nonEmptyString(record.code);
  const reason = nonEmptyString(record.reason);
  const nextStep = nonEmptyString(record.next_step);
  const requestId = nonEmptyString(record.request_id);
  const reasonCode = nonEmptyString(record.reason_code);
  const message = nonEmptyString(record.message);
  if (!code || (!message && !reason && !nextStep && !requestId && !reasonCode)) return null;

  return {
    code,
    message: message ?? '',
    reason: reason ?? '',
    next_step: nextStep ?? '',
    request_id: requestId ?? '',
    ...(reasonCode ? { reason_code: reasonCode } : {}),
  };
}

/** Backend messages already combine reason and next step, preventing duplicates. */
export function formatPublicErrorDetail(value: unknown, fallback: string): string {
  const detail = getPublicErrorDetail(value);
  if (!detail) return fallback;
  if (detail.message) return detail.message;
  return [detail.reason, detail.next_step].filter(Boolean).join('\n\n') || fallback;
}
