import type { UserSpacePreviewSandboxFlagOption } from '@/types';

export function getUserSpacePreviewSandboxFlagValues(
  options: UserSpacePreviewSandboxFlagOption[],
): string[] {
  return options.map((option) => option.value);
}

export function normalizeUserSpacePreviewSandboxFlags(
  flags: string[] | null | undefined,
  allowedFlags: string[],
  fallbackFlags: string[],
): string[] {
  const selected = new Set((flags ?? fallbackFlags).map((flag) => flag.trim()).filter(Boolean));
  return allowedFlags.filter((value) => selected.has(value));
}

export function buildUserSpacePreviewSandboxAttribute(flags?: string[] | null): string {
  return (flags ?? []).join(' ');
}