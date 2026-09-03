import { useEffect, useMemo, useState } from 'react';

import { api } from '@/api/client';
import { buildUserSpacePreviewSandboxAttribute } from '@/utils/userspacePreview/sandbox';

import { CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS } from './constants';

export const CHAT_HTML_COMPONENT_SCRIPTS_BLOCKED_REASON =
  'Interactive components are disabled by the preview sandbox policy (allow-scripts is off).';
export const CHAT_HTML_COMPONENT_POLICY_UNAVAILABLE_REASON =
  'Interactive components are disabled because the preview sandbox policy could not be loaded.';

const DENIED_FLAGS: ReadonlySet<string> = new Set(CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS);

/**
 * Derives the sandbox flags an in-chat component may use from the admin
 * policy. The deny-list is always stripped: a srcdoc frame inherits the app
 * origin, so allow-same-origin or any top-navigation flag would hand
 * agent-authored code the user's session or the app shell.
 */
export function buildChatComponentSandboxFlags(adminFlags: string[]): {
  flags: string[];
  blockedReason: string | null;
} {
  const flags: string[] = [];
  for (const rawFlag of adminFlags ?? []) {
    const flag = typeof rawFlag === 'string' ? rawFlag.trim() : '';
    if (!flag || DENIED_FLAGS.has(flag) || flags.includes(flag)) continue;
    flags.push(flag);
  }
  return {
    flags,
    blockedReason: flags.includes('allow-scripts')
      ? null
      : CHAT_HTML_COMPONENT_SCRIPTS_BLOCKED_REASON,
  };
}

let cachedFlagsPromise: Promise<string[]> | null = null;

/**
 * Loads the admin preview sandbox policy once per page and resolves with the
 * chat-safe flags (deny-list already stripped, so callers can never obtain
 * allow-same-origin from this function). A rejection clears the memo so the
 * next consumer retries instead of inheriting a permanent failure.
 */
export function loadChatComponentSandboxFlags(): Promise<string[]> {
  if (cachedFlagsPromise) return cachedFlagsPromise;

  const pending = api
    .getUserSpacePreviewSettings()
    .then(
      (response) =>
        buildChatComponentSandboxFlags(response?.userspace_preview_sandbox_flags ?? []).flags,
    );
  cachedFlagsPromise = pending;
  pending.catch(() => {
    if (cachedFlagsPromise === pending) cachedFlagsPromise = null;
  });
  return pending;
}

export function resetChatComponentSandboxFlagsCache(): void {
  cachedFlagsPromise = null;
}

type PolicyStatus = 'loading' | 'ready' | 'error';

interface PolicyState {
  status: PolicyStatus;
  flags: string[];
}

const INITIAL_POLICY_STATE: PolicyState = { status: 'loading', flags: [] };

export function useChatComponentSandboxPolicy(): {
  status: PolicyStatus;
  sandboxAttribute: string;
  blockedReason: string | null;
} {
  const [state, setState] = useState<PolicyState>(INITIAL_POLICY_STATE);

  useEffect(() => {
    let cancelled = false;

    loadChatComponentSandboxFlags()
      .then((flags) => {
        if (!cancelled) setState({ status: 'ready', flags });
      })
      .catch(() => {
        if (!cancelled) setState({ status: 'error', flags: [] });
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return useMemo(() => {
    const { flags, blockedReason } = buildChatComponentSandboxFlags(state.flags);
    const resolvedReason =
      state.status === 'error'
        ? CHAT_HTML_COMPONENT_POLICY_UNAVAILABLE_REASON
        : state.status === 'ready'
          ? blockedReason
          : null;
    return {
      status: state.status,
      sandboxAttribute: buildUserSpacePreviewSandboxAttribute(flags),
      blockedReason: resolvedReason,
    };
  }, [state]);
}
