import { renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const { apiMock } = vi.hoisted(() => ({
  apiMock: {
    getUserSpacePreviewSettings: vi.fn(),
  },
}));

vi.mock('@/api/client', () => ({ api: apiMock }));

import { CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS } from './constants';
import {
  CHAT_HTML_COMPONENT_POLICY_UNAVAILABLE_REASON,
  CHAT_HTML_COMPONENT_SCRIPTS_BLOCKED_REASON,
  buildChatComponentSandboxFlags,
  loadChatComponentSandboxFlags,
  resetChatComponentSandboxFlagsCache,
  useChatComponentSandboxPolicy,
} from './sandboxPolicy';

const ADMIN_FLAGS = [
  'allow-scripts',
  'allow-same-origin',
  'allow-forms',
  'allow-top-navigation',
  'allow-top-navigation-by-user-activation',
  'allow-top-navigation-to-custom-protocols',
  'allow-popups',
];

function settings(flags: string[]) {
  return { userspace_preview_sandbox_flags: flags };
}

beforeEach(() => {
  resetChatComponentSandboxFlagsCache();
  apiMock.getUserSpacePreviewSettings.mockResolvedValue(settings(ADMIN_FLAGS));
});

afterEach(() => {
  vi.clearAllMocks();
});

describe('buildChatComponentSandboxFlags', () => {
  it('strips every denied flag regardless of admin settings', () => {
    const { flags, blockedReason } = buildChatComponentSandboxFlags(ADMIN_FLAGS);
    expect(flags).toEqual(['allow-scripts', 'allow-forms', 'allow-popups']);
    for (const denied of CHAT_HTML_COMPONENT_DENIED_SANDBOX_FLAGS) {
      expect(flags).not.toContain(denied);
    }
    expect(blockedReason).toBeNull();
  });

  it('reports a blocked reason when allow-scripts is missing', () => {
    const { flags, blockedReason } = buildChatComponentSandboxFlags(['allow-forms']);
    expect(flags).toEqual(['allow-forms']);
    expect(blockedReason).toBe(CHAT_HTML_COMPONENT_SCRIPTS_BLOCKED_REASON);
  });

  it('trims, drops blanks and de-duplicates', () => {
    const { flags } = buildChatComponentSandboxFlags([' allow-scripts ', '', 'allow-scripts']);
    expect(flags).toEqual(['allow-scripts']);
  });
});

describe('loadChatComponentSandboxFlags', () => {
  it('memoizes the settings request and resolves with chat-safe flags', async () => {
    const [first, second] = await Promise.all([
      loadChatComponentSandboxFlags(),
      loadChatComponentSandboxFlags(),
    ]);
    expect(first).toEqual(['allow-scripts', 'allow-forms', 'allow-popups']);
    expect(second).toBe(first);
    expect(apiMock.getUserSpacePreviewSettings).toHaveBeenCalledTimes(1);
  });

  it('clears the memo on rejection so the next call retries', async () => {
    apiMock.getUserSpacePreviewSettings.mockRejectedValueOnce(new Error('offline'));
    await expect(loadChatComponentSandboxFlags()).rejects.toThrow('offline');

    apiMock.getUserSpacePreviewSettings.mockResolvedValueOnce(settings(['allow-scripts']));
    await expect(loadChatComponentSandboxFlags()).resolves.toEqual(['allow-scripts']);
    expect(apiMock.getUserSpacePreviewSettings).toHaveBeenCalledTimes(2);
  });
});

describe('useChatComponentSandboxPolicy', () => {
  it('starts loading with an empty sandbox and settles to ready with the filtered attribute', async () => {
    const { result } = renderHook(() => useChatComponentSandboxPolicy());
    expect(result.current).toEqual({
      status: 'loading',
      sandboxAttribute: '',
      blockedReason: null,
    });

    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(result.current.sandboxAttribute).toBe('allow-scripts allow-forms allow-popups');
    expect(result.current.sandboxAttribute).not.toContain('allow-same-origin');
    expect(result.current.sandboxAttribute).not.toContain('allow-top-navigation');
    expect(result.current.blockedReason).toBeNull();
  });

  it('shares one settings request across two hook consumers', async () => {
    const first = renderHook(() => useChatComponentSandboxPolicy());
    const second = renderHook(() => useChatComponentSandboxPolicy());

    await waitFor(() => expect(first.result.current.status).toBe('ready'));
    await waitFor(() => expect(second.result.current.status).toBe('ready'));
    expect(apiMock.getUserSpacePreviewSettings).toHaveBeenCalledTimes(1);
    expect(second.result.current.sandboxAttribute).toBe(first.result.current.sandboxAttribute);
  });

  it('exposes the allow-scripts blocked reason once ready', async () => {
    apiMock.getUserSpacePreviewSettings.mockResolvedValue(settings(['allow-forms']));
    const { result } = renderHook(() => useChatComponentSandboxPolicy());

    await waitFor(() => expect(result.current.status).toBe('ready'));
    expect(result.current.sandboxAttribute).toBe('allow-forms');
    expect(result.current.blockedReason).toBe(CHAT_HTML_COMPONENT_SCRIPTS_BLOCKED_REASON);
  });

  it('fails closed when the policy cannot be loaded', async () => {
    apiMock.getUserSpacePreviewSettings.mockRejectedValue(new Error('offline'));
    const { result } = renderHook(() => useChatComponentSandboxPolicy());

    await waitFor(() => expect(result.current.status).toBe('error'));
    expect(result.current.sandboxAttribute).toBe('');
    expect(result.current.blockedReason).toBe(CHAT_HTML_COMPONENT_POLICY_UNAVAILABLE_REASON);
  });

  it('does not update state after unmount', async () => {
    let resolveSettings: (value: { userspace_preview_sandbox_flags: string[] }) => void = () => {};
    apiMock.getUserSpacePreviewSettings.mockReturnValue(
      new Promise((resolve) => {
        resolveSettings = resolve;
      }),
    );
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const { result, unmount } = renderHook(() => useChatComponentSandboxPolicy());
    unmount();
    resolveSettings(settings(['allow-scripts']));
    await loadChatComponentSandboxFlags();

    expect(result.current.status).toBe('loading');
    expect(errorSpy).not.toHaveBeenCalled();
  });
});
