import { act, cleanup, renderHook, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { deferred } from '@/testHelpers/deferred';
import { api } from '@/api';
import { isGitAuthenticationError, useGitTokenValidation } from './gitTokenValidation';

vi.mock('@/api', () => ({ api: { fetchBranches: vi.fn() } }));
const fetchBranches = vi.mocked(api.fetchBranches);
afterEach(cleanup);

describe('useGitTokenValidation', () => {
  beforeEach(() => fetchBranches.mockReset());

  it('deduplicates simultaneous checks for the same repository and token', async () => {
    const check = deferred<{ branches: string[]; error: null; needs_token: boolean }>();
    fetchBranches.mockReturnValue(check.promise);
    const { result } = renderHook(() =>
      useGitTokenValidation('https://github.com/acme/repo.git', 'repo'),
    );

    let first!: Promise<boolean>;
    let second!: Promise<boolean>;
    act(() => {
      first = result.current.validate('candidate');
      second = result.current.validate('candidate');
    });
    expect(fetchBranches).toHaveBeenCalledTimes(1);
    check.resolve({ branches: ['main'], error: null, needs_token: false });
    await act(async () => {
      expect(await first).toBe(true);
      expect(await second).toBe(true);
    });
  });

  it('starts a fresh same-key check after invalidation and ignores the old completion', async () => {
    const oldCheck = deferred<{ branches: string[]; error: null; needs_token: boolean }>();
    const freshCheck = deferred<{ branches: string[]; error: null; needs_token: boolean }>();
    fetchBranches.mockReturnValueOnce(oldCheck.promise).mockReturnValueOnce(freshCheck.promise);
    const { result } = renderHook(() =>
      useGitTokenValidation('https://github.com/acme/repo.git', 'repo'),
    );

    act(() => void result.current.validate('candidate'));
    act(() => result.current.invalidate());
    let fresh!: Promise<boolean>;
    act(() => {
      fresh = result.current.validate('candidate');
    });
    expect(fetchBranches).toHaveBeenCalledTimes(2);

    oldCheck.resolve({ branches: ['main'], error: null, needs_token: false });
    await act(async () => Promise.resolve());
    expect(result.current.state).toBe('checking');
    freshCheck.resolve({ branches: ['main'], error: null, needs_token: false });
    await act(async () => expect(await fresh).toBe(true));
    await waitFor(() => expect(result.current.state).toBe('valid'));
  });

  it('treats backend timeout results as failed checks rather than invalid credentials', async () => {
    fetchBranches.mockResolvedValue({
      branches: [],
      error: 'Repository access check timed out',
      needs_token: false,
    });
    const { result } = renderHook(() =>
      useGitTokenValidation('https://github.com/acme/repo.git', 'repo'),
    );
    await act(async () => void (await result.current.validate('candidate')));
    expect(result.current.state).toBe('failed');
  });
});

describe('isGitAuthenticationError', () => {
  it('does not treat rate-limit responses as credential failures', () => {
    expect(isGitAuthenticationError('API rate limit exceeded; HTTP 403')).toBe(false);
    expect(isGitAuthenticationError('Too many requests (429)')).toBe(false);
  });
  it('recognizes repository selection failures without misclassifying local filesystem permissions', () => {
    expect(
      isGitAuthenticationError(
        'Repository could not be found or read: token may not have this repository selected',
      ),
    ).toBe(true);
    expect(
      isGitAuthenticationError(
        "Git fetch failed: cannot open '.git/FETCH_HEAD': Permission denied",
      ),
    ).toBe(false);
  });
});
