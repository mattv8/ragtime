import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ContentProtectionApiError,
  contentProtectionApi,
  requirementModeFor,
  updateContentProtectionConfigSlice,
  userOverrideMode,
  withGroupProfile,
  withRequirement,
  withUserOverride,
  type ContentProtectionConfig,
} from './contentProtection';

afterEach(() => vi.unstubAllGlobals());

const config = (revision = 1): ContentProtectionConfig => ({
  revision,
  enabled: true,
  classifier_model: null,
  coverage_mode: 'selected_scopes',
  profiles: [],
  group_profiles: [],
  requirements: [],
  user_overrides: [],
});

describe('contentProtectionApi', () => {
  it('uses the canonical config URL and exposes safe structured error messages', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: {
            code: 'content_denied',
            message: 'Content is unavailable.',
            request_id: 'req-1',
          },
        }),
        {
          status: 403,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );
    vi.stubGlobal('fetch', fetch);

    await expect(contentProtectionApi.getConfig()).rejects.toEqual(
      expect.objectContaining<Partial<ContentProtectionApiError>>({
        message: 'Content is unavailable.',
        status: 403,
      }),
    );
    expect(fetch).toHaveBeenCalledWith(
      '/indexes/content-protection/config',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('falls back to a stable status message when an error detail has no safe text', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: {} }), { status: 503 })),
    );
    await expect(contentProtectionApi.getCatalog()).rejects.toThrow(
      'Request failed with status 503',
    );
  });
});

describe('updateContentProtectionConfigSlice', () => {
  it('fetches a fresh config and saves the mutated slice with its revision', async () => {
    const freshConfig = config(4);
    const savedConfig = withUserOverride(freshConfig, 'user-1', 'always_classify');
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(freshConfig)))
      .mockResolvedValueOnce(new Response(JSON.stringify(savedConfig)));
    vi.stubGlobal('fetch', fetch);

    await expect(
      updateContentProtectionConfigSlice((current) =>
        withUserOverride(current, 'user-1', 'always_classify'),
      ),
    ).resolves.toEqual(savedConfig);

    expect(fetch).toHaveBeenNthCalledWith(
      2,
      '/indexes/content-protection/config',
      expect.objectContaining({
        body: JSON.stringify({ expected_revision: 4, config: savedConfig }),
        method: 'PUT',
      }),
    );
  });

  it('refetches and retries once after a revision conflict', async () => {
    const firstConfig = config(4);
    const secondConfig = config(5);
    const savedConfig = withRequirement(secondConfig, 'tool', 'tool-1', 'require');
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(firstConfig)))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Conflict' }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(secondConfig)))
      .mockResolvedValueOnce(new Response(JSON.stringify(savedConfig)));
    vi.stubGlobal('fetch', fetch);

    await expect(
      updateContentProtectionConfigSlice((current) =>
        withRequirement(current, 'tool', 'tool-1', 'require'),
      ),
    ).resolves.toEqual(savedConfig);

    expect(fetch).toHaveBeenCalledTimes(4);
    expect(
      fetch.mock.calls.filter(([path]) => path === '/indexes/content-protection/config'),
    ).toHaveLength(4);
    expect(fetch).toHaveBeenNthCalledWith(
      4,
      '/indexes/content-protection/config',
      expect.objectContaining({
        body: JSON.stringify({ expected_revision: 5, config: savedConfig }),
        method: 'PUT',
      }),
    );
  });

  it('surfaces the second revision conflict', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(config(1))))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Conflict' }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(config(2))))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Still conflicted' }), { status: 409 }),
      );
    vi.stubGlobal('fetch', fetch);

    await expect(updateContentProtectionConfigSlice((current) => current)).rejects.toEqual(
      expect.objectContaining<Partial<ContentProtectionApiError>>({
        message: 'Still conflicted',
        status: 409,
      }),
    );
    expect(fetch).toHaveBeenCalledTimes(4);
  });
});

describe('content protection config helpers', () => {
  it('adds, replaces, and clears user overrides', () => {
    const initial = {
      ...config(),
      user_overrides: [{ user_id: 'user-1', mode: 'never_classify' as const }],
    };
    const added = withUserOverride(initial, 'user-2', 'always_classify');
    const replaced = withUserOverride(added, 'user-1', 'always_classify');
    const cleared = withUserOverride(replaced, 'user-1', 'inherit');

    expect(userOverrideMode(initial, 'missing')).toBe('inherit');
    expect(added.user_overrides).toEqual([
      { user_id: 'user-1', mode: 'never_classify' },
      { user_id: 'user-2', mode: 'always_classify' },
    ]);
    expect(replaced.user_overrides).toContainEqual({ user_id: 'user-1', mode: 'always_classify' });
    expect(cleared.user_overrides).not.toContainEqual({
      user_id: 'user-1',
      mode: 'always_classify',
    });
  });

  it('adds, replaces, and clears requirements and group profiles', () => {
    const initial = {
      ...config(),
      requirements: [
        { scope_kind: 'group' as const, scope_key: 'group-1', mode: 'require' as const },
      ],
      group_profiles: [{ group_id: 'group-1', profile_id: 'profile-1' }],
    };
    const addedRequirement = withRequirement(initial, 'tool', 'tool-1', 'require');
    const replacedRequirement = withRequirement(addedRequirement, 'group', 'group-1', 'require');
    const clearedRequirement = withRequirement(replacedRequirement, 'group', 'group-1', 'inherit');
    const replacedProfile = withGroupProfile(initial, 'group-1', 'profile-2');
    const clearedProfile = withGroupProfile(replacedProfile, 'group-1', '');

    expect(requirementModeFor(initial, 'tool', 'missing')).toBe('inherit');
    expect(addedRequirement.requirements).toContainEqual({
      scope_kind: 'tool',
      scope_key: 'tool-1',
      mode: 'require',
    });
    expect(replacedRequirement.requirements).toContainEqual({
      scope_kind: 'group',
      scope_key: 'group-1',
      mode: 'require',
    });
    expect(clearedRequirement.requirements).not.toContainEqual({
      scope_kind: 'group',
      scope_key: 'group-1',
      mode: 'require',
    });
    expect(replacedProfile.group_profiles).toContainEqual({
      group_id: 'group-1',
      profile_id: 'profile-2',
    });
    expect(clearedProfile.group_profiles).not.toContainEqual({
      group_id: 'group-1',
      profile_id: 'profile-2',
    });
  });
});
