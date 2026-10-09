import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  ContentProtectionApiError,
  DELETED_ACCESS_LEVEL_MESSAGE,
  contentProtectionApi,
  requirementModeFor,
  updateContentProtectionConfigSlice,
  userOverrideMode,
  withExistingAccessLevel,
  withGroupAccessLevel,
  withRequirement,
  withUserOverride,
  type ContentProtectionConfig,
} from './contentProtection';
afterEach(() => vi.unstubAllGlobals());
const config = (revision = 1): ContentProtectionConfig => ({
  schema_version: 2,
  revision,
  enabled: false,
  share_with_assistant: false,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: 'jev-latest' }, llm_model: null },
  strictness: 'strict',
  categories: [
    {
      id: 'operational',
      name: 'Operational',
      description: 'Operations.',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: 'Restricted.',
      threshold_override: null,
      system: false,
    },
    {
      id: 'rule_override',
      name: 'Rule override',
      description: 'Policy overrides.',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: 'Cannot override.',
      threshold_override: null,
      system: true,
    },
  ],
  access_levels: [
    { id: 'standard', name: 'Standard', granted_category_ids: ['operational'], guidance: '' },
  ],
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'selected_scopes',
  requirements: [],
  user_overrides: [],
});
describe('contentProtectionApi', () => {
  it('uses a stable error when the response has no safe detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('upstream unavailable', { status: 503 })),
    );
    await expect(contentProtectionApi.getCatalog()).rejects.toThrow(
      'Request failed with status 503',
    );
  });
  it('uses the canonical config URL and exposes safe structured error messages', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: { message: 'Content is unavailable.' } }), {
        status: 403,
        headers: { 'Content-Type': 'application/json' },
      }),
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

  it('sends an explicit access-level selector and preserves the synthetic preview response', async () => {
    const response = {
      required: null,
      provenance: 'synthetic_access_levels',
      access_levels: [[{ id: 'finance', name: 'Finance', granted_category_ids: ['operational'] }]],
      granted_category_ids: ['operational'],
      categories: [{ id: 'operational', name: 'Operational' }],
      guidance: ['Keep customer data internal.'],
      policy_revision: 7,
      guidance_revision: 'guidance-7',
      share_with_assistant: true,
      prompt_fragment: 'SERVER-BUILT PROMPT',
    };
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(response)));
    vi.stubGlobal('fetch', fetch);

    const preview = await contentProtectionApi.preview({ access_level_ids: ['finance'] });

    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ access_level_ids: ['finance'] });
    expect(preview.required).toBeNull();
    expect(preview.provenance).toBe('synthetic_access_levels');
    expect(preview.prompt_fragment).toBe('SERVER-BUILT PROMPT');
  });

  it('keeps an empty access-level selector distinct from omitting it', async () => {
    const response = () =>
      new Response(
        JSON.stringify({
          required: false,
          provenance: 'authenticated_context',
          access_levels: [],
          granted_category_ids: [],
          categories: [],
          guidance: [],
          policy_revision: 7,
          guidance_revision: 'guidance-7',
          share_with_assistant: false,
        }),
      );
    const fetch = vi.fn().mockImplementation(response);
    vi.stubGlobal('fetch', fetch);

    await contentProtectionApi.preview({ access_level_ids: [] });
    await contentProtectionApi.preview({});

    expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ access_level_ids: [] });
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({});
  });
});
describe('content protection config helpers', () => {
  it('guards access-level mutations against a concurrently deleted level', () => {
    const initial = config();
    expect(withExistingAccessLevel(initial, 'standard', (level) => level.name)).toBe('Standard');
    expect(() => withExistingAccessLevel(initial, 'missing', () => undefined)).toThrow(
      DELETED_ACCESS_LEVEL_MESSAGE,
    );
  });
  it('changes only the selected user override and removes inherit entries', () => {
    const initial: ContentProtectionConfig = {
      ...config(),
      user_overrides: [
        { user_id: 'alice', mode: 'never_classify' },
        { user_id: 'bob', mode: 'always_classify' },
      ],
    };
    const changed = withUserOverride(initial, 'alice', 'always_classify');
    expect(userOverrideMode(initial, 'unknown')).toBe('inherit');
    expect(changed.user_overrides).toEqual([
      { user_id: 'bob', mode: 'always_classify' },
      { user_id: 'alice', mode: 'always_classify' },
    ]);
    expect(withUserOverride(changed, 'alice', 'inherit').user_overrides).toEqual([
      { user_id: 'bob', mode: 'always_classify' },
    ]);
    expect(userOverrideMode(initial, 'alice')).toBe('never_classify');
  });

  it('changes one scoped requirement without removing other scopes or duplicating it', () => {
    const initial: ContentProtectionConfig = {
      ...config(),
      requirements: [{ scope_kind: 'group', scope_key: 'shared-id', mode: 'require' }],
    };
    const added = withRequirement(initial, 'tool', 'shared-id', 'require');
    const repeated = withRequirement(added, 'tool', 'shared-id', 'require');
    expect(requirementModeFor(initial, 'tool', 'shared-id')).toBe('inherit');
    expect(repeated.requirements).toEqual([
      { scope_kind: 'group', scope_key: 'shared-id', mode: 'require' },
      { scope_kind: 'tool', scope_key: 'shared-id', mode: 'require' },
    ]);
    expect(withRequirement(repeated, 'tool', 'shared-id', 'inherit').requirements).toEqual(
      initial.requirements,
    );
  });
  it('keeps other access-level mappings while toggling one group level', () => {
    const initial = {
      ...config(),
      group_access_levels: [
        { group_id: 'group-1', access_level_id: 'standard' },
        { group_id: 'group-2', access_level_id: 'standard' },
      ],
    };
    const mapped = withGroupAccessLevel(initial, 'group-1', 'finance');
    const unmapped = withGroupAccessLevel(mapped, 'group-1', 'standard', false);
    expect(mapped.group_access_levels).toContainEqual({
      group_id: 'group-1',
      access_level_id: 'finance',
    });
    expect(unmapped.group_access_levels).toEqual([
      { group_id: 'group-2', access_level_id: 'standard' },
      { group_id: 'group-1', access_level_id: 'finance' },
    ]);
  });
  it('saves a mutated config slice', async () => {
    const fresh = config(4);
    const saved = withGroupAccessLevel(fresh, 'group-1', 'standard');
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(fresh)))
      .mockResolvedValueOnce(new Response(JSON.stringify(saved)));
    vi.stubGlobal('fetch', fetch);
    await expect(
      updateContentProtectionConfigSlice((current) =>
        withGroupAccessLevel(current, 'group-1', 'standard'),
      ),
    ).resolves.toEqual(saved);
    expect(JSON.parse(fetch.mock.calls[1][1].body)).toEqual({
      expected_revision: 4,
      config: {
        ...fresh,
        group_access_levels: [{ group_id: 'group-1', access_level_id: 'standard' }],
      },
    });
  });

  it('rebases only its slice after one conflict, preserving concurrent admin edits', async () => {
    const newer: ContentProtectionConfig = {
      ...config(5),
      share_with_assistant: true,
      requirements: [{ scope_kind: 'surface', scope_key: 'mcp', mode: 'require' }],
    };
    const expected = {
      ...newer,
      user_overrides: [{ user_id: 'alice', mode: 'always_classify' }],
    };
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(config(4))))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Conflict' }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(newer)))
      .mockResolvedValueOnce(new Response(JSON.stringify({ ...expected, revision: 6 })));
    vi.stubGlobal('fetch', fetch);
    const saved = await updateContentProtectionConfigSlice((current) =>
      withUserOverride(current, 'alice', 'always_classify'),
    );
    expect(JSON.parse(fetch.mock.calls[3][1].body)).toEqual({
      expected_revision: 5,
      config: expected,
    });
    expect(saved.revision).toBe(6);
    expect(saved.requirements).toEqual(newer.requirements);
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it('surfaces a second conflict rather than retrying indefinitely', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(config(1))))
      .mockResolvedValueOnce(new Response(JSON.stringify({ detail: 'Conflict' }), { status: 409 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(config(2))))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Still conflicted' }), { status: 409 }),
      );
    vi.stubGlobal('fetch', fetch);
    await expect(updateContentProtectionConfigSlice((current) => current)).rejects.toMatchObject({
      status: 409,
      message: 'Still conflicted',
    });
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it('does not retry authorization failures', async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(new Response(JSON.stringify(config())))
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ detail: 'Forbidden' }), { status: 403 }),
      );
    vi.stubGlobal('fetch', fetch);
    await expect(updateContentProtectionConfigSlice((current) => current)).rejects.toMatchObject({
      status: 403,
    });
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
