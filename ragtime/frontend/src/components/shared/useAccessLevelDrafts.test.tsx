import { act, renderHook, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentProtectionApiError, type ContentProtectionConfig } from '@/api/contentProtection';
import { useAccessLevelDrafts } from './useAccessLevelDrafts';

const api = vi.hoisted(() => ({ getConfig: vi.fn(), saveConfig: vi.fn() }));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: api,
}));

const level = (id: string, name = id) => ({
  id,
  name,
  granted_category_ids: ['operational'],
  guidance: '',
});
const config = (revision = 1, levels = [level('standard')]): ContentProtectionConfig => ({
  schema_version: 2,
  revision,
  enabled: true,
  share_with_assistant: true,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: 'jev' }, llm_model: null },
  strictness: 'strict',
  categories: [
    {
      id: 'operational',
      name: 'Operational',
      description: '',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: '',
      threshold_override: null,
      system: false,
    },
    {
      id: 'removed',
      name: 'Removed',
      description: '',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: '',
      threshold_override: null,
      system: false,
    },
  ],
  access_levels: levels,
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'selected_scopes',
  requirements: [],
  user_overrides: [],
});
function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function drafts(initial = config()) {
  return renderHook(
    ({ current }) =>
      useAccessLevelDrafts({
        config: current,
        onConfigSaved: vi.fn(),
        toast: { success: vi.fn(), error: vi.fn() },
      }),
    { initialProps: { current: initial } },
  );
}
afterEach(() => vi.clearAllMocks());

describe('useAccessLevelDrafts', () => {
  it('rebases a 409 using the fresh complete config', async () => {
    const initial = config();
    const latest = { ...config(2), strictness: 'permissive' as const };
    const retry = deferred<ContentProtectionConfig>();
    api.saveConfig
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockReturnValueOnce(retry.promise);
    api.getConfig.mockResolvedValue(latest);
    const { result } = drafts(initial);
    act(() => result.current.updateDraft('standard', { name: 'Edited' }));
    act(() => {
      void result.current.saveLevel('standard');
    });
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(2));
    expect(api.saveConfig.mock.calls[1][0]).toBe(2);
    expect(api.saveConfig.mock.calls[1][1].strictness).toBe('permissive');
    await act(async () =>
      retry.resolve({ ...latest, revision: 3, access_levels: [level('standard', 'Edited')] }),
    );
  });

  it('keeps a changed-level draft in conflict', async () => {
    api.saveConfig.mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409));
    api.getConfig.mockResolvedValue(config(2, [level('standard', 'Server edit')]));
    const { result } = drafts();
    act(() => result.current.updateDraft('standard', { name: 'Local edit' }));
    await act(async () => result.current.saveLevel('standard'));
    expect(result.current.conflicts.standard?.type).toBe('changed');
  });

  it('retains a dirty deleted level across config sync', () => {
    const hook = drafts(config(1, [level('standard'), level('finance')]));
    act(() => hook.result.current.updateDraft('finance', { name: 'Local finance' }));
    hook.rerender({ current: config(2, [level('standard')]) });
    expect(hook.result.current.drafts.finance.name).toBe('Local finance');
    expect(hook.result.current.conflicts.finance?.type).toBe('deleted');
  });

  it('inserts a new level, retries insertion when absent after 409, and reports a collision for the update path', async () => {
    const created = level('new', 'New');
    const initial = config();
    const latest = config(2);
    api.saveConfig
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockResolvedValueOnce({
        ...latest,
        revision: 3,
        access_levels: [...latest.access_levels, created],
      });
    api.getConfig.mockResolvedValueOnce(latest);
    const { result } = drafts(initial);
    act(() => result.current.createLevel(created));
    await act(async () => result.current.saveLevel('new'));
    expect(api.saveConfig.mock.calls[1][1].access_levels).toContainEqual(created);

    const existingLatest = config(4, [level('standard'), level('new', 'Server')]);
    api.saveConfig.mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409));
    api.getConfig.mockResolvedValueOnce(existingLatest);
    act(() => result.current.updateDraft('new', { name: 'New' }));
    await act(async () => result.current.saveLevel('new'));
    expect(result.current.conflicts.new?.type).toBe('changed');
  });

  it('prunes grants missing from the latest config before inserting', async () => {
    const initial = config();
    const latest = {
      ...config(2),
      categories: initial.categories.filter((category) => category.id !== 'removed'),
    };
    api.saveConfig
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockResolvedValueOnce({
        ...latest,
        revision: 3,
        access_levels: [...latest.access_levels, level('new')],
      });
    api.getConfig.mockResolvedValueOnce(latest);
    const { result } = drafts(initial);
    act(() =>
      result.current.createLevel({
        ...level('new'),
        granted_category_ids: ['operational', 'removed'],
      }),
    );
    await act(async () => result.current.saveLevel('new'));
    expect(
      api.saveConfig.mock.calls[1][1].access_levels.find(
        (item: { id: string }) => item.id === 'new',
      ).granted_category_ids,
    ).toEqual(['operational']);
    expect(result.current.prunedGrantsLevelId).toBe('new');
  });

  it('sends only one save while the first request is pending', async () => {
    const pending = deferred<ContentProtectionConfig>();
    api.saveConfig.mockReturnValueOnce(pending.promise);
    const { result } = drafts();
    act(() => result.current.updateDraft('standard', { name: 'Edited' }));
    act(() => {
      void result.current.saveLevel('standard');
      void result.current.saveLevel('standard');
    });

    expect(api.saveConfig).toHaveBeenCalledTimes(1);
    await act(async () => pending.resolve(config(2, [level('standard', 'Edited')])));
  });

  it('marks new drafts dirty and removes them when discarded', () => {
    const { result } = drafts();
    act(() => result.current.createLevel(level('new')));
    expect(result.current.isDirty('new')).toBe(true);
    act(() => result.current.discardLevel('new'));
    expect(result.current.drafts.new).toBeUndefined();
  });

  it('clears one conflict and error on discard while discardAll preserves conflicts', async () => {
    api.saveConfig.mockRejectedValueOnce(new Error('save failed'));
    const { result } = drafts();
    act(() => result.current.updateDraft('standard', { name: 'Local' }));
    await act(async () => result.current.saveLevel('standard'));
    expect(result.current.saveErrors.standard).toBe('save failed');
    act(() => result.current.discardLevel('standard'));
    expect(result.current.saveErrors.standard).toBeUndefined();
    act(() => result.current.updateDraft('standard', { name: 'Local again' }));
    api.saveConfig.mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409));
    api.getConfig.mockResolvedValueOnce(config(2, [level('standard', 'Server')]));
    await act(async () => result.current.saveLevel('standard'));
    act(() => result.current.discardAll());
    expect(result.current.conflicts.standard?.type).toBe('changed');
  });
});
