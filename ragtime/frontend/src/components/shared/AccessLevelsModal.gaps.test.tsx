import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ContentProtectionApiError,
  contentProtectionApi,
  DELETED_ACCESS_LEVEL_MESSAGE,
  type ContentProtectionConfig,
} from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { AccessLevelsModal } from './AccessLevelsModal';

const contentProtectionMock = vi.hoisted(() => ({
  updateConfigSlice:
    vi.fn<
      (
        mutate: (config: ContentProtectionConfig) => ContentProtectionConfig,
      ) => Promise<ContentProtectionConfig>
    >(),
}));

vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  updateContentProtectionConfigSlice: contentProtectionMock.updateConfigSlice,
}));

const accessLevel = (id: string, name = id) => ({
  id,
  name,
  granted_category_ids: ['internal'],
  guidance: '',
});

const config = (
  revision = 1,
  access_levels = [accessLevel('standard', 'Standard')],
): ContentProtectionConfig => ({
  schema_version: 2,
  revision,
  enabled: true,
  share_with_assistant: true,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: '' }, llm_model: null },
  strictness: 'balanced',
  categories: [
    {
      id: 'internal',
      name: 'Internal',
      description: '',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: '',
      threshold_override: null,
      system: false,
    },
  ],
  access_levels,
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'all_supported_traffic',
  requirements: [],
  user_overrides: [],
});

const authGroup = {
  id: 'group',
  display_name: 'Group',
  provider: 'local_managed',
  member_count: 1,
} as AuthGroup;
const toast = { success: vi.fn(), error: vi.fn() };

function renderModal() {
  return render(
    <AccessLevelsModal open onClose={vi.fn()} authGroups={[authGroup]} toast={toast} />,
  );
}

beforeEach(() => {
  vi.restoreAllMocks();
  contentProtectionMock.updateConfigSlice.mockReset();
  toast.success.mockClear();
  toast.error.mockClear();
  vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue(config());
  vi.spyOn(contentProtectionApi, 'saveConfig').mockImplementation(async (_revision, next) => ({
    ...next,
    revision: next.revision + 1,
  }));
  vi.spyOn(contentProtectionApi, 'preview').mockResolvedValue({
    prompt_fragment: 'saved preview',
  } as never);
});

afterEach(() => {
  vi.useRealTimers();
  cleanup();
});

describe('AccessLevelsModal remaining data conflicts', () => {
  it('aborts a conflicted delete for changed, default, and mapped levels and adopts each latest config', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const cases = [
      {
        latest: {
          ...config(2, [accessLevel('standard', 'Server Edit')]),
          default_access_level_id: '',
        },
        reason: 'This access level was edited by another admin.',
        railName: /server edit/i,
      },
      {
        latest: config(2, [accessLevel('standard', 'Now Default')]),
        reason: 'This access level is now the default level.',
        railName: /now default/i,
      },
      {
        latest: {
          ...config(2, [accessLevel('standard', 'Now Mapped')]),
          default_access_level_id: '',
          group_access_levels: [{ group_id: 'group', access_level_id: 'standard' }],
        },
        reason: 'This access level is now mapped to a group.',
        railName: /now mapped/i,
      },
    ];

    for (const testCase of cases) {
      cleanup();
      vi.clearAllMocks();
      const initial = { ...config(), default_access_level_id: '' };
      vi.mocked(contentProtectionApi.getConfig)
        .mockResolvedValueOnce(initial)
        .mockResolvedValueOnce(testCase.latest);
      vi.mocked(contentProtectionApi.saveConfig).mockRejectedValueOnce(
        new ContentProtectionApiError('conflict', 409),
      );
      vi.mocked(contentProtectionApi.preview).mockResolvedValue({
        prompt_fragment: 'saved preview',
      } as never);

      renderModal();
      await screen.findByDisplayValue('Standard');
      await userEvent.click(screen.getByRole('button', { name: 'Delete level' }));
      await vi.advanceTimersByTimeAsync(3000);
      await userEvent.click(screen.getByRole('button', { name: 'Confirm?' }));

      await waitFor(() => expect(toast.error).toHaveBeenCalledWith(testCase.reason));
      expect(contentProtectionApi.saveConfig).toHaveBeenCalledTimes(1);
      expect(await screen.findByDisplayValue(testCase.latest.access_levels[0].name)).toBeTruthy();
      expect(screen.getByRole('button', { name: testCase.railName })).toBeTruthy();
    }
  });

  it('adds and removes a group with scoped mutators and updates the group chips', async () => {
    let fresh = { ...config(), default_access_level_id: '' };
    contentProtectionMock.updateConfigSlice.mockImplementation(async (mutate) => {
      fresh = { ...mutate(fresh), revision: fresh.revision + 1 };
      return fresh;
    });
    renderModal();
    await screen.findByDisplayValue('Standard');

    await userEvent.selectOptions(screen.getByLabelText('Add group to this level'), 'group');
    await waitFor(() => expect(contentProtectionMock.updateConfigSlice).toHaveBeenCalledTimes(1));
    const addMutator = contentProtectionMock.updateConfigSlice.mock.calls[0][0];
    expect(addMutator(config()).group_access_levels).toEqual([
      { group_id: 'group', access_level_id: 'standard' },
    ]);
    expect(document.querySelector('[data-level-group-chip="group"]')).toBeTruthy();
    const configWithMapping = fresh;

    await userEvent.click(screen.getByRole('button', { name: 'Remove Group from this level' }));
    await waitFor(() => expect(contentProtectionMock.updateConfigSlice).toHaveBeenCalledTimes(2));
    const removeMutator = contentProtectionMock.updateConfigSlice.mock.calls[1][0];
    expect(removeMutator(configWithMapping).group_access_levels).toEqual([]);
    await waitFor(() =>
      expect(document.querySelector('[data-level-group-chip="group"]')).toBeNull(),
    );
  });

  it('does not write when making a deleted level the default in a fresh config', async () => {
    const freshWithoutLevel = { ...config(2, []), default_access_level_id: '' };
    contentProtectionMock.updateConfigSlice.mockImplementation(async (mutate) => {
      const next = mutate(freshWithoutLevel);
      return contentProtectionApi.saveConfig(freshWithoutLevel.revision, next);
    });
    vi.mocked(contentProtectionApi.getConfig).mockResolvedValue({
      ...config(),
      default_access_level_id: '',
    });
    renderModal();
    await screen.findByDisplayValue('Standard');

    await userEvent.click(screen.getByRole('button', { name: 'Make default level' }));

    await waitFor(() => expect(toast.error).toHaveBeenCalledWith(DELETED_ACCESS_LEVEL_MESSAGE));
    expect(contentProtectionApi.saveConfig).not.toHaveBeenCalled();
  });

  it('disables Save after the edited level is found deleted during conflict recovery', async () => {
    const freshWithoutLevel = { ...config(2, []), default_access_level_id: '' };
    vi.mocked(contentProtectionApi.getConfig)
      .mockResolvedValueOnce(config())
      .mockResolvedValueOnce(freshWithoutLevel);
    vi.mocked(contentProtectionApi.saveConfig).mockRejectedValueOnce(
      new ContentProtectionApiError('conflict', 409),
    );
    renderModal();
    const name = await screen.findByLabelText('Name');
    await userEvent.clear(name);
    await userEvent.type(name, 'Local Edit');

    await userEvent.click(screen.getByRole('button', { name: 'Save' }));

    expect(await screen.findByText(/draft is retained read-only/i)).toBeTruthy();
    expect((screen.getByRole('button', { name: 'Save' }) as HTMLButtonElement).disabled).toBe(true);
  });

  it('disables deleted-conflict writes other than Save', async () => {
    const freshWithoutLevel = { ...config(2, []), default_access_level_id: '' };
    vi.mocked(contentProtectionApi.getConfig)
      .mockResolvedValueOnce({ ...config(), default_access_level_id: '' })
      .mockResolvedValueOnce(freshWithoutLevel);
    vi.mocked(contentProtectionApi.saveConfig).mockRejectedValueOnce(
      new ContentProtectionApiError('conflict', 409),
    );
    renderModal();
    const name = await screen.findByLabelText('Name');
    await userEvent.clear(name);
    await userEvent.type(name, 'Local Edit');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    await screen.findByText(/draft is retained read-only/i);

    expect(
      (screen.getByRole('button', { name: 'Delete level' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect(
      (screen.getByRole('button', { name: 'Make default level' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    expect((screen.getByLabelText('Add group to this level') as HTMLSelectElement).disabled).toBe(
      true,
    );
  });

  it('uses a neutral mapping fallback for a non-Error rejection', async () => {
    contentProtectionMock.updateConfigSlice.mockRejectedValue('boom');
    vi.mocked(contentProtectionApi.getConfig).mockResolvedValue({
      ...config(),
      default_access_level_id: '',
    });
    renderModal();
    await screen.findByDisplayValue('Standard');

    await userEvent.selectOptions(screen.getByLabelText('Add group to this level'), 'group');

    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith('Could not update group mapping.'),
    );
    expect(toast.error).not.toHaveBeenCalledWith(DELETED_ACCESS_LEVEL_MESSAGE);
  });
});
