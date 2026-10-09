import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ContentProtectionApiError,
  contentProtectionApi,
  type ContentProtectionConfig,
} from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { AccessLevelsModal } from './AccessLevelsModal';

const level = (id: string, name = id) => ({
  id,
  name,
  granted_category_ids: ['internal'],
  guidance: '',
});
const config = (
  revision = 1,
  access_levels = [level('standard', 'Standard')],
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
const toast = { success: vi.fn(), error: vi.fn() };
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
function renderModal() {
  const authGroups = [
    { id: 'group', display_name: 'Group', provider: 'local_managed', member_count: 1 } as AuthGroup,
  ];
  return render(<AccessLevelsModal open onClose={vi.fn()} authGroups={authGroups} toast={toast} />);
}

beforeEach(() => {
  vi.restoreAllMocks();
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
afterEach(cleanup);

describe('AccessLevelsModal data operations', () => {
  it('clears a stale preview when the selected level changes and requests the new saved level', async () => {
    const first = deferred<{ prompt_fragment: string }>();
    vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue(
      config(1, [level('standard', 'Standard'), level('other', 'Other')]),
    );
    vi.spyOn(contentProtectionApi, 'preview')
      .mockReturnValueOnce(first.promise as never)
      .mockResolvedValueOnce({ prompt_fragment: 'other preview' } as never);
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.click(screen.getByRole('button', { name: /other/i }));
    first.resolve({ prompt_fragment: 'old preview' });
    expect(await screen.findByText('other preview')).toBeTruthy();
    expect(screen.queryByText('old preview')).toBeNull();
  });

  it('shows that an unsaved access level must be saved before previewing', async () => {
    renderModal();
    await screen.findByText('Standard');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    expect(screen.getByText('Save to preview.')).toBeTruthy();
  });

  it('treats an already deleted level as a successful confirmed delete without a ghost row', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    vi.spyOn(contentProtectionApi, 'getConfig')
      .mockResolvedValueOnce({ ...config(), default_access_level_id: '' })
      .mockResolvedValueOnce({ ...config(2, []), default_access_level_id: '' });
    vi.spyOn(contentProtectionApi, 'saveConfig').mockRejectedValueOnce(
      new ContentProtectionApiError('conflict', 409),
    );
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.click(screen.getByRole('button', { name: 'Delete level' }));
    await vi.advanceTimersByTimeAsync(3000);
    await userEvent.click(screen.getByRole('button', { name: /confirm/i }));
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith('Access level deleted'));
    expect(screen.queryByRole('button', { name: /standard/i })).toBeNull();
    vi.useRealTimers();
  });

  it('retries one unchanged delete at the latest revision', async () => {
    const initial = { ...config(), default_access_level_id: '' };
    const latest = { ...config(2), default_access_level_id: '' };
    vi.spyOn(contentProtectionApi, 'getConfig')
      .mockResolvedValueOnce(initial)
      .mockResolvedValueOnce(latest);
    const save = vi
      .spyOn(contentProtectionApi, 'saveConfig')
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockResolvedValueOnce({ ...latest, revision: 3, access_levels: [] });
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.click(screen.getByRole('button', { name: 'Delete level' }));
    await new Promise((resolve) => setTimeout(resolve, 3100));
    await userEvent.click(screen.getByRole('button', { name: /confirm/i }));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
    expect(save.mock.calls[1][0]).toBe(2);
  });

  it('disables mapping and default writes for an unsaved level', async () => {
    renderModal();
    await screen.findByText('Standard');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    expect(screen.queryByLabelText('Add group to this level')).toBeNull();
    expect(
      (screen.getByRole('button', { name: 'Make default level' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('uses scoped mapping writes and rejects a mapping when the selected level was deleted', async () => {
    const freshWithoutLevel = config(2, []);
    vi.spyOn(contentProtectionApi, 'getConfig')
      .mockResolvedValueOnce(config())
      .mockResolvedValueOnce(freshWithoutLevel);
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.selectOptions(screen.getByLabelText('Add group to this level'), 'group');
    await waitFor(() =>
      expect(toast.error).toHaveBeenCalledWith('This access level was deleted by another admin.'),
    );
    expect(contentProtectionApi.saveConfig).not.toHaveBeenCalled();
  });

  it('replaces the config after making a default', async () => {
    const levels = [level('standard', 'Standard'), level('other', 'Other')];
    const saved = { ...config(3, levels), default_access_level_id: 'other' };
    vi.spyOn(contentProtectionApi, 'getConfig')
      .mockResolvedValueOnce(config(1, levels))
      .mockResolvedValueOnce(config(2, levels));
    vi.spyOn(contentProtectionApi, 'saveConfig').mockResolvedValueOnce(saved);
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.click(screen.getByRole('button', { name: /other/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Make default level' }));
    expect((await screen.findAllByText('Default')).length).toBeGreaterThan(0);
  });

  it('keeps saved row counts while a new level is selected', async () => {
    const initial = {
      ...config(1, [level('standard', 'Standard')]),
      group_access_levels: [{ group_id: 'group', access_level_id: 'standard' }],
    };
    vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue(initial);
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    expect(screen.getByRole('button', { name: /standard.*1 group.*1 category/i })).toBeTruthy();
  });

  it('shows changed-save conflict actions and overwrites at the latest revision', async () => {
    const latest = config(2, [level('standard', 'Server edit')]);
    const save = vi
      .spyOn(contentProtectionApi, 'saveConfig')
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockResolvedValueOnce({
        ...latest,
        revision: 3,
        access_levels: [level('standard', 'Local edit')],
      });
    vi.spyOn(contentProtectionApi, 'getConfig')
      .mockResolvedValueOnce(config())
      .mockResolvedValueOnce(latest)
      .mockResolvedValueOnce(latest);
    renderModal();
    const name = await screen.findByLabelText('Name');
    await userEvent.clear(name);
    await userEvent.type(name, 'Local edit');
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    await screen.findByRole('button', { name: 'Overwrite' });
    await userEvent.click(screen.getByRole('button', { name: 'Overwrite' }));
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2));
    expect(save.mock.calls[1][0]).toBe(2);
  });

  it('shows a notice when create rebase prunes grants removed by another admin', async () => {
    const initial = {
      ...config(),
      categories: [
        ...config().categories,
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
    };
    const latest = config(2);
    vi.spyOn(contentProtectionApi, 'getConfig')
      .mockResolvedValueOnce(initial)
      .mockResolvedValueOnce(latest);
    vi.spyOn(contentProtectionApi, 'saveConfig')
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockResolvedValueOnce({
        ...latest,
        revision: 3,
        access_levels: [...latest.access_levels, level('new', 'New')],
      });
    renderModal();
    await screen.findByDisplayValue('Standard');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    await userEvent.type(screen.getByLabelText('Name'), 'New');
    await userEvent.click(screen.getByRole('checkbox', { name: /removed/i }));
    await userEvent.click(screen.getByRole('button', { name: 'Save' }));
    expect(await screen.findByRole('status')).toBeTruthy();
  });
});
