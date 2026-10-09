import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { contentProtectionApi, type ContentProtectionConfig } from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { AccessLevelsModal } from './AccessLevelsModal';

const level = (id = 'standard', name = 'Standard') => ({
  id,
  name,
  granted_category_ids: ['internal'],
  guidance: '',
});
const config = (revision = 1): ContentProtectionConfig => ({
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
      description: 'Internal data',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: '',
      threshold_override: null,
      system: false,
    },
    {
      id: 'rule_override',
      name: 'Rule override',
      description: '',
      includes: [],
      excludes: [],
      examples: [],
      denial_message: '',
      threshold_override: null,
      system: true,
    },
  ],
  access_levels: [level()],
  group_access_levels: [],
  default_access_level_id: 'standard',
  coverage_mode: 'all_supported_traffic',
  requirements: [],
  user_overrides: [],
});

const toast = { success: vi.fn(), error: vi.fn() };
const authGroup = {
  id: 'group-1',
  display_name: 'Editors',
  provider: 'local_managed',
  member_count: 1,
} as AuthGroup;
const renderModal = (options: Partial<React.ComponentProps<typeof AccessLevelsModal>> = {}) => {
  const props = {
    open: true,
    onClose: vi.fn(),
    authGroups: [authGroup],
    toast,
    ...options,
  };
  return {
    ...render(
      <>
        <button type="button">Launcher</button>
        <AccessLevelsModal {...props} />
      </>,
    ),
    props,
  };
};

beforeEach(() => {
  vi.restoreAllMocks();
  toast.success.mockClear();
  toast.error.mockClear();
  vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue(config());
  vi.spyOn(contentProtectionApi, 'preview').mockResolvedValue({
    prompt_fragment: 'fragment',
  } as never);
  vi.spyOn(contentProtectionApi, 'saveConfig').mockImplementation(async (_revision, next) => ({
    ...next,
    revision: next.revision + 1,
  }));
});
afterEach(cleanup);

describe('AccessLevelsModal', () => {
  it('loads levels, retries a load error, and ignores a stale request after close', async () => {
    const getConfig = vi
      .spyOn(contentProtectionApi, 'getConfig')
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValue(config(2));
    const { rerender } = renderModal();
    expect(await screen.findByRole('alert')).toBeTruthy();
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect((await screen.findAllByText('Standard')).length).toBeGreaterThan(0);
    let resolveOld: (value: ContentProtectionConfig) => void = () => undefined;
    getConfig.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveOld = resolve;
        }),
    );
    rerender(<AccessLevelsModal open={false} onClose={vi.fn()} authGroups={[]} toast={toast} />);
    rerender(<AccessLevelsModal open onClose={vi.fn()} authGroups={[]} toast={toast} />);
    resolveOld(config(99));
    await waitFor(() => expect(screen.queryByText('Standard')).toBeTruthy());
  });

  it('guards dirty selection changes and removes a discarded new draft', async () => {
    vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue({
      ...config(),
      access_levels: [level(), level('other', 'Other')],
    });
    renderModal({ authGroups: [authGroup] });
    await userEvent.click(await screen.findByText('Standard'));
    await userEvent.type(screen.getByLabelText('Name'), ' changed');
    await userEvent.click(screen.getByText('Other'));
    expect(screen.getByRole('alertdialog')).toBeTruthy();
    await userEvent.click(screen.getByText('Keep editing'));
    expect(screen.getByDisplayValue('Standard changed')).toBeTruthy();
    await userEvent.click(screen.getByText('Other'));
    await userEvent.click(screen.getByText('Discard and continue'));
    expect(await screen.findByDisplayValue('Other')).toBeTruthy();
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    await userEvent.click(screen.getByLabelText('Close access editor'));
    await userEvent.click(screen.getByText('Discard and continue'));
    expect(screen.queryByText('(new level)')).toBeNull();
  });

  it('inserts a new level', async () => {
    const save = vi.spyOn(contentProtectionApi, 'saveConfig');
    renderModal();
    await screen.findByText('Standard');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    await userEvent.type(screen.getByLabelText('Name'), 'New level');
    await userEvent.click(screen.getByText('Save'));
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith(
        1,
        expect.objectContaining({
          access_levels: expect.arrayContaining([expect.objectContaining({ name: 'New level' })]),
        }),
      ),
    );
  });

  it('renders unsaved mapping and default safeguards', async () => {
    renderModal();
    await screen.findByText('Standard');
    await userEvent.click(screen.getByRole('button', { name: /new access level/i }));
    expect(screen.getByText('Save this level before mapping groups.')).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Delete level' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('filters rail rows and closes the modal with Escape', async () => {
    const { props } = renderModal();
    await screen.findByText('Standard');
    await userEvent.type(screen.getByLabelText('Search access levels'), 'missing');
    expect(screen.getByText('No matching access levels.')).toBeTruthy();
    await userEvent.keyboard('{Escape}');
    await userEvent.keyboard('{Escape}');
    await waitFor(() => expect(props.onClose).toHaveBeenCalled());
  });

  it('shows the default-level delete guard', async () => {
    renderModal();
    await userEvent.click(await screen.findByText('Standard'));
    expect(
      (screen.getByRole('button', { name: 'Delete level' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    vi.spyOn(contentProtectionApi, 'getConfig').mockResolvedValue({
      ...config(),
      default_access_level_id: '',
      group_access_levels: [],
    });
  });
});
