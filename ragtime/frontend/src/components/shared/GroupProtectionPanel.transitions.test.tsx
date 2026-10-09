import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ContentProtectionApiError, type ContentProtectionConfig } from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { GroupProtectionPanel } from './GroupProtectionPanel';

const api = vi.hoisted(() => ({ getConfig: vi.fn(), saveConfig: vi.fn(), preview: vi.fn() }));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: api,
}));
const GROUP: AuthGroup = {
  id: 'finance',
  key: 'finance',
  display_name: 'Finance',
  description: '',
  provider: 'local_managed',
  role: null,
  member_count: 2,
  manual_member_count: 2,
  ldap_member_count: 0,
  member_previews: [],
  is_logon_group: false,
};
const DEFAULT_LEVEL = {
  id: 'default',
  name: 'Default',
  granted_category_ids: [],
  guidance: 'Default guidance',
};
const FINANCE_LEVEL = {
  id: 'finance-level',
  name: 'Finance restricted',
  granted_category_ids: ['operational'],
  guidance: 'Finance guidance',
};
const SECOND_LEVEL = {
  id: 'second-level',
  name: 'Second level',
  granted_category_ids: [],
  guidance: 'Second guidance',
};
function makeConfig(overrides: Partial<ContentProtectionConfig> = {}): ContentProtectionConfig {
  return {
    schema_version: 2,
    revision: 1,
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
    ],
    access_levels: [DEFAULT_LEVEL, FINANCE_LEVEL, SECOND_LEVEL],
    group_access_levels: [],
    default_access_level_id: DEFAULT_LEVEL.id,
    coverage_mode: 'selected_scopes',
    requirements: [],
    user_overrides: [],
    ...overrides,
  };
}
interface Control {
  setConfig?: Dispatch<SetStateAction<ContentProtectionConfig>>;
}
function StatefulPanel({
  initialConfig,
  control,
  updateMapping = async () => undefined,
}: {
  initialConfig: ContentProtectionConfig;
  control?: Control;
  updateMapping?: (id: string, enabled: boolean) => Promise<void>;
}) {
  const [config, setConfig] = useState(initialConfig);
  useEffect(() => {
    if (control) control.setConfig = setConfig;
  }, [control]);
  const handleMapping = async (levelId: string, enabled: boolean) => {
    await updateMapping(levelId, enabled);
    setConfig((current) => ({
      ...current,
      revision: current.revision + 1,
      group_access_levels: enabled
        ? [
            ...current.group_access_levels.filter(
              (mapping) => mapping.group_id !== GROUP.id || mapping.access_level_id !== levelId,
            ),
            { group_id: GROUP.id, access_level_id: levelId },
          ]
        : current.group_access_levels.filter(
            (mapping) => mapping.group_id !== GROUP.id || mapping.access_level_id !== levelId,
          ),
    }));
  };
  return (
    <GroupProtectionPanel
      group={GROUP}
      authGroups={[GROUP]}
      config={config}
      onConfigSaved={setConfig}
      onUpdateMapping={handleMapping}
      onUpdateRequirement={async () => undefined}
      onBack={vi.fn()}
      toast={{ success: vi.fn(), error: vi.fn() }}
    />
  );
}
function editor(container: HTMLElement, id: string): HTMLElement {
  const result = container.querySelector<HTMLElement>(`[data-group-shared-level="${id}"]`);
  if (!result) throw new Error(`Missing editor for ${id}`);
  return result;
}
async function rename(target: HTMLElement, name: string) {
  const user = userEvent.setup();
  const input = within(target).getByLabelText('Name');
  await user.clear(input);
  await user.type(input, name);
}
function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
beforeEach(() => {
  api.preview.mockResolvedValue({ prompt_fragment: 'preview' });
});
afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('GroupProtectionPanel state transitions', () => {
  it('confirms before the first mapping hides a dirty default-level editor', async () => {
    const user = userEvent.setup();
    const updateMapping = vi.fn().mockResolvedValue(undefined);
    const { container } = render(
      <StatefulPanel initialConfig={makeConfig()} updateMapping={updateMapping} />,
    );
    await rename(editor(container, DEFAULT_LEVEL.id), 'Dirty default');
    const choice = within(screen.getByRole('group', { name: 'Access levels' })).getByLabelText(
      /Finance restricted/,
    );
    await user.click(choice);
    expect(updateMapping).not.toHaveBeenCalled();
    expect(choice).toHaveProperty('checked', false);
    const confirmation = await screen.findByRole('alertdialog');
    await user.click(within(confirmation).getByRole('button', { name: /^Discard/i }));
    await waitFor(() => expect(updateMapping).toHaveBeenCalledWith(FINANCE_LEVEL.id, true));
    expect(updateMapping).toHaveBeenCalledTimes(1);
    await waitFor(() =>
      expect(container.querySelector('[data-group-shared-level="default"]')).toBeNull(),
    );
  });

  it('discards a dirty unmapped draft and performs the network mutation exactly once', async () => {
    const user = userEvent.setup();
    const network = deferred<void>();
    const updateMapping = vi.fn(() => network.promise);
    const initialConfig = makeConfig({
      group_access_levels: [{ group_id: GROUP.id, access_level_id: FINANCE_LEVEL.id }],
    });
    const { container } = render(
      <StatefulPanel initialConfig={initialConfig} updateMapping={updateMapping} />,
    );
    await rename(editor(container, FINANCE_LEVEL.id), 'Dirty finance');
    const choice = within(screen.getByRole('group', { name: 'Access levels' })).getByLabelText(
      /Finance restricted/,
    );
    await user.click(choice);
    expect(updateMapping).not.toHaveBeenCalled();
    const confirmation = await screen.findByRole('alertdialog', { name: /confirm unmap/i });
    await user.click(within(confirmation).getByRole('button', { name: /discard and unmap/i }));
    expect(updateMapping).toHaveBeenCalledTimes(1);
    expect(updateMapping).toHaveBeenCalledWith(FINANCE_LEVEL.id, false);
    await act(async () => network.resolve());
    await waitFor(() => expect(choice).toHaveProperty('checked', false));
    expect(updateMapping).toHaveBeenCalledTimes(1);
  });

  it('uses the loaded-latest config as the base for the next edited save', async () => {
    const user = userEvent.setup();
    const initial = makeConfig({
      group_access_levels: [{ group_id: GROUP.id, access_level_id: FINANCE_LEVEL.id }],
    });
    const latest = makeConfig({
      revision: 2,
      strictness: 'permissive',
      group_access_levels: initial.group_access_levels,
      access_levels: initial.access_levels.map((level) =>
        level.id === FINANCE_LEVEL.id ? { ...level, name: 'Server finance' } : level,
      ),
    });
    api.saveConfig
      .mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409))
      .mockResolvedValueOnce({
        ...latest,
        revision: 3,
        access_levels: latest.access_levels.map((level) =>
          level.id === FINANCE_LEVEL.id ? { ...level, name: 'Edited latest' } : level,
        ),
      });
    api.getConfig.mockResolvedValueOnce(latest);
    const { container } = render(<StatefulPanel initialConfig={initial} />);
    await rename(editor(container, FINANCE_LEVEL.id), 'First draft');
    await user.click(
      within(editor(container, FINANCE_LEVEL.id)).getByRole('button', { name: 'Save' }),
    );
    const conflict = await screen.findByText(/updated by another admin/i);
    await user.click(
      within(conflict.parentElement as HTMLElement).getByRole('button', { name: 'Load latest' }),
    );
    expect(within(editor(container, FINANCE_LEVEL.id)).getByLabelText('Name')).toHaveProperty(
      'value',
      'Server finance',
    );
    await rename(editor(container, FINANCE_LEVEL.id), 'Edited latest');
    await user.click(
      within(editor(container, FINANCE_LEVEL.id)).getByRole('button', { name: 'Save' }),
    );
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(2));
    expect(api.saveConfig.mock.calls[1][0]).toBe(2);
    expect(api.saveConfig.mock.calls[1][1].strictness).toBe('permissive');
  });

  it('retains a dirty draft as read-only when a config refresh deletes its level', async () => {
    const control: Control = {};
    const initial = makeConfig({
      group_access_levels: [{ group_id: GROUP.id, access_level_id: FINANCE_LEVEL.id }],
    });
    const { container } = render(<StatefulPanel initialConfig={initial} control={control} />);
    await rename(editor(container, FINANCE_LEVEL.id), 'Retained finance draft');
    await act(async () =>
      control.setConfig?.((current) => ({
        ...current,
        revision: 2,
        access_levels: current.access_levels.filter((level) => level.id !== FINANCE_LEVEL.id),
        group_access_levels: [],
      })),
    );
    const retained = editor(container, FINANCE_LEVEL.id);
    expect(within(retained).getByLabelText('Name')).toHaveProperty(
      'value',
      'Retained finance draft',
    );
    expect(within(retained).getByLabelText('Name')).toHaveProperty('disabled', true);
    expect(within(retained).getByText(/deleted by another admin/i)).toBeTruthy();
  });

  it('clears an inline save error after a later successful save', async () => {
    const user = userEvent.setup();
    const initial = makeConfig({
      group_access_levels: [{ group_id: GROUP.id, access_level_id: FINANCE_LEVEL.id }],
    });
    const saved = makeConfig({
      revision: 2,
      group_access_levels: initial.group_access_levels,
      access_levels: initial.access_levels.map((level) =>
        level.id === FINANCE_LEVEL.id ? { ...level, name: 'Retry succeeds' } : level,
      ),
    });
    api.saveConfig
      .mockRejectedValueOnce(new Error('temporary save failure'))
      .mockResolvedValueOnce(saved);
    const { container } = render(<StatefulPanel initialConfig={initial} />);
    const target = editor(container, FINANCE_LEVEL.id);
    await rename(target, 'Retry succeeds');
    await user.click(within(target).getByRole('button', { name: 'Save' }));
    expect(await within(target).findByText('temporary save failure')).toBeTruthy();
    await user.click(within(target).getByRole('button', { name: 'Save' }));
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(2));
    await waitFor(() =>
      expect(
        within(editor(container, FINANCE_LEVEL.id)).queryByText('temporary save failure'),
      ).toBeNull(),
    );
  });

  it('preserves another level draft and its original conflict base across save and refresh', async () => {
    const user = userEvent.setup();
    const control: Control = {};
    const mappings = [
      { group_id: GROUP.id, access_level_id: FINANCE_LEVEL.id },
      { group_id: GROUP.id, access_level_id: SECOND_LEVEL.id },
    ];
    const initial = makeConfig({ group_access_levels: mappings });
    const afterSecond = makeConfig({
      revision: 2,
      group_access_levels: mappings,
      access_levels: initial.access_levels.map((level) =>
        level.id === SECOND_LEVEL.id ? { ...level, name: 'Second saved' } : level,
      ),
    });
    api.saveConfig.mockResolvedValueOnce(afterSecond);
    const { container } = render(<StatefulPanel initialConfig={initial} control={control} />);
    await rename(editor(container, FINANCE_LEVEL.id), 'Finance draft survives');
    await rename(editor(container, SECOND_LEVEL.id), 'Second saved');
    await user.click(
      within(editor(container, SECOND_LEVEL.id)).getByRole('button', { name: 'Save' }),
    );
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(1));
    expect(within(editor(container, FINANCE_LEVEL.id)).getByLabelText('Name')).toHaveProperty(
      'value',
      'Finance draft survives',
    );
    await act(async () =>
      control.setConfig?.((current) => ({ ...current, revision: 3, strictness: 'balanced' })),
    );
    const latest = makeConfig({
      revision: 4,
      strictness: 'balanced',
      group_access_levels: mappings,
      access_levels: afterSecond.access_levels.map((level) =>
        level.id === FINANCE_LEVEL.id
          ? { ...level, guidance: 'Concurrent server guidance' }
          : level,
      ),
    });
    api.saveConfig.mockRejectedValueOnce(new ContentProtectionApiError('conflict', 409));
    api.getConfig.mockResolvedValueOnce(latest);
    await user.click(
      within(editor(container, FINANCE_LEVEL.id)).getByRole('button', { name: 'Save' }),
    );
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(2));
    expect(api.saveConfig.mock.calls[1][0]).toBe(1);
    expect(await screen.findByText(/updated by another admin/i)).toBeTruthy();
    expect(within(editor(container, FINANCE_LEVEL.id)).getByLabelText('Name')).toHaveProperty(
      'value',
      'Finance draft survives',
    );
  });

  it('overwrites against a freshly fetched CAS base and keeps a second 409 visible', async () => {
    const user = userEvent.setup();
    const initial = makeConfig({
      group_access_levels: [{ group_id: GROUP.id, access_level_id: FINANCE_LEVEL.id }],
    });
    const conflictConfig = makeConfig({
      revision: 2,
      strictness: 'balanced',
      group_access_levels: initial.group_access_levels,
      access_levels: initial.access_levels.map((level) =>
        level.id === FINANCE_LEVEL.id ? { ...level, guidance: 'Concurrent edit' } : level,
      ),
    });
    const overwriteBase = { ...conflictConfig, revision: 3, strictness: 'permissive' as const };
    api.saveConfig
      .mockRejectedValueOnce(new ContentProtectionApiError('first conflict', 409))
      .mockRejectedValueOnce(new ContentProtectionApiError('second conflict', 409));
    api.getConfig.mockResolvedValueOnce(conflictConfig).mockResolvedValueOnce(overwriteBase);
    const { container } = render(<StatefulPanel initialConfig={initial} />);
    await rename(editor(container, FINANCE_LEVEL.id), 'Force my draft');
    await user.click(
      within(editor(container, FINANCE_LEVEL.id)).getByRole('button', { name: 'Save' }),
    );
    const conflict = await screen.findByText(/updated by another admin/i);
    await user.click(
      within(conflict.parentElement as HTMLElement).getByRole('button', {
        name: 'Overwrite latest',
      }),
    );
    await waitFor(() => expect(api.saveConfig).toHaveBeenCalledTimes(2));
    expect(api.getConfig).toHaveBeenCalledTimes(2);
    expect(api.saveConfig.mock.calls[1][0]).toBe(3);
    expect(api.saveConfig.mock.calls[1][1].strictness).toBe('permissive');
    expect(
      api.saveConfig.mock.calls[1][1].access_levels.find(
        (level: { id: string }) => level.id === FINANCE_LEVEL.id,
      ).name,
    ).toBe('Force my draft');
    expect(await screen.findByText('second conflict')).toBeTruthy();
    expect(screen.getByText(/updated by another admin/i)).toBeTruthy();
  });
});
