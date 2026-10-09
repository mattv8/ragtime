import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ComponentProps } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ContentProtectionConfig } from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { AuthAdminModalHost } from './AuthAdminModals';

type HostProps = ComponentProps<typeof AuthAdminModalHost>;

const { getConfig, updateContentProtectionConfigSlice, deleteAuthGroup } = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
  deleteAuthGroup: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: { listAuthGroups: vi.fn().mockResolvedValue([]), deleteAuthGroup },
}));

vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getConfig, preview: vi.fn(), saveConfig: vi.fn() },
  updateContentProtectionConfigSlice,
}));

const GROUP: AuthGroup = {
  id: 'group-1',
  key: 'engineering',
  display_name: 'Engineering',
  description: '',
  provider: 'local_managed',
  role: null,
  member_count: 0,
  manual_member_count: 0,
  ldap_member_count: 0,
  member_previews: [],
  is_logon_group: false,
};

const CONFIG: ContentProtectionConfig = {
  revision: 1,
  enabled: true,
  schema_version: 2,
  share_with_assistant: false,
  classifier: { backend: 'jev', jev: { transport: 'auto', model: 'jev-latest' }, llm_model: null },
  strictness: 'strict',
  categories: [
    {
      id: 'operational',
      name: 'Operational',
      description: 'Operational information.',
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
      description: 'Policy override requests.',
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
};

function renderModal() {
  const toast = { success: vi.fn(), error: vi.fn() };
  render(
    <AuthAdminModalHost
      createUserOpen={false}
      manageGroupsOpen
      authGroups={[GROUP]}
      onAuthGroupsChange={vi.fn()}
      onCloseCreateUser={vi.fn()}
      onCloseManageGroups={vi.fn()}
      toast={toast}
    />,
  );
  return toast;
}

function hostProps(overrides: Partial<HostProps> = {}): HostProps {
  return {
    createUserOpen: false,
    manageGroupsOpen: true,
    authGroups: [GROUP],
    onAuthGroupsChange: vi.fn(),
    onCloseCreateUser: vi.fn(),
    onCloseManageGroups: vi.fn(),
    toast: { success: vi.fn(), error: vi.fn() },
    ...overrides,
  };
}

describe('Group modal child Escape handling', () => {
  it.each(['Edit Engineering name', 'Edit Engineering description', 'New Group'])(
    'cancels %s without dismissing Manage Group Memberships',
    async (action) => {
      getConfig.mockResolvedValue(CONFIG);
      const props = hostProps();
      render(<AuthAdminModalHost {...props} />);
      const user = userEvent.setup();
      await user.click(screen.getByRole('button', { name: action }));
      await user.type(screen.getByRole('textbox'), 'Draft');
      await user.keyboard('{Escape}');
      expect(props.onCloseManageGroups).not.toHaveBeenCalled();
      expect(screen.queryByRole('textbox')).toBeNull();
      expect(screen.getByRole('heading', { name: 'Manage Group Memberships' })).toBeTruthy();
    },
  );
});

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

async function openProtectionPanel(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole('button', { name: 'Edit protection for Engineering' }));
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('AuthAdminModalHost content protection controls', () => {
  it('uses a compact selected rail and moves group controls into the detail', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();

    await openProtectionPanel(user);

    const railRow = screen.getByRole('button', {
      name: 'Engineering Internal 0 members Default only',
    });
    expect(railRow.getAttribute('aria-current')).toBe('true');
    expect(railRow.hasAttribute('aria-pressed')).toBe(false);
    const controls = document.querySelector('[data-group-detail-controls]') as HTMLElement;
    expect(
      within(controls).getByRole('group', { name: 'Access mode for Engineering' }),
    ).toBeTruthy();
    expect(within(controls).getByRole('button', { name: 'Delete' })).toBeTruthy();
  });

  it('closes the detail with the header controls and restores focus to its rail row', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();

    await openProtectionPanel(user);
    await user.click(screen.getByRole('button', { name: 'Close access editor' }));

    await waitFor(() => expect(document.querySelector('[data-group-detail]')).toBeNull());
    expect(
      document.querySelector('[data-auth-group-modal-body]')?.hasAttribute('data-detail-open'),
    ).toBe(false);
    await waitFor(() =>
      expect(document.activeElement).toBe(
        screen.getByRole('button', { name: 'Edit protection for Engineering' }),
      ),
    );

    await openProtectionPanel(user);
    await user.click(screen.getByRole('button', { name: 'Close access editor' }));
    expect(
      document.querySelector('[data-auth-group-modal-body]')?.hasAttribute('data-detail-open'),
    ).toBe(false);
  });

  it('guards Close and Back when the protection panel is dirty', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();

    await openProtectionPanel(user);
    const panel = await screen.findByRole('region', {
      name: 'Protection settings for Engineering',
    });
    await user.type(within(panel).getByLabelText('Name'), ' edited');
    await user.click(screen.getByRole('button', { name: 'Close access editor' }));
    expect(await screen.findByRole('alertdialog', { name: 'Unsaved changes' })).toBeTruthy();
    expect(document.querySelector('[data-group-detail]')).toBeTruthy();
  });

  it('closes the group modal and navigates from the disabled-policy advisory', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue({ ...CONFIG, enabled: false });
    const props = hostProps({ onNavigateToSetting: vi.fn() });
    render(<AuthAdminModalHost {...props} />);

    await openProtectionPanel(user);
    await user.click(screen.getByRole('link', { name: /enable in settings/i }));
    expect(props.onCloseManageGroups).toHaveBeenCalledOnce();
    expect(props.onNavigateToSetting).toHaveBeenCalledWith('content_protection');
  });

  it('keeps the group modal open when the disabled-policy advisory guard is cancelled', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue({ ...CONFIG, enabled: false });
    const props = hostProps({ onNavigateToSetting: vi.fn() });
    render(<AuthAdminModalHost {...props} />);

    await openProtectionPanel(user);
    await user.type(screen.getByLabelText('Name'), ' edited');
    await user.click(screen.getByRole('link', { name: /enable in settings/i }));
    await user.click(screen.getByText('Keep editing'));
    expect(props.onCloseManageGroups).not.toHaveBeenCalled();
    expect(props.onNavigateToSetting).not.toHaveBeenCalled();
  });

  it('focuses the detail heading after selecting a group', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();

    await openProtectionPanel(user);

    expect(document.activeElement).toBe(document.querySelector('#group-detail-heading-group-1'));
  });

  it('renders LDAP details without local edit controls', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    render(
      <AuthAdminModalHost
        {...hostProps({
          authGroups: [
            { ...GROUP, provider: 'ldap', source_dn: 'CN=Engineering,DC=example,DC=com' },
          ],
        })}
      />,
    );

    await openProtectionPanel(user);

    expect(screen.queryByRole('button', { name: 'Edit Engineering name' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Edit Engineering description' })).toBeNull();
    expect(document.querySelector('.auth-group-detail-dn')?.getAttribute('title')).toBe(
      'CN=Engineering,DC=example,DC=com',
    );
    expect(screen.getByText('Access')).toBeTruthy();
  });

  it('handles backdrop mousedown once without closing from an inside drag', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    const props = hostProps();
    render(<AuthAdminModalHost {...props} />);
    const overlay = document.querySelector('.modal-overlay') as HTMLElement;

    fireEvent.mouseDown(overlay);
    expect(props.onCloseManageGroups).toHaveBeenCalledTimes(1);

    cleanup();
    render(<AuthAdminModalHost {...props} />);
    await openProtectionPanel(user);
    const selectedOverlay = document.querySelector('.modal-overlay') as HTMLElement;
    fireEvent.mouseDown(selectedOverlay);
    expect(document.querySelector('[data-group-detail]')).toBeNull();
    expect(props.onCloseManageGroups).toHaveBeenCalledTimes(1);

    await openProtectionPanel(user);
    const panel = await screen.findByRole('region', {
      name: 'Protection settings for Engineering',
    });
    fireEvent.mouseDown(panel);
    fireEvent.mouseUp(document.querySelector('.modal-overlay') as HTMLElement);
    expect(document.querySelector('[data-group-detail]')).toBeTruthy();
  });

  it('guards dirty detail navigation from the backdrop', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();
    await openProtectionPanel(user);
    const panel = await screen.findByRole('region', {
      name: 'Protection settings for Engineering',
    });
    await user.type(within(panel).getByLabelText('Name'), ' edited');

    fireEvent.mouseDown(document.querySelector('.modal-overlay') as HTMLElement);

    expect(await screen.findByRole('alertdialog', { name: 'Unsaved changes' })).toBeTruthy();
    expect(document.querySelector('[data-group-detail]')).toBeTruthy();
  });

  it('renders per-group classification and access-level controls without the old settings anchor', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);

    renderModal();
    await user.click(
      await screen.findByRole('button', { name: 'Edit protection for Engineering' }),
    );

    expect(
      (await screen.findByLabelText('Require classification')) as HTMLSelectElement,
    ).toHaveProperty('value', 'inherit');
    const accessLevels = screen.getByRole('group', { name: 'Access levels' });
    expect(accessLevels).toBeTruthy();
    expect(within(accessLevels).getByLabelText(/Standard/)).toHaveProperty('checked', false);
    expect(document.querySelector('#content-protection-groups-tab')).toBeNull();
  });

  it('saves classification and access-level changes immediately', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    updateContentProtectionConfigSlice.mockImplementation(async (mutate) => mutate(CONFIG));
    const toast = renderModal();
    await user.click(
      await screen.findByRole('button', { name: 'Edit protection for Engineering' }),
    );

    await user.selectOptions(await screen.findByLabelText('Require classification'), 'require');
    await waitFor(() => expect(updateContentProtectionConfigSlice).toHaveBeenCalledTimes(1));
    expect(updateContentProtectionConfigSlice.mock.calls[0][0](CONFIG).requirements).toEqual([
      { scope_kind: 'group', scope_key: 'group-1', mode: 'require' },
    ]);

    await user.click(screen.getByLabelText(/Standard/));
    await waitFor(() => expect(updateContentProtectionConfigSlice).toHaveBeenCalledTimes(2));
    expect(updateContentProtectionConfigSlice.mock.calls[1][0](CONFIG).group_access_levels).toEqual(
      [{ group_id: 'group-1', access_level_id: 'standard' }],
    );
    expect(toast.success).toHaveBeenCalledTimes(2);
  });

  it('keeps controls editable when protection is disabled', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue({
      ...CONFIG,
      enabled: false,
    });

    renderModal();
    await user.click(
      await screen.findByRole('button', { name: 'Edit protection for Engineering' }),
    );

    expect(await screen.findByLabelText('Require classification')).toBeTruthy();
    expect(screen.getByRole('group', { name: 'Access levels' })).toBeTruthy();
    expect(updateContentProtectionConfigSlice).not.toHaveBeenCalled();
  });

  it('keeps classification editable and annotates all-traffic coverage', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue({ ...CONFIG, coverage_mode: 'all_supported_traffic' });

    renderModal();
    await user.click(
      await screen.findByRole('button', { name: 'Edit protection for Engineering' }),
    );

    expect(
      (await screen.findByLabelText('Require classification')) as HTMLSelectElement,
    ).toHaveProperty('disabled', false);
  });

  it('shows group controls when classification is advisory-only', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue({ ...CONFIG, enabled: false, share_with_assistant: true });

    renderModal();
    await user.click(
      await screen.findByRole('button', { name: 'Edit protection for Engineering' }),
    );

    expect(await screen.findByLabelText('Require classification')).toBeTruthy();
    expect(screen.getByRole('group', { name: 'Access levels' })).toBeTruthy();
  });

  it('saves only the selected access level when a group has multiple levels', async () => {
    const user = userEvent.setup();
    const config = {
      ...CONFIG,
      access_levels: [
        ...CONFIG.access_levels,
        { id: 'restricted', name: 'Restricted', granted_category_ids: [], guidance: '' },
      ],
      group_access_levels: [{ group_id: 'group-1', access_level_id: 'standard' }],
    };
    getConfig.mockResolvedValue(config);
    updateContentProtectionConfigSlice.mockImplementation(async (mutate) => mutate(config));

    renderModal();
    await user.click(
      await screen.findByRole('button', { name: 'Edit protection for Engineering' }),
    );

    const accessLevels = await screen.findByRole('group', { name: 'Access levels' });
    await user.click(within(accessLevels).getByLabelText(/Restricted/));

    await waitFor(() => expect(updateContentProtectionConfigSlice).toHaveBeenCalledTimes(1));
    expect(updateContentProtectionConfigSlice.mock.calls[0][0](config).group_access_levels).toEqual(
      [
        { group_id: 'group-1', access_level_id: 'standard' },
        { group_id: 'group-1', access_level_id: 'restricted' },
      ],
    );
  });
});

describe('AuthAdminModalHost protection state guards', () => {
  it('ignores a config response from a previous open after close and reopen', async () => {
    const user = userEvent.setup();
    const stale = deferred<ContentProtectionConfig>();
    const fresh = deferred<ContentProtectionConfig>();
    getConfig.mockReturnValueOnce(stale.promise).mockReturnValueOnce(fresh.promise);
    const props = hostProps();
    const { rerender } = render(<AuthAdminModalHost {...props} />);
    rerender(<AuthAdminModalHost {...props} manageGroupsOpen={false} />);
    rerender(<AuthAdminModalHost {...props} manageGroupsOpen />);

    await act(async () => {
      fresh.resolve({
        ...CONFIG,
        group_access_levels: [{ group_id: 'group-1', access_level_id: 'standard' }],
      });
    });
    await openProtectionPanel(user);
    const accessLevels = await screen.findByRole('group', { name: 'Access levels' });
    expect(within(accessLevels).getByLabelText(/Standard/)).toHaveProperty('checked', true);

    await act(async () => {
      stale.resolve(CONFIG);
    });
    expect(within(accessLevels).getByLabelText(/Standard/)).toHaveProperty('checked', true);
  });

  it('recovers from a config load error with Retry', async () => {
    const user = userEvent.setup();
    getConfig.mockRejectedValueOnce(new Error('Config offline')).mockResolvedValueOnce(CONFIG);
    renderModal();

    await openProtectionPanel(user);
    expect(await screen.findByText(/Config offline/)).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Retry' }));

    expect(await screen.findByLabelText('Require classification')).toBeTruthy();
    expect(getConfig).toHaveBeenCalledTimes(2);
  });

  it('refuses a mapping when the target level is missing on a fresh config', async () => {
    const user = userEvent.setup();
    const config = {
      ...CONFIG,
      access_levels: [
        ...CONFIG.access_levels,
        { id: 'restricted', name: 'Restricted', granted_category_ids: [], guidance: '' },
      ],
    };
    getConfig.mockResolvedValue(config);
    // First attempt sees the level; the retried fresh config no longer has it.
    updateContentProtectionConfigSlice.mockImplementation(async (mutate) => {
      mutate(config);
      return mutate({
        ...config,
        access_levels: config.access_levels.filter((level) => level.id !== 'restricted'),
      });
    });
    const props = hostProps();
    render(<AuthAdminModalHost {...props} />);

    await openProtectionPanel(user);
    const accessLevels = await screen.findByRole('group', { name: 'Access levels' });
    await user.click(within(accessLevels).getByLabelText(/Restricted/));

    await waitFor(() =>
      expect(props.toast.error).toHaveBeenCalledWith(
        'This access level was deleted by another admin.',
      ),
    );
    expect(props.toast.success).not.toHaveBeenCalled();
  });

  it('keeps a dirty draft visible when the selected group disappears externally', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    const props = hostProps();
    const { rerender } = render(<AuthAdminModalHost {...props} />);

    await openProtectionPanel(user);
    const panel = await screen.findByRole('region', {
      name: 'Protection settings for Engineering',
    });
    await user.type(within(panel).getByLabelText('Name'), ' edited');

    rerender(<AuthAdminModalHost {...props} authGroups={[]} />);

    expect(await screen.findByText(/no longer available/)).toBeTruthy();
    expect(within(panel).getByLabelText('Name')).toHaveProperty('value', 'Standard edited');
    await user.selectOptions(within(panel).getByLabelText('Require classification'), 'require');
    expect(updateContentProtectionConfigSlice).not.toHaveBeenCalled();
    expect(props.toast.error).toHaveBeenCalledWith(expect.stringContaining('no longer available'));
  });

  it('disables group name and description edit affordances while protection saves are busy', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    updateContentProtectionConfigSlice.mockReturnValue(new Promise(() => {}));
    renderModal();

    await openProtectionPanel(user);
    await user.selectOptions(await screen.findByLabelText('Require classification'), 'require');

    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Edit Engineering name' })).toHaveProperty(
        'disabled',
        true,
      ),
    );
    expect(screen.getByRole('button', { name: 'Edit Engineering description' })).toHaveProperty(
      'disabled',
      true,
    );
  });

  it('keeps a dirty selected group and its draft when the delete guard is cancelled', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();

    await openProtectionPanel(user);
    const panel = await screen.findByRole('region', {
      name: 'Protection settings for Engineering',
    });
    await user.type(within(panel).getByLabelText('Name'), ' edited');

    const row = document.querySelector('[data-group-detail-controls]') as HTMLElement;
    await user.click(within(row).getByRole('button', { name: 'Delete' }));
    await waitFor(
      () => expect(within(row).getByRole('button', { name: 'Confirm?' })).toBeTruthy(),
      {
        timeout: 4000,
      },
    );
    await user.click(within(row).getByRole('button', { name: 'Confirm?' }));

    expect(await screen.findByRole('alertdialog', { name: 'Unsaved changes' })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Keep editing' }));

    expect(deleteAuthGroup).not.toHaveBeenCalled();
    expect(within(panel).getByLabelText('Name')).toHaveProperty('value', 'Standard edited');
  });

  it('deletes a dirty selected group once after discard and clears the selection', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    renderModal();

    await openProtectionPanel(user);
    const panel = await screen.findByRole('region', {
      name: 'Protection settings for Engineering',
    });
    await user.type(within(panel).getByLabelText('Name'), ' edited');

    const row = document.querySelector('[data-group-detail-controls]') as HTMLElement;
    await user.click(within(row).getByRole('button', { name: 'Delete' }));
    await waitFor(
      () => expect(within(row).getByRole('button', { name: 'Confirm?' })).toBeTruthy(),
      {
        timeout: 4000,
      },
    );
    await user.click(within(row).getByRole('button', { name: 'Confirm?' }));
    await user.click(await screen.findByRole('button', { name: 'Discard and continue' }));

    await waitFor(() => expect(deleteAuthGroup).toHaveBeenCalledTimes(1));
    expect(deleteAuthGroup).toHaveBeenCalledWith('group-1');
    await waitFor(() => expect(document.querySelector('[data-group-protection-panel]')).toBeNull());
  });
});
