import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { ContentProtectionConfig } from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { AuthAdminModalHost } from './AuthAdminModals';

const { getConfig, updateContentProtectionConfigSlice } = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: { listAuthGroups: vi.fn().mockResolvedValue([]) },
}));

vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getConfig },
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
  classifier_model: null,
  coverage_mode: 'selected_scopes',
  profiles: [{ id: 'profile-1', name: 'Internal', level: 1, scope: 'group' }],
  group_profiles: [],
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

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('AuthAdminModalHost content protection controls', () => {
  it('renders per-group classification and profile controls without the old settings anchor', async () => {
    getConfig.mockResolvedValue(CONFIG);

    renderModal();

    expect((await screen.findByLabelText('Classification')) as HTMLSelectElement).toHaveProperty(
      'value',
      'inherit',
    );
    expect(screen.getByLabelText('Profile') as HTMLSelectElement).toHaveProperty('value', '');
    expect(screen.getByRole('option', { name: 'Internal' })).toBeTruthy();
    expect(document.querySelector('#content-protection-groups-tab')).toBeNull();
  });

  it('saves classification and profile changes immediately', async () => {
    const user = userEvent.setup();
    getConfig.mockResolvedValue(CONFIG);
    updateContentProtectionConfigSlice.mockImplementation(async (mutate) => mutate(CONFIG));
    const toast = renderModal();

    await user.selectOptions(await screen.findByLabelText('Classification'), 'require');
    await waitFor(() => expect(updateContentProtectionConfigSlice).toHaveBeenCalledTimes(1));
    expect(updateContentProtectionConfigSlice.mock.calls[0][0](CONFIG).requirements).toEqual([
      { scope_kind: 'group', scope_key: 'group-1', mode: 'require' },
    ]);

    await user.selectOptions(screen.getByLabelText('Profile'), 'profile-1');
    await waitFor(() => expect(updateContentProtectionConfigSlice).toHaveBeenCalledTimes(2));
    expect(updateContentProtectionConfigSlice.mock.calls[1][0](CONFIG).group_profiles).toEqual([
      { group_id: 'group-1', profile_id: 'profile-1' },
    ]);
    expect(toast.success).toHaveBeenCalledTimes(2);
  });

  it('hides controls when protection is disabled', async () => {
    getConfig.mockResolvedValue({
      ...CONFIG,
      enabled: false,
    });

    renderModal();

    await screen.findByText('Engineering');
    expect(screen.queryByLabelText('Classification')).toBeNull();
    expect(screen.queryByLabelText('Profile')).toBeNull();
    expect(updateContentProtectionConfigSlice).not.toHaveBeenCalled();
  });

  it('keeps classification editable and annotates all-traffic coverage', async () => {
    getConfig.mockResolvedValue({ ...CONFIG, coverage_mode: 'all_supported_traffic' });

    renderModal();

    expect((await screen.findByLabelText('Classification')) as HTMLSelectElement).toHaveProperty(
      'disabled',
      false,
    );
    expect(
      screen.getByText(
        'Coverage is currently All supported traffic; scope requirements apply when coverage is Selected scopes.',
      ),
    ).toBeTruthy();
  });
});
