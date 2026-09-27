import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ContentProtectionConfig } from '@/api/contentProtection';
import type { User } from '@/types';
import { UsersPanel } from './UsersPanel';

const apiMock = vi.hoisted(() => ({
  listUsers: vi.fn(),
  getUser: vi.fn(),
  listAuthGroups: vi.fn(),
  listUserSpaceWorkspaces: vi.fn(),
  updateUserGenerationPolicy: vi.fn(),
  updateUserRole: vi.fn(),
  setUserGroups: vi.fn(),
  listConversationSummaries: vi.fn(),
  getUsageSummary: vi.fn(),
  getUsageProviders: vi.fn(),
  getUsageDaily: vi.fn(),
  getUsageApi: vi.fn(),
  getUsageMcp: vi.fn(),
  getUsageRange: vi.fn(),
  getUsageUsersDaily: vi.fn(),
}));
const contentProtectionMock = vi.hoisted(() => ({
  getConfig: vi.fn(),
  updateContentProtectionConfigSlice: vi.fn(),
}));
vi.mock('@/api', () => ({ api: apiMock }));
vi.mock('@/api/contentProtection', async () => ({
  ...(await vi.importActual<typeof import('@/api/contentProtection')>('@/api/contentProtection')),
  contentProtectionApi: { getConfig: contentProtectionMock.getConfig },
  updateContentProtectionConfigSlice: contentProtectionMock.updateContentProtectionConfigSlice,
}));
vi.mock('@/theme', () => ({ subscribeToThemeChanges: () => () => {} }));
vi.mock('react-chartjs-2', () => ({ Bar: () => null, Chart: () => null, Line: () => null }));
vi.mock('./shared/AuthAdminModals', () => ({
  AuthAdminModalHost: ({ manageGroupsOpen }: { manageGroupsOpen: boolean }) =>
    manageGroupsOpen ? <div>Manage Groups</div> : null,
}));

const config = (): ContentProtectionConfig => ({
  revision: 1,
  enabled: true,
  classifier_model: null,
  coverage_mode: 'selected_scopes',
  profiles: [],
  group_profiles: [],
  requirements: [],
  user_overrides: [],
});
const user = (id: string, provider: User['auth_provider'] = 'local_managed') =>
  ({
    id,
    username: id === 'user-1' ? 'alex' : 'sam',
    display_name: id === 'user-1' ? 'Alex' : 'Sam',
    email: `${id}@example.test`,
    role: 'admin',
    auth_provider: provider,
    chat_enabled: null,
    userspace_generation_enabled: null,
    mfa_enabled: false,
    mfa_required: false,
  }) as User;

describe('UsersPanel directory and modal', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.listUsers.mockResolvedValue([user('user-1'), user('user-2', 'ldap')]);
    apiMock.getUser.mockResolvedValue(user('user-1'));
    apiMock.listAuthGroups.mockResolvedValue([]);
    apiMock.listUserSpaceWorkspaces.mockResolvedValue({ items: [], total: 0 });
    apiMock.listConversationSummaries.mockResolvedValue([]);
    apiMock.getUsageSummary.mockResolvedValue({ users: [] });
    apiMock.getUsageProviders.mockResolvedValue({ providers: [] });
    apiMock.getUsageDaily.mockResolvedValue({ daily: [] });
    apiMock.getUsageApi.mockResolvedValue({ daily: [] });
    apiMock.getUsageMcp.mockResolvedValue({ users: [], daily: [], routes: [] });
    apiMock.getUsageRange.mockResolvedValue({ earliest_date: '2025-01-01' });
    apiMock.getUsageUsersDaily.mockResolvedValue({ series: [] });
    contentProtectionMock.getConfig.mockResolvedValue(config());
  });
  afterEach(cleanup);

  it('keeps directory toolbar available for filtered empty results and opens the unified modal', async () => {
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);
    await screen.findByText('Alex');
    fireEvent.change(screen.getByLabelText('Search users'), { target: { value: 'nobody' } });
    expect(screen.getByText('No users match these filters.')).toBeTruthy();
    expect(screen.getByRole('button', { name: /create internal user/i })).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Search users'), { target: { value: 'alex' } });
    fireEvent.click(screen.getByRole('button', { name: 'Manage' }));
    expect(screen.getByRole('dialog', { name: 'Alex' })).toBeTruthy();
    expect(screen.getByRole('tab', { name: 'Policies' })).toBeTruthy();
  });
  it('filters Internal users as local_managed, not the local admin account', async () => {
    apiMock.listUsers.mockResolvedValue([user('user-1', 'local_managed'), user('user-2', 'local')]);
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);
    await screen.findByText('Alex');
    fireEvent.change(screen.getByLabelText('Provider filter'), {
      target: { value: 'local_managed' },
    });
    expect(screen.getByText('Alex')).toBeTruthy();
    expect(screen.queryByText('Sam')).toBeNull();
  });
  it('keeps resource counts unknown until chats load and after a workspace load failure', async () => {
    apiMock.listUserSpaceWorkspaces.mockRejectedValueOnce(new Error('offline'));
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);
    await screen.findByText('Alex');
    expect(screen.getAllByText('Resources unknown')).toHaveLength(2);
  });
  it('keeps default management ordering independent of usage values', async () => {
    const zeta = { ...user('user-1'), username: 'zeta', display_name: 'Same' };
    const alpha = { ...user('user-2'), username: 'alpha', display_name: 'Same' };
    apiMock.listUsers.mockResolvedValue([zeta, alpha]);
    apiMock.getUsageSummary.mockResolvedValue({
      users: [
        {
          user_id: zeta.id,
          username: zeta.username,
          display_name: zeta.display_name,
          total_requests: 99,
          total_input_tokens: 0,
          total_output_tokens: 0,
          total_tokens: 0,
          completed_count: 99,
          failed_count: 0,
          interrupted_count: 0,
        },
        {
          user_id: alpha.id,
          username: alpha.username,
          display_name: alpha.display_name,
          total_requests: 1,
          total_input_tokens: 0,
          total_output_tokens: 0,
          total_tokens: 0,
          completed_count: 1,
          failed_count: 0,
          interrupted_count: 0,
        },
      ],
    });
    render(<UsersPanel currentUser={zeta} onOpenWorkspace={vi.fn()} />);
    await screen.findAllByText('Same');
    fireEvent.click(screen.getByRole('button', { name: 'Usage' }));
    await screen.findByText(/Requests \(30d\)/);
    fireEvent.click(screen.getByRole('button', { name: 'Users' }));
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows.map((row) => row.getAttribute('data-user-directory-row'))).toEqual([
      alpha.id,
      zeta.id,
    ]);
  });
  it('sorts the directory from keyboard-accessible visible header controls', async () => {
    const alex = { ...user('user-1'), mfa_enabled: false, mfa_required: false };
    const sam = { ...user('user-2'), mfa_enabled: true, mfa_required: false };
    const charlie = {
      ...user('user-3'),
      username: 'charlie',
      display_name: 'Charlie',
      mfa_enabled: false,
      mfa_required: true,
    };
    apiMock.listUsers.mockResolvedValue([alex, sam, charlie]);
    render(<UsersPanel currentUser={alex} onOpenWorkspace={vi.fn()} />);
    await screen.findByText('Charlie');
    expect(screen.getByRole('button', { name: 'Sort by User' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Sort by Role' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Sort by Resources' })).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Sort by MFA' }));
    const rows = screen.getAllByRole('row').slice(1);
    expect(rows.map((row) => row.getAttribute('data-user-directory-row'))).toEqual([
      sam.id,
      charlie.id,
      alex.id,
    ]);
  });

  it('preserves the manage-groups and user-policies hash targets', async () => {
    const original = window.location.hash;
    window.location.hash = '#manage-groups';
    const { unmount } = render(
      <UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />,
    );
    await screen.findByText('Manage Groups');
    unmount();
    window.location.hash = '#user-policies';
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);
    await screen.findByText('Alex');
    expect(document.getElementById('user-policies')).toBeTruthy();
    window.location.hash = original;
  });

  it('reports content protection load failure and retries from the modal', async () => {
    contentProtectionMock.getConfig
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(config());
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);
    await screen.findByText('Alex');
    fireEvent.click(screen.getAllByRole('button', { name: 'Manage' })[0]);
    fireEvent.click(screen.getByRole('tab', { name: 'Policies' }));
    expect(screen.getByText(/Content protection settings could not be loaded/)).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await waitFor(() => expect(contentProtectionMock.getConfig).toHaveBeenCalledTimes(2));
  });
});
