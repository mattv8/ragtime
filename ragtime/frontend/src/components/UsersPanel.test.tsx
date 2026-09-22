import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ContentProtectionConfig } from '@/api/contentProtection';
import type { User } from '@/types';
import { UsersPanel } from './UsersPanel';

const apiMock = vi.hoisted(() => ({
  listUsers: vi.fn(),
  listAuthGroups: vi.fn(),
  listUserSpaceWorkspaces: vi.fn(),
  updateUserGenerationPolicy: vi.fn(),
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
vi.mock('./shared/AuthAdminModals', () => ({
  AuthAdminModalHost: ({ manageGroupsOpen }: { manageGroupsOpen: boolean }) =>
    manageGroupsOpen ? <div>Manage Groups</div> : null,
}));

const config = (enabled = true): ContentProtectionConfig => ({
  revision: 1,
  enabled,
  classifier_model: null,
  coverage_mode: 'selected_scopes',
  profiles: [],
  group_profiles: [],
  requirements: [],
  user_overrides: [{ user_id: 'user-1', mode: 'always_classify' }],
});

const user = (
  id: string,
  chatEnabled: boolean | null = null,
  userspaceGenerationEnabled: boolean | null = null,
  chatEnabledEffective?: boolean,
  userspaceGenerationEnabledEffective?: boolean,
) =>
  ({
    id,
    username: id === 'user-1' ? 'alex' : 'sam',
    display_name: id === 'user-1' ? 'Alex' : 'Sam',
    role: 'admin',
    auth_provider: 'local',
    chat_enabled: chatEnabled,
    userspace_generation_enabled: userspaceGenerationEnabled,
    chat_enabled_effective: chatEnabledEffective,
    userspace_generation_enabled_effective: userspaceGenerationEnabledEffective,
    mfa_enabled: false,
    mfa_required: false,
  }) as User;

describe('UsersPanel user policies', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.listUsers.mockResolvedValue([user('user-1'), user('user-2', true, false)]);
    apiMock.listAuthGroups.mockResolvedValue([]);
    apiMock.listUserSpaceWorkspaces.mockResolvedValue({ items: [], total: 0 });
    contentProtectionMock.getConfig.mockResolvedValue(config());
    contentProtectionMock.updateContentProtectionConfigSlice.mockResolvedValue(config());
  });

  afterEach(cleanup);

  it('renders status and policies buttons for regular and self rows', async () => {
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Alex');
    await waitFor(() =>
      expect(document.querySelector('[data-user-policies-cell="user-1"]')?.textContent).toContain(
        'Chat: inherit · User Space: inherit · Protection: always',
      ),
    );
    expect(document.querySelector('[data-user-policies-cell="user-2"]')?.textContent).toContain(
      'Chat: enabled · User Space: disabled · Protection: inherit',
    );
    expect(screen.getAllByRole('button', { name: /user policies/i })).toHaveLength(2);
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[0]);
    expect(screen.queryByLabelText('Chat')).toBeNull();
    expect(screen.queryByLabelText('User Space AI generation')).toBeNull();
    expect(screen.getByLabelText('Content protection')).not.toBeNull();
  });

  it('opens group management and scrolls to user policies from recognized hashes', async () => {
    const scrollIntoView = vi.fn();
    const focus = vi.fn();
    const originalHash = window.location.hash;
    const originalScrollIntoView = HTMLElement.prototype.scrollIntoView;
    const originalFocus = HTMLElement.prototype.focus;
    HTMLElement.prototype.scrollIntoView = scrollIntoView;
    HTMLElement.prototype.focus = focus;

    window.location.hash = '#manage-groups';
    const { unmount } = render(
      <UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />,
    );
    await screen.findByText('Alex');
    expect(screen.getByText('Manage Groups')).toBeTruthy();

    unmount();
    window.location.hash = '#user-policies';
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);
    await waitFor(() => expect(scrollIntoView).toHaveBeenCalled());
    expect(document.getElementById('user-policies')).toBeTruthy();
    expect(focus).toHaveBeenCalled();

    window.location.hash = originalHash;
    HTMLElement.prototype.scrollIntoView = originalScrollIntoView;
    HTMLElement.prototype.focus = originalFocus;
  });

  it('updates Chat and User Space policies independently, including inherit resets', async () => {
    apiMock.updateUserGenerationPolicy
      .mockResolvedValueOnce(user('user-2', false, false))
      .mockResolvedValueOnce(user('user-2', false, null));
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);
    fireEvent.change(screen.getByLabelText('Chat'), { target: { value: 'disabled' } });

    await waitFor(() =>
      expect(apiMock.updateUserGenerationPolicy).toHaveBeenCalledWith('user-2', {
        chat_enabled: false,
      }),
    );
    fireEvent.change(screen.getByLabelText('User Space AI generation'), {
      target: { value: 'inherit' },
    });
    await waitFor(() =>
      expect(apiMock.updateUserGenerationPolicy).toHaveBeenLastCalledWith('user-2', {
        userspace_generation_enabled: null,
      }),
    );
  });

  it('shows backend effective policy states for inherited settings and independent global vetoes', async () => {
    apiMock.listUsers.mockResolvedValue([
      user('user-1', null, null, true, false),
      user('user-2', true, true, false, false),
    ]);
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);

    const dialog = screen.getByRole('dialog', { name: 'User policies: Sam' });
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.textContent).toContain(
      'Enabled for this user, but Disabled effectively: this instance disables Chat.',
    );
    expect(dialog.textContent).toContain(
      'Enabled for this user, but Disabled effectively: this instance disables User Space AI generation.',
    );

    fireEvent.keyDown(document, { key: 'Escape' });
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull());
  });

  it('explains inherited effective settings without inferring them from overrides', async () => {
    apiMock.listUsers.mockResolvedValue([user('user-1'), user('user-2', null, null, true, false)]);
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);
    const dialog = screen.getByRole('dialog', { name: 'User policies: Sam' });
    expect(dialog.textContent).toContain('Inherited. Effective: Enabled.');
    expect(dialog.textContent).toContain('Inherited. Effective: Disabled.');
  });

  it('saves a content-protection override through the shared slice helper', async () => {
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);
    fireEvent.change(screen.getByLabelText('Content protection'), {
      target: { value: 'never_classify' },
    });

    await waitFor(() =>
      expect(contentProtectionMock.updateContentProtectionConfigSlice).toHaveBeenCalled(),
    );
    const mutate = contentProtectionMock.updateContentProtectionConfigSlice.mock.calls[0][0];
    expect(mutate(config()).user_overrides).toEqual([
      { user_id: 'user-1', mode: 'always_classify' },
      { user_id: 'user-2', mode: 'never_classify' },
    ]);
  });

  it('hides content-protection status and self policy button when the master policy is off', async () => {
    contentProtectionMock.getConfig.mockResolvedValue(config(false));
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    expect(document.querySelector('[data-user-policies-cell="user-1"]')?.textContent).not.toContain(
      'Protection:',
    );
    expect(screen.getAllByRole('button', { name: /user policies/i })).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: /user policies/i }));
    expect(screen.getByLabelText('Chat')).not.toBeNull();
    expect(screen.getByLabelText('User Space AI generation')).not.toBeNull();
    expect(screen.queryByLabelText('Content protection')).toBeNull();
  });

  it('shows the degraded status and hides the content-protection field after config failure', async () => {
    contentProtectionMock.getConfig.mockRejectedValue(new Error('unavailable'));
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    await waitFor(() =>
      expect(
        Array.from(document.querySelectorAll('[data-user-policies-cell]')).every((cell) =>
          cell.textContent?.includes('Protection: —'),
        ),
      ).toBe(true),
    );
    fireEvent.click(screen.getByRole('button', { name: /user policies/i }));
    expect(screen.queryByLabelText('Content protection')).toBeNull();
  });
});
