import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { ContentProtectionConfig } from '@/api/contentProtection';
import type { User } from '@/types';
import { UsersPanel } from './UsersPanel';

const apiMock = vi.hoisted(() => ({
  listUsers: vi.fn(),
  listAuthGroups: vi.fn(),
  listUserSpaceWorkspaces: vi.fn(),
  updateUserHostedChatEnabled: vi.fn(),
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
vi.mock('./shared/AuthAdminModals', () => ({ AuthAdminModalHost: () => null }));

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

const user = (id: string, hostedChatEnabled: boolean | null = null) =>
  ({
    id,
    username: id === 'user-1' ? 'alex' : 'sam',
    display_name: id === 'user-1' ? 'Alex' : 'Sam',
    role: 'admin',
    auth_provider: 'local',
    hosted_chat_enabled: hostedChatEnabled,
    mfa_enabled: false,
    mfa_required: false,
  }) as User;

describe('UsersPanel user policies', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    apiMock.listUsers.mockResolvedValue([user('user-1'), user('user-2', true)]);
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
        'Chat: inherit · Protection: always',
      ),
    );
    expect(document.querySelector('[data-user-policies-cell="user-2"]')?.textContent).toContain(
      'Chat: enabled · Protection: inherit',
    );
    expect(screen.getAllByRole('button', { name: /user policies/i })).toHaveLength(2);
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[0]);
    expect(screen.queryByLabelText('Hosted chat')).toBeNull();
    expect(screen.getByLabelText('Content protection')).not.toBeNull();
  });

  it('opens the modal and moves hosted chat policy editing there', async () => {
    apiMock.updateUserHostedChatEnabled.mockResolvedValue(user('user-2', false));
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);
    fireEvent.change(screen.getByLabelText('Hosted chat'), { target: { value: 'disabled' } });

    await waitFor(() =>
      expect(apiMock.updateUserHostedChatEnabled).toHaveBeenCalledWith('user-2', false),
    );
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

  it('disables content-protection editing when the master policy is off', async () => {
    contentProtectionMock.getConfig.mockResolvedValue(config(false));
    render(<UsersPanel currentUser={user('user-1')} onOpenWorkspace={vi.fn()} />);

    await screen.findByText('Sam');
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);
    expect((screen.getByLabelText('Content protection') as HTMLSelectElement).disabled).toBe(true);
    expect(screen.getByText('Content protection is disabled in Settings.')).not.toBeNull();
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
    fireEvent.click(screen.getAllByRole('button', { name: /user policies/i })[1]);
    expect(screen.queryByLabelText('Content protection')).toBeNull();
  });
});
