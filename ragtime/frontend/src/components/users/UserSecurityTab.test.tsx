import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';
import type { User } from '@/types';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { UserSecurityTab } from './UserSecurityTab';

const apiMock = vi.hoisted(() => ({
  getUser: vi.fn(),
  resetUserMfa: vi.fn(),
  updateLocalUser: vi.fn(),
}));
const recoveryMock = vi.hoisted(() => ({
  getRecoveryPass: vi.fn(),
  verifyAdminSecurity: vi.fn(),
  issueRecoveryPass: vi.fn(),
  revokeRecoveryPass: vi.fn(),
}));
vi.mock('@/api', () => ({ api: apiMock }));
vi.mock('@/api/userRecovery', () => ({ userRecoveryApi: recoveryMock }));

const user: User = {
  id: 'u1',
  username: 'user',
  auth_provider: 'local_managed' as const,
  role: 'user' as const,
  display_name: null,
  email: null,
  mfa_enabled: true,
};
const grant = {
  id: 'g1',
  status: 'issued' as const,
  created_at: '2026-01-01T00:00:00Z',
  expires_at: '2026-01-01T01:00:00Z',
  redeemed_at: null,
  completed_at: null,
};
const renderTab = (overrides = {}) =>
  render(
    <UserSecurityTab
      user={user}
      isSelf={false}
      onUserUpdated={vi.fn()}
      onDirtyChange={vi.fn()}
      {...overrides}
    />,
  );

describe('UserSecurityTab', () => {
  beforeEach(() => {
    vi.resetAllMocks();
    recoveryMock.getRecoveryPass.mockResolvedValue(grant);
    apiMock.getUser.mockResolvedValue(user);
    recoveryMock.verifyAdminSecurity.mockResolvedValue({ verification_token: 'fresh' });
  });
  afterEach(cleanup);

  it('loads once when a parent stores refreshed users with fresh callback identities', async () => {
    const never = new Promise(() => {});
    recoveryMock.getRecoveryPass.mockResolvedValueOnce(grant).mockImplementation(() => never);

    function Parent() {
      const [selectedUser, setSelectedUser] = useState(user);
      return (
        <UserSecurityTab
          user={selectedUser}
          isSelf={false}
          onUserUpdated={(updated) => setSelectedUser({ ...updated })}
          onDirtyChange={() => undefined}
          onBusyChange={() => undefined}
        />
      );
    }

    render(<Parent />);

    await screen.findByRole('button', { name: 'Reissue recovery pass' });
    await waitFor(() => expect(recoveryMock.getRecoveryPass).toHaveBeenCalledTimes(1));
    expect(apiMock.getUser).toHaveBeenCalledTimes(1);
  });
  it('preserves password drafts and a matching issued secret across same-user rerenders', async () => {
    recoveryMock.issueRecoveryPass.mockResolvedValue({ grant, pass: 'one-time-secret' });

    function Parent() {
      const [selectedUser, setSelectedUser] = useState(user);
      const [, setParentRender] = useState(0);
      return (
        <>
          <button type="button" onClick={() => setParentRender((value) => value + 1)}>
            Rerender parent
          </button>
          <UserSecurityTab
            user={selectedUser}
            isSelf={false}
            onUserUpdated={(updated) => setSelectedUser({ ...updated })}
            onDirtyChange={() => undefined}
            onBusyChange={() => undefined}
          />
        </>
      );
    }

    render(<Parent />);
    await screen.findByRole('button', { name: 'Reissue recovery pass' });
    fireEvent.change(screen.getByLabelText('New password'), {
      target: { value: 'draft-password' },
    });
    fireEvent.change(screen.getByLabelText('Current password'), { target: { value: 'admin' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reissue recovery pass' }));

    expect(await screen.findByLabelText('One-time pass — save it now')).toHaveProperty(
      'value',
      'one-time-secret',
    );
    expect(screen.getByLabelText('New password')).toHaveProperty('value', 'draft-password');

    fireEvent.click(screen.getByRole('button', { name: 'Rerender parent' }));
    expect(screen.getByLabelText('One-time pass — save it now')).toHaveProperty(
      'value',
      'one-time-secret',
    );
    expect(screen.getByLabelText('New password')).toHaveProperty('value', 'draft-password');
    expect(recoveryMock.getRecoveryPass).toHaveBeenCalledTimes(2);
    expect(apiMock.getUser).toHaveBeenCalledTimes(2);
  });

  it('clears an issued secret when refresh returns a different grant', async () => {
    const replacement = { ...grant, id: 'replacement' };
    recoveryMock.getRecoveryPass.mockResolvedValueOnce(grant).mockResolvedValueOnce(replacement);
    recoveryMock.issueRecoveryPass.mockResolvedValue({ grant, pass: 'stale-secret' });
    renderTab();
    await screen.findByRole('button', { name: 'Reissue recovery pass' });
    fireEvent.change(screen.getByLabelText('Current password'), { target: { value: 'admin' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reissue recovery pass' }));

    await waitFor(() => expect(recoveryMock.getRecoveryPass).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(screen.queryByLabelText('One-time pass — save it now')).toBeNull());
  });

  it('ignores a mutation result after the selected user changes', async () => {
    let resolveIssue!: (value: { grant: typeof grant; pass: string }) => void;
    recoveryMock.issueRecoveryPass.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveIssue = resolve;
        }),
    );
    const callbacks = {
      onUserUpdated: vi.fn(),
      onDirtyChange: vi.fn(),
    };
    const view = renderTab(callbacks);
    await screen.findByRole('button', { name: 'Reissue recovery pass' });
    fireEvent.change(screen.getByLabelText('Current password'), { target: { value: 'admin' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reissue recovery pass' }));
    await waitFor(() => expect(recoveryMock.issueRecoveryPass).toHaveBeenCalledWith('u1', 'fresh'));

    const nextUser = { ...user, id: 'u2', username: 'other' };
    view.rerender(<UserSecurityTab user={nextUser} isSelf={false} {...callbacks} />);
    await act(async () => resolveIssue({ grant, pass: 'wrong-user-secret' }));

    expect(screen.queryByDisplayValue('wrong-user-secret')).toBeNull();
  });

  it('does not emit a second busy completion after unmount', async () => {
    let resolveIssue!: (value: { grant: typeof grant; pass: string }) => void;
    recoveryMock.issueRecoveryPass.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveIssue = resolve;
        }),
    );
    const onBusyChange = vi.fn();
    const view = renderTab({ onBusyChange });
    await screen.findByRole('button', { name: 'Reissue recovery pass' });
    fireEvent.change(screen.getByLabelText('Current password'), { target: { value: 'admin' } });
    fireEvent.click(screen.getByRole('button', { name: 'Reissue recovery pass' }));
    await waitFor(() => expect(onBusyChange).toHaveBeenCalledWith(true));

    view.unmount();
    await act(async () => resolveIssue({ grant, pass: 'ignored' }));

    expect(onBusyChange.mock.calls).toEqual([[true], [false]]);
  });

  it('allows revoking a redeemed enrollment-pending grant', async () => {
    recoveryMock.getRecoveryPass.mockResolvedValue({ ...grant, status: 'redeemed' });
    recoveryMock.revokeRecoveryPass.mockResolvedValue(undefined);
    renderTab();
    await screen.findByRole('button', { name: 'Revoke pass' });
    fireEvent.change(screen.getByLabelText('Current password'), { target: { value: 'admin' } });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Revoke pass' }));
    });
    expect(recoveryMock.revokeRecoveryPass).toHaveBeenCalledWith('u1', 'fresh');
  });
  it('requires matching password confirmation and keeps self actions disabled', async () => {
    renderTab({ isSelf: true });
    await screen.findByText(/MFA:/);
    expect(screen.getByRole('button', { name: 'Reissue recovery pass' })).toHaveProperty(
      'disabled',
      true,
    );
    fireEvent.change(screen.getByLabelText('New password'), { target: { value: 'new' } });
    expect(screen.getByRole('button', { name: 'Reset password…' })).toHaveProperty(
      'disabled',
      true,
    );
  });
  it('offers a retry when status loading fails', async () => {
    recoveryMock.getRecoveryPass
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(grant);
    renderTab();
    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    await screen.findByRole('button', { name: 'Reissue recovery pass' });
  });
});
