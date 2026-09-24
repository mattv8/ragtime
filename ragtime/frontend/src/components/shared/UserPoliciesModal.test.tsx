import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { User } from '@/types';
import { UserPoliciesModal } from './UserPoliciesModal';

const user = {
  id: 'admin-1',
  username: 'admin',
  display_name: 'Admin',
  role: 'admin',
  auth_provider: 'local',
  chat_enabled: true,
  userspace_generation_enabled: null,
  chat_enabled_effective: false,
  userspace_generation_enabled_effective: false,
} as User;

describe('UserPoliciesModal generation policies', () => {
  it('allows an admin to edit their own independent generation overrides', () => {
    const onGenerationPolicyChange = vi.fn().mockResolvedValue(undefined);
    render(
      <UserPoliciesModal
        user={user}
        actionLoading={false}
        contentProtectionAvailable={false}
        contentProtectionEnabled={false}
        contentProtectionMode="inherit"
        onClose={vi.fn()}
        onGenerationPolicyChange={onGenerationPolicyChange}
        onContentProtectionConfigChange={vi.fn()}
        onSuccess={vi.fn()}
        onError={vi.fn()}
      />,
    );

    expect(screen.getAllByRole('option', { name: 'Use global default' })).toHaveLength(2);
    expect(
      screen.getByText(
        'Explicit enabled override. Effective: Disabled; the server has not enabled this policy.',
      ),
    ).toBeTruthy();
    expect(screen.getByText('Inherited. Effective: Disabled.')).toBeTruthy();

    fireEvent.change(screen.getByLabelText('Chat'), { target: { value: 'enabled' } });
    fireEvent.change(screen.getByLabelText('User Space AI generation'), {
      target: { value: 'disabled' },
    });
    expect(onGenerationPolicyChange).toHaveBeenNthCalledWith(
      1,
      'admin-1',
      'chat_enabled',
      'enabled',
    );
    expect(onGenerationPolicyChange).toHaveBeenNthCalledWith(
      2,
      'admin-1',
      'userspace_generation_enabled',
      'disabled',
    );
  });
});
