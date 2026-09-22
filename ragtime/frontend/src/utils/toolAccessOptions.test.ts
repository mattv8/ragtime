import { describe, expect, it } from 'vitest';

import { toToolAccessGroupOptions, toToolAccessUserOptions } from './toolAccessOptions';
import type { AuthGroup, User } from '@/types';

type ToolAccessUserInput = Pick<User, 'id' | 'username' | 'display_name' | 'role'>;
type ToolAccessGroupInput = Pick<
  AuthGroup,
  'id' | 'key' | 'display_name' | 'provider' | 'member_count'
>;

describe('tool access option projections', () => {
  it('projects administrator status from role while preserving user identity and labels', () => {
    const users: readonly ToolAccessUserInput[] = Object.freeze([
      Object.freeze({
        id: 'user-admin',
        username: 'administrator',
        display_name: null,
        role: 'admin' as const,
      }),
      Object.freeze({
        id: 'user-named-admin',
        username: 'admin-like-user',
        display_name: '',
        role: 'user' as const,
      }),
      Object.freeze({
        id: 'user-whitespace-name',
        username: 'sloane',
        display_name: '  Sloane  ',
        role: 'user' as const,
      }),
    ]);

    const options = toToolAccessUserOptions(users);

    expect(options).toEqual([
      { id: 'user-admin', username: 'administrator', display_name: null, is_admin: true },
      { id: 'user-named-admin', username: 'admin-like-user', display_name: '', is_admin: false },
      {
        id: 'user-whitespace-name',
        username: 'sloane',
        display_name: '  Sloane  ',
        is_admin: false,
      },
    ]);
    expect(options).not.toBe(users);
    expect(options[0]).not.toBe(users[0]);
  });

  it('preserves ordered group metadata in fresh options', () => {
    const groups: readonly ToolAccessGroupInput[] = Object.freeze([
      Object.freeze({
        id: 'group-ldap',
        key: 'cn=engineering,ou=groups,dc=example,dc=com',
        display_name: 'Engineering',
        provider: 'ldap' as const,
        member_count: 0,
      }),
      Object.freeze({
        id: 'group-local',
        key: 'release-managers',
        display_name: 'Release Managers',
        provider: 'local' as const,
        member_count: 3,
      }),
    ]);

    const options = toToolAccessGroupOptions(groups);

    expect(options).toEqual([
      {
        id: 'group-ldap',
        key: 'cn=engineering,ou=groups,dc=example,dc=com',
        display_name: 'Engineering',
        provider: 'ldap',
        member_count: 0,
      },
      {
        id: 'group-local',
        key: 'release-managers',
        display_name: 'Release Managers',
        provider: 'local',
        member_count: 3,
      },
    ]);
    expect(options).not.toBe(groups);
    expect(options[0]).not.toBe(groups[0]);
  });

  it('returns a fresh empty array for empty principal lists', () => {
    const users: readonly ToolAccessUserInput[] = Object.freeze([]);
    const groups: readonly ToolAccessGroupInput[] = Object.freeze([]);

    expect(toToolAccessUserOptions(users)).toEqual([]);
    expect(toToolAccessUserOptions(users)).not.toBe(users);
    expect(toToolAccessGroupOptions(groups)).toEqual([]);
    expect(toToolAccessGroupOptions(groups)).not.toBe(groups);
  });
});
