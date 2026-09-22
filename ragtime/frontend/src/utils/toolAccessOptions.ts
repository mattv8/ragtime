import type { AuthGroup, User } from '@/types';
import type { ToolAccessGroupOption, ToolAccessUserOption } from '@/components/ToolAccessEditor';

type ToolAccessUser = Pick<User, 'id' | 'username' | 'display_name' | 'role'>;
type ToolAccessGroup = Pick<AuthGroup, 'id' | 'key' | 'display_name' | 'provider' | 'member_count'>;

export function toToolAccessUserOptions(users: readonly ToolAccessUser[]): ToolAccessUserOption[] {
  return users.map((user) => ({
    id: user.id,
    username: user.username,
    display_name: user.display_name,
    is_admin: user.role === 'admin',
  }));
}

export function toToolAccessGroupOptions(
  groups: readonly ToolAccessGroup[],
): ToolAccessGroupOption[] {
  return groups.map((group) => ({
    id: group.id,
    key: group.key,
    display_name: group.display_name,
    provider: group.provider,
    member_count: group.member_count,
  }));
}
