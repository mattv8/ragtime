import type { AuthGroup } from '@/types';
import type { PolicyConflictUser } from '@/api/contentProtection';

interface Props {
  conflicts: PolicyConflictUser[];
  total: number;
  truncated: boolean;
  authGroups: AuthGroup[];
}

function groupNames(groupIds: string[], authGroups: AuthGroup[]): string {
  return groupIds
    .map((id) => authGroups.find((group) => group.id === id)?.display_name ?? `Unknown (${id})`)
    .join(', ');
}

export function PolicyConflictList({ conflicts, total, truncated, authGroups }: Props) {
  return (
    <div data-conflict-user-list>
      <ul>
        {conflicts.map((conflict) => (
          <li key={conflict.user_id} data-conflict-user={conflict.user_id}>
            <strong>{conflict.display_name || conflict.username}</strong>
            <div className="field-help">
              Skip: {groupNames(conflict.skip_group_ids, authGroups)} · Require:{' '}
              {groupNames(conflict.require_group_ids, authGroups)}
            </div>
          </li>
        ))}
      </ul>
      {truncated && (
        <p className="field-help" data-conflict-truncated>
          Showing {conflicts.length} of {total}. Reload to see the latest.
        </p>
      )}
    </div>
  );
}
