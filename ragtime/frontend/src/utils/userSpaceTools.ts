import { api } from '@/api';
import type { UserSpaceAvailableTool } from '@/types';

export interface UserSpaceToolGroupInfo {
  id: string;
  name: string;
}

export interface UserSpaceToolCatalog {
  availableTools: UserSpaceAvailableTool[];
  toolGroups: UserSpaceToolGroupInfo[];
  toolsError: unknown | null;
  toolGroupsError: unknown | null;
}

export async function fetchUserSpaceToolCatalog(): Promise<UserSpaceToolCatalog> {
  const [toolsResult, groupsResult] = await Promise.allSettled([
    api.listUserSpaceAvailableTools(),
    api.listUserSpaceToolGroups(),
  ]);

  return {
    availableTools: toolsResult.status === 'fulfilled' ? toolsResult.value : [],
    toolGroups: groupsResult.status === 'fulfilled'
      ? groupsResult.value.map((group) => ({ id: group.id, name: group.name }))
      : [],
    toolsError: toolsResult.status === 'rejected' ? toolsResult.reason : null,
    toolGroupsError: groupsResult.status === 'rejected' ? groupsResult.reason : null,
  };
}

export function getUserSpaceGroupToolIds(tools: UserSpaceAvailableTool[], groupId: string): string[] {
  return tools
    .filter((tool) => tool.group_id === groupId)
    .map((tool) => tool.id);
}