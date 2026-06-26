import { api } from '@/api';
import type { ToolSelectionMode, UserSpaceAvailableTool } from '@/types';

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

export interface UserSpaceToolSelection {
  mode: ToolSelectionMode;
  toolIds: string[];
  toolGroupIds: string[];
}

export type UserSpaceToolGroupCheckState = 'all' | 'some' | 'none';

export async function fetchUserSpaceToolCatalog(): Promise<UserSpaceToolCatalog> {
  const [toolsResult, groupsResult] = await Promise.allSettled([
    api.listUserSpaceAvailableTools(),
    api.listUserSpaceToolGroups(),
  ]);

  return {
    availableTools: toolsResult.status === 'fulfilled' ? toolsResult.value : [],
    toolGroups:
      groupsResult.status === 'fulfilled'
        ? groupsResult.value.map((group) => ({ id: group.id, name: group.name }))
        : [],
    toolsError: toolsResult.status === 'rejected' ? toolsResult.reason : null,
    toolGroupsError: groupsResult.status === 'rejected' ? groupsResult.reason : null,
  };
}

export function getUserSpaceGroupToolIds(
  tools: UserSpaceAvailableTool[],
  groupId: string,
): string[] {
  return tools.filter((tool) => tool.group_id === groupId).map((tool) => tool.id);
}

export function isUserSpaceToolAvailable(tool: UserSpaceAvailableTool): boolean {
  return tool.available !== false;
}

export function normalizeToolSelectionMode(mode?: string | null): ToolSelectionMode {
  return mode === 'default_all' ? 'default_all' : 'custom';
}

export function getSelectableUserSpaceToolIds(tools: UserSpaceAvailableTool[]): string[] {
  return tools.filter(isUserSpaceToolAvailable).map((tool) => tool.id);
}

export function resolveDefaultSelectedToolIds(
  selectedToolIds: string[],
  selectedToolGroupIds: string[],
  availableTools: UserSpaceAvailableTool[],
  toolSelectionMode?: ToolSelectionMode | null,
): string[] {
  if (toolSelectionMode === 'default_all') {
    return getSelectableUserSpaceToolIds(availableTools);
  }
  if (
    selectedToolIds.length > 0 ||
    selectedToolGroupIds.length > 0 ||
    toolSelectionMode === 'custom'
  ) {
    return selectedToolIds;
  }
  return getSelectableUserSpaceToolIds(availableTools);
}

export function getEffectiveUserSpaceToolIdSet(
  selection: UserSpaceToolSelection,
  availableTools: UserSpaceAvailableTool[],
): Set<string> {
  if (selection.mode === 'default_all') {
    return new Set(getSelectableUserSpaceToolIds(availableTools));
  }

  const ids = new Set(selection.toolIds);
  const groupIds = new Set(selection.toolGroupIds);
  for (const tool of availableTools) {
    if (isUserSpaceToolAvailable(tool) && tool.group_id && groupIds.has(tool.group_id)) {
      ids.add(tool.id);
    }
  }
  return ids;
}

export function getUserSpaceGroupCheckState(
  selection: UserSpaceToolSelection,
  availableTools: UserSpaceAvailableTool[],
  groupId: string,
  tools: UserSpaceAvailableTool[],
): UserSpaceToolGroupCheckState {
  const selectableTools = tools.filter(isUserSpaceToolAvailable);
  if (selectableTools.length === 0) return 'none';
  if (selection.mode === 'default_all' || selection.toolGroupIds.includes(groupId)) return 'all';
  const effectiveIds = getEffectiveUserSpaceToolIdSet(selection, availableTools);
  const selectedCount = selectableTools.filter((tool) => effectiveIds.has(tool.id)).length;
  if (selectedCount === 0) return 'none';
  if (selectedCount === selectableTools.length) return 'all';
  return 'some';
}

function uniqueToolIds(ids: Iterable<string>, availableTools: UserSpaceAvailableTool[]): string[] {
  const selectableIds = new Set(getSelectableUserSpaceToolIds(availableTools));
  const next: string[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (!selectableIds.has(id) || seen.has(id)) continue;
    next.push(id);
    seen.add(id);
  }
  return next;
}

function uniqueGroupIds(ids: Iterable<string>): string[] {
  const next: string[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (!id || seen.has(id)) continue;
    next.push(id);
    seen.add(id);
  }
  return next;
}

export function normalizeUserSpaceToolSelection(
  selection: UserSpaceToolSelection,
  availableTools: UserSpaceAvailableTool[],
): UserSpaceToolSelection {
  if (selection.mode === 'default_all') {
    return { mode: 'default_all', toolIds: [], toolGroupIds: [] };
  }
  return {
    mode: 'custom',
    toolIds: uniqueToolIds(selection.toolIds, availableTools),
    toolGroupIds: uniqueGroupIds(selection.toolGroupIds),
  };
}

export function toggleUserSpaceToolSelection(
  selection: UserSpaceToolSelection,
  availableTools: UserSpaceAvailableTool[],
  toolId: string,
): UserSpaceToolSelection {
  const targetTool = availableTools.find((tool) => tool.id === toolId);
  if (!targetTool || !isUserSpaceToolAvailable(targetTool)) return selection;

  const effectiveIds = getEffectiveUserSpaceToolIdSet(selection, availableTools);
  const nextIds = new Set(effectiveIds);
  const nextGroupIds = new Set(selection.mode === 'default_all' ? [] : selection.toolGroupIds);

  if (targetTool.group_id) {
    nextGroupIds.delete(targetTool.group_id);
  }

  if (effectiveIds.has(toolId)) {
    nextIds.delete(toolId);
  } else {
    nextIds.add(toolId);
  }

  return normalizeUserSpaceToolSelection(
    {
      mode: 'custom',
      toolIds: Array.from(nextIds),
      toolGroupIds: Array.from(nextGroupIds),
    },
    availableTools,
  );
}

export function toggleUserSpaceToolGroupSelection(
  selection: UserSpaceToolSelection,
  availableTools: UserSpaceAvailableTool[],
  groupId: string,
  groupTools: UserSpaceAvailableTool[],
): UserSpaceToolSelection {
  const selectableGroupIds = groupTools.filter(isUserSpaceToolAvailable).map((tool) => tool.id);
  if (selectableGroupIds.length === 0) return selection;

  const state = getUserSpaceGroupCheckState(selection, availableTools, groupId, groupTools);
  const effectiveIds = getEffectiveUserSpaceToolIdSet(selection, availableTools);
  const nextIds = new Set(effectiveIds);
  const nextGroupIds = new Set(selection.mode === 'default_all' ? [] : selection.toolGroupIds);

  if (state === 'all') {
    nextGroupIds.delete(groupId);
    for (const toolId of selectableGroupIds) nextIds.delete(toolId);
  } else {
    nextGroupIds.add(groupId);
    for (const toolId of selectableGroupIds) nextIds.delete(toolId);
  }

  return normalizeUserSpaceToolSelection(
    {
      mode: 'custom',
      toolIds: Array.from(nextIds),
      toolGroupIds: Array.from(nextGroupIds),
    },
    availableTools,
  );
}

export function setUserSpaceToolSelectionForTools(
  selection: UserSpaceToolSelection,
  availableTools: UserSpaceAvailableTool[],
  scopedToolIds: string[],
  selected: boolean,
): UserSpaceToolSelection {
  const selectableScope = new Set(uniqueToolIds(scopedToolIds, availableTools));
  if (selectableScope.size === 0) return selection;

  const allSelectableIds = getSelectableUserSpaceToolIds(availableTools);
  if (selected && selectableScope.size === allSelectableIds.length) {
    return { mode: 'default_all', toolIds: [], toolGroupIds: [] };
  }

  const nextIds = getEffectiveUserSpaceToolIdSet(selection, availableTools);
  if (selected) {
    for (const toolId of selectableScope) nextIds.add(toolId);
  } else {
    for (const toolId of selectableScope) nextIds.delete(toolId);
  }

  return normalizeUserSpaceToolSelection(
    {
      mode: 'custom',
      toolIds: Array.from(nextIds),
      toolGroupIds: [],
    },
    availableTools,
  );
}
