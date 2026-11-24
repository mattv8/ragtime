import type { ToolConfig, ToolGroup } from '@/types';

export const UNGROUPED_SECTION_ID = 'ungrouped';
export interface ToolSection {
  id: string;
  group: ToolGroup | null;
  tools: ToolConfig[];
}
export interface ToolMoveResult {
  orderedIds: string[];
  groupChanged: boolean;
}

function effectiveGroup(tool: ToolConfig, knownGroupIds: ReadonlySet<string>): string | null {
  return tool.group_id && knownGroupIds.has(tool.group_id) ? tool.group_id : null;
}

export function buildToolSections(
  tools: ToolConfig[],
  groups: ToolGroup[],
  o?: { includeEmptyUngrouped?: boolean },
): ToolSection[] {
  const sections = [...groups]
    .sort(
      (a, b) =>
        a.sort_order - b.sort_order ||
        a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }),
    )
    .map((group) => ({ id: group.id, group, tools: [] as ToolConfig[] }));
  const byId = new Map<string, ToolSection>(sections.map((section) => [section.id, section]));
  const ungrouped: ToolSection = { id: UNGROUPED_SECTION_ID, group: null, tools: [] };
  for (const tool of tools) (byId.get(tool.group_id ?? '') ?? ungrouped).tools.push(tool);
  return ungrouped.tools.length || o?.includeEmptyUngrouped ? [...sections, ungrouped] : sections;
}

export function computeToolMove(i: {
  tools: ToolConfig[];
  knownGroupIds: ReadonlySet<string>;
  toolId: string;
  targetGroupId: string | null;
  insertIndex: number;
}): ToolMoveResult | null {
  const tool = i.tools.find((item) => item.id === i.toolId);
  if (!tool) return null;
  const currentGroup = effectiveGroup(tool, i.knownGroupIds);
  const targetGroup =
    i.targetGroupId && i.knownGroupIds.has(i.targetGroupId) ? i.targetGroupId : null;
  const groupChanged = currentGroup !== targetGroup;
  const targetTools = i.tools.filter(
    (item) => effectiveGroup(item, i.knownGroupIds) === targetGroup,
  );
  const fromIndex = targetTools.findIndex((item) => item.id === i.toolId);
  let index = Number.isFinite(i.insertIndex) ? i.insertIndex : targetTools.length;
  if (fromIndex !== -1 && fromIndex < index) index--;
  const remaining = i.tools.filter((item) => item.id !== i.toolId);
  const mates = targetTools.filter((item) => item.id !== i.toolId);
  index = Math.max(0, Math.min(index, mates.length));
  if (!groupChanged && index === fromIndex) return null;
  let globalIndex = remaining.length;
  if (mates.length) {
    const anchor = mates[index] ?? mates[mates.length - 1];
    globalIndex =
      remaining.findIndex((item) => item.id === anchor.id) + (index === mates.length ? 1 : 0);
  }
  remaining.splice(globalIndex, 0, tool);
  const orderedIds = remaining.map((item) => item.id);
  if (!groupChanged && orderedIds.every((id, position) => id === i.tools[position].id)) return null;
  return { orderedIds, groupChanged };
}

export function moveToolWithinSection(i: {
  tools: ToolConfig[];
  knownGroupIds: ReadonlySet<string>;
  toolId: string;
  direction: 'earlier' | 'later';
}): ToolMoveResult | null {
  const tool = i.tools.find((item) => item.id === i.toolId);
  if (!tool) return null;
  const group = effectiveGroup(tool, i.knownGroupIds);
  const mates = i.tools.filter((item) => effectiveGroup(item, i.knownGroupIds) === group);
  const index = mates.findIndex((item) => item.id === i.toolId);
  const neighbour = mates[index + (i.direction === 'earlier' ? -1 : 1)];
  if (!neighbour) return null;
  const orderedIds = i.tools.map((item) =>
    item.id === tool.id ? neighbour.id : item.id === neighbour.id ? tool.id : item.id,
  );
  return { orderedIds, groupChanged: false };
}
