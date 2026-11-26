import type {
  InstructionSkill,
  InstructionSkillCount,
  InstructionSkillMode,
  InstructionSkillSummary,
} from '@/types';

export const EMPTY_INSTRUCTION_SKILL_SUMMARY: InstructionSkillSummary = { tools: {}, routes: {} };
export const DEFAULT_MCP_ROUTE_OWNER_ID = 'default';

/** One row of `GET /indexes/tools/{tool_id}/skills/builtin`; shipped with Ragtime, read-only. */
export interface BuiltinInstructionSkill {
  name: string;
  description: string;
  mode: InstructionSkillMode;
  applies: boolean;
  reason: string | null;
  /** Server view of whether an admin skill with the same name exists (enabled or not). */
  overridden: boolean;
}

export type BuiltinInstructionSkillState = 'active' | 'inactive' | 'replaced';

/**
 * Built-in status as the dialog shows it. A same-name admin skill wins over
 * every other state. `adminSkillNames` is null while the admin list is not
 * known yet, and then the server's `overridden` flag decides.
 */
export function builtinSkillState(
  item: BuiltinInstructionSkill,
  adminSkillNames: ReadonlySet<string> | null,
): BuiltinInstructionSkillState {
  const replaced = adminSkillNames ? adminSkillNames.has(item.name) : item.overridden;
  if (replaced) return 'replaced';
  return item.applies ? 'active' : 'inactive';
}

export function countInstructionSkills(skills: InstructionSkill[]): InstructionSkillCount {
  return {
    total: skills.length,
    enabled: skills.filter((skill) => skill.enabled).length,
  };
}

/** Skills menu description. Built-ins are never counted, so a SolidWorks tool with none says so. */
export function describeInstructionSkillCount(
  count: InstructionSkillCount | undefined,
  toolType?: string | null,
): string {
  if (!count || count.total === 0)
    return toolType === 'solidworks' ? 'Built-in skills only' : 'None yet';
  return `${count.total} attached, ${count.enabled} enabled`;
}

// Closing the skills dialog returns focus to its opener; when that opener was
// removed (for example a context-menu item), focus falls back to the owner's card.
export function restoreFocusIfLost(fallbackId: string): void {
  window.setTimeout(() => {
    const active = document.activeElement;
    if (!active || active === document.body) {
      document.getElementById(fallbackId)?.focus();
    }
  }, 0);
}
