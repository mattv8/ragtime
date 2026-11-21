import type { InstructionSkill, InstructionSkillCount, InstructionSkillSummary } from '@/types';

export const EMPTY_INSTRUCTION_SKILL_SUMMARY: InstructionSkillSummary = { tools: {}, routes: {} };

export function countInstructionSkills(skills: InstructionSkill[]): InstructionSkillCount {
  return {
    total: skills.length,
    enabled: skills.filter((skill) => skill.enabled).length,
  };
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
