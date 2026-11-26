import { describe, expect, it } from 'vitest';
import {
  builtinSkillState,
  describeInstructionSkillCount,
  type BuiltinInstructionSkill,
} from './instructionSkills';

const builtin = (overrides: Partial<BuiltinInstructionSkill> = {}): BuiltinInstructionSkill => ({
  name: 'solidworks-scripts',
  description: 'Edit a copy with a script.',
  mode: 'on_demand',
  applies: true,
  reason: null,
  overridden: false,
  ...overrides,
});

describe('builtinSkillState', () => {
  it('reports active and not-active built-ins', () => {
    expect(builtinSkillState(builtin(), new Set())).toBe('active');
    expect(builtinSkillState(builtin({ applies: false }), new Set())).toBe('inactive');
  });

  it('lets an admin skill with the same name replace the built-in, enabled or not', () => {
    expect(builtinSkillState(builtin(), new Set(['solidworks-scripts']))).toBe('replaced');
    expect(builtinSkillState(builtin({ applies: false }), new Set(['solidworks-scripts']))).toBe(
      'replaced',
    );
  });

  it('falls back to the server flag while the admin list is unknown', () => {
    expect(builtinSkillState(builtin({ overridden: true }), null)).toBe('replaced');
    expect(builtinSkillState(builtin({ overridden: false }), null)).toBe('active');
  });
});

describe('describeInstructionSkillCount', () => {
  it('says built-ins only for an empty SolidWorks tool', () => {
    expect(describeInstructionSkillCount(undefined, 'solidworks')).toBe('Built-in skills only');
    expect(describeInstructionSkillCount({ total: 0, enabled: 0 }, 'solidworks')).toBe(
      'Built-in skills only',
    );
  });

  it('keeps the existing copy for other tools', () => {
    expect(describeInstructionSkillCount({ total: 0, enabled: 0 }, 'ssh_shell')).toBe('None yet');
    expect(describeInstructionSkillCount(undefined)).toBe('None yet');
  });

  it('counts admin skills only', () => {
    expect(describeInstructionSkillCount({ total: 3, enabled: 2 }, 'solidworks')).toBe(
      '3 attached, 2 enabled',
    );
  });
});
