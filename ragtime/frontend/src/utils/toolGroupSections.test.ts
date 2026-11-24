import { describe, expect, it } from 'vitest';
import type { ToolConfig, ToolGroup } from '@/types';
import { buildToolSections, computeToolMove, moveToolWithinSection } from './toolGroupSections';

const tools = ['t1', 't2', 't3', 't4', 't5'].map(
  (id, i) => ({ id, group_id: i === 1 || i === 3 ? 'g1' : null }) as ToolConfig,
);
const groups = [
  { id: 'g1', name: 'Beta', sort_order: 0 },
  { id: 'g2', name: 'alpha', sort_order: 0 },
] as ToolGroup[];
const knownGroupIds = new Set(['g1', 'g2']);

describe('tool sections', () => {
  it('sorts groups, retains tool order, and puts unknown groups in ungrouped', () => {
    const input = [...tools, { id: 'ghost', group_id: 'ghost' } as ToolConfig];
    expect(buildToolSections(input, groups).map((s) => [s.id, s.tools.map((t) => t.id)])).toEqual([
      ['g2', []],
      ['g1', ['t2', 't4']],
      ['ungrouped', ['t1', 't3', 't5', 'ghost']],
    ]);
    expect(buildToolSections([], groups).map((s) => s.id)).toEqual(['g2', 'g1']);
    expect(buildToolSections([], groups, { includeEmptyUngrouped: true }).map((s) => s.id)).toEqual(
      ['g2', 'g1', 'ungrouped'],
    );
  });
  it.each([
    ['t5', 'g1', 0, ['t1', 't5', 't2', 't3', 't4'], true],
    ['t5', 'g1', 2, ['t1', 't2', 't3', 't4', 't5'], true],
    ['t2', 'g1', 2, ['t1', 't3', 't4', 't2', 't5'], false],
    ['t1', 'g2', 0, ['t2', 't3', 't4', 't5', 't1'], true],
  ])(
    'moves %s to %s at %i with a complete order',
    (toolId, targetGroupId, insertIndex, orderedIds, groupChanged) => {
      const result = computeToolMove({ tools, knownGroupIds, toolId, targetGroupId, insertIndex });
      expect(result).toEqual({ orderedIds, groupChanged });
      expect(new Set(result?.orderedIds)).toEqual(new Set(tools.map((t) => t.id)));
    },
  );
  it.each([
    ['t2', 0],
    ['t2', 1],
    ['missing', 0],
  ])('returns null for no-op or missing %s at %i', (toolId, insertIndex) => {
    expect(
      computeToolMove({ tools, knownGroupIds, toolId, targetGroupId: 'g1', insertIndex }),
    ).toBeNull();
  });
  it('treats unknown group membership as ungrouped', () => {
    const input = tools.map((t) => (t.id === 't3' ? { ...t, group_id: 'ghost' } : t));
    expect(
      computeToolMove({
        tools: input,
        knownGroupIds,
        toolId: 't3',
        targetGroupId: null,
        insertIndex: 3,
      }),
    ).toEqual({ orderedIds: ['t1', 't2', 't4', 't5', 't3'], groupChanged: false });
  });
  it.each([
    ['t4', 'earlier', ['t1', 't4', 't3', 't2', 't5']],
    ['t3', 'later', ['t1', 't2', 't5', 't4', 't3']],
    ['t2', 'earlier', null],
    ['t4', 'later', null],
    ['missing', 'later', null],
  ] as const)('swaps %s %s with its section neighbour', (toolId, direction, orderedIds) => {
    expect(moveToolWithinSection({ tools, knownGroupIds, toolId, direction })).toEqual(
      orderedIds ? { orderedIds, groupChanged: false } : null,
    );
  });
});
