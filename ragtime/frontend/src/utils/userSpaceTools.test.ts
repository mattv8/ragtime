import { describe, expect, it } from 'vitest';

import {
  applyUserSpaceToolAvailabilityCap,
  getCappedUserSpaceToolIdSet,
  type UserSpaceToolSelection,
} from './userSpaceTools';
import type { UserSpaceAvailableTool } from '@/types';

const tools: UserSpaceAvailableTool[] = [
  {
    id: 'tool-a',
    name: 'Tool A',
    tool_type: 'postgres',
    available: true,
    group_id: 'group-1',
    group_name: 'Group 1',
  },
  {
    id: 'tool-b',
    name: 'Tool B',
    tool_type: 'ssh_shell',
    available: true,
    group_id: 'group-1',
    group_name: 'Group 1',
  },
  {
    id: 'tool-c',
    name: 'Tool C',
    tool_type: 'odoo',
    available: true,
  },
];

describe('userSpaceTools workspace capping', () => {
  it('marks tools outside the workspace effective selection unavailable', () => {
    const workspaceSelection: UserSpaceToolSelection = {
      mode: 'custom',
      toolIds: ['tool-c'],
      toolGroupIds: [],
    };

    expect(applyUserSpaceToolAvailabilityCap(tools, workspaceSelection)).toMatchObject([
      { id: 'tool-a', available: false, disabled_reason: 'Disabled in Workspace' },
      { id: 'tool-b', available: false, disabled_reason: 'Disabled in Workspace' },
      { id: 'tool-c', available: true },
    ]);
  });

  it('preserves existing heartbeat failures when applying workspace cap', () => {
    const workspaceSelection: UserSpaceToolSelection = {
      mode: 'custom',
      toolIds: ['tool-c'],
      toolGroupIds: [],
    };
    const offlineTools: UserSpaceAvailableTool[] = [
      {
        ...tools[0],
        available: false,
        disabled_reason: 'No recent heartbeat',
      },
      tools[2],
    ];

    expect(applyUserSpaceToolAvailabilityCap(offlineTools, workspaceSelection)).toMatchObject([
      { id: 'tool-a', available: false, disabled_reason: 'No recent heartbeat' },
      { id: 'tool-c', available: true },
    ]);
  });

  it('intersects conversation and workspace effective selections', () => {
    const conversationSelection: UserSpaceToolSelection = {
      mode: 'custom',
      toolIds: ['tool-a', 'tool-c'],
      toolGroupIds: [],
    };
    const workspaceSelection: UserSpaceToolSelection = {
      mode: 'custom',
      toolIds: ['tool-b'],
      toolGroupIds: ['group-1'],
    };

    expect(
      Array.from(getCappedUserSpaceToolIdSet(conversationSelection, tools, workspaceSelection)),
    ).toEqual(['tool-a']);
  });

  it('treats default-all conversation selection as all tools inside the workspace cap', () => {
    const conversationSelection: UserSpaceToolSelection = {
      mode: 'default_all',
      toolIds: [],
      toolGroupIds: [],
    };
    const workspaceSelection: UserSpaceToolSelection = {
      mode: 'custom',
      toolIds: ['tool-c'],
      toolGroupIds: [],
    };

    expect(
      Array.from(getCappedUserSpaceToolIdSet(conversationSelection, tools, workspaceSelection)),
    ).toEqual(['tool-c']);
  });
});
