import { beforeEach, describe, expect, it, vi } from 'vitest';

import {
  applyUserSpaceToolAvailabilityCap,
  canManageUserSpaceToolWriteForWorkspace,
  fetchUserSpaceToolCatalog,
  getCappedUserSpaceToolIdSet,
  getNextWorkspaceToolOptions,
  getUserSpaceToolWorkspaceWriteState,
  hasUserSpaceToolWriteAccessLevel,
  isUserSpaceToolWriteEnabledForWorkspace,
  type UserSpaceToolSelection,
} from './userSpaceTools';
import type { UserSpaceAvailableTool, WorkspaceToolOptionState } from '@/types';

const apiMock = vi.hoisted(() => ({
  listUserSpaceAvailableTools: vi.fn(),
  listUserSpaceToolGroups: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

const tools: UserSpaceAvailableTool[] = [
  {
    id: 'tool-a',
    name: 'Tool A',
    tool_type: 'postgres',
    available: true,
    access_level: 'read_write',
    group_id: 'group-1',
    group_name: 'Group 1',
  },
  {
    id: 'tool-b',
    name: 'Tool B',
    tool_type: 'ssh_shell',
    available: true,
    access_level: 'read',
    group_id: 'group-1',
    group_name: 'Group 1',
  },
  {
    id: 'tool-c',
    name: 'Tool C',
    tool_type: 'odoo',
    available: true,
    access_level: 'read_write',
  },
];

describe('userSpaceTools workspace capping', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('passes the requested catalog surface to the available-tools API', async () => {
    apiMock.listUserSpaceAvailableTools.mockResolvedValue([]);
    apiMock.listUserSpaceToolGroups.mockResolvedValue([]);

    await fetchUserSpaceToolCatalog('chat');

    expect(apiMock.listUserSpaceAvailableTools).toHaveBeenCalledWith('chat');
  });

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

  it('honors an explicit workspace write grant even when the tool is globally read-only', () => {
    const options: Record<string, WorkspaceToolOptionState> = {
      'tool-a': { write_access_enabled: true },
      'tool-c': { write_access_enabled: true },
    };

    expect(
      isUserSpaceToolWriteEnabledForWorkspace({ ...tools[0], allow_write: true }, options),
    ).toBe(true);
    expect(
      isUserSpaceToolWriteEnabledForWorkspace({ ...tools[2], allow_write: false }, options),
    ).toBe(true);
  });

  it('rejects workspace write enablement when ACL is only read', () => {
    expect(
      isUserSpaceToolWriteEnabledForWorkspace(
        { ...tools[1], allow_write: true },
        { 'tool-b': { write_access_enabled: true } },
      ),
    ).toBe(false);
  });

  it('returns enabled when a tool is enabled for the workspace, even if globally read-only', () => {
    expect(
      getUserSpaceToolWorkspaceWriteState(
        { ...tools[0], allow_write: false },
        { 'tool-a': { write_access_enabled: true } },
      ),
    ).toBe('enabled');
  });

  it('returns eligible when a globally read-only tool has ACL read_write but no workspace write grant', () => {
    expect(getUserSpaceToolWorkspaceWriteState({ ...tools[0], allow_write: false }, {})).toBe(
      'eligible',
    );
  });

  it('returns ineligible when ACL does not allow workspace writes even if global write is enabled', () => {
    expect(getUserSpaceToolWorkspaceWriteState({ ...tools[1], allow_write: true }, {})).toBe(
      'ineligible',
    );
  });

  it('adds and removes workspace write option entries without leaving empty records', () => {
    const enabled = getNextWorkspaceToolOptions({}, 'tool-a', true);
    expect(enabled).toEqual({ 'tool-a': { write_access_enabled: true } });

    const mixed = getNextWorkspaceToolOptions(
      {
        'tool-a': { write_access_enabled: true },
        'tool-b': { write_access_enabled: true },
      },
      'tool-a',
      false,
    );
    expect(mixed).toEqual({ 'tool-b': { write_access_enabled: true } });
  });

  it('allows workspace owners to manage workspace grants only for ACL read_write tools', () => {
    expect(
      canManageUserSpaceToolWriteForWorkspace({ ...tools[0], allow_write: true }, true, false),
    ).toBe(true);
    expect(
      canManageUserSpaceToolWriteForWorkspace({ ...tools[0], allow_write: false }, true, false),
    ).toBe(true);
    expect(
      canManageUserSpaceToolWriteForWorkspace({ ...tools[1], allow_write: true }, true, false),
    ).toBe(false);
    expect(
      canManageUserSpaceToolWriteForWorkspace({ ...tools[0], allow_write: true }, false, true),
    ).toBe(false);
    expect(
      canManageUserSpaceToolWriteForWorkspace({ ...tools[0], allow_write: false }, false, true),
    ).toBe(false);
  });

  it('fails closed when workspace tool ACL context is missing', () => {
    const { access_level: _accessLevel, ...toolWithoutAccessLevel } = tools[0];

    expect(hasUserSpaceToolWriteAccessLevel(toolWithoutAccessLevel)).toBe(false);
    expect(canManageUserSpaceToolWriteForWorkspace(toolWithoutAccessLevel, true, false)).toBe(
      false,
    );
  });
});
