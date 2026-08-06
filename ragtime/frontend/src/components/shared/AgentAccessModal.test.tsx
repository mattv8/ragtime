import { cleanup, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { WorkspaceAgentGrant } from '@/types';
import { AgentAccessModal } from './AgentAccessModal';

const SOURCE_WORKSPACE = { id: 'ws-source', name: 'Source Workspace' };

function createGrant(overrides: Partial<WorkspaceAgentGrant> = {}): WorkspaceAgentGrant {
  return {
    id: 'grant-1',
    source_workspace_id: 'ws-source',
    source_workspace_name: 'Source Workspace',
    target_workspace_id: 'ws-target',
    target_workspace_name: 'Target Workspace',
    access_mode: 'read',
    sqlite_access_mode: 'none',
    granted_by_user_id: 'user-1',
    created_at: '2026-08-05T12:00:00Z',
    updated_at: '2026-08-05T12:00:00Z',
    ...overrides,
  };
}

function renderModal({
  grants = [],
  availableWorkspaces = [],
  onUpsert = vi.fn().mockResolvedValue(undefined),
  onRevoke = vi.fn().mockResolvedValue(undefined),
}: {
  grants?: Array<ReturnType<typeof createGrant>>;
  availableWorkspaces?: Array<{
    id: string;
    name: string;
    canGrantReadWrite?: boolean;
    canGrantSqlite?: boolean;
  }>;
  onUpsert?: ReturnType<typeof vi.fn>;
  onRevoke?: ReturnType<typeof vi.fn>;
} = {}) {
  render(
    <AgentAccessModal
      isOpen
      onClose={vi.fn()}
      sourceWorkspace={SOURCE_WORKSPACE}
      availableWorkspaces={availableWorkspaces}
      grants={grants}
      onUpsert={onUpsert}
      onRevoke={onRevoke}
    />,
  );

  return { onUpsert, onRevoke };
}

function getGrantRow(targetName: string): HTMLElement {
  const targetText = screen
    .getAllByText(targetName)
    .find((element) => element.tagName.toLowerCase() === 'span');
  if (!(targetText instanceof HTMLElement)) {
    throw new Error(`Missing target label for ${targetName}`);
  }
  const row = targetText.closest('.userspace-agent-access-grant-row');
  if (!(row instanceof HTMLElement)) {
    throw new Error(`Missing grant row for ${targetName}`);
  }
  return row;
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('AgentAccessModal', () => {
  it('renders the SQLite warning, accessible toggle groups, and fixture modes for none/read/read_write', () => {
    renderModal({
      grants: [
        createGrant({
          id: 'grant-none',
          target_workspace_id: 'ws-a',
          target_workspace_name: 'Alpha',
          sqlite_access_mode: 'none',
        }),
        createGrant({
          id: 'grant-read',
          target_workspace_id: 'ws-b',
          target_workspace_name: 'Beta',
          sqlite_access_mode: 'read',
        }),
        createGrant({
          id: 'grant-read-write',
          target_workspace_id: 'ws-c',
          target_workspace_name: 'Gamma',
          sqlite_access_mode: 'read_write',
          access_mode: 'read_write',
        }),
      ],
      availableWorkspaces: [
        { id: 'ws-a', name: 'Alpha', canGrantReadWrite: true, canGrantSqlite: true },
        { id: 'ws-b', name: 'Beta', canGrantReadWrite: true, canGrantSqlite: true },
        { id: 'ws-c', name: 'Gamma', canGrantReadWrite: true, canGrantSqlite: true },
      ],
    });

    expect(screen.getByRole('note').textContent).toContain(
      'A SQLite grant covers all data in app.sqlite3, including future data. Only the target workspace owner can change it.',
    );

    const modal = document.querySelector('.userspace-agent-access-modal');
    expect(modal).not.toBeNull();

    const alphaRow = getGrantRow('Alpha');
    expect(alphaRow.querySelector('.userspace-agent-access-control-row')).not.toBeNull();
    const alphaSqliteGroup = within(alphaRow).getByRole('group', {
      name: 'Shared SQLite access for Alpha',
    });
    expect(
      within(alphaSqliteGroup).getByRole('button', { name: 'None' }).getAttribute('aria-pressed'),
    ).toBe('true');

    const betaRow = getGrantRow('Beta');
    const betaSqliteGroup = within(betaRow).getByRole('group', {
      name: 'Shared SQLite access for Beta',
    });
    expect(
      within(betaSqliteGroup).getByRole('button', { name: 'Read' }).getAttribute('aria-pressed'),
    ).toBe('true');

    const gammaRow = getGrantRow('Gamma');
    const gammaFileGroup = within(gammaRow).getByRole('group', {
      name: 'File/runtime access for Gamma',
    });
    expect(
      within(gammaFileGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
    const gammaSqliteGroup = within(gammaRow).getByRole('group', {
      name: 'Shared SQLite access for Gamma',
    });
    expect(
      within(gammaSqliteGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
  });

  it('lets an owner change SQLite mode while preserving the current file mode', async () => {
    const user = userEvent.setup();
    const { onUpsert } = renderModal({
      grants: [
        createGrant({
          target_workspace_id: 'ws-owner',
          target_workspace_name: 'Owner Target',
          access_mode: 'read_write',
          sqlite_access_mode: 'read',
        }),
      ],
      availableWorkspaces: [
        { id: 'ws-owner', name: 'Owner Target', canGrantReadWrite: true, canGrantSqlite: true },
      ],
    });

    const row = getGrantRow('Owner Target');
    const sqliteGroup = within(row).getByRole('group', {
      name: 'Shared SQLite access for Owner Target',
    });

    await user.click(within(sqliteGroup).getByRole('button', { name: 'Read / Write' }));

    expect(onUpsert).toHaveBeenCalledWith({
      target_workspace_id: 'ws-owner',
      access_mode: 'read_write',
      sqlite_access_mode: 'read_write',
    });
  });

  it('omits sqlite_access_mode during file-mode edits and keeps the current SQLite state visible', async () => {
    const user = userEvent.setup();
    const { onUpsert } = renderModal({
      grants: [
        createGrant({
          target_workspace_id: 'ws-owner',
          target_workspace_name: 'Owner Target',
          access_mode: 'read',
          sqlite_access_mode: 'read_write',
        }),
      ],
      availableWorkspaces: [
        { id: 'ws-owner', name: 'Owner Target', canGrantReadWrite: true, canGrantSqlite: true },
      ],
    });

    const row = getGrantRow('Owner Target');
    const fileGroup = within(row).getByRole('group', {
      name: 'File/runtime access for Owner Target',
    });
    const sqliteGroup = within(row).getByRole('group', {
      name: 'Shared SQLite access for Owner Target',
    });

    await user.click(within(fileGroup).getByRole('button', { name: 'Read / Write' }));

    expect(onUpsert).toHaveBeenCalledWith({
      target_workspace_id: 'ws-owner',
      access_mode: 'read_write',
    });
    expect(
      within(sqliteGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
  });

  it('shows current SQLite mode to a non-owner, disables SQLite changes, and blocks revoke when SQLite access exists', () => {
    renderModal({
      grants: [
        createGrant({
          id: 'grant-read',
          target_workspace_id: 'ws-member',
          target_workspace_name: 'Member Target',
          sqlite_access_mode: 'read',
        }),
        createGrant({
          id: 'grant-none',
          target_workspace_id: 'ws-member-none',
          target_workspace_name: 'Member Target None',
          sqlite_access_mode: 'none',
        }),
      ],
      availableWorkspaces: [
        { id: 'ws-member', name: 'Member Target', canGrantReadWrite: true, canGrantSqlite: false },
        {
          id: 'ws-member-none',
          name: 'Member Target None',
          canGrantReadWrite: true,
          canGrantSqlite: false,
        },
      ],
    });

    const row = getGrantRow('Member Target');
    const sqliteGroup = within(row).getByRole('group', {
      name: 'Shared SQLite access for Member Target',
    });
    const buttons = within(sqliteGroup).getAllByRole('button');
    expect(buttons.every((button) => (button as HTMLButtonElement).disabled)).toBe(true);
    expect(
      within(sqliteGroup).getByRole('button', { name: 'Read' }).getAttribute('aria-pressed'),
    ).toBe('true');
    expect(
      (within(row).getByRole('button', { name: /revoke read only access/i }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);

    const noneRow = getGrantRow('Member Target None');
    expect(
      (
        within(noneRow).getByRole('button', {
          name: /revoke read only access/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });

  it('omits sqlite_access_mode when a non-owner adds or updates a file-only grant', async () => {
    const user = userEvent.setup();
    const { onUpsert } = renderModal({
      availableWorkspaces: [
        { id: 'ws-member', name: 'Member Target', canGrantReadWrite: true, canGrantSqlite: false },
      ],
    });

    const sqliteGroup = screen.getByRole('group', { name: 'New grant Shared SQLite access' });
    const sqliteButtons = within(sqliteGroup).getAllByRole('button');
    expect(sqliteButtons.every((button) => (button as HTMLButtonElement).disabled)).toBe(true);

    await user.click(screen.getByRole('button', { name: 'Add / Update Grant' }));

    expect(onUpsert).toHaveBeenCalledWith({
      target_workspace_id: 'ws-member',
      access_mode: 'read',
    });
  });

  it('includes sqlite_access_mode for an owned target and resets the local SQLite choice when switching targets', async () => {
    const user = userEvent.setup();
    const { onUpsert } = renderModal({
      availableWorkspaces: [
        { id: 'ws-alpha', name: 'Alpha Owned', canGrantReadWrite: true, canGrantSqlite: true },
        { id: 'ws-zulu', name: 'Zulu Owned', canGrantReadWrite: true, canGrantSqlite: true },
      ],
    });

    const sqliteGroup = screen.getByRole('group', { name: 'New grant Shared SQLite access' });
    await user.click(within(sqliteGroup).getByRole('button', { name: 'Read / Write' }));
    expect(
      within(sqliteGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');

    await user.selectOptions(screen.getByLabelText('Target workspace'), 'ws-zulu');

    expect(
      within(sqliteGroup).getByRole('button', { name: 'None' }).getAttribute('aria-pressed'),
    ).toBe('true');

    await user.click(within(sqliteGroup).getByRole('button', { name: 'Read' }));
    await user.click(screen.getByRole('button', { name: 'Add / Update Grant' }));

    expect(onUpsert).toHaveBeenLastCalledWith({
      target_workspace_id: 'ws-zulu',
      access_mode: 'read',
      sqlite_access_mode: 'read',
    });
  });
});
