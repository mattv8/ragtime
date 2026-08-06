import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
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
  const renderResult = render(
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

  return { ...renderResult, onUpsert, onRevoke };
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
  it('renders one combined help note, keeps existing-row labels and groups, and removes add-form permission controls', () => {
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

    expect(screen.getByRole('note').textContent).toBe(
      "Allow this workspace's agent to use file/runtime tools in another workspace you can access. A SQLite grant covers all data in app.sqlite3, including future data. Only the target workspace owner can change it.",
    );
    expect(
      screen.queryAllByText(
        'A SQLite grant covers all data in app.sqlite3, including future data. Only the target workspace owner can change it.',
      ),
    ).toHaveLength(0);
    expect(screen.getByText('Target workspace')).not.toBeNull();

    const modal = document.querySelector('.userspace-agent-access-modal');
    expect(modal).not.toBeNull();

    const alphaRow = getGrantRow('Alpha');
    expect(alphaRow.querySelector('.userspace-agent-access-control-row')).not.toBeNull();
    expect(within(alphaRow).getByText('File/runtime')).not.toBeNull();
    expect(within(alphaRow).getByText('Shared SQLite')).not.toBeNull();
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

    const newGrantSection = screen
      .getByLabelText('Target workspace')
      .closest('.userspace-agent-access-add-row');
    expect(newGrantSection).not.toBeNull();
    if (!(newGrantSection instanceof HTMLElement)) {
      throw new Error('Missing new grant section');
    }
    expect(
      within(newGrantSection).queryByRole('group', { name: 'New grant file/runtime access' }),
    ).toBeNull();
    expect(
      within(newGrantSection).queryByRole('group', { name: 'New grant Shared SQLite access' }),
    ).toBeNull();
    expect(within(newGrantSection).queryByText('File/runtime')).toBeNull();
    expect(within(newGrantSection).queryByText('Shared SQLite')).toBeNull();
    expect(within(newGrantSection).queryByText(/choose the permissions/i)).toBeNull();
  });

  it('keeps existing-grant edits local until Save is clicked and preserves the current file mode', async () => {
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
    const saveButton = within(row).getByRole('button', { name: 'Save grant for Owner Target' });

    expect((saveButton as HTMLButtonElement).disabled).toBe(true);
    expect(saveButton.classList.contains('btn-sm')).toBe(true);

    await user.click(within(sqliteGroup).getByRole('button', { name: 'Read / Write' }));

    expect(onUpsert).not.toHaveBeenCalled();
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);

    await user.click(saveButton);

    expect(onUpsert).toHaveBeenCalledWith({
      target_workspace_id: 'ws-owner',
      access_mode: 'read_write',
      sqlite_access_mode: 'read_write',
    });
  });

  it('enables Save only while a row draft differs and persists the full owner-capable draft on Save', async () => {
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
    const saveButton = within(row).getByRole('button', { name: 'Save grant for Owner Target' });

    expect((saveButton as HTMLButtonElement).disabled).toBe(true);
    expect(saveButton.classList.contains('btn-sm')).toBe(true);

    await user.click(within(fileGroup).getByRole('button', { name: 'Read / Write' }));

    expect(onUpsert).not.toHaveBeenCalled();
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);

    await user.click(within(fileGroup).getByRole('button', { name: 'Read' }));
    expect((saveButton as HTMLButtonElement).disabled).toBe(true);

    await user.click(within(fileGroup).getByRole('button', { name: 'Read / Write' }));
    await user.click(saveButton);

    expect(onUpsert).toHaveBeenCalledWith({
      target_workspace_id: 'ws-owner',
      access_mode: 'read_write',
      sqlite_access_mode: 'read_write',
    });
    expect(
      within(sqliteGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
  });

  it('resynchronizes a dirty row draft when the persisted grant changes after rerender', async () => {
    const user = userEvent.setup();
    const onUpsert = vi.fn().mockResolvedValue(undefined);
    const onRevoke = vi.fn().mockResolvedValue(undefined);
    const initialGrant = createGrant({
      target_workspace_id: 'ws-owner',
      target_workspace_name: 'Owner Target',
      access_mode: 'read',
      sqlite_access_mode: 'none',
    });
    const updatedGrant = createGrant({
      target_workspace_id: 'ws-owner',
      target_workspace_name: 'Owner Target',
      access_mode: 'read_write',
      sqlite_access_mode: 'read',
      updated_at: '2026-08-06T12:00:00Z',
    });
    const { rerender } = renderModal({
      grants: [initialGrant],
      availableWorkspaces: [
        { id: 'ws-owner', name: 'Owner Target', canGrantReadWrite: true, canGrantSqlite: true },
      ],
      onUpsert,
      onRevoke,
    });

    const row = getGrantRow('Owner Target');
    const fileGroup = within(row).getByRole('group', {
      name: 'File/runtime access for Owner Target',
    });
    const saveButton = within(row).getByRole('button', { name: 'Save grant for Owner Target' });

    await user.click(within(fileGroup).getByRole('button', { name: 'Read / Write' }));
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);

    rerender(
      <AgentAccessModal
        isOpen
        onClose={vi.fn()}
        sourceWorkspace={SOURCE_WORKSPACE}
        availableWorkspaces={[
          { id: 'ws-owner', name: 'Owner Target', canGrantReadWrite: true, canGrantSqlite: true },
        ]}
        grants={[updatedGrant]}
        onUpsert={onUpsert}
        onRevoke={onRevoke}
      />,
    );

    const rerenderedRow = getGrantRow('Owner Target');
    const rerenderedFileGroup = within(rerenderedRow).getByRole('group', {
      name: 'File/runtime access for Owner Target',
    });
    const rerenderedSqliteGroup = within(rerenderedRow).getByRole('group', {
      name: 'Shared SQLite access for Owner Target',
    });
    const rerenderedSaveButton = within(rerenderedRow).getByRole('button', {
      name: 'Save grant for Owner Target',
    });

    expect(
      within(rerenderedFileGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
    expect(
      within(rerenderedSqliteGroup)
        .getByRole('button', { name: 'Read' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
    expect((rerenderedSaveButton as HTMLButtonElement).disabled).toBe(true);
  });

  it('shows current SQLite mode to a non-owner, disables SQLite changes, and preserves revoke restrictions', () => {
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

  it('uses a blank placeholder, disables granted targets, and enables Add only for valid ungranted targets', async () => {
    const user = userEvent.setup();
    const { onUpsert } = renderModal({
      grants: [
        createGrant({
          target_workspace_id: 'ws-granted',
          target_workspace_name: 'Already Granted',
        }),
      ],
      availableWorkspaces: [
        {
          id: 'ws-granted',
          name: 'Already Granted',
          canGrantReadWrite: true,
          canGrantSqlite: true,
        },
        { id: 'ws-member', name: 'Member Target', canGrantReadWrite: true, canGrantSqlite: false },
      ],
    });

    const targetSelect = screen.getByLabelText('Target workspace') as HTMLSelectElement;
    const addButton = screen.getByRole('button', { name: 'Add grant' });
    const footer = document.querySelector('.modal-footer');

    expect(
      (screen.getByRole('option', { name: 'Select a workspace...' }) as HTMLOptionElement).value,
    ).toBe('');
    expect(targetSelect.value).toBe('');
    expect((addButton as HTMLButtonElement).disabled).toBe(true);
    expect(footer).not.toBeNull();
    if (!(footer instanceof HTMLElement)) {
      throw new Error('Missing modal footer');
    }
    expect(within(footer).getByRole('button', { name: 'Close' })).toBe(
      screen.getByRole('button', { name: 'Close' }),
    );
    expect(within(footer).getByRole('button', { name: 'Add grant' })).toBe(addButton);
    expect(
      within(footer)
        .getAllByRole('button')
        .map((button) => button.textContent?.trim()),
    ).toEqual(['Close', 'Add grant']);
    expect(
      (
        screen.getByRole('option', {
          name: 'Already Granted (already granted)',
        }) as HTMLOptionElement
      ).disabled,
    ).toBe(true);
    expect(screen.queryByRole('group', { name: 'New grant file/runtime access' })).toBeNull();
    expect(screen.queryByRole('group', { name: 'New grant Shared SQLite access' })).toBeNull();

    await user.selectOptions(targetSelect, 'ws-member');

    expect((addButton as HTMLButtonElement).disabled).toBe(false);

    await user.click(addButton);

    expect(onUpsert).toHaveBeenCalledWith({
      target_workspace_id: 'ws-member',
      access_mode: 'read',
    });
  });

  it('submits the least-privilege payload for an owned target and resets the target to the placeholder after Add succeeds', async () => {
    const user = userEvent.setup();
    const { onUpsert } = renderModal({
      availableWorkspaces: [
        { id: 'ws-alpha', name: 'Alpha Owned', canGrantReadWrite: true, canGrantSqlite: true },
        { id: 'ws-zulu', name: 'Zulu Owned', canGrantReadWrite: true, canGrantSqlite: true },
      ],
    });

    const targetSelect = screen.getByLabelText('Target workspace') as HTMLSelectElement;
    await user.selectOptions(targetSelect, 'ws-zulu');

    await user.click(screen.getByRole('button', { name: 'Add grant' }));

    expect(onUpsert).toHaveBeenLastCalledWith({
      target_workspace_id: 'ws-zulu',
      access_mode: 'read',
    });
    await waitFor(() => {
      expect(targetSelect.value).toBe('');
    });
  });

  it('keeps a persisted read_write mode active when write capability is unavailable and does not mark the row dirty', () => {
    renderModal({
      grants: [
        createGrant({
          target_workspace_id: 'ws-limited',
          target_workspace_name: 'Limited Target',
          access_mode: 'read_write',
          sqlite_access_mode: 'none',
        }),
      ],
      availableWorkspaces: [
        {
          id: 'ws-limited',
          name: 'Limited Target',
          canGrantReadWrite: false,
          canGrantSqlite: true,
        },
      ],
    });

    const row = getGrantRow('Limited Target');
    const fileGroup = within(row).getByRole('group', {
      name: 'File/runtime access for Limited Target',
    });

    expect(
      within(fileGroup).getByRole('button', { name: 'Read / Write' }).getAttribute('aria-pressed'),
    ).toBe('true');
    expect(
      (
        within(row).getByRole('button', {
          name: 'Save grant for Limited Target',
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  it('retains a dirty draft after a rejected Save and surfaces an in-modal error', async () => {
    const user = userEvent.setup();
    const onUpsert = vi.fn().mockRejectedValue(new Error('Save failed'));
    renderModal({
      grants: [
        createGrant({
          target_workspace_id: 'ws-owner',
          target_workspace_name: 'Owner Target',
          access_mode: 'read',
          sqlite_access_mode: 'read',
        }),
      ],
      availableWorkspaces: [
        { id: 'ws-owner', name: 'Owner Target', canGrantReadWrite: true, canGrantSqlite: true },
      ],
      onUpsert,
    });

    const row = getGrantRow('Owner Target');
    const sqliteGroup = within(row).getByRole('group', {
      name: 'Shared SQLite access for Owner Target',
    });
    const saveButton = within(row).getByRole('button', { name: 'Save grant for Owner Target' });

    await user.click(within(sqliteGroup).getByRole('button', { name: 'Read / Write' }));
    await user.click(saveButton);

    expect(await within(row).findByText('Save failed')).not.toBeNull();
    expect((saveButton as HTMLButtonElement).disabled).toBe(false);
    expect(
      within(sqliteGroup)
        .getByRole('button', { name: 'Read / Write' })
        .getAttribute('aria-pressed'),
    ).toBe('true');
  });
});
