import { useState } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { PdmFolderChild, PdmRights } from '../api/solidworks';
import { SolidWorksApprovedFolders } from './SolidWorksApprovedFolders';

class ListError extends Error {
  constructor(
    message: string,
    readonly code: string,
  ) {
    super(message);
  }
}

const listChildren = vi.hoisted(() => vi.fn());
vi.mock('../utils/pdmFolderListing', () => ({ listAllPdmFolderChildren: listChildren }));

afterEach(cleanup);

const FULL: PdmRights = { read: true, add_files: true, check_out: true, add_folders: true };
const READ_ONLY: PdmRights = { read: true, add_files: false, check_out: false, add_folders: false };
const NO_READ: PdmRights = { read: false, add_files: false, check_out: false, add_folders: false };

function child(vaultPath: string, rights: PdmRights | null): PdmFolderChild {
  const name = vaultPath.slice(vaultPath.lastIndexOf('\\') + 1);
  return { name, vault_path: vaultPath, rights, write_ready: rights === FULL };
}

const TREE: Record<string, PdmFolderChild[]> = {
  '\\': [
    child('\\04_Custom', READ_ONLY),
    child('\\13_Ragtime', FULL),
    child('\\99_Archive', NO_READ),
    child('\\Unchecked', null),
  ],
  '\\04_Custom': [
    child('\\04_Custom\\01_Family-Platform', READ_ONLY),
    child('\\04_Custom\\02_Mounting_Bodies', READ_ONLY),
    child('\\04_Custom\\03_Light_Body', FULL),
  ],
};

interface HarnessProps {
  initial?: string[];
  disabled?: boolean;
  suggestions?: boolean;
  onFoldersChange?: (next: string[]) => void;
  onSubmit?: () => void;
}

function Harness({
  initial = [],
  disabled,
  suggestions = false,
  onFoldersChange,
  onSubmit,
}: HarnessProps) {
  const [folders, setFolders] = useState(initial);
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onSubmit?.();
      }}
    >
      <SolidWorksApprovedFolders
        folders={folders}
        vaultRoot={'C:\\PDM'}
        disabled={disabled}
        suggestions={
          suggestions
            ? { toolId: 'sw-a', pdmToolId: 'pdm-a' }
            : { unavailableReason: 'Suggestions are off.' }
        }
        onChange={(next) => {
          onFoldersChange?.(next);
          setFolders(next);
        }}
      />
    </form>
  );
}

const input = () => document.getElementById('solidworks-pdm-folder-input') as HTMLInputElement;
const addButton = () => screen.getByRole('button', { name: 'Add folder' });
const removeButton = (folder: string) =>
  screen.getByRole('button', { name: `Remove folder ${folder}` });
const rowFor = (key: string) =>
  Array.from(document.querySelectorAll<HTMLElement>('li[data-sw-folder]')).find(
    (row) => row.getAttribute('data-sw-folder') === key,
  );
const optionFor = (key: string) =>
  Array.from(document.querySelectorAll<HTMLElement>('li[role="option"]')).find(
    (row) => row.getAttribute('data-sw-folder') === key,
  );
const statusText = () => document.getElementById('solidworks-pdm-folders-status')?.textContent;
const feedback = () => document.getElementById('solidworks-pdm-folder-feedback');
const popup = () => document.getElementById('solidworks-pdm-folder-suggest');
const popupMessage = () =>
  document.getElementById('solidworks-pdm-folder-suggest-message')?.textContent;

function typeFolder(value: string) {
  fireEvent.change(input(), { target: { value } });
}

function press(key: string) {
  return fireEvent.keyDown(input(), { key });
}

async function openRoot() {
  fireEvent.click(input());
  await waitFor(() => expect(optionFor('\\04_custom')).toBeTruthy());
}

describe('SolidWorksApprovedFolders', () => {
  it('shows folder rights badges and defaults unchecked folders', () => {
    render(
      <SolidWorksApprovedFolders
        folders={['\\Ready', '\\Missing', '\\Hidden', '\\Unknown']}
        vaultRoot={null}
        suggestions={{ unavailableReason: 'Suggestions are off.' }}
        onChange={vi.fn()}
        rights={{ '\\ready': 'write_ready', '\\missing': 'missing_write', '\\hidden': 'not_found' }}
      />,
    );
    expect(rowFor('\\ready')?.getAttribute('data-sw-rights')).toBe('write_ready');
    expect(rowFor('\\ready')?.textContent).toContain('Can add and check out');
    expect(rowFor('\\missing')?.textContent).toContain('Missing add or check-out rights');
    expect(rowFor('\\hidden')?.textContent).toContain('Not found in the vault');
    expect(rowFor('\\unknown')?.getAttribute('data-sw-rights')).toBe('not_checked');
  });

  it('names the group, explains why suggestions are off and exposes folder help', () => {
    render(<Harness />);
    expect(screen.getByRole('group', { name: 'Approved folders' })).toBeTruthy();
    expect(input().getAttribute('aria-describedby')).toBe(
      'solidworks-pdm-folder-suggest-unavailable',
    );
    expect(document.getElementById('solidworks-pdm-folder-suggest-unavailable')?.textContent).toBe(
      'Suggestions are off.',
    );
    fireEvent.click(screen.getByRole('button', { name: 'About approved folders' }));
    expect(document.getElementById('solidworks-pdm-folders-help-content')?.textContent).toContain(
      'Changes anywhere else in the vault are refused.',
    );
    expect(document.getElementById('solidworks-pdm-folder-list')).toBeNull();
  });

  it('adds a folder with the Add folder button and keeps focus in the input', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    typeFolder('\\13_Ragtime');
    fireEvent.click(addButton());
    expect(onFoldersChange).toHaveBeenLastCalledWith(['\\13_Ragtime']);
    expect(rowFor('\\13_ragtime')?.textContent).toContain('\\13_Ragtime');
    expect(input().value).toBe('');
    expect(document.activeElement).toBe(input());
    expect(statusText()).toBe('Added \\13_Ragtime. 1 folder approved.');
  });

  it('adds a folder on Enter, prevents the default submit, and does not submit the form', () => {
    const onFoldersChange = vi.fn();
    const onSubmit = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} onSubmit={onSubmit} />);
    typeFolder('\\13_Ragtime');
    const notPrevented = press('Enter');
    expect(notPrevented).toBe(false);
    expect(onFoldersChange).toHaveBeenLastCalledWith(['\\13_Ragtime']);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('ignores other keys', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    typeFolder('\\13_Ragtime');
    press('a');
    expect(onFoldersChange).not.toHaveBeenCalled();
  });

  it('turns forward slashes into vault separators while typing', () => {
    render(<Harness />);
    typeFolder('/13_Ragtime/Rev');
    expect(input().value).toBe('\\13_Ragtime\\Rev');
  });

  it('shows the empty-entry error and keeps the input value', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    fireEvent.click(addButton());
    expect(onFoldersChange).not.toHaveBeenCalled();
    expect(feedback()?.textContent).toBe('Enter a vault folder, for example \\13_Ragtime.');
    expect(feedback()?.getAttribute('data-tone')).toBe('error');
  });

  it('marks an invalid entry, links the error, keeps the value, and adds nothing', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    typeFolder('\\13_Ragtime\\..\\x');
    fireEvent.click(addButton());
    expect(onFoldersChange).not.toHaveBeenCalled();
    expect(feedback()?.textContent).toBe(
      'Folder paths cannot contain “..” or any of * ? " < > | :',
    );
    expect(input().getAttribute('aria-invalid')).toBe('true');
    expect(input().getAttribute('aria-describedby')).toContain('solidworks-pdm-folder-feedback');
    expect(input().value).toBe('\\13_Ragtime\\..\\x');
  });

  it('clears the error once the entry changes', () => {
    render(<Harness />);
    fireEvent.click(addButton());
    expect(feedback()).not.toBeNull();
    typeFolder('\\1');
    expect(feedback()).toBeNull();
  });

  it('reports a covered folder as info, not as an invalid entry', () => {
    render(<Harness initial={['\\13_Ragtime']} />);
    typeFolder('\\13_Ragtime\\Rev');
    fireEvent.click(addButton());
    expect(feedback()?.textContent).toBe('\\13_Ragtime\\Rev is already covered by \\13_Ragtime.');
    expect(feedback()?.getAttribute('data-tone')).toBe('info');
    expect(input().getAttribute('aria-invalid')).toBe('false');
    expect(input().value).toBe('\\13_Ragtime\\Rev');
  });

  it('announces a converted view-root prefix and shows it', () => {
    render(<Harness />);
    typeFolder('C:\\PDM\\13_Ragtime');
    fireEvent.click(addButton());
    expect(statusText()).toBe(
      'Converted C:\\PDM\\13_Ragtime to \\13_Ragtime. Added \\13_Ragtime. 1 folder approved.',
    );
    expect(feedback()?.getAttribute('data-tone')).toBe('info');
  });

  it('refuses a folder beyond the cap of 25', () => {
    const full = Array.from({ length: 25 }, (_, index) => `\\F${index}`);
    const onFoldersChange = vi.fn();
    render(<Harness initial={full} onFoldersChange={onFoldersChange} />);
    typeFolder('\\F99');
    fireEvent.click(addButton());
    expect(onFoldersChange).not.toHaveBeenCalled();
    expect(feedback()?.textContent).toBe('You can approve up to 25 folders.');
  });

  it('removes a folder, announces it, and moves focus to the next Remove button', () => {
    render(<Harness initial={['\\Alpha', '\\Beta']} />);
    fireEvent.click(removeButton('\\Alpha'));
    expect(rowFor('\\alpha')).toBeUndefined();
    expect(statusText()).toBe('Removed \\Alpha. 1 folder approved.');
    expect(document.activeElement).toBe(removeButton('\\Beta'));
  });

  it('moves focus to the previous Remove button when the last folder is removed', () => {
    render(<Harness initial={['\\Alpha', '\\Beta']} />);
    fireEvent.click(removeButton('\\Beta'));
    expect(document.activeElement).toBe(removeButton('\\Alpha'));
  });

  it('moves focus to the input when the last folder is removed', () => {
    render(<Harness initial={['\\Alpha']} />);
    fireEvent.click(removeButton('\\Alpha'));
    expect(document.getElementById('solidworks-pdm-folder-list')).toBeNull();
    expect(document.activeElement).toBe(input());
  });

  it('moves from an empty input to the last chip, between chips, and removes with Backspace', () => {
    render(<Harness initial={['\\Alpha', '\\Beta']} />);
    input().focus();
    expect(press('Backspace')).toBe(false);
    expect(document.activeElement).toBe(removeButton('\\Beta'));
    expect(statusText()).toBe('Last folder \\Beta. Press Enter or Backspace to remove it.');
    fireEvent.keyDown(removeButton('\\Beta'), { key: 'ArrowLeft' });
    expect(document.activeElement).toBe(removeButton('\\Alpha'));
    fireEvent.keyDown(removeButton('\\Alpha'), { key: 'ArrowRight' });
    fireEvent.keyDown(removeButton('\\Beta'), { key: 'ArrowRight' });
    expect(document.activeElement).toBe(input());
    fireEvent.keyDown(removeButton('\\Alpha'), { key: 'Backspace' });
    expect(rowFor('\\alpha')).toBeUndefined();
  });

  it('does not remove folders while Backspace is held down', () => {
    render(<Harness initial={['\\Alpha', '\\Beta']} />);
    input().focus();
    fireEvent.keyDown(input(), { key: 'Backspace', repeat: true });
    expect(document.activeElement).toBe(input());
    press('Backspace');
    fireEvent.keyDown(removeButton('\\Beta'), { key: 'Backspace', repeat: true });
    expect(rowFor('\\beta')).toBeTruthy();
  });

  it('returns from a chip to the input with Escape without closing an outer dialog', () => {
    const outer = vi.fn();
    render(
      <div onKeyDown={outer}>
        <Harness initial={['\\Alpha']} />
      </div>,
    );
    removeButton('\\Alpha').focus();
    fireEvent.keyDown(removeButton('\\Alpha'), { key: 'Escape' });
    expect(document.activeElement).toBe(input());
    expect(outer).not.toHaveBeenCalled();
  });

  it('keeps Backspace in an empty input when there are no folders', () => {
    render(<Harness />);
    input().focus();
    expect(press('Backspace')).toBe(true);
    expect(document.activeElement).toBe(input());
  });

  it('exposes each folder as a list item with a named Remove button', () => {
    render(<Harness initial={['\\13_Ragtime']} />);
    const list = document.getElementById('solidworks-pdm-folder-list');
    expect(list?.tagName).toBe('UL');
    expect(list?.querySelectorAll('li[data-sw-folder]').length).toBe(1);
    expect(removeButton('\\13_Ragtime').getAttribute('data-sw-action')).toBe('remove-folder');
  });

  it('disables the input, Add folder and every Remove button while disabled', () => {
    render(<Harness initial={['\\Alpha']} disabled />);
    expect(input().disabled).toBe(true);
    expect(addButton().hasAttribute('disabled')).toBe(true);
    expect(removeButton('\\Alpha').hasAttribute('disabled')).toBe(true);
  });

  it('is a plain text box without suggestions and never lists the vault', () => {
    render(<Harness />);
    input().focus();
    fireEvent.click(input());
    typeFolder('\\04');
    expect(input().getAttribute('role')).toBeNull();
    expect(popup()).toBeNull();
    expect(listChildren).not.toHaveBeenCalled();
  });
});

describe('SolidWorksApprovedFolders vault suggestions', () => {
  beforeEach(() => {
    listChildren.mockReset();
    listChildren.mockImplementation(async (_tool: string, _pdm: string, path: string) => {
      const items = TREE[path];
      if (!items) throw new ListError('That vault folder no longer exists.', 'pdm_not_found');
      return { items, total: items.length };
    });
  });

  it('lists the vault root with each folder’s rights for the current PDM user', async () => {
    render(<Harness suggestions />);
    expect(input().getAttribute('role')).toBe('combobox');
    expect(input().getAttribute('aria-describedby')).toBe('solidworks-pdm-folder-hint');
    await openRoot();
    expect(listChildren).toHaveBeenCalledWith('sw-a', 'pdm-a', '\\', expect.any(AbortSignal));
    expect(input().getAttribute('aria-expanded')).toBe('true');
    const states = Array.from(document.querySelectorAll('li[role="option"]')).map((row) =>
      row.getAttribute('data-sw-suggest-state'),
    );
    expect(states).toEqual(['read_only', 'can_change', 'no_access', 'not_checked']);
    expect(optionFor('\\04_custom')?.getAttribute('aria-disabled')).toBeNull();
    expect(optionFor('\\04_custom')?.textContent).toContain('can still be approved');
  });

  it('moves through options with the arrows, wrapping through the typed text', async () => {
    render(<Harness suggestions />);
    await openRoot();
    expect(input().getAttribute('aria-activedescendant')).toBeNull();
    press('ArrowDown');
    expect(input().getAttribute('aria-activedescendant')).toBe(optionFor('\\04_custom')?.id);
    expect(optionFor('\\04_custom')?.getAttribute('aria-selected')).toBe('true');
    press('ArrowUp');
    expect(input().getAttribute('aria-activedescendant')).toBeNull();
    press('ArrowUp');
    expect(input().getAttribute('aria-activedescendant')).toBe(optionFor('\\unchecked')?.id);
  });

  it('adds a read-only folder on Enter with a warning and keeps the list open', async () => {
    const onFoldersChange = vi.fn();
    render(<Harness suggestions onFoldersChange={onFoldersChange} />);
    await openRoot();
    press('ArrowDown');
    press('Enter');
    expect(onFoldersChange).toHaveBeenLastCalledWith(['\\04_Custom']);
    expect(feedback()?.getAttribute('data-tone')).toBe('warning');
    expect(feedback()?.textContent).toBe(
      "Added \\04_Custom. The PDM user can't add or check out files there yet, so the AI's changes there will be refused until you grant those rights in the PDM Administration tool.",
    );
    expect(statusText()).toBe(
      "Added \\04_Custom. 1 folder approved. The PDM user can't change files there yet.",
    );
    expect(input().value).toBe('');
    expect(optionFor('\\04_custom')?.getAttribute('data-sw-suggest-state')).toBe('approved');
    expect(input().getAttribute('aria-activedescendant')).toBeNull();
  });

  it('opens a folder with the right arrow and adds siblings with clicks', async () => {
    const onFoldersChange = vi.fn();
    render(<Harness suggestions onFoldersChange={onFoldersChange} />);
    await openRoot();
    press('ArrowDown');
    press('ArrowRight');
    expect(input().value).toBe('\\04_Custom\\');
    await waitFor(() => expect(optionFor('\\04_custom\\03_light_body')).toBeTruthy());
    fireEvent.click(optionFor('\\04_custom\\03_light_body') as HTMLElement);
    expect(onFoldersChange).toHaveBeenLastCalledWith(['\\04_Custom\\03_Light_Body']);
    expect(feedback()).toBeNull();
    expect(input().value).toBe('\\04_Custom\\');
    fireEvent.click(optionFor('\\04_custom\\01_family-platform') as HTMLElement);
    expect(onFoldersChange).toHaveBeenLastCalledWith([
      '\\04_Custom\\03_Light_Body',
      '\\04_Custom\\01_Family-Platform',
    ]);
    expect(optionFor('\\04_custom\\03_light_body')?.getAttribute('aria-disabled')).toBe('true');
  });

  it('opens a folder from its chevron and filters by the typed prefix', async () => {
    render(<Harness suggestions />);
    await openRoot();
    const chevron = optionFor('\\04_custom')?.querySelector('.sw-suggest-drill') as HTMLElement;
    fireEvent.click(chevron);
    await waitFor(() => expect(optionFor('\\04_custom\\02_mounting_bodies')).toBeTruthy());
    typeFolder('\\04_Custom\\02');
    expect(document.querySelectorAll('li[role="option"]')).toHaveLength(1);
    expect(optionFor('\\99_archive')).toBeUndefined();
  });

  it('completes the active folder with Tab instead of leaving the field', async () => {
    render(<Harness suggestions />);
    await openRoot();
    press('ArrowDown');
    expect(press('Tab')).toBe(false);
    expect(input().value).toBe('\\04_Custom\\');
  });

  it('closes the list on Escape, then clears the entry, without closing an outer dialog', async () => {
    const outer = vi.fn();
    render(
      <div onKeyDown={outer}>
        <Harness suggestions />
      </div>,
    );
    typeFolder('\\04');
    await waitFor(() => expect(popup()).not.toBeNull());
    press('Escape');
    expect(popup()).toBeNull();
    expect(input().value).toBe('\\04');
    press('Escape');
    expect(input().value).toBe('');
    expect(outer).not.toHaveBeenCalled();
  });

  it('adds the typed path on Enter when nothing in the list is chosen', async () => {
    const onFoldersChange = vi.fn();
    render(<Harness suggestions onFoldersChange={onFoldersChange} />);
    typeFolder('\\04_Custom\\Zeta');
    await waitFor(() =>
      expect(popupMessage()).toBe(
        'No folder in \\04_Custom starts with “Zeta”. Press Enter to add \\04_Custom\\Zeta anyway.',
      ),
    );
    expect(input().getAttribute('aria-expanded')).toBe('false');
    press('Enter');
    expect(onFoldersChange).toHaveBeenLastCalledWith(['\\04_Custom\\Zeta']);
    expect(input().value).toBe('');
  });

  it('warns when a typed path matches a read-only folder in the list', async () => {
    render(<Harness suggestions />);
    typeFolder('\\04_Custom\\01_Family-Platform');
    await waitFor(() => expect(optionFor('\\04_custom\\01_family-platform')).toBeTruthy());
    press('Enter');
    expect(feedback()?.getAttribute('data-tone')).toBe('warning');
  });

  it('refuses approved and included folders with the existing messages', async () => {
    const onFoldersChange = vi.fn();
    render(
      <Harness
        suggestions
        initial={['\\13_Ragtime', '\\04_Custom']}
        onFoldersChange={onFoldersChange}
      />,
    );
    await openRoot();
    expect(optionFor('\\13_ragtime')?.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(optionFor('\\13_ragtime') as HTMLElement);
    expect(feedback()?.textContent).toBe('\\13_Ragtime is already in the list.');
    typeFolder('\\04_Custom\\');
    await waitFor(() =>
      expect(optionFor('\\04_custom\\03_light_body')?.getAttribute('data-sw-suggest-state')).toBe(
        'included',
      ),
    );
    press('ArrowDown');
    press('Enter');
    expect(feedback()?.getAttribute('data-tone')).toBe('info');
    expect(onFoldersChange).not.toHaveBeenCalled();
  });

  it('ignores the answer for a folder the entry has moved away from', async () => {
    const pending: Record<string, (items: PdmFolderChild[]) => void> = {};
    listChildren.mockImplementation(
      (_tool: string, _pdm: string, path: string) =>
        new Promise((resolve) => {
          pending[path] = (items) => resolve({ items, total: items.length });
        }),
    );
    render(<Harness suggestions />);
    typeFolder('\\A\\');
    await waitFor(() => expect(pending['\\A']).toBeDefined());
    typeFolder('\\B\\');
    await waitFor(() => expect(pending['\\B']).toBeDefined());
    await act(async () => pending['\\B']([child('\\B\\Bee', FULL)]));
    await act(async () => pending['\\A']([child('\\A\\Aye', FULL)]));
    expect(optionFor('\\b\\bee')).toBeTruthy();
    expect(optionFor('\\a\\aye')).toBeUndefined();
  });

  it('says when only part of a large folder is loaded', async () => {
    listChildren.mockResolvedValue({ items: TREE['\\'], total: 700 });
    render(<Harness suggestions />);
    await openRoot();
    expect(popupMessage()).toBe(
      'Showing the first 4 of 700 folders in \\. Keep typing to narrow the list, or type the full path and press Enter.',
    );
    typeFolder('Zz');
    expect(popupMessage()).toBe(
      'No loaded folder in \\ starts with “Zz”. Only the first 4 of 700 are loaded, so it may still exist. Press Enter to add \\Zz anyway.',
    );
  });

  it('treats a folder missing from the vault as no match', async () => {
    render(<Harness suggestions />);
    typeFolder('\\Gone\\x');
    await waitFor(() =>
      expect(popupMessage()).toBe(
        'No folder in \\Gone starts with “x”. Press Enter to add \\Gone\\x anyway.',
      ),
    );
  });

  it('shows a load error with Try again', async () => {
    listChildren.mockRejectedValueOnce(
      new ListError('The SolidWorks agent did not answer in time. Try again.', 'operation_timeout'),
    );
    render(<Harness suggestions />);
    fireEvent.click(input());
    await waitFor(() =>
      expect(popupMessage()).toBe(
        "Couldn't load folders in \\: The SolidWorks agent did not answer in time. Try again. Press the down arrow or choose Try again to retry. You can still type a path and press Enter.",
      ),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    await waitFor(() => expect(optionFor('\\04_custom')).toBeTruthy());
    expect(listChildren).toHaveBeenCalledTimes(2);
  });

  it('retries a failed load with the down arrow', async () => {
    listChildren.mockRejectedValueOnce(new ListError('Busy.', 'operation_timeout'));
    render(<Harness suggestions />);
    fireEvent.click(input());
    await waitFor(() => expect(popupMessage()).toContain("Couldn't load folders"));
    press('ArrowDown');
    await waitFor(() => expect(optionFor('\\04_custom')).toBeTruthy());
    expect(listChildren).toHaveBeenCalledTimes(2);
  });

  it('adds a folder with unchecked rights without claiming it is read only', async () => {
    render(<Harness suggestions />);
    await openRoot();
    fireEvent.click(optionFor('\\unchecked') as HTMLElement);
    expect(rowFor('\\unchecked')).toBeTruthy();
    expect(feedback()).toBeNull();
    expect(statusText()).toBe('Added \\Unchecked. 1 folder approved.');
  });

  it('leaves Shift+Tab to the browser', async () => {
    render(<Harness suggestions />);
    await openRoot();
    press('ArrowDown');
    expect(fireEvent.keyDown(input(), { key: 'Tab', shiftKey: true })).toBe(true);
    expect(input().value).toBe('');
    expect(popup()).toBeNull();
  });

  it('closes the list when the field loses focus', async () => {
    render(<Harness suggestions />);
    await openRoot();
    fireEvent.blur(input());
    expect(popup()).toBeNull();
    expect(input().getAttribute('aria-expanded')).toBe('false');
    expect(input().getAttribute('aria-controls')).toBeNull();
  });

  it('keeps the feedback above the list so the open list never covers it', async () => {
    render(<Harness suggestions />);
    await openRoot();
    press('ArrowDown');
    press('Enter');
    const combobox = document.getElementById('solidworks-pdm-folder-combobox') as HTMLElement;
    const order = Array.from(combobox.children).map((element) => element.id);
    expect(order).toEqual([
      'solidworks-pdm-folder-box',
      'solidworks-pdm-folder-feedback',
      'solidworks-pdm-folder-suggest',
    ]);
  });
});
