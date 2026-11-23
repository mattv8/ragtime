import { useState } from 'react';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { SolidWorksApprovedFolders } from './SolidWorksApprovedFolders';

afterEach(cleanup);

interface HarnessProps {
  initial?: string[];
  disabled?: boolean;
  onFoldersChange?: (next: string[]) => void;
  onSubmit?: () => void;
}

function Harness({ initial = [], disabled, onFoldersChange, onSubmit }: HarnessProps) {
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
const statusText = () => document.getElementById('solidworks-pdm-folders-status')?.textContent;
const feedbackError = () => document.getElementById('solidworks-pdm-folder-input-error');

function typeFolder(value: string) {
  fireEvent.change(input(), { target: { value } });
}

describe('SolidWorksApprovedFolders', () => {
  it('shows folder rights badges and defaults unchecked folders', () => {
    render(
      <SolidWorksApprovedFolders
        folders={['\\Ready', '\\Missing', '\\Hidden', '\\Unknown']}
        vaultRoot={null}
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

  it('explains when browsing is disabled and invokes the enabled action', () => {
    const onBrowse = vi.fn();
    const props = { folders: [], vaultRoot: null, onChange: vi.fn(), onBrowse };
    const { rerender } = render(
      <SolidWorksApprovedFolders {...props} browseDisabledReason="Log in first." />,
    );
    const browse = screen.getByRole('button', { name: 'Browse vault…' });
    expect(browse.hasAttribute('disabled')).toBe(true);
    expect(browse.getAttribute('aria-describedby')).toBe('solidworks-pdm-folder-browse-help');
    expect(document.getElementById('solidworks-pdm-folder-browse-help')?.textContent).toBe(
      'Log in first.',
    );
    rerender(<SolidWorksApprovedFolders {...props} />);
    fireEvent.click(browse);
    expect(onBrowse).toHaveBeenCalledOnce();
  });

  it('names the group and exposes folder help', () => {
    render(<Harness />);
    expect(screen.getByRole('group', { name: 'Approved folders' })).toBeTruthy();
    expect(input().getAttribute('aria-describedby')).toBeNull();
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
    const notPrevented = fireEvent.keyDown(input(), { key: 'Enter' });
    expect(notPrevented).toBe(false);
    expect(onFoldersChange).toHaveBeenLastCalledWith(['\\13_Ragtime']);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('ignores other keys', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    typeFolder('\\13_Ragtime');
    fireEvent.keyDown(input(), { key: 'a' });
    expect(onFoldersChange).not.toHaveBeenCalled();
  });

  it('shows the empty-entry error and keeps the input value', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    fireEvent.click(addButton());
    expect(onFoldersChange).not.toHaveBeenCalled();
    expect(feedbackError()?.textContent).toBe('Enter a vault folder, for example \\13_Ragtime.');
    expect(feedbackError()?.getAttribute('data-tone')).toBe('error');
  });

  it('marks an invalid entry, links the error, keeps the value, and adds nothing', () => {
    const onFoldersChange = vi.fn();
    render(<Harness onFoldersChange={onFoldersChange} />);
    typeFolder('\\13_Ragtime\\..\\x');
    fireEvent.click(addButton());
    expect(onFoldersChange).not.toHaveBeenCalled();
    expect(feedbackError()?.textContent).toBe(
      'Folder paths cannot contain “..” or any of * ? " < > | :',
    );
    expect(input().getAttribute('aria-invalid')).toBe('true');
    expect(input().getAttribute('aria-describedby')).toContain('solidworks-pdm-folder-input-error');
    expect(input().value).toBe('\\13_Ragtime\\..\\x');
  });

  it('reports a covered folder as info, not as an invalid entry', () => {
    render(<Harness initial={['\\13_Ragtime']} />);
    typeFolder('\\13_Ragtime\\Rev');
    fireEvent.click(addButton());
    expect(feedbackError()?.textContent).toBe(
      '\\13_Ragtime\\Rev is already covered by \\13_Ragtime.',
    );
    expect(feedbackError()?.getAttribute('data-tone')).toBe('info');
    expect(input().getAttribute('aria-invalid')).toBe('false');
    expect(input().value).toBe('\\13_Ragtime\\Rev');
  });

  it('announces a converted view-root prefix in the status line', () => {
    render(<Harness />);
    typeFolder('C:\\PDM\\13_Ragtime');
    fireEvent.click(addButton());
    expect(statusText()).toBe(
      'Converted C:\\PDM\\13_Ragtime to \\13_Ragtime. Added \\13_Ragtime. 1 folder approved.',
    );
  });

  it('refuses a folder beyond the cap of 25', () => {
    const full = Array.from({ length: 25 }, (_, index) => `\\F${index}`);
    const onFoldersChange = vi.fn();
    render(<Harness initial={full} onFoldersChange={onFoldersChange} />);
    typeFolder('\\F99');
    fireEvent.click(addButton());
    expect(onFoldersChange).not.toHaveBeenCalled();
    expect(feedbackError()?.textContent).toBe('You can approve up to 25 folders.');
  });

  it('removes a folder, announces it, and moves focus to the next Remove button', () => {
    render(<Harness initial={['\\Alpha', '\\Beta']} />);
    fireEvent.click(removeButton('\\Alpha'));
    expect(rowFor('\\alpha')).toBeUndefined();
    expect(statusText()).toBe('Removed \\Alpha.');
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
});
