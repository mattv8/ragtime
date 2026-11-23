import { useState } from 'react';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { solidworksApi, type PdmFolderChild, type PdmFoldersResponse } from '../api/solidworks';
import { SolidWorksPdmFolderPicker } from './SolidWorksPdmFolderPicker';

vi.mock('../api/solidworks', () => ({ solidworksApi: { listPdmFolders: vi.fn() } }));
const list = vi.mocked(solidworksApi.listPdmFolders);
const rights = { read: true, add_files: true, check_out: true, add_folders: true };
const child = (name: string, overrides: Partial<PdmFolderChild> = {}): PdmFolderChild => ({
  name,
  vault_path: `\\${name}`,
  rights,
  write_ready: true,
  ...overrides,
});
const page = (
  items: PdmFolderChild[],
  overrides: Partial<PdmFoldersResponse> = {},
): PdmFoldersResponse => ({
  items,
  offset: 0,
  next_offset: null,
  total: items.length,
  snapshot: 'snapshot',
  error: null,
  ...overrides,
});
const node = (path: string) =>
  document.querySelector<HTMLElement>(
    `[data-sw-folder-node="${path.toLowerCase().replace(/\\/g, '\\\\')}"]`,
  )!;
const status = () => document.getElementById('solidworks-pdm-folder-picker-status')?.textContent;
const change = vi.fn();
const close = vi.fn();
function Harness({ initial = [] }: { initial?: string[] }) {
  const [folders, setFolders] = useState(initial);
  return (
    <SolidWorksPdmFolderPicker
      toolId="tool"
      pdmToolId="pdm"
      vaultName="Vault"
      folders={folders}
      vaultRoot={null}
      onChange={(next) => {
        change(next);
        setFolders(next);
      }}
      onClose={close}
    />
  );
}
beforeEach(() => {
  list.mockReset();
  change.mockReset();
  close.mockReset();
});
afterEach(cleanup);

describe('SolidWorksPdmFolderPicker', () => {
  it('focuses the first child, navigates, expands, collapses and adds without closing', async () => {
    list
      .mockResolvedValueOnce(page([child('A'), child('B')]))
      .mockResolvedValueOnce(page([child('Nested', { vault_path: '\\A\\Nested' })]));
    render(<Harness />);
    await waitFor(() => expect(document.activeElement).toBe(node('\\A')));
    expect(node('\\').getAttribute('aria-expanded')).toBe('true');
    fireEvent.keyDown(node('\\A'), { key: 'ArrowRight' });
    await screen.findByText('Nested');
    fireEvent.keyDown(node('\\A'), { key: 'ArrowRight' });
    expect(document.activeElement).toBe(node('\\A\\Nested'));
    fireEvent.keyDown(node('\\A\\Nested'), { key: 'ArrowLeft' });
    expect(document.activeElement).toBe(node('\\A'));
    fireEvent.keyDown(node('\\A'), { key: 'ArrowLeft' });
    expect(node('\\A').getAttribute('aria-expanded')).toBe('false');
    fireEvent.keyDown(node('\\A'), { key: 'End' });
    expect(document.activeElement).toBe(node('\\B'));
    fireEvent.keyDown(node('\\B'), { key: 'Home' });
    expect(document.activeElement).toBe(node('\\'));
    fireEvent.keyDown(node('\\'), { key: 'ArrowDown' });
    fireEvent.keyDown(node('\\A'), { key: 'Enter' });
    expect(change).toHaveBeenCalledWith(['\\A']);
    expect(status()).toBe('Added \\A. 1 folder approved.');
    expect(screen.getByRole('dialog')).toBeTruthy();
    expect(node('\\A').textContent).toContain('Approved');
  });

  it('shows disabled reasons in precedence order and prevents selection', async () => {
    list.mockResolvedValueOnce(
      page([
        child('Approved'),
        child('Inside', { vault_path: '\\Approved\\Inside' }),
        child('No', { rights: { ...rights, read: false }, write_ready: false }),
        child('Unknown', { rights: null, write_ready: false }),
        child('é'),
        child('Read', { write_ready: false }),
      ]),
    );
    render(<Harness initial={['\\Approved']} />);
    await screen.findByText('Read only');
    for (const label of [
      'Approved',
      'Included',
      'No access',
      'Not checked',
      "Can't approve",
      'Read only',
    ]) {
      const badge = screen.getByText(label, { selector: '.sw-tree-rights' });
      const item = badge.closest('[role=treeitem]')!;
      expect(item.getAttribute('aria-disabled')).toBe('true');
      fireEvent.click(badge.previousElementSibling!);
    }
    expect(change).not.toHaveBeenCalled();
    expect(node('\\No').hasAttribute('aria-expanded')).toBe(false);
    expect(screen.getByText("Can't approve").getAttribute('title')).toContain('ASCII');
  });

  it('pages with snapshot and moves focus to the first new folder', async () => {
    list
      .mockResolvedValueOnce(page([child('A')], { next_offset: 1 }))
      .mockResolvedValueOnce(page([child('B')], { offset: 1 }));
    render(<Harness />);
    const more = await screen.findByText('Load more folders');
    fireEvent.click(more);
    await waitFor(() => expect(document.activeElement).toBe(node('\\B')));
    expect(list.mock.calls[1][1]).toEqual({
      pdm_tool_id: 'pdm',
      path: '\\',
      offset: 1,
      snapshot: 'snapshot',
    });
    expect(node('\\A')).toBeTruthy();
  });

  it('uses an explicit Add button and names parents without their descendants', async () => {
    list
      .mockResolvedValueOnce(page([child('A')]))
      .mockResolvedValueOnce(page([child('Nested', { vault_path: '\\A\\Nested' })]));
    render(<Harness />);
    const label = await screen.findByText('A', { selector: '.sw-tree-label' });
    fireEvent.click(label);
    await screen.findByText('Nested');
    expect(change).not.toHaveBeenCalled();
    expect(node('\\A').getAttribute('aria-label')).toBe('A, Can change');
    expect(node('\\A').getAttribute('aria-selected')).toBeNull();
    expect(node('\\A').querySelector('[data-sw-rights="write_ready"]')?.getAttribute('title')).toBe(
      'Can add and check out files',
    );
    const add = screen.getByRole('button', { name: 'Add \\A' });
    add.focus();
    fireEvent.click(add);
    expect(change).toHaveBeenCalledWith(['\\A']);
    expect(screen.queryByRole('button', { name: 'Add \\A' })).toBeNull();
    expect(document.activeElement).toBe(node('\\A'));
    expect(screen.getByText('1 folder approved.')).toBeTruthy();
    expect(status()).toBe('Added \\A. 1 folder approved.');
  });

  it('reports disabled reasons and re-announces the same message', async () => {
    list.mockResolvedValueOnce(page([child('Read', { write_ready: false })]));
    render(<Harness />);
    await screen.findByText('Read only');
    const item = node('\\Read');
    act(() => item.focus());
    fireEvent.keyDown(item, { key: 'Enter' });
    expect(status()).toBe("The PDM user can't add and check out files here.");
    const previous = document.querySelector('#solidworks-pdm-folder-picker-status span');
    fireEvent.keyDown(item, { key: ' ' });
    expect(document.querySelector('#solidworks-pdm-folder-picker-status span')).not.toBe(previous);
    expect(document.getElementById('solidworks-pdm-folder-picker-error')?.textContent).toContain(
      "The PDM user can't add and check out files here.",
    );
    expect(
      document
        .getElementById('solidworks-pdm-folder-picker-error')!
        .compareDocumentPosition(document.getElementById('solidworks-pdm-folder-tree')!) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it('rechecks unknown children when re-expanding and moves left from an empty leaf', async () => {
    list
      .mockResolvedValueOnce(page([child('A')]))
      .mockResolvedValueOnce(
        page([child('Unknown', { vault_path: '\\A\\Unknown', rights: null, write_ready: false })]),
      )
      .mockResolvedValueOnce(page([]));
    render(<Harness />);
    await screen.findByText('A', { selector: '.sw-tree-label' });
    fireEvent.keyDown(node('\\A'), { key: 'ArrowRight' });
    await screen.findByText('Not checked');
    fireEvent.keyDown(node('\\A'), { key: 'ArrowLeft' });
    fireEvent.keyDown(node('\\A'), { key: 'ArrowRight' });
    await waitFor(() => expect(list).toHaveBeenCalledTimes(3));
    await waitFor(() => expect(node('\\A').hasAttribute('aria-expanded')).toBe(false));
    fireEvent.keyDown(node('\\A'), { key: 'ArrowLeft' });
    expect(document.activeElement).toBe(node('\\'));
  });

  it('does not steal focus after loading and uses browse fallback on close', async () => {
    let resolvePage!: (response: PdmFoldersResponse) => void;
    list.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolvePage = resolve;
        }),
    );
    const browse = document.createElement('button');
    browse.id = 'solidworks-pdm-folder-browse';
    document.body.appendChild(browse);
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const view = render(<Harness />);
    const done = screen.getByRole('button', { name: 'Done' });
    act(() => done.focus());
    await act(async () => {
      resolvePage(page([child('A')]));
    });
    expect(document.activeElement).toBe(done);
    opener.remove();
    view.unmount();
    expect(document.activeElement).toBe(browse);
    browse.remove();
  });

  it('loads a page only once while repeated activations are pending', async () => {
    let resolvePage!: (response: PdmFoldersResponse) => void;
    list.mockResolvedValueOnce(page([child('A')], { next_offset: 1 })).mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolvePage = resolve;
        }),
    );
    render(<Harness />);
    const more = await screen.findByText('Load more folders');
    act(() => more.focus());
    fireEvent.keyDown(more, { key: 'Enter' });
    expect(more.textContent).toBe('Loading…');
    expect(more.getAttribute('aria-disabled')).toBe('true');
    fireEvent.keyDown(more, { key: 'Enter' });
    fireEvent.click(more);
    expect(list).toHaveBeenCalledTimes(2);
    await act(async () => {
      resolvePage(page([child('B')], { offset: 1 }));
    });
    expect(document.querySelectorAll('[data-sw-folder-node="\\\\b"]')).toHaveLength(1);
    expect(list).toHaveBeenCalledTimes(2);
  });

  it('keeps a tab stop on the parent when stale recovery fails', async () => {
    list
      .mockResolvedValueOnce(page([child('A')], { next_offset: 1 }))
      .mockResolvedValueOnce(page([], { error: { code: 'stale_snapshot', message: 'stale' } }))
      .mockResolvedValueOnce(page([], { error: { code: 'failed', message: 'offline' } }));
    render(<Harness />);
    const more = await screen.findByText('Load more folders');
    act(() => more.focus());
    fireEvent.keyDown(more, { key: 'Enter' });
    await screen.findByText('Could not load folders: offline');
    const tabbable = document.querySelectorAll('[role="treeitem"][tabindex="0"]');
    expect(tabbable).toHaveLength(1);
    expect(tabbable[0]).toBe(node('\\'));
    expect(document.activeElement).toBe(node('\\'));
    const closeButton = screen.getByRole('button', { name: 'Close' });
    const focusable = Array.from(
      document.querySelectorAll<HTMLElement>(
        '#solidworks-pdm-folder-picker button:not([disabled]), #solidworks-pdm-folder-picker [tabindex]:not([tabindex="-1"])',
      ),
    );
    expect(focusable[focusable.indexOf(closeButton) + 1]).toBe(node('\\'));
    expect(
      node('\\').querySelector('[role="group"]')?.getAttribute('data-sw-folder-children'),
    ).toBe('\\');
  });

  it('reloads stale snapshots', async () => {
    list
      .mockResolvedValueOnce(page([child('A')], { next_offset: 1 }))
      .mockResolvedValueOnce(page([], { error: { code: 'stale_snapshot', message: 'stale' } }))
      .mockResolvedValueOnce(page([child('B')]));
    render(<Harness />);
    fireEvent.click(await screen.findByText('Load more folders'));
    await screen.findByText('B');
    expect(list.mock.calls[2][1].offset).toBe(0);
    expect(list.mock.calls[2][1].snapshot).toBeNull();
    expect(status()).toBe('The folder changed. Reloaded it.');
  });

  it('retries errors and shows an empty root', async () => {
    list
      .mockResolvedValueOnce(page([], { error: { code: 'failed', message: 'offline' } }))
      .mockResolvedValueOnce(page([]));
    render(<Harness />);
    await screen.findByText('Could not load folders: offline');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    await screen.findByText('This vault has no folders.');
    expect(node('\\').hasAttribute('aria-expanded')).toBe(false);
  });

  it('traps Tab, closes, restores focus, and aborts requests on unmount', async () => {
    list.mockImplementation(() => new Promise(() => {}));
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const view = render(<Harness />);
    const signal = list.mock.calls[0][2]!;
    const done = screen.getByRole('button', { name: 'Done' });
    done.focus();
    fireEvent.keyDown(done, { key: 'Tab' });
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Close' }));
    fireEvent.keyDown(document.activeElement!, { key: 'Tab', shiftKey: true });
    expect(document.activeElement).toBe(done);
    fireEvent.keyDown(done, { key: 'Escape' });
    fireEvent.click(done);
    expect(close).toHaveBeenCalledTimes(2);
    view.unmount();
    expect(signal.aborted).toBe(true);
    expect(document.activeElement).toBe(opener);
    opener.remove();
  });
});
