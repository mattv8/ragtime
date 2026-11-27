import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { solidworksApi } from '../api/solidworks';
import { SolidWorksPdmFolderBrowser } from './SolidWorksPdmFolderBrowser';

vi.mock('../api/solidworks', () => ({ solidworksApi: { listPdmFolders: vi.fn() } }));

type Listing = Awaited<ReturnType<typeof solidworksApi.listPdmFolders>>;
const listing = (overrides: Partial<Listing> = {}): Listing => ({
  items: [],
  offset: 0,
  next_offset: null,
  total: 0,
  snapshot: null,
  error: null,
  ...overrides,
});
const rights = { read: true, add_files: false, check_out: false, add_folders: false };
const child = { name: 'A', vault_path: '\\A', rights, write_ready: false };
const browser = (onPick = vi.fn(), idPrefix?: string) => (
  <SolidWorksPdmFolderBrowser
    toolId="s"
    pdmToolId="p"
    vaultName="Vault"
    onPick={onPick}
    idPrefix={idPrefix}
  />
);

describe('SolidWorksPdmFolderBrowser', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.mocked(solidworksApi.listPdmFolders).mockReset();
  });

  it('opens by default, uses the prefix and selects a read-only parent', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) =>
      request.path === '\\' ? listing({ items: [child], total: 1 }) : listing(),
    );
    const onPick = vi.fn();
    render(browser(onPick, 'sandbox-browser'));
    expect(document.getElementById('sandbox-browser-help')?.textContent).toBe(
      'Open a folder, then choose Use as parent.',
    );
    expect(
      (screen.getByRole('button', { name: 'Use as parent' }) as HTMLButtonElement).disabled,
    ).toBe(true);
    fireEvent.click(await screen.findByRole('button', { name: 'A' }));
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Use as parent' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Use as parent' }));
    expect(onPick).toHaveBeenCalledWith('\\A');
    expect(document.getElementById('sandbox-browser-status')?.textContent).toBe(
      'Using \\A as the parent.',
    );
  });

  it('blocks selection when rights are not checked', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue(
      listing({ items: [{ ...child, rights: null }], total: 1 }),
    );
    render(browser());
    fireEvent.click(await screen.findByRole('button', { name: /A Not checked/ }));
    expect(
      await screen.findByText(
        "Rights for this folder aren't checked. Choose Reload folders to try again.",
      ),
    ).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Use as parent' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('reports errors and reloads the open folder', async () => {
    vi.mocked(solidworksApi.listPdmFolders)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce(listing());
    render(browser());
    await screen.findByText('Could not load folders: offline. Choose Reload folders to try again.');
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    await screen.findByText('No subfolders.');
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(2);
  });

  it('ignores a listing that was in flight before reload', async () => {
    let finish!: (value: Listing) => void;
    vi.mocked(solidworksApi.listPdmFolders)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      )
      .mockResolvedValueOnce(listing());
    render(browser());
    await waitFor(() => expect(solidworksApi.listPdmFolders).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    await screen.findByText('No subfolders.');
    await act(async () => finish(listing({ items: [child], total: 1 })));
    expect(screen.queryByRole('button', { name: 'A' })).toBeNull();
  });

  it('refreshes the current folder rights from its parent on reload', async () => {
    let roots = 0;
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => {
      if (request.path !== '\\') return listing();
      roots += 1;
      return listing({ items: [{ ...child, rights: { ...rights, read: roots === 1 } }], total: 1 });
    });
    render(browser());
    fireEvent.click(await screen.findByRole('button', { name: 'A' }));
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Use as parent' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    expect(await screen.findByText("The PDM user can't see inside this folder.")).toBeTruthy();
  });

  it('restarts stale snapshots and displays the pick truncation notice', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => {
      if (request.offset === 1 && vi.mocked(solidworksApi.listPdmFolders).mock.calls.length === 2)
        return listing({ error: { code: 'stale_snapshot', message: 'stale' } });
      return listing({
        items: [{ ...child, name: `F${request.offset}`, vault_path: `\\F${request.offset}` }],
        next_offset: request.offset + 1,
        total: 20,
        snapshot: 'snap',
      });
    });
    render(browser());
    const notice = await screen.findByText(/Showing the first 10 of 20 folders/);
    expect(notice.textContent).toContain(
      "To choose one that isn't listed, type its path in the filter box.",
    );
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(12);
  });

  it('shows an error when the restarted snapshot is also stale', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue(
      listing({ error: { code: 'stale_snapshot', message: 'changed' } }),
    );
    render(browser());
    await screen.findByText('Could not load folders: changed. Choose Reload folders to try again.');
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(2);
  });

  it('keeps IDs unique with two browsers mounted', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue(listing());
    render(
      <>
        {browser()}
        {browser(vi.fn(), 'sandbox-browser')}
      </>,
    );
    await screen.findAllByText('No subfolders.');
    const ids = Array.from(document.querySelectorAll('[id]'), (element) => element.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});
