import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { solidworksApi } from '../api/solidworks';
import { SolidWorksPdmFolderBrowser } from './SolidWorksPdmFolderBrowser';

vi.mock('../api/solidworks', () => ({ solidworksApi: { listPdmFolders: vi.fn() } }));

describe('SolidWorksPdmFolderBrowser', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.mocked(solidworksApi.listPdmFolders).mockReset();
  });

  it('loads only when opened', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue({
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: null,
    });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    expect(solidworksApi.listPdmFolders).not.toHaveBeenCalled();
    expect(document.getElementById('solidworks-pdm-folder-browser-status')?.textContent).toBe('');
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    await waitFor(() => expect(solidworksApi.listPdmFolders).toHaveBeenCalledOnce());
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledWith(
      's',
      expect.objectContaining({ path: '\\' }),
    );
  });

  it('re-announces repeated reloads and explains disabled approval', async () => {
    const { rerender } = render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    const status = document.getElementById('solidworks-pdm-folder-browser-status')!;
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    const first = status.firstElementChild;
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    expect(status.textContent).toBe('Reloading folders.');
    expect(status.firstElementChild).not.toBe(first);
    rerender(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
        disabled
      />,
    );
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue({
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: null,
    });
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    expect(await screen.findByText('Wait for the current action to finish.')).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });

  it('shows an error when a restarted snapshot is also stale', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue({
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: { code: 'stale_snapshot', message: 'changed' },
    });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    expect(
      await screen.findByText(
        'Could not load folders: changed. Choose Reload folders to try again.',
      ),
    ).toBeTruthy();
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(2);
  });

  it('ignores a listing that was in flight before reload', async () => {
    const empty = {
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: null,
    };
    let finish!: (value: Awaited<ReturnType<typeof solidworksApi.listPdmFolders>>) => void;
    vi.mocked(solidworksApi.listPdmFolders)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            finish = resolve;
          }),
      )
      .mockResolvedValueOnce(empty);
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    await waitFor(() => expect(solidworksApi.listPdmFolders).toHaveBeenCalledOnce());
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    await screen.findByText('No subfolders.');
    await act(async () =>
      finish({
        ...empty,
        total: 1,
        items: [{ name: 'Old', vault_path: '\\Old', rights: null, write_ready: false }],
      }),
    );
    expect(screen.queryByRole('button', { name: /Old/ })).toBeNull();
  });

  it('navigates, approves a writable folder, and focuses the filter', async () => {
    const writable = { read: true, add_files: true, check_out: true, add_folders: true };
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => ({
      items:
        request.path === '\\'
          ? [
              { name: 'A', vault_path: '\\A', rights: writable, write_ready: true },
              {
                name: 'Hidden',
                vault_path: '\\Hidden',
                rights: { ...writable, read: false },
                write_ready: false,
              },
            ]
          : request.path === '\\A'
            ? [{ name: 'B', vault_path: '\\A\\B', rights: writable, write_ready: true }]
            : [],
      offset: 0,
      next_offset: null,
      total: request.path === '\\' ? 2 : request.path === '\\A' ? 1 : 0,
      snapshot: 's',
      error: null,
    }));
    const onChange = vi.fn();
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    const hidden = await screen.findByRole('button', { name: /Hidden No access/ });
    expect(hidden.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(hidden);
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(1);
    const filter = screen.getByRole('textbox', { name: 'Filter folders' });
    fireEvent.change(filter, { target: { value: 'Hidden' } });
    fireEvent.keyDown(filter, { key: 'Enter' });
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(1);
    fireEvent.change(filter, { target: { value: '' } });
    fireEvent.click(screen.getByRole('button', { name: /A Can change/ }));
    fireEvent.click(await screen.findByRole('button', { name: /B Can change/ }));
    await waitFor(() =>
      expect(solidworksApi.listPdmFolders).toHaveBeenCalledWith(
        's',
        expect.objectContaining({ path: '\\A\\B' }),
      ),
    );
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Approve this folder' }));
    expect(onChange).toHaveBeenCalledWith(['\\A\\B']);
    expect(document.getElementById('solidworks-pdm-folder-browser-status')?.textContent).toContain(
      'Added \\A\\B.',
    );
    expect(document.activeElement).toBe(screen.getByRole('textbox', { name: 'Filter folders' }));
  });

  it('shows the replacement notice when approving a parent', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => ({
      items:
        request.path === '\\'
          ? [
              {
                name: 'A',
                vault_path: '\\A',
                rights: { read: true, add_files: true, check_out: true, add_folders: true },
                write_ready: true,
              },
            ]
          : [],
      offset: 0,
      next_offset: null,
      total: request.path === '\\' ? 1 : 0,
      snapshot: null,
      error: null,
    }));
    const onChange = vi.fn();
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={['\\A\\B']}
        vaultRoot={null}
        onChange={onChange}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    fireEvent.click(await screen.findByRole('button', { name: /A Can change/ }));
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Approve this folder' }));
    expect(onChange).toHaveBeenCalledWith(['\\A']);
    expect(
      document.getElementById('solidworks-pdm-folder-browser-feedback')?.textContent,
    ).toContain('Replaced \\A\\B with \\A');
  });

  it('re-announces identical approval refusals', async () => {
    const rights = { read: true, add_files: true, check_out: true, add_folders: true };
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => ({
      items:
        request.path === '\\' ? [{ name: 'A', vault_path: '\\A', rights, write_ready: true }] : [],
      offset: 0,
      next_offset: null,
      total: request.path === '\\' ? 1 : 0,
      snapshot: null,
      error: null,
    }));
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={Array.from({ length: 25 }, (_, index) => `\\Existing${index}`)}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    fireEvent.click(await screen.findByRole('button', { name: /A Can change/ }));
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Approve this folder' }));
    const status = document.getElementById('solidworks-pdm-folder-browser-status')!;
    const first = status.firstElementChild;
    expect(status.textContent).toContain('25 folders');
    fireEvent.click(screen.getByRole('button', { name: 'Approve this folder' }));
    expect(status.firstElementChild).not.toBe(first);
    expect(status.textContent).toContain('25 folders');
  });

  it('restarts one stale snapshot and caps pages with a truncation notice', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => {
      if (request.offset === 1 && vi.mocked(solidworksApi.listPdmFolders).mock.calls.length === 2) {
        return {
          items: [],
          offset: 1,
          next_offset: null,
          total: 0,
          snapshot: null,
          error: { code: 'stale_snapshot', message: 'stale' },
        };
      }
      return {
        items: [
          {
            name: `F${request.offset}`,
            vault_path: `\\F${request.offset}`,
            rights: null,
            write_ready: false,
          },
        ],
        offset: request.offset,
        next_offset: request.offset + 1,
        total: 20,
        snapshot: 'snap',
        error: null,
      };
    });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    await screen.findByText(/Showing the first 10 of 20 folders/);
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(12);
    expect(vi.mocked(solidworksApi.listPdmFolders).mock.calls[1][1]).toMatchObject({
      offset: 1,
      snapshot: 'snap',
    });
    expect(vi.mocked(solidworksApi.listPdmFolders).mock.calls[2][1]).toMatchObject({
      offset: 0,
      snapshot: null,
    });
  });

  it('reports errors and reloads the open folder', async () => {
    vi.mocked(solidworksApi.listPdmFolders)
      .mockRejectedValueOnce(new Error('offline'))
      .mockResolvedValueOnce({
        items: [],
        offset: 0,
        next_offset: null,
        total: 0,
        snapshot: null,
        error: null,
      });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    await screen.findByText('Could not load folders: offline. Choose Reload folders to try again.');
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    await screen.findByText('No subfolders.');
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(2);
  });

  it('retains rights during reload and refreshes the current folder from its parent', async () => {
    const rights = { read: true, add_files: false, check_out: false, add_folders: false };
    const empty = {
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: null,
    };
    let finishParent!: (value: Awaited<ReturnType<typeof solidworksApi.listPdmFolders>>) => void;
    let rootRequests = 0;
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => {
      if (request.path !== '\\') return empty;
      rootRequests += 1;
      if (rootRequests === 1)
        return {
          ...empty,
          total: 1,
          items: [{ name: 'A', vault_path: '\\A', rights, write_ready: false }],
        };
      return new Promise((resolve) => {
        finishParent = resolve;
      });
    });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    fireEvent.click(await screen.findByRole('button', { name: /A Read only/ }));
    await screen.findByText(
      "The PDM user can read this folder but can't add and check out files here.",
    );
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    await waitFor(() => expect(rootRequests).toBe(2));
    expect(
      screen.getByText("The PDM user can read this folder but can't add and check out files here."),
    ).toBeTruthy();
    expect(
      screen.queryByText(
        "Rights for this folder aren't checked. Choose Reload folders to try again.",
      ),
    ).toBeNull();
    finishParent({
      ...empty,
      total: 1,
      items: [
        {
          name: 'A',
          vault_path: '\\A',
          rights: { ...rights, add_files: true, check_out: true },
          write_ready: true,
        },
      ],
    });
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
  });

  it.each(['no-read', 'failure'] as const)('handles a parent refresh: %s', async (outcome) => {
    const rights = { read: true, add_files: true, check_out: true, add_folders: true };
    const empty = {
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: null,
    };
    let roots = 0;
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => {
      if (request.path !== '\\') return empty;
      roots += 1;
      if (roots > 1 && outcome === 'failure') throw new Error('offline');
      return {
        ...empty,
        total: 1,
        items: [
          {
            name: 'A',
            vault_path: '\\A',
            rights: { ...rights, read: roots === 1 },
            write_ready: true,
          },
        ],
      };
    });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    fireEvent.click(await screen.findByRole('button', { name: /A Can change/ }));
    await waitFor(() =>
      expect(
        (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
      ).toBe(false),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    if (outcome === 'failure') {
      await waitFor(() =>
        expect(document.getElementById('solidworks-pdm-folder-browser-status')?.textContent).toBe(
          'Could not reload folders.',
        ),
      );
      expect(screen.queryByText(/Could not load folders: offline/)).toBeNull();
    } else {
      expect(await screen.findByText("The PDM user can't see inside this folder.")).toBeTruthy();
    }
  });

  it('names the current folder in the truncation notice', async () => {
    const rights = { read: true, add_files: true, check_out: true, add_folders: true };
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) =>
      request.path === '\\'
        ? {
            items: [{ name: 'A', vault_path: '\\A', rights, write_ready: true }],
            offset: 0,
            next_offset: null,
            total: 1,
            snapshot: null,
            error: null,
          }
        : { items: [], offset: 0, next_offset: null, total: 5, snapshot: null, error: null },
    );
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    fireEvent.click(await screen.findByRole('button', { name: /A Can change/ }));
    expect(await screen.findByText(/Showing the first 0 of 5 folders in \\A/)).toBeTruthy();
  });

  it('ignores a parent refresh after navigating elsewhere', async () => {
    const rights = { read: true, add_files: false, check_out: false, add_folders: false };
    const empty = {
      items: [],
      offset: 0,
      next_offset: null,
      total: 0,
      snapshot: null,
      error: null,
    };
    const listing = {
      ...empty,
      total: 1,
      items: [{ name: 'A', vault_path: '\\A', rights, write_ready: false }],
    };
    let finishParent!: (value: Awaited<ReturnType<typeof solidworksApi.listPdmFolders>>) => void;
    let rootRequests = 0;
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => {
      if (request.path !== '\\') return empty;
      rootRequests += 1;
      if (rootRequests === 2)
        return new Promise((resolve) => {
          finishParent = resolve;
        });
      return listing;
    });
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={[]}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    fireEvent.click(await screen.findByRole('button', { name: /A Read only/ }));
    await screen.findByText(
      "The PDM user can read this folder but can't add and check out files here.",
    );
    fireEvent.click(screen.getByRole('button', { name: 'Reload folders' }));
    await waitFor(() => expect(rootRequests).toBe(2));
    await waitFor(() =>
      expect((screen.getByRole('button', { name: '..' }) as HTMLButtonElement).disabled).toBe(
        false,
      ),
    );
    fireEvent.click(screen.getByRole('button', { name: '..' }));
    await screen.findByRole('button', { name: /A Read only/ });
    await act(async () =>
      finishParent({ ...listing, items: [{ ...listing.items[0], write_ready: true }] }),
    );
    expect(screen.getByRole('button', { name: /A Read only/ })).toBeTruthy();
  });

  it.each([
    ['Approved', ['\\Approved'], true, 'Approved', 'This folder is already approved.'],
    ['Parent\\Child', ['\\Parent'], true, 'Included', 'Already included in \\Parent.'],
    [
      'Unknown',
      [],
      null,
      'Not checked',
      "Rights for this folder aren't checked. Choose Reload folders to try again.",
    ],
    ['Bad:name', [], true, "Can't approve", 'Folder paths cannot contain'],
    [
      'Readonly',
      [],
      false,
      'Read only',
      "The PDM user can read this folder but can't add and check out files here.",
    ],
  ])('explains why %s cannot be approved', async (name, folders, writable, badge, reason) => {
    const folder = `\\${name}`;
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) => ({
      items:
        request.path === '\\'
          ? [
              {
                name,
                vault_path: folder,
                rights:
                  writable === null
                    ? null
                    : { read: true, add_files: writable, check_out: writable, add_folders: false },
                write_ready: Boolean(writable),
              },
            ]
          : [],
      offset: 0,
      next_offset: null,
      total: request.path === '\\' ? 1 : 0,
      snapshot: null,
      error: null,
    }));
    render(
      <SolidWorksPdmFolderBrowser
        toolId="s"
        pdmToolId="p"
        vaultName="Vault"
        folders={folders}
        vaultRoot={null}
        onChange={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: '/ Vault' }));
    expect(
      screen.getByText("Open a folder to approve it. The whole vault can't be approved."),
    ).toBeTruthy();
    fireEvent.click(
      await screen.findByRole('button', {
        name: new RegExp(badge.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')),
      }),
    );
    expect(
      await screen.findByText(
        (text, element) => element?.className === 'field-help' && text.includes(reason),
      ),
    ).toBeTruthy();
    expect(
      (screen.getByRole('button', { name: 'Approve this folder' }) as HTMLButtonElement).disabled,
    ).toBe(true);
  });
});
