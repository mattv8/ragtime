import { beforeEach, describe, expect, it, vi } from 'vitest';
import { solidworksApi } from '../api/solidworks';
import { listAllPdmFolderChildren, PdmFolderListError } from './pdmFolderListing';

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
const item = (name: string) => ({
  name,
  vault_path: `\\${name}`,
  rights: null,
  write_ready: false,
});

describe('listAllPdmFolderChildren', () => {
  beforeEach(() => {
    vi.mocked(solidworksApi.listPdmFolders).mockReset();
  });

  it('accumulates pages, forwards snapshot and signal', async () => {
    const signal = new AbortController().signal;
    vi.mocked(solidworksApi.listPdmFolders)
      .mockResolvedValueOnce(
        listing({ items: [item('A')], total: 2, next_offset: 50, snapshot: 'snap' }),
      )
      .mockResolvedValueOnce(listing({ items: [item('B')], total: 2 }));
    expect(await listAllPdmFolderChildren('s', 'p', '\\Parent', signal)).toEqual({
      items: [item('A'), item('B')],
      total: 2,
    });
    expect(solidworksApi.listPdmFolders).toHaveBeenNthCalledWith(
      2,
      's',
      { pdm_tool_id: 'p', path: '\\Parent', offset: 50, snapshot: 'snap' },
      signal,
    );
  });

  it('caps at ten pages and retains total', async () => {
    vi.mocked(solidworksApi.listPdmFolders).mockImplementation(async (_tool, request) =>
      listing({
        items: Array.from({ length: 50 }, (_, i) => item(`${request.offset + i}`)),
        total: 600,
        next_offset: request.offset + 50,
      }),
    );
    const result = await listAllPdmFolderChildren('s', 'p', '\\');
    expect(result.items).toHaveLength(500);
    expect(result.total).toBe(600);
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(10);
  });

  it('restarts once on stale snapshot, discarding previous items', async () => {
    vi.mocked(solidworksApi.listPdmFolders)
      .mockResolvedValueOnce(listing({ items: [item('Old')], next_offset: 50, snapshot: 'old' }))
      .mockResolvedValueOnce(listing({ error: { code: 'stale_snapshot', message: 'stale' } }))
      .mockResolvedValueOnce(listing({ items: [item('New')], total: 1 }));
    expect(await listAllPdmFolderChildren('s', 'p', '\\')).toEqual({
      items: [item('New')],
      total: 1,
    });
    expect(vi.mocked(solidworksApi.listPdmFolders).mock.calls[2][1]).toMatchObject({
      offset: 0,
      snapshot: null,
    });
  });

  it.each(['pdm_not_found', 'stale_snapshot'])('propagates %s error code', async (code) => {
    vi.mocked(solidworksApi.listPdmFolders).mockResolvedValue(
      listing({ error: { code, message: 'failed' } }),
    );
    const promise = listAllPdmFolderChildren('s', 'p', '\\');
    await expect(promise).rejects.toBeInstanceOf(PdmFolderListError);
    await expect(promise).rejects.toMatchObject({ code, message: 'failed' });
    expect(solidworksApi.listPdmFolders).toHaveBeenCalledTimes(code === 'stale_snapshot' ? 2 : 1);
  });
});
