import { solidworksApi, type PdmFolderChild } from '../api/solidworks';

const MAX_PAGES = 10;

export class PdmFolderListError extends Error {
  constructor(
    message: string,
    public readonly code: string,
  ) {
    super(message);
    this.name = 'PdmFolderListError';
  }
}

export async function listAllPdmFolderChildren(
  toolId: string,
  pdmToolId: string,
  vaultPath: string,
  signal?: AbortSignal,
): Promise<{ items: PdmFolderChild[]; total: number }> {
  let items: PdmFolderChild[] = [];
  let offset = 0;
  let snapshot: string | null = null;
  let restarted = false;
  let pages = 0;
  let total = 0;
  while (pages < MAX_PAGES) {
    const result = await solidworksApi.listPdmFolders(
      toolId,
      { pdm_tool_id: pdmToolId, path: vaultPath, offset, snapshot },
      signal,
    );
    if (result.error?.code === 'stale_snapshot' && !restarted) {
      restarted = true;
      items = [];
      offset = 0;
      snapshot = null;
      pages = 0;
      continue;
    }
    if (result.error) throw new PdmFolderListError(result.error.message, result.error.code);
    items.push(...result.items);
    total = result.total;
    snapshot = result.snapshot;
    pages += 1;
    if (result.next_offset === null) break;
    offset = result.next_offset;
  }
  return { items, total };
}
