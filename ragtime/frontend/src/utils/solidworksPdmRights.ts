import type { PdmRightsItem, PdmRightsResponse } from '../api/solidworks';
import { vaultFolderKey } from './solidworksPdmLink';

export type FolderRightsState = 'write_ready' | 'missing_write' | 'not_found' | 'not_checked';

export interface RightsState {
  root: 'ok' | 'no_read' | 'not_checked';
  folders: Record<string, FolderRightsState>;
  extraWriteInfo: boolean;
  user: string | null;
}

function folderState(item: PdmRightsItem | undefined): FolderRightsState {
  if (!item?.checked) return 'not_checked';
  if (item.exists === false) return 'not_found';
  return item.write_ready ? 'write_ready' : 'missing_write';
}

export function resolveRightsState({
  response,
  allowChanges,
  folders,
}: {
  response: PdmRightsResponse | null;
  allowChanges: boolean;
  folders: string[];
}): RightsState {
  const root = response?.root;
  const items = new Map(response?.folders.map((item) => [vaultFolderKey(item.path), item]));
  const folderRights: Record<string, FolderRightsState> = {};
  for (const folder of folders) {
    const key = vaultFolderKey(folder);
    folderRights[key] = allowChanges ? folderState(items.get(key)) : 'not_checked';
  }
  const rootState =
    !root?.checked || root.exists === null
      ? 'not_checked'
      : root.exists && root.rights?.read
        ? 'ok'
        : 'no_read';
  return {
    root: rootState,
    folders: folderRights,
    extraWriteInfo: Boolean(
      !allowChanges &&
      root?.checked &&
      root.exists &&
      (root.rights?.add_files || root.rights?.check_out),
    ),
    user: response?.user ?? null,
  };
}
