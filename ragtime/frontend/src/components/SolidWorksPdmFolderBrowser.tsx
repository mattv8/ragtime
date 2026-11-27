import { useRef, useState } from 'react';
import type { PdmFolderChild } from '../api/solidworks';
import type { BrowseResponse } from '../types/api';
import { normalizeMountBrowserPath } from '../utils/mountPaths';
import { listAllPdmFolderChildren } from '../utils/pdmFolderListing';
import { ConstrainedPathBrowser } from './ConstrainedPathBrowser';
import { PdmMessage } from './shared/PdmMessage';

export interface SolidWorksPdmFolderBrowserProps {
  toolId: string;
  pdmToolId: string;
  vaultName: string;
  onPick: (vaultPath: string) => void;
  /** Prefix for every id in the browser; use a different one when two browsers are mounted. */
  idPrefix?: string;
  disabled?: boolean;
}

interface FolderBadge {
  label: string;
  tone: 'info' | 'muted' | 'success';
  title: string;
}

interface FolderState {
  reason?: string;
  badge?: FolderBadge;
}

const DEFAULT_ID_PREFIX = 'solidworks-pdm-folder-browser';
const PICK_ROOT_REASON = "Open a folder to use as the parent. The vault root can't be the parent.";
const PICK_HELP = 'Open a folder, then choose Use as parent.';
const NOT_CHECKED_REASON =
  "Rights for this folder aren't checked. Choose Reload folders to try again.";
const NOT_CHECKED_BADGE: FolderBadge = {
  label: 'Not checked',
  tone: 'muted',
  title: 'Rights for this folder have not been checked.',
};
const vaultPath = (path: string) => normalizeMountBrowserPath(path).replace(/\//g, '\\');
const browserPath = (path: string) => normalizeMountBrowserPath(path);

export function SolidWorksPdmFolderBrowser({
  toolId,
  pdmToolId,
  vaultName,
  onPick,
  idPrefix = DEFAULT_ID_PREFIX,
  disabled,
}: SolidWorksPdmFolderBrowserProps) {
  const [reloadCount, setReloadCount] = useState(0);
  const [children, setChildren] = useState<Record<string, PdmFolderChild>>({});
  const [truncations, setTruncations] = useState<Record<string, { shown: number; total: number }>>(
    {},
  );
  const [currentPath, setCurrentPath] = useState('/');
  const [status, setStatus] = useState({ text: '', revision: 0 });
  const generation = useRef(0);
  const pathRevision = useRef(0);
  const currentPathRef = useRef('/');
  const announcedPath = useRef<string | null>(null);

  function announce(text: string) {
    setStatus((previous) => ({ text, revision: previous.revision + 1 }));
  }

  async function browse(
    path: string,
    isCurrent: () => boolean = () => true,
  ): Promise<BrowseResponse> {
    const requestGeneration = generation.current;
    try {
      const { items, total } = await listAllPdmFolderChildren(toolId, pdmToolId, vaultPath(path));
      if (requestGeneration === generation.current && isCurrent()) {
        setChildren((previous) => ({
          ...previous,
          ...Object.fromEntries(items.map((item) => [browserPath(item.vault_path), item])),
        }));
        setTruncations((previous) => {
          const next = { ...previous };
          if (items.length < total) next[path] = { shown: items.length, total };
          else delete next[path];
          return next;
        });
      }
      return {
        path,
        entries: items.map((item) => ({
          name: item.name,
          path: browserPath(item.vault_path),
          is_dir: true,
        })),
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return {
        path,
        entries: [],
        error: `Could not load folders: ${message}. Choose Reload folders to try again.`,
      };
    }
  }

  function pickState(path: string): FolderState {
    if (path === '/') return { reason: PICK_ROOT_REASON };
    const child = children[browserPath(path)];
    if (!child?.rights) return { reason: NOT_CHECKED_REASON, badge: NOT_CHECKED_BADGE };
    if (!child.rights.read) return { reason: "The PDM user can't see inside this folder." };
    return {};
  }

  function select(path: string) {
    onPick(vaultPath(path));
    announce(`Using ${vaultPath(path)} as the parent.`);
  }

  const truncated = truncations[currentPath];
  return (
    <div id={idPrefix} data-testid={idPrefix} role="group" aria-labelledby={`${idPrefix}-label`}>
      <div id={`${idPrefix}-header`}>
        <span id={`${idPrefix}-label`} className="sw-group-label">
          Browse the vault
        </span>
        <button
          id={`${idPrefix}-reload`}
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={disabled}
          onClick={() => {
            generation.current += 1;
            setTruncations({});
            setReloadCount((count) => count + 1);
            announce('Reloading folders.');
            if (currentPath !== '/') {
              const revision = pathRevision.current;
              const requestGeneration = generation.current;
              const parent = currentPath.slice(0, currentPath.lastIndexOf('/')) || '/';
              void browse(parent, () => pathRevision.current === revision).then((result) => {
                if (
                  result.error &&
                  pathRevision.current === revision &&
                  generation.current === requestGeneration
                ) {
                  announce('Could not reload folders.');
                }
              });
            }
          }}
        >
          Reload folders
        </button>
      </div>
      <p id={`${idPrefix}-help`} className="field-help">
        {PICK_HELP}
      </p>
      <div id={`${idPrefix}-drilldown`}>
        <ConstrainedPathBrowser
          currentPath=""
          rootPath="/"
          rootLabel={vaultName}
          onBrowsePath={browse}
          onSelectPath={select}
          defaultExpanded
          showFiles={false}
          keepExpandedOnSelect
          selectLabel="Use as parent"
          filterPlaceholder="Filter folders"
          emptyMessage="No subfolders."
          cacheKey={`${toolId}:${pdmToolId}:${reloadCount}`}
          blockDisabledNavigation
          isPathDisabled={(path) =>
            children[browserPath(path)]?.rights?.read === false ? 'No access' : null
          }
          pathBadge={(path) => pickState(path).badge ?? null}
          canSelectPath={(path) => !disabled && !pickState(path).reason}
          cannotSelectPathMessage={(path) =>
            disabled ? 'Wait for the current action to finish.' : pickState(path).reason
          }
          onBrowsePathChange={(path) => {
            if (currentPathRef.current !== path) {
              currentPathRef.current = path;
              pathRevision.current += 1;
            }
            setCurrentPath(path);
            if (announcedPath.current !== null && announcedPath.current !== path) {
              announce(path === '/' ? 'Opened the vault root.' : `Opened ${vaultPath(path)}.`);
            }
            announcedPath.current = path;
          }}
        />
      </div>
      {truncated && (
        <PdmMessage id={`${idPrefix}-truncated`} tone="info">
          Showing the first {truncated.shown} of {truncated.total} folders in{' '}
          {vaultPath(currentPath)}. To choose one that isn't listed, type its path in the filter
          box.
        </PdmMessage>
      )}
      <p id={`${idPrefix}-status`} className="sw-visually-hidden" role="status" aria-live="polite">
        <span key={status.revision}>{status.text}</span>
      </p>
    </div>
  );
}
