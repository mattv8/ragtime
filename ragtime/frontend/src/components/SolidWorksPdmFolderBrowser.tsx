import { useRef, useState } from 'react';
import { solidworksApi, type PdmFolderChild } from '../api/solidworks';
import type { BrowseResponse } from '../types/api';
import { normalizeMountBrowserPath } from '../utils/mountPaths';
import {
  addApprovedFolder,
  isWithinFolder,
  normalizeVaultFolder,
  vaultFolderKey,
} from '../utils/solidworksPdmLink';
import { ConstrainedPathBrowser } from './ConstrainedPathBrowser';
import { approvedCountLabel } from './SolidWorksApprovedFolders';
import { PdmMessage } from './shared/PdmMessage';

/** `approve` adds folders to the approved list; `pick` chooses one folder and changes nothing else. */
export type SolidWorksPdmFolderBrowserMode = 'approve' | 'pick';

export interface SolidWorksPdmFolderBrowserProps {
  toolId: string;
  pdmToolId: string;
  vaultName: string;
  folders: string[];
  vaultRoot: string | null;
  /** Approve mode only: receives the approved list after a folder is added. */
  onChange?: (next: string[]) => void;
  /** Pick mode only: receives the chosen vault path. */
  onPick?: (vaultPath: string) => void;
  mode?: SolidWorksPdmFolderBrowserMode;
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

const MAX_PAGES = 10;
const DEFAULT_ID_PREFIX = 'solidworks-pdm-folder-browser';
const READ_ONLY_REASON =
  "The PDM user can read this folder but can't add and check out files here.";
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
  folders,
  vaultRoot,
  onChange,
  onPick,
  mode = 'approve',
  idPrefix = DEFAULT_ID_PREFIX,
  disabled,
}: SolidWorksPdmFolderBrowserProps) {
  const pick = mode === 'pick';
  const [reloadCount, setReloadCount] = useState(0);
  const [children, setChildren] = useState<Record<string, PdmFolderChild>>({});
  const [truncations, setTruncations] = useState<Record<string, { shown: number; total: number }>>(
    {},
  );
  const [currentPath, setCurrentPath] = useState('/');
  const [feedback, setFeedback] = useState<{ text: string; info: boolean } | null>(null);
  const [status, setStatus] = useState({ text: '', revision: 0 });
  const wrapper = useRef<HTMLDivElement>(null);
  const generation = useRef(0);
  const pathRevision = useRef(0);
  const currentPathRef = useRef('/');
  const announcedPath = useRef<string | null>(null);

  function announce(text: string) {
    setStatus((previous) => ({ text, revision: previous.revision + 1 }));
  }

  async function listAllPages(path: string) {
    let items: PdmFolderChild[] = [];
    let offset = 0;
    let snapshot: string | null = null;
    let restarted = false;
    let pages = 0;
    let total = 0;
    while (pages < MAX_PAGES) {
      const result = await solidworksApi.listPdmFolders(toolId, {
        pdm_tool_id: pdmToolId,
        path: vaultPath(path),
        offset,
        snapshot,
      });
      if (result.error?.code === 'stale_snapshot' && !restarted) {
        restarted = true;
        items = [];
        offset = 0;
        snapshot = null;
        pages = 0;
        continue;
      }
      if (result.error) throw new Error(result.error.message);
      items.push(...result.items);
      total = result.total;
      snapshot = result.snapshot;
      pages += 1;
      if (result.next_offset === null) break;
      offset = result.next_offset;
    }
    return { items, total };
  }

  async function browse(
    path: string,
    isCurrent: () => boolean = () => true,
  ): Promise<BrowseResponse> {
    const requestGeneration = generation.current;
    try {
      const { items, total } = await listAllPages(path);
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

  function approveState(path: string): FolderState {
    const folder = vaultPath(path);
    if (path === '/')
      return { reason: "Open a folder to approve it. The whole vault can't be approved." };
    if (folders.some((item) => vaultFolderKey(item) === vaultFolderKey(folder)))
      return {
        reason: 'This folder is already approved.',
        badge: { label: 'Approved', tone: 'info', title: 'This folder is approved.' },
      };
    const parent = folders.find((item) => isWithinFolder(folder, item));
    if (parent)
      return {
        reason: `Already included in ${parent}.`,
        badge: {
          label: 'Included',
          tone: 'info',
          title: `Inside the approved folder ${parent}.`,
        },
      };
    const child = children[browserPath(path)];
    if (!child?.rights) return { reason: NOT_CHECKED_REASON, badge: NOT_CHECKED_BADGE };
    if (!child.rights.read) return { reason: "The PDM user can't see inside this folder." };
    const normalized = normalizeVaultFolder(folder, vaultRoot);
    if (!normalized.ok)
      return {
        reason: normalized.error,
        badge: { label: "Can't approve", tone: 'muted', title: normalized.error },
      };
    if (!child.write_ready)
      return {
        reason: READ_ONLY_REASON,
        badge: {
          label: 'Read only',
          tone: 'muted',
          title: READ_ONLY_REASON,
        },
      };
    return {
      badge: {
        label: 'Can change',
        tone: 'success',
        title: 'Can add and check out files',
      },
    };
  }

  function state(path: string): FolderState {
    return pick ? pickState(path) : approveState(path);
  }

  function select(path: string) {
    if (pick) {
      onPick?.(vaultPath(path));
      announce(`Using ${vaultPath(path)} as the parent.`);
      return;
    }
    const result = addApprovedFolder(folders, vaultPath(path), vaultRoot);
    if (!result.ok) {
      setFeedback({ text: result.error, info: result.info });
      announce(result.error);
      return;
    }
    onChange?.(result.folders);
    setFeedback(result.notice ? { text: result.notice, info: true } : null);
    announce(`Added ${result.added}. ${approvedCountLabel(result.folders.length)}`);
    wrapper.current?.querySelector<HTMLInputElement>('.browser-filter-input')?.focus();
  }

  const truncated = truncations[currentPath];
  return (
    <div
      id={idPrefix}
      data-testid={idPrefix}
      role="group"
      aria-labelledby={`${idPrefix}-label`}
      ref={wrapper}
    >
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
        {pick ? (
          PICK_HELP
        ) : (
          <>
            Open a folder, then choose Approve this folder. It is added above as a vault path, like{' '}
            <code>\13_Ragtime</code>.
          </>
        )}
      </p>
      <div id={`${idPrefix}-drilldown`}>
        <ConstrainedPathBrowser
          currentPath=""
          rootPath="/"
          rootLabel={vaultName}
          onBrowsePath={browse}
          onSelectPath={select}
          defaultExpanded={pick}
          showFiles={false}
          keepExpandedOnSelect
          selectLabel={pick ? 'Use as parent' : 'Approve this folder'}
          filterPlaceholder="Filter folders"
          emptyMessage="No subfolders."
          cacheKey={`${toolId}:${pdmToolId}:${reloadCount}`}
          blockDisabledNavigation
          isPathDisabled={(path) =>
            children[browserPath(path)]?.rights?.read === false ? 'No access' : null
          }
          pathBadge={(path) => state(path).badge ?? null}
          canSelectPath={(path) => !disabled && !state(path).reason}
          cannotSelectPathMessage={(path) =>
            disabled ? 'Wait for the current action to finish.' : state(path).reason
          }
          onBrowsePathChange={(path) => {
            if (currentPathRef.current !== path) {
              currentPathRef.current = path;
              pathRevision.current += 1;
            }
            setCurrentPath(path);
            setFeedback(null);
            if (announcedPath.current !== null && announcedPath.current !== path) {
              announce(path === '/' ? 'Opened the vault root.' : `Opened ${vaultPath(path)}.`);
            }
            announcedPath.current = path;
          }}
        />
      </div>
      {feedback && (
        <PdmMessage id={`${idPrefix}-feedback`} tone={feedback.info ? 'info' : 'error'}>
          {feedback.text}
        </PdmMessage>
      )}
      {truncated && (
        <PdmMessage id={`${idPrefix}-truncated`} tone="info">
          Showing the first {truncated.shown} of {truncated.total} folders in{' '}
          {vaultPath(currentPath)}.{' '}
          {pick
            ? "To choose one that isn't listed, type its path in the filter box."
            : "To approve one that isn't listed, type its path in the box above."}
        </PdmMessage>
      )}
      <p id={`${idPrefix}-status`} className="sw-visually-hidden" role="status" aria-live="polite">
        <span key={status.revision}>{status.text}</span>
      </p>
    </div>
  );
}
