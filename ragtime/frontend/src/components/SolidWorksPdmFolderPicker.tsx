import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
} from 'react';
import { ChevronDown, ChevronRight, X } from 'lucide-react';
import { solidworksApi, type PdmFolderChild } from '../api/solidworks';
import {
  addApprovedFolder,
  isWithinFolder,
  normalizeVaultFolder,
  vaultFolderKey,
} from '../utils/solidworksPdmLink';
import { PdmMessage } from './shared/PdmMessage';
import { approvedCountLabel } from './SolidWorksApprovedFolders';

export interface SolidWorksPdmFolderPickerProps {
  toolId: string;
  pdmToolId: string;
  vaultName: string;
  folders: string[];
  vaultRoot: string | null;
  onChange: (next: string[]) => void;
  onClose: () => void;
}

interface Branch {
  expanded: boolean;
  loaded: boolean;
  loading: boolean;
  items: PdmFolderChild[];
  next: number | null;
  snapshot: string | null;
  error: string | null;
}

const emptyBranch = (): Branch => ({
  expanded: false,
  loaded: false,
  loading: false,
  items: [],
  next: null,
  snapshot: null,
  error: null,
});
const ROOT = '\\';
const moreKey = (path: string) => `more:${vaultFolderKey(path)}`;

function folderBadge(child: PdmFolderChild, folders: string[]) {
  const key = vaultFolderKey(child.vault_path);
  const approved = folders.map(vaultFolderKey);
  const normalized = normalizeVaultFolder(child.vault_path, null);
  const parent = folders.find((folder) => isWithinFolder(child.vault_path, folder));
  if (approved.includes(key))
    return { code: 'approved', label: 'Approved', tone: 'info', reason: 'Already approved.' };
  if (parent)
    return { code: 'included', label: 'Included', tone: 'info', reason: `Included in ${parent}.` };
  if (child.rights && !child.rights.read)
    return {
      code: 'no_access',
      label: 'No access',
      tone: 'muted',
      reason: "The PDM user can't see this folder's contents.",
    };
  if (!child.rights)
    return {
      code: 'not_checked',
      label: 'Not checked',
      tone: 'muted',
      reason: 'Rights not checked. Collapse and expand the parent folder to check again.',
    };
  if (!normalized.ok)
    return {
      code: 'cant_approve',
      label: "Can't approve",
      tone: 'muted',
      title: normalized.error,
      reason: normalized.error,
    };
  if (!child.write_ready)
    return {
      code: 'read_only',
      label: 'Read only',
      tone: 'muted',
      reason: "The PDM user can't add and check out files here.",
    };
  return {
    code: 'write_ready',
    label: 'Can change',
    tone: 'success',
    title: 'Can add and check out files',
    reason: '',
  };
}

function useFolderTree(toolId: string, pdmToolId: string) {
  const [branches, setBranches] = useState<Record<string, Branch>>({
    [ROOT]: { ...emptyBranch(), expanded: true },
  });
  const [focus, setFocus] = useState(ROOT);
  const [status, setStatusState] = useState({ text: '', revision: 0 });
  const setStatus = useCallback(
    (text: string) => setStatusState((old) => ({ text, revision: old.revision + 1 })),
    [],
  );
  const controllers = useRef(new Set<AbortController>());
  const inFlight = useRef(new Map<string, AbortController>());
  const nodes = useRef(new Map<string, HTMLElement>());
  const pendingFocus = useRef<string | null>(ROOT);
  const update = useCallback((path: string, change: (branch: Branch) => Branch) => {
    setBranches((current) => ({ ...current, [path]: change(current[path] ?? emptyBranch()) }));
  }, []);
  const moveFocus = useCallback((key: string) => {
    pendingFocus.current = key;
    setFocus(key);
    nodes.current.get(key)?.focus();
  }, []);
  useLayoutEffect(() => {
    if (pendingFocus.current === focus) {
      nodes.current.get(focus)?.focus();
      pendingFocus.current = null;
    }
  });
  const load = useCallback(
    async (path: string, offset: number, snapshot: string | null, initial = false) => {
      if (inFlight.current.has(path)) return;
      const controller = new AbortController();
      inFlight.current.set(path, controller);
      controllers.current.add(controller);
      update(path, (branch) => ({ ...branch, loading: true, error: null, expanded: true }));
      try {
        let result = await solidworksApi.listPdmFolders(
          toolId,
          { pdm_tool_id: pdmToolId, path, offset, snapshot },
          controller.signal,
        );
        let stale = false;
        if (controller.signal.aborted) return;
        if (result.error?.code === 'stale_snapshot') {
          stale = true;
          moveFocus(vaultFolderKey(path));
          update(path, (branch) => ({ ...branch, items: [], next: null, snapshot: null }));
          result = await solidworksApi.listPdmFolders(
            toolId,
            { pdm_tool_id: pdmToolId, path, offset: 0, snapshot: null },
            controller.signal,
          );
        }
        if (controller.signal.aborted) return;
        if (result.error) throw new Error(result.error.message);
        update(path, (branch) => ({
          ...branch,
          loading: false,
          loaded: true,
          error: null,
          items: offset === 0 || stale ? result.items : [...branch.items, ...result.items],
          next: result.next_offset,
          snapshot: result.snapshot,
        }));
        if (stale) setStatus('The folder changed. Reloaded it.');
        const initialFocusAllowed =
          document.activeElement === nodes.current.get(ROOT) ||
          document.activeElement?.id === 'solidworks-pdm-folder-picker';
        if (((initial && initialFocusAllowed) || offset > 0) && result.items.length)
          moveFocus(vaultFolderKey(result.items[0].vault_path));
        else if (offset > 0) moveFocus(vaultFolderKey(path));
      } catch (error) {
        if (!controller.signal.aborted)
          update(path, (branch) => ({
            ...branch,
            loading: false,
            error: error instanceof Error ? error.message : String(error),
          }));
      } finally {
        controllers.current.delete(controller);
        if (inFlight.current.get(path) === controller) inFlight.current.delete(path);
      }
    },
    [toolId, pdmToolId, update, moveFocus, setStatus],
  );
  useEffect(() => {
    void load(ROOT, 0, null, true);
    const active = controllers.current;
    const requests = inFlight.current;
    return () => {
      active.forEach((controller) => controller.abort());
      requests.clear();
    };
  }, [load]);
  return { branches, update, load, focus, moveFocus, nodes, status, setStatus };
}

type TreeState = ReturnType<typeof useFolderTree>;

function FolderTree({
  tree,
  folders,
  vaultName,
  onAdd,
  onDisabled,
}: {
  tree: TreeState;
  folders: string[];
  vaultName: string;
  onAdd: (path: string) => void;
  onDisabled: (reason: string) => void;
}) {
  const visible: { key: string; path: string; parent: string | null; more: boolean }[] = [];
  const collect = (path: string, parent: string | null) => {
    const key = vaultFolderKey(path);
    visible.push({ key, path, parent, more: false });
    const branch = tree.branches[path];
    if (!branch?.expanded) return;
    branch.items.forEach((item) => collect(item.vault_path, path));
    if (branch.next !== null) visible.push({ key: moreKey(path), path, parent: path, more: true });
  };
  collect(ROOT, null);
  useLayoutEffect(() => {
    if (visible.some((node) => node.key === tree.focus)) return;
    let path = tree.focus.startsWith('more:') ? tree.focus.slice(5) : tree.focus;
    while (path !== ROOT) {
      const separator = path.lastIndexOf('\\');
      path = separator <= 0 ? ROOT : path.slice(0, separator);
      const parent = visible.find((node) => !node.more && node.key === path);
      if (parent) {
        tree.moveFocus(parent.key);
        return;
      }
    }
    tree.moveFocus(ROOT);
  });
  const toggle = (path: string) => {
    const branch = tree.branches[path] ?? emptyBranch();
    if (branch.loading) return;
    if (branch.expanded) tree.update(path, (current) => ({ ...current, expanded: false }));
    else if (!branch.loaded || branch.items.some((item) => item.rights === null))
      void tree.load(path, 0, null);
    else tree.update(path, (current) => ({ ...current, expanded: true }));
  };
  const keyboard = (event: KeyboardEvent<HTMLUListElement>) => {
    const index = visible.findIndex((node) => node.key === tree.focus);
    const node = visible[index];
    if (!node) return;
    const branch = tree.branches[node.path] ?? emptyBranch();
    const target = event.target as HTMLElement;
    if (!target.matches('[role="treeitem"]')) return;
    if (event.key === 'ArrowDown')
      tree.moveFocus(visible[Math.min(index + 1, visible.length - 1)].key);
    else if (event.key === 'ArrowUp') tree.moveFocus(visible[Math.max(index - 1, 0)].key);
    else if (event.key === 'Home') tree.moveFocus(visible[0].key);
    else if (event.key === 'End') tree.moveFocus(visible[visible.length - 1].key);
    else if (event.key === 'ArrowLeft') {
      if (!node.more && target.getAttribute('aria-expanded') === 'true') toggle(node.path);
      else if (node.parent) tree.moveFocus(vaultFolderKey(node.parent));
    } else if (event.key === 'ArrowRight' && !node.more) {
      if (target.getAttribute('aria-expanded') === 'false') toggle(node.path);
      else if (branch.expanded && visible[index + 1]?.parent === node.path)
        tree.moveFocus(visible[index + 1].key);
    } else if (event.key === 'Enter' || event.key === ' ') {
      if (node.more) {
        if (!branch.loading && branch.next !== null)
          void tree.load(node.path, branch.next, branch.snapshot);
      } else if (target.getAttribute('aria-disabled') !== 'true') onAdd(node.path);
      else {
        const item = Object.values(tree.branches)
          .flatMap((entry) => entry.items)
          .find((child) => vaultFolderKey(child.vault_path) === node.key);
        onDisabled(item ? folderBadge(item, folders).reason : 'The vault root cannot be approved.');
      }
    } else return;
    event.preventDefault();
    event.stopPropagation();
  };
  const renderNode = (
    path: string,
    name: string,
    level: number,
    child?: PdmFolderChild,
  ): JSX.Element => {
    const key = vaultFolderKey(path);
    const branch = tree.branches[path] ?? emptyBranch();
    const badge = child ? folderBadge(child, folders) : null;
    const expandable =
      (!child?.rights || child.rights.read) &&
      !(branch.loaded && branch.items.length === 0 && branch.next === null);
    const disabled = !badge || badge.label !== 'Can change';
    return (
      <li
        key={key}
        role="treeitem"
        aria-level={level}
        aria-expanded={expandable ? branch.expanded : undefined}
        aria-disabled={disabled || undefined}
        aria-label={badge ? `${name}, ${badge.label}` : `${name}, vault root`}
        aria-busy={branch.loading || undefined}
        data-sw-folder-node={key}
        tabIndex={tree.focus === key ? 0 : -1}
        ref={(element) => {
          if (element) tree.nodes.current.set(key, element);
          else tree.nodes.current.delete(key);
        }}
        onFocus={(event) => {
          if (event.target === event.currentTarget) tree.moveFocus(key);
        }}
      >
        <div className="sw-tree-row">
          <span
            className="sw-tree-chevron"
            aria-hidden="true"
            onClick={() => {
              tree.moveFocus(key);
              if (expandable) toggle(path);
            }}
          >
            {expandable &&
              (branch.expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />)}
          </span>
          <span
            className="sw-tree-label"
            onClick={() => {
              tree.moveFocus(key);
              if (disabled) onDisabled(badge?.reason ?? 'The vault root cannot be approved.');
              if (expandable) toggle(path);
            }}
          >
            {name}
          </span>
          {badge && (
            <span
              className="sw-tree-rights sw-pill"
              data-sw-rights={badge.code}
              data-tone={badge.tone}
              title={badge.title}
            >
              {badge.label}
            </span>
          )}
          {!badge && (
            <span className="sw-tree-rights sw-pill" data-tone="muted">
              Vault
            </span>
          )}
          {!disabled && (
            <button
              type="button"
              className="btn btn-secondary btn-sm sw-tree-add"
              data-sw-action="add-folder"
              aria-label={`Add ${path}`}
              onClick={() => {
                // The button disappears once the folder is approved; keep focus on its treeitem.
                tree.moveFocus(key);
                onAdd(path);
              }}
            >
              Add
            </button>
          )}
        </div>
        {branch.expanded && (
          <ul role="group" data-sw-folder-children={key}>
            {branch.items.map((item) => renderNode(item.vault_path, item.name, level + 1, item))}
            {branch.next !== null && (
              <li
                role="treeitem"
                aria-level={level + 1}
                aria-disabled={branch.loading || undefined}
                data-sw-load-more
                tabIndex={tree.focus === moreKey(path) ? 0 : -1}
                ref={(element) => {
                  if (element) tree.nodes.current.set(moreKey(path), element);
                  else tree.nodes.current.delete(moreKey(path));
                }}
                onFocus={(event) => {
                  if (event.target === event.currentTarget) tree.moveFocus(moreKey(path));
                }}
                onClick={(event) => {
                  event.stopPropagation();
                  tree.moveFocus(moreKey(path));
                  if (!branch.loading && branch.next !== null)
                    void tree.load(path, branch.next, branch.snapshot);
                }}
              >
                {branch.loading ? 'Loading…' : 'Load more folders'}
              </li>
            )}
            {branch.loading && (
              <li role="none" className="sw-muted">
                Loading…
              </li>
            )}
            {branch.error && (
              <li role="none">
                <PdmMessage tone="error">Could not load folders: {branch.error}</PdmMessage>
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  data-sw-action="retry-folders"
                  disabled={branch.loading}
                  onClick={() => void tree.load(path, branch.next ?? 0, branch.snapshot)}
                >
                  Try again
                </button>
              </li>
            )}
            {path === ROOT &&
              branch.loaded &&
              !branch.items.length &&
              !branch.loading &&
              !branch.error && (
                <li role="none" className="sw-muted">
                  This vault has no folders.
                </li>
              )}
          </ul>
        )}
      </li>
    );
  };
  return (
    <ul
      id="solidworks-pdm-folder-tree"
      className="sw-tree"
      role="tree"
      aria-labelledby="solidworks-pdm-folder-picker-title"
      onKeyDown={keyboard}
    >
      {renderNode(ROOT, vaultName, 1)}
    </ul>
  );
}

export function SolidWorksPdmFolderPicker({
  toolId,
  pdmToolId,
  vaultName,
  folders,
  vaultRoot,
  onChange,
  onClose,
}: SolidWorksPdmFolderPickerProps): JSX.Element {
  const opener = useRef<HTMLElement | null>(null);
  useLayoutEffect(() => {
    const active = document.activeElement;
    opener.current =
      active instanceof HTMLElement &&
      active !== document.body &&
      active !== document.documentElement
        ? active
        : null;
    return () => {
      const target = opener.current?.isConnected
        ? opener.current
        : document.getElementById('solidworks-pdm-folder-browse');
      target?.focus();
    };
  }, []);
  const tree = useFolderTree(toolId, pdmToolId);
  const [feedback, setFeedback] = useState<{ text: string; info: boolean } | null>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const disabledFeedback = (text: string) => {
    setFeedback({ text, info: true });
    tree.setStatus(text);
  };
  const add = (path: string) => {
    const result = addApprovedFolder(folders, path, vaultRoot);
    if (!result.ok) {
      setFeedback({ text: result.error, info: result.info });
      tree.setStatus(result.error);
      return;
    }
    onChange(result.folders);
    setFeedback(result.notice ? { text: result.notice, info: true } : null);
    tree.setStatus(`Added ${result.added}. ${approvedCountLabel(result.folders.length)}`);
  };
  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = Array.from(
      dialog.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) ?? [],
    ).filter((element) => getComputedStyle(element).visibility !== 'hidden');
    if (!focusable.length) return;
    const first = focusable[0],
      last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };
  return (
    <div className="modal-overlay">
      <div
        id="solidworks-pdm-folder-picker"
        className="modal-content"
        role="dialog"
        aria-modal="true"
        aria-labelledby="solidworks-pdm-folder-picker-title"
        aria-describedby="solidworks-pdm-folder-picker-description"
        ref={dialog}
        onKeyDown={handleKeyDown}
      >
        <div className="modal-header">
          <h3 id="solidworks-pdm-folder-picker-title">Browse vault folders</h3>
          <button type="button" className="modal-close" aria-label="Close" onClick={onClose}>
            <X size={18} />
          </button>
        </div>
        <div className="modal-body">
          <p id="solidworks-pdm-folder-picker-description">
            Choose folders the AI may change. Folders where the PDM user cannot add and check out
            files can't be chosen.
          </p>
          {feedback && (
            <PdmMessage
              id="solidworks-pdm-folder-picker-error"
              tone={feedback.info ? 'info' : 'error'}
            >
              {feedback.text}
            </PdmMessage>
          )}
          <FolderTree
            tree={tree}
            folders={folders}
            vaultName={vaultName}
            onAdd={add}
            onDisabled={disabledFeedback}
          />
          <p
            id="solidworks-pdm-folder-picker-status"
            role="status"
            aria-live="polite"
            className="sw-visually-hidden"
          >
            <span key={tree.status.revision}>{tree.status.text}</span>
          </p>
        </div>
        <div className="modal-footer sw-tree-footer">
          <span>{approvedCountLabel(folders.length)}</span>
          <button
            type="button"
            id="solidworks-pdm-folder-picker-done"
            className="btn"
            onClick={onClose}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
