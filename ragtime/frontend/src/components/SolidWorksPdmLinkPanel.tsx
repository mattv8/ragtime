import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  Circle,
  Info,
  RefreshCw,
  XCircle,
  ExternalLink,
  type LucideIcon,
} from 'lucide-react';
import { api } from '../api/client';
import type { PdmSandboxResponse, SolidworksHostStatus } from '../api/solidworks';
import { usePdmTools } from '../hooks/usePdmTools';
import { usePdmRights } from '../hooks/usePdmRights';
import type { RightsState } from '../utils/solidworksPdmRights';
import { SolidWorksPdmSandboxSetup, type SandboxTarget } from './SolidWorksPdmSandboxSetup';
import type {
  PdmIndexJob,
  PdmIndexStatusResponse,
  SolidworksPdmConnectionConfig,
  ToolConfig,
} from '../types/api';
import {
  findMatchingView,
  isWithinFolder,
  resolveLinkState,
  resolveMachineState,
  vaultFolderKey,
  type PdmLinkState,
  type PdmMachineState,
  type PdmViewFacts,
} from '../utils/solidworksPdmLink';
import {
  SolidWorksApprovedFolders,
  type FolderSuggestionSource,
} from './SolidWorksApprovedFolders';
import { SolidWorksPdmSetupForm } from './SolidWorksPdmSetupForm';
import { InfoPopover } from './shared/InfoPopover';
import { MESSAGE_ICON, PdmMessage, type PdmMessageTone } from './shared/PdmMessage';

export interface PdmLinkValue {
  pdm_tool_id: string | null;
  approved_folders: string[];
}

export interface PdmTarget {
  tool: ToolConfig;
  status: PdmIndexStatusResponse | null;
}
export type PdmResolution = 'loading' | 'failed' | 'missing' | 'ready';

export interface SolidWorksPdmLinkPanelProps {
  toolId: string | null;
  onRightsChange?: (state: RightsState | null) => void;
  value: PdmLinkValue;
  onChange: (next: PdmLinkValue) => void;
  allowChanges: boolean;
  onAllowChangesChange: (next: boolean) => void;
  hostStatus: SolidworksHostStatus | null;
  /** The tool's Allow scripts setting, read only here: the sandbox copy says whether scripts are on. */
  allowScripts?: boolean;
  disabled?: boolean;
  /** True while the setup form or the sandbox action runs (blocks Back/Continue like the host actions). */
  onBusyChange?: (busy: boolean) => void;
  /** For the Review step: the resolved tool and its index status. */
  onTargetChange?: (target: PdmTarget | null) => void;
  onResolutionChange?: (state: PdmResolution) => void;
}

type Tone = 'success' | 'warning' | 'info' | 'danger' | 'muted';
type PendingAction = 'index' | 'enable';
type FocusTarget = 'select' | 'trigger';
type PdmHostStatus = NonNullable<SolidworksHostStatus['pdm']>;

interface Pill {
  label: string;
  tone: Tone;
  spinner?: boolean;
}

interface Suggestion {
  tool: ToolConfig;
  view: PdmViewFacts;
}

const TONE_ICON: Record<Tone, LucideIcon> = {
  success: CheckCircle,
  warning: AlertTriangle,
  info: Info,
  danger: XCircle,
  muted: Circle,
};
const TONE_SEVERITY: Record<Tone, number> = {
  success: 0,
  muted: 1,
  info: 2,
  warning: 3,
  danger: 4,
};
const INDEX_ACTION_STATES: readonly PdmLinkState[] = [
  'never_indexed',
  'index_empty',
  'index_failed',
];

const PDM_TOOL_HELP =
  'The SolidWorks PDM tool whose search index the AI uses. Everyone who can use this tool can search that vault. The PDM tool also names the vault that vault access on this machine uses.';
const RIGHTS_CAVEAT =
  "These are folder rights. A file's workflow state can still refuse a check-out or check-in even when the folder allows it.";
const ALLOW_CHANGES_HELP =
  'Off by default. When off, the AI can search the vault, find paths, get the latest copy and open files, but cannot add files, check out or check in. When on, it can add files, check out, undo check-out and check in, only in the approved folders below. It never deletes files or changes workflow state.';

const LINK_PILLS: Record<PdmLinkState, Pill> = {
  loading: { label: 'Loading…', tone: 'muted', spinner: true },
  load_failed: { label: 'Load failed', tone: 'danger' },
  no_tools: { label: 'Not linked', tone: 'muted' },
  none_linked: { label: 'Not linked', tone: 'muted' },
  linked_missing: { label: 'Tool not found', tone: 'danger' },
  linked_disabled: { label: 'Tool disabled', tone: 'warning' },
  never_indexed: { label: 'Not indexed', tone: 'warning' },
  indexing: { label: 'Indexing…', tone: 'info', spinner: true },
  index_failed: { label: 'Index failed', tone: 'danger' },
  index_empty: { label: 'Index empty', tone: 'warning' },
  linked_ready: { label: 'Linked', tone: 'success' },
};
const MACHINE_PILLS: Record<PdmMachineState, Pill> = {
  unknown: { label: 'Not checked', tone: 'muted' },
  client_missing: { label: 'PDM client missing', tone: 'warning' },
  no_views: { label: 'No vault view', tone: 'warning' },
  vault_mismatch: { label: 'Vault mismatch', tone: 'warning' },
  logged_out: { label: 'Vault view logged out', tone: 'warning' },
  ready: { label: 'Vault view ready', tone: 'success' },
};

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));
const pdmConfigOf = (tool: ToolConfig) => tool.connection_config as SolidworksPdmConnectionConfig;
const databaseOf = (tool: ToolConfig) => pdmConfigOf(tool).database;
const quoted = (name: string) => `“${name}”`;
const formatCount = (value: number) => value.toLocaleString();
const formatDate = (iso: string) => new Date(iso).toLocaleString();

export function formatLastIndexed(status: PdmIndexStatusResponse | null): string {
  if (!status) return 'Unknown';
  return status.last_indexed ? formatDate(status.last_indexed) : 'Never';
}

/**
 * Adds a sandbox folder to the draft. A draft folder the new one sits inside is already covered, so
 * it stays; a draft folder inside the new one is dropped, since the new folder now covers it.
 */
function withApprovedFolder(folders: string[], folder: string): string[] {
  const covered = folders.some(
    (item) => vaultFolderKey(item) === vaultFolderKey(folder) || isWithinFolder(folder, item),
  );
  if (covered) return folders;
  return [...folders.filter((item) => !isWithinFolder(item, folder)), folder];
}

type VaultBrowseBlock = 'unsaved' | 'loading' | 'tool_disabled' | 'agent';

/**
 * Why the vault can't be used yet; null when it can. Listing folders works with a disabled PDM
 * tool, so only actions that need an enabled one pass `requireEnabledTool`.
 */
function vaultBrowseBlock(input: {
  toolId: string | null;
  linkedTool: ToolConfig | undefined;
  loading: boolean;
  database: string | null;
  machineState: PdmMachineState;
  requireEnabledTool: boolean;
}): VaultBrowseBlock | null {
  if (!input.toolId) return 'unsaved';
  if (!input.linkedTool || input.loading || !input.database) return 'loading';
  if (input.requireEnabledTool && !input.linkedTool.enabled) return 'tool_disabled';
  if (input.machineState !== 'ready') return 'agent';
  return null;
}

const SANDBOX_BLOCK_REASON: Record<VaultBrowseBlock, string> = {
  unsaved: 'Save the connection first to browse the vault.',
  loading: 'Wait for the search index tool to load.',
  tool_disabled: 'Enable the PDM tool first.',
  agent: 'Install and start the SolidWorks agent and log in to the vault view to browse the vault.',
};

const TYPED_PATH_STILL_WORKS = 'You can still type a path and press Enter.';
const SUGGESTIONS_BLOCK_REASON: Record<Exclude<VaultBrowseBlock, 'tool_disabled'>, string> = {
  unsaved: `Save the connection first to get folder suggestions. ${TYPED_PATH_STILL_WORKS}`,
  loading: `Wait for the search index tool to load to get folder suggestions. ${TYPED_PATH_STILL_WORKS}`,
  agent: `Install and start the SolidWorks agent and log in to the vault view to get folder suggestions. ${TYPED_PATH_STILL_WORKS}`,
};

function folderSuggestionSource(
  toolId: string | null,
  pdmToolId: string | null,
  block: VaultBrowseBlock | null,
): FolderSuggestionSource {
  if (block === null && toolId && pdmToolId) return { toolId, pdmToolId };
  // Without `requireEnabledTool` the block is never 'tool_disabled'.
  const reason = block === 'unsaved' || block === 'agent' ? block : 'loading';
  return { unavailableReason: SUGGESTIONS_BLOCK_REASON[reason] };
}

function linkPill(state: PdmLinkState, status: PdmIndexStatusResponse | null): Pill {
  if (state === 'index_failed' && (status?.document_count ?? 0) > 0) {
    return { label: 'Index outdated', tone: 'warning' };
  }
  return LINK_PILLS[state];
}

/** The header shows the worst state; a machine problem only wins when it is a warning or worse. */
function headerPill(link: Pill, machine: Pill | null): Pill {
  if (
    machine &&
    TONE_SEVERITY[machine.tone] >= TONE_SEVERITY.warning &&
    TONE_SEVERITY[machine.tone] > TONE_SEVERITY[link.tone]
  ) {
    return machine;
  }
  return link;
}

function findSuggestion(tools: ToolConfig[], views: PdmViewFacts[]): Suggestion | null {
  const matches = tools.flatMap((tool) => {
    const view = findMatchingView(views, databaseOf(tool));
    return view ? [{ tool, view }] : [];
  });
  return matches.length === 1 ? (matches[0] ?? null) : null;
}

function indexFailedText(status: PdmIndexStatusResponse | null): string {
  const failure = `The last index failed: ${status?.current_job?.error_message ?? 'no details were recorded'}.`;
  if (status && status.document_count > 0 && status.last_indexed) {
    return `${failure} The AI uses the index from ${formatDate(status.last_indexed)}.`;
  }
  return failure;
}

function linkText(
  state: PdmLinkState,
  {
    name,
    loadError,
    status,
  }: { name: string; loadError: string; status: PdmIndexStatusResponse | null },
): string {
  const named = quoted(name);
  switch (state) {
    case 'loading':
      return 'Loading your PDM tools.';
    case 'load_failed':
      return `Could not load PDM tools: ${loadError}`;
    case 'no_tools':
      return 'No PDM tools yet. Linking one is optional.';
    case 'none_linked':
      return '';
    case 'linked_missing':
      return 'The linked PDM tool no longer exists. Vault access and all changes are off.';
    case 'linked_disabled':
      return `${named} is disabled, so the AI cannot search it.`;
    case 'never_indexed':
      return `${named} has not been indexed, so the AI cannot find vault files.`;
    case 'indexing':
      return `Indexing ${named}. You can continue.`;
    case 'index_failed':
      return indexFailedText(status);
    case 'index_empty':
      return `${named} finished indexing but holds no documents.`;
    case 'linked_ready':
      return `Linked to ${named}.`;
  }
}

function machineMessage(
  state: PdmMachineState,
  {
    pdm,
    database,
    view,
  }: { pdm: PdmHostStatus | null; database: string; view: PdmViewFacts | null },
): string | null {
  const version = pdm?.client_version ? ` ${pdm.client_version}` : '';
  const views = pdm?.views ?? [];
  switch (state) {
    case 'unknown':
      return 'Install the SolidWorks agent and start it above to check the PDM client and vault view on this machine.';
    case 'client_missing':
      return 'The PDM client is not installed on this machine. Vault search works, but the AI cannot get, open or change vault files here. Install the PDM client, then choose Check again in the SolidWorks agent panel.';
    case 'no_views':
      return `The PDM client${version} is installed but has no vault view. Create a vault view for “${database}” with the PDM View Setup wizard, then choose Check again in the SolidWorks agent panel.`;
    case 'vault_mismatch':
      return `This machine has vault views for ${views.map((item) => item.vault).join(', ')}, but the linked PDM tool reads “${database}”. Choose the PDM tool for one of these vaults, or add a vault view for “${database}”. Files cannot be opened or changed here until the names match.`;
    case 'logged_out':
      return `The vault view “${view?.vault ?? database}” is not logged in. Log in to it in the PDM client on this machine, then choose Check again in the SolidWorks agent panel. The AI can open and change files only while the view is logged in.`;
    case 'ready':
      return null;
  }
}

function MachineCheck({
  name,
  tone,
  children,
}: {
  name: string;
  tone: 'success' | 'warning' | 'muted';
  children: ReactNode;
}) {
  const Icon = TONE_ICON[tone];
  const prefix = tone === 'success' ? 'Passed: ' : tone === 'warning' ? 'Needs attention: ' : '';
  return (
    <li className="sw-check" data-sw-check={name} data-tone={tone}>
      <Icon size={14} aria-hidden="true" />
      <span>
        <span className="sw-visually-hidden">{prefix}</span>
        {children}
      </span>
    </li>
  );
}

function IndexProgress({ name, job }: { name: string; job: PdmIndexJob }) {
  const known = job.total_documents > 0;
  const processed = formatCount(job.processed_documents);
  const total = formatCount(job.total_documents);
  return (
    <>
      <progress
        id="solidworks-pdm-index-progress"
        className="sw-progress"
        aria-label={`Indexing ${quoted(name)}`}
        max={known ? job.total_documents : undefined}
        value={known ? job.processed_documents : undefined}
        aria-valuetext={known ? `Indexing ${processed} of ${total} documents` : undefined}
      />
      <p id="solidworks-pdm-index-progress-count" className="field-help">
        {known ? `${processed} of ${total} documents.` : 'Waiting for indexing to start.'}
      </p>
    </>
  );
}

export function SolidWorksPdmLinkPanel({
  toolId,
  onRightsChange,
  value,
  onChange,
  allowChanges,
  onAllowChangesChange,
  hostStatus,
  allowScripts = false,
  disabled,
  onBusyChange,
  onTargetChange,
  onResolutionChange,
}: SolidWorksPdmLinkPanelProps) {
  const { tools, loading, loadError, indexStatus, statusError, refresh, refreshStatus } =
    usePdmTools(value.pdm_tool_id);
  const [formOpen, setFormOpen] = useState(false);
  const [setupBusy, setSetupBusy] = useState(false);
  const [pending, setPending] = useState<PendingAction | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState('');
  const [refreshRequested, setRefreshRequested] = useState(false);
  const [createdNote, setCreatedNote] = useState(false);
  const focusTarget = useRef<FocusTarget | null>(null);
  const indexStartFailed = useRef(false);
  const pendingIndexError = useRef<string | null>(null);
  const setupRef = useRef<HTMLDivElement>(null);
  const selectRef = useRef<HTMLSelectElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const busyCallback = useRef(onBusyChange);
  const targetCallback = useRef(onTargetChange);
  const resolutionCallback = useRef(onResolutionChange);
  const rightsCallback = useRef(onRightsChange);
  const lastRights = useRef<string | undefined>(undefined);
  const previousLink = useRef(value.pdm_tool_id);
  const requestedLink = useRef<string | null | undefined>(undefined);
  const busyParts = useRef({ setup: false, sandbox: false });
  const latestValue = useRef(value);

  useEffect(() => {
    busyCallback.current = onBusyChange;
    targetCallback.current = onTargetChange;
    resolutionCallback.current = onResolutionChange;
    rightsCallback.current = onRightsChange;
    latestValue.current = value;
  });

  const linkedId = value.pdm_tool_id;
  const linkedTool = tools.find((tool) => tool.id === linkedId);
  const linked = linkedId !== null;
  const name = linkedTool?.name ?? '';
  const database = linkedTool ? databaseOf(linkedTool) : null;
  const resolvedState = resolveLinkState({
    loading,
    loadError,
    tools,
    linkedToolId: linkedId,
    indexStatus,
  });
  const indexError = !!linkedTool && !indexStatus && !!statusError && !loading && !loadError;
  const state = indexError && resolvedState === 'loading' ? 'load_failed' : resolvedState;
  const resolution: PdmResolution = loading
    ? 'loading'
    : loadError
      ? 'failed'
      : linkedId && !linkedTool
        ? 'missing'
        : 'ready';
  const job = indexStatus?.current_job ?? null;
  const pdm = hostStatus?.pdm ?? null;
  const views = pdm?.views ?? [];
  const view = findMatchingView(views, database);
  const machineState = resolveMachineState(hostStatus, database);
  const rights = usePdmRights({
    toolId,
    pdmToolId: linkedTool ? linkedId : null,
    folders: value.approved_folders,
    allowChanges,
    enabled: linked && machineState === 'ready',
  });
  const machineText = machineMessage(machineState, { pdm, database: database ?? '', view });
  const vaultRoot = view?.root ?? null;
  const user = rights.state.user || view?.user || null;
  const folderStates = Object.values(rights.state.folders);
  const missingWrite = folderStates.filter(
    (item) => item === 'missing_write' || item === 'not_found',
  ).length;
  const folderUnchecked = folderStates.includes('not_checked');
  const vaultTone =
    machineState !== 'ready' ||
    rights.loading ||
    rights.error ||
    rights.state.root === 'not_checked'
      ? 'muted'
      : rights.state.root === 'ok'
        ? 'success'
        : 'warning';
  const loadAlert = state === 'load_failed' && refreshRequested;
  const stateText = indexError
    ? `Could not load the index status: ${statusError}`
    : !loading && state === 'loading' && linkedTool
      ? 'Loading index status.'
      : linkText(state, {
          name,
          loadError: loadError ?? '',
          status: indexStatus,
        });
  const alertText = loadAlert ? stateText : actionError;
  const pill = headerPill(
    indexError ? { label: 'Load failed', tone: 'danger' } : linkPill(state, indexStatus),
    linked ? MACHINE_PILLS[machineState] : null,
  );
  const PillIcon = TONE_ICON[pill.tone];
  const suggestion = linked ? null : findSuggestion(tools, views);
  const noFolders = allowChanges && value.approved_folders.length === 0;
  const machineShort: Record<Exclude<PdmMachineState, 'ready'>, string> = {
    unknown: 'Start the SolidWorks agent above to check this machine.',
    client_missing: 'Without the PDM client the AI cannot get, open or change vault files here.',
    no_views: 'No vault view on this machine, so the AI cannot open or change vault files.',
    vault_mismatch: `This machine has no vault view for “${database}”.`,
    logged_out:
      'The AI can search the vault but cannot open or change files here until the vault view is logged in.',
  };
  const stateHelp: Partial<Record<PdmLinkState, { id: string; label: string; text: string }>> = {
    no_tools: {
      id: 'solidworks-pdm-no-tools-help',
      label: 'About linking a PDM tool',
      text: 'No SolidWorks PDM tools exist yet. Set up a search index so the AI can find vault files. You can skip this: the tool works on SolidWorks models without a vault link.',
    },
    linked_missing: {
      id: 'solidworks-pdm-missing-help',
      label: 'About the missing PDM tool',
      text: 'Choose another PDM tool or set up a new one. Until then the AI cannot use the vault and all changes are refused.',
    },
    indexing: {
      id: 'solidworks-pdm-indexing-help',
      label: 'About indexing',
      text: 'Vault search is complete when indexing finishes.',
    },
    index_empty: {
      id: 'solidworks-pdm-index-empty-help',
      label: 'About the empty index',
      text: 'Check the database name and the file extension filter in the PDM tool.',
    },
  };
  const help = stateHelp[state];
  const hiddenState = state === 'loading' || state === 'none_linked' || state === 'linked_ready';
  const stateTone: PdmMessageTone =
    state === 'load_failed' || state === 'linked_missing' || state === 'index_failed'
      ? 'error'
      : state === 'no_tools' || state === 'indexing'
        ? 'info'
        : 'warning';
  const StateIcon = MESSAGE_ICON[stateTone];

  const serializedRights = JSON.stringify(linked ? rights.state : null);
  useEffect(() => {
    if (lastRights.current === serializedRights) return;
    lastRights.current = serializedRights;
    rightsCallback.current?.(JSON.parse(serializedRights) as RightsState | null);
  }, [serializedRights]);

  // The target is reported after every change in the linked tool or its index status.
  useEffect(() => {
    targetCallback.current?.(linkedTool ? { tool: linkedTool, status: indexStatus } : null);
  }, [linkedTool, indexStatus]);
  useEffect(() => {
    resolutionCallback.current?.(resolution);
  }, [resolution, linkedId]);
  useEffect(() => {
    if (previousLink.current === linkedId) return;
    previousLink.current = linkedId;
    if (requestedLink.current !== linkedId) {
      setActionError(null);
      setCreatedNote(false);
      setNotice('');
    }
    requestedLink.current = undefined;
  }, [linkedId]);

  // The new-tab fallback (Tools) can add a tool; coming back to the wizard reloads the list.
  useEffect(() => {
    if (state !== 'no_tools' && state !== 'load_failed') return;
    const onFocus = () => {
      if (!setupBusy) void refresh();
    };
    window.addEventListener('focus', onFocus);
    return () => window.removeEventListener('focus', onFocus);
  }, [state, refresh, setupBusy]);

  useEffect(() => {
    if (formOpen) {
      setupRef.current?.querySelector<HTMLInputElement>('#solidworks-pdm-setup-host')?.focus();
      return;
    }
    const target = focusTarget.current;
    if (target === 'select') {
      const select = selectRef.current;
      if (loading || !linkedTool || !select || select.disabled || select.value !== linkedId) return;
      focusTarget.current = null;
      select.focus();
    }
    if (target === 'trigger') {
      focusTarget.current = null;
      triggerRef.current?.focus();
    }
  }, [formOpen, loading, linkedTool, linkedId, disabled]);

  // Stable identity: the setup form and the sandbox action list these callbacks in dependencies.
  const publishBusy = useCallback((part: 'setup' | 'sandbox', busy: boolean) => {
    busyParts.current = { ...busyParts.current, [part]: busy };
    busyCallback.current?.(busyParts.current.setup || busyParts.current.sandbox);
  }, []);
  const reportSetupBusy = useCallback(
    (busy: boolean) => {
      setSetupBusy(busy);
      publishBusy('setup', busy);
    },
    [publishBusy],
  );
  const reportSandboxBusy = useCallback(
    (busy: boolean) => publishBusy('sandbox', busy),
    [publishBusy],
  );

  const selectTool = (nextId: string | null) => {
    if (nextId === linkedId) return;
    requestedLink.current = nextId;
    setActionError(null);
    setCreatedNote(false);
    if (nextId === null) {
      onChange({ pdm_tool_id: null, approved_folders: [] });
      if (allowChanges) onAllowChangesChange(false);
      setNotice(allowChanges ? 'Changes turned off because no PDM tool is linked.' : '');
      return;
    }
    const foldersCleared = linkedId !== null && value.approved_folders.length > 0;
    onChange({ pdm_tool_id: nextId, approved_folders: [] });
    setCreatedNote(false);
    setNotice(
      foldersCleared ? 'Approved folders cleared because the linked PDM tool changed.' : '',
    );
  };

  const openSetup = () => {
    setActionError(null);
    setNotice('Setting up a new PDM connection.');
    setFormOpen(true);
  };

  const closeSetup = () => {
    focusTarget.current = 'trigger';
    setFormOpen(false);
  };

  const handleIndexError = (message: string) => {
    indexStartFailed.current = true;
    pendingIndexError.current = message;
  };

  const handleCreated = (tool: ToolConfig) => {
    const indexFailed = indexStartFailed.current;
    indexStartFailed.current = false;
    selectTool(tool.id);
    if (indexFailed) setActionError(pendingIndexError.current);
    pendingIndexError.current = null;
    setNotice(
      indexFailed
        ? `PDM tool “${tool.name}” created and linked.`
        : `PDM tool “${tool.name}” created and linked. Indexing started.`,
    );
    setCreatedNote(true);
    focusTarget.current = 'select';
    setFormOpen(false);
    void refresh();
  };

  const reload = () => {
    setRefreshRequested(true);
    void refresh();
  };

  const startIndex = async () => {
    if (!linkedTool) return;
    setPending('index');
    setActionError(null);
    try {
      await api.triggerPdmIndex(linkedTool.id);
      await refreshStatus();
    } catch (error) {
      setActionError(`Could not start indexing: ${errorMessage(error)}`);
    } finally {
      setPending(null);
    }
  };

  const enableTool = async () => {
    if (!linkedTool) return;
    setPending('enable');
    setActionError(null);
    try {
      await api.toggleToolConfig(linkedTool.id, true);
      await refresh();
    } catch (error) {
      setActionError(`Could not enable ${quoted(linkedTool.name)}: ${errorMessage(error)}.`);
    } finally {
      setPending(null);
    }
  };

  const changeFolders = (approved_folders: string[]) =>
    onChange({ pdm_tool_id: linkedId, approved_folders });

  // Merges only the new folder into the draft (U3): a later Save keeps it, and unsaved edits stay.
  const handleSandboxCreated = (result: PdmSandboxResponse) => {
    const current = latestValue.current;
    onChange({
      pdm_tool_id: current.pdm_tool_id,
      approved_folders: withApprovedFolder(current.approved_folders, result.folder),
    });
    onAllowChangesChange(true);
  };

  const vaultState = { toolId, linkedTool, loading, database, machineState };
  const sandboxBlockCode = vaultBrowseBlock({ ...vaultState, requireEnabledTool: true });
  const sandboxBlock = sandboxBlockCode && SANDBOX_BLOCK_REASON[sandboxBlockCode];
  const folderSuggestions = folderSuggestionSource(
    toolId,
    linkedId,
    vaultBrowseBlock({ ...vaultState, requireEnabledTool: false }),
  );
  const sandboxTarget: SandboxTarget | null =
    toolId && linkedId && database && sandboxBlock === null
      ? { toolId, pdmToolId: linkedId, vaultName: database }
      : null;

  const toolOptions = loading ? (
    <option value="">Loading…</option>
  ) : (
    <>
      <option value="">None (no search index or vault access)</option>
      {state === 'load_failed' && linkedId !== null && !linkedTool && (
        <option value={linkedId}>Linked tool (not loaded)</option>
      )}
      {tools.map((tool) => (
        <option key={tool.id} value={tool.id}>
          {tool.enabled ? tool.name : `${tool.name} (disabled)`}
        </option>
      ))}
      {state === 'linked_missing' && (
        <option value={linkedId ?? ''} disabled>
          Missing PDM tool
        </option>
      )}
    </>
  );

  return (
    <section
      id="solidworks-pdm-link-panel"
      className="sw-panel"
      data-testid="solidworks-pdm-link-panel"
      data-sw-pdm-state={state}
      data-sw-pdm-machine-state={machineState}
      aria-labelledby="solidworks-pdm-link-title"
      aria-busy={loading || setupBusy || pending !== null}
    >
      <header className="sw-panel-header">
        <h3 id="solidworks-pdm-link-title">SolidWorks PDM</h3>
        <span className="sw-pill" data-tone={pill.tone}>
          {pill.spinner ? (
            <span className="rc-polling-indicator" aria-hidden="true" />
          ) : (
            <PillIcon size={14} aria-hidden="true" />
          )}
          {pill.label}
        </span>
        <button
          id="solidworks-pdm-refresh"
          data-sw-action="refresh"
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={disabled || setupBusy || pending !== null}
          onClick={reload}
        >
          <RefreshCw size={14} aria-hidden="true" /> Refresh
        </button>
      </header>
      <p id="solidworks-pdm-link-notice" role="status" aria-live="polite">
        {notice}
      </p>
      {alertText && (
        <PdmMessage id="solidworks-pdm-link-error" tone="error" alert>
          {alertText}
        </PdmMessage>
      )}
      {formOpen && (
        <p
          id="solidworks-pdm-link-state"
          role="status"
          aria-live="polite"
          className="sw-visually-hidden"
        >
          {loadAlert ? null : stateText}
        </p>
      )}

      {!formOpen && (
        <div id="solidworks-pdm-columns" className="sw-vault-grid">
          <div
            id="solidworks-pdm-vault"
            className="sw-box"
            data-sw-box="vault"
            aria-labelledby="solidworks-pdm-vault-title"
          >
            <h4 id="solidworks-pdm-vault-title" className="sw-box-title">
              Search index (SQL){' '}
              <InfoPopover id="solidworks-pdm-index-box-help" label="About the search index">
                Optional. A SolidWorks PDM tool reads the vault database with a read-only SQL login
                and keeps a search index. The AI uses it to find similar parts by description, part
                number and data-card values. The SQL connection, index and schedule live on the PDM
                tool card in Tools.
              </InfoPopover>
            </h4>
            {state !== 'no_tools' && (
              <div className="form-group">
                <div className="sw-field-label">
                  <label htmlFor="solidworks-pdm-tool">PDM tool</label>
                  <InfoPopover id="solidworks-pdm-tool-help" label="About PDM tool">
                    {PDM_TOOL_HELP}
                  </InfoPopover>
                </div>
                <select
                  id="solidworks-pdm-tool"
                  ref={selectRef}
                  value={linkedId ?? ''}
                  disabled={disabled || loading || setupBusy || pending !== null}
                  onChange={(event) => selectTool(event.target.value || null)}
                >
                  {toolOptions}
                </select>
              </div>
            )}
            {state === 'no_tools' && (
              <div
                id="solidworks-pdm-link-state"
                role="status"
                aria-live="polite"
                className="sw-message"
                data-tone="info"
              >
                <Info size={14} className="sw-message-icon" aria-hidden="true" />
                <span>
                  {stateText}{' '}
                  {help && (
                    <InfoPopover id={help.id} label={help.label}>
                      {help.text}
                    </InfoPopover>
                  )}
                </span>
              </div>
            )}
            <div className="sw-actions">
              {state === 'no_tools' ? (
                <button
                  id="solidworks-pdm-setup-open"
                  ref={triggerRef}
                  type="button"
                  className="btn"
                  disabled={disabled || setupBusy || pending !== null}
                  onClick={openSetup}
                >
                  Set up a search index
                </button>
              ) : (
                <button
                  id="solidworks-pdm-setup-open"
                  ref={triggerRef}
                  type="button"
                  className="btn btn-secondary btn-sm"
                  aria-label="Set up a new search index"
                  disabled={disabled || setupBusy || pending !== null}
                  onClick={openSetup}
                >
                  New…
                </button>
              )}
              {linkedTool && (
                <a
                  id="solidworks-pdm-open-tools"
                  className="sw-inline-link"
                  href="?view=tools#tools-connections"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Open in Tools (opens in a new tab)"
                >
                  Open in Tools <ExternalLink size={12} aria-hidden="true" />
                </a>
              )}
              {state === 'no_tools' && (
                <a href="?view=tools#tools-connections" target="_blank" rel="noopener noreferrer">
                  Or add one in Tools
                </a>
              )}
            </div>
            {state !== 'no_tools' && (
              <div
                id="solidworks-pdm-link-state"
                role="status"
                aria-live="polite"
                className={hiddenState ? 'sw-visually-hidden' : 'sw-message'}
                data-tone={hiddenState ? undefined : stateTone}
              >
                {!hiddenState && !loadAlert && (
                  <StateIcon className="sw-message-icon" size={14} aria-hidden="true" />
                )}
                <span>
                  {loadAlert ? null : stateText}
                  {!loadAlert && help && (
                    <>
                      {' '}
                      <InfoPopover id={help.id} label={help.label}>
                        {help.text}
                      </InfoPopover>
                    </>
                  )}
                </span>
              </div>
            )}
            {createdNote && (
              <div id="solidworks-pdm-created-note" className="sw-message" data-tone="info">
                <Info className="sw-message-icon" size={14} aria-hidden="true" />
                <span>
                  The PDM tool stays in your Tools list if you cancel this wizard.{' '}
                  <InfoPopover id="solidworks-pdm-created-help" label="About the new PDM tool">
                    To change filters, schedule or SSH tunnel, edit it in Tools.
                  </InfoPopover>
                </span>
              </div>
            )}
            {state === 'load_failed' && (
              <div className="sw-actions">
                <button
                  id="solidworks-pdm-retry"
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={disabled}
                  onClick={indexError ? () => void refreshStatus() : reload}
                >
                  Try again
                </button>
              </div>
            )}
            {linkedTool && (
              <>
                {statusError && !indexError && (
                  <PdmMessage id="solidworks-pdm-index-error" tone="warning">
                    {`Could not load the index status: ${statusError}`}
                  </PdmMessage>
                )}
                {state === 'indexing' && job && <IndexProgress name={name} job={job} />}
                {INDEX_ACTION_STATES.includes(state) && (
                  <div className="sw-actions">
                    <button
                      id="solidworks-pdm-index-start"
                      type="button"
                      className="btn"
                      disabled={disabled || setupBusy || pending !== null}
                      onClick={() => void startIndex()}
                    >
                      {pending === 'index'
                        ? 'Starting…'
                        : state === 'index_failed'
                          ? 'Try again'
                          : 'Index now'}
                    </button>
                  </div>
                )}
                {state === 'linked_disabled' && (
                  <div className="sw-actions">
                    <button
                      id="solidworks-pdm-enable-tool"
                      type="button"
                      className="btn btn-sm"
                      disabled={disabled || setupBusy || pending !== null}
                      onClick={() => void enableTool()}
                    >
                      {`Enable ${quoted(name)}`}
                    </button>
                  </div>
                )}
              </>
            )}
          </div>
          <div
            id="solidworks-pdm-machine"
            className="sw-box"
            data-sw-box="machine"
            aria-labelledby="solidworks-pdm-machine-title"
          >
            <h4 id="solidworks-pdm-machine-title" className="sw-box-title">
              Vault access (PDM client){' '}
              <InfoPopover id="solidworks-pdm-access-box-help" label="About vault access">
                The PDM client on the SolidWorks machine gets, opens and changes vault files. It
                works as the PDM user logged in to the vault view there. This is the only PDM user
                the tool uses: it needs read access, and add and check-out rights in the approved
                folders if you allow vault changes. The SQL login of the search index is never used
                for this.
              </InfoPopover>
            </h4>
            {linked ? (
              <ul id="solidworks-pdm-machine-checks" className="sw-checks">
                <MachineCheck
                  name="client"
                  tone={
                    machineState === 'unknown'
                      ? 'muted'
                      : machineState === 'client_missing'
                        ? 'warning'
                        : 'success'
                  }
                >
                  {machineState === 'unknown' ? (
                    'PDM client not checked'
                  ) : machineState === 'client_missing' ? (
                    <>
                      PDM client <strong>not installed</strong>
                    </>
                  ) : (
                    <>
                      PDM client <strong>{pdm?.client_version || 'installed'}</strong>
                    </>
                  )}
                </MachineCheck>
                <MachineCheck
                  name="view"
                  tone={
                    machineState === 'unknown' || machineState === 'client_missing'
                      ? 'muted'
                      : machineState === 'no_views' || machineState === 'vault_mismatch'
                        ? 'warning'
                        : 'success'
                  }
                >
                  {machineState === 'unknown' || machineState === 'client_missing' ? (
                    'Vault view not checked'
                  ) : machineState === 'no_views' ? (
                    <strong>No vault view</strong>
                  ) : machineState === 'vault_mismatch' ? (
                    <>
                      No view for <strong>{quoted(database ?? '')}</strong>
                    </>
                  ) : (
                    <>
                      Vault view <strong>{view?.vault}</strong> at <strong>{view?.root}</strong>
                    </>
                  )}
                </MachineCheck>
                <MachineCheck
                  name="user"
                  tone={
                    machineState === 'ready'
                      ? 'success'
                      : machineState === 'logged_out'
                        ? 'warning'
                        : 'muted'
                  }
                >
                  {machineState === 'ready' ? (
                    user ? (
                      <>
                        Logged in as <strong>{user}</strong>
                      </>
                    ) : (
                      'Logged in, user unknown'
                    )
                  ) : machineState === 'logged_out' ? (
                    <>
                      Vault view <strong>not logged in</strong>
                    </>
                  ) : (
                    'Logged-in user not checked'
                  )}
                </MachineCheck>
                <MachineCheck name="vault-rights" tone={vaultTone}>
                  {machineState !== 'ready' ? (
                    'Vault access not checked'
                  ) : rights.loading ? (
                    'Checking vault access…'
                  ) : rights.error ? (
                    'Vault access could not be checked'
                  ) : rights.state.root === 'ok' ? (
                    'Can read the vault'
                  ) : rights.state.root === 'no_read' ? (
                    <>
                      <strong>Cannot read</strong> the vault
                    </>
                  ) : (
                    'Vault access not checked'
                  )}
                </MachineCheck>
                {allowChanges && value.approved_folders.length > 0 && (
                  <MachineCheck
                    name="folder-rights"
                    tone={folderUnchecked ? 'muted' : missingWrite ? 'warning' : 'success'}
                  >
                    {folderUnchecked
                      ? 'Folder rights not checked'
                      : missingWrite
                        ? `${missingWrite} of ${value.approved_folders.length} approved folders can't be changed`
                        : value.approved_folders.length === 1
                          ? 'Can add and check out in the approved folder'
                          : `Can add and check out in all ${value.approved_folders.length} approved folders`}
                  </MachineCheck>
                )}
              </ul>
            ) : suggestion ? (
              <div id="solidworks-pdm-suggestion" className="sw-box">
                <ul className="sw-checks">
                  <li className="sw-check" data-tone="success">
                    <CheckCircle size={14} aria-hidden="true" />
                    <span>
                      PDM client{' '}
                      <strong>{pdm?.client_version ? pdm.client_version : 'installed'}</strong>
                    </span>
                  </li>
                  <li className="sw-check" data-tone="success">
                    <CheckCircle size={14} aria-hidden="true" />
                    <span>
                      Vault view <strong>{suggestion.view.vault}</strong>
                    </span>
                  </li>
                </ul>
                <button
                  id="solidworks-pdm-suggestion-link"
                  type="button"
                  className="btn btn-sm"
                  disabled={disabled || setupBusy}
                  onClick={() => selectTool(suggestion.tool.id)}
                >
                  Link {suggestion.tool.name}
                </button>
              </div>
            ) : (
              <p className="sw-muted">
                Vault access needs a linked search index, which names the vault. Link one to check
                this machine.
              </p>
            )}
            {linked && machineState === 'ready' && (
              <div className="sw-actions">
                <button
                  id="solidworks-pdm-rights-recheck"
                  type="button"
                  className="btn btn-secondary btn-sm"
                  disabled={disabled || setupBusy || pending !== null}
                  aria-disabled={rights.loading || undefined}
                  onClick={() => {
                    if (!rights.loading) rights.recheck();
                  }}
                >
                  {rights.loading ? 'Checking…' : 'Check access again'}
                </button>
              </div>
            )}
            <p
              id="solidworks-pdm-rights-status"
              role="status"
              aria-live="polite"
              className="sw-visually-hidden"
            >
              {linked && linkedTool && toolId && machineState === 'ready' && !rights.loading
                ? `${rights.error || rights.state.root === 'not_checked' ? 'Vault access could not be checked.' : rights.state.root === 'ok' ? 'Vault access checked: can read the vault.' : 'Vault access checked: the PDM user cannot read the vault.'}${allowChanges && missingWrite > 0 ? ` ${missingWrite} of ${value.approved_folders.length} approved folders can't be changed.` : ''}`
                : ''}
            </p>
          </div>
        </div>
      )}
      {!formOpen && linked && rights.error && (
        <PdmMessage id="solidworks-pdm-rights-error" tone="warning">
          Could not check vault access: {rights.error}
        </PdmMessage>
      )}
      {!formOpen && linked && rights.state.root === 'no_read' && (
        <PdmMessage id="solidworks-pdm-rights-no-read" tone="warning">
          The PDM user cannot read the vault, so the AI cannot get or open vault files. Give this
          user read access in the PDM Administration tool, then choose Check access again.
        </PdmMessage>
      )}
      {!formOpen && linked && allowChanges && missingWrite > 0 && (
        <div id="solidworks-pdm-rights-missing-write" className="sw-message" data-tone="warning">
          <AlertTriangle className="sw-message-icon" size={14} aria-hidden="true" />
          <span>
            {`The PDM user cannot add or check out files in ${missingWrite} approved ${missingWrite === 1 ? 'folder' : 'folders'}, or cannot see ${missingWrite === 1 ? 'it' : 'them'}. The AI's changes there will be refused. Grant the rights in the PDM Administration tool or remove ${missingWrite === 1 ? 'that folder' : 'those folders'}.`}{' '}
            <InfoPopover id="solidworks-pdm-rights-caveat-help" label="About PDM rights">
              {RIGHTS_CAVEAT}
            </InfoPopover>
          </span>
        </div>
      )}
      {!formOpen && linked && !allowChanges && rights.state.extraWriteInfo && (
        <PdmMessage id="solidworks-pdm-rights-extra-write" tone="info">
          This PDM user can also add or check out files. Ragtime refuses every change while vault
          changes are off. For a tool that only reads, a read-only PDM user is the safer choice.
        </PdmMessage>
      )}
      <div ref={setupRef} hidden={!formOpen}>
        <SolidWorksPdmSetupForm
          onCreated={handleCreated}
          onCancel={closeSetup}
          disabled={disabled || setupBusy || pending !== null}
          onBusyChange={reportSetupBusy}
          onIndexError={handleIndexError}
        />
      </div>
      {!formOpen && linked && machineState !== 'ready' && machineText && (
        <div
          id="solidworks-pdm-machine-message"
          className="sw-message"
          data-tone={machineState === 'unknown' ? 'info' : 'warning'}
          data-boxed={machineState !== 'unknown' ? 'true' : undefined}
        >
          {machineState === 'unknown' ? (
            <Info className="sw-message-icon" size={14} aria-hidden="true" />
          ) : (
            <AlertTriangle className="sw-message-icon" size={14} aria-hidden="true" />
          )}
          <span>
            {machineShort[machineState]}{' '}
            <InfoPopover id="solidworks-pdm-machine-help" label="About this check">
              {machineText}
            </InfoPopover>
          </span>
        </div>
      )}

      {linked && (
        <fieldset
          id="solidworks-pdm-changes"
          className="sw-pdm-controls"
          disabled={disabled || setupBusy}
        >
          <legend className="sw-visually-hidden">Vault changes</legend>
          <div id="solidworks-allow-changes-row" className="sw-switch-row">
            <label>
              <input
                id="solidworks-allow-changes"
                type="checkbox"
                role="switch"
                className="sw-switch"
                checked={allowChanges}
                aria-describedby="solidworks-allow-changes-summary"
                onChange={(event) => onAllowChangesChange(event.target.checked)}
              />
              Allow vault changes
            </label>
            <InfoPopover id="solidworks-allow-changes-help" label="About vault changes">
              {ALLOW_CHANGES_HELP}
            </InfoPopover>
            <span id="solidworks-allow-changes-summary" className="sw-switch-summary">
              {allowChanges
                ? 'Add, check out and check in, approved folders only'
                : 'Off: search, get latest and open only'}
            </span>
          </div>
          <SolidWorksPdmSandboxSetup
            target={sandboxTarget}
            unavailableReason={sandboxBlock}
            allowScripts={allowScripts}
            disabled={disabled || setupBusy || pending !== null}
            onBusyChange={reportSandboxBusy}
            onCreated={handleSandboxCreated}
          />
          {allowChanges && (
            <div id="solidworks-pdm-changes-detail" className="sw-subsection">
              <div id="solidworks-pdm-identity">
                {user ? (
                  <>
                    Changes appear in PDM as <strong>{user}</strong>. This user needs add and
                    check-out rights in the approved folders.
                  </>
                ) : (
                  'Changes appear in PDM as the vault view user.'
                )}{' '}
                <InfoPopover id="solidworks-pdm-identity-help" label="About the PDM user">
                  {user
                    ? 'The PDM user logged in to the vault view on this machine. Ragtime adds the tool name and the Ragtime user to each check-in comment. Use a dedicated PDM user with only the rights this tool needs.'
                    : 'PDM records changes under the PDM user logged in to the vault view on this machine. Install the SolidWorks agent and log in to the vault view to see which user that is.'}{' '}
                  {RIGHTS_CAVEAT}
                </InfoPopover>
              </div>
              {noFolders && (
                <PdmMessage id="solidworks-pdm-folders-warning" tone="warning">
                  Changes are on, but no folders are approved. The AI cannot change anything in the
                  vault.
                </PdmMessage>
              )}
              <SolidWorksApprovedFolders
                folders={value.approved_folders}
                onChange={changeFolders}
                vaultRoot={vaultRoot}
                disabled={disabled || setupBusy}
                rights={rights.state.folders}
                suggestions={folderSuggestions}
              />
            </div>
          )}
        </fieldset>
      )}
    </section>
  );
}
