import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  Circle,
  Info,
  RefreshCw,
  XCircle,
  type LucideIcon,
} from 'lucide-react';
import { api } from '../api/client';
import type { SolidworksHostStatus } from '../api/solidworks';
import { usePdmTools } from '../hooks/usePdmTools';
import type {
  PdmIndexJob,
  PdmIndexStatusResponse,
  SolidworksPdmConnectionConfig,
  ToolConfig,
} from '../types/api';
import {
  findMatchingView,
  formatPdmSchedule,
  resolveLinkState,
  resolveMachineState,
  type PdmLinkState,
  type PdmMachineState,
  type PdmViewFacts,
} from '../utils/solidworksPdmLink';
import { SolidWorksApprovedFolders } from './SolidWorksApprovedFolders';
import { SolidWorksPdmSetupForm } from './SolidWorksPdmSetupForm';

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
  value: PdmLinkValue;
  onChange: (next: PdmLinkValue) => void;
  allowChanges: boolean;
  onAllowChangesChange: (next: boolean) => void;
  hostStatus: SolidworksHostStatus | null;
  disabled?: boolean;
  /** True while the setup form is running (blocks Back/Continue like the host actions). */
  onBusyChange?: (busy: boolean) => void;
  /** For the Review step: the resolved tool and its index status. */
  onTargetChange?: (target: PdmTarget | null) => void;
  onResolutionChange?: (state: PdmResolution) => void;
}

type Tone = 'success' | 'warning' | 'info' | 'danger' | 'muted';
export type PdmMessageTone = 'info' | 'warning' | 'error';
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
const MESSAGE_ICON: Record<PdmMessageTone, LucideIcon> = {
  info: Info,
  warning: AlertTriangle,
  error: XCircle,
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
  "Optional. Link a SolidWorks PDM tool so the AI can search the vault, find a file's path in this machine's vault view, get the latest copy and open it. Everyone who can use this tool can search that vault through it.";
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

function indexedText(status: PdmIndexStatusResponse): string {
  return `${formatCount(status.document_count)} documents, ${formatCount(status.embedding_count)} embeddings`;
}

function indexFailedText(named: string, status: PdmIndexStatusResponse | null): string {
  const failure = `The last index of ${named} failed: ${status?.current_job?.error_message ?? 'no details were recorded'}.`;
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
      return `Ragtime could not load your PDM tools: ${loadError}`;
    case 'no_tools':
      return 'No SolidWorks PDM tools exist yet. Set up a connection to your PDM vault database so the AI can search it through this tool. You can skip this: the tool works on SolidWorks models without a vault link.';
    case 'none_linked':
      return '';
    case 'linked_missing':
      return 'The linked PDM tool no longer exists. Choose another PDM tool or set up a new one. Until then the AI cannot use the vault and all changes are refused.';
    case 'linked_disabled':
      return `${named} is disabled, so the AI cannot search it. Enable it or choose another tool.`;
    case 'never_indexed':
      return `${named} has not been indexed yet, so the AI cannot find vault files. Start the first index.`;
    case 'indexing':
      return `Indexing ${named}. You can continue. Vault search is complete when indexing finishes.`;
    case 'index_failed':
      return indexFailedText(named, status);
    case 'index_empty':
      return `${named} finished indexing but holds no documents. Check the database name and the file extension filter in the PDM tool.`;
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
      return 'Install the SolidWorks host and start it above to check the PDM client and vault view on this machine.';
    case 'client_missing':
      return 'The PDM client is not installed on this machine. Vault search works, but the AI cannot get, open or change vault files here. Install the PDM client, then choose Check again in the SolidWorks host panel.';
    case 'no_views':
      return `The PDM client${version} is installed but has no vault view. Create a vault view for “${database}” with the PDM View Setup wizard, then choose Check again in the SolidWorks host panel.`;
    case 'vault_mismatch':
      return `This machine has vault views for ${views.map((item) => item.vault).join(', ')}, but the linked PDM tool reads “${database}”. Choose the PDM tool for one of these vaults, or add a vault view for “${database}”. Files cannot be opened or changed here until the names match.`;
    case 'logged_out':
      return `The vault view “${view?.vault ?? database}” is not logged in. Log in to it in the PDM client on this machine, then choose Check again in the SolidWorks host panel. The AI can open and change files only while the view is logged in.`;
    case 'ready':
      return null;
  }
}

function clientText(pdm: PdmHostStatus): string {
  if (!pdm.client_installed) return 'Not installed';
  return pdm.client_version ? `Installed, ${pdm.client_version}` : 'Installed';
}

function changesWarning(user: string | null): ReactNode {
  if (!user) {
    return 'The AI can now add files, check out, undo check-out and check in inside the approved folders. Engineers review the results in PDM.';
  }
  return (
    <>
      The AI can now add files, check out, undo check-out and check in inside the approved folders,
      as <strong>{user}</strong>. Engineers review the results in PDM.
    </>
  );
}

function identityText(user: string | null): ReactNode {
  if (!user) {
    return 'PDM records changes under the PDM user logged in to the vault view on this machine. Install the host and log in to the vault view to see which user that is.';
  }
  return (
    <>
      Changes appear in PDM as <strong>{user}</strong>, the PDM user logged in to the vault view on
      this machine. Ragtime adds the tool name and the Ragtime user to each check-in comment. Use a
      dedicated PDM user with only the permissions this tool needs.
    </>
  );
}

export function PdmMessage({
  id,
  tone,
  boxed,
  alert,
  children,
}: {
  id?: string;
  tone: PdmMessageTone;
  boxed?: boolean;
  /** Only for failures the user just caused (ux-spec §6). */
  alert?: boolean;
  children: ReactNode;
}) {
  const Icon = MESSAGE_ICON[tone];
  return (
    <p
      id={id}
      className="sw-message"
      data-tone={tone}
      data-boxed={boxed ? 'true' : undefined}
      role={alert ? 'alert' : undefined}
    >
      <Icon className="sw-message-icon" size={14} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}

function Fact({ name, label, children }: { name: string; label: string; children: ReactNode }) {
  return (
    <div data-sw-fact={name}>
      <dt>{label}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function IndexFacts({ tool, status }: { tool: ToolConfig; status: PdmIndexStatusResponse | null }) {
  const config = pdmConfigOf(tool);
  return (
    <dl id="solidworks-pdm-index-facts" className="rc-facts-list">
      <Fact name="database" label="Vault database">
        {config.database}
      </Fact>
      <Fact name="server" label="SQL server">
        {`${config.host}:${config.port ?? 1433}`}
      </Fact>
      <Fact name="documents" label="Documents indexed">
        {status ? indexedText(status) : 'Unknown'}
      </Fact>
      <Fact name="last-indexed" label="Last indexed">
        {formatLastIndexed(status)}
      </Fact>
      <Fact name="updates" label="Updates">
        {formatPdmSchedule(config.reindex_interval_hours)}
      </Fact>
    </dl>
  );
}

function MachineFacts({ pdm, view }: { pdm: PdmHostStatus; view: PdmViewFacts | null }) {
  return (
    <dl id="solidworks-pdm-machine-facts" className="rc-facts-list">
      <Fact name="pdm-client" label="PDM client">
        {clientText(pdm)}
      </Fact>
      {view && (
        <Fact name="vault-view" label="Vault view">
          {`${view.vault} at ${view.root}`}
        </Fact>
      )}
      {view && (
        <Fact name="pdm-user" label="PDM user">
          {view.user ?? 'Unknown'}
        </Fact>
      )}
    </dl>
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
  value,
  onChange,
  allowChanges,
  onAllowChangesChange,
  hostStatus,
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
  const previousLink = useRef(value.pdm_tool_id);
  const requestedLink = useRef<string | null | undefined>(undefined);

  useEffect(() => {
    busyCallback.current = onBusyChange;
    targetCallback.current = onTargetChange;
    resolutionCallback.current = onResolutionChange;
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
  const machineText = machineMessage(machineState, { pdm, database: database ?? '', view });
  const vaultRoot = view?.root ?? null;
  const user = view?.user || null;
  const loadAlert = !!loadError && state === 'load_failed' && refreshRequested;
  const stateText = indexError
    ? `Ragtime could not load the index status: ${statusError}`
    : !loading && state === 'loading' && linkedTool
      ? 'Loading index status…'
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
  const allowDescribedBy = allowChanges
    ? 'solidworks-allow-changes-help solidworks-allow-changes-warning'
    : 'solidworks-allow-changes-help';

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

  // Stable identity: the setup form lists this callback in an effect dependency.
  const reportSetupBusy = useCallback((busy: boolean) => {
    setSetupBusy(busy);
    busyCallback.current?.(busy);
  }, []);

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

  const toolOptions = loading ? (
    <option value="">Loading…</option>
  ) : (
    <>
      <option value="">None (no vault access)</option>
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
      <p id="solidworks-pdm-link-state" role="status" aria-live="polite">
        {loadAlert ? null : stateText}
      </p>
      {alertText && (
        <PdmMessage id="solidworks-pdm-link-error" tone="error" alert>
          {alertText}
        </PdmMessage>
      )}
      <p id="solidworks-pdm-link-notice" role="status" aria-live="polite">
        {notice}
      </p>

      {!formOpen && (
        <>
          {state !== 'no_tools' && (
            <div className="form-group">
              <label htmlFor="solidworks-pdm-tool">PDM tool</label>
              <select
                id="solidworks-pdm-tool"
                ref={selectRef}
                value={linkedId ?? ''}
                disabled={disabled || loading || setupBusy || pending !== null}
                aria-describedby="solidworks-pdm-tool-help"
                onChange={(event) => selectTool(event.target.value || null)}
              >
                {toolOptions}
              </select>
              <p id="solidworks-pdm-tool-help" className="field-help">
                {PDM_TOOL_HELP}
              </p>
            </div>
          )}
          {createdNote && (
            <PdmMessage id="solidworks-pdm-created-note" tone="info">
              The PDM tool stays in your Tools list even if you cancel this wizard. To change
              filters, schedule or SSH tunnel, edit it in{' '}
              <a href="?view=tools#tools-connections" target="_blank" rel="noopener noreferrer">
                Tools
              </a>
              .
            </PdmMessage>
          )}
          {suggestion && (
            <PdmMessage id="solidworks-pdm-suggestion" tone="info">
              {`${suggestion.tool.name} matches the vault view ${suggestion.view.vault} on this machine.`}{' '}
              <button
                id="solidworks-pdm-suggestion-link"
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={disabled || setupBusy}
                onClick={() => selectTool(suggestion.tool.id)}
              >
                {`Link ${suggestion.tool.name}`}
              </button>
            </PdmMessage>
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
                Set up a PDM connection
              </button>
            ) : (
              <button
                id="solidworks-pdm-setup-open"
                ref={triggerRef}
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={disabled || setupBusy || pending !== null}
                onClick={openSetup}
              >
                Set up new…
              </button>
            )}
            {state === 'load_failed' && (
              <button
                id="solidworks-pdm-retry"
                type="button"
                className="btn btn-secondary btn-sm"
                disabled={disabled}
                onClick={indexError ? () => void refreshStatus() : reload}
              >
                Try again
              </button>
            )}
          </div>
          {state === 'no_tools' && (
            <p className="field-help">
              <a href="?view=tools#tools-connections" target="_blank" rel="noopener noreferrer">
                Or add one in Tools
              </a>
            </p>
          )}
        </>
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

      {linkedTool && (
        <>
          <h4 className="sw-group-label">Vault index</h4>
          <IndexFacts tool={linkedTool} status={indexStatus} />
          {statusError && !indexError && (
            <PdmMessage id="solidworks-pdm-index-error" tone="warning">
              {`Ragtime could not load the index status: ${statusError}`}
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
          <h4 className="sw-group-label">On this machine</h4>
          {pdm && machineState !== 'unknown' && <MachineFacts pdm={pdm} view={view} />}
          {machineText && (
            <PdmMessage
              id="solidworks-pdm-machine-message"
              tone={machineState === 'unknown' ? 'info' : 'warning'}
            >
              {machineText}
            </PdmMessage>
          )}
        </>
      )}

      {linked && (
        <fieldset
          id="solidworks-pdm-changes"
          className="sw-pdm-controls"
          disabled={disabled || setupBusy}
        >
          <legend>Changes</legend>
          <label className="checkbox-label">
            <input
              id="solidworks-allow-changes"
              type="checkbox"
              checked={allowChanges}
              aria-describedby={allowDescribedBy}
              onChange={(event) => onAllowChangesChange(event.target.checked)}
            />
            Allow changes
          </label>
          <p id="solidworks-allow-changes-help" className="field-help">
            {ALLOW_CHANGES_HELP}
          </p>
          <p id="solidworks-pdm-identity">{identityText(user)}</p>
          {allowChanges && (
            <PdmMessage id="solidworks-allow-changes-warning" tone="warning" boxed>
              {changesWarning(user)}
            </PdmMessage>
          )}
          {noFolders && (
            <PdmMessage id="solidworks-pdm-folders-warning" tone="warning">
              Changes are on, but no folders are approved, so the AI cannot change anything in the
              vault. Add a folder below.
            </PdmMessage>
          )}
          <SolidWorksApprovedFolders
            folders={value.approved_folders}
            onChange={changeFolders}
            vaultRoot={vaultRoot}
            disabled={disabled || setupBusy}
          />
          {/* Reserved: solidworks-allow-scripts, a later switch below the folders (ux-spec §8, Decision 9). Nothing renders yet. */}
        </fieldset>
      )}
    </section>
  );
}
