import type { ReactNode } from 'react';
import type { SolidworksHostStatus } from '../api/solidworks';
import type { SolidworksPdmConnectionConfig, ToolConfig } from '../types/api';
import {
  findMatchingView,
  resolveLinkState,
  resolveMachineState,
  type PdmLinkState,
  type PdmMachineState,
  type PdmViewFacts,
} from '../utils/solidworksPdmLink';
import {
  formatLastIndexed,
  PdmMessage,
  type PdmLinkValue,
  type PdmResolution,
  type PdmTarget,
} from './SolidWorksPdmLinkPanel';

export interface SolidWorksPdmReviewProps {
  link: PdmLinkValue;
  allowChanges: boolean;
  target: PdmTarget | null;
  hostStatus: SolidworksHostStatus | null;
  resolution?: PdmResolution;
}

const warning = (text: string) => <PdmMessage tone="warning">{text}</PdmMessage>;
const databaseOf = (tool: ToolConfig) =>
  (tool.connection_config as SolidworksPdmConnectionConfig).database;

function indexLine(state: PdmLinkState, status: PdmTarget['status']): ReactNode {
  switch (state) {
    case 'linked_ready':
      return `Index: ${status?.document_count.toLocaleString()} documents, last indexed ${formatLastIndexed(status)}`;
    case 'linked_disabled':
      return warning('Index: the PDM tool is disabled, so the AI cannot search the vault.');
    case 'never_indexed':
      return warning('Index: not indexed yet.');
    case 'indexing':
      return warning('Index: indexing in progress.');
    case 'index_failed':
      return warning('Index: the last index failed.');
    case 'index_empty':
      return warning('Index: empty, the last index found no documents.');
    default:
      return 'Index: not checked yet';
  }
}

function machineLine(
  state: PdmMachineState,
  database: string,
  view: PdmViewFacts | null,
): ReactNode {
  switch (state) {
    case 'unknown':
      return 'Vault view on the machine: not checked';
    case 'client_missing':
      return warning('Vault view on the machine: the PDM client is not installed.');
    case 'no_views':
      return warning('Vault view on the machine: none. Create one with the PDM View Setup wizard.');
    case 'vault_mismatch':
      return warning(`Vault view on the machine: none for “${database}”.`);
    case 'logged_out':
      return warning(`Vault view on the machine: “${view?.vault ?? database}” is not logged in.`);
    case 'ready':
      return `Vault view on the machine: ${view?.vault ?? database} at ${view?.root ?? ''}, logged in${view?.user ? ` as ${view.user}` : ''}.`;
  }
}

function ChangesRows({ folders, user }: { folders: string[]; user: string | null }) {
  const count = `${folders.length} ${folders.length === 1 ? 'folder' : 'folders'}`;
  return (
    <>
      <li>
        Vault changes: On, in {count}
        {folders.length > 0 && (
          <ul id="solidworks-review-pdm-folders">
            {folders.map((folder) => (
              <li key={folder} data-sw-folder={folder.toLowerCase()}>
                <code>{folder}</code>
              </li>
            ))}
          </ul>
        )}
      </li>
      {folders.length === 0 && (
        <li>
          {warning('Changes are on but no folders are approved, so every change will be refused.')}
        </li>
      )}
      <li>
        {user
          ? `Changes appear in PDM as ${user}.`
          : 'Changes appear in PDM under the PDM user logged in to the vault view on this machine.'}
      </li>
    </>
  );
}

function LinkedRows({
  link,
  allowChanges,
  target,
  hostStatus,
}: SolidWorksPdmReviewProps & { target: PdmTarget }) {
  const { tool, status } = target;
  const database = databaseOf(tool);
  const state = resolveLinkState({
    loading: false,
    loadError: null,
    tools: [tool],
    linkedToolId: tool.id,
    indexStatus: status,
  });
  const view = findMatchingView(hostStatus?.pdm?.views ?? [], database);
  const machineState = resolveMachineState(hostStatus, database);
  return (
    <ul id="solidworks-review-pdm-rows">
      <li>PDM tool: {tool.name}</li>
      <li>Vault database: {database}</li>
      <li>{indexLine(state, status)}</li>
      <li>{machineLine(machineState, database, view)}</li>
      {allowChanges ? (
        <ChangesRows folders={link.approved_folders} user={view?.user ?? null} />
      ) : (
        <li>Vault changes: Off</li>
      )}
    </ul>
  );
}

/** Review step rows for the PDM link (ux-spec §4). Display only. */
export function SolidWorksPdmReview({
  link,
  allowChanges,
  target,
  hostStatus,
  resolution,
}: SolidWorksPdmReviewProps) {
  const matchingTarget = target?.tool.id === link.pdm_tool_id ? target : null;
  return (
    <div
      id="solidworks-review-pdm"
      className="review-section"
      data-testid="solidworks-review-pdm"
      data-sw-review="pdm"
    >
      <h4>PDM vault</h4>
      {link.pdm_tool_id === null ? (
        <p>Not linked</p>
      ) : matchingTarget === null ? (
        resolution === 'missing' ? (
          warning(
            'The linked PDM tool no longer exists. Choose another PDM tool or set up a new one.',
          )
        ) : resolution === 'failed' ? (
          warning('PDM tools could not be loaded. Return to the SolidWorks step and try again.')
        ) : (
          <p>Loading PDM tool…</p>
        )
      ) : (
        <LinkedRows
          link={link}
          allowChanges={allowChanges}
          target={matchingTarget}
          hostStatus={hostStatus}
        />
      )}
    </div>
  );
}
