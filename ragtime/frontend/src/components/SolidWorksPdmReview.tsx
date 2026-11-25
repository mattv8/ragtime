import type { ReactNode } from 'react';
import type { SolidworksHostStatus } from '../api/solidworks';
import type { SolidworksPdmConnectionConfig, ToolConfig } from '../types/api';
import {
  findMatchingView,
  resolveLinkState,
  resolveMachineState,
  vaultFolderKey,
  type PdmLinkState,
  type PdmMachineState,
  type PdmViewFacts,
} from '../utils/solidworksPdmLink';
import type { RightsState, FolderRightsState } from '../utils/solidworksPdmRights';
import {
  formatLastIndexed,
  type PdmLinkValue,
  type PdmResolution,
  type PdmTarget,
} from './SolidWorksPdmLinkPanel';
import { PdmMessage } from './shared/PdmMessage';

export interface SolidWorksPdmReviewProps {
  link: PdmLinkValue;
  allowChanges: boolean;
  target: PdmTarget | null;
  hostStatus: SolidworksHostStatus | null;
  resolution?: PdmResolution;
  rights?: RightsState | null;
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

const folderLabels: Record<FolderRightsState, string> = {
  write_ready: 'can add and check out',
  missing_write: 'missing add or check-out rights',
  not_found: 'not found, or hidden from this PDM user',
  not_checked: 'rights not checked',
};

function ChangesRows({
  folders,
  user,
  rights,
}: {
  folders: string[];
  user: string | null;
  rights?: RightsState | null;
}) {
  const count = `${folders.length} ${folders.length === 1 ? 'folder' : 'folders'}`;
  return (
    <>
      <li>
        Vault changes: On, in {count}
        {folders.length > 0 && (
          <ul id="solidworks-review-pdm-folders">
            {folders.map((folder) => {
              const state = rights?.folders[vaultFolderKey(folder)] ?? 'not_checked';
              return (
                <li key={folder} data-sw-folder={folder.toLowerCase()} data-sw-rights={state}>
                  <code>{folder}</code>
                  {state === 'missing_write' || state === 'not_found' ? (
                    <PdmMessage tone="warning"> — {folderLabels[state]}</PdmMessage>
                  ) : (
                    <> — {folderLabels[state]}</>
                  )}
                </li>
              );
            })}
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
  rights,
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
      <li>Search index: {tool.name}</li>
      <li>Vault database: {database}</li>
      <li>{indexLine(state, status)}</li>
      <li>{machineLine(machineState, database, view)}</li>
      {!allowChanges && (
        <li data-sw-review-row="pdm-user">
          PDM user on this machine: {rights?.user ?? 'not checked'}
        </li>
      )}
      <li data-sw-review-row="vault-access">
        {rights?.root === 'ok'
          ? 'Vault access: can read the vault'
          : rights?.root === 'no_read'
            ? warning('Vault access: the PDM user cannot read the vault.')
            : 'Vault access: not checked'}
      </li>
      {allowChanges ? (
        <ChangesRows
          folders={link.approved_folders}
          user={rights?.user ?? view?.user ?? null}
          rights={rights}
        />
      ) : (
        <li>Vault changes: Off</li>
      )}
      {!allowChanges && rights?.extraWriteInfo && (
        <li data-sw-review-row="extra-write">
          <PdmMessage tone="info">
            The PDM user can also change files. Ragtime refuses changes while vault changes are off.
          </PdmMessage>
        </li>
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
  rights,
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
          rights={rights}
        />
      )}
    </div>
  );
}
