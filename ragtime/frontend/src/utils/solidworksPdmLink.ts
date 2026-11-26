import type { SolidworksConnectionConfig, SolidworksHostStatus } from '../api/solidworks';
import type { PdmIndexStatus, PdmIndexStatusResponse, ToolConfig } from '../types/api';

export const MAX_APPROVED_FOLDERS = 25;

const MAX_PATH_LENGTH = 260;
const EXAMPLE_FOLDER = '\\13_Ragtime';
const ILLEGAL_CHARACTER = /[*?"<>|:]/;
const ABSOLUTE_PATH = /^(?:[a-z]:|\\\\)/i;
const BACKSLASHES_ONLY = /^\\*$/;
const ACTIVE_INDEX_JOB_STATES: readonly PdmIndexStatus[] = ['pending', 'indexing'];

export type PdmLinkState =
  | 'loading'
  | 'load_failed'
  | 'no_tools'
  | 'none_linked'
  | 'linked_missing'
  | 'linked_disabled'
  | 'never_indexed'
  | 'indexing'
  | 'index_failed'
  | 'index_empty'
  | 'linked_ready';

export type PdmMachineState =
  | 'unknown'
  | 'client_missing'
  | 'no_views'
  | 'vault_mismatch'
  | 'logged_out'
  | 'ready';

export interface PdmViewFacts {
  vault: string;
  root: string;
  logged_in: boolean;
  user: string | null;
}

export interface PdmLinkStateInput {
  loading: boolean;
  loadError: string | null;
  tools: ToolConfig[];
  linkedToolId: string | null;
  indexStatus: PdmIndexStatusResponse | null;
}

export type FolderNormalizeResult =
  | { ok: true; path: string; converted: boolean }
  | { ok: false; error: string };

export type AddFolderResult =
  | { ok: true; folders: string[]; added: string; notice: string | null }
  | { ok: false; error: string; info: boolean };

export interface PdmLinkRelation {
  solidworksToolId: string;
  solidworksToolName: string;
  /** Whether the SolidWorks tool is enabled; disabled tools no longer serve the PDM tool. */
  solidworksEnabled: boolean;
  pdmToolId: string;
  allowWrite: boolean;
  approvedFolderCount: number;
}

const MAX_NAMED_TOOLS = 3;

/** "A", "A and B", or "A, B, and C" (the Oxford comma, as Intl.ListFormat('en') writes it). */
function joinConjunction(items: string[]): string {
  if (items.length <= 2) return items.join(' and ');
  return `${items.slice(0, -1).join(', ')}, and ${items[items.length - 1]}`;
}

/**
 * Card note for a PDM tool that enabled SolidWorks tools serve. Returns null
 * when none is enabled, so the note is omitted and the PDM tool stays visible.
 */
export function describePdmLinkedNote(relations: PdmLinkRelation[]): string | null {
  const names = relations
    .filter((relation) => relation.solidworksEnabled)
    .map((relation) => relation.solidworksToolName);
  if (names.length === 0) return null;
  if (names.length === 1) {
    return `Served through “${names[0]}” for callers who can use it. They don't see this tool separately.`;
  }
  const quoted = names.slice(0, MAX_NAMED_TOOLS).map((name) => `“${name}”`);
  const hidden = names.length - quoted.length;
  const listed = joinConjunction(hidden > 0 ? [...quoted, `${hidden} more`] : quoted);
  const who = names.length === 2 ? 'either' : 'any of them';
  return `Served through ${listed} for callers who can use ${who}. They don't see this tool separately.`;
}

const MESSAGES = {
  empty: `Enter a vault folder, for example ${EXAMPLE_FOLDER}.`,
  control: 'Folder paths cannot contain control characters.',
  nonAscii: 'Use plain ASCII characters in approved folder paths.',
  tooLong: `Folder paths can be at most ${MAX_PATH_LENGTH} characters.`,
  rootOnly: 'Approving the whole vault is not allowed. Enter a folder inside it.',
  illegal: 'Folder paths cannot contain “..” or any of * ? " < > | :',
  edgeSpace: 'Folder names cannot start or end with a space, and cannot end with a dot.',
  limit: `You can approve up to ${MAX_APPROVED_FOLDERS} folders.`,
  duplicate: (folder: string) => `${folder} is already in the list.`,
  covered: (folder: string, parent: string) => `${folder} is already covered by ${parent}.`,
};

function hasControlCharacter(value: string): boolean {
  return [...value].some((char) => {
    const code = char.charCodeAt(0);
    return code < 0x20 || code === 0x7f;
  });
}

function hasNonAsciiCharacter(value: string): boolean {
  return /[^\x20-\x7E]/.test(value);
}

function toBackslashes(value: string): string {
  return value.replace(/\//g, '\\');
}

function viewRootPrefix(vaultRoot: string | null): string | null {
  const root = toBackslashes(vaultRoot?.trim() ?? '').replace(/\\+$/, '');
  return root === '' ? null : root;
}

function isWithinViewRoot(value: string, root: string): boolean {
  const lower = value.toLowerCase();
  const prefix = root.toLowerCase();
  return lower === prefix || lower.startsWith(`${prefix}\\`);
}

function outsideMessage(root: string | null): string {
  const where = root ? ` ${root}` : '';
  return `That path is not inside the vault view${where}. Enter the folder inside the vault, for example ${EXAMPLE_FOLDER}.`;
}

function rejected(error: string): FolderNormalizeResult {
  return { ok: false, error };
}

function isDotSegment(segment: string): boolean {
  return segment === '.' || segment === '..';
}

function hasEdgeSpaceOrDot(segment: string): boolean {
  return segment !== segment.trim() || segment.endsWith('.');
}

export function vaultFolderKey(path: string): string {
  return path.toLowerCase();
}

export function isWithinFolder(folder: string, parent: string): boolean {
  return vaultFolderKey(folder).startsWith(`${vaultFolderKey(parent)}\\`);
}

export function normalizeVaultFolder(
  input: string,
  vaultRoot: string | null,
): FolderNormalizeResult {
  const trimmed = input.trim();
  if (trimmed === '') return rejected(MESSAGES.empty);
  if (hasControlCharacter(trimmed)) return rejected(MESSAGES.control);
  if (hasNonAsciiCharacter(trimmed)) return rejected(MESSAGES.nonAscii);
  if (trimmed.length > MAX_PATH_LENGTH) return rejected(MESSAGES.tooLong);

  const root = viewRootPrefix(vaultRoot);
  const slashed = toBackslashes(trimmed);
  const relative =
    root !== null && isWithinViewRoot(slashed, root) ? slashed.slice(root.length) : slashed;
  if (BACKSLASHES_ONLY.test(relative)) return rejected(MESSAGES.rootOnly);
  // Checked before repeated separators are collapsed, so \\server\share is refused rather than read as \server\share.
  if (ABSOLUTE_PATH.test(relative)) return rejected(outsideMessage(root));

  const rooted = relative.startsWith('\\') ? relative : `\\${relative}`;
  const folder = rooted.replace(/\\{2,}/g, '\\').replace(/\\$/, '');
  if (ILLEGAL_CHARACTER.test(folder) || folder.split('\\').some(isDotSegment)) {
    return rejected(MESSAGES.illegal);
  }
  if (folder.slice(1).split('\\').some(hasEdgeSpaceOrDot)) return rejected(MESSAGES.edgeSpace);
  return { ok: true, path: folder, converted: relative !== slashed };
}

function buildNotice(parts: {
  input: string;
  converted: boolean;
  path: string;
  replaced: string[];
}): string | null {
  const notices: string[] = [];
  if (parts.converted) notices.push(`Converted ${parts.input.trim()} to ${parts.path}.`);
  if (parts.replaced.length > 0) {
    notices.push(`Replaced ${parts.replaced.join(', ')} with ${parts.path}, which covers them.`);
  }
  return notices.length > 0 ? notices.join(' ') : null;
}

export function addApprovedFolder(
  folders: string[],
  input: string,
  vaultRoot: string | null,
): AddFolderResult {
  const normalized = normalizeVaultFolder(input, vaultRoot);
  if (!normalized.ok) return { ok: false, error: normalized.error, info: false };
  const { path } = normalized;

  const existing = folders.find((folder) => vaultFolderKey(folder) === vaultFolderKey(path));
  if (existing !== undefined) {
    return { ok: false, error: MESSAGES.duplicate(existing), info: false };
  }
  const parent = folders.find((folder) => isWithinFolder(path, folder));
  if (parent !== undefined) {
    return { ok: false, error: MESSAGES.covered(path, parent), info: true };
  }

  const replaced = folders.filter((folder) => isWithinFolder(folder, path));
  const kept = folders.filter((folder) => !isWithinFolder(folder, path));
  if (kept.length >= MAX_APPROVED_FOLDERS) return { ok: false, error: MESSAGES.limit, info: false };

  return {
    ok: true,
    folders: [...kept, path],
    added: path,
    notice: buildNotice({ input, converted: normalized.converted, path, replaced }),
  };
}

export function removeApprovedFolder(folders: string[], path: string): string[] {
  return folders.filter((folder) => vaultFolderKey(folder) !== vaultFolderKey(path));
}

function isPdmTool(tool: ToolConfig): boolean {
  return tool.tool_type === 'solidworks_pdm';
}

function resolveIndexState(status: PdmIndexStatusResponse | null): PdmLinkState {
  if (!status) return 'loading';
  const job = status.current_job;
  if (job && ACTIVE_INDEX_JOB_STATES.includes(job.status)) return 'indexing';
  if (job?.status === 'failed') return 'index_failed';
  if (status.document_count === 0) return status.last_indexed ? 'index_empty' : 'never_indexed';
  return 'linked_ready';
}

export function resolveLinkState(input: PdmLinkStateInput): PdmLinkState {
  if (input.loading) return 'loading';
  if (input.loadError !== null) return 'load_failed';
  const pdmTools = input.tools.filter(isPdmTool);
  if (!input.linkedToolId) return pdmTools.length === 0 ? 'no_tools' : 'none_linked';
  const linked = pdmTools.find((tool) => tool.id === input.linkedToolId);
  if (!linked) return 'linked_missing';
  if (!linked.enabled) return 'linked_disabled';
  return resolveIndexState(input.indexStatus);
}

export function findMatchingView(
  views: PdmViewFacts[],
  database: string | null | undefined,
): PdmViewFacts | null {
  if (!database) return null;
  const key = vaultFolderKey(database);
  return views.find((view) => vaultFolderKey(view.vault) === key) ?? null;
}

export function resolveMachineState(
  hostStatus: SolidworksHostStatus | null,
  database: string | null | undefined,
): PdmMachineState {
  const pdm = hostStatus?.running ? hostStatus.pdm : null;
  if (!pdm || !database) return 'unknown';
  if (!pdm.client_installed) return 'client_missing';
  if (pdm.views.length === 0) return 'no_views';
  const view = findMatchingView(pdm.views, database);
  if (view === null) return 'vault_mismatch';
  return view.logged_in ? 'ready' : 'logged_out';
}

export function getPdmLinkRelations(tools: ToolConfig[]): PdmLinkRelation[] {
  return tools.flatMap((tool) => {
    if (tool.tool_type !== 'solidworks') return [];
    const config = tool.connection_config as Partial<SolidworksConnectionConfig>;
    if (!config.pdm_tool_id) return [];
    return [
      {
        solidworksToolId: tool.id,
        solidworksToolName: tool.name,
        solidworksEnabled: tool.enabled,
        pdmToolId: config.pdm_tool_id,
        allowWrite: tool.allow_write,
        approvedFolderCount: config.pdm_approved_folders?.length ?? 0,
      },
    ];
  });
}

export function formatPdmSchedule(intervalHours: number | null | undefined): string {
  if (!intervalHours) return 'Manual only';
  return intervalHours === 1 ? 'Every hour' : `Every ${intervalHours} hours`;
}
