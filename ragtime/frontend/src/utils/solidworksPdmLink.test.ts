import { describe, expect, it } from 'vitest';
import type { SolidworksHostStatus } from '../api/solidworks';
import type { PdmIndexJob, PdmIndexStatus, PdmIndexStatusResponse, ToolConfig } from '../types/api';
import {
  addApprovedFolder,
  findMatchingView,
  formatPdmSchedule,
  describePdmLinkedNote,
  getPdmLinkRelations,
  MAX_APPROVED_FOLDERS,
  normalizeVaultFolder,
  removeApprovedFolder,
  resolveLinkState,
  resolveMachineState,
  type PdmLinkStateInput,
  type PdmViewFacts,
} from './solidworksPdmLink';
import { splitVaultDraft } from './solidworksPdmLink';

describe('splitVaultDraft', () => {
  it.each([
    ['', '\\', ''],
    ['   ', '\\', ''],
    ['\\', '\\', ''],
    ['\\04_Cu', '\\', '04_Cu'],
    ['04_Cu', '\\', '04_Cu'],
    ['\\04_Custom\\', '\\04_Custom', ''],
    ['/04_Custom/01', '\\04_Custom', '01'],
    ['\\04_Custom\\\\01', '\\04_Custom', '01'],
    ['C:\\Vault\\04_Custom\\x', '\\04_Custom', 'x'],
    ['  \\folder\\partial ', '\\folder', 'partial '],
  ])('splits %j, preserving trailing prefix spaces', (draft, parent, prefix) => {
    expect(splitVaultDraft(draft, 'C:\\Vault')).toEqual({ parent, prefix });
  });

  it.each([
    'D:\\other',
    '\\\\server\\share',
    '\\a\\..\\b',
    '\\a\\.',
    '\\a*',
    '\\a\\bad\u0001',
    '\\a\\é',
    '\\bad \\child',
    '\\bad.\\child',
  ])('rejects invalid draft %j', (draft) => {
    expect(splitVaultDraft(draft, 'C:\\Vault')).toBeNull();
  });
});

const VAULT_ROOT = 'C:\\PDM';
const EMPTY_MESSAGE = 'Enter a vault folder, for example \\13_Ragtime.';
const NON_ASCII_MESSAGE = 'Use plain ASCII characters in approved folder paths.';
const ROOT_ONLY_MESSAGE = 'Approving the whole vault is not allowed. Enter a folder inside it.';
const ILLEGAL_MESSAGE = 'Folder paths cannot contain “..” or any of * ? " < > | :';
const outsideMessage = (root: string | null) =>
  `That path is not inside the vault view${root ? ` ${root}` : ''}. Enter the folder inside the vault, for example \\13_Ragtime.`;

function tool(overrides: Partial<ToolConfig> & { id: string }): ToolConfig {
  return {
    name: overrides.id,
    tool_type: 'solidworks_pdm',
    enabled: true,
    description: '',
    connection_config: {} as ToolConfig['connection_config'],
    max_results: 10,
    timeout_max_seconds: 100,
    allow_write: false,
    sort_order: 0,
    undecryptable_fields: [],
    configured_secret_fields: [],
    last_test_at: null,
    last_test_result: null,
    last_test_error: null,
    created_at: '',
    updated_at: '',
    ...overrides,
  };
}

const solidworksTool = (overrides: Partial<ToolConfig> = {}) =>
  tool({ id: 'sw-1', tool_type: 'solidworks', name: 'SolidWorks', ...overrides });

function job(status: PdmIndexStatus, errorMessage: string | null = null): PdmIndexJob {
  return {
    id: 'job-1',
    tool_config_id: 'pdm-1',
    status,
    index_name: 'pdm_main',
    progress_percent: 0,
    total_documents: 10,
    processed_documents: 3,
    skipped_documents: 0,
    total_chunks: 0,
    processed_chunks: 0,
    error_message: errorMessage,
    cancel_requested: false,
    created_at: '',
    started_at: null,
    completed_at: null,
  };
}

function indexStatus(overrides: Partial<PdmIndexStatusResponse> = {}): PdmIndexStatusResponse {
  return {
    last_indexed: '2026-11-19T10:00:00Z',
    document_count: 12480,
    embedding_count: 11902,
    current_job: null,
    ...overrides,
  };
}

const pdmTool = tool({ id: 'pdm-1', name: 'Main vault' });

function linkInput(overrides: Partial<PdmLinkStateInput> = {}): PdmLinkStateInput {
  return {
    loading: false,
    loadError: null,
    tools: [solidworksTool(), pdmTool],
    linkedToolId: 'pdm-1',
    indexStatus: indexStatus(),
    ...overrides,
  };
}

function hostStatus(overrides: Partial<SolidworksHostStatus> = {}): SolidworksHostStatus {
  return {
    reachable: true,
    installed: true,
    running: true,
    host_version: '0.2.0',
    expected_host_version: '0.2.0',
    update_available: false,
    account: null,
    session: null,
    solidworks: null,
    pdm: {
      client_installed: true,
      client_version: '32.3.1',
      views: [{ vault: 'PDM', root: 'C:\\PDM', logged_in: true, user: 'readonly' }],
    },
    error: null,
    ...overrides,
  };
}

const view = (overrides: Partial<PdmViewFacts> = {}): PdmViewFacts => ({
  vault: 'PDM',
  root: 'C:\\PDM',
  logged_in: true,
  user: 'readonly',
  ...overrides,
});

describe('resolveLinkState', () => {
  it.each([
    ['loading', linkInput({ loading: true, loadError: 'boom' }), 'loading'],
    ['load_failed', linkInput({ loadError: 'Network down' }), 'load_failed'],
    ['no_tools', linkInput({ tools: [solidworksTool()], linkedToolId: null }), 'no_tools'],
    ['none_linked', linkInput({ linkedToolId: null }), 'none_linked'],
    ['linked_missing', linkInput({ linkedToolId: 'gone' }), 'linked_missing'],
    [
      'linked_missing with zero PDM tools (dangling link wins over no_tools)',
      linkInput({ tools: [solidworksTool()], linkedToolId: 'pdm-1' }),
      'linked_missing',
    ],
    [
      'linked_disabled',
      linkInput({ tools: [solidworksTool(), { ...pdmTool, enabled: false }] }),
      'linked_disabled',
    ],
    [
      'indexing (pending)',
      linkInput({ indexStatus: indexStatus({ current_job: job('pending') }) }),
      'indexing',
    ],
    [
      'indexing (running)',
      linkInput({ indexStatus: indexStatus({ current_job: job('indexing') }) }),
      'indexing',
    ],
    [
      'index_failed',
      linkInput({ indexStatus: indexStatus({ current_job: job('failed', 'x') }) }),
      'index_failed',
    ],
    [
      'never_indexed',
      linkInput({ indexStatus: indexStatus({ last_indexed: null, document_count: 0 }) }),
      'never_indexed',
    ],
    ['index_empty', linkInput({ indexStatus: indexStatus({ document_count: 0 }) }), 'index_empty'],
    ['linked_ready', linkInput(), 'linked_ready'],
    ['loading while the index status is not loaded', linkInput({ indexStatus: null }), 'loading'],
  ] as const)('resolves %s', (_name, input, expected) => {
    expect(resolveLinkState(input)).toBe(expected);
  });

  it('gives a disabled tool precedence over its index state', () => {
    const input = linkInput({
      tools: [solidworksTool(), { ...pdmTool, enabled: false }],
      indexStatus: indexStatus({ current_job: job('indexing') }),
    });
    expect(resolveLinkState(input)).toBe('linked_disabled');
  });

  it('treats a linked id that is not a PDM tool as missing', () => {
    expect(resolveLinkState(linkInput({ linkedToolId: 'sw-1' }))).toBe('linked_missing');
  });
});

describe('findMatchingView', () => {
  const views = [view({ vault: 'HAM-PDM' }), view({ vault: 'PDM' })];

  it.each([
    ['exact name', 'PDM', 'PDM'],
    ['case-insensitive name', 'pdm', 'PDM'],
    ['other view', 'ham-pdm', 'HAM-PDM'],
  ])('matches %s', (_name, database, expectedVault) => {
    expect(findMatchingView(views, database)?.vault).toBe(expectedVault);
  });

  it.each([
    ['no view with that name', 'Other'],
    ['empty database', ''],
    ['null database', null],
    ['undefined database', undefined],
  ])('returns null for %s', (_name, database) => {
    expect(findMatchingView(views, database)).toBeNull();
  });

  it('returns null when there are no views', () => {
    expect(findMatchingView([], 'PDM')).toBeNull();
  });
});

describe('resolveMachineState', () => {
  it.each([
    ['host status unknown', null, 'PDM', 'unknown'],
    ['host not running', hostStatus({ running: false }), 'PDM', 'unknown'],
    ['pdm block missing', hostStatus({ pdm: null }), 'PDM', 'unknown'],
    ['no database to compare', hostStatus(), null, 'unknown'],
    [
      'client not installed',
      hostStatus({ pdm: { client_installed: false, client_version: null, views: [] } }),
      'PDM',
      'client_missing',
    ],
    [
      'client installed without views',
      hostStatus({ pdm: { client_installed: true, client_version: '32.3.1', views: [] } }),
      'PDM',
      'no_views',
    ],
    ['views but none match', hostStatus(), 'Other', 'vault_mismatch'],
    [
      'matching view logged out',
      hostStatus({
        pdm: {
          client_installed: true,
          client_version: '32.3.1',
          views: [{ vault: 'PDM', root: 'C:\\PDM', logged_in: false, user: null }],
        },
      }),
      'pdm',
      'logged_out',
    ],
    ['matching view logged in', hostStatus(), 'pdm', 'ready'],
  ] as const)('resolves %s', (_name, status, database, expected) => {
    expect(resolveMachineState(status, database)).toBe(expected);
  });
});

describe('normalizeVaultFolder', () => {
  it.each([
    ['a vault-relative folder', '\\13_Ragtime', '\\13_Ragtime', false],
    ['a folder without the leading backslash', '13_Ragtime', '\\13_Ragtime', false],
    ['forward slashes and spaces around the path', '  /13_Ragtime/  ', '\\13_Ragtime', false],
    ['repeated backslashes', '\\13_Ragtime\\\\Block', '\\13_Ragtime\\Block', false],
    ['an absolute path under the view root', 'C:\\PDM\\13_Ragtime', '\\13_Ragtime', true],
    ['the view root in another case with slashes', 'c:/pdm/13_Ragtime/', '\\13_Ragtime', true],
    ['the printable ASCII range edges', '\\Work ~1', '\\Work ~1', false],
    [
      'a file path under the view root',
      'C:\\PDM\\13_Ragtime\\Block1.SLDPRT',
      '\\13_Ragtime\\Block1.SLDPRT',
      true,
    ],
  ])('accepts %s', (_name, input, path, converted) => {
    expect(normalizeVaultFolder(input, VAULT_ROOT)).toEqual({ ok: true, path, converted });
  });

  it('strips a view root configured with a trailing backslash', () => {
    expect(normalizeVaultFolder('C:\\PDM\\13_Ragtime', 'C:\\PDM\\')).toEqual({
      ok: true,
      path: '\\13_Ragtime',
      converted: true,
    });
  });

  it('keeps a folder whose name only starts like the view root', () => {
    expect(normalizeVaultFolder('\\PDMX\\13', VAULT_ROOT)).toEqual({
      ok: true,
      path: '\\PDMX\\13',
      converted: false,
    });
  });

  it.each([
    ['an empty entry', '', EMPTY_MESSAGE],
    ['a blank entry', '   ', EMPTY_MESSAGE],
    ['the vault root', '\\', ROOT_ONLY_MESSAGE],
    ['the vault root as a path', 'C:\\PDM', ROOT_ONLY_MESSAGE],
    ['the vault root with a trailing separator', 'C:\\PDM\\', ROOT_ONLY_MESSAGE],
    ['a drive path outside the view', 'D:\\Other\\x', outsideMessage(VAULT_ROOT)],
    ['a UNC path', '\\\\server\\share', outsideMessage(VAULT_ROOT)],
    ['a UNC path with forward slashes', '//server/share', outsideMessage(VAULT_ROOT)],
    ['a parent segment', '\\13_Ragtime\\..\\x', ILLEGAL_MESSAGE],
    ['a current-directory segment', '\\13_Ragtime\\.', ILLEGAL_MESSAGE],
    ['an asterisk', '\\a*b', ILLEGAL_MESSAGE],
    ['a question mark', '\\a?b', ILLEGAL_MESSAGE],
    ['a double quote', '\\a"b', ILLEGAL_MESSAGE],
    ['a less-than sign', '\\a<b', ILLEGAL_MESSAGE],
    ['a greater-than sign', '\\a>b', ILLEGAL_MESSAGE],
    ['a pipe', '\\a|b', ILLEGAL_MESSAGE],
    ['a colon after the root', '\\a:b', ILLEGAL_MESSAGE],
    [
      'a segment with a leading space',
      '\\13_Ragtime\\ Block',
      'Folder names cannot start or end with a space, and cannot end with a dot.',
    ],
    [
      'a segment with a trailing dot',
      '\\13_Ragtime\\Block.',
      'Folder names cannot start or end with a space, and cannot end with a dot.',
    ],
    [
      'a segment with a trailing space',
      '\\Block \\x',
      'Folder names cannot start or end with a space, and cannot end with a dot.',
    ],
    ['a control character', '\\13_Rag\ttime', 'Folder paths cannot contain control characters.'],
    ['a bell character', '\\13_Rag\u0007time', 'Folder paths cannot contain control characters.'],
    ['a DEL character', '\\13_Rag\u007ftime', 'Folder paths cannot contain control characters.'],
    ['a non-ASCII letter', '\\Projet\u00e9', NON_ASCII_MESSAGE],
    ['a non-breaking space', '\\13\u00a0Ragtime', NON_ASCII_MESSAGE],
    ['an emoji', '\\13_\u{1F4C1}', NON_ASCII_MESSAGE],
    ['a non-ASCII character in an absolute path', 'C:\\PDM\\Ordn\u00e9r', NON_ASCII_MESSAGE],
  ])('rejects %s', (_name, input, error) => {
    expect(normalizeVaultFolder(input, VAULT_ROOT)).toEqual({ ok: false, error });
  });

  it('names no view root when none is known', () => {
    expect(normalizeVaultFolder('C:\\PDM\\13_Ragtime', null)).toEqual({
      ok: false,
      error: outsideMessage(null),
    });
  });

  it('accepts a path of exactly 260 characters and rejects 261', () => {
    const body = 'a'.repeat(259);
    expect(normalizeVaultFolder(`\\${body}`, VAULT_ROOT)).toEqual({
      ok: true,
      path: `\\${body}`,
      converted: false,
    });
    expect(normalizeVaultFolder(`\\${body}a`, VAULT_ROOT)).toEqual({
      ok: false,
      error: 'Folder paths can be at most 260 characters.',
    });
  });
});

describe('addApprovedFolder', () => {
  it('adds a normalized folder to an empty list without a notice', () => {
    expect(addApprovedFolder([], '13_Ragtime', VAULT_ROOT)).toEqual({
      ok: true,
      folders: ['\\13_Ragtime'],
      added: '\\13_Ragtime',
      notice: null,
    });
  });

  it('announces a converted view-root prefix', () => {
    expect(addApprovedFolder([], 'C:\\PDM\\13_Ragtime', VAULT_ROOT)).toEqual({
      ok: true,
      folders: ['\\13_Ragtime'],
      added: '\\13_Ragtime',
      notice: 'Converted C:\\PDM\\13_Ragtime to \\13_Ragtime.',
    });
  });

  it('rejects a duplicate case-insensitively and names the stored spelling', () => {
    expect(addApprovedFolder(['\\13_Ragtime'], '\\13_RAGTIME', VAULT_ROOT)).toEqual({
      ok: false,
      error: '\\13_Ragtime is already in the list.',
      info: false,
    });
  });

  it('reports a folder covered by an existing parent as info, not an error', () => {
    expect(addApprovedFolder(['\\13_Ragtime'], '\\13_Ragtime\\Rev', VAULT_ROOT)).toEqual({
      ok: false,
      error: '\\13_Ragtime\\Rev is already covered by \\13_Ragtime.',
      info: true,
    });
  });

  it('detects coverage case-insensitively', () => {
    expect(addApprovedFolder(['\\13_Ragtime'], '\\13_ragtime\\rev', VAULT_ROOT)).toEqual({
      ok: false,
      error: '\\13_ragtime\\rev is already covered by \\13_Ragtime.',
      info: true,
    });
  });

  it('does not treat a sibling with a shared prefix as covered', () => {
    expect(addApprovedFolder(['\\13_Ragtime'], '\\13_Ragtime2', VAULT_ROOT)).toEqual({
      ok: true,
      folders: ['\\13_Ragtime', '\\13_Ragtime2'],
      added: '\\13_Ragtime2',
      notice: null,
    });
  });

  it('adds a parent of existing entries, removes the children it covers, and says so', () => {
    expect(
      addApprovedFolder(
        ['\\13_Ragtime\\A', '\\Other', '\\13_Ragtime\\B'],
        '\\13_Ragtime',
        VAULT_ROOT,
      ),
    ).toEqual({
      ok: true,
      folders: ['\\Other', '\\13_Ragtime'],
      added: '\\13_Ragtime',
      notice: 'Replaced \\13_Ragtime\\A, \\13_Ragtime\\B with \\13_Ragtime, which covers them.',
    });
  });

  it('replaces children case-insensitively', () => {
    expect(addApprovedFolder(['\\13_RAGTIME\\a'], '\\13_ragtime', VAULT_ROOT)).toEqual({
      ok: true,
      folders: ['\\13_ragtime'],
      added: '\\13_ragtime',
      notice: 'Replaced \\13_RAGTIME\\a with \\13_ragtime, which covers them.',
    });
  });

  it('joins a converted notice and a replacement notice', () => {
    const result = addApprovedFolder(['\\13_Ragtime\\A'], 'C:\\PDM\\13_Ragtime', VAULT_ROOT);
    expect(result).toEqual({
      ok: true,
      folders: ['\\13_Ragtime'],
      added: '\\13_Ragtime',
      notice:
        'Converted C:\\PDM\\13_Ragtime to \\13_Ragtime. Replaced \\13_Ragtime\\A with \\13_Ragtime, which covers them.',
    });
  });

  it('passes a normalization error through as an error', () => {
    expect(addApprovedFolder(['\\13_Ragtime'], '', VAULT_ROOT)).toEqual({
      ok: false,
      error: EMPTY_MESSAGE,
      info: false,
    });
  });

  it(`refuses a folder beyond the cap of ${MAX_APPROVED_FOLDERS}`, () => {
    const full = Array.from({ length: MAX_APPROVED_FOLDERS }, (_, index) => `\\F${index}`);
    expect(addApprovedFolder(full, '\\F99', VAULT_ROOT)).toEqual({
      ok: false,
      error: 'You can approve up to 25 folders.',
      info: false,
    });
  });

  it('allows the last folder below the cap', () => {
    const almostFull = Array.from(
      { length: MAX_APPROVED_FOLDERS - 1 },
      (_, index) => `\\F${index}`,
    );
    const result = addApprovedFolder(almostFull, '\\F99', VAULT_ROOT);
    expect(result.ok).toBe(true);
  });

  it('counts a parent replacing its children against the cap', () => {
    const children = Array.from(
      { length: MAX_APPROVED_FOLDERS - 1 },
      (_, index) => `\\P\\c${index}`,
    );
    expect(addApprovedFolder([...children, '\\Q'], '\\P', VAULT_ROOT)).toEqual({
      ok: true,
      folders: ['\\Q', '\\P'],
      added: '\\P',
      notice: expect.stringContaining('Replaced'),
    });
  });
});

describe('removeApprovedFolder', () => {
  it.each([
    ['removes an exact match', ['\\A', '\\B'], '\\A', ['\\B']],
    ['removes case-insensitively', ['\\13_Ragtime', '\\B'], '\\13_RAGTIME', ['\\B']],
    ['leaves the list alone when absent', ['\\A'], '\\B', ['\\A']],
  ])('%s', (_name, folders, path, expected) => {
    expect(removeApprovedFolder(folders, path)).toEqual(expected);
  });

  it('returns a new array and does not mutate the input', () => {
    const folders = ['\\A', '\\B'];
    const next = removeApprovedFolder(folders, '\\A');
    expect(next).not.toBe(folders);
    expect(folders).toEqual(['\\A', '\\B']);
  });
});

describe('getPdmLinkRelations', () => {
  it('lists each SolidWorks tool that links a PDM tool, with its write state and folder count', () => {
    const linked = solidworksTool({
      id: 'sw-2',
      name: 'Render PC',
      allow_write: true,
      connection_config: {
        pdm_tool_id: 'pdm-1',
        pdm_approved_folders: ['\\13_Ragtime', '\\14_Other'],
      } as ToolConfig['connection_config'],
    });
    expect(getPdmLinkRelations([solidworksTool(), linked, pdmTool])).toEqual([
      {
        solidworksToolId: 'sw-2',
        solidworksToolName: 'Render PC',
        solidworksEnabled: true,
        pdmToolId: 'pdm-1',
        allowWrite: true,
        approvedFolderCount: 2,
      },
    ]);
  });

  it('reports whether the SolidWorks tool is enabled', () => {
    const disabled = solidworksTool({
      enabled: false,
      connection_config: { pdm_tool_id: 'pdm-1' } as ToolConfig['connection_config'],
    });
    expect(getPdmLinkRelations([disabled])[0].solidworksEnabled).toBe(false);
  });

  it('counts zero folders when none are stored', () => {
    const linked = solidworksTool({
      connection_config: { pdm_tool_id: 'pdm-1' } as ToolConfig['connection_config'],
    });
    expect(getPdmLinkRelations([linked])[0].approvedFolderCount).toBe(0);
  });

  it('ignores PDM tools and SolidWorks tools without a link', () => {
    expect(getPdmLinkRelations([pdmTool, solidworksTool()])).toEqual([]);
  });
});

describe('describePdmLinkedNote', () => {
  const relation = (name: string, solidworksEnabled = true) => ({
    solidworksToolId: `sw-${name}`,
    solidworksToolName: name,
    solidworksEnabled,
    pdmToolId: 'pdm-1',
    allowWrite: false,
    approvedFolderCount: 0,
  });

  it('omits the note when no SolidWorks tool is enabled', () => {
    expect(describePdmLinkedNote([])).toBeNull();
    expect(describePdmLinkedNote([relation('Render PC', false)])).toBeNull();
  });

  it('names the one enabled SolidWorks tool and skips disabled ones', () => {
    expect(describePdmLinkedNote([relation('Render PC'), relation('Spare', false)])).toBe(
      "Served through “Render PC” for callers who can use it. They don't see this tool separately.",
    );
  });

  it('lists several enabled tools with a conjunction', () => {
    expect(describePdmLinkedNote([relation('A'), relation('B')])).toBe(
      "Served through “A” and “B” for callers who can use either. They don't see this tool separately.",
    );
  });

  it('caps the named tools at three and counts the rest', () => {
    const names = ['A', 'B', 'C', 'D', 'E'].map((name) => relation(name));
    expect(describePdmLinkedNote(names)).toBe(
      "Served through “A”, “B”, “C”, and 2 more for callers who can use any of them. They don't see this tool separately.",
    );
  });
});

describe('formatPdmSchedule', () => {
  it.each([
    [null, 'Manual only'],
    [undefined, 'Manual only'],
    [0, 'Manual only'],
    [1, 'Every hour'],
    [6, 'Every 6 hours'],
    [24, 'Every 24 hours'],
  ])('formats %p as %s', (hours, expected) => {
    expect(formatPdmSchedule(hours)).toBe(expected);
  });
});
