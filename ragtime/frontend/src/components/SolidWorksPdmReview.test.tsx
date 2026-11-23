import { cleanup, render } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';
import type { SolidworksHostStatus } from '../api/solidworks';
import type { PdmIndexJob, PdmIndexStatusResponse, ToolConfig } from '../types/api';
import { SolidWorksPdmReview, type SolidWorksPdmReviewProps } from './SolidWorksPdmReview';

type PdmHost = NonNullable<SolidworksHostStatus['pdm']>;

const VIEW = { vault: 'Ragtime', root: 'C:\\PDM\\Ragtime', logged_in: true, user: 'pdm.svc' };
const PDM_OK: PdmHost = { client_installed: true, client_version: '2024', views: [VIEW] };
const READY: PdmIndexStatusResponse = {
  last_indexed: '2026-11-19T10:00:00Z',
  document_count: 12480,
  embedding_count: 11902,
  current_job: null,
};
const FOLDER = '\\13_Ragtime';

function pdmTool(overrides: Partial<ToolConfig> = {}): ToolConfig {
  return {
    id: 'pdm-a',
    name: 'PDM Ragtime',
    tool_type: 'solidworks_pdm',
    enabled: true,
    description: '',
    connection_config: {
      host: 'pdm-sql',
      port: 1433,
      user: 'sa',
      password: '',
      database: 'Ragtime',
      reindex_interval_hours: 24,
    } as ToolConfig['connection_config'],
    max_results: 10,
    timeout_max_seconds: 100,
    allow_write: false,
    sort_order: 0,
    undecryptable_fields: [],
    configured_secret_fields: [],
    last_test_at: null,
    last_test_result: null,
    last_test_error: null,
    created_at: '2026-11-01T00:00:00Z',
    updated_at: '2026-11-01T00:00:00Z',
    ...overrides,
  };
}

function hostWithPdm(pdm: PdmHost | null): SolidworksHostStatus {
  return {
    reachable: true,
    installed: true,
    running: true,
    host_version: '0.1.0',
    expected_host_version: '0.1.0',
    update_available: false,
    account: null,
    session: { interactive: true },
    solidworks: null,
    error: null,
    pdm,
  };
}

function job(status: PdmIndexJob['status']): PdmIndexJob {
  return {
    id: 'job-1',
    tool_config_id: 'pdm-a',
    status,
    index_name: 'ragtime',
    progress_percent: 0,
    total_documents: 0,
    processed_documents: 0,
    skipped_documents: 0,
    total_chunks: 0,
    processed_chunks: 0,
    error_message: null,
    cancel_requested: false,
    created_at: '2026-11-19T10:00:00Z',
    started_at: null,
    completed_at: null,
  };
}

function renderReview(overrides: Partial<SolidWorksPdmReviewProps> = {}): string {
  const props: SolidWorksPdmReviewProps = {
    link: { pdm_tool_id: 'pdm-a', approved_folders: [] },
    allowChanges: false,
    target: { tool: pdmTool(), status: READY },
    hostStatus: hostWithPdm(PDM_OK),
    ...overrides,
  };
  render(<SolidWorksPdmReview {...props} />);
  return document.getElementById('solidworks-review-pdm')?.textContent ?? '';
}

describe('SolidWorksPdmReview', () => {
  afterEach(cleanup);

  it.each([
    ['write_ready', 'can add and check out'],
    ['missing_write', 'missing add or check-out rights'],
    ['not_found', 'not found, or hidden from this PDM user'],
    ['not_checked', 'rights not checked'],
  ] as const)('shows folder rights %s', (state, label) => {
    renderReview({
      allowChanges: true,
      link: { pdm_tool_id: 'pdm-a', approved_folders: [FOLDER] },
      rights: {
        root: 'ok',
        folders: { [FOLDER.toLowerCase()]: state },
        user: 'checked.user',
        extraWriteInfo: false,
      },
    });
    const folder = document.querySelector('#solidworks-review-pdm-folders li');
    expect(folder?.getAttribute('data-sw-rights')).toBe(state);
    expect(folder?.textContent).toContain(label);
    if (state === 'missing_write' || state === 'not_found') {
      expect(folder?.querySelector('.sw-message[data-tone="warning"]')?.textContent).toContain(
        label,
      );
    } else {
      expect(folder?.querySelector('.sw-message')).toBeNull();
    }
    expect(document.querySelector('[data-sw-review-row="vault-access"]')?.textContent).toContain(
      'can read the vault',
    );
    expect(document.body.textContent).toContain('Changes appear in PDM as checked.user.');
    expect(document.querySelector('[data-sw-review-row="pdm-user"]')).toBeNull();
  });

  it('shows the checked user when changes are off', () => {
    renderReview({
      rights: { root: 'ok', folders: {}, user: 'checked.user', extraWriteInfo: false },
    });
    expect(document.querySelector('[data-sw-review-row="pdm-user"]')?.textContent).toContain(
      'checked.user',
    );
  });

  it('shows a warning for no read and information about extra write rights', () => {
    renderReview({ rights: { root: 'no_read', folders: {}, user: null, extraWriteInfo: true } });
    expect(document.querySelector('[data-sw-review-row="vault-access"]')?.textContent).toContain(
      'cannot read',
    );
    expect(document.querySelector('[data-sw-review-row="extra-write"]')?.textContent).toContain(
      'Ragtime refuses changes',
    );
  });

  it.each([null, undefined])('shows not checked when rights are %s', (rights) => {
    renderReview({
      rights,
      allowChanges: true,
      link: { pdm_tool_id: 'pdm-a', approved_folders: [FOLDER] },
    });
    expect(document.querySelector('[data-sw-review-row="pdm-user"]')).toBeNull();
    expect(document.querySelector('[data-sw-review-row="vault-access"]')?.textContent).toContain(
      'not checked',
    );
    expect(
      document.querySelector('#solidworks-review-pdm-folders li')?.getAttribute('data-sw-rights'),
    ).toBe('not_checked');
  });

  it('shows an unchecked user when changes are off and rights are unavailable', () => {
    renderReview({ rights: null });
    expect(document.querySelector('[data-sw-review-row="pdm-user"]')?.textContent).toContain(
      'not checked',
    );
  });

  it('is a review section with stable hooks', () => {
    renderReview();
    const root = document.getElementById('solidworks-review-pdm') as HTMLElement;
    expect(root.dataset.swReview).toBe('pdm');
    expect(root.dataset.testid).toBe('solidworks-review-pdm');
    expect(root.className).toBe('review-section');
  });

  it('says Not linked when no PDM tool is chosen', () => {
    const text = renderReview({ link: { pdm_tool_id: null, approved_folders: [] }, target: null });
    expect(text).toBe('PDM vaultNot linked');
  });

  it('lists a healthy link with its index and machine view', () => {
    const text = renderReview();
    expect(text).toContain('Search index: PDM Ragtime');
    expect(text).toContain('Vault database: Ragtime');
    expect(text).toContain('Index: 12,480 documents, last indexed');
    expect(text).toContain(
      'Vault view on the machine: Ragtime at C:\\PDM\\Ragtime, logged in as pdm.svc.',
    );
    expect(text).toContain('Vault changes: Off');
  });

  it('reports a missing linked tool as a warning', () => {
    const text = renderReview({ target: null, resolution: 'missing' });
    expect(text).toContain('The linked PDM tool no longer exists.');
  });

  it('does not report an unresolved or mismatched target as deleted', () => {
    const text = renderReview({ target: { tool: pdmTool({ id: 'other' }), status: READY } });
    expect(text).toContain('Loading PDM tool');
    expect(text).not.toContain('no longer exists');
  });

  it('distinguishes list failure from a missing tool', () => {
    const text = renderReview({ target: null, resolution: 'failed' });
    expect(text).toContain('could not be loaded');
    expect(text).not.toContain('no longer exists');
  });

  it.each([
    [
      'never_indexed',
      { ...READY, document_count: 0, last_indexed: null },
      'Index: not indexed yet.',
    ],
    ['indexing', { ...READY, current_job: job('indexing') }, 'Index: indexing in progress.'],
    [
      'index_empty',
      { ...READY, document_count: 0 },
      'Index: empty, the last index found no documents.',
    ],
    [
      'index_failed',
      { ...READY, document_count: 0, current_job: job('failed') },
      'Index: the last index failed.',
    ],
  ])('warns about the %s index', (_state, status, sentence) => {
    expect(renderReview({ target: { tool: pdmTool(), status } })).toContain(sentence);
  });

  it('warns when the linked tool is disabled', () => {
    const text = renderReview({ target: { tool: pdmTool({ enabled: false }), status: READY } });
    expect(text).toContain('Index: the PDM tool is disabled, so the AI cannot search the vault.');
  });

  it.each([
    [
      'client missing',
      hostWithPdm({ client_installed: false, client_version: null, views: [] }),
      'Vault view on the machine: the PDM client is not installed.',
    ],
    [
      'no views',
      hostWithPdm({ client_installed: true, client_version: '2024', views: [] }),
      'Vault view on the machine: none. Create one with the PDM View Setup wizard.',
    ],
    [
      'vault mismatch',
      hostWithPdm({ ...PDM_OK, views: [{ ...VIEW, vault: 'Other' }] }),
      'Vault view on the machine: none for “Ragtime”.',
    ],
    [
      'logged out',
      hostWithPdm({ ...PDM_OK, views: [{ ...VIEW, logged_in: false }] }),
      'Vault view on the machine: “Ragtime” is not logged in.',
    ],
  ])('warns about a machine problem: %s', (_name, hostStatus, sentence) => {
    expect(renderReview({ hostStatus })).toContain(sentence);
  });

  it('says the machine is not checked when the host is not running', () => {
    expect(renderReview({ hostStatus: null })).toContain('Vault view on the machine: not checked');
  });

  it('lists approved folders and the PDM user when changes are on', () => {
    const text = renderReview({
      link: { pdm_tool_id: 'pdm-a', approved_folders: [FOLDER] },
      allowChanges: true,
    });
    expect(text).toContain('Vault changes: On, in 1 folder');
    expect(text).toContain(FOLDER);
    expect(text).toContain('Changes appear in PDM as pdm.svc.');
    expect(document.getElementById('solidworks-review-pdm-rows')).not.toBeNull();
    expect(
      document.querySelector<HTMLElement>('#solidworks-review-pdm-folders [data-sw-folder]')
        ?.dataset.swFolder,
    ).toBe(FOLDER.toLowerCase());
  });

  it('warns when changes are on with no folders', () => {
    const text = renderReview({ allowChanges: true });
    expect(text).toContain('Vault changes: On, in 0 folders');
    expect(text).toContain(
      'Changes are on but no folders are approved, so every change will be refused.',
    );
  });

  it('uses the fallback PDM user line when no user is known', () => {
    const text = renderReview({
      link: { pdm_tool_id: 'pdm-a', approved_folders: [FOLDER] },
      allowChanges: true,
      hostStatus: null,
    });
    expect(text).toContain(
      'Changes appear in PDM under the PDM user logged in to the vault view on this machine.',
    );
  });
});
