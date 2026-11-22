import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { useState } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { SolidworksHostStatus } from '../api/solidworks';
import type { PdmIndexJob, PdmIndexStatusResponse, ToolConfig } from '../types/api';
import { SolidWorksPdmLinkPanel, type SolidWorksPdmLinkPanelProps } from './SolidWorksPdmLinkPanel';

const client = vi.hoisted(() => ({
  listToolConfigs: vi.fn(),
  getPdmIndexStatus: vi.fn(),
  toggleToolConfig: vi.fn(),
  triggerPdmIndex: vi.fn(),
  discoverMssqlDatabases: vi.fn(),
  discoverPdmSchema: vi.fn(),
  createToolConfig: vi.fn(),
  updateToolAccessPolicy: vi.fn(),
}));
vi.mock('../api/client', () => ({ api: client }));

type PdmHost = NonNullable<SolidworksHostStatus['pdm']>;

const VIEW = { vault: 'Ragtime', root: 'C:\\PDM\\Ragtime', logged_in: true, user: 'pdm.svc' };
const READY: PdmIndexStatusResponse = {
  last_indexed: '2026-11-19T10:00:00Z',
  document_count: 12480,
  embedding_count: 11902,
  current_job: null,
};
const EMPTY_INDEX: PdmIndexStatusResponse = { ...READY, document_count: 0, last_indexed: null };
const PDM_OK: PdmHost = { client_installed: true, client_version: '2024', views: [VIEW] };
const LINKED_FOLDERS = ['\\13_Ragtime'];

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

function job(overrides: Partial<PdmIndexJob>): PdmIndexJob {
  return {
    id: 'job-1',
    tool_config_id: 'pdm-a',
    status: 'pending',
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

function section(): HTMLElement {
  return document.getElementById('solidworks-pdm-link-panel') as HTMLElement;
}

function byId(id: string): HTMLElement | null {
  return document.getElementById(id);
}

function textOf(id: string): string {
  return byId(id)?.textContent ?? '';
}

function renderPanel(overrides: Partial<SolidWorksPdmLinkPanelProps> = {}) {
  const props: SolidWorksPdmLinkPanelProps = {
    value: { pdm_tool_id: null, approved_folders: [] },
    onChange: vi.fn(),
    allowChanges: false,
    onAllowChangesChange: vi.fn(),
    hostStatus: null,
    ...overrides,
  };
  render(<SolidWorksPdmLinkPanel {...props} />);
  return props;
}

async function settled(state: string) {
  await waitFor(() => expect(section().dataset.swPdmState).toBe(state));
}

describe('SolidWorksPdmLinkPanel link states', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.clearAllMocks();
    client.listToolConfigs.mockResolvedValue([]);
    client.getPdmIndexStatus.mockResolvedValue(READY);
  });

  it.each([
    ['none_linked', [pdmTool()], null, READY, 'Not linked', ''],
    ['no_tools', [], null, READY, 'Not linked', 'No PDM tools yet. Linking one is optional.'],
    [
      'linked_missing',
      [],
      'gone',
      READY,
      'Tool not found',
      'The linked PDM tool no longer exists.',
    ],
    [
      'linked_disabled',
      [pdmTool({ enabled: false })],
      'pdm-a',
      READY,
      'Tool disabled',
      '“PDM Ragtime” is disabled, so the AI cannot search it.',
    ],
    [
      'never_indexed',
      [pdmTool()],
      'pdm-a',
      EMPTY_INDEX,
      'Not indexed',
      '“PDM Ragtime” has not been indexed',
    ],
    ['linked_ready', [pdmTool()], 'pdm-a', READY, 'Linked', 'Linked to “PDM Ragtime”.'],
  ])(
    'shows the %s state with its pill and sentence',
    async (state, tools, linked, status, pill, sentence) => {
      client.listToolConfigs.mockResolvedValue(tools);
      client.getPdmIndexStatus.mockResolvedValue(status);
      renderPanel({ value: { pdm_tool_id: linked, approved_folders: [] } });
      await settled(state);
      expect(section().querySelector('.sw-pill')?.textContent).toBe(pill);
      expect(textOf('solidworks-pdm-link-state')).toContain(sentence);
    },
  );

  it('shows the loading state while the tool list is pending', () => {
    client.listToolConfigs.mockReturnValue(new Promise(() => undefined));
    renderPanel();
    expect(section().dataset.swPdmState).toBe('loading');
    expect(section().querySelector('.sw-pill')?.textContent).toBe('Loading…');
    expect(textOf('solidworks-pdm-link-state')).toBe('Loading your PDM tools.');
  });

  it('shows load_failed with a polite sentence and a Try again action', async () => {
    client.listToolConfigs.mockRejectedValue(new Error('offline'));
    renderPanel();
    await settled('load_failed');
    expect(section().querySelector('.sw-pill')?.textContent).toBe('Load failed');
    expect(textOf('solidworks-pdm-link-state')).toBe('Could not load PDM tools: offline');
    expect(byId('solidworks-pdm-link-error')).toBeNull();
    expect(byId('solidworks-pdm-retry')?.textContent).toBe('Try again');
  });

  it('shows index failure with the error and the "outdated" pill when documents exist', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockResolvedValue({
      ...READY,
      current_job: job({ status: 'failed', error_message: 'boom' }),
    });
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled('index_failed');
    expect(section().querySelector('.sw-pill')?.textContent).toBe('Index outdated');
    expect(textOf('solidworks-pdm-link-state')).toContain('The last index failed: boom.');
    expect(textOf('solidworks-pdm-link-state')).toContain('The AI uses the index from');
  });

  it('reports indexing with a progress bar that carries the counts', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockResolvedValue({
      ...READY,
      current_job: job({ status: 'indexing', total_documents: 12480, processed_documents: 3200 }),
    });
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled('indexing');
    const bar = byId('solidworks-pdm-index-progress') as HTMLProgressElement;
    expect(bar.getAttribute('aria-valuetext')).toBe('Indexing 3,200 of 12,480 documents');
    expect(textOf('solidworks-pdm-index-progress-count')).toBe('3,200 of 12,480 documents.');
  });

  it('reports an empty index', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockResolvedValue({
      ...EMPTY_INDEX,
      last_indexed: '2026-11-19T10:00:00Z',
    });
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled('index_empty');
    expect(section().querySelector('.sw-pill')?.textContent).toBe('Index empty');
    expect(textOf('solidworks-pdm-link-state')).toContain('holds no documents');
  });

  it.each([
    ['unknown', null, 'Start the SolidWorks host above to check this machine.'],
    [
      'client_missing',
      hostWithPdm({ client_installed: false, client_version: null, views: [] }),
      'Without the PDM client the AI cannot get, open or change vault files here.',
    ],
    [
      'no_views',
      hostWithPdm({ client_installed: true, client_version: '2024', views: [] }),
      'No vault view on this machine, so the AI cannot open or change vault files.',
    ],
    [
      'vault_mismatch',
      hostWithPdm({ ...PDM_OK, views: [{ ...VIEW, vault: 'Other' }] }),
      'This machine has no vault view for “Ragtime”.',
    ],
    [
      'logged_out',
      hostWithPdm({ ...PDM_OK, views: [{ ...VIEW, logged_in: false }] }),
      'The AI can search the vault but cannot open or change files here until the vault view is logged in.',
    ],
    ['ready', hostWithPdm(PDM_OK), ''],
  ])('shows the %s machine state', async (machine, hostStatus, sentence) => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] }, hostStatus });
    await settled('linked_ready');
    expect(section().dataset.swPdmMachineState).toBe(machine);
    expect(textOf('solidworks-pdm-machine-message')).toContain(sentence);
    expect(byId('solidworks-pdm-machine-checks')?.querySelectorAll('[data-sw-check]').length).toBe(
      3,
    );
    if (machine === 'logged_out') {
      expect(section().querySelector('[data-sw-check="user"]')?.getAttribute('data-tone')).toBe(
        'warning',
      );
      expect(section().querySelector('[data-sw-check="user"]')?.textContent).toContain(
        'not logged in',
      );
    }
    if (machine === 'ready') expect(byId('solidworks-pdm-machine-message')).toBeNull();
  });
});

describe('SolidWorksPdmLinkPanel picker and link rules', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.clearAllMocks();
    client.listToolConfigs.mockResolvedValue([
      pdmTool(),
      pdmTool({ id: 'pdm-b', name: 'PDM Second' }),
    ]);
    client.getPdmIndexStatus.mockResolvedValue(READY);
  });

  it('calls onChange with the chosen tool and clears approved folders', async () => {
    const props = renderPanel({
      value: { pdm_tool_id: 'pdm-a', approved_folders: LINKED_FOLDERS },
    });
    await settled('linked_ready');
    fireEvent.change(byId('solidworks-pdm-tool') as HTMLSelectElement, {
      target: { value: 'pdm-b' },
    });
    expect(props.onChange).toHaveBeenCalledWith({ pdm_tool_id: 'pdm-b', approved_folders: [] });
    expect(textOf('solidworks-pdm-link-notice')).toBe(
      'Approved folders cleared because the linked PDM tool changed.',
    );
  });

  it('turns Allow changes off and says so when the link is cleared', async () => {
    const props = renderPanel({
      value: { pdm_tool_id: 'pdm-a', approved_folders: LINKED_FOLDERS },
      allowChanges: true,
    });
    await settled('linked_ready');
    fireEvent.change(byId('solidworks-pdm-tool') as HTMLSelectElement, { target: { value: '' } });
    expect(props.onChange).toHaveBeenCalledWith({ pdm_tool_id: null, approved_folders: [] });
    expect(props.onAllowChangesChange).toHaveBeenCalledWith(false);
    expect(textOf('solidworks-pdm-link-notice')).toBe(
      'Changes turned off because no PDM tool is linked.',
    );
  });

  it('offers the vault match as a suggestion that links the tool', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    const props = renderPanel({
      hostStatus: hostWithPdm({ ...PDM_OK, views: [{ ...VIEW, vault: 'ragtime' }] }),
    });
    await waitFor(() => expect(byId('solidworks-pdm-suggestion')).not.toBeNull());
    expect(textOf('solidworks-pdm-suggestion')).toContain('Vault view ragtime');
    fireEvent.click(byId('solidworks-pdm-suggestion-link') as HTMLButtonElement);
    expect(props.onChange).toHaveBeenCalledWith({ pdm_tool_id: 'pdm-a', approved_folders: [] });
  });

  it('shows the picker help with the who-can-search sentence', async () => {
    renderPanel();
    await settled('none_linked');
    fireEvent.click(byId('solidworks-pdm-tool-help') as HTMLButtonElement);
    expect(textOf('solidworks-pdm-tool-help-content')).toContain(
      'Everyone who can use this tool can search that vault.',
    );
  });

  it('opens the setup form and keeps it mounted when cancelled', async () => {
    client.listToolConfigs.mockResolvedValue([]);
    renderPanel();
    await settled('no_tools');
    fireEvent.click(byId('solidworks-pdm-setup-open') as HTMLButtonElement);
    const setup = byId('solidworks-pdm-setup') as HTMLElement;
    expect(setup.closest('div[hidden]')).toBeNull();
    fireEvent.click(byId('solidworks-pdm-setup-cancel') as HTMLButtonElement);
    expect(byId('solidworks-pdm-setup')).toBe(setup);
    expect(setup.closest('div[hidden]')).not.toBeNull();
    expect(document.activeElement).toBe(byId('solidworks-pdm-setup-open'));
  });

  it('focuses the newly selected tool only after the refreshed list arrives', async () => {
    const created = pdmTool({ id: 'pdm-new' });
    let resolveList!: (tools: ToolConfig[]) => void;
    client.listToolConfigs.mockResolvedValueOnce([]).mockImplementationOnce(
      () =>
        new Promise<ToolConfig[]>((resolve) => {
          resolveList = resolve;
        }),
    );
    client.discoverPdmSchema.mockResolvedValue({ success: true });
    client.createToolConfig.mockResolvedValue(created);
    client.updateToolAccessPolicy.mockResolvedValue({});
    client.triggerPdmIndex.mockResolvedValue({});
    function Harness() {
      const [value, setValue] = useState({
        pdm_tool_id: null as string | null,
        approved_folders: [] as string[],
      });
      return (
        <SolidWorksPdmLinkPanel
          value={value}
          onChange={setValue}
          allowChanges={false}
          onAllowChangesChange={vi.fn()}
          hostStatus={null}
        />
      );
    }
    render(<Harness />);
    await settled('no_tools');
    fireEvent.click(byId('solidworks-pdm-setup-open') as HTMLButtonElement);
    fillSetup();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    await waitFor(() => expect(resolveList).toBeDefined());
    expect((byId('solidworks-pdm-tool') as HTMLSelectElement).disabled).toBe(true);
    await act(async () => resolveList([created]));
    await waitFor(() => expect(document.activeElement).toBe(byId('solidworks-pdm-tool')));
    expect((byId('solidworks-pdm-tool') as HTMLSelectElement).value).toBe('pdm-new');
  });

  it('links the created tool and reports it, without the index message when indexing started', async () => {
    client.listToolConfigs.mockResolvedValue([]);
    client.discoverPdmSchema.mockResolvedValue({ success: true });
    client.createToolConfig.mockResolvedValue(pdmTool({ id: 'pdm-new', name: 'PDM Ragtime' }));
    client.updateToolAccessPolicy.mockResolvedValue({});
    client.triggerPdmIndex.mockResolvedValue({});
    const props = renderPanel();
    await settled('no_tools');
    fireEvent.click(byId('solidworks-pdm-setup-open') as HTMLButtonElement);
    fillSetup();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    await waitFor(() =>
      expect(props.onChange).toHaveBeenCalledWith({ pdm_tool_id: 'pdm-new', approved_folders: [] }),
    );
    expect(textOf('solidworks-pdm-link-notice')).toBe(
      'PDM tool “PDM Ragtime” created and linked. Indexing started.',
    );
    expect(textOf('solidworks-pdm-created-note')).toContain('stays in your Tools list');
  });

  it('keeps the tool linked and shows the start failure when indexing cannot start after creation', async () => {
    client.listToolConfigs.mockResolvedValue([]);
    client.discoverPdmSchema.mockResolvedValue({ success: true });
    client.createToolConfig.mockResolvedValue(pdmTool({ id: 'pdm-new', name: 'PDM Ragtime' }));
    client.updateToolAccessPolicy.mockResolvedValue({});
    client.triggerPdmIndex.mockRejectedValue(new Error('boom'));
    const props = renderPanel();
    await settled('no_tools');
    fireEvent.click(byId('solidworks-pdm-setup-open') as HTMLButtonElement);
    fillSetup();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    await waitFor(() => expect(props.onChange).toHaveBeenCalled());
    expect(textOf('solidworks-pdm-link-notice')).toBe('PDM tool “PDM Ragtime” created and linked.');
    expect(byId('solidworks-pdm-link-error')?.getAttribute('role')).toBe('alert');
    expect(textOf('solidworks-pdm-link-error')).toBe('Could not start indexing: boom');
  });

  it('reports busy while the setup form runs', async () => {
    client.listToolConfigs.mockResolvedValue([]);
    client.discoverPdmSchema.mockReturnValue(new Promise(() => undefined));
    const onBusyChange = vi.fn();
    renderPanel({ onBusyChange });
    await settled('no_tools');
    fireEvent.click(byId('solidworks-pdm-setup-open') as HTMLButtonElement);
    fillSetup();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    await waitFor(() => expect(onBusyChange).toHaveBeenCalledWith(true));
    expect((byId('solidworks-pdm-refresh') as HTMLButtonElement).disabled).toBe(true);
  });

  it('reports list resolution independently of index status', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockReturnValue(new Promise(() => undefined));
    const onResolutionChange = vi.fn();
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] }, onResolutionChange });
    await waitFor(() => expect(onResolutionChange).toHaveBeenLastCalledWith('ready'));
    expect(textOf('solidworks-pdm-link-state')).toBe('Loading index status.');
  });

  it('re-emits ready resolution when switching between resolved tools', async () => {
    const onResolutionChange = vi.fn();
    const props: SolidWorksPdmLinkPanelProps = {
      value: { pdm_tool_id: 'pdm-a', approved_folders: [] },
      onChange: vi.fn(),
      allowChanges: false,
      onAllowChangesChange: vi.fn(),
      hostStatus: null,
      onResolutionChange,
    };
    const { rerender } = render(<SolidWorksPdmLinkPanel {...props} />);
    await waitFor(() => expect(onResolutionChange).toHaveBeenLastCalledWith('ready'));
    onResolutionChange.mockClear();
    rerender(
      <SolidWorksPdmLinkPanel {...props} value={{ pdm_tool_id: 'pdm-b', approved_folders: [] }} />,
    );
    await waitFor(() => expect(onResolutionChange).toHaveBeenCalledTimes(1));
    expect(onResolutionChange).toHaveBeenLastCalledWith('ready');
  });

  it('reports a confirmed missing tool separately from a list failure', async () => {
    client.listToolConfigs.mockResolvedValue([]);
    const onResolutionChange = vi.fn();
    renderPanel({ value: { pdm_tool_id: 'gone', approved_folders: [] }, onResolutionChange });
    await waitFor(() => expect(onResolutionChange).toHaveBeenLastCalledWith('missing'));
    client.listToolConfigs.mockRejectedValueOnce(new Error('offline'));
    fireEvent.click(byId('solidworks-pdm-refresh') as HTMLButtonElement);
    await waitFor(() => expect(onResolutionChange).toHaveBeenLastCalledWith('failed'));
  });

  it('clears a target-specific error when the linked tool changes externally', async () => {
    client.getPdmIndexStatus.mockResolvedValue(EMPTY_INDEX);
    client.triggerPdmIndex.mockRejectedValueOnce(new Error('failed'));
    const props: SolidWorksPdmLinkPanelProps = {
      value: { pdm_tool_id: 'pdm-a', approved_folders: [] },
      onChange: vi.fn(),
      allowChanges: false,
      onAllowChangesChange: vi.fn(),
      hostStatus: null,
    };
    const { rerender } = render(<SolidWorksPdmLinkPanel {...props} />);
    await settled('never_indexed');
    fireEvent.click(byId('solidworks-pdm-index-start') as HTMLButtonElement);
    await waitFor(() => expect(textOf('solidworks-pdm-link-error')).toContain('failed'));
    rerender(
      <SolidWorksPdmLinkPanel {...props} value={{ pdm_tool_id: 'pdm-b', approved_folders: [] }} />,
    );
    await waitFor(() => expect(byId('solidworks-pdm-link-error')).toBeNull());
  });

  it('offers retry when index status fails', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockRejectedValue(new Error('offline'));
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await waitFor(() =>
      expect(textOf('solidworks-pdm-link-state')).toContain('index status: offline'),
    );
    expect(section().querySelector('.rc-polling-indicator')).toBeNull();
    client.getPdmIndexStatus.mockResolvedValue(READY);
    fireEvent.click(byId('solidworks-pdm-retry') as HTMLButtonElement);
    await settled('linked_ready');
  });

  it('reports the resolved target to the wizard', async () => {
    const onTargetChange = vi.fn();
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] }, onTargetChange });
    await settled('linked_ready');
    await waitFor(() =>
      expect(onTargetChange).toHaveBeenLastCalledWith({
        tool: expect.objectContaining({ id: 'pdm-a' }),
        status: READY,
      }),
    );
  });

  it('reports no target when nothing is linked', async () => {
    const onTargetChange = vi.fn();
    renderPanel({ onTargetChange });
    await settled('none_linked');
    expect(onTargetChange).toHaveBeenLastCalledWith(null);
  });
});

describe('SolidWorksPdmLinkPanel actions', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.clearAllMocks();
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockResolvedValue({ ...EMPTY_INDEX });
  });

  it.each(['index', 'enable'])('prevents setup while %s is pending', async (action) => {
    client.listToolConfigs.mockResolvedValue([pdmTool({ enabled: action !== 'enable' })]);
    const pendingRequest = new Promise(() => undefined);
    client.triggerPdmIndex.mockReturnValue(pendingRequest);
    client.toggleToolConfig.mockReturnValue(pendingRequest);
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled(action === 'index' ? 'never_indexed' : 'linked_disabled');
    fireEvent.click(
      byId(
        action === 'index' ? 'solidworks-pdm-index-start' : 'solidworks-pdm-enable-tool',
      ) as HTMLButtonElement,
    );
    const trigger = byId('solidworks-pdm-setup-open') as HTMLButtonElement;
    expect(trigger.disabled).toBe(true);
    expect(byId('solidworks-pdm-setup')?.querySelector('fieldset')?.disabled).toBe(true);
    fireEvent.click(trigger);
    expect(byId('solidworks-pdm-setup')?.closest('[hidden]')).not.toBeNull();
  });

  it('starts the first index and reports a failure as an alert', async () => {
    client.triggerPdmIndex.mockRejectedValueOnce(new Error('boom'));
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled('never_indexed');
    fireEvent.click(byId('solidworks-pdm-index-start') as HTMLButtonElement);
    expect(client.triggerPdmIndex).toHaveBeenCalledWith('pdm-a');
    await waitFor(() =>
      expect(textOf('solidworks-pdm-link-error')).toBe('Could not start indexing: boom'),
    );
    expect(byId('solidworks-pdm-link-error')?.getAttribute('role')).toBe('alert');
  });

  it('enables a disabled tool through the toggle', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool({ enabled: false })]);
    client.toggleToolConfig.mockRejectedValueOnce(new Error('nope'));
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled('linked_disabled');
    fireEvent.click(byId('solidworks-pdm-enable-tool') as HTMLButtonElement);
    expect(client.toggleToolConfig).toHaveBeenCalledWith('pdm-a', true);
    await waitFor(() =>
      expect(textOf('solidworks-pdm-link-error')).toBe('Could not enable “PDM Ragtime”: nope.'),
    );
  });

  it('refreshes the tool list and reports a failed refresh as an alert', async () => {
    client.getPdmIndexStatus.mockResolvedValue(READY);
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] } });
    await settled('linked_ready');
    client.listToolConfigs.mockRejectedValueOnce(new Error('offline'));
    fireEvent.click(byId('solidworks-pdm-refresh') as HTMLButtonElement);
    expect(client.listToolConfigs).toHaveBeenCalledTimes(2);
    await settled('load_failed');
    expect(byId('solidworks-pdm-link-error')?.getAttribute('role')).toBe('alert');
    expect(textOf('solidworks-pdm-link-error')).toBe('Could not load PDM tools: offline');
  });
});

describe('SolidWorksPdmLinkPanel Changes block', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.clearAllMocks();
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    client.getPdmIndexStatus.mockResolvedValue(READY);
  });

  it('is not rendered without a link', async () => {
    renderPanel();
    await settled('none_linked');
    expect(byId('solidworks-pdm-changes')).toBeNull();
  });

  it('toggles Allow changes through the checkbox', async () => {
    const props = renderPanel({
      value: { pdm_tool_id: 'pdm-a', approved_folders: LINKED_FOLDERS },
    });
    await settled('linked_ready');
    fireEvent.click(byId('solidworks-allow-changes') as HTMLInputElement);
    expect(props.onAllowChangesChange).toHaveBeenCalledWith(true);
  });

  it('shows the summary and no boxed warning when changes are on', async () => {
    renderPanel({
      value: { pdm_tool_id: 'pdm-a', approved_folders: LINKED_FOLDERS },
      allowChanges: true,
      hostStatus: hostWithPdm(PDM_OK),
    });
    await settled('linked_ready');
    expect(byId('solidworks-allow-changes-warning')).toBeNull();
    expect(textOf('solidworks-allow-changes-summary')).toBe(
      'Add, check out and check in, approved folders only',
    );
    expect(byId('solidworks-pdm-index-facts')).toBeNull();
  });

  it('warns when changes are on but no folders are approved', async () => {
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] }, allowChanges: true });
    await settled('linked_ready');
    expect(textOf('solidworks-pdm-folders-warning')).toBe(
      'Changes are on, but no folders are approved. The AI cannot change anything in the vault.',
    );
  });

  it('shows the PDM user in the identity text when the vault view reports one', async () => {
    renderPanel({
      value: { pdm_tool_id: 'pdm-a', approved_folders: [] },
      allowChanges: true,
      hostStatus: hostWithPdm(PDM_OK),
    });
    await settled('linked_ready');
    expect(textOf('solidworks-pdm-identity')).toContain('Changes appear in PDM as pdm.svc');
    fireEvent.click(byId('solidworks-pdm-identity-help') as HTMLButtonElement);
    expect(textOf('solidworks-pdm-identity-help-content')).toContain(
      'Ragtime adds the tool name and the Ragtime user',
    );
  });

  it('shows the fallback identity text when no PDM user is known', async () => {
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] }, allowChanges: true });
    await settled('linked_ready');
    expect(textOf('solidworks-pdm-identity')).toContain(
      'Changes appear in PDM as the vault view user.',
    );
    fireEvent.click(byId('solidworks-pdm-identity-help') as HTMLButtonElement);
    expect(textOf('solidworks-pdm-identity-help-content')).toContain(
      'PDM records changes under the PDM user logged in to the vault view on this machine.',
    );
  });
});

describe('SolidWorksPdmLinkPanel setup helpers', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('reloads the tool list on window focus while no tool exists', async () => {
    client.listToolConfigs.mockResolvedValue([]);
    renderPanel();
    await settled('no_tools');
    window.dispatchEvent(new Event('focus'));
    await waitFor(() => expect(client.listToolConfigs).toHaveBeenCalledTimes(2));
  });

  it('disables the whole panel when disabled', async () => {
    client.listToolConfigs.mockResolvedValue([pdmTool()]);
    renderPanel({ value: { pdm_tool_id: 'pdm-a', approved_folders: [] }, disabled: true });
    await settled('linked_ready');
    expect((byId('solidworks-pdm-refresh') as HTMLButtonElement).disabled).toBe(true);
    expect((byId('solidworks-pdm-tool') as HTMLSelectElement).disabled).toBe(true);
  });
});

function fillSetup() {
  fireEvent.change(screen.getByLabelText('SQL Server host'), { target: { value: 'sql' } });
  fireEvent.change(screen.getByLabelText('User'), { target: { value: 'user' } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
  fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'Ragtime' } });
}
