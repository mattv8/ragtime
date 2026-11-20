import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SolidWorksPdmSetupForm } from './SolidWorksPdmSetupForm';

const client = vi.hoisted(() => ({
  discoverMssqlDatabases: vi.fn(),
  discoverPdmSchema: vi.fn(),
  createToolConfig: vi.fn(),
  updateToolAccessPolicy: vi.fn(),
  triggerPdmIndex: vi.fn(),
}));
vi.mock('../api/client', () => ({ api: client }));

function fill() {
  fireEvent.change(screen.getByLabelText('SQL Server host'), { target: { value: 'sql' } });
  fireEvent.change(screen.getByLabelText('User'), { target: { value: 'user' } });
  fireEvent.change(screen.getByLabelText('Password'), { target: { value: 'secret' } });
  fireEvent.change(screen.getByLabelText('Database'), { target: { value: 'PDM' } });
}

describe('SolidWorksPdmSetupForm', () => {
  afterEach(cleanup);
  beforeEach(() => {
    vi.clearAllMocks();
    client.discoverMssqlDatabases.mockResolvedValue({ success: true, databases: ['PDM'] });
    client.discoverPdmSchema.mockResolvedValue({ success: true });
    client.createToolConfig.mockResolvedValue({ id: 'new', name: 'PDM PDM' });
    client.updateToolAccessPolicy.mockResolvedValue({});
    client.triggerPdmIndex.mockResolvedValue({});
  });

  it('requires fields, discovers databases and creates with a deny policy', async () => {
    const onCreated = vi.fn();
    render(<SolidWorksPdmSetupForm onCreated={onCreated} onCancel={vi.fn()} />);
    expect(
      (screen.getByRole('button', { name: 'Connect and start indexing' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Discover' }));
    await waitFor(() => expect(screen.getByLabelText('Database').tagName).toBe('SELECT'));
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    await waitFor(() => expect(onCreated).toHaveBeenCalled());
    expect(client.createToolConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        name: 'PDM PDM',
        tool_type: 'solidworks_pdm',
        connection_config: expect.objectContaining({ reindex_interval_hours: 24 }),
      }),
    );
    expect(client.updateToolAccessPolicy).toHaveBeenCalledWith('new', {
      tool_id: 'new',
      default_chat_access: 'deny',
      default_workspace_access: 'deny',
      users: [],
      groups: [],
    });
    expect(client.discoverPdmSchema.mock.invocationCallOrder[0]).toBeLessThan(
      client.createToolConfig.mock.invocationCallOrder[0],
    );
    expect(client.updateToolAccessPolicy.mock.invocationCallOrder[0]).toBeLessThan(
      client.triggerPdmIndex.mock.invocationCallOrder[0],
    );
  });

  it('retains fields and creates nothing if testing fails', async () => {
    client.discoverPdmSchema.mockRejectedValueOnce(new Error('offline'));
    render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} />);
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    expect((await screen.findByRole('alert')).textContent).toContain(
      'Could not connect: offline. Nothing was created.',
    );
    expect((screen.getByLabelText('Password') as HTMLInputElement).value).toBe('secret');
    expect(client.createToolConfig).not.toHaveBeenCalled();
  });

  it('reports create errors', async () => {
    client.createToolConfig.mockRejectedValueOnce(new Error('failed'));
    render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} />);
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    expect((await screen.findByRole('alert')).textContent).toContain(
      'Could not create the PDM tool: failed.',
    );
  });

  it('still returns a created tool when indexing fails', async () => {
    client.triggerPdmIndex.mockRejectedValueOnce(new Error('failed'));
    const onCreated = vi.fn();
    const onIndexError = vi.fn();
    render(
      <SolidWorksPdmSetupForm
        onCreated={onCreated}
        onCancel={vi.fn()}
        onIndexError={onIndexError}
      />,
    );
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    await waitFor(() =>
      expect(onCreated).toHaveBeenCalledWith(expect.objectContaining({ id: 'new' })),
    );
    expect(onIndexError).toHaveBeenCalledWith('Could not start indexing: failed');
  });

  it('reports a policy failure without allowing a duplicate creation', async () => {
    client.updateToolAccessPolicy.mockRejectedValueOnce(new Error('unavailable'));
    render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} />);
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Connect and start indexing' }));
    expect((await screen.findByRole('alert')).textContent).toContain('The PDM tool was created');
    expect(
      (screen.getByRole('button', { name: 'Connect and start indexing' }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(client.triggerPdmIndex).not.toHaveBeenCalled();
  });

  it('cancels on Escape and Cancel', () => {
    const cancel = vi.fn();
    render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={cancel} />);
    fireEvent.keyDown(screen.getByLabelText('SQL Server host'), { key: 'Escape' });
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(cancel).toHaveBeenCalledTimes(2);
  });

  it('marks required fields and explains invalid values after blur', () => {
    render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} />);
    for (const name of ['SQL Server host', 'Port', 'User', 'Password', 'Database']) {
      const input = screen.getByLabelText(name) as HTMLInputElement;
      expect(input.required).toBe(true);
      if (name !== 'Port') {
        fireEvent.blur(input);
        expect(input.getAttribute('aria-invalid')).toBe('true');
        const errorId = `solidworks-pdm-setup-${name === 'SQL Server host' ? 'host' : name.toLowerCase()}-error`;
        expect(input.getAttribute('aria-describedby')).toContain(errorId);
        expect(document.getElementById(errorId)?.textContent).toBeTruthy();
      }
    }
    for (const name of ['SQL Server host', 'User', 'Database']) {
      expect((screen.getByLabelText(name) as HTMLInputElement).type).toBe('text');
    }
    const port = screen.getByLabelText('Port') as HTMLInputElement;
    fireEvent.change(port, { target: { value: '2.5' } });
    fireEvent.blur(port);
    expect(port.getAttribute('aria-invalid')).toBe('true');
    expect(document.getElementById('solidworks-pdm-setup-port-error')?.textContent).toBeTruthy();
    fill();
    expect((screen.getByRole('button', { name: 'Discover' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });

  it.each(['SQL Server host', 'User', 'Password', 'Port'])(
    'invalidates discovered databases when %s changes',
    async (field) => {
      render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} />);
      fill();
      fireEvent.click(screen.getByRole('button', { name: 'Discover' }));
      await waitFor(() => expect(screen.getByLabelText('Database').tagName).toBe('SELECT'));
      fireEvent.change(screen.getByLabelText(field), {
        target: { value: field === 'Port' ? '1434' : 'changed' },
      });
      expect(screen.getByLabelText('Database').tagName).toBe('INPUT');
    },
  );

  it('keeps manual database entry and explains an empty discovery result', async () => {
    client.discoverMssqlDatabases.mockResolvedValueOnce({ success: true, databases: [] });
    render(<SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} />);
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Discover' }));
    expect(await screen.findByText(/No databases found/)).toBeTruthy();
    expect((screen.getByLabelText('Database') as HTMLInputElement).value).toBe('PDM');
  });

  it('shows discovery progress and reports busy until discovery settles', async () => {
    let resolve!: (value: { success: boolean; databases: string[] }) => void;
    client.discoverMssqlDatabases.mockImplementationOnce(
      () =>
        new Promise((done) => {
          resolve = done;
        }),
    );
    const onBusyChange = vi.fn();
    render(
      <SolidWorksPdmSetupForm onCreated={vi.fn()} onCancel={vi.fn()} onBusyChange={onBusyChange} />,
    );
    fill();
    fireEvent.click(screen.getByRole('button', { name: 'Discover' }));
    expect(screen.getByRole('button', { name: 'Discovering…' })).toBeTruthy();
    expect(onBusyChange).toHaveBeenCalledWith(true);
    resolve({ success: true, databases: [] });
    await waitFor(() => expect(onBusyChange).toHaveBeenLastCalledWith(false));
  });
});
