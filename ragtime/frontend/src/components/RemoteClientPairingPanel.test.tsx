import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { RemoteClientPairingPanel } from './RemoteClientPairingPanel';
import type { RemotePairingCreatedResponse } from '@/types';

const apiMock = vi.hoisted(() => ({
  createRemotePairing: vi.fn(),
  getRemotePairing: vi.fn(),
  regenerateRemotePairing: vi.fn(),
  completeRemotePairingOffline: vi.fn(),
  deleteRemotePairing: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

const createdPairing: RemotePairingCreatedResponse = {
  id: 'pairing-1',
  tool_id: 'tool-1',
  status: 'pending',
  expires_at: new Date(Date.now() + 60_000).toISOString(),
  connection_config: { host: '', user: '', pairing_id: 'pairing-1' },
  public_key: 'ssh-ed25519 AAAA',
  commands: {
    sh: 'run-sh',
    powershell: 'run-ps',
    offline_sh: 'offline-sh',
    offline_powershell: 'offline-ps',
  },
};

function createDeferredPromise<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

afterEach(() => {
  vi.useRealTimers();
  cleanup();
  vi.restoreAllMocks();
  vi.clearAllMocks();
});

describe('RemoteClientPairingPanel', () => {
  it('creates one draft, polls it, and exposes paired facts without private credentials', async () => {
    apiMock.createRemotePairing.mockResolvedValue(createdPairing);
    apiMock.getRemotePairing.mockResolvedValue({
      ...createdPairing,
      status: 'paired',
      facts: {
        os: 'windows',
        username: 'Administrator',
        hostname: 'host-1',
        addresses: [],
        sshd_port: 22,
        host_keys: ['ssh-ed25519 AAAA'],
        shared_admin_keyfile: false,
        errors: [],
        python_available: false,
      },
      transport: 'tunnel',
    });
    const onPairingCreated = vi.fn();

    render(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId={null}
        onPairingCreated={onPairingCreated}
        onPairingUpdated={vi.fn()}
      />,
    );

    await screen.findByText('Connected — host-1');
    expect(apiMock.createRemotePairing).toHaveBeenCalledTimes(1);
    expect(onPairingCreated).toHaveBeenCalledWith(expect.objectContaining({ tool_id: 'tool-1' }));
    expect(screen.queryByText(/private key/i)).toBeNull();
  });

  it('reuses a supplied pairing id instead of creating another draft', async () => {
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'paired' });

    render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );
    await screen.findByText('Connected — target');
    await waitFor(() => expect(apiMock.createRemotePairing).not.toHaveBeenCalled());
  });

  it('binds a failed verification diagnostic once in an accessible alert', async () => {
    const diagnostic = 'SSH authentication failed: transport shut down or saw EOF';
    apiMock.getRemotePairing.mockResolvedValue({
      ...createdPairing,
      status: 'failed',
      error: diagnostic,
      facts: {
        os: 'windows',
        username: 'VISNOVSKY\\Matt',
        hostname: 'ELFYPCMINI',
        addresses: [],
        sshd_port: 22,
        host_keys: [],
        shared_admin_keyfile: false,
        errors: [],
        python_available: false,
      },
    });

    render(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    const alert = await screen.findByRole('alert');
    expect(alert.id).toBe('rc-verification-failed');
    expect(alert.dataset.rcSection).toBe('verification-failed');
    expect(alert.textContent).toContain(diagnostic);
    expect(screen.getAllByText(diagnostic)).toHaveLength(1);
    expect(document.querySelector('[data-rc-section="facts"]')?.getAttribute('aria-label')).toBe(
      'Target details reported by installer (not verified)',
    );
    expect(alert.textContent).toMatch(/Event Viewer/i);
    expect(alert.textContent).toMatch(/domain controller/i);
  });

  it('limits Windows EOF guidance to the applicable failed connection', async () => {
    const localPairing = {
      ...createdPairing,
      status: 'failed' as const,
      error: 'SSH authentication failed',
      facts: {
        os: 'windows' as const,
        username: 'ELFYPCMINI\\admin',
        hostname: 'ELFYPCMINI',
        addresses: [],
        sshd_port: 22,
        host_keys: [],
        shared_admin_keyfile: false,
        errors: [],
        python_available: false,
      },
    };
    apiMock.getRemotePairing.mockResolvedValue(localPairing);

    const view = render(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).not.toMatch(/Event Viewer/i);

    const localDiagnostic = 'SSH connection failed: transport shut down or saw EOF';
    apiMock.getRemotePairing.mockResolvedValue({ ...localPairing, error: localDiagnostic });
    view.rerender(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId="pairing-local"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );
    await screen.findByText(localDiagnostic);
    expect(screen.getByRole('alert').textContent).toMatch(/Event Viewer/i);
    expect(screen.getByRole('alert').textContent).not.toMatch(/domain controller/i);

    apiMock.getRemotePairing.mockResolvedValue({
      ...createdPairing,
      status: 'failed',
      error: 'transport shut down or saw EOF',
      facts: {
        os: 'linux',
        username: 'deploy',
        hostname: 'server-1',
        addresses: [],
        sshd_port: 22,
        host_keys: [],
        shared_admin_keyfile: false,
        errors: [],
        python_available: true,
      },
    });
    view.rerender(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-2"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    await screen.findByText('server-1');
    expect(screen.getByRole('alert').textContent).not.toMatch(/Event Viewer/i);
  });

  it('clears a failed alert and resumes polling after generating a new command', async () => {
    apiMock.getRemotePairing
      .mockResolvedValueOnce({
        ...createdPairing,
        status: 'failed',
        error: 'verification failed',
      })
      .mockResolvedValueOnce({ ...createdPairing, id: 'pairing-2', status: 'paired' });
    apiMock.regenerateRemotePairing.mockResolvedValue({
      ...createdPairing,
      id: 'pairing-2',
      expires_at: new Date(Date.now() + 60_000).toISOString(),
    });

    render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    await screen.findByRole('alert');
    fireEvent.click(screen.getByRole('button', { name: 'Generate new command' }));

    await waitFor(() => expect(apiMock.regenerateRemotePairing).toHaveBeenCalledWith('pairing-1'));
    await waitFor(() => expect(apiMock.getRemotePairing).toHaveBeenCalledWith('pairing-2'));
    await screen.findByText('Connected — target');
    expect(document.querySelector('#rc-verification-failed')).toBeNull();
  });

  it('stops polling on unmount without cancelling the persisted pairing', async () => {
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'pending' });
    const view = render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    await screen.findByText('Waiting for installer to run…');
    expect(
      document.querySelector('section[aria-label="Installation command"]')?.textContent,
    ).toContain('Waiting for installer to run…');
    view.unmount();

    expect(apiMock.deleteRemotePairing).not.toHaveBeenCalled();
  });

  it('discards a stale poll result after the pairing changes', async () => {
    const staleResponse = createDeferredPromise<typeof createdPairing>();
    apiMock.getRemotePairing.mockReturnValueOnce(staleResponse.promise).mockResolvedValueOnce({
      ...createdPairing,
      id: 'pairing-2',
      status: 'paired',
      facts: {
        os: 'linux',
        username: 'deploy',
        hostname: 'new-host',
        addresses: [],
        sshd_port: 22,
        host_keys: [],
        shared_admin_keyfile: false,
        errors: [],
        python_available: true,
      },
    });
    const onPairingUpdated = vi.fn();
    const view = render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={onPairingUpdated}
      />,
    );

    view.rerender(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-2"
        onPairingCreated={vi.fn()}
        onPairingUpdated={onPairingUpdated}
      />,
    );
    await screen.findByText('Connected — new-host');
    staleResponse.resolve({
      ...createdPairing,
      status: 'paired',
      facts: {
        os: 'linux',
        username: 'old',
        hostname: 'old-host',
        addresses: [],
        sshd_port: 22,
        host_keys: [],
        shared_admin_keyfile: false,
        errors: [],
        python_available: true,
      },
    });
    await waitFor(() => expect(screen.queryByText('Connected — old-host')).toBeNull());
  });

  it('regenerates an expired command', async () => {
    apiMock.createRemotePairing.mockResolvedValue({
      ...createdPairing,
      expires_at: new Date(Date.now() - 1_000).toISOString(),
    });
    const { commands: _commands, public_key: _publicKey, ...pollingPairing } = createdPairing;
    apiMock.getRemotePairing
      .mockResolvedValueOnce({ ...pollingPairing, status: 'expired' })
      .mockResolvedValueOnce({ ...pollingPairing, status: 'pending' });
    apiMock.regenerateRemotePairing.mockResolvedValue({
      ...createdPairing,
      id: 'pairing-2',
      expires_at: new Date(Date.now() + 60_000).toISOString(),
      commands: { ...createdPairing.commands, sh: 'fresh-sh' },
    });

    const onPairingCreated = vi.fn();
    render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId={null}
        onPairingCreated={onPairingCreated}
        onPairingUpdated={vi.fn()}
      />,
    );
    await screen.findByRole('button', { name: 'Generate new command' });
    fireEvent.click(screen.getByLabelText('Linux'));
    fireEvent.click(screen.getByRole('button', { name: 'Generate new command' }));
    await waitFor(() => expect(apiMock.regenerateRemotePairing).toHaveBeenCalledWith('pairing-1'));
    await waitFor(() => expect(apiMock.getRemotePairing).toHaveBeenCalledWith('pairing-2'));
    expect(onPairingCreated).toHaveBeenLastCalledWith(
      expect.objectContaining({ id: 'pairing-2', tool_id: 'tool-1' }),
    );
    await screen.findByText('fresh-sh');
  });

  it('allows importing offline installer output while the pairing is pending', async () => {
    apiMock.createRemotePairing.mockResolvedValue(createdPairing);
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'pending' });
    apiMock.completeRemotePairingOffline.mockResolvedValue({ ...createdPairing, status: 'paired' });

    render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId={null}
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );
    await screen.findByText('Waiting for installer to run…');
    fireEvent.click(screen.getByLabelText('Linux'));
    fireEvent.click(screen.getByLabelText('Offline'));
    expect(screen.getByText('offline-sh')).toBeTruthy();
    const installerOutput = screen.getByLabelText('Installer output') as HTMLTextAreaElement;
    expect(installerOutput.disabled).toBe(false);
    fireEvent.change(installerOutput, { target: { value: 'blob' } });
    fireEvent.click(screen.getByRole('button', { name: 'Import' }));
    await waitFor(() =>
      expect(apiMock.completeRemotePairingOffline).toHaveBeenCalledWith('pairing-1', 'blob'),
    );
  });

  const targetUserHelpName = 'Choose a Windows SSH account';

  it('offers Windows target-user help beside the command label in online and offline modes', async () => {
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'pending' });

    render(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    const trigger = await screen.findByRole('button', { name: targetUserHelpName });
    expect(trigger.id).toBe('rc-target-user-help-trigger');
    expect(trigger.getAttribute('aria-haspopup')).toBe('dialog');
    const label = screen.getByText('Run on the target machine as the account Ragtime should use');
    expect(label.parentElement?.contains(trigger)).toBe(true);
    const copyButton = screen.getByRole('button', { name: 'Copy installation command' });
    expect(label.parentElement?.contains(copyButton)).toBe(false);

    fireEvent.click(screen.getByLabelText('Offline'));
    expect(screen.getByRole('button', { name: targetUserHelpName })).toBe(trigger);
  });

  it('omits target-user help from POSIX commands', async () => {
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'pending' });

    render(
      <RemoteClientPairingPanel
        toolType="ssh_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    await screen.findByText('Waiting for installer to run…');
    expect(screen.queryByRole('button', { name: targetUserHelpName })).toBeNull();
    fireEvent.click(screen.getByLabelText('macOS'));
    expect(screen.queryByRole('button', { name: targetUserHelpName })).toBeNull();
  });

  it('shows the literal -TargetUser flag in a real dialog and closes it with Escape', async () => {
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'pending' });

    render(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    const trigger = await screen.findByRole('button', { name: targetUserHelpName });
    trigger.focus();
    fireEvent.click(trigger);

    const dialog = await screen.findByRole('alertdialog', { name: 'Choose a Windows SSH account' });
    expect(within(dialog).getByText("-TargetUser 'PCNAME\\username'").tagName).toBe('CODE');
    expect(dialog.textContent).toMatch(/elevated PowerShell/);
    await waitFor(() =>
      expect(document.activeElement).toBe(within(dialog).getByRole('button', { name: 'Got it' })),
    );

    fireEvent.keyDown(dialog, { key: 'Escape' });

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
    expect(document.activeElement).toBe(trigger);
    expect(screen.getByText('run-ps')).toBeTruthy();
    expect(screen.getByText('Waiting for installer to run…')).toBeTruthy();
  });

  it('closes target-user help from Got it without changing the command', async () => {
    apiMock.getRemotePairing.mockResolvedValue({ ...createdPairing, status: 'pending' });

    render(
      <RemoteClientPairingPanel
        toolType="windows_shell"
        pairingId="pairing-1"
        onPairingCreated={vi.fn()}
        onPairingUpdated={vi.fn()}
      />,
    );

    const trigger = await screen.findByRole('button', { name: targetUserHelpName });
    trigger.focus();
    fireEvent.click(trigger);
    const dialog = await screen.findByRole('alertdialog', { name: 'Choose a Windows SSH account' });
    await waitFor(() =>
      expect(document.activeElement).toBe(within(dialog).getByRole('button', { name: 'Got it' })),
    );
    fireEvent.click(within(dialog).getByRole('button', { name: 'Got it' }));

    await waitFor(() => expect(screen.queryByRole('alertdialog')).toBeNull());
    expect(document.activeElement).toBe(trigger);
    expect(screen.getByText('run-ps')).toBeTruthy();
  });
});
