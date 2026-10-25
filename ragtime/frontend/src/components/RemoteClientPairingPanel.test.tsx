import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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
});
