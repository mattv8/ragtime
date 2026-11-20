import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  solidworksApi,
  SolidworksApiError,
  type SolidworksHostArtifact,
  type SolidworksHostStatus,
} from '../api/solidworks';
import { deferred } from '../testHelpers/deferred';
import { SolidWorksHostPanel } from './SolidWorksHostPanel';

vi.mock('../api/solidworks', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/solidworks')>();
  return {
    ...actual,
    solidworksApi: {
      getHostArtifact: vi.fn(),
      getHostStatus: vi.fn(),
      installHost: vi.fn(),
      uninstallHost: vi.fn(),
      hostDownloadUrl: vi.fn(() => '/indexes/solidworks/host/download'),
    },
  };
});

const ARTIFACT_OK: SolidworksHostArtifact = {
  available: true,
  host_version: '0.1.0',
  protocol: 1,
  sha256: 'a'.repeat(64),
  size_bytes: 2516582,
};
const ARTIFACT_MISSING: SolidworksHostArtifact = {
  available: false,
  host_version: null,
  protocol: null,
  sha256: null,
  size_bytes: null,
};
const BASE: SolidworksHostStatus = {
  reachable: true,
  installed: true,
  running: true,
  host_version: '0.1.0',
  expected_host_version: '0.1.0',
  update_available: false,
  account: { name: 'HAMMERTON\\cad-agent', is_administrator: false },
  session: { interactive: true },
  solidworks: { installed_versions: [2024, 2022], instances: [] },
  error: null,
};
const ST_RUNNING = BASE;
const ST_RUNNING_ADMIN: SolidworksHostStatus = {
  ...BASE,
  account: { name: 'HAMMERTON\\mvisnovsky.admin', is_administrator: true },
};
const ST_UPDATE: SolidworksHostStatus = {
  ...BASE,
  expected_host_version: '0.2.0',
  update_available: true,
};
const ST_NOT_INSTALLED: SolidworksHostStatus = {
  ...BASE,
  installed: false,
  running: false,
  host_version: null,
  solidworks: { installed_versions: [2024], instances: [] },
};
const ST_NO_SESSION: SolidworksHostStatus = {
  ...BASE,
  running: false,
  session: { interactive: false },
  error: { code: 'no_interactive_session', message: 'No interactive session.' },
};
const ST_NOT_RESPONDING: SolidworksHostStatus = {
  ...BASE,
  running: false,
  error: { code: 'host_not_running', message: 'The host did not answer within 20 seconds.' },
};
const ST_UNREACHABLE: SolidworksHostStatus = {
  reachable: false,
  installed: false,
  running: false,
  host_version: null,
  expected_host_version: null,
  update_available: false,
  account: null,
  session: null,
  solidworks: null,
  error: { code: 'ssh_unreachable', message: 'Connection timed out.' },
};
const ST_INSTALL_FAILED: SolidworksHostStatus = {
  ...ST_NOT_INSTALLED,
  error: { code: 'hash_mismatch', message: 'The uploaded package did not match its SHA-256 hash.' },
};

const getStatus = vi.mocked(solidworksApi.getHostStatus);
const getArtifact = vi.mocked(solidworksApi.getHostArtifact);
const install = vi.mocked(solidworksApi.installHost);
const uninstall = vi.mocked(solidworksApi.uninstallHost);

const panelState = () =>
  document.getElementById('solidworks-host-panel')?.getAttribute('data-sw-host-state');
const pillText = () => document.querySelector('#solidworks-host-panel .sw-pill')?.textContent;
const liveText = () => document.getElementById('solidworks-host-state')?.textContent;
const actionButtons = () =>
  [...document.querySelectorAll<HTMLButtonElement>('#solidworks-host-actions button')].map(
    (button) => button,
  );
const factKeys = () =>
  [...document.querySelectorAll('#solidworks-host-facts [data-sw-fact]')].map((row) =>
    row.getAttribute('data-sw-fact'),
  );
const flush = () => act(async () => undefined);
const advance = (ms: number) =>
  act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });

function renderPanel(props: Partial<Parameters<typeof SolidWorksHostPanel>[0]> = {}) {
  return render(<SolidWorksHostPanel toolId="tool-1" connectionReady {...props} />);
}

beforeEach(() => {
  getArtifact.mockResolvedValue(ARTIFACT_OK);
  getStatus.mockResolvedValue(ST_RUNNING);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe('SolidWorksHostPanel states', () => {
  it('blocked without a saved connection: no request, no buttons, onStatusChange(null)', () => {
    const onStatusChange = vi.fn();
    renderPanel({ toolId: null, connectionReady: false, onStatusChange });
    expect(panelState()).toBe('blocked');
    expect(pillText()).toBe('Waiting for connection');
    expect(liveText()).toContain('Finish the connection in the previous step first.');
    expect(actionButtons()).toHaveLength(0);
    expect(document.getElementById('solidworks-host-refresh')).toBeNull();
    expect(getStatus).not.toHaveBeenCalled();
    expect(getArtifact).not.toHaveBeenCalled();
    expect(onStatusChange).toHaveBeenCalledWith(null);
  });

  it('checking on first load: busy, Check again disabled, last facts absent', async () => {
    const pending = deferred<SolidworksHostStatus>();
    getStatus.mockReturnValueOnce(pending.promise);
    renderPanel();
    expect(panelState()).toBe('checking');
    expect(pillText()).toBe('Checking…');
    expect(liveText()).toContain('This can take up to 30 seconds.');
    expect(document.getElementById('solidworks-host-panel')?.getAttribute('aria-busy')).toBe(
      'true',
    );
    expect((document.getElementById('solidworks-host-refresh') as HTMLButtonElement).disabled).toBe(
      true,
    );
    await act(async () => pending.resolve(ST_RUNNING));
    expect(panelState()).toBe('running_current');
  });

  it('running and current: pill, sentence, all five facts, uninstall on the right, no update pill', async () => {
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(pillText()).toBe('Running');
    expect(liveText()).toBe('The host is installed and answering.');
    expect(document.getElementById('solidworks-host-state')?.className).toBe('sw-visually-hidden');
    expect(factKeys()).toEqual(['account', 'session', 'solidworks', 'starts', 'host-version']);
    expect(
      document.querySelector('#solidworks-host-facts [data-sw-fact="solidworks"] dd')?.textContent,
    ).toBe('2024, 2022');
    expect(actionButtons()).toHaveLength(0);
    expect(screen.getByRole('button', { name: 'Uninstall host' })).toBeTruthy();
    expect(document.querySelector('header #solidworks-host-uninstall')).not.toBeNull();
    expect(document.querySelectorAll('.sw-pill')).toHaveLength(1);
    fireEvent.click(screen.getByRole('button', { name: 'About session' }));
    expect(document.getElementById('solidworks-session-help-content')?.textContent).toBe(
      'The tool works only while this account is logged in on the machine. A disconnected remote desktop session counts as logged in.',
    );
  });

  it('running with no SolidWorks found shows the warning message', async () => {
    getStatus.mockResolvedValue({
      ...ST_RUNNING,
      solidworks: { installed_versions: [], instances: [] },
    });
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(
      screen.getByText(
        'The host is running but found no SolidWorks installation for this account.',
      ),
    ).toBeTruthy();
    expect(
      document.querySelector('#solidworks-host-facts [data-sw-fact="solidworks"] dd')?.textContent,
    ).toBe('None found');
  });

  it('update available: Update host is primary, pills show both labels, uninstall stays', async () => {
    getStatus.mockResolvedValue(ST_UPDATE);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_update'));
    expect(pillText()).toBe('Running');
    expect(document.querySelectorAll('.sw-pill')[1]?.textContent).toBe('Update available');
    expect(liveText()).toBe('Version 0.1.0 is running; version 0.2.0 is available.');
    fireEvent.click(screen.getByRole('button', { name: 'About updating' }));
    expect(document.getElementById('solidworks-host-update-help-content')?.textContent).toBe(
      'Updating restarts the host and closes any SolidWorks it started. Calls in progress fail.',
    );
    expect(actionButtons().map((b) => b.getAttribute('data-sw-action'))).toEqual(['update']);
    expect(document.getElementById('solidworks-host-update')?.textContent).toBe('Update host');
  });

  it('not installed: Install host, facts without host version, collapsed manual checklist', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    expect(pillText()).toBe('Not installed');
    expect(liveText()).toContain('uploads the package (2.4\u00a0MB), verifies its SHA-256 hash');
    expect(factKeys()).toEqual(['account', 'session', 'solidworks']);
    expect(actionButtons().map((b) => b.getAttribute('data-sw-action'))).toEqual(['install']);
    const manual = document.getElementById('solidworks-manual-install') as HTMLDetailsElement;
    expect(manual.open).toBe(false);
    expect(document.getElementById('solidworks-session-help')).not.toBeNull();
  });

  it('artifact unavailable: installer missing, no install button and no manual checklist', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    getArtifact.mockResolvedValue(ARTIFACT_MISSING);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('artifact_unavailable'));
    expect(pillText()).toBe('Installer missing');
    expect(liveText()).toContain('does not include the SolidWorks host installer');
    expect(actionButtons()).toHaveLength(0);
    expect(document.getElementById('solidworks-manual-install')).toBeNull();
    expect(factKeys()).toEqual(['account', 'session', 'solidworks']);
  });

  it('unreachable: danger pill, error message and code, three checks, no facts', async () => {
    getStatus.mockResolvedValue(ST_UNREACHABLE);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('unreachable'));
    expect(pillText()).toBe('Unreachable');
    expect(liveText()).toBe('Ragtime could not reach this machine over SSH.');
    expect(screen.getByText('Connection timed out.')).toBeTruthy();
    expect(screen.getByText('ssh_unreachable')).toBeTruthy();
    expect(document.querySelectorAll('#solidworks-host-panel ul li')).toHaveLength(3);
    expect(document.getElementById('solidworks-host-facts')).toBeNull();
    expect(actionButtons()).toHaveLength(0);
  });

  it('no desktop session: warning pill with polling indicator and Check again only', async () => {
    getStatus.mockResolvedValue(ST_NO_SESSION);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('no_desktop_session'));
    expect(pillText()).toBe('Waiting for desktop session');
    expect(liveText()).toContain('has no desktop session on this machine');
    expect(document.querySelector('#solidworks-host-panel .rc-polling-indicator')).not.toBeNull();
    expect(factKeys()).toEqual(['account', 'session', 'solidworks', 'host-version']);
    expect(actionButtons()).toHaveLength(0);
  });

  it('not responding: Reinstall host primary, host log copy, uninstall hidden', async () => {
    getStatus.mockResolvedValue(ST_NOT_RESPONDING);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_responding'));
    expect(pillText()).toBe('Not running');
    expect(liveText()).toContain('did not answer. Reinstalling registers its logon task again');
    expect(liveText()).toContain('logs\\host.log');
    expect(screen.getByText('The host did not answer within 20 seconds.')).toBeTruthy();
    expect(actionButtons().map((b) => b.getAttribute('data-sw-action'))).toEqual(['reinstall']);
    expect(document.getElementById('solidworks-manual-install')).not.toBeNull();
  });

  it('a failed first check is announced politely, not as an alert', async () => {
    getStatus.mockRejectedValue(new SolidworksApiError('Internal error', 500));
    const onStatusChange = vi.fn();
    renderPanel({ onStatusChange });
    await waitFor(() => expect(panelState()).toBe('request_failed'));
    expect(pillText()).toBe('Check failed');
    expect(liveText()).toBe('Ragtime could not check the host.');
    expect(document.getElementById('solidworks-host-panel')?.textContent).toContain(
      'Internal error',
    );
    expect(screen.queryByRole('alert')).toBeNull();
    expect(actionButtons()).toHaveLength(0);
    expect(onStatusChange).not.toHaveBeenCalledWith(expect.anything());
  });

  it('a failed user check shows the alert and clears on the next successful check', async () => {
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    getStatus.mockRejectedValueOnce(new SolidworksApiError('Gateway timeout', 504));
    fireEvent.click(document.getElementById('solidworks-host-refresh') as HTMLButtonElement);
    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('Gateway timeout'));
    expect(screen.getByRole('alert').textContent).toContain('Check failed');
    fireEvent.click(document.getElementById('solidworks-host-refresh') as HTMLButtonElement);
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(screen.queryByRole('alert')).toBeNull();
  });
});

describe('SolidWorksHostPanel actions', () => {
  it('install success: pressed button keeps its label, others disabled, response becomes the status', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    const pending = deferred<SolidworksHostStatus>();
    install.mockReturnValue(pending.promise);
    const onStatusChange = vi.fn();
    renderPanel({ onStatusChange });
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    fireEvent.click(screen.getByRole('button', { name: 'Install host' }));
    expect(panelState()).toBe('installing');
    expect(pillText()).toBe('Installing…');
    expect(liveText()).toContain('can take about a minute');
    const pressed = document.getElementById('solidworks-host-install') as HTMLButtonElement;
    expect(pressed.textContent).toBe('Installing…');
    expect(pressed.disabled).toBe(true);
    expect((document.getElementById('solidworks-host-refresh') as HTMLButtonElement).disabled).toBe(
      true,
    );
    await act(async () => pending.resolve(ST_RUNNING));
    expect(panelState()).toBe('running_current');
    expect(getStatus).toHaveBeenCalledTimes(1);
    expect(onStatusChange).toHaveBeenLastCalledWith(ST_RUNNING);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('install failure: alert with code, hint, checklist open, Try again re-runs install', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    install.mockResolvedValue(ST_INSTALL_FAILED);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    fireEvent.click(screen.getByRole('button', { name: 'Install host' }));
    await waitFor(() => expect(panelState()).toBe('install_failed'));
    const alert = screen.getByRole('alert');
    expect(alert.querySelector('strong')?.textContent).toBe('Details');
    expect(liveText()).toBe('');
    expect(pillText()).toBe('Install failed');
    expect(alert.textContent).toContain('The uploaded package did not match its SHA-256 hash.');
    expect(alert.textContent).toContain('hash_mismatch');
    expect(alert.textContent).toContain('compare the hash yourself');
    expect(
      document.getElementById('solidworks-host-panel')?.getAttribute('data-sw-host-action-error'),
    ).toBe('install');
    expect((document.getElementById('solidworks-manual-install') as HTMLDetailsElement).open).toBe(
      true,
    );
    expect(actionButtons().map((b) => b.getAttribute('data-sw-action'))).toEqual(['retry']);
    install.mockResolvedValue(ST_RUNNING);
    fireEvent.click(document.getElementById('solidworks-host-retry') as HTMLButtonElement);
    expect(
      (document.getElementById('solidworks-host-retry') as HTMLButtonElement).textContent,
    ).toBe('Installing…');
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(install).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('artifact 503 on install shows the install error and drops the checklist', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    install.mockRejectedValue(
      new SolidworksApiError('The SolidWorks host package is not available on this server.', 503),
    );
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    fireEvent.click(screen.getByRole('button', { name: 'Install host' }));
    await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy());
    expect(screen.getByRole('alert').textContent).toContain('not available on this server');
    expect(panelState()).toBe('install_failed');
    expect(document.getElementById('solidworks-manual-install')).toBeNull();
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it('uninstall confirm: focus on Cancel, Escape closes and returns focus, then confirm uninstalls', async () => {
    const onStatusChange = vi.fn();
    renderPanel({ onStatusChange });
    await waitFor(() => expect(panelState()).toBe('running_current'));
    uninstall.mockResolvedValue(ST_NOT_INSTALLED);
    fireEvent.click(screen.getByRole('button', { name: 'Uninstall host' }));
    const confirm = document.getElementById('solidworks-uninstall-confirm') as HTMLElement;
    expect(confirm.getAttribute('role')).toBe('group');
    expect(document.activeElement).toBe(document.getElementById('solidworks-uninstall-cancel'));
    expect(document.getElementById('solidworks-host-actions')).toBeNull();
    expect(confirm.textContent).toContain(
      'Your SolidWorks installation and documents are not touched.',
    );
    fireEvent.keyDown(confirm, { key: 'Escape' });
    expect(document.getElementById('solidworks-uninstall-confirm')).toBeNull();
    expect(document.activeElement).toBe(document.getElementById('solidworks-host-uninstall'));

    fireEvent.click(screen.getByRole('button', { name: 'Uninstall host' }));
    fireEvent.click(document.getElementById('solidworks-uninstall-cancel') as HTMLButtonElement);
    expect(document.getElementById('solidworks-uninstall-confirm')).toBeNull();
    expect(document.activeElement).toBe(document.getElementById('solidworks-host-uninstall'));

    fireEvent.click(screen.getByRole('button', { name: 'Uninstall host' }));
    fireEvent.click(
      document.getElementById('solidworks-uninstall-confirm-button') as HTMLButtonElement,
    );
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    expect(liveText()).toBe('Host uninstalled.');
    expect(uninstall).toHaveBeenCalledWith('tool-1');
    expect(onStatusChange).toHaveBeenLastCalledWith(ST_NOT_INSTALLED);
  });

  it('uninstalling keeps the pressed button labelled and disables the rest', async () => {
    const pending = deferred<SolidworksHostStatus>();
    uninstall.mockReturnValue(pending.promise);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    fireEvent.click(screen.getByRole('button', { name: 'Uninstall host' }));
    fireEvent.click(
      document.getElementById('solidworks-uninstall-confirm-button') as HTMLButtonElement,
    );
    expect(panelState()).toBe('uninstalling');
    expect(liveText()).toBe('Removing the SolidWorks host…');
    const pressed = document.getElementById('solidworks-host-uninstall') as HTMLButtonElement;
    expect(pressed.textContent).toBe('Uninstalling…');
    expect(pressed.disabled).toBe(true);
    await act(async () => pending.resolve(ST_NOT_INSTALLED));
    expect(panelState()).toBe('not_installed');
  });

  it('uninstall failure shows the uninstall alert and Try again', async () => {
    uninstall.mockResolvedValue({
      ...ST_RUNNING,
      error: { code: 'uninstall_failed', message: 'The uninstall script returned an error.' },
    });
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    fireEvent.click(screen.getByRole('button', { name: 'Uninstall host' }));
    fireEvent.click(
      document.getElementById('solidworks-uninstall-confirm-button') as HTMLButtonElement,
    );
    await waitFor(() => expect(panelState()).toBe('uninstall_failed'));
    expect(screen.getByRole('alert').textContent).toContain('Uninstall failed');
    expect(
      document.getElementById('solidworks-host-panel')?.getAttribute('data-sw-host-action-error'),
    ).toBe('uninstall');
    expect(actionButtons().map((b) => b.getAttribute('data-sw-action'))).toEqual(['retry']);
  });

  it('shows Update host as pressed while the update runs', async () => {
    getStatus.mockResolvedValue(ST_UPDATE);
    const pending = deferred<SolidworksHostStatus>();
    install.mockReturnValue(pending.promise);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_update'));
    fireEvent.click(screen.getByRole('button', { name: 'Update host' }));
    expect(liveText()).toContain('Updating also closes SolidWorks instances the host started.');
    expect(document.getElementById('solidworks-host-update')?.textContent).toBe('Updating…');
    await act(async () => pending.resolve(ST_RUNNING));
    expect(panelState()).toBe('running_current');
  });
});

describe('SolidWorksHostPanel account warning and manual checklist', () => {
  it('shows the administrator warning only when is_administrator is true', async () => {
    getStatus.mockResolvedValue(ST_RUNNING_ADMIN);
    const { unmount } = renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    const warning = document.getElementById('solidworks-account-warning');
    expect(warning?.getAttribute('role')).toBe('note');
    expect(warning?.textContent).toContain('Runs as an administrator account.');
    fireEvent.click(screen.getByRole('button', { name: 'About the administrator account' }));
    expect(
      document.getElementById('solidworks-account-warning-help-content')?.textContent,
    ).toContain('HAMMERTON\\mvisnovsky.admin');
    unmount();
    getStatus.mockResolvedValue(ST_RUNNING);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(document.getElementById('solidworks-account-warning')).toBeNull();
  });

  it.each([
    ['not_installed', ST_NOT_INSTALLED, true, false],
    ['running_update', ST_UPDATE, true, false],
    ['not_responding', ST_NOT_RESPONDING, true, false],
    ['running_current', ST_RUNNING, false, false],
    ['no_desktop_session', ST_NO_SESSION, false, false],
  ] as const)('manual checklist visibility for %s', async (state, status, visible, open) => {
    getStatus.mockResolvedValue(status);
    const { unmount } = renderPanel();
    await waitFor(() => expect(panelState()).toBe(state));
    const manual = document.getElementById(
      'solidworks-manual-install',
    ) as HTMLDetailsElement | null;
    expect(manual !== null).toBe(visible);
    if (manual) expect(manual.open).toBe(open);
    unmount();
  });

  it('renders the five manual steps with the copy and hooks from the spec', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    const steps = [
      ...document.querySelectorAll('#solidworks-manual-steps [data-sw-manual-step]'),
    ].map((li) => li.getAttribute('data-sw-manual-step'));
    expect(steps).toEqual(['download', 'verify', 'unzip', 'install', 'recheck']);
    expect(
      document.getElementById('solidworks-host-download')?.getAttribute('download'),
    ).not.toBeNull();
    expect(document.getElementById('solidworks-host-sha256')?.textContent).toBe('a'.repeat(64));
    expect(document.getElementById('solidworks-host-sha256-copy')?.getAttribute('aria-label')).toBe(
      'Copy SHA-256',
    );
    expect(document.body.textContent).toContain(
      'Get-FileHash .\\ragtime-solidworks-host-win-x64.zip -Algorithm SHA256',
    );
    expect(document.body.textContent).toContain('ragtime-solidworks-host.exe" install');
    expect(document.getElementById('solidworks-manual-copy-verify')).not.toBeNull();
    expect(document.getElementById('solidworks-manual-copy-unzip')).not.toBeNull();
    expect(document.getElementById('solidworks-manual-copy-install')).not.toBeNull();
  });
});

describe('SolidWorksHostPanel polling and refresh', () => {
  it('polls every 10 seconds in no_desktop_session, at most 30 times, then says it stopped', async () => {
    vi.useFakeTimers();
    getStatus.mockResolvedValue(ST_NO_SESSION);
    renderPanel();
    await flush();
    expect(getStatus).toHaveBeenCalledTimes(1);
    await advance(10_000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    for (let i = 0; i < 29; i += 1) await advance(10_000);
    expect(getStatus).toHaveBeenCalledTimes(31);
    await advance(60_000);
    expect(getStatus).toHaveBeenCalledTimes(31);
    expect(liveText()).toContain('Ragtime stopped checking automatically.');
    expect(document.querySelector('#solidworks-host-panel .rc-polling-indicator')).toBeNull();
  });

  it('polls in not_responding too', async () => {
    vi.useFakeTimers();
    getStatus.mockResolvedValue(ST_NOT_RESPONDING);
    renderPanel();
    await flush();
    await advance(10_000);
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it('never polls in other states', async () => {
    vi.useFakeTimers();
    getStatus.mockResolvedValue(ST_RUNNING);
    renderPanel();
    await flush();
    await advance(120_000);
    expect(getStatus).toHaveBeenCalledTimes(1);
  });

  it('pauses while the document is hidden and stops on unmount', async () => {
    vi.useFakeTimers();
    getStatus.mockResolvedValue(ST_NO_SESSION);
    const { unmount } = renderPanel();
    await flush();
    Object.defineProperty(document, 'hidden', { configurable: true, value: true });
    try {
      await advance(30_000);
      expect(getStatus).toHaveBeenCalledTimes(1);
    } finally {
      Object.defineProperty(document, 'hidden', { configurable: true, value: false });
    }
    await advance(10_000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    unmount();
    await advance(60_000);
    expect(getStatus).toHaveBeenCalledTimes(2);
  });

  it('a poll does not report busy to the wizard', async () => {
    vi.useFakeTimers();
    getStatus.mockResolvedValue(ST_NO_SESSION);
    const onBusyChange = vi.fn();
    renderPanel({ onBusyChange });
    await flush();
    const before = onBusyChange.mock.calls.map((call) => call[0]);
    await advance(10_000);
    expect(getStatus).toHaveBeenCalledTimes(2);
    expect(onBusyChange.mock.calls.map((call) => call[0])).toEqual(before);
    expect(before[before.length - 1]).toBe(false);
  });

  it('drops a stale response after the tool changes', async () => {
    const first = deferred<SolidworksHostStatus>();
    getStatus.mockReturnValueOnce(first.promise).mockResolvedValueOnce(ST_RUNNING);
    const onStatusChange = vi.fn();
    const { rerender } = render(
      <SolidWorksHostPanel toolId="tool-a" connectionReady onStatusChange={onStatusChange} />,
    );
    rerender(
      <SolidWorksHostPanel toolId="tool-b" connectionReady onStatusChange={onStatusChange} />,
    );
    await waitFor(() => expect(panelState()).toBe('running_current'));
    await act(async () => first.resolve(ST_NOT_INSTALLED));
    expect(panelState()).toBe('running_current');
    expect(onStatusChange).not.toHaveBeenCalledWith(ST_NOT_INSTALLED);
  });

  it('reports only settled statuses, null only when blocked', async () => {
    const onStatusChange = vi.fn();
    renderPanel({ onStatusChange });
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(onStatusChange).toHaveBeenCalledWith(ST_RUNNING);
    expect(onStatusChange).not.toHaveBeenCalledWith(null);
  });

  it('reports busy for the check and for actions, in order', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    install.mockResolvedValue(ST_RUNNING);
    const onBusyChange = vi.fn();
    renderPanel({ onBusyChange });
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    fireEvent.click(screen.getByRole('button', { name: 'Install host' }));
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(onBusyChange.mock.calls.map((call) => call[0])).toEqual([
      false,
      true,
      false,
      true,
      false,
    ]);
  });
});

describe('SolidWorksHostPanel presentation, warnings and busy signals', () => {
  it('uses its own status pill with a tone attribute and not the shared pill class', async () => {
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    expect(
      document.querySelector('#solidworks-host-panel .sw-pill')?.getAttribute('data-tone'),
    ).toBe('success');
    expect(document.querySelector('#solidworks-host-panel .userspace-status-pill')).toBeNull();
  });

  it('boxes the administrator warning as a warning callout', async () => {
    getStatus.mockResolvedValue(ST_RUNNING_ADMIN);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    const warning = document.getElementById('solidworks-account-warning');
    expect(warning?.getAttribute('data-boxed')).toBe('true');
    expect(warning?.getAttribute('data-tone')).toBe('warning');
  });

  it('shows a status warning under the state sentence in the resolved state', async () => {
    getStatus.mockResolvedValue({ ...ST_RUNNING, warning: 'Some host files are still locked.' });
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    const note = document.getElementById('solidworks-host-warning');
    expect(note?.getAttribute('data-tone')).toBe('warning');
    expect(note?.textContent).toContain('Some host files are still locked.');
  });

  it('an uninstall with leftover files resolves to not installed, and Check again uses the normal rules', async () => {
    const leftover: SolidworksHostStatus = {
      ...ST_NOT_RESPONDING,
      error: null,
      warning:
        'The host was stopped and its logon task removed, but some files could not be deleted.',
    };
    uninstall.mockResolvedValue(leftover);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('running_current'));
    fireEvent.click(screen.getByRole('button', { name: 'Uninstall host' }));
    fireEvent.click(
      document.getElementById('solidworks-uninstall-confirm-button') as HTMLButtonElement,
    );
    await waitFor(() => expect(uninstall).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    expect(pillText()).toBe('Not installed');
    expect(liveText()).toBe('Host uninstalled.');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(document.getElementById('solidworks-host-warning')?.textContent).toContain(
      'some files could not be deleted',
    );
    expect(actionButtons().map((b) => b.getAttribute('data-sw-action'))).toEqual(['install']);
    expect(document.getElementById('solidworks-manual-install')).not.toBeNull();

    getStatus.mockResolvedValue(leftover);
    fireEvent.click(document.getElementById('solidworks-host-refresh') as HTMLButtonElement);
    await waitFor(() => expect(panelState()).toBe('not_responding'));
    expect(pillText()).toBe('Not running');
    expect(document.getElementById('solidworks-host-reinstall')).not.toBeNull();
    expect(document.getElementById('solidworks-host-warning')?.textContent).toContain(
      'some files could not be deleted',
    );
  });

  it('reports only actions to onActionBusyChange and keeps onBusyChange for checks too', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    install.mockResolvedValue(ST_RUNNING);
    const onBusyChange = vi.fn();
    const onActionBusyChange = vi.fn();
    renderPanel({ onBusyChange, onActionBusyChange });
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    expect(onActionBusyChange.mock.calls.map((call) => call[0])).toEqual([false]);
    fireEvent.click(screen.getByRole('button', { name: 'Install host' }));
    await waitFor(() => expect(panelState()).toBe('running_current'));
    fireEvent.click(document.getElementById('solidworks-host-refresh') as HTMLButtonElement);
    await waitFor(() => expect(getStatus).toHaveBeenCalledTimes(2));
    await flush();
    expect(onActionBusyChange.mock.calls.map((call) => call[0])).toEqual([false, true, false]);
    expect(onBusyChange.mock.calls.map((call) => call[0])).toEqual([
      false,
      true,
      false,
      true,
      false,
      true,
      false,
    ]);
  });

  it('gives each manual command card an outer hook', async () => {
    getStatus.mockResolvedValue(ST_NOT_INSTALLED);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_installed'));
    for (const step of ['verify', 'unzip', 'install']) {
      expect(document.getElementById(`solidworks-manual-command-${step}`)).not.toBeNull();
    }
  });

  it('labels a missing desktop session as Not logged in and wraps the log path in code', async () => {
    getStatus.mockResolvedValue(ST_NO_SESSION);
    const { unmount } = renderPanel();
    await waitFor(() => expect(panelState()).toBe('no_desktop_session'));
    expect(document.querySelector('[data-sw-fact="session"] dd')?.textContent).toBe(
      'Not logged in',
    );
    unmount();
    getStatus.mockResolvedValue(ST_NOT_RESPONDING);
    renderPanel();
    await waitFor(() => expect(panelState()).toBe('not_responding'));
    expect(document.querySelector('#solidworks-host-state code')?.textContent).toBe(
      '%LOCALAPPDATA%\\RagtimeSolidWorks\\logs\\host.log',
    );
  });
});
