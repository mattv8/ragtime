import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import {
  AlertTriangle,
  CheckCircle,
  Circle,
  Info,
  RefreshCw,
  XCircle,
  type LucideIcon,
} from 'lucide-react';
import {
  solidworksApi,
  SolidworksApiError,
  type SolidworksHostArtifact,
  type SolidworksHostStatus,
} from '../api/solidworks';
import { InlineCopyButton } from './shared/InlineCopyButton';
import { InfoPopover } from './shared/InfoPopover';

export interface SolidWorksHostPanelProps {
  toolId: string | null;
  connectionReady: boolean;
  onStatusChange?: (status: SolidworksHostStatus | null) => void;
  /** Reports checking, installing, updating, reinstalling and uninstalling. Background polls do not count. */
  onBusyChange?: (busy: boolean) => void;
  /** Reports only install, update, reinstall and uninstall in flight (not status checks). */
  onActionBusyChange?: (busy: boolean) => void;
}

type Action = 'install' | 'update' | 'reinstall' | 'uninstall';
type Via = 'primary' | 'retry' | 'uninstall';
type Busy = { kind: 'check' } | { kind: 'action'; action: Action; via: Via };
interface Failure {
  action: 'check' | Action;
  /** Only user-initiated failures use role="alert"; the first automatic check stays polite. */
  user: boolean;
  message: string;
  code?: string;
}
type Button = 'install' | 'update' | 'reinstall' | 'retry' | 'uninstall';
type Tone = 'success' | 'warning' | 'info' | 'danger' | 'muted';
type FactKey = 'host-version' | 'account' | 'session' | 'solidworks' | 'starts';
type State =
  | 'blocked'
  | 'checking'
  | 'installing'
  | 'uninstalling'
  | 'install_failed'
  | 'uninstall_failed'
  | 'request_failed'
  | 'unreachable'
  | 'artifact_unavailable'
  | 'not_installed'
  | 'no_desktop_session'
  | 'not_responding'
  | 'running_current'
  | 'running_update';

const POLL_INTERVAL_MS = 10_000;
const POLL_LIMIT = 30;
const MISSING_ARTIFACT: SolidworksHostArtifact = {
  available: false,
  host_version: null,
  protocol: null,
  sha256: null,
  size_bytes: null,
};
const MANUAL_STATES: State[] = [
  'not_installed',
  'running_update',
  'not_responding',
  'install_failed',
];
const POLLING_STATES: State[] = ['no_desktop_session', 'not_responding'];
const PRIMARY_BY_STATE: Partial<Record<State, Button>> = {
  not_installed: 'install',
  running_update: 'update',
  not_responding: 'reinstall',
};
const STATE_META: Record<State, { label: string; tone: Tone; spinner: boolean }> = {
  blocked: { label: 'Waiting for connection', tone: 'muted', spinner: false },
  checking: { label: 'Checking…', tone: 'muted', spinner: true },
  installing: { label: 'Installing…', tone: 'info', spinner: true },
  uninstalling: { label: 'Uninstalling…', tone: 'info', spinner: true },
  install_failed: { label: 'Install failed', tone: 'danger', spinner: false },
  uninstall_failed: { label: 'Uninstall failed', tone: 'danger', spinner: false },
  request_failed: { label: 'Check failed', tone: 'danger', spinner: false },
  unreachable: { label: 'Unreachable', tone: 'danger', spinner: false },
  artifact_unavailable: { label: 'Installer missing', tone: 'warning', spinner: false },
  not_installed: { label: 'Not installed', tone: 'muted', spinner: false },
  no_desktop_session: { label: 'Waiting for desktop session', tone: 'warning', spinner: false },
  not_responding: { label: 'Not running', tone: 'warning', spinner: false },
  running_current: { label: 'Running', tone: 'success', spinner: false },
  running_update: { label: 'Running', tone: 'success', spinner: false },
};
const TONE_ICON: Record<Tone, LucideIcon> = {
  success: CheckCircle,
  danger: XCircle,
  warning: AlertTriangle,
  info: Info,
  muted: Circle,
};
const BUTTON_LABEL: Record<Button, string> = {
  install: 'Install host',
  update: 'Update host',
  reinstall: 'Reinstall host',
  retry: 'Try again',
  uninstall: 'Uninstall host',
};
const PENDING_LABEL: Record<Action, string> = {
  install: 'Installing…',
  update: 'Updating…',
  reinstall: 'Reinstalling…',
  uninstall: 'Uninstalling…',
};

function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : 'Request failed';
}

function megabytes(bytes: number | null | undefined): string {
  return bytes ? `${(bytes / 1048576).toFixed(1)}\u00a0MB` : '';
}

/** Status-only classification (B.1 steps 2 and 6-10). */
function observedState(
  status: SolidworksHostStatus | null,
  artifact: SolidworksHostArtifact | null,
): State {
  if (!status) return 'checking';
  if (!status.reachable) return 'unreachable';
  if (!status.installed)
    return artifact?.available === false ? 'artifact_unavailable' : 'not_installed';
  if (!status.running)
    return status.session?.interactive === false ? 'no_desktop_session' : 'not_responding';
  return status.update_available ? 'running_update' : 'running_current';
}

/** The state a failed user action or check leaves behind (B.1 steps 4-5). */
function failureState(failure: Failure | null): State | null {
  if (!failure) return null;
  if (failure.action === 'check') return 'request_failed';
  return failure.action === 'uninstall' ? 'uninstall_failed' : 'install_failed';
}

function factRowsFor(state: State): readonly FactKey[] {
  if (state === 'running_current' || state === 'running_update')
    return ['account', 'session', 'solidworks', 'starts', 'host-version'];
  if (state === 'no_desktop_session' || state === 'not_responding')
    return ['account', 'session', 'solidworks', 'host-version'];
  if (state === 'not_installed' || state === 'artifact_unavailable')
    return ['account', 'session', 'solidworks'];
  return [];
}

function failureHint(code: string | undefined): string | null {
  if (code === 'hash_mismatch')
    return 'Try again. If it keeps failing, install manually and compare the hash yourself.';
  if (code === 'upload_failed' || code === 'ssh_unreachable')
    return 'Check the connection and the free disk space on the machine.';
  if (code === 'no_interactive_session')
    return 'Log in to the machine as this account, then try again.';
  return null;
}

interface TextContext {
  status: SolidworksHostStatus | null;
  artifact: SolidworksHostArtifact | null;
  action: Action | null;
  failure: Failure | null;
  uninstalled: boolean;
  pollingStopped: boolean;
}

const REQUEST_FAILED_TEXT = 'Ragtime could not check the host.';

function stateText(state: State, context: TextContext): ReactNode {
  const account = context.status?.account?.name ?? 'this account';
  const size = megabytes(context.artifact?.size_bytes);
  switch (state) {
    case 'blocked':
      return 'Finish the connection in the previous step first. The host is installed over that connection.';
    case 'checking':
      return 'Checking the SSH connection and the SolidWorks host. This can take up to 30 seconds.';
    case 'installing':
      if (context.action === 'update')
        return 'Updating the SolidWorks host… Updating also closes SolidWorks instances the host started.';
      if (context.action === 'reinstall') return 'Reinstalling the SolidWorks host…';
      return 'Installing the SolidWorks host. Uploading, verifying and starting it can take about a minute.';
    case 'uninstalling':
      return 'Removing the SolidWorks host…';
    case 'install_failed':
      return 'Install failed. See the details below.';
    case 'uninstall_failed':
      return 'Uninstall failed. See the details below.';
    case 'request_failed':
      return REQUEST_FAILED_TEXT;
    case 'unreachable':
      return 'Ragtime could not reach this machine over SSH.';
    case 'artifact_unavailable':
      return 'This Ragtime server does not include the SolidWorks host installer, so the host cannot be installed from here. Update or rebuild Ragtime with the host package, then check again.';
    case 'not_installed':
      if (context.uninstalled) return 'Host uninstalled.';
      return `The SolidWorks host is not installed for this account. Ragtime can install it over SSH: it uploads the package${size ? ` (${size})` : ''}, verifies its SHA-256 hash, installs it for this account only and starts it at logon.`;
    case 'no_desktop_session':
      if (context.pollingStopped)
        return `Ragtime stopped checking automatically. Log in as ${account}, then choose Check again.`;
      return `The host is installed, but ${account} has no desktop session on this machine, so it cannot start. Log in to the machine as ${account}; a remote desktop session is enough and it may stay disconnected. Ragtime checks again every 10 seconds.`;
    case 'not_responding':
      return (
        <>
          The host is installed and {account} is logged in, but the host did not answer.
          Reinstalling registers its logon task again and starts it. The host log is{' '}
          <code>%LOCALAPPDATA%\RagtimeSolidWorks\logs\host.log</code> on the machine.
        </>
      );
    case 'running_current':
      return 'The host is installed and answering.';
    case 'running_update':
      return `Version ${context.status?.host_version ?? 'unknown'} is running; version ${context.status?.expected_host_version ?? 'unknown'} is available.`;
  }
}

function Command({ id, command }: { id: string; command: string }) {
  return (
    <div id={`solidworks-manual-command-${id}`} className="rc-command-block">
      <div className="rc-command-toolbar">
        <span className="rc-command-label">PowerShell</span>
        <InlineCopyButton
          id={`solidworks-manual-copy-${id}`}
          copyText={command}
          className="btn btn-secondary btn-sm"
          title="Copy command"
          ariaLabel="Copy command"
        />
      </div>
      <pre>
        <code className="rc-command-code font-mono">{command}</code>
      </pre>
    </div>
  );
}

function Manual({ artifact, open }: { artifact: SolidworksHostArtifact; open: boolean }) {
  const details = useRef<HTMLDetailsElement>(null);
  useEffect(() => {
    if (open && details.current) details.current.open = true;
  }, [open]);
  const size = megabytes(artifact.size_bytes) || 'unknown size';
  return (
    <details id="solidworks-manual-install" className="sw-manual" ref={details}>
      <summary>Install manually instead</summary>
      <p>
        Use this if automatic install fails or you prefer to install by hand. Do it as the SSH
        account, in that account&apos;s desktop session.
      </p>
      <ol id="solidworks-manual-steps" className="sw-manual-steps">
        <li data-sw-manual-step="download">
          Download the host package ({artifact.host_version ?? 'unknown version'}, {size}).
          <div>
            <a
              id="solidworks-host-download"
              className="btn btn-secondary btn-sm"
              href={solidworksApi.hostDownloadUrl()}
              download
            >
              Download host (.zip)
            </a>
          </div>
          <div>
            SHA-256{' '}
            <code id="solidworks-host-sha256" className="sw-hash">
              {artifact.sha256}
            </code>{' '}
            <InlineCopyButton
              id="solidworks-host-sha256-copy"
              copyText={artifact.sha256 ?? ''}
              className="btn btn-secondary btn-sm"
              title="Copy SHA-256"
              ariaLabel="Copy SHA-256"
            />
          </div>
        </li>
        <li data-sw-manual-step="verify">
          Copy the zip to the machine and check its hash. The result must match the SHA-256 above.
          <Command
            id="verify"
            command={'Get-FileHash .\\ragtime-solidworks-host-win-x64.zip -Algorithm SHA256'}
          />
        </li>
        <li data-sw-manual-step="unzip">
          Unzip it to <code>%LOCALAPPDATA%\RagtimeSolidWorks\host</code>.
          <Command
            id="unzip"
            command={
              'Expand-Archive .\\ragtime-solidworks-host-win-x64.zip -DestinationPath "$env:LOCALAPPDATA\\RagtimeSolidWorks\\host" -Force'
            }
          />
        </li>
        <li data-sw-manual-step="install">
          Register and start the host:
          <Command
            id="install"
            command={
              '& "$env:LOCALAPPDATA\\RagtimeSolidWorks\\host\\ragtime-solidworks-host.exe" install'
            }
          />
          <p>
            Run this as the SSH account (not another administrator), in a PowerShell window on that
            account&apos;s desktop.
          </p>
        </li>
        <li data-sw-manual-step="recheck">Come back here and choose Check again.</li>
      </ol>
    </details>
  );
}

export function SolidWorksHostPanel({
  toolId,
  connectionReady,
  onStatusChange,
  onBusyChange,
  onActionBusyChange,
}: SolidWorksHostPanelProps) {
  const [status, setStatus] = useState<SolidworksHostStatus | null>(null);
  const [artifact, setArtifact] = useState<SolidworksHostArtifact | null>(null);
  const [busy, setBusy] = useState<Busy | null>(null);
  const [requesting, setRequesting] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [uninstalled, setUninstalled] = useState(false);
  const [attempts, setAttempts] = useState(0);
  const [confirming, setConfirming] = useState(false);
  const generation = useRef(0);
  const inFlight = useRef(false);
  const controller = useRef<AbortController | null>(null);
  const statusCallback = useRef(onStatusChange);
  const busyCallback = useRef(onBusyChange);
  const actionBusyCallback = useRef(onActionBusyChange);
  const cancelButton = useRef<HTMLButtonElement>(null);
  const uninstallButton = useRef<HTMLButtonElement>(null);
  const restoreUninstallFocus = useRef(false);

  useEffect(() => {
    statusCallback.current = onStatusChange;
    busyCallback.current = onBusyChange;
    actionBusyCallback.current = onActionBusyChange;
  });

  const beginRequest = useCallback((next: Busy | null) => {
    const gen = ++generation.current;
    inFlight.current = true;
    setRequesting(true);
    setBusy(next);
    return gen;
  }, []);

  const endRequest = useCallback((gen: number) => {
    if (gen !== generation.current) return;
    inFlight.current = false;
    setRequesting(false);
    setBusy(null);
  }, []);

  const checkStatus = useCallback(
    async (id: string, kind: 'initial' | 'user' | 'poll') => {
      if (inFlight.current) return;
      const signal = controller.current?.signal;
      const gen = beginRequest(kind === 'poll' ? null : { kind: 'check' });
      if (kind !== 'poll') {
        setFailure(null);
        setUninstalled(false);
      }
      if (kind === 'user') setAttempts(0);
      try {
        const next = await solidworksApi.getHostStatus(id, signal);
        if (gen !== generation.current) return;
        setStatus(next);
        statusCallback.current?.(next);
      } catch (error) {
        if (gen !== generation.current || signal?.aborted) return;
        setFailure({ action: 'check', user: kind === 'user', message: messageOf(error) });
      } finally {
        endRequest(gen);
      }
    },
    [beginRequest, endRequest],
  );

  useEffect(() => {
    setStatus(null);
    setArtifact(null);
    setFailure(null);
    setUninstalled(false);
    setAttempts(0);
    setBusy(null);
    setRequesting(false);
    if (!toolId || !connectionReady) {
      statusCallback.current?.(null);
      return;
    }
    const abort = new AbortController();
    controller.current = abort;
    solidworksApi.getHostArtifact(abort.signal).then(
      (next) => {
        if (!abort.signal.aborted) setArtifact(next);
      },
      () => undefined,
    );
    void checkStatus(toolId, 'initial');
    return () => {
      abort.abort();
      generation.current += 1;
      inFlight.current = false;
      controller.current = null;
    };
  }, [toolId, connectionReady, checkStatus]);

  useEffect(() => {
    busyCallback.current?.(busy !== null);
  }, [busy]);

  const actionBusy = busy?.kind === 'action';
  useEffect(() => {
    actionBusyCallback.current?.(actionBusy);
  }, [actionBusy]);

  // A successful uninstall means "not installed" even if leftover files still report installed.
  // The marker is cleared by the next check or action, so later checks resolve from the status.
  const observed =
    uninstalled && status?.reachable ? 'not_installed' : observedState(status, artifact);
  const resting = failureState(failure) ?? observed;
  const polling = POLLING_STATES.includes(resting) && attempts < POLL_LIMIT;
  const pollable = Boolean(toolId && connectionReady && polling);

  useEffect(() => {
    if (!pollable || !toolId) return;
    const timer = window.setInterval(() => {
      if (document.hidden || inFlight.current) return;
      setAttempts((count) => count + 1);
      void checkStatus(toolId, 'poll');
    }, POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [pollable, toolId, checkStatus]);

  useEffect(() => {
    if (confirming) cancelButton.current?.focus();
  }, [confirming]);

  useEffect(() => {
    if (!confirming && restoreUninstallFocus.current) {
      restoreUninstallFocus.current = false;
      uninstallButton.current?.focus();
    }
  }, [confirming]);

  const closeConfirm = () => {
    restoreUninstallFocus.current = true;
    setConfirming(false);
  };

  const runAction = async (action: Action, via: Via) => {
    if (!toolId || inFlight.current) return;
    const id = toolId;
    setConfirming(false);
    setFailure(null);
    setUninstalled(false);
    setAttempts(0);
    const gen = beginRequest({ kind: 'action', action, via });
    try {
      const next =
        action === 'uninstall'
          ? await solidworksApi.uninstallHost(id)
          : await solidworksApi.installHost(id);
      if (gen !== generation.current) return;
      setStatus(next);
      statusCallback.current?.(next);
      if (next.error)
        setFailure({ action, user: true, message: next.error.message, code: next.error.code });
      else setUninstalled(action === 'uninstall');
    } catch (error) {
      if (gen !== generation.current) return;
      setFailure({
        action,
        user: true,
        message: messageOf(error),
        code: error instanceof SolidworksApiError ? error.code : undefined,
      });
      // HTTP 503 on install/update/reinstall means the server has no host package.
      if (action !== 'uninstall' && error instanceof SolidworksApiError && error.status === 503)
        setArtifact(MISSING_ARTIFACT);
    } finally {
      endRequest(gen);
    }
  };

  const state: State =
    !toolId || !connectionReady
      ? 'blocked'
      : busy?.kind === 'check' || (!status && !failure)
        ? 'checking'
        : busy?.kind === 'action'
          ? busy.action === 'uninstall'
            ? 'uninstalling'
            : 'installing'
          : resting;
  const meta = STATE_META[state];
  const pendingAction = busy?.kind === 'action' ? busy.action : null;
  const retryAction = failure && failure.action !== 'check' ? failure.action : null;
  const retryMode = busy?.kind === 'action' ? busy.via === 'retry' : retryAction !== null;
  const buttons: Button[] = (() => {
    if (state === 'blocked' || state === 'checking' || state === 'request_failed') return [];
    if (retryMode) return ['retry'];
    const primary = PRIMARY_BY_STATE[observed];
    const removable = observed === 'running_current' || observed === 'running_update';
    return [...(primary ? [primary] : []), ...(removable ? (['uninstall'] as const) : [])];
  })();
  const showFacts =
    status !== null && !['blocked', 'request_failed', 'unreachable'].includes(state);
  const facts = showFacts ? factRowsFor(observed) : [];
  const versions = status?.solidworks?.installed_versions;
  const factValue = (key: FactKey): string => {
    if (!status) return 'Unknown';
    if (key === 'host-version') return status.host_version ?? 'Unknown';
    if (key === 'account') return status.account?.name ?? 'Unknown';
    if (key === 'session')
      return status.session?.interactive === true
        ? 'Logged in'
        : status.session?.interactive === false
          ? 'Not logged in'
          : 'Unknown';
    if (key === 'solidworks') return versions ? versions.join(', ') || 'None found' : 'Unknown';
    return 'At logon';
  };
  const hint = retryAction ? failureHint(failure?.code) : null;
  const failureTitle =
    failure?.action === 'uninstall'
      ? 'Uninstall failed'
      : failure?.action === 'check'
        ? 'Check failed'
        : 'Details';
  const showAlert = failure !== null && (failure.action !== 'check' || failure.user);
  const text = stateText(state, {
    status,
    artifact,
    action: pendingAction,
    failure,
    uninstalled,
    pollingStopped: attempts >= POLL_LIMIT,
  });
  const Icon = TONE_ICON[meta.tone];
  const showSpinner = meta.spinner || (polling && state === resting);

  const renderButton = (kind: Button) => {
    const pressed =
      busy?.kind === 'action' &&
      (kind === 'retry'
        ? busy.via === 'retry'
        : kind === 'uninstall'
          ? busy.via === 'uninstall'
          : busy.via === 'primary' && busy.action === kind);
    const label = pressed && pendingAction ? PENDING_LABEL[pendingAction] : BUTTON_LABEL[kind];
    if (kind === 'uninstall')
      return (
        <button
          key={kind}
          ref={uninstallButton}
          id="solidworks-host-uninstall"
          data-sw-action="uninstall"
          className="btn btn-secondary btn-sm"
          type="button"
          disabled={requesting}
          onClick={() => setConfirming(true)}
        >
          {label}
        </button>
      );
    return (
      <button
        key={kind}
        id={`solidworks-host-${kind}`}
        data-sw-action={kind}
        className="btn"
        type="button"
        disabled={requesting}
        onClick={() => {
          if (kind === 'retry') {
            if (retryAction) void runAction(retryAction, 'retry');
          } else void runAction(kind, 'primary');
        }}
      >
        {label}
      </button>
    );
  };

  return (
    <section
      id="solidworks-host-panel"
      className="sw-panel"
      data-testid="solidworks-host-panel"
      data-sw-host-state={state}
      data-sw-host-action-error={
        failure && failure.action !== 'check'
          ? failure.action === 'uninstall'
            ? 'uninstall'
            : 'install'
          : undefined
      }
      aria-labelledby="solidworks-host-title"
      aria-busy={requesting}
    >
      <header className="sw-panel-header">
        <h3 id="solidworks-host-title">SolidWorks host</h3>
        <span className="sw-pill" data-tone={meta.tone}>
          {showSpinner ? (
            <span className="rc-polling-indicator" aria-hidden="true" />
          ) : (
            <Icon size={14} aria-hidden="true" />
          )}
          {meta.label}
        </span>
        {state === 'running_update' && (
          <span className="sw-pill" data-tone="info">
            <Info size={14} aria-hidden="true" />
            Update available
          </span>
        )}
        {state !== 'blocked' && (
          <button
            id="solidworks-host-refresh"
            data-sw-action="refresh"
            type="button"
            className="btn btn-secondary btn-sm"
            disabled={requesting}
            onClick={() => toolId && void checkStatus(toolId, 'user')}
          >
            <RefreshCw size={14} aria-hidden="true" /> Check again
          </button>
        )}
        {buttons.includes('uninstall') && renderButton('uninstall')}
      </header>
      <div className={state === 'running_update' ? 'sw-message' : undefined}>
        <p
          id="solidworks-host-state"
          role="status"
          aria-live="polite"
          className={state === 'running_current' ? 'sw-visually-hidden' : undefined}
        >
          {showAlert ? null : text}
        </p>
        {state === 'running_update' && (
          <InfoPopover id="solidworks-host-update-help" label="About updating">
            Updating restarts the host and closes any SolidWorks it started. Calls in progress fail.
          </InfoPopover>
        )}
      </div>
      {state === 'request_failed' && !showAlert && failure?.message && <p>{failure.message}</p>}
      {status?.warning && (
        <p id="solidworks-host-warning" className="sw-message" data-tone="warning">
          <AlertTriangle className="sw-message-icon" size={14} aria-hidden="true" />
          <span>{status.warning}</span>
        </p>
      )}
      {showAlert && failure && (
        <div id="solidworks-host-error" className="sw-panel" data-tone="error" role="alert">
          <strong>{failureTitle}</strong>
          <p>{failure.message}</p>
          {failure.code && (
            <p>
              Error code: <code className="font-mono">{failure.code}</code>
            </p>
          )}
          {hint && <p>{hint}</p>}
        </div>
      )}
      {(state === 'unreachable' || state === 'not_responding') && status?.error && (
        <div>
          <p>{status.error.message}</p>
          <p>
            Error code: <code className="font-mono">{status.error.code}</code>
          </p>
        </div>
      )}
      {state === 'unreachable' && (
        <ul>
          <li>Check that the machine is on and the OpenSSH Server service is running.</li>
          <li>Check the host name, port and credentials from the previous step.</li>
          <li>For a paired machine, check that the Ragtime remote client is running on it.</li>
        </ul>
      )}
      {facts.length > 0 && (
        <dl id="solidworks-host-facts" className="sw-facts" aria-label="SolidWorks host details">
          {facts.map((key) => (
            <div key={key} data-sw-fact={key}>
              <dt>
                {key === 'host-version'
                  ? 'Host version'
                  : key === 'account'
                    ? 'Account'
                    : key === 'session'
                      ? 'Session'
                      : key === 'solidworks'
                        ? 'SolidWorks'
                        : 'Starts'}
                {key === 'session' && (
                  <InfoPopover id="solidworks-session-help" label="About session">
                    The tool works only while this account is logged in on the machine. A
                    disconnected remote desktop session counts as logged in.
                  </InfoPopover>
                )}
              </dt>
              <dd>{factValue(key)}</dd>
            </div>
          ))}
        </dl>
      )}
      {observed === 'running_current' && versions?.length === 0 && (
        <div className="sw-message" data-tone="warning">
          <AlertTriangle className="sw-message-icon" size={14} aria-hidden="true" />
          <span>The host is running but found no SolidWorks installation for this account.</span>
        </div>
      )}
      {status?.account?.is_administrator === true && (
        <div
          id="solidworks-account-warning"
          className="sw-message"
          data-tone="warning"
          data-boxed="true"
          role="note"
        >
          <AlertTriangle className="sw-message-icon" size={14} aria-hidden="true" />
          <span>Runs as an administrator account.</span>
          <InfoPopover id="solidworks-account-warning-help" label="About the administrator account">
            The tool acts as <strong>{status.account.name}</strong>, an account with administrator
            rights. Choose an account that only has the access SolidWorks work needs.
          </InfoPopover>
        </div>
      )}
      {confirming ? (
        <div
          id="solidworks-uninstall-confirm"
          className="sw-panel"
          data-tone="error"
          role="group"
          aria-labelledby="solidworks-uninstall-title"
          onKeyDown={(event) => {
            if (event.key === 'Escape') closeConfirm();
          }}
        >
          <strong id="solidworks-uninstall-title">Uninstall the SolidWorks host?</strong>
          <p>
            This stops the host, closes any SolidWorks it started, removes its logon task and
            deletes its files from this machine. Every Ragtime tool that uses this account on this
            machine stops working until you install the host again. Your SolidWorks installation and
            documents are not touched.
          </p>
          <div className="sw-actions">
            <button
              id="solidworks-uninstall-confirm-button"
              className="btn btn-danger"
              type="button"
              onClick={() => void runAction('uninstall', 'uninstall')}
            >
              Uninstall host
            </button>
            <button
              ref={cancelButton}
              id="solidworks-uninstall-cancel"
              className="btn btn-secondary"
              type="button"
              onClick={closeConfirm}
            >
              Cancel
            </button>
          </div>
        </div>
      ) : (
        <div id="solidworks-host-actions" className="sw-actions">
          {buttons.filter((button) => button !== 'uninstall').map(renderButton)}
        </div>
      )}
      {artifact?.available === true && MANUAL_STATES.includes(resting) && (
        <Manual artifact={artifact} open={resting === 'install_failed'} />
      )}
    </section>
  );
}
