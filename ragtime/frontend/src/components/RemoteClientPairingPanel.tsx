import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import type { RemotePairingCreatedResponse, RemotePairingResponse, ToolType } from '@/types';
import { InlineCopyButton } from './shared/InlineCopyButton';

type TargetOs = 'linux' | 'windows' | 'mac';
type ConnectivityMode = 'online' | 'offline';
const POLL_INTERVAL_MS = 2000;
const RETRY_INTERVAL_MS = 2500;

interface RemoteClientPairingPanelProps {
  toolType: Extract<ToolType, 'ssh_shell' | 'windows_shell'>;
  pairingId: string | null;
  toolId?: string;
  onPairingCreated: (pairing: RemotePairingCreatedResponse) => void;
  onPairingUpdated: (pairing: RemotePairingResponse) => void;
}

function initialTargetOs(): TargetOs {
  if (typeof navigator !== 'undefined' && /win/i.test(navigator.userAgent)) return 'windows';
  if (typeof navigator !== 'undefined' && /mac/i.test(navigator.userAgent)) return 'mac';
  return 'linux';
}

function targetOsOptions(
  toolType: RemoteClientPairingPanelProps['toolType'],
): Array<[TargetOs, string]> {
  return toolType === 'windows_shell'
    ? [['windows', 'Windows']]
    : [
        ['linux', 'Linux'],
        ['mac', 'macOS'],
      ];
}

function secondsRemaining(expiresAt: string): number {
  return Math.max(0, Math.ceil((new Date(expiresAt).getTime() - Date.now()) / 1000));
}

function commandFor(
  pairing: RemotePairingResponse & Partial<Pick<RemotePairingCreatedResponse, 'commands'>>,
  targetOs: TargetOs,
  mode: ConnectivityMode,
): string {
  const windows = targetOs === 'windows';
  if (!pairing.commands) return '';
  return mode === 'offline'
    ? pairing.commands[windows ? 'offline_powershell' : 'offline_sh']
    : pairing.commands[windows ? 'powershell' : 'sh'];
}

export function RemoteClientPairingPanel({
  toolType,
  pairingId,
  toolId,
  onPairingCreated,
  onPairingUpdated,
}: RemoteClientPairingPanelProps) {
  const generationRef = useRef(0);
  const timerRef = useRef<number | null>(null);
  const [pairing, setPairing] = useState<
    (RemotePairingResponse & Partial<Pick<RemotePairingCreatedResponse, 'commands'>>) | null
  >(null);
  const [targetOs, setTargetOs] = useState<TargetOs>(initialTargetOs);
  const [connectivityMode, setConnectivityMode] = useState<ConnectivityMode>('online');
  const [seconds, setSeconds] = useState(0);
  const [networkError, setNetworkError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [offlineBlob, setOfflineBlob] = useState('');
  const [importing, setImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);

  useEffect(() => {
    const options = targetOsOptions(toolType);
    if (!options.some(([value]) => value === targetOs)) setTargetOs(options[0][0]);
  }, [targetOs, toolType]);

  const applyPairing = useCallback(
    (next: RemotePairingResponse) => {
      setPairing((current) => ({ ...current, ...next }));
      onPairingUpdated(next);
    },
    [onPairingUpdated],
  );

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  const poll = useCallback(
    async (id: string, generation: number) => {
      if (generation !== generationRef.current) return;
      try {
        const next = await api.getRemotePairing(id);
        if (generation !== generationRef.current) return;
        setNetworkError(null);
        setRetryCount(0);
        applyPairing(next);
        if (next.status === 'pending' || next.status === 'consumed') {
          timerRef.current = window.setTimeout(() => void poll(id, generation), POLL_INTERVAL_MS);
        }
      } catch {
        if (generation !== generationRef.current) return;
        setNetworkError('Lost connection to Ragtime. Retrying…');
        setRetryCount((current) => current + 1);
        timerRef.current = window.setTimeout(() => void poll(id, generation), RETRY_INTERVAL_MS);
      }
    },
    [applyPairing],
  );

  useEffect(() => {
    const generation = ++generationRef.current;
    clearTimer();
    const start = async () => {
      try {
        if (pairingId) {
          const existing = await api.getRemotePairing(pairingId);
          if (generation !== generationRef.current) return;
          applyPairing(existing);
          if (existing.status === 'pending' || existing.status === 'consumed') {
            void poll(existing.id, generation);
          }
          return;
        }
        const created = await api.createRemotePairing(toolType, toolId);
        if (generation !== generationRef.current) return;
        setPairing(created);
        onPairingCreated(created);
        void poll(created.id, generation);
      } catch {
        if (generation === generationRef.current)
          setNetworkError('Unable to start remote pairing. Try again.');
      }
    };
    void start();
    return () => {
      generationRef.current += 1;
      clearTimer();
    };
  }, [applyPairing, clearTimer, onPairingCreated, pairingId, poll, toolId, toolType]);

  useEffect(() => {
    if (!pairing) return;
    const update = () => setSeconds(secondsRemaining(pairing.expires_at));
    update();
    const interval = window.setInterval(update, 1000);
    return () => window.clearInterval(interval);
  }, [pairing]);

  const regenerate = async () => {
    if (!pairing) return;
    const generation = ++generationRef.current;
    clearTimer();
    try {
      const next = await api.regenerateRemotePairing(pairing.id);
      if (generation !== generationRef.current) return;
      setPairing(next);
      onPairingCreated(next);
      setNetworkError(null);
      setRetryCount(0);
      void poll(next.id, generation);
    } catch {
      if (generation === generationRef.current)
        setNetworkError('Could not generate a new command. Try again.');
    }
  };

  const importOffline = async () => {
    if (!pairing || !offlineBlob.trim()) return;
    setImporting(true);
    setImportError(null);
    try {
      applyPairing(await api.completeRemotePairingOffline(pairing.id, offlineBlob.trim()));
      setOfflineBlob('');
    } catch {
      setImportError('Could not import installer output. Check the pasted blob and try again.');
    } finally {
      setImporting(false);
    }
  };

  if (!pairing) {
    return (
      <p className="wizard-help" role="status">
        Preparing secure connection instructions…
      </p>
    );
  }

  const expired = seconds === 0 || pairing.status === 'expired';
  const command = commandFor(pairing, targetOs, connectivityMode);
  const facts = pairing.facts;
  return (
    <section
      data-wizard-step="connection"
      data-remote-client-pairing="true"
      className="remote-client-pairing"
    >
      <p className="wizard-help">
        Run the command on the target machine. The installer uses the target's invoking user.
      </p>
      <fieldset id="rc-os-selector">
        <legend>Target operating system</legend>
        {targetOsOptions(toolType).map(([value, label]) => (
          <label key={value} className="radio-label">
            <input
              id={`rc-os-${value}`}
              type="radio"
              name="rc-target-os"
              checked={targetOs === value}
              onChange={() => setTargetOs(value)}
            />
            {label}
          </label>
        ))}
      </fieldset>
      <fieldset id="rc-mode-selector">
        <legend>Connectivity mode</legend>
        <label className="radio-label">
          <input
            id="rc-mode-online"
            type="radio"
            name="rc-mode"
            checked={connectivityMode === 'online'}
            onChange={() => setConnectivityMode('online')}
          />
          Online
        </label>
        <label className="radio-label">
          <input
            id="rc-mode-offline"
            type="radio"
            name="rc-mode"
            checked={connectivityMode === 'offline'}
            onChange={() => setConnectivityMode('offline')}
          />
          Offline / paste-back
        </label>
      </fieldset>
      <section
        className={`rc-command-block${expired ? ' rc-command-expired' : ''}`}
        data-rc-command-os={targetOs}
        data-rc-command-mode={connectivityMode}
        aria-label="Installation command"
      >
        <div className="rc-command-toolbar">
          <span className="rc-command-label">Run on target machine</span>
          <InlineCopyButton
            className="btn btn-secondary"
            copyText={command}
            title="Copy installation command"
            ariaLabel="Copy installation command"
            disabled={expired || !command}
            feedbackMs={2000}
          />
        </div>
        <pre>
          <code className="rc-command-code font-mono">
            {expired
              ? 'Token expired.'
              : command || 'Generate a new command to continue this pairing.'}
          </code>
        </pre>
        <p className="field-help" aria-live="polite">
          {expired
            ? 'Token expired. '
            : `Token expires in ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')} — `}
          <button type="button" className="link-btn" onClick={() => void regenerate()}>
            {expired ? 'Generate new command' : 'Generate new'}
          </button>
        </p>
        {expired && (
          <span className="rc-visually-hidden" role="alert">
            Token expired.
          </span>
        )}
      </section>
      {networkError && (
        <div className="tool-access-callout tool-access-callout-warning" role="alert">
          {networkError}
          {retryCount >= 3 && ` Retry ${retryCount}.`}
        </div>
      )}
      <div id="rc-enrollment-status" role="status" aria-live="polite">
        {(pairing.status === 'pending' || pairing.status === 'consumed') && (
          <span
            aria-label="Polling for enrollment…"
            role="status"
            className="rc-polling-indicator"
          />
        )}
        {pairing.status === 'pending' && 'Waiting for installer to run…'}
        {pairing.status === 'consumed' && 'Installer ran. Probing connection…'}
        {pairing.status === 'failed' &&
          (pairing.error ||
            'The installer could not complete pairing. Generate a new command to try again.')}
        {pairing.status === 'expired' && 'Token expired. Generate a new command to continue.'}
        {pairing.status === 'revoked' &&
          'This pairing was revoked. Create a new tool to reconnect.'}
        {pairing.status === 'paired' && `Connected — ${facts?.hostname ?? 'target'}`}
      </div>
      {(pairing.status === 'paired' ||
        pairing.status === 'expired' ||
        pairing.status === 'revoked') && (
        <span className="rc-visually-hidden" role="alert">
          {pairing.status === 'paired'
            ? 'Pairing completed.'
            : pairing.status === 'expired'
              ? 'Enrollment token expired.'
              : 'Pairing was revoked.'}
        </span>
      )}
      {facts && (
        <>
          <dl data-rc-section="facts" className="rc-facts-list" aria-label="Target details">
            <div>
              <dt>Host</dt>
              <dd>{facts.hostname}</dd>
            </div>
            <div>
              <dt>User</dt>
              <dd>{facts.username}</dd>
            </div>
            <div>
              <dt>OS</dt>
              <dd>
                <span className="badge">{facts.os}</span>
              </dd>
            </div>
            <div>
              <dt>Transport</dt>
              <dd>
                <span className="badge">{pairing.transport ?? 'direct'}</span>
              </dd>
            </div>
            <div>
              <dt>Pinned public host key</dt>
              <dd>
                <code className="font-mono rc-fingerprint">
                  {facts.host_keys[0] ?? 'Not reported'}
                </code>
                {facts.host_keys[0] && (
                  <InlineCopyButton
                    className="btn btn-secondary"
                    copyText={facts.host_keys[0]}
                    title="Copy pinned public host key"
                    ariaLabel="Copy pinned public host key"
                  />
                )}
              </dd>
            </div>
          </dl>
          {facts.os !== 'windows' && !facts.python_available && (
            <p className="field-help warning">
              Python was not found on this POSIX target. Install it to enable Python-dependent
              capabilities.
            </p>
          )}
          {facts.shared_admin_keyfile && (
            <div data-rc-section="shared-admin-notice" className="rc-warning-notice" role="note">
              <strong>Windows shared admin key</strong>
              <p>
                The server authorized the key in{' '}
                <code>%ProgramData%\ssh\administrators_authorized_keys</code>. This grants access to
                all local administrator accounts on this host, not only the current user. Review and
                restrict the key if this is not intended.
              </p>
            </div>
          )}
        </>
      )}
      {connectivityMode === 'offline' && (
        <p className="field-help">
          <a href="/indexes/remote/install.sh" download>
            Download install.sh
          </a>
          {' · '}
          <a href="/indexes/remote/install.ps1" download>
            Download install.ps1
          </a>
        </p>
      )}
      {connectivityMode === 'offline' &&
        (pairing.status === 'pending' ||
          pairing.status === 'consumed' ||
          pairing.status === 'failed') && (
          <section data-rc-section="offline-import" className="rc-offline-import">
            <h3>Paste installer output</h3>
            <label htmlFor="rc-offline-blob">Installer output</label>
            <textarea
              id="rc-offline-blob"
              aria-describedby="rc-offline-blob-help"
              value={offlineBlob}
              onChange={(event) => setOfflineBlob(event.target.value)}
              rows={8}
            />
            <p id="rc-offline-blob-help" className="field-help">
              The output is a base64-encoded JSON blob. It does not contain private keys.
            </p>
            <button
              type="button"
              className="btn"
              aria-busy={importing}
              disabled={importing || !offlineBlob.trim()}
              onClick={() => void importOffline()}
            >
              {importing ? 'Importing…' : 'Import'}
            </button>
            {importError && (
              <p className="field-help error" role="alert">
                {importError}
              </p>
            )}
          </section>
        )}
    </section>
  );
}
