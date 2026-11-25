import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { ToolConfig } from '../types/api';
import { quickSetupPdmConfig } from './pdm/pdmDefaults';

export interface SolidWorksPdmSetupFormProps {
  onCreated: (tool: ToolConfig) => void;
  onCancel: () => void;
  disabled?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onIndexError?: (message: string) => void;
}

const errorMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));

export function SolidWorksPdmSetupForm({
  onCreated,
  onCancel,
  disabled,
  onBusyChange,
  onIndexError,
}: SolidWorksPdmSetupFormProps) {
  const [host, setHost] = useState('');
  const [port, setPort] = useState(1433);
  const [user, setUser] = useState('');
  const [password, setPassword] = useState('');
  const [database, setDatabase] = useState('');
  const [databases, setDatabases] = useState<string[] | null>(null);
  const [noDatabases, setNoDatabases] = useState(false);
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [createdWithoutPolicy, setCreatedWithoutPolicy] = useState(false);
  const hostInput = useRef<HTMLInputElement>(null);
  useEffect(() => {
    hostInput.current?.focus();
  }, []);
  useEffect(() => {
    onBusyChange?.(busy);
    return () => {
      if (busy) onBusyChange?.(false);
    };
  }, [busy, onBusyChange]);

  const portValid = Number.isInteger(port) && port >= 1 && port <= 65535;
  const validConnection = Boolean(host.trim() && user.trim() && password && portValid);
  const errors: Record<string, string> = {
    host: host.trim() ? '' : 'Enter a SQL Server host.',
    port: portValid ? '' : 'Enter a whole number from 1 to 65535.',
    user: user.trim() ? '' : 'Enter a user.',
    password: password ? '' : 'Enter a password.',
    database: database.trim() ? '' : 'Enter a database.',
  };
  const fieldError = (field: string) => touched[field] && errors[field];
  const markTouched = (field: string) => setTouched((previous) => ({ ...previous, [field]: true }));
  const clearDiscovery = () => {
    setDatabases(null);
    setNoDatabases(false);
  };
  const discover = async () => {
    setBusy(true);
    setError(null);
    setStatus('Discovering…');
    try {
      const result = await api.discoverMssqlDatabases({ host, port, user, password });
      if (!result.success) throw new Error(result.error || 'Discovery failed');
      const names = result.database_options?.length
        ? result.database_options.filter((item) => item.accessible).map((item) => item.name)
        : result.databases;
      setDatabases(names.length ? names : null);
      setNoDatabases(names.length === 0);
      if (names.length && !names.includes(database)) setDatabase('');
    } catch (err) {
      setError(`Could not discover databases: ${errorMessage(err)}.`);
    } finally {
      setBusy(false);
      setStatus('');
    }
  };

  const submit = async () => {
    setBusy(true);
    setError(null);
    setStatus('Testing the connection…');
    const fields = {
      host: host.trim(),
      port,
      user: user.trim(),
      password,
      database: database.trim(),
    };
    try {
      const result = await api.discoverPdmSchema(fields);
      if (!result.success) throw new Error(result.error || 'Connection failed');
    } catch (err) {
      setError(`Could not connect: ${errorMessage(err)}. Nothing was created.`);
      setBusy(false);
      setStatus('');
      return;
    }
    setStatus('Creating the PDM tool…');
    let tool: ToolConfig;
    try {
      tool = await api.createToolConfig({
        name: `PDM ${fields.database}`,
        tool_type: 'solidworks_pdm',
        connection_config: quickSetupPdmConfig(fields),
        skip_indexing: true,
      });
    } catch (err) {
      setError(`Could not create the PDM tool: ${errorMessage(err)}.`);
      setBusy(false);
      setStatus('');
      return;
    }
    try {
      await api.updateToolAccessPolicy(tool.id, {
        tool_id: tool.id,
        default_chat_access: 'deny',
        default_workspace_access: 'deny',
        users: [],
        groups: [],
      });
    } catch (err) {
      setCreatedWithoutPolicy(true);
      setError(
        `The PDM tool was created, but its access policy could not be saved: ${errorMessage(err)}. Find it in Tools.`,
      );
      setBusy(false);
      setStatus('');
      return;
    }
    setStatus('Starting the first index…');
    try {
      await api.triggerPdmIndex(tool.id);
    } catch (err) {
      const indexError = `Could not start indexing: ${errorMessage(err)}`;
      setError(indexError);
      onIndexError?.(indexError);
    }
    setBusy(false);
    setStatus('');
    onCreated(tool);
  };

  return (
    <div
      id="solidworks-pdm-setup"
      role="group"
      aria-label="New search index connection"
      aria-busy={busy}
      onKeyDown={(event) => {
        if (event.key === 'Escape' && !busy) {
          event.preventDefault();
          onCancel();
        }
      }}
    >
      <p id="solidworks-pdm-setup-intro" className="field-help">
        Ragtime reads the PDM vault database with this SQL login to build the search index. A
        read-only login is enough. Vault access on this machine uses the PDM user, not this login.
      </p>
      <fieldset className="sw-pdm-controls" disabled={disabled || busy}>
        <div className="form-group">
          <label htmlFor="solidworks-pdm-setup-host">SQL Server host</label>
          <input
            ref={hostInput}
            id="solidworks-pdm-setup-host"
            type="text"
            required
            value={host}
            placeholder="pdm-sql.example.local"
            onChange={(event) => {
              setHost(event.target.value);
              clearDiscovery();
            }}
            onBlur={() => markTouched('host')}
            aria-invalid={!!fieldError('host')}
            aria-describedby={`solidworks-pdm-setup-host-help${fieldError('host') ? ' solidworks-pdm-setup-host-error' : ''}`}
          />
          <p id="solidworks-pdm-setup-host-help" className="field-help">
            The server that hosts the PDM vault database.
          </p>
          {fieldError('host') && <p id="solidworks-pdm-setup-host-error">{errors.host}</p>}
        </div>
        <div className="form-group">
          <label htmlFor="solidworks-pdm-setup-port">Port</label>
          <input
            id="solidworks-pdm-setup-port"
            type="number"
            required
            min={1}
            max={65535}
            step={1}
            value={port}
            onChange={(event) => {
              setPort(Number(event.target.value));
              clearDiscovery();
            }}
            onBlur={() => markTouched('port')}
            aria-invalid={!!fieldError('port')}
            aria-describedby={fieldError('port') ? 'solidworks-pdm-setup-port-error' : undefined}
          />
          {fieldError('port') && <p id="solidworks-pdm-setup-port-error">{errors.port}</p>}
        </div>
        <div className="form-group">
          <label htmlFor="solidworks-pdm-setup-user">User</label>
          <input
            id="solidworks-pdm-setup-user"
            type="text"
            required
            value={user}
            placeholder={'sa or domain\\user'}
            onChange={(event) => {
              setUser(event.target.value);
              clearDiscovery();
            }}
            onBlur={() => markTouched('user')}
            aria-invalid={!!fieldError('user')}
            aria-describedby={fieldError('user') ? 'solidworks-pdm-setup-user-error' : undefined}
          />
          {fieldError('user') && <p id="solidworks-pdm-setup-user-error">{errors.user}</p>}
        </div>
        <div className="form-group">
          <label htmlFor="solidworks-pdm-setup-password">Password</label>
          <input
            id="solidworks-pdm-setup-password"
            type="password"
            required
            autoComplete="off"
            value={password}
            onChange={(event) => {
              setPassword(event.target.value);
              clearDiscovery();
            }}
            onBlur={() => markTouched('password')}
            aria-invalid={!!fieldError('password')}
            aria-describedby={
              fieldError('password') ? 'solidworks-pdm-setup-password-error' : undefined
            }
          />
          {fieldError('password') && (
            <p id="solidworks-pdm-setup-password-error">{errors.password}</p>
          )}
        </div>
        <div className="form-group">
          <label htmlFor="solidworks-pdm-setup-database">Database</label>
          <div className="sw-input-with-action">
            {databases === null ? (
              <input
                id="solidworks-pdm-setup-database"
                type="text"
                required
                value={database}
                onChange={(event) => setDatabase(event.target.value)}
                onBlur={() => markTouched('database')}
                aria-invalid={!!fieldError('database')}
                aria-describedby={`solidworks-pdm-setup-database-help${fieldError('database') ? ' solidworks-pdm-setup-database-error' : ''}`}
              />
            ) : (
              <select
                id="solidworks-pdm-setup-database"
                required
                value={database}
                onChange={(event) => setDatabase(event.target.value)}
                onBlur={() => markTouched('database')}
                aria-invalid={!!fieldError('database')}
                aria-describedby={`solidworks-pdm-setup-database-help${fieldError('database') ? ' solidworks-pdm-setup-database-error' : ''}`}
              >
                <option value="">Choose a database</option>
                {databases.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            )}
            <button
              id="solidworks-pdm-setup-discover"
              type="button"
              className="btn btn-secondary"
              disabled={!validConnection}
              onClick={() => void discover()}
            >
              {status === 'Discovering…' ? 'Discovering…' : 'Discover'}
            </button>
          </div>
          <p id="solidworks-pdm-setup-database-help" className="field-help">
            The PDM vault database. It usually has the same name as the vault.
          </p>
          {fieldError('database') && (
            <p id="solidworks-pdm-setup-database-error">{errors.database}</p>
          )}
          {noDatabases && (
            <p className="field-help">
              No databases found. Enter the vault database name manually.
            </p>
          )}
        </div>
        <div className="sw-actions">
          <button
            id="solidworks-pdm-setup-submit"
            type="button"
            className="btn"
            disabled={createdWithoutPolicy || !validConnection || !database.trim()}
            onClick={() => void submit()}
          >
            Connect and start indexing
          </button>
          <button
            id="solidworks-pdm-setup-cancel"
            type="button"
            className="btn btn-secondary"
            onClick={onCancel}
          >
            Cancel
          </button>
        </div>
      </fieldset>
      <p id="solidworks-pdm-setup-status" role="status" aria-live="polite">
        {status}
      </p>
      {error && (
        <p id="solidworks-pdm-setup-error" role="alert">
          {error}
        </p>
      )}
      <p id="solidworks-pdm-setup-fallback" className="field-help">
        Need an SSH tunnel, document filters or a schedule?{' '}
        <a href="?view=tools#tools-connections" target="_blank" rel="noopener noreferrer">
          Add the PDM tool in Tools
        </a>{' '}
        (opens in a new tab), then come back and choose Refresh.
      </p>
    </div>
  );
}
