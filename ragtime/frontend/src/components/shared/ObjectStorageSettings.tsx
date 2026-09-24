import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { api } from '@/api/client';
import type { ObjectStorageAdminSettings, ObjectStorageMigrationJob } from '@/types';

const ACTIVE_STATES = new Set(['pending', 'copying', 'verifying']);

export interface ObjectStorageSettingsHandle {
  save(): Promise<void>;
}

export const ObjectStorageSettings = forwardRef<ObjectStorageSettingsHandle>(
  function ObjectStorageSettings(_props, ref): JSX.Element {
    const [settings, setSettings] = useState<ObjectStorageAdminSettings | null>(null);
    const [mode, setMode] = useState<'local' | 'external'>('local');
    const [fields, setFields] = useState({
      endpoint: '',
      region: 'us-east-1',
      bucket: '',
      access_key_id: '',
      secret_access_key: '',
    });
    const [createBucket, setCreateBucket] = useState(false);
    const [migrate, setMigrate] = useState(false);
    const [busy, setBusy] = useState<'test' | 'save' | 'migrate' | null>(null);
    const [message, setMessage] = useState<string | null>(null);
    const mounted = useRef(true);
    const busyRef = useRef<typeof busy>(null);

    const setOperation = useCallback((operation: typeof busy) => {
      busyRef.current = operation;
      setBusy(operation);
    }, []);
    const startOperation = useCallback(
      (operation: Exclude<typeof busy, null>) => {
        if (busyRef.current !== null) return false;
        setOperation(operation);
        return true;
      },
      [setOperation],
    );

    const load = useCallback(async () => {
      try {
        const next = await api.getObjectStorageAdminSettings();
        if (!mounted.current) return;
        setSettings(next);
        setMode(next.mode);
        setFields((current) => ({
          ...current,
          endpoint: next.endpoint ?? '',
          region: next.region ?? 'us-east-1',
          bucket: next.bucket ?? '',
        }));
      } catch (error) {
        if (mounted.current)
          setMessage(
            error instanceof Error ? error.message : 'Unable to load object storage settings',
          );
      }
    }, []);
    useEffect(() => {
      void load();
      return () => {
        mounted.current = false;
      };
    }, [load]);
    useEffect(() => {
      if (!settings?.migrations.some((job) => ACTIVE_STATES.has(job.state))) return;
      const refresh = () => {
        if (document.visibilityState === 'visible') void load();
      };
      const timer = window.setInterval(refresh, 5000);
      document.addEventListener('visibilitychange', refresh);
      return () => {
        window.clearInterval(timer);
        document.removeEventListener('visibilitychange', refresh);
      };
    }, [load, settings?.migrations]);
    const request = useCallback(
      () => ({
        mode,
        ...(mode === 'external' ? { ...fields, create_bucket: createBucket } : {}),
      }),
      [createBucket, fields, mode],
    );
    const test = async () => {
      if (!startOperation('test')) return;
      setMessage(null);
      try {
        await api.testObjectStorageAdminSettings(request());
        setMessage('Connection test succeeded. Settings have not been saved.');
      } catch (e) {
        setMessage(e instanceof Error ? e.message : 'Connection test failed');
      } finally {
        setOperation(null);
      }
    };
    const save = useCallback(async () => {
      if (settings === null) {
        throw new Error('Object storage settings are still loading');
      }
      if (busyRef.current !== null) {
        throw new Error('Another object storage operation is in progress');
      }
      startOperation('save');
      setMessage(null);
      try {
        const next = await api.updateObjectStorageAdminSettings(request());
        setSettings(next);
        setFields((current) => ({ ...current, access_key_id: '', secret_access_key: '' }));
        setMessage('Object storage settings saved. Existing workspace bindings are unchanged.');
        if (migrate && next.existing_local_workspaces > 0) {
          setOperation('migrate');
          const result = await api.startObjectStorageMigration();
          setSettings({ ...next, migrations: result.jobs });
          setMessage(
            'Migration started. Existing workspace bindings will switch only after verification.',
          );
        }
      } catch (error) {
        const errorMessage =
          error instanceof Error ? error.message : 'Unable to save object storage settings';
        setMessage(errorMessage);
        throw error instanceof Error ? error : new Error(errorMessage);
      } finally {
        setOperation(null);
      }
    }, [migrate, request, setOperation, settings, startOperation]);
    useImperativeHandle(ref, () => ({ save }), [save]);
    const retry = async (job: ObjectStorageMigrationJob) => {
      if (!startOperation('migrate')) return;
      try {
        await api.retryObjectStorageMigration(job.id);
        await load();
      } catch (e) {
        setMessage(e instanceof Error ? e.message : 'Unable to retry migration');
      } finally {
        setOperation(null);
      }
    };
    return (
      <section
        id="object-storage-settings"
        data-settings-filter-card="true"
        aria-labelledby="object-storage-settings-title"
      >
        <h4 id="object-storage-settings-title">Object Storage</h4>
        <p className="field-help">
          Local storage is ready with no setup. External credentials are never displayed; leave
          secret fields empty to preserve saved values.
        </p>
        <div className="form-group">
          <label>
            <input type="radio" checked={mode === 'local'} onChange={() => setMode('local')} />{' '}
            Local (default)
          </label>{' '}
          <label>
            <input
              type="radio"
              checked={mode === 'external'}
              onChange={() => setMode('external')}
            />{' '}
            External S3-compatible storage
          </label>
        </div>
        {mode === 'external' && (
          <div className="settings-form-grid">
            {(
              [
                ['endpoint', 'Endpoint'],
                ['region', 'Region'],
                ['bucket', 'Private root bucket'],
                [
                  'access_key_id',
                  settings?.access_key_configured ? 'Access key (configured)' : 'Access key',
                ],
                [
                  'secret_access_key',
                  settings?.secret_key_configured
                    ? 'Secret key (configured; leave empty to preserve)'
                    : 'Secret key',
                ],
              ] as const
            ).map(([key, label]) => (
              <div className="form-group" key={key}>
                <label htmlFor={`object-storage-${key}`}>{label}</label>
                <input
                  id={`object-storage-${key}`}
                  type={key === 'secret_access_key' ? 'password' : 'text'}
                  value={fields[key]}
                  onChange={(event) => setFields({ ...fields, [key]: event.target.value })}
                />
              </div>
            ))}
            <div className="form-group">
              <label>
                <input
                  type="checkbox"
                  checked={createBucket}
                  onChange={(event) => setCreateBucket(event.target.checked)}
                />{' '}
                Create the private root bucket if needed
              </label>
            </div>
          </div>
        )}
        {mode === 'external' && (
          <div className="form-group">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => void test()}
              disabled={busy !== null}
            >
              {busy === 'test' ? <Loader2 size={14} className="spinning" /> : null} Test connection
            </button>
          </div>
        )}
        {settings && settings.existing_local_workspaces > 0 && (
          <div id="object-storage-migrations" data-settings-filter-card="true">
            <p className="field-help">
              {settings.existing_local_workspaces} existing workspace
              {settings.existing_local_workspaces === 1 ? '' : 's'} remain on their current backend
              until migrated.
            </p>
            <label>
              <input
                type="checkbox"
                checked={migrate}
                onChange={(event) => setMigrate(event.target.checked)}
              />{' '}
              Migrate existing workspaces after saving
            </label>
            <div>
              {settings.migrations.map((job) => (
                <div key={job.id}>
                  <strong>{job.workspace_id}</strong>: {job.state} ({job.objects_copied} objects,{' '}
                  {job.bytes_copied} bytes){' '}
                  {job.error && <span className="userspace-object-error">{job.error}</span>}{' '}
                  {job.state === 'failed' && (
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => void retry(job)}
                      disabled={busy !== null}
                    >
                      Retry
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
        {mode === 'local' && (
          <div className="form-group">
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => void load()}
              disabled={busy !== null}
            >
              <RefreshCw size={14} /> Refresh status
            </button>
          </div>
        )}
        {message && (
          <p role="status" className="field-help">
            {message}
          </p>
        )}
      </section>
    );
  },
);
