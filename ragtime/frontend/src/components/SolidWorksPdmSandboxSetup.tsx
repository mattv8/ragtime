import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { CheckCircle } from 'lucide-react';
import { solidworksApi, SolidworksApiError, type PdmSandboxResponse } from '../api/solidworks';
import { SolidWorksPdmFolderBrowser } from './SolidWorksPdmFolderBrowser';
import { PdmMessage } from './shared/PdmMessage';

/** What the sandbox needs from the linked tool; present only when the trigger can work. */
export interface SandboxTarget {
  toolId: string;
  pdmToolId: string;
  vaultName: string;
}

export interface SolidWorksPdmSandboxSetupProps {
  target: SandboxTarget | null;
  /** Why the trigger is disabled; shown as help while `target` is null. */
  unavailableReason: string | null;
  allowScripts: boolean;
  disabled?: boolean;
  onBusyChange: (busy: boolean) => void;
  onCreated: (result: PdmSandboxResponse) => void;
}

interface Failure {
  tone: 'error' | 'warning';
  text: string;
  retry: boolean;
}

const DEFAULT_NAME = 'Ragtime Sandbox';
const FORM_ID = 'solidworks-pdm-sandbox-form';
const NAME_HELP =
  "Letters, digits, spaces, dots, hyphens and underscores. Start with a letter or digit; don't end with a dot or space.";
const NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$/;
const NO_DELETE_TEXT =
  'Ragtime cannot delete vault folders. To remove one later, ask a PDM administrator.';
const SCRIPTS_OFF =
  "Allow scripts is off, so the AI can't run scripts. This setup doesn't change it.";
const SCRIPTS_ON =
  "Allow scripts is on. Scripts run as this computer's Windows user, so they are not limited to the sandbox folder. This setup doesn't change it.";
const MISSING_RIGHT_LABELS: Record<string, string> = {
  read: 'read',
  add_folders: 'add folders',
  add_files: 'add files',
  check_out: 'check out',
};

/** The first problem with a folder name, or null when it may be created. */
export function sandboxNameProblem(name: string): string | null {
  if (name === '') return 'Enter a folder name.';
  if (!/^[A-Za-z0-9]/.test(name)) return 'Start with a letter or digit.';
  if (!NAME_PATTERN.test(name)) {
    return 'Use only letters, digits, spaces, dots, hyphens and underscores, up to 64 characters.';
  }
  if (/[. ]$/.test(name)) return "Don't end with a dot or space.";
  return null;
}

function missingRightsText(missing: readonly string[]): string {
  const labels = missing.map((right) => MISSING_RIGHT_LABELS[right] ?? right);
  return labels.length > 0 ? `: ${labels.join(', ')}` : '';
}

/** `sandbox_exists` is a field error on the name, not a form-level failure. */
function nameConflictMessage(error: unknown): string | null {
  return error instanceof SolidworksApiError && error.code === 'sandbox_exists'
    ? error.message
    : null;
}

function describeFailure(error: unknown, parent: string): Failure {
  if (!(error instanceof SolidworksApiError)) {
    const message = error instanceof Error ? error.message : String(error);
    return { tone: 'error', text: `Could not create the folder: ${message}`, retry: false };
  }
  switch (error.code) {
    case 'pdm_rights_missing':
      return {
        tone: 'error',
        text: `The PDM user is missing rights in ${parent}${missingRightsText(error.missing)}. Grant them in the PDM Administration tool, or choose another parent.`,
        retry: true,
      };
    case 'host_updating':
      return {
        tone: 'warning',
        text: 'The SolidWorks agent is updating. Try again in a few minutes.',
        retry: false,
      };
    case 'host_not_running':
      return {
        tone: 'warning',
        text: "The SolidWorks agent isn't running. Start it in the agent panel above, then try again.",
        retry: false,
      };
    case 'sandbox_overlap':
    case 'pdm_link_not_saved':
    case 'tool_disabled':
      return { tone: 'error', text: error.message, retry: false };
    default:
      return { tone: 'error', text: `Could not create the folder: ${error.message}`, retry: false };
  }
}

function goToAllowScripts() {
  const field = document.getElementById('solidworks-allow-scripts');
  field?.scrollIntoView({ block: 'center' });
  field?.focus();
}

export function SolidWorksPdmSandboxSetup({
  target,
  unavailableReason,
  allowScripts,
  disabled,
  onBusyChange,
  onCreated,
}: SolidWorksPdmSandboxSetupProps) {
  const [open, setOpen] = useState(false);
  const [parent, setParent] = useState<string | null>(null);
  const [name, setName] = useState(DEFAULT_NAME);
  const [nameError, setNameError] = useState<string | null>(null);
  const [touched, setTouched] = useState(false);
  const [busy, setBusy] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [success, setSuccess] = useState<PdmSandboxResponse | null>(null);
  const targetKey = target ? `${target.toolId}\n${target.pdmToolId}\n${target.vaultName}` : null;
  const [scope, setScope] = useState(targetKey);
  // A different tool, PDM tool or vault starts a new draft: the parent and errors belong to the old one.
  if (scope !== targetKey) {
    setScope(targetKey);
    setParent(null);
    setFailure(null);
    setNameError(null);
  }
  const triggerRef = useRef<HTMLButtonElement>(null);
  const formRef = useRef<HTMLDivElement>(null);
  const nameRef = useRef<HTMLInputElement>(null);
  const createRef = useRef<HTMLButtonElement>(null);
  const successRef = useRef<HTMLDivElement>(null);
  const focusParentOnOpen = useRef(false);
  const focusTriggerOnClose = useRef(false);

  const formVisible = open && target !== null;
  const locked = busy || disabled === true;
  const problem = sandboxNameProblem(name);
  const nameInvalid = nameError !== null || (touched && problem !== null);
  const nameMessage = nameError ?? problem ?? NAME_HELP;
  const canCreate = formVisible && parent !== null && problem === null && !locked;
  const phase = busy ? 'busy' : formVisible ? 'open' : success ? 'success' : 'idle';

  // The browser mounts with the form, so the filter input exists only after the form is visible.
  useEffect(() => {
    if (!formVisible || !focusParentOnOpen.current) return;
    focusParentOnOpen.current = false;
    formRef.current?.querySelector<HTMLInputElement>('.browser-filter-input')?.focus();
  }, [formVisible]);

  // The trigger is unmounted while the form is open, so focus returns once it is back.
  useEffect(() => {
    if (formVisible || !focusTriggerOnClose.current) return;
    focusTriggerOnClose.current = false;
    triggerRef.current?.focus();
  }, [formVisible]);

  // A failed attempt keeps focus on Create; it was disabled while the request ran.
  useEffect(() => {
    if (failure) createRef.current?.focus();
  }, [failure]);

  // A name the vault already has sends focus back to the name, with the parent still chosen.
  useEffect(() => {
    if (nameError) nameRef.current?.focus();
  }, [nameError]);

  useEffect(() => {
    if (success) successRef.current?.focus();
  }, [success]);

  const setBusyState = (next: boolean) => {
    setBusy(next);
    onBusyChange(next);
  };

  const openForm = () => {
    focusParentOnOpen.current = true;
    setSuccess(null);
    setFailure(null);
    setNameError(null);
    setOpen(true);
  };

  const closeForm = () => {
    focusTriggerOnClose.current = true;
    setOpen(false);
    setFailure(null);
    setNameError(null);
    setTouched(false);
  };

  const onFormKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key !== 'Escape') return;
    event.stopPropagation();
    if (!busy) closeForm();
  };

  const pickParent = (path: string) => {
    setParent(path);
    setFailure(null);
    setNameError(null);
  };

  const changeName = (value: string) => {
    setName(value);
    setFailure(null);
    setNameError(null);
  };

  const create = async () => {
    if (!canCreate || target === null || parent === null) return;
    setBusyState(true);
    setFailure(null);
    setNameError(null);
    try {
      const result = await solidworksApi.createPdmSandbox(target.toolId, {
        pdm_tool_id: target.pdmToolId,
        parent,
        name,
      });
      setOpen(false);
      setTouched(false);
      setSuccess(result);
      onCreated(result);
    } catch (error) {
      const conflict = nameConflictMessage(error);
      if (conflict !== null) setNameError(conflict);
      else setFailure(describeFailure(error, parent));
    } finally {
      setBusyState(false);
    }
  };

  const createAnother = () => {
    setName(DEFAULT_NAME);
    openForm();
  };

  return (
    <div
      id="solidworks-pdm-sandbox"
      className="sw-box"
      data-sw-box="sandbox"
      data-sw-sandbox-state={phase}
      role="group"
      aria-labelledby="solidworks-pdm-sandbox-title"
    >
      <h4 id="solidworks-pdm-sandbox-title" className="sw-box-title">
        Sandbox folder
      </h4>
      <p id="solidworks-pdm-sandbox-intro" className="field-help">
        A vault folder for the AI to work in. One step creates it, approves it and turns on Allow
        changes.
      </p>
      <PdmMessage id="solidworks-pdm-sandbox-no-delete" tone="info">
        {NO_DELETE_TEXT}
      </PdmMessage>
      {!formVisible && !success && (
        <>
          <div className="sw-actions">
            <button
              id="solidworks-pdm-sandbox-open"
              ref={triggerRef}
              type="button"
              className="btn btn-secondary btn-sm"
              aria-describedby={unavailableReason ? 'solidworks-pdm-sandbox-reason' : undefined}
              disabled={disabled === true || target === null}
              onClick={openForm}
            >
              Create sandbox folder…
            </button>
          </div>
          {target === null && unavailableReason && (
            <p id="solidworks-pdm-sandbox-reason" className="field-help">
              {unavailableReason}
            </p>
          )}
        </>
      )}
      {success && !formVisible && (
        <div id="solidworks-pdm-sandbox-success" ref={successRef} tabIndex={-1}>
          <div className="sw-check" data-tone="success">
            <CheckCircle size={14} aria-hidden="true" />
            <span>
              Created <code>{success.folder}</code>. It is approved and Allow changes is on.
            </span>
          </div>
          <div className="sw-actions">
            <button
              id="solidworks-pdm-sandbox-again"
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={disabled === true || target === null}
              onClick={createAnother}
            >
              Create another
            </button>
          </div>
        </div>
      )}
      {formVisible && target && (
        <div
          id={FORM_ID}
          ref={formRef}
          role="group"
          aria-label="Create sandbox folder"
          aria-busy={busy || undefined}
          onKeyDown={onFormKeyDown}
        >
          <div id="solidworks-pdm-sandbox-parent-group">
            <span className="sw-group-label">Parent folder</span>
            <code id="solidworks-pdm-sandbox-parent">{parent ?? 'None chosen'}</code>
            {parent === null && (
              <p id="solidworks-pdm-sandbox-parent-help" className="field-help">
                Choose a parent folder first.
              </p>
            )}
            <SolidWorksPdmFolderBrowser
              key={`${target.toolId}:${target.pdmToolId}:${target.vaultName}`}
              idPrefix="solidworks-pdm-sandbox-browser"
              mode="pick"
              toolId={target.toolId}
              pdmToolId={target.pdmToolId}
              vaultName={target.vaultName}
              folders={[]}
              vaultRoot={null}
              onPick={pickParent}
              disabled={locked}
            />
          </div>
          <div className="form-group">
            <label htmlFor="solidworks-pdm-sandbox-name">Folder name</label>
            <input
              id="solidworks-pdm-sandbox-name"
              ref={nameRef}
              type="text"
              value={name}
              maxLength={64}
              autoComplete="off"
              spellCheck={false}
              disabled={locked}
              aria-invalid={nameInvalid ? true : undefined}
              aria-describedby="solidworks-pdm-sandbox-name-help"
              onChange={(event) => changeName(event.target.value)}
              onBlur={() => setTouched(true)}
            />
            <p
              id="solidworks-pdm-sandbox-name-help"
              className="field-help"
              role={nameError ? 'alert' : undefined}
              data-invalid={nameInvalid ? 'true' : undefined}
            >
              {nameMessage}
            </p>
          </div>
          {parent !== null && (
            <p id="solidworks-pdm-sandbox-target" className="sw-muted">
              Will create <code className="sw-sandbox-target">{`${parent}\\${name}`}</code>
            </p>
          )}
          <ul id="solidworks-pdm-sandbox-effects" className="sw-checks">
            <li>Creates the folder in the vault.</li>
            <li>Approves it for vault changes.</li>
            <li>Turns on Allow changes.</li>
          </ul>
          <PdmMessage id="solidworks-pdm-sandbox-scripts" tone="info">
            {allowScripts ? SCRIPTS_ON : SCRIPTS_OFF}{' '}
            <button
              id="solidworks-pdm-sandbox-scripts-link"
              type="button"
              className="sw-inline-link"
              disabled={locked}
              onClick={goToAllowScripts}
            >
              Go to Allow scripts
            </button>
          </PdmMessage>
          <div id="solidworks-pdm-sandbox-result" role="status" aria-live="polite">
            {busy && (
              <p id="solidworks-pdm-sandbox-busy" className="sw-muted">
                Checking rights and creating the folder…
              </p>
            )}
          </div>
          {failure && (
            <>
              <PdmMessage id="solidworks-pdm-sandbox-error" tone={failure.tone} alert>
                {failure.text}
              </PdmMessage>
              {failure.retry && (
                <div className="sw-actions">
                  <button
                    id="solidworks-pdm-sandbox-retry"
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={locked}
                    onClick={() => void create()}
                  >
                    Try again
                  </button>
                </div>
              )}
            </>
          )}
          <div className="sw-actions">
            <button
              id="solidworks-pdm-sandbox-create"
              ref={createRef}
              type="button"
              className="btn"
              disabled={!canCreate}
              onClick={() => void create()}
            >
              {busy ? 'Creating…' : 'Create folder'}
            </button>
            <button
              id="solidworks-pdm-sandbox-cancel"
              type="button"
              className="btn btn-secondary"
              disabled={locked}
              onClick={closeForm}
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
