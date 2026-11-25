import { useLayoutEffect, useRef, useState } from 'react';
import type { DeleteToolConfigOptions, DeleteToolConfigResponse } from '@/api/client';
import {
  Check,
  CircleAlert,
  CircleCheck,
  CircleX,
  Copy,
  Info,
  Minus,
  Terminal,
  TriangleAlert,
  type LucideIcon,
} from 'lucide-react';

type DialogPhase = 'confirm' | 'working' | 'result';
const DELETE_WAIT_TIMEOUT_MS = 45_000;
type CleanupStepStatus = NonNullable<
  DeleteToolConfigResponse['cleanup']
>['steps'][keyof NonNullable<DeleteToolConfigResponse['cleanup']>['steps']];

interface ToolDeleteDialogProps {
  toolName: string;
  onDelete: (options: DeleteToolConfigOptions) => Promise<DeleteToolConfigResponse>;
  onClose: () => void;
  fallbackFocusRef?: React.RefObject<HTMLElement>;
}

interface DeleteFailureDetail {
  message?: string;
  cleanup?: DeleteToolConfigResponse['cleanup'];
}

function getDeleteFailureDetail(error: unknown): DeleteFailureDetail {
  if (!error || typeof error !== 'object') return {};
  const detail = 'structuredDetail' in error ? error.structuredDetail : undefined;
  if (!detail || typeof detail !== 'object') return {};
  return detail as DeleteFailureDetail;
}

function hasIncompleteCleanup(cleanup: DeleteToolConfigResponse['cleanup']): boolean {
  return Boolean(
    cleanup &&
    Object.values(cleanup.steps).some((step) => step === 'failed' || step === 'unconfirmed'),
  );
}

type StatusTone = 'success' | 'warning' | 'danger' | 'muted';
type StepDisplayStatus = CleanupStepStatus | 'done' | 'required';

const STEP_STATUS_DISPLAY: Record<
  StepDisplayStatus,
  { label: string; tone: StatusTone; Icon: LucideIcon }
> = {
  confirmed: { label: 'Confirmed', tone: 'success', Icon: CircleCheck },
  done: { label: 'Done', tone: 'success', Icon: CircleCheck },
  failed: { label: 'Failed', tone: 'danger', Icon: CircleX },
  unconfirmed: { label: 'Not confirmed', tone: 'warning', Icon: CircleAlert },
  not_requested: { label: 'Not requested', tone: 'muted', Icon: Minus },
  required: { label: 'Required', tone: 'warning', Icon: TriangleAlert },
};

const SUMMARY_DISPLAY: Record<
  Exclude<StatusTone, 'muted'>,
  { lead: string | null; Icon: LucideIcon }
> = {
  success: { lead: null, Icon: CircleCheck },
  warning: { lead: 'Manual cleanup needed.', Icon: TriangleAlert },
  danger: { lead: 'Delete failed.', Icon: CircleX },
};

interface CleanupStepRowProps {
  stepId: string;
  label: string;
  status: StepDisplayStatus;
  detail?: string;
}

function CleanupStepRow({ stepId, label, status, detail }: CleanupStepRowProps) {
  const { label: statusLabel, tone, Icon } = STEP_STATUS_DISPLAY[status];
  return (
    <li className="tool-delete-step" data-tool-delete-step={stepId}>
      <Icon className="tool-delete-icon" data-tone={tone} size={16} aria-hidden="true" />
      <span className="tool-delete-step-label">
        {label}
        {detail && <span className="tool-delete-step-detail">{detail}</span>}
      </span>
      <span
        className={`userspace-status-pill userspace-status-pill-${tone}`}
        data-tool-delete-step-status={status}
      >
        <Icon size={12} aria-hidden="true" />
        {statusLabel}
      </span>
    </li>
  );
}

export function ToolDeleteDialog({
  toolName,
  onDelete,
  onClose,
  fallbackFocusRef,
}: ToolDeleteDialogProps) {
  const [cleanupMode, setCleanupMode] =
    useState<DeleteToolConfigOptions['sshCleanup']>('disable_ssh');
  const [machineWideConfirmed, setMachineWideConfirmed] = useState(false);
  const [phase, setPhase] = useState<DialogPhase>('confirm');
  const [result, setResult] = useState<DeleteToolConfigResponse | null>(null);
  const [failureMessage, setFailureMessage] = useState<string | null>(null);
  const [copyError, setCopyError] = useState<string | null>(null);
  const [copySuccess, setCopySuccess] = useState(false);
  const openerRef = useRef<HTMLElement | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const workingStatusRef = useRef<HTMLParagraphElement>(null);
  const copyCommandRef = useRef<HTMLButtonElement>(null);
  const dismissRef = useRef<HTMLButtonElement>(null);
  const requestGenerationRef = useRef(0);

  useLayoutEffect(() => {
    const activeElement = document.activeElement;
    openerRef.current =
      activeElement instanceof HTMLElement &&
      activeElement !== document.body &&
      activeElement !== document.documentElement
        ? activeElement
        : null;
    const fallbackFocusTarget = fallbackFocusRef?.current;
    dialogRef.current?.querySelector<HTMLInputElement>('input[type="radio"]:checked')?.focus();
    return () => {
      requestGenerationRef.current += 1;
      if (openerRef.current?.isConnected) openerRef.current.focus();
      else fallbackFocusTarget?.focus();
    };
  }, [fallbackFocusRef]);

  useLayoutEffect(() => {
    if (phase === 'working') workingStatusRef.current?.focus();
    else if (phase === 'result') (copyCommandRef.current ?? dismissRef.current)?.focus();
  }, [phase]);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape' && phase !== 'working') {
      event.preventDefault();
      onClose();
      return;
    }
    if (event.key !== 'Tab') return;
    const focusable = Array.from(
      dialogRef.current?.querySelectorAll<HTMLElement>(
        'button:not([disabled]), input:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ) ?? [],
    );
    if (focusable.length === 0) {
      event.preventDefault();
      workingStatusRef.current?.focus();
      return;
    }
    const [first] = focusable;
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  };

  const handleDelete = async () => {
    if (cleanupMode === 'disable_ssh' && !machineWideConfirmed) return;
    setPhase('working');
    setCopyError(null);
    setCopySuccess(false);
    const requestGeneration = ++requestGenerationRef.current;
    const timeoutError = new Error(
      'The request did not finish within 45 seconds. It may still be running on the server; check the tool and Windows host before trying again.',
    );
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    try {
      const deletionResult = await Promise.race([
        onDelete({
          sshCleanup: cleanupMode,
          ...(cleanupMode === 'disable_ssh' ? { confirmMachineWide: true } : {}),
        }),
        new Promise<never>((_, reject) => {
          timeoutId = setTimeout(() => reject(timeoutError), DELETE_WAIT_TIMEOUT_MS);
        }),
      ]);
      if (requestGeneration !== requestGenerationRef.current) return;
      setResult(deletionResult);
    } catch (error) {
      if (requestGeneration !== requestGenerationRef.current) return;
      const detail = getDeleteFailureDetail(error);
      setFailureMessage(
        detail.message ?? (error instanceof Error ? error.message : 'Delete failed'),
      );
      setResult({ cleanup: detail.cleanup });
    } finally {
      clearTimeout(timeoutId);
      if (requestGeneration === requestGenerationRef.current) setPhase('result');
    }
  };

  const cleanup = result?.cleanup;
  const localRemovalCommand = cleanup?.local_removal_command;
  const localRemovalInstructions = cleanup?.local_removal_instructions;
  const cleanupDetailsMissing = phase === 'result' && !failureMessage && !cleanup;
  const cleanupNeedsRecovery = cleanupDetailsMissing || hasIncompleteCleanup(cleanup);
  const uniqueWarnings = Array.from(new Set(cleanup?.warnings ?? []));
  const resultSummary = failureMessage
    ? cleanup
      ? `${failureMessage} Host changes may already have been attempted.`
      : `${failureMessage} We could not confirm deletion or host cleanup status.`
    : cleanupDetailsMissing
      ? 'Tool deleted. Host cleanup was requested but not confirmed.'
      : cleanupNeedsRecovery
        ? 'Tool deleted. Host cleanup not confirmed or failed.'
        : 'The tool was deleted.';
  const summaryTone: Exclude<StatusTone, 'muted'> = failureMessage
    ? 'danger'
    : cleanupNeedsRecovery
      ? 'warning'
      : 'success';
  const { lead: summaryLead, Icon: SummaryIcon } = SUMMARY_DISPLAY[summaryTone];
  const toolRecordStatus: StepDisplayStatus = failureMessage ? 'unconfirmed' : 'done';
  const manualCleanupDetail = localRemovalCommand
    ? 'Run the command below as an Administrator.'
    : localRemovalInstructions
      ? 'Follow the instructions below.'
      : cleanupDetailsMissing
        ? 'The server did not return host cleanup details. Complete cleanup manually on the Windows host.'
        : 'Complete cleanup manually on the Windows host.';

  const handleCopyCommand = async () => {
    if (!localRemovalCommand) return;
    try {
      await navigator.clipboard.writeText(localRemovalCommand);
      setCopyError(null);
      setCopySuccess(true);
    } catch {
      setCopySuccess(false);
      setCopyError('Could not copy the command. Select and copy it manually.');
    }
  };

  return (
    <div className="modal-overlay" data-tool-delete-dialog-overlay>
      <div
        ref={dialogRef}
        id="tool-delete-dialog"
        className="modal-content"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tool-delete-dialog-title"
        aria-describedby="tool-delete-dialog-description"
        aria-busy={phase === 'working'}
        data-tool-delete-dialog
        onKeyDown={handleKeyDown}
      >
        <div className="modal-header">
          <h3 id="tool-delete-dialog-title">
            {phase === 'result' ? 'Tool deletion result' : 'Delete paired Windows tool'}
          </h3>
          {phase !== 'working' && (
            <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
              ×
            </button>
          )}
        </div>
        <div className="modal-body">
          {phase === 'confirm' && (
            <>
              <p id="tool-delete-dialog-description">
                Delete <strong>{toolName}</strong> and choose how to clean up its paired Windows
                host.
              </p>
              <fieldset data-tool-delete-cleanup-options>
                <legend>Windows host cleanup</legend>
                <label className="tool-delete-dialog-radio-label">
                  <input
                    type="radio"
                    name="ssh-cleanup"
                    value="disable_ssh"
                    checked={cleanupMode === 'disable_ssh'}
                    onChange={() => setCleanupMode('disable_ssh')}
                  />
                  Remove and disable SSH
                </label>
                <label className="tool-delete-dialog-radio-label">
                  <input
                    type="radio"
                    name="ssh-cleanup"
                    value="pairing_only"
                    checked={cleanupMode === 'pairing_only'}
                    onChange={() => setCleanupMode('pairing_only')}
                  />
                  Remove Ragtime only
                </label>
              </fieldset>
              {cleanupMode === 'disable_ssh' && (
                <label
                  className="tool-delete-dialog-checkbox-label"
                  data-tool-delete-machine-wide-confirmation
                >
                  <input
                    type="checkbox"
                    checked={machineWideConfirmed}
                    onChange={(event) => setMachineWideConfirmed(event.target.checked)}
                  />
                  I understand this stops SSH for all users and other Ragtime connections on this
                  machine.
                </label>
              )}
            </>
          )}
          {phase === 'working' && (
            <p
              ref={workingStatusRef}
              id="tool-delete-dialog-description"
              role="status"
              tabIndex={-1}
            >
              Deleting the tool and requesting host cleanup…
            </p>
          )}
          {phase === 'result' && (
            <div className="tool-delete-result" data-tool-delete-result>
              <p
                id="tool-delete-dialog-description"
                className="tool-delete-result-summary"
                role={summaryTone === 'success' ? 'status' : 'alert'}
              >
                <SummaryIcon
                  className="tool-delete-icon"
                  data-tone={summaryTone}
                  size={16}
                  aria-hidden="true"
                />
                <span>
                  {summaryLead && (
                    <>
                      <strong>{summaryLead}</strong>{' '}
                    </>
                  )}
                  {resultSummary}
                </span>
              </p>
              <ol
                className="tool-delete-steps"
                role="list"
                aria-label="Deletion steps"
                data-tool-delete-steps
              >
                <CleanupStepRow
                  stepId="tool-record"
                  label="Remove tool from Ragtime"
                  status={toolRecordStatus}
                />
                {cleanup && (
                  <>
                    <CleanupStepRow
                      stepId="pairing-cleanup"
                      label="Pairing cleanup"
                      status={cleanup.steps.pairing_cleanup}
                    />
                    <CleanupStepRow
                      stepId="ssh-service"
                      label="SSH service"
                      status={cleanup.steps.ssh_service}
                    />
                  </>
                )}
                {cleanupNeedsRecovery && (
                  <CleanupStepRow
                    stepId="manual-cleanup"
                    label="Finish cleanup on the host"
                    status="required"
                    detail={manualCleanupDetail}
                  />
                )}
              </ol>
              {uniqueWarnings.length > 0 && (
                <ul
                  className="tool-delete-notes"
                  role="list"
                  aria-label="Cleanup notes"
                  data-tool-delete-notes
                >
                  {uniqueWarnings.map((warning) => (
                    <li key={warning} className="tool-delete-note">
                      <Info
                        className="tool-delete-icon"
                        data-tone="muted"
                        size={14}
                        aria-hidden="true"
                      />
                      <span>{warning}</span>
                    </li>
                  ))}
                </ul>
              )}
              {cleanupNeedsRecovery && localRemovalCommand && (
                <div
                  className="tool-delete-command"
                  role="group"
                  aria-label="Recovery command"
                  data-tool-delete-recovery-command
                >
                  <div className="tool-delete-command-bar">
                    <span className="tool-delete-command-label">
                      <Terminal size={14} aria-hidden="true" />
                      PowerShell
                    </span>
                    <button
                      ref={copyCommandRef}
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => void handleCopyCommand()}
                    >
                      {copySuccess ? (
                        <>
                          <Check size={14} aria-hidden="true" />
                          Copied
                        </>
                      ) : (
                        <>
                          <Copy size={14} aria-hidden="true" />
                          Copy<span className="rc-visually-hidden"> command</span>
                        </>
                      )}
                    </button>
                  </div>
                  <pre tabIndex={0} role="region" aria-label="PowerShell command">
                    <code>{localRemovalCommand}</code>
                  </pre>
                  <p className="rc-visually-hidden" role="status">
                    {copySuccess ? 'Command copied.' : ''}
                  </p>
                </div>
              )}
              {copyError && (
                <p className="tool-delete-copy-error" role="alert">
                  {copyError}
                </p>
              )}
              {cleanupNeedsRecovery && !localRemovalCommand && localRemovalInstructions && (
                <div
                  className="tool-delete-instructions"
                  role="group"
                  aria-labelledby="tool-delete-recovery-instructions-title"
                  data-tool-delete-recovery-instructions
                >
                  <p
                    id="tool-delete-recovery-instructions-title"
                    className="tool-delete-instructions-title"
                  >
                    Manual cleanup instructions:
                  </p>
                  <p>{localRemovalInstructions}</p>
                </div>
              )}
            </div>
          )}
        </div>
        <div className="modal-footer">
          {phase === 'confirm' && (
            <>
              <button type="button" className="btn btn-secondary" onClick={onClose}>
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-danger"
                onClick={() => void handleDelete()}
                disabled={cleanupMode === 'disable_ssh' && !machineWideConfirmed}
              >
                Delete tool
              </button>
            </>
          )}
          {phase === 'result' && (
            <button ref={dismissRef} type="button" className="btn btn-secondary" onClick={onClose}>
              Dismiss
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
