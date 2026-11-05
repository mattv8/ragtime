import { useLayoutEffect, useRef, useState } from 'react';
import type { DeleteToolConfigOptions, DeleteToolConfigResponse } from '@/api/client';

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

function getCleanupStepLabel(step: CleanupStepStatus): string {
  switch (step) {
    case 'confirmed':
      return 'Confirmed';
    case 'failed':
      return 'Failed';
    case 'unconfirmed':
      return 'Not confirmed';
    case 'not_requested':
      return 'Not requested';
  }
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
  const resultActionRef = useRef<HTMLButtonElement>(null);
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
    else if (phase === 'result') resultActionRef.current?.focus();
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
            <div data-tool-delete-result>
              {failureMessage || cleanupNeedsRecovery ? (
                <p id="tool-delete-dialog-description" role="alert">
                  {resultSummary}
                </p>
              ) : (
                <p id="tool-delete-dialog-description" role="status">
                  {resultSummary}
                </p>
              )}
              {cleanup && (
                <>
                  <p>
                    Pairing cleanup: {getCleanupStepLabel(cleanup.steps.pairing_cleanup)}. SSH
                    service: {getCleanupStepLabel(cleanup.steps.ssh_service)}.
                  </p>
                  {uniqueWarnings.map((warning) => (
                    <p key={warning} className="field-help">
                      {warning}
                    </p>
                  ))}
                </>
              )}
              {cleanupDetailsMissing && (
                <p className="field-help">
                  The server did not return host cleanup details. Complete cleanup manually on the
                  Windows host.
                </p>
              )}
              {cleanupNeedsRecovery && localRemovalCommand && (
                <div
                  role="group"
                  aria-labelledby="tool-delete-recovery-command-title"
                  data-tool-delete-recovery-command
                >
                  <p id="tool-delete-recovery-command-title">
                    Run this command as an Administrator to finish cleanup:
                  </p>
                  <pre>
                    <code>{localRemovalCommand}</code>
                  </pre>
                  <button
                    ref={resultActionRef}
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void handleCopyCommand()}
                  >
                    Copy command
                  </button>
                  {copyError && <p role="alert">{copyError}</p>}
                  {copySuccess && (
                    <p role="status" aria-label="Command copied">
                      Command copied.
                    </p>
                  )}
                </div>
              )}
              {cleanupNeedsRecovery && !localRemovalCommand && localRemovalInstructions && (
                <div role="group" aria-labelledby="tool-delete-recovery-instructions-title">
                  <p id="tool-delete-recovery-instructions-title">Manual cleanup instructions:</p>
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
            <button
              ref={resultActionRef}
              type="button"
              className="btn btn-secondary"
              onClick={onClose}
            >
              Dismiss
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
