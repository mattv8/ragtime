import {
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
} from 'react';
import { createPortal } from 'react-dom';

import type { WorkspaceExternalApiCredentialItem } from '@/types';

export type ExternalApiCredentialDialogTokenState = {
  token: string;
  prefix: string;
  label: string;
  operation: 'Created' | 'Rotated';
  endpointPath?: string | null;
  method?: string | null;
};

export interface TokenDialogProps {
  workspaceId: string;
  tokenState: ExternalApiCredentialDialogTokenState;
  onClose: () => void;
}

export type CredentialAction = 'rotate' | 'revoke' | 'delete';

export interface CredentialConfirmDialogProps {
  action: CredentialAction;
  credential: WorkspaceExternalApiCredentialItem;
  isSubmitting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}

function getFocusableElements(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];
  return Array.from(
    container.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  );
}

function useDialogFocusTrap(
  containerRef: React.RefObject<HTMLDivElement | null>,
  initialFocusRef: React.RefObject<HTMLElement | null>,
  { disableEscape, onEscape }: { disableEscape?: boolean; onEscape?: () => void } = {},
) {
  const returnFocusRef = useRef<HTMLElement | null>(null);
  const disableEscapeRef = useRef(disableEscape);
  const onEscapeRef = useRef(onEscape);

  useEffect(() => {
    disableEscapeRef.current = disableEscape;
  }, [disableEscape]);

  useEffect(() => {
    onEscapeRef.current = onEscape;
  }, [onEscape]);

  useEffect(() => {
    returnFocusRef.current =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const focusTimer = window.setTimeout(() => {
      initialFocusRef.current?.focus();
    }, 0);

    const handleKeyDown = (event: KeyboardEvent) => {
      const container = containerRef.current;
      if (!container) return;

      if (event.key === 'Escape') {
        if (!disableEscapeRef.current) {
          event.preventDefault();
          onEscapeRef.current?.();
        }
        return;
      }

      if (event.key !== 'Tab') {
        return;
      }

      const focusableElements = getFocusableElements(container);
      if (focusableElements.length === 0) {
        event.preventDefault();
        return;
      }

      const activeElement = document.activeElement as HTMLElement | null;
      const currentIndex = activeElement ? focusableElements.indexOf(activeElement) : -1;
      const nextIndex = event.shiftKey
        ? currentIndex <= 0
          ? focusableElements.length - 1
          : currentIndex - 1
        : currentIndex === -1 || currentIndex === focusableElements.length - 1
          ? 0
          : currentIndex + 1;

      event.preventDefault();
      focusableElements[nextIndex]?.focus();
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.clearTimeout(focusTimer);
      document.removeEventListener('keydown', handleKeyDown);
      returnFocusRef.current?.focus();
    };
  }, [containerRef, initialFocusRef, disableEscapeRef, onEscapeRef]);
}

function buildWorkspaceEndpoint(workspaceId: string, endpointPath?: string | null): string {
  const normalizedPath = endpointPath && endpointPath !== '/' ? endpointPath : '/your-endpoint';
  return `${window.location.origin}/indexes/userspace/workspaces/${encodeURIComponent(workspaceId)}/external-api${normalizedPath}`;
}

function buildCurlExample(
  workspaceId: string,
  tokenState: ExternalApiCredentialDialogTokenState,
): string {
  return `curl -X ${tokenState.method ?? 'GET'} "${buildWorkspaceEndpoint(workspaceId, tokenState.endpointPath)}" -H "Authorization: Bearer ${tokenState.token}"`;
}

function buildPowerQueryExample(
  workspaceId: string,
  tokenState: ExternalApiCredentialDialogTokenState,
): string {
  return `let\n    Source = Json.Document(Web.Contents("${buildWorkspaceEndpoint(workspaceId, tokenState.endpointPath)}", [Headers=[Authorization="Bearer ${tokenState.token}"]]))\nin\n    Source`;
}

export function ExternalApiCredentialTokenDialog({
  workspaceId,
  tokenState,
  onClose,
}: TokenDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const copyButtonRef = useRef<HTMLButtonElement>(null);
  const curlTabRef = useRef<HTMLButtonElement>(null);
  const powerQueryTabRef = useRef<HTMLButtonElement>(null);
  const copiedTimerRef = useRef<number | null>(null);
  const [activeTab, setActiveTab] = useState<'curl' | 'power-query'>('curl');
  const [copied, setCopied] = useState(false);
  const curlPanelId = useId();
  const powerQueryPanelId = useId();

  useDialogFocusTrap(dialogRef, copyButtonRef, { disableEscape: true });

  useEffect(() => {
    return () => {
      if (copiedTimerRef.current !== null) {
        window.clearTimeout(copiedTimerRef.current);
      }
    };
  }, []);

  const curlExample = useMemo(
    () => buildCurlExample(workspaceId, tokenState),
    [workspaceId, tokenState],
  );
  const powerQueryExample = useMemo(
    () => buildPowerQueryExample(workspaceId, tokenState),
    [workspaceId, tokenState],
  );

  const moveTabSelection = (nextTab: 'curl' | 'power-query') => {
    setActiveTab(nextTab);
    if (nextTab === 'curl') {
      curlTabRef.current?.focus();
      return;
    }
    powerQueryTabRef.current?.focus();
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(tokenState.token);
    setCopied(true);
    if (copiedTimerRef.current !== null) {
      window.clearTimeout(copiedTimerRef.current);
    }
    copiedTimerRef.current = window.setTimeout(() => setCopied(false), 1200);
  };

  const handleTabKeyDown = (
    event: ReactKeyboardEvent<HTMLButtonElement>,
    currentTab: 'curl' | 'power-query',
  ) => {
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      moveTabSelection(currentTab === 'curl' ? 'power-query' : 'curl');
      return;
    }
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      moveTabSelection(currentTab === 'curl' ? 'power-query' : 'curl');
      return;
    }
    if (event.key === 'Home') {
      event.preventDefault();
      moveTabSelection('curl');
      return;
    }
    if (event.key === 'End') {
      event.preventDefault();
      moveTabSelection('power-query');
    }
  };

  return createPortal(
    <div
      className="userspace-external-api-dialog-backdrop"
      data-userspace-external-api-backdrop="token"
    >
      <div
        ref={dialogRef}
        id="userspace-external-api-token-dialog"
        className="userspace-external-api-dialog-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="userspace-external-api-token-dialog-title"
      >
        <div
          className="userspace-external-api-dialog-content"
          data-userspace-external-api-region="token-dialog"
        >
          <h3
            id="userspace-external-api-token-dialog-title"
            className="userspace-external-api-section-heading"
          >
            {tokenState.operation} credential
          </h3>
          <div
            className="userspace-external-api-dialog-metadata"
            data-userspace-external-api-region="token-metadata"
          >
            <span>{tokenState.label}</span>
            <code>{tokenState.prefix}</code>
          </div>
          <p>Copy this token now. It cannot be shown again.</p>
          <div
            className="userspace-external-api-dialog-example"
            data-userspace-external-api-region="token-value"
          >
            <label
              className="userspace-external-api-field-label"
              htmlFor="userspace-external-api-token-value"
            >
              Bearer token
            </label>
            <div className="userspace-share-url-copy-wrap">
              <pre
                id="userspace-external-api-token-value"
                className="userspace-external-api-dialog-code"
              >
                {tokenState.token}
              </pre>
              <button
                ref={copyButtonRef}
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => void handleCopy()}
              >
                {copied ? 'Copied' : 'Copy token'}
              </button>
            </div>
          </div>
          <div
            className="userspace-external-api-dialog-tabs"
            data-userspace-external-api-region="token-tabs"
            role="tablist"
            aria-label="Credential usage examples"
          >
            <button
              ref={curlTabRef}
              type="button"
              id="userspace-external-api-tab-curl"
              className="userspace-external-api-dialog-tab"
              role="tab"
              aria-selected={activeTab === 'curl'}
              aria-controls={curlPanelId}
              tabIndex={activeTab === 'curl' ? 0 : -1}
              onClick={() => setActiveTab('curl')}
              onKeyDown={(event) => handleTabKeyDown(event, 'curl')}
            >
              curl
            </button>
            <button
              ref={powerQueryTabRef}
              type="button"
              id="userspace-external-api-tab-power-query"
              className="userspace-external-api-dialog-tab"
              role="tab"
              aria-selected={activeTab === 'power-query'}
              aria-controls={powerQueryPanelId}
              tabIndex={activeTab === 'power-query' ? 0 : -1}
              onClick={() => setActiveTab('power-query')}
              onKeyDown={(event) => handleTabKeyDown(event, 'power-query')}
            >
              Power Query
            </button>
          </div>
          <div
            id={curlPanelId}
            className="userspace-external-api-dialog-example"
            data-userspace-external-api-region="token-tabpanel-curl"
            role="tabpanel"
            aria-labelledby="userspace-external-api-tab-curl"
            aria-label="curl"
            hidden={activeTab !== 'curl'}
          >
            <pre className="userspace-external-api-dialog-code">{curlExample}</pre>
          </div>
          <div
            id={powerQueryPanelId}
            className="userspace-external-api-dialog-example"
            data-userspace-external-api-region="token-tabpanel-power-query"
            role="tabpanel"
            aria-labelledby="userspace-external-api-tab-power-query"
            aria-label="Power Query"
            hidden={activeTab !== 'power-query'}
          >
            <pre className="userspace-external-api-dialog-code">{powerQueryExample}</pre>
          </div>
          <div
            className="userspace-external-api-dialog-actions"
            data-userspace-external-api-region="token-actions"
          >
            <button type="button" className="btn btn-primary" onClick={onClose}>
              I saved this token
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}

const CONFIRM_DIALOG_COPY: Record<
  CredentialAction,
  { title: string; description: string; confirmLabel: string; confirmClassName: string }
> = {
  rotate: {
    title: 'Rotate credential',
    description:
      'The current token will stop working immediately. You will need to update every client with the replacement token.',
    confirmLabel: 'Rotate token',
    confirmClassName: 'btn btn-primary',
  },
  revoke: {
    title: 'Revoke credential',
    description:
      'This credential will stop working immediately. You can keep it for audit purposes or delete the revoked record later.',
    confirmLabel: 'Revoke credential',
    confirmClassName: 'btn btn-danger',
  },
  delete: {
    title: 'Delete credential',
    description:
      'This permanently removes the revoked credential record. Request history and management audit history are preserved.',
    confirmLabel: 'Delete permanently',
    confirmClassName: 'btn btn-danger',
  },
};

export function ExternalApiCredentialConfirmDialog({
  action,
  credential,
  isSubmitting,
  onCancel,
  onConfirm,
}: CredentialConfirmDialogProps) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const cancelButtonRef = useRef<HTMLButtonElement>(null);
  const dialogCopy = CONFIRM_DIALOG_COPY[action];

  useDialogFocusTrap(dialogRef, cancelButtonRef, {
    disableEscape: isSubmitting,
    onEscape: onCancel,
  });

  return createPortal(
    <div
      className="userspace-external-api-dialog-backdrop"
      onClick={() => {
        if (!isSubmitting) {
          onCancel();
        }
      }}
    >
      <div
        ref={dialogRef}
        id="userspace-external-api-confirm-dialog"
        className="userspace-external-api-dialog-panel"
        role="dialog"
        aria-modal="true"
        aria-labelledby="userspace-external-api-confirm-dialog-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div
          className="userspace-external-api-dialog-content"
          data-userspace-external-api-region="confirm-dialog"
        >
          <h3
            id="userspace-external-api-confirm-dialog-title"
            className="userspace-external-api-section-heading"
          >
            {dialogCopy.title}
          </h3>
          <div
            className="userspace-external-api-dialog-metadata"
            data-userspace-external-api-region="confirm-metadata"
          >
            <span>{credential.label}</span>
            <code>{credential.token_prefix}</code>
          </div>
          <p>{dialogCopy.description}</p>
          <div
            className="userspace-external-api-dialog-actions"
            data-userspace-external-api-region="confirm-actions"
          >
            <button
              ref={cancelButtonRef}
              type="button"
              className="btn btn-secondary"
              disabled={isSubmitting}
              onClick={onCancel}
            >
              Cancel
            </button>
            <button
              type="button"
              className={dialogCopy.confirmClassName}
              disabled={isSubmitting}
              onClick={onConfirm}
            >
              {dialogCopy.confirmLabel}
            </button>
          </div>
        </div>
      </div>
    </div>,
    document.body,
  );
}
