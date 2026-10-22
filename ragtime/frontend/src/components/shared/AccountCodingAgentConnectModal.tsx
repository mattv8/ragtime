import { useRef, useState } from 'react';
import { ModalTabs, type ModalTab } from './ModalTabs';
import { useDialogFocusTrap } from '@/hooks/useDialogFocusTrap';
import {
  CodingAgentOAuthConnection,
  type OAuthCodingAgentClientId,
} from './CodingAgentOAuthConnection';
import { CopyableSnippet } from './CodingAgentSetupPrimitives';
import './CodingAgentSetup.css';

interface AccountCodingAgentConnectModalProps {
  isOpen: boolean;
  onClose: () => void;
  mcpUrl: string;
}

const DISCOVERY_PROMPT =
  'Call workspace_development_workspaces to list workspaces I can access. Report their names and full IDs. Do not create a workspace, change files, or run commands.';

export function AccountCodingAgentConnectModal({
  isOpen,
  onClose,
  mcpUrl,
}: AccountCodingAgentConnectModalProps) {
  if (!isOpen) return null;
  return <AccountCodingAgentConnectDialog onClose={onClose} mcpUrl={mcpUrl} />;
}

function AccountCodingAgentConnectDialog({
  onClose,
  mcpUrl,
}: Omit<AccountCodingAgentConnectModalProps, 'isOpen'>) {
  const [activeTab, setActiveTab] = useState<OAuthCodingAgentClientId>('claude-desktop');
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  useDialogFocusTrap(dialogRef, closeButtonRef, { onEscape: onClose });

  const tabs: ModalTab[] = [
    ['claude-desktop', 'Claude Cowork / Desktop'],
    ['chatgpt', 'ChatGPT'],
    ['opencode', 'OpenCode V2'],
  ].map(([clientId, label]) => ({
    id: clientId,
    label,
    content: (
      <CodingAgentOAuthConnection clientId={clientId as OAuthCodingAgentClientId} mcpUrl={mcpUrl} />
    ),
  }));

  return (
    <div
      className="modal-overlay"
      onClick={onClose}
      data-userspace-modal="account-coding-agent-connect"
    >
      <div
        ref={dialogRef}
        id="userspace-account-coding-agent-connect-modal"
        className="modal-content modal-small modal-with-tabs userspace-connect-agent"
        role="dialog"
        aria-modal="true"
        aria-labelledby="userspace-account-coding-agent-connect-title"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id="userspace-account-coding-agent-connect-title">Connect your account agent</h3>
          <button
            ref={closeButtonRef}
            type="button"
            className="modal-close"
            onClick={onClose}
            aria-label="Close account agent connection"
          >
            &times;
          </button>
        </div>
        <div className="modal-body" data-userspace-panel="account-agent-connection">
          <p>
            Connect first. Connecting does not create a workspace; create one only when you
            explicitly ask your agent to do so.
          </p>
          <ModalTabs
            idPrefix="userspace-account-agent-connect"
            label="Account agent client"
            tabs={tabs}
            activeTabId={activeTab}
            onChange={(tabId) => setActiveTab(tabId as OAuthCodingAgentClientId)}
          />
          <section
            id="userspace-account-agent-discovery-prompt"
            className="coding-agent-guide-section"
            aria-label="Read-only workspace discovery"
          >
            <h4>Verify account access</h4>
            <CopyableSnippet value={DISCOVERY_PROMPT} label="Copy read-only discovery prompt" />
          </section>
        </div>
      </div>
    </div>
  );
}
