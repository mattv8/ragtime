import type { CSSProperties } from 'react';
import claudeIcon from '@lobehub/icons-static-svg/icons/claude.svg?url';
import claudeCodeIcon from '@lobehub/icons-static-svg/icons/claudecode.svg?url';
import clineIcon from '@lobehub/icons-static-svg/icons/cline.svg?url';
import codexIcon from '@lobehub/icons-static-svg/icons/codex.svg?url';
import cursorIcon from '@lobehub/icons-static-svg/icons/cursor.svg?url';
import hermesIcon from '@lobehub/icons-static-svg/icons/hermesagent.svg?url';
import openAIIcon from '@lobehub/icons-static-svg/icons/openai.svg?url';
import openCodeIcon from '@lobehub/icons-static-svg/icons/opencode.svg?url';
import { X } from 'lucide-react';

import { CODING_AGENT_CLIENTS, type CodingAgentClientId } from './codingAgentClients';
import './UserSpaceAgentOnboardingRail.css';

interface UserSpaceAgentOnboardingRailProps {
  onSelectClient: (clientId: CodingAgentClientId, trigger: HTMLButtonElement) => void;
  onDismiss: () => void;
}

const CLIENT_ICON_URLS: Record<Exclude<CodingAgentClientId, 'continue'>, string> = {
  'claude-desktop': claudeIcon,
  'claude-code': claudeCodeIcon,
  cursor: cursorIcon,
  chatgpt: openAIIcon,
  cline: clineIcon,
  opencode: openCodeIcon,
  hermes: hermesIcon,
  codex: codexIcon,
};

// Icon path source: https://github.com/continuedev/continue/blob/main/gui/src/components/svg/ContinueLogo.tsx
function ContinueIcon() {
  return (
    <svg
      className="userspace-agent-onboarding-brand-icon"
      viewBox="82 82 146 136"
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M199.142 100.633L191.304 114.263L211.044 148.774C211.334 149.065 211.334 149.355 211.334 149.645C211.334 149.935 211.334 150.225 211.044 150.515L191.304 185.026L199.142 198.656L227.3 149.645L199.142 100.633ZM188.401 112.523L196.239 98.8928H180.563L172.725 112.523H188.401ZM172.435 116.003L190.723 147.904H206.399L188.111 116.003H172.435ZM188.401 182.996L206.689 151.095H191.014L172.725 182.996H188.401ZM172.435 186.766L180.273 200.396H195.949L188.111 186.766H172.435ZM119.311 203.586C119.021 203.586 118.73 203.586 118.44 203.296C118.15 203.006 117.859 203.006 117.859 202.716L97.8292 168.205H82.1533L110.312 217.217H166.919L159.081 203.586H119.311ZM161.984 201.846L169.822 215.477L177.66 201.846L169.822 188.216L161.984 201.846ZM166.919 186.476H130.052L122.214 200.106H158.791L166.919 186.476ZM127.149 184.736L108.86 152.835L101.022 166.465L119.311 198.366L127.149 184.736ZM82.1533 164.725H97.8292L105.667 151.095H89.9913L82.1533 164.725ZM117.569 96.5728C117.569 96.2827 117.859 95.9927 118.15 95.9927C118.44 95.9927 118.73 95.7027 119.021 95.7027H158.791L166.629 82.0723H110.312L82.1533 131.084H97.8292L117.569 96.5728ZM105.667 148.194L97.8292 134.564H82.1533L89.9913 148.194H105.667ZM119.021 100.923L100.732 132.824L108.57 146.454L126.859 114.553L119.021 100.923ZM159.081 99.1828H122.214L130.052 112.813H166.919L159.081 99.1828ZM169.822 111.073L177.66 97.4428L169.822 83.8123L161.984 97.4428L169.822 111.073Z" />
    </svg>
  );
}

export function UserSpaceAgentOnboardingRail({
  onSelectClient,
  onDismiss,
}: UserSpaceAgentOnboardingRailProps) {
  return (
    <section id="userspace-agent-onboarding-rail" className="userspace-agent-onboarding-rail">
      <button
        type="button"
        className="userspace-agent-onboarding-label"
        title="Set up OpenCode"
        aria-label="Connect your agent"
        onClick={(event) => onSelectClient('opencode', event.currentTarget)}
      >
        Connect your agent
      </button>
      <div className="userspace-agent-onboarding-actions">
        {CODING_AGENT_CLIENTS.map((client) => {
          const iconUrl = client.id === 'continue' ? undefined : CLIENT_ICON_URLS[client.id];
          return (
            <button
              key={client.id}
              type="button"
              className="userspace-agent-onboarding-client"
              title={`Set up ${client.label}`}
              aria-label={`Set up ${client.label}`}
              onClick={(event) => onSelectClient(client.id, event.currentTarget)}
            >
              {client.id === 'continue' ? (
                <ContinueIcon />
              ) : (
                <span
                  className="userspace-agent-onboarding-brand-icon userspace-agent-onboarding-brand-icon-mask"
                  aria-hidden="true"
                  style={{ '--provider-icon-url': `url("${iconUrl}")` } as CSSProperties}
                />
              )}
            </button>
          );
        })}
      </div>
      <button
        type="button"
        className="userspace-agent-onboarding-dismiss"
        title="Dismiss agent setup"
        aria-label="Dismiss agent setup"
        onClick={onDismiss}
      >
        <X size={14} aria-hidden="true" />
      </button>
    </section>
  );
}
