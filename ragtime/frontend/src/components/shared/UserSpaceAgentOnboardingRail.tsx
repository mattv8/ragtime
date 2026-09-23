import {
  Bot,
  Braces,
  Code2,
  Layers3,
  MessageSquare,
  Monitor,
  MousePointer2,
  Sparkles,
  Terminal,
  X,
  type LucideIcon,
} from 'lucide-react';

import { CODING_AGENT_CLIENTS, type CodingAgentClientId } from './codingAgentClients';
import './UserSpaceAgentOnboardingRail.css';

interface UserSpaceAgentOnboardingRailProps {
  onSelectClient: (clientId: CodingAgentClientId, trigger: HTMLButtonElement) => void;
  onDismiss: () => void;
}

const CLIENT_ICONS: Record<CodingAgentClientId, LucideIcon> = {
  'claude-desktop': Monitor,
  'claude-code': Sparkles,
  cursor: MousePointer2,
  chatgpt: MessageSquare,
  cline: Bot,
  continue: Layers3,
  opencode: Code2,
  hermes: Braces,
  codex: Terminal,
};

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
          const Icon = CLIENT_ICONS[client.id];
          return (
            <button
              key={client.id}
              type="button"
              className="userspace-agent-onboarding-client"
              title={`Set up ${client.label}`}
              aria-label={`Set up ${client.label}`}
              onClick={(event) => onSelectClient(client.id, event.currentTarget)}
            >
              <Icon size={16} aria-hidden="true" />
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
