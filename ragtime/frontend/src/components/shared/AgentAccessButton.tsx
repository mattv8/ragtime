import { forwardRef } from 'react';
import { ArrowLeftRight } from 'lucide-react';

interface AgentAccessButtonProps {
  onClick: () => void;
  title?: string;
  disabled?: boolean;
  className?: string;
}

export const AgentAccessButton = forwardRef<HTMLButtonElement, AgentAccessButtonProps>(
  (
    {
      onClick,
      title = 'Manage agent access',
      disabled = false,
      className = 'btn btn-secondary btn-sm',
    },
    ref,
  ) => (
    <button
      ref={ref}
      type="button"
      className={className}
      onClick={onClick}
      title={title}
      aria-label={title}
      disabled={disabled}
    >
      <ArrowLeftRight size={14} />
    </button>
  ),
);

AgentAccessButton.displayName = 'AgentAccessButton';

export default AgentAccessButton;
