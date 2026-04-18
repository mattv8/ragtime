import { ArrowLeftRight } from 'lucide-react';

interface AgentAccessButtonProps {
  onClick: () => void;
  title?: string;
  disabled?: boolean;
  className?: string;
}

export function AgentAccessButton({
  onClick,
  title = 'Manage agent access',
  disabled = false,
  className = 'btn btn-secondary btn-sm',
}: AgentAccessButtonProps) {
  return (
    <button type="button" className={className} onClick={onClick} title={title} disabled={disabled}>
      <ArrowLeftRight size={14} />
    </button>
  );
}

export default AgentAccessButton;
