import { Users } from 'lucide-react';

interface MemberManagementButtonProps {
  onClick: () => void;
  title?: string;
  disabled?: boolean;
  className?: string;
}

export function MemberManagementButton({
  onClick,
  title = 'Manage members',
  disabled = false,
  className = 'btn btn-secondary btn-sm',
}: MemberManagementButtonProps) {
  return (
    <button type="button" className={className} onClick={onClick} title={title} disabled={disabled}>
      <Users size={14} />
    </button>
  );
}

export default MemberManagementButton;