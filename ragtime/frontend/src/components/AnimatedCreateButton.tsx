import { X } from 'lucide-react';

interface AnimatedCreateButtonProps {
  isExpanded: boolean;
  onClick: () => void;
  label: string;
  className?: string;
}

export function AnimatedCreateButton({ isExpanded, onClick, label, className = '' }: AnimatedCreateButtonProps) {
  if (isExpanded) {
    return (
      <button
        type="button"
        className={`close-btn ${className}`}
        onClick={onClick}
        aria-label="Close"
      >
        <X size={18} />
      </button>
    );
  }

  return (
    <button
      type="button"
      className={`btn ${className}`}
      onClick={onClick}
    >
      {label}
    </button>
  );
}
