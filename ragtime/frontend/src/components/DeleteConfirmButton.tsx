import { useState, useEffect } from 'react';

interface DeleteConfirmButtonProps {
  onDelete: () => void;
  disabled?: boolean;
  className?: string;
  title?: string;
  buttonText?: string;
  deleting?: boolean;
}

export function DeleteConfirmButton({
  onDelete,
  disabled = false,
  className = "btn btn-sm btn-danger",
  title = "Delete",
  buttonText = "Delete",
  deleting = false
}: DeleteConfirmButtonProps) {
  const [confirming, setConfirming] = useState(false);
  const [countdown, setCountdown] = useState(0);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    if (confirming && countdown > 0) {
      timer = setTimeout(() => {
        setCountdown((prev) => prev - 1);
      }, 1000);
    }
    return () => clearTimeout(timer);
  }, [confirming, countdown]);

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();

    // Ignore clicks during countdown
    if (confirming && countdown > 0) return;

    if (!confirming) {
      setConfirming(true);
      setCountdown(3);
    } else {
      onDelete();
      // Optional: don't reset confirming immediately if we are about to unmount or show deleting state
      // But if operation fails, we might want to reset?
      // For now, parent controlling 'deleting' prop handles the visual feedback
      setConfirming(false);
    }
  };

  return (
    <button
      type="button"
      className={className}
      onClick={handleClick}
      // Don't disable during countdown so we keep focus (and can detect onBlur)
      disabled={disabled || deleting}
      title={confirming ? "Click to confirm deletion" : title}
      style={{
        minWidth: confirming ? '100px' : undefined,
        ...(confirming && countdown > 0 ? { opacity: 0.65, cursor: 'not-allowed' } : {})
      }}
      onBlur={() => {
        setConfirming(false);
      }}
    >
      {deleting
        ? 'Deleting...'
        : (confirming
            ? (countdown > 0 ? `Confirm? (${countdown})` : 'Confirm?')
            : buttonText
          )
      }
    </button>
  );
}
