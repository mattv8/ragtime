import { useState, useCallback, useRef, useEffect } from 'react';
import { createPortal } from 'react-dom';

type ToastType = 'success' | 'error';

interface ToastItem {
  id: number;
  type: ToastType;
  message: string;
}

interface ToastActions {
  success: (message: string, durationMs?: number) => void;
  error: (message: string, durationMs?: number) => void;
  dismiss: (id: number) => void;
  clear: () => void;
}

const DEFAULT_SUCCESS_DURATION = 3000;
const DEFAULT_ERROR_DURATION = 8000;

export function useToast(): [ToastItem[], ToastActions] {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const nextId = useRef(0);
  const timers = useRef<Map<number, ReturnType<typeof setTimeout>>>(new Map());

  const dismiss = useCallback((id: number) => {
    const timer = timers.current.get(id);
    if (timer) {
      clearTimeout(timer);
      timers.current.delete(id);
    }
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const add = useCallback((type: ToastType, message: string, durationMs?: number) => {
    const id = nextId.current++;
    setToasts((prev) => [...prev, { id, type, message }]);
    const timeout = durationMs ?? (type === 'success' ? DEFAULT_SUCCESS_DURATION : DEFAULT_ERROR_DURATION);
    const timer = setTimeout(() => {
      timers.current.delete(id);
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, timeout);
    timers.current.set(id, timer);
  }, []);

  const success = useCallback((message: string, durationMs?: number) => add('success', message, durationMs), [add]);
  const error = useCallback((message: string, durationMs?: number) => add('error', message, durationMs), [add]);

  const clear = useCallback(() => {
    for (const timer of timers.current.values()) clearTimeout(timer);
    timers.current.clear();
    setToasts([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    const ref = timers.current;
    return () => {
      for (const timer of ref.values()) clearTimeout(timer);
    };
  }, []);

  return [toasts, { success, error, dismiss, clear }];
}

interface ToastContainerProps {
  toasts: ToastItem[];
  onDismiss: (id: number) => void;
}

export function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  if (toasts.length === 0) return null;

  return createPortal(
    <div className="toast-container" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast-item toast-${toast.type}`} role="alert">
          <span className="toast-message">{toast.message}</span>
          <button
            type="button"
            className="toast-dismiss"
            onClick={() => onDismiss(toast.id)}
            aria-label="Dismiss"
          >
            &times;
          </button>
        </div>
      ))}
    </div>,
    document.body,
  );
}
