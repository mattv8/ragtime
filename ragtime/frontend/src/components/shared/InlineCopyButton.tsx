import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, Copy } from 'lucide-react';

type CopySource = string | (() => string | null | undefined | Promise<string | null | undefined>);

interface InlineCopyButtonProps {
  copyText: CopySource;
  className: string;
  title: string;
  ariaLabel: string;
  copiedTitle?: string;
  copiedAriaLabel?: string;
  disabled?: boolean;
  iconSize?: number;
  feedbackMs?: number;
  onCopySuccess?: () => void;
  onCopyError?: (error: Error) => void;
}

async function writeTextToClipboard(value: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(value);
    return;
  }

  const textarea = document.createElement('textarea');
  textarea.value = value;
  textarea.setAttribute('readonly', '');
  textarea.style.position = 'absolute';
  textarea.style.left = '-9999px';
  document.body.appendChild(textarea);
  textarea.select();

  try {
    const copied = document.execCommand('copy');
    if (!copied) {
      throw new Error('Copy command was rejected');
    }
  } finally {
    document.body.removeChild(textarea);
  }
}

export function InlineCopyButton({
  copyText,
  className,
  title,
  ariaLabel,
  copiedTitle,
  copiedAriaLabel,
  disabled = false,
  iconSize = 14,
  feedbackMs = 1500,
  onCopySuccess,
  onCopyError,
}: InlineCopyButtonProps) {
  const [copied, setCopied] = useState(false);
  const resetTimerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimerRef.current !== null) {
        window.clearTimeout(resetTimerRef.current);
      }
    };
  }, []);

  const handleClick = useCallback(async () => {
    if (disabled) {
      return;
    }

    try {
      const value = typeof copyText === 'function' ? await copyText() : copyText;
      if (!value) {
        return;
      }

      await writeTextToClipboard(value);
      onCopySuccess?.();
      setCopied(true);
      if (resetTimerRef.current !== null) {
        window.clearTimeout(resetTimerRef.current);
      }
      resetTimerRef.current = window.setTimeout(() => {
        setCopied(false);
        resetTimerRef.current = null;
      }, feedbackMs);
    } catch (error) {
      onCopyError?.(error instanceof Error ? error : new Error('Failed to copy'));
    }
  }, [copyText, disabled, feedbackMs, onCopyError, onCopySuccess]);

  return (
    <button
      type="button"
      className={`${className} ${copied ? 'is-copied' : ''}`.trim()}
      onClick={() => {
        void handleClick();
      }}
      disabled={disabled}
      title={copied ? (copiedTitle ?? title) : title}
      aria-label={copied ? (copiedAriaLabel ?? ariaLabel) : ariaLabel}
    >
      {copied ? <Check size={iconSize} /> : <Copy size={iconSize} />}
    </button>
  );
}