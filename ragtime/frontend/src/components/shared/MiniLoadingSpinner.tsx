import { Loader2 } from 'lucide-react';

interface MiniLoadingSpinnerProps {
  /** 'css' renders the CSS border spinner; 'icon' renders the Loader2 SVG spinner. Defaults to 'css'. */
  variant?: 'css' | 'icon';
  /** Icon size in pixels. Only used when variant='icon'. Defaults to 14. */
  size?: number;
  className?: string;
  title?: string;
  ariaHidden?: boolean;
}

export function MiniLoadingSpinner({ variant = 'css', size, className, title, ariaHidden }: MiniLoadingSpinnerProps) {
  if (variant === 'icon') {
    const iconClassName = ['userspace-icon-spin', className].filter(Boolean).join(' ');
    const icon = <Loader2 size={size ?? 14} className={iconClassName} aria-hidden={ariaHidden} />;
    return title ? <span className="mini-loading-spinner-icon-wrapper" title={title}>{icon}</span> : icon;
  }
  const spinnerClassName = ['userspace-toolbar-live-spinner', className].filter(Boolean).join(' ');
  return <span className={spinnerClassName} title={title} aria-hidden={ariaHidden} />;
}
