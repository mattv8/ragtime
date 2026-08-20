import type { CSSProperties, ReactNode } from 'react';

export interface ThemeChromeIconProps {
  fallback: ReactNode;
  codicon: string;
  size?: number;
  className?: string;
}

export function ThemeChromeIcon({ fallback, codicon, size = 16, className }: ThemeChromeIconProps) {
  const classes = ['theme-chrome-icon', className].filter(Boolean).join(' ');

  return (
    <span
      aria-hidden="true"
      className={classes}
      style={{ '--theme-chrome-icon-size': `${size}px` } as CSSProperties}
    >
      <span aria-hidden="true" className="theme-chrome-icon-fallback">
        {fallback}
      </span>
      <span aria-hidden="true" className={`theme-chrome-icon-codicon codicon codicon-${codicon}`} />
    </span>
  );
}
