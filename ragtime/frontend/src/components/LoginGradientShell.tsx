import type { HTMLAttributes } from 'react';

import WebGLGradient from './WebGLGradient';

export function LoginGradientShell({
  className = '',
  children,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  const classes = ['login-container', 'login-gradient-container', className]
    .filter(Boolean)
    .join(' ');

  return (
    <div data-auth-surface="gradient" className={classes} {...props}>
      <WebGLGradient className="login-background-gradient" fullscreen />
      {children}
    </div>
  );
}
