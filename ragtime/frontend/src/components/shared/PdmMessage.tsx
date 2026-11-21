import type { ReactNode } from 'react';
import { AlertTriangle, Info, XCircle, type LucideIcon } from 'lucide-react';

export type PdmMessageTone = 'info' | 'warning' | 'error';

export const MESSAGE_ICON: Record<PdmMessageTone, LucideIcon> = {
  info: Info,
  warning: AlertTriangle,
  error: XCircle,
};

export function PdmMessage({
  id,
  tone,
  boxed,
  alert,
  children,
}: {
  id?: string;
  tone: PdmMessageTone;
  boxed?: boolean;
  alert?: boolean;
  children: ReactNode;
}) {
  const Icon = MESSAGE_ICON[tone];
  return (
    <p
      id={id}
      className="sw-message"
      data-tone={tone}
      data-boxed={boxed ? 'true' : undefined}
      role={alert ? 'alert' : undefined}
    >
      <Icon className="sw-message-icon" size={14} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}
