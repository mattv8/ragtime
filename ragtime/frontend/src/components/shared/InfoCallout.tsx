import { Info } from 'lucide-react';
import type { ReactNode } from 'react';

interface InfoCalloutProps {
  id: string;
  children: ReactNode;
}

export function InfoCallout({ id, children }: InfoCalloutProps) {
  return (
    <div
      id={id}
      className="tool-access-callout tool-access-callout-info"
      role="note"
      data-info-callout
    >
      <Info size={14} aria-hidden="true" className="info-callout-icon" />
      <div className="info-callout-body">{children}</div>
    </div>
  );
}
