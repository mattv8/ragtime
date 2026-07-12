import type { ReactNode } from 'react';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

interface SettingsAccordionSectionProps {
  id: SettingsAccordionSectionId;
  title: ReactNode;
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  status?: ReactNode;
  className?: string;
  children: ReactNode;
}

export function SettingsAccordionSection(props: SettingsAccordionSectionProps): JSX.Element {
  const { id, title, open, onToggle, status, className = '', children } = props;
  const bodyId = `settings-accordion-body-${id}`;

  return (
    <section
      className={`settings-accordion-item ${open ? 'settings-accordion-item--open' : ''} ${className}`.trim()}
      data-settings-accordion-section={id}
      data-settings-accordion-open={open ? 'true' : 'false'}
    >
      <button
        type="button"
        className="settings-accordion-header"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => onToggle(id)}
      >
        <span className="settings-accordion-header-main">
          <span className="settings-accordion-title">{title}</span>
          {status != null && <span className="settings-accordion-header-status">{status}</span>}
        </span>
        <span className="settings-accordion-chevron" aria-hidden="true">
          &gt;
        </span>
      </button>
      <div id={bodyId} className="settings-accordion-body" hidden={!open}>
        {children}
      </div>
    </section>
  );
}
