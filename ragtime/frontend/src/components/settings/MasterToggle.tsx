import type { ReactNode } from 'react';

export interface MasterToggleProps {
  settingId?: string;
  inputId: string;
  label: string;
  help?: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  className?: string;
}

/** Standard master toggle for settings sections. */
export function MasterToggle({
  settingId,
  inputId,
  label,
  help,
  checked,
  onChange,
  disabled,
  className,
}: MasterToggleProps): JSX.Element {
  const helpId = help != null ? `${inputId}-help` : undefined;

  return (
    <div className={'form-group master-toggle' + (className ? ` ${className}` : '')} id={settingId}>
      <label className="master-toggle-control" htmlFor={inputId}>
        <span className="toggle-switch">
          <input
            id={inputId}
            type="checkbox"
            role="switch"
            aria-describedby={helpId}
            checked={checked}
            disabled={disabled}
            onChange={(e) => onChange(e.target.checked)}
          />
          <span className="toggle-slider" />
        </span>
        <span className="master-toggle-label">{label}</span>
      </label>
      {help != null && (
        <p className="field-help" id={helpId}>
          {help}
        </p>
      )}
    </div>
  );
}
