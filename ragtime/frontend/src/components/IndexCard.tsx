import { ReactNode } from 'react';

interface IndexCardProps {
  title: string;
  enabled: boolean;
  onToggle: (enabled: boolean) => void;
  metaPills?: ReactNode;
  actions?: ReactNode;
  children?: ReactNode; // For description or other content in the info area
  className?: string;
  as?: 'li' | 'div';
  id?: string;
  toggleTitle?: string;
  titleChildren?: ReactNode; // Elements to render alongside the title
}

export function IndexCard({
  title,
  enabled,
  onToggle,
  metaPills,
  actions,
  children,
  className = '',
  as = 'div',
  id,
  toggleTitle,
  titleChildren,
}: IndexCardProps) {
  const Component = as;

  return (
    <Component
      className={`index-item ${!enabled ? 'index-disabled' : ''} ${className}`}
      id={id}
    >
      <div className="index-toggle">
        <label
          className="toggle-switch"
          title={toggleTitle || (enabled ? 'Enabled' : 'Disabled')}
        >
          <input
            type="checkbox"
            checked={enabled}
            onChange={(e) => onToggle(e.target.checked)}
          />
          <span className="toggle-slider"></span>
        </label>
      </div>
      <div className="index-info">
        <h3>
          {title}
          {titleChildren}
        </h3>
        {metaPills && <div className="index-meta-pills">{metaPills}</div>}
        {children}
      </div>
      {actions && <div className="index-actions">{actions}</div>}
    </Component>
  );
}
