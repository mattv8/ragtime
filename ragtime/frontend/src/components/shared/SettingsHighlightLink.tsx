import type { MouseEvent, ReactNode } from 'react';

export const CONTENT_PROTECTION_SETTING_ID = 'content_protection';

interface SettingsHighlightLinkProps {
  settingId: string;
  onNavigate?: (settingId: string) => void;
  className?: string;
  children: ReactNode;
}

export function SettingsHighlightLink({
  settingId,
  onNavigate,
  className,
  children,
}: SettingsHighlightLinkProps): JSX.Element {
  const handleClick = (event: MouseEvent<HTMLAnchorElement>) => {
    if (
      !onNavigate ||
      event.button !== 0 ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey
    )
      return;
    event.preventDefault();
    onNavigate(settingId);
  };

  return (
    <a
      href={`?view=settings&highlight=${encodeURIComponent(settingId)}`}
      className={className}
      data-settings-highlight-link={settingId}
      onClick={handleClick}
    >
      {children}
    </a>
  );
}
