import { useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { AuthStatus } from '@/types';
import { API_KEY_INFO_HIGHLIGHT, renderApiKeySecurityWarning, renderHttpSecurityWarning } from './shared/securityWarnings';

const DISMISS_KEY = 'ragtime_security_banner_dismissed';
const DISMISSED_NOTICES_KEY = 'ragtime_security_banner_dismissed_notices';
const BRANDING_NOTICE_KEY = 'ragtime_branding_restart_notice';

const NOTICE_API_KEY = 'api-key';
const NOTICE_CORS = 'cors';
const NOTICE_HTTP = 'http';
const NOTICE_BRANDING = 'branding-restart';

interface SecurityBannerProps {
  authStatus: AuthStatus | null;
  isAdmin: boolean;
  hidden?: boolean;
  onNavigateToSettings?: (highlightSetting?: string) => void;
}

interface NoticeItem {
  id: string;
  title: string;
  message: ReactNode;
  highlightSetting: string;
}

interface NoticeDefinition extends NoticeItem {
  visible: boolean;
}

interface NoticeBannerProps {
  notice: NoticeItem;
  onDismiss: (noticeId: string) => void;
  onNavigateToSettings?: (highlightSetting?: string) => void;
}

function NoticeBanner({ notice, onDismiss, onNavigateToSettings }: NoticeBannerProps) {
  return (
    <div className="security-banner">
      <div className="security-banner-content">
        <strong>{notice.title}:</strong>
        <span>{notice.message}</span>
        <div className="security-banner-actions">
          {onNavigateToSettings && (
            <button
              type="button"
              className="security-banner-link"
              onClick={() => onNavigateToSettings(notice.highlightSetting)}
            >
              View in Settings
            </button>
          )}
          <button
            type="button"
            className="security-banner-dismiss"
            onClick={() => onDismiss(notice.id)}
            title="Dismiss for this session"
          >
            Dismiss
          </button>
        </div>
      </div>
    </div>
  );
}

export function SecurityBanner({ authStatus, isAdmin, hidden, onNavigateToSettings }: SecurityBannerProps) {
  const [dismissedNoticeIds, setDismissedNoticeIds] = useState<string[]>([]);
  const [showBrandingNotice, setShowBrandingNotice] = useState(false);

  // Check sessionStorage on mount
  useEffect(() => {
    const storedDismissedNotices = sessionStorage.getItem(DISMISSED_NOTICES_KEY);
    if (storedDismissedNotices) {
      try {
        const parsed = JSON.parse(storedDismissedNotices);
        if (Array.isArray(parsed)) {
          setDismissedNoticeIds(parsed.filter((n): n is string => typeof n === 'string'));
        }
      } catch {
        // Ignore invalid session storage payload.
      }
    } else if (sessionStorage.getItem(DISMISS_KEY) === 'true') {
      // Migrate legacy one-shot dismissal key without suppressing branding notice.
      const migrated = [NOTICE_API_KEY, NOTICE_CORS, NOTICE_HTTP];
      setDismissedNoticeIds(migrated);
      sessionStorage.setItem(DISMISSED_NOTICES_KEY, JSON.stringify(migrated));
      sessionStorage.removeItem(DISMISS_KEY);
    }

    const brandingNoticePending = sessionStorage.getItem(BRANDING_NOTICE_KEY) === 'true';
    setShowBrandingNotice(brandingNoticePending);

    const handleBrandingNoticeUpdate = () => {
      const brandingNoticePendingNow = sessionStorage.getItem(BRANDING_NOTICE_KEY) === 'true';
      setShowBrandingNotice(brandingNoticePendingNow);
      if (brandingNoticePendingNow) {
        setDismissedNoticeIds((prev) => {
          if (!prev.includes(NOTICE_BRANDING)) {
            return prev;
          }
          const next = prev.filter((id) => id !== NOTICE_BRANDING);
          sessionStorage.setItem(DISMISSED_NOTICES_KEY, JSON.stringify(next));
          return next;
        });
      }
    };

    window.addEventListener('ragtime:branding-notice-updated', handleBrandingNoticeUpdate);

    return () => {
      window.removeEventListener('ragtime:branding-notice-updated', handleBrandingNoticeUpdate);
    };
  }, []);

  // Don't show banner if we don't have auth status yet
  if (!authStatus) return null;

  // Only show to admins - regular users can't fix these issues
  if (!isAdmin) return null;

  // Hide when userspace is fullscreen
  if (hidden) return null;

  // Check security issues
  const showApiKeyWarning = !authStatus.api_key_configured;
  const showCorsWarning = authStatus.allowed_origins_open;
  const isHttp = window.location.protocol === 'http:';

  const securityNoticeDefinitions: NoticeDefinition[] = [
    {
      id: NOTICE_API_KEY,
      title: 'Security',
      message: renderApiKeySecurityWarning(),
      highlightSetting: API_KEY_INFO_HIGHLIGHT,
      visible: showApiKeyWarning && !dismissedNoticeIds.includes(NOTICE_API_KEY),
    },
    {
      id: NOTICE_CORS,
      title: 'Security',
      message: (
        <>
          <code>ALLOWED_ORIGINS=*</code> allows requests from any website.
          Consider restricting to specific domains.
        </>
      ),
      highlightSetting: API_KEY_INFO_HIGHLIGHT,
      visible: showCorsWarning && !dismissedNoticeIds.includes(NOTICE_CORS),
    },
    {
      id: NOTICE_HTTP,
      title: 'Security',
      message: renderHttpSecurityWarning(),
      highlightSetting: API_KEY_INFO_HIGHLIGHT,
      visible: isHttp && !dismissedNoticeIds.includes(NOTICE_HTTP),
    },
  ];

  const securityNotices = securityNoticeDefinitions
    .filter((notice) => notice.visible)
    .map(({ visible: _visible, ...notice }) => notice);

  const brandingNotice: NoticeItem | null =
    showBrandingNotice && !dismissedNoticeIds.includes(NOTICE_BRANDING)
      ? {
          id: NOTICE_BRANDING,
          title: 'Branding',
          message: (
            <>
              Server branding changed: UI updates immediately, but restart Ragtime to fully apply MCP server identity changes.
            </>
          ),
          highlightSetting: 'server_branding',
        }
      : null;

  if (!securityNotices.length && !brandingNotice) return null;

  const dismissNotice = (noticeId: string) => {
    if (dismissedNoticeIds.includes(noticeId)) {
      return;
    }

    const nextDismissed = [...dismissedNoticeIds, noticeId];
    setDismissedNoticeIds(nextDismissed);
    sessionStorage.setItem(DISMISSED_NOTICES_KEY, JSON.stringify(nextDismissed));

    if (noticeId === NOTICE_BRANDING) {
      sessionStorage.removeItem(BRANDING_NOTICE_KEY);
      setShowBrandingNotice(false);
    }
  };

  return (
    <>
      {securityNotices.map((notice) => (
        <NoticeBanner
          key={notice.id}
          notice={notice}
          onDismiss={dismissNotice}
          onNavigateToSettings={onNavigateToSettings}
        />
      ))}
      {brandingNotice && (
        <NoticeBanner
          notice={brandingNotice}
          onDismiss={dismissNotice}
          onNavigateToSettings={onNavigateToSettings}
        />
      )}
    </>
  );
}
