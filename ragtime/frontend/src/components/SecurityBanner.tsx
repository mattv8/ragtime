import { useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import type { AuthStatus } from '@/types';
import {
  API_KEY_INFO_HIGHLIGHT,
  hasAuthenticatedSecurityPosture,
  renderApiKeySecurityWarning,
  renderHttpSecurityWarning,
  renderRuntimeAuthSecurityWarning,
} from './shared/securityWarnings';

const DISMISS_KEY = 'ragtime_security_banner_dismissed';
const DISMISSED_NOTICES_KEY = 'ragtime_security_banner_dismissed_notices';
const BRANDING_NOTICE_KEY = 'ragtime_branding_restart_notice';

const NOTICE_API_KEY = 'api-key';
const NOTICE_CORS = 'cors';
const NOTICE_HTTP = 'http';
const NOTICE_RUNTIME_AUTH = 'runtime-auth';
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

function readDismissedNoticeIds(): string[] {
  if (typeof window === 'undefined') return [];

  try {
    const storedDismissedNotices = window.sessionStorage.getItem(DISMISSED_NOTICES_KEY);
    if (storedDismissedNotices) {
      const parsed = JSON.parse(storedDismissedNotices);
      return Array.isArray(parsed) ? parsed.filter((n): n is string => typeof n === 'string') : [];
    }

    // Return the migrated set synchronously to avoid a first-render flicker,
    // but do not write to storage here; writes happen in a mount effect.
    if (window.sessionStorage.getItem(DISMISS_KEY) === 'true') {
      return [NOTICE_API_KEY, NOTICE_CORS, NOTICE_HTTP];
    }
  } catch {
    // Treat unavailable or invalid session storage as no dismissed notices.
  }

  return [];
}

function readBrandingNoticePending(): boolean {
  if (typeof window === 'undefined') return false;

  try {
    return window.sessionStorage.getItem(BRANDING_NOTICE_KEY) === 'true';
  } catch {
    return false;
  }
}

function writeDismissedNoticeIds(noticeIds: string[]): void {
  if (typeof window === 'undefined') return;

  try {
    window.sessionStorage.setItem(DISMISSED_NOTICES_KEY, JSON.stringify(noticeIds));
  } catch {
    // Ignore unavailable storage.
  }
}

function clearBrandingNoticePending(): void {
  if (typeof window === 'undefined') return;

  try {
    window.sessionStorage.removeItem(BRANDING_NOTICE_KEY);
  } catch {
    // Ignore unavailable storage.
  }
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

export function SecurityBanner({
  authStatus,
  isAdmin,
  hidden,
  onNavigateToSettings,
}: SecurityBannerProps) {
  const [dismissedNoticeIds, setDismissedNoticeIds] = useState<string[]>(readDismissedNoticeIds);
  const [showBrandingNotice, setShowBrandingNotice] = useState(readBrandingNoticePending);

  // Persist legacy migration to storage on mount so it happens outside render.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    try {
      if (window.sessionStorage.getItem(DISMISS_KEY) === 'true') {
        if (window.sessionStorage.getItem(DISMISSED_NOTICES_KEY) === null) {
          window.sessionStorage.setItem(
            DISMISSED_NOTICES_KEY,
            JSON.stringify([NOTICE_API_KEY, NOTICE_CORS, NOTICE_HTTP]),
          );
        }
        window.sessionStorage.removeItem(DISMISS_KEY);
      }
    } catch {
      // Ignore unavailable storage.
    }
  }, []);

  // Keep branding notice state in sync after Settings changes it at runtime.
  useEffect(() => {
    const handleBrandingNoticeUpdate = () => {
      const brandingNoticePendingNow = readBrandingNoticePending();
      setShowBrandingNotice(brandingNoticePendingNow);
      if (brandingNoticePendingNow) {
        const storedDismissed = readDismissedNoticeIds();
        if (storedDismissed.includes(NOTICE_BRANDING)) {
          writeDismissedNoticeIds(storedDismissed.filter((id) => id !== NOTICE_BRANDING));
        }
        setDismissedNoticeIds((prev) => {
          if (!prev.includes(NOTICE_BRANDING)) {
            return prev;
          }
          return prev.filter((id) => id !== NOTICE_BRANDING);
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
  const hasAuthenticatedPosture = hasAuthenticatedSecurityPosture(authStatus);
  const showApiKeyWarning = hasAuthenticatedPosture && !authStatus.api_key_configured;
  const showCorsWarning = hasAuthenticatedPosture && authStatus.allowed_origins_open;
  const showRuntimeAuthWarning =
    hasAuthenticatedPosture && Boolean(authStatus.runtime_auth_token_warning);
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
          <code>ALLOWED_ORIGINS=*</code> allows requests from any website. Consider restricting to
          specific domains.
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
    {
      id: NOTICE_RUNTIME_AUTH,
      title: 'Security',
      message: renderRuntimeAuthSecurityWarning(),
      highlightSetting: API_KEY_INFO_HIGHLIGHT,
      visible: showRuntimeAuthWarning && !dismissedNoticeIds.includes(NOTICE_RUNTIME_AUTH),
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
              Server branding changed: UI updates immediately, but restart Ragtime to fully apply
              MCP server identity changes.
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
    writeDismissedNoticeIds(nextDismissed);

    if (noticeId === NOTICE_BRANDING) {
      clearBrandingNoticePending();
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
