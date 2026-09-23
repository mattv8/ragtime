import { Suspense, lazy, useState, useEffect, useLayoutEffect, useCallback, useRef } from 'react';
import { MoreHorizontal, Waves } from 'lucide-react';
import { api, apiFetch, isResponseAuthContextCurrent } from '@/api';
import WebGLGradient from '@/components/WebGLGradient';
import { ConfigurationBanner } from './components/ConfigurationBanner';
import { LoginGradientShell } from './components/LoginGradientShell';
import { LoginPage } from './components/LoginPage';
import { MemoryStatus } from './components/MemoryStatus';
import { OAuthCallbackError } from './components/OAuthCallbackError';
import { OAuthLoginPage } from './components/OAuthLoginPage';
import type { OAuthParams } from './components/OAuthLoginPage';
import { buildAuthorizeForm, parseAuthorizeError } from './auth/oauthAuthorization';
import { useAuthSession } from './auth/useAuthSession';
import { sessionLifecycle } from './auth/sessionLifecycle';
import { AuthRecoveryState } from './components/AuthRecoveryState';
import { SecurityBanner } from './components/SecurityBanner';
import { ToastContainer, useToast } from '@/components/shared/Toast';
import { UserMenu } from './components/UserMenu';
import { WarningsBanner } from './components/WarningsBanner';
import { AvailableModelsProvider } from '@/contexts/AvailableModelsContext';
import type {
  IndexJob,
  IndexInfo,
  User,
  AuthStatus,
  FilesystemIndexJob,
  SchemaIndexJob,
  PdmIndexJob,
  UserSpaceCodeIndexJob,
  ConfigurationWarning,
  UserSpacePreviewWarning,
  ServerBackupJob,
  ServerRestoreJob,
  OpenRouterCreditStatus,
} from '@/types';
import { BrandName } from '@/utils/buildEnvironment';
import { setThemePack, resolveThemePackId } from '@/theme';
import { ThemeChromeIcon } from '@/components/shared/ThemeChromeIcon';
import { SERVER_BACKUP_RESTORE_HIGHLIGHT } from '@/components/shared/securityWarnings';
import '@/styles/global.css';

type ViewType = 'chat' | 'userspace' | 'indexer' | 'tools' | 'users' | 'settings';
type UserSpaceSharedRoute =
  | { mode: 'token'; token: string }
  | { mode: 'slug'; ownerUsername: string; shareSlug: string }
  | null;

interface WorkspaceOpenRequest {
  workspaceId: string;
  requestId: number;
}

interface ChatOpenRequest {
  conversationId: string;
  requestId: number;
}

function getInitialView(): ViewType {
  const params = new URLSearchParams(window.location.search);
  const view = params.get('view');
  if (view === 'settings') return 'settings';
  if (view === 'users') return 'users';
  if (view === 'tools') return 'tools';
  if (view === 'indexer') return 'indexer';
  if (view === 'userspace') return 'userspace';
  return 'chat';
}

function getInitialHighlight(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get('highlight');
}

function getInitialConversationId(): string | null {
  const params = new URLSearchParams(window.location.search);
  const conversationId = params.get('conversation');
  return conversationId && conversationId.trim() ? conversationId.trim() : null;
}

const INDEXER_ACTIVE_POLL_MS = 2000;
const ENCRYPTION_KEY_ERROR_DISMISS_KEY = 'ragtime_encryption_key_error';
const ENCRYPTION_BACKUP_REMINDER_DISMISS_KEY = 'ragtime_encryption_backup_reminder';
const OPENROUTER_CREDIT_POLL_MS = 60_000;

const LazyChatPage = lazy(async () => ({
  default: (await import('./components/ChatPage')).ChatPage,
}));
const LazyUserSpacePanel = lazy(async () => ({
  default: (await import('./components/UserSpacePanel')).UserSpacePanel,
}));
const LazySettingsPanel = lazy(async () => ({
  default: (await import('./components/SettingsPanel')).SettingsPanel,
}));
const LazyToolsPanel = lazy(async () => ({
  default: (await import('./components/ToolsPanel')).ToolsPanel,
}));
const LazyUsersPanel = lazy(async () => ({
  default: (await import('./components/UsersPanel')).UsersPanel,
}));
const LazyIndexerAdminView = lazy(async () => ({
  default: (await import('./components/IndexerAdminView')).IndexerAdminView,
}));
const LazyPublicSharedChatView = lazy(async () => ({
  default: (await import('./components/PublicSharedChatView')).PublicSharedChatView,
}));

type ObservedServerJobKind = 'backup' | 'restore';

function isObservedServerJobTerminal(
  _kind: ObservedServerJobKind,
  job: ServerBackupJob | ServerRestoreJob | null,
): boolean {
  if (!job) {
    return false;
  }
  if (job.status === 'cancelled' || job.status === 'failed' || job.status === 'interrupted') {
    return true;
  }
  return job.status === 'completed';
}

function getObservedServerJobToastMessage(
  kind: ObservedServerJobKind,
  job: ServerBackupJob | ServerRestoreJob,
): { type: 'success' | 'error'; message: string } | null {
  if (job.status === 'failed' || job.status === 'interrupted') {
    return {
      type: 'error',
      message: job.error || job.message || `Server ${kind} ${job.status}.`,
    };
  }
  if (job.status === 'cancelled') {
    return null;
  }
  if (job.status === 'completed') {
    return {
      type: 'success',
      message:
        kind === 'backup'
          ? 'Server backup completed successfully.'
          : 'Server restore completed successfully.',
    };
  }
  return null;
}

function RouteViewFallback() {
  return (
    <div className="auth-loading" aria-live="polite">
      <div className="spinner"></div>
      <p>Loading view...</p>
    </div>
  );
}

function readPersistentDismissed(dismissKey: string): boolean {
  try {
    return window.localStorage.getItem(dismissKey) === 'true';
  } catch {
    return false;
  }
}

/**
 * Check if URL contains OAuth authorization parameters.
 * Returns OAuthParams if this is an OAuth flow, null otherwise.
 */
function getOAuthParams(): OAuthParams | null {
  const params = new URLSearchParams(window.location.search);
  const client_id = params.get('client_id');
  const redirect_uri = params.get('redirect_uri');
  const response_type = params.get('response_type');
  const code_challenge = params.get('code_challenge');

  // All required OAuth params must be present
  if (client_id && redirect_uri && response_type === 'code' && code_challenge) {
    return {
      client_id,
      redirect_uri,
      response_type,
      code_challenge,
      code_challenge_method: params.get('code_challenge_method') || 'S256',
      state: params.get('state') || '',
      resource: params.get('resource') || undefined,
      scope: params.get('scope') || undefined,
    };
  }
  return null;
}

interface OAuthCallbackErrorParams {
  title: string;
  summary: string;
  nextSteps: string[];
  runtime?: boolean;
}

function getOAuthCallbackError(): OAuthCallbackErrorParams | null {
  const params = new URLSearchParams(window.location.search);
  const title = params.get('oauth_error_title');
  const summary = params.get('oauth_error_summary');
  if (title && summary) {
    return { title, summary, nextSteps: params.getAll('oauth_next_steps') };
  }
  return null;
}

function getUserSpaceSharedRoute(): UserSpaceSharedRoute {
  const params = new URLSearchParams(window.location.search);
  const token = params.get('userspace_share_token');
  if (token && token.trim()) {
    return { mode: 'token', token: token.trim() };
  }

  const parts = window.location.pathname.split('/').filter(Boolean);
  if (parts.length >= 2 && parts[0] === 'shared' && parts[1]) {
    return {
      mode: 'token',
      token: decodeURIComponent(parts[1]),
    };
  }
  if (parts.length === 2) {
    const [ownerUsername, shareSlug] = parts;
    if (ownerUsername && shareSlug) {
      return {
        mode: 'slug',
        ownerUsername: decodeURIComponent(ownerUsername),
        shareSlug: decodeURIComponent(shareSlug),
      };
    }
  }

  return null;
}

export function App() {
  const {
    authStatus,
    currentUser,
    phase: authPhase,
    refresh,
    completeLogin,
    logout,
    retryBootstrap,
    retryLogout,
    recoveryAction,
    recoveryBusy,
    refreshError,
    generation: authGeneration,
    updatePresentation,
  } = useAuthSession();
  const authLoading =
    authPhase === 'bootstrapping' || authPhase === 'establishing' || authPhase === 'logging-out';

  // OAuth flow state - capture on mount
  const [oauthCallbackError, setOauthCallbackError] = useState<OAuthCallbackErrorParams | null>(
    getOAuthCallbackError,
  );
  const [oauthRetry, setOauthRetry] = useState(0);
  const oauthAuthorizationPromiseRef = useRef<Promise<void> | null>(null);
  const [oauthParams, setOauthParams] = useState<OAuthParams | null>(() => {
    const params = getOAuthParams();
    return params;
  });
  const [userspaceSharedRoute] = useState<UserSpaceSharedRoute>(getUserSpaceSharedRoute);
  const sharedRouteMountedRef = useRef(false);

  // App state
  const [activeView, setActiveView] = useState<ViewType>(getInitialView);
  const [highlightSetting, setHighlightSetting] = useState<string | null>(getInitialHighlight);
  const [initialConversationId] = useState<string | null>(getInitialConversationId);
  const [highlightToolsSection, setHighlightToolsSection] = useState<string | null>(null);
  const [serverName, setServerName] = useState<string>('Ragtime');
  const [authenticatedWebglBackgroundEnabled, setAuthenticatedWebglBackgroundEnabled] =
    useState(true);
  const [webglBackgroundPausedForBattery, setWebglBackgroundPausedForBattery] = useState(false);
  const [isNavOverflowOpen, setIsNavOverflowOpen] = useState(false);

  // Per-user motion background override (stored in localStorage)
  const [webglBackgroundUserOverride, setWebglBackgroundUserOverride] = useState<boolean | null>(
    () => {
      const stored = localStorage.getItem('ragtime-webgl-background');
      if (stored === 'true') return true;
      if (stored === 'false') return false;
      return null;
    },
  );

  const toggleWebglBackground = useCallback(() => {
    setWebglBackgroundUserOverride((prev) => {
      const next = prev === null ? false : !prev;
      if (next === null) {
        localStorage.removeItem('ragtime-webgl-background');
      } else {
        localStorage.setItem('ragtime-webgl-background', String(next));
      }
      return next;
    });
  }, []);

  const handleViewSelect = useCallback((view: ViewType) => {
    setActiveView(view);
    setIsNavOverflowOpen(false);
  }, []);

  // User override takes precedence over the global setting
  const effectiveWebglEnabled =
    webglBackgroundUserOverride !== null
      ? webglBackgroundUserOverride
      : authenticatedWebglBackgroundEnabled;

  useEffect(() => {
    if (!effectiveWebglEnabled) {
      setWebglBackgroundPausedForBattery(false);
    }
  }, [effectiveWebglEnabled]);

  const [jobs, setJobs] = useState<IndexJob[]>([]);
  const [indexes, setIndexes] = useState<IndexInfo[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [indexesLoading, setIndexesLoading] = useState(true);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [indexesError, setIndexesError] = useState<string | null>(null);

  // Filesystem indexer state
  const [filesystemToolIds, setFilesystemToolIds] = useState<string[]>([]);
  const [filesystemToolIdsLoaded, setFilesystemToolIdsLoaded] = useState(false);
  const [filesystemJobs, setFilesystemJobs] = useState<FilesystemIndexJob[]>([]);
  const [aggregateSearch, setAggregateSearch] = useState(true);
  const [embeddingDimensions, setEmbeddingDimensions] = useState<number | null>(null);
  const [previewWarning, setPreviewWarning] = useState<UserSpacePreviewWarning | null>(null);
  const [observedServerBackupJob, setObservedServerBackupJob] = useState<ServerBackupJob | null>(
    null,
  );
  const [observedServerRestoreJob, setObservedServerRestoreJob] = useState<ServerRestoreJob | null>(
    null,
  );
  const [toasts, toast] = useToast();
  const observedServerTerminalToastsRef = useRef<Set<string>>(new Set());
  const firstNavButtonRef = useRef<HTMLButtonElement | null>(null);
  const pendingRecoveryFocusRef = useRef(false);

  // Schema indexer state
  const [schemaJobs, setSchemaJobs] = useState<SchemaIndexJob[]>([]);

  // PDM indexer state
  const [pdmJobs, setPdmJobs] = useState<PdmIndexJob[]>([]);

  // Hidden User Space code index jobs
  const [userspaceCodeJobs, setUserspaceCodeJobs] = useState<UserSpaceCodeIndexJob[]>([]);

  // Configuration warnings state
  const [configurationWarnings, setConfigurationWarnings] = useState<ConfigurationWarning[]>([]);
  const [openRouterCreditStatus, setOpenRouterCreditStatus] =
    useState<OpenRouterCreditStatus | null>(null);
  const authIdentityRef = useRef({
    generation: authGeneration,
    userId: currentUser?.id ?? null,
    role: currentUser?.role ?? null,
  });
  authIdentityRef.current = {
    generation: authGeneration,
    userId: currentUser?.id ?? null,
    role: currentUser?.role ?? null,
  };
  const principalKeyRef = useRef<string | null>(null);
  const settingsRequestRef = useRef(0);
  const [encryptionBackupReminderDismissed, setEncryptionBackupReminderDismissed] = useState(() =>
    readPersistentDismissed(ENCRYPTION_BACKUP_REMINDER_DISMISS_KEY),
  );

  // Userspace fullscreen state
  const [userspaceFullscreen, setUserspaceFullscreen] = useState(false);
  const [chatFullscreen, setChatFullscreen] = useState(false);
  const [workspaceOpenRequest, setWorkspaceOpenRequest] = useState<WorkspaceOpenRequest | null>(
    null,
  );
  const [chatOpenRequest, setChatOpenRequest] = useState<ChatOpenRequest | null>(null);
  // Missing effective flags are unavailable until the server supplies the current policy.
  const chatEnabled = authStatus?.chat_enabled === true;
  const userspaceGenerationEnabled = authStatus?.userspace_generation_enabled === true;

  const handleOpenWorkspaceFromUsers = useCallback((workspaceId: string) => {
    setWorkspaceOpenRequest((prev) => ({
      workspaceId,
      requestId: (prev?.requestId ?? 0) + 1,
    }));
    setActiveView('userspace');
  }, []);

  const handleOpenChatFromUsers = useCallback(
    (conversationId: string) => {
      if (!chatEnabled) return;
      setChatOpenRequest((prev) => ({
        conversationId,
        requestId: (prev?.requestId ?? 0) + 1,
      }));
      setActiveView('chat');
    },
    [chatEnabled],
  );

  const clearPrincipalState = useCallback(() => {
    setObservedServerBackupJob(null);
    setObservedServerRestoreJob(null);
    observedServerTerminalToastsRef.current.clear();
  }, []);

  const applyAuthStatusPresentation = useCallback((status: AuthStatus, user?: User | null) => {
    const authServerName = (status.server_name || '').trim();
    if (authServerName) {
      setServerName(authServerName);
      document.title = authServerName;
    }
    setAuthenticatedWebglBackgroundEnabled(status.authenticated_webgl_background_enabled ?? true);
    setThemePack(resolveThemePackId(user?.theme_pack, status.default_theme_pack));
  }, []);

  const presentationServerName = authStatus?.server_name;
  const presentationWebglEnabled = authStatus?.authenticated_webgl_background_enabled;
  const presentationDefaultTheme = authStatus?.default_theme_pack;
  const presentationUserTheme = currentUser?.theme_pack;
  useLayoutEffect(() => {
    if (!authStatus) return;
    applyAuthStatusPresentation(
      {
        ...authStatus,
        server_name: presentationServerName,
        authenticated_webgl_background_enabled: presentationWebglEnabled,
        default_theme_pack: presentationDefaultTheme,
      },
      presentationUserTheme === currentUser?.theme_pack ? currentUser : null,
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps -- only presentation primitives may restore preferences
  }, [
    applyAuthStatusPresentation,
    presentationDefaultTheme,
    presentationServerName,
    presentationUserTheme,
    presentationWebglEnabled,
  ]);

  // Handle post-recovery focus after authenticated DOM mounts
  useLayoutEffect(() => {
    // Only consume the pending flag after recovery completes and DOM settles
    if (
      pendingRecoveryFocusRef.current &&
      !recoveryBusy &&
      (authPhase === 'authenticated' || authPhase === 'anonymous' || authPhase === 'unavailable')
    ) {
      // Focus only if authenticated
      if (authPhase === 'authenticated' && firstNavButtonRef.current) {
        firstNavButtonRef.current.focus();
      }
      // Clear the pending flag - recovery is settled
      pendingRecoveryFocusRef.current = false;
    }
  }, [authPhase, recoveryBusy]);

  // Derive stable primitive values to avoid effect dependency on full user object
  const userId = currentUser?.id;
  const userRole = currentUser?.role;

  useEffect(() => {
    const principalKey = userId && userRole ? `${userId}:${userRole}` : null;
    if (principalKeyRef.current === principalKey) return;
    principalKeyRef.current = principalKey;
    settingsRequestRef.current += 1;
    clearPrincipalState();
    setConfigurationWarnings([]);
    setOpenRouterCreditStatus(null);
  }, [clearPrincipalState, userId, userRole]);

  useEffect(() => {
    if (currentUser && !chatEnabled && activeView === 'chat') {
      setActiveView('userspace');
    }
  }, [activeView, currentUser, chatEnabled]);

  const refreshConfigurationWarnings = useCallback(async () => {
    const expected = authIdentityRef.current;
    if (!expected.userId || expected.role !== 'admin') return;
    const requestId = ++settingsRequestRef.current;
    const isCurrent = () => {
      const current = authIdentityRef.current;
      return (
        requestId === settingsRequestRef.current &&
        current.generation === expected.generation &&
        current.userId === expected.userId &&
        current.role === expected.role
      );
    };

    try {
      const { settings, configuration_warnings } = await api.getSettings();
      if (!isCurrent()) return;
      const configuredServerName = (settings.server_name || '').trim();
      const resolvedServerName = configuredServerName || 'Ragtime';
      const nextWarnings = configuration_warnings ?? [];

      setServerName(resolvedServerName);
      document.title = resolvedServerName;
      setAuthenticatedWebglBackgroundEnabled(
        settings.authenticated_webgl_background_enabled ?? true,
      );
      setAggregateSearch(settings.aggregate_search ?? true);
      setEmbeddingDimensions(settings.embedding_dimensions ?? null);
      setConfigurationWarnings(nextWarnings);

      const hasEncryptionWarning = nextWarnings.some(
        (warning) => warning.category === 'encryption',
      );
      if (!hasEncryptionWarning) {
        window.sessionStorage.removeItem(ENCRYPTION_KEY_ERROR_DISMISS_KEY);
      }
    } catch (error) {
      if (isCurrent()) console.error('Failed to refresh configuration warnings', error);
    }
  }, []);

  const handleSettingsSaved = useCallback(async () => {
    await refreshConfigurationWarnings();
    try {
      await refresh('policy-save');
    } catch (error) {
      console.error('Failed to refresh authenticated state after saving settings', error);
    }
  }, [refresh, refreshConfigurationWarnings]);

  useEffect(() => {
    if (currentUser?.role === 'admin') void refreshConfigurationWarnings();
  }, [authGeneration, currentUser?.id, currentUser?.role, refreshConfigurationWarnings]);

  // Callback to update server name from SettingsPanel
  const handleServerNameChange = useCallback((name: string) => {
    const resolvedName = name.trim() || 'Ragtime';
    setServerName(resolvedName);
    document.title = resolvedName;
  }, []);

  const handleChatCompactionThresholdChange = useCallback(
    (threshold: number) => {
      const normalizedThreshold = Math.max(1, Math.min(100, Math.round(threshold)));
      updatePresentation((previous) => ({
        ...previous,
        chat_compaction_threshold_percent: normalizedThreshold,
      }));
    },
    [updatePresentation],
  );

  const handleChatAutoCompactionThresholdChange = useCallback(
    (threshold: number) => {
      const normalizedThreshold = Math.max(1, Math.min(100, Math.round(threshold)));
      updatePresentation((previous) => ({
        ...previous,
        chat_auto_compaction_threshold_percent: normalizedThreshold,
      }));
    },
    [updatePresentation],
  );

  const handleEncryptedArtifactDelivered = useCallback(() => {
    try {
      window.localStorage.setItem(ENCRYPTION_BACKUP_REMINDER_DISMISS_KEY, 'true');
    } catch {
      // Ignore unavailable storage.
    }
    setEncryptionBackupReminderDismissed(true);
  }, []);

  const handleServerOperationError = useCallback(
    (message: string) => {
      toast.error(message);
    },
    [toast],
  );

  const observeServerBackupJob = useCallback(
    (job: ServerBackupJob) => {
      const toastInfo = getObservedServerJobToastMessage('backup', job);
      if (toastInfo) {
        const toastKey = `backup:${job.id}:${job.status}`;
        if (!observedServerTerminalToastsRef.current.has(toastKey)) {
          observedServerTerminalToastsRef.current.add(toastKey);
          if (toastInfo.type === 'success') {
            toast.success(toastInfo.message);
          } else {
            toast.error(toastInfo.message);
          }
        }
        setObservedServerBackupJob(null);
        return;
      }
      if (!isObservedServerJobTerminal('backup', job)) {
        setObservedServerBackupJob(job);
      } else {
        setObservedServerBackupJob(null);
      }
    },
    [toast],
  );

  const observeServerRestoreJob = useCallback(
    (job: ServerRestoreJob) => {
      const toastInfo = getObservedServerJobToastMessage('restore', job);
      if (toastInfo) {
        const toastKey = `restore:${job.id}:${job.status}`;
        if (!observedServerTerminalToastsRef.current.has(toastKey)) {
          observedServerTerminalToastsRef.current.add(toastKey);
          if (toastInfo.type === 'success') {
            toast.success(toastInfo.message);
          } else {
            toast.error(toastInfo.message);
          }
        }
        setObservedServerRestoreJob(null);
        return;
      }
      if (!isObservedServerJobTerminal('restore', job)) {
        setObservedServerRestoreJob(job);
      } else {
        setObservedServerRestoreJob(null);
      }
    },
    [toast],
  );

  useEffect(() => {
    const refreshOnReturn = () => {
      if (document.visibilityState === 'visible') {
        void refresh('return').catch((error) => {
          console.warn('Failed to refresh authenticated state after returning to the app', error);
        });
      }
    };
    window.addEventListener('focus', refreshOnReturn);
    document.addEventListener('visibilitychange', refreshOnReturn);
    return () => {
      window.removeEventListener('focus', refreshOnReturn);
      document.removeEventListener('visibilitychange', refreshOnReturn);
    };
  }, [refresh]);

  const handleLoginSuccess = async (user: User) => {
    if ((await completeLogin(user)) !== 'applied') return;
    // Redirect only after the current generation accepted the verified pair.
    if (user.role !== 'admin' && activeView !== 'chat' && activeView !== 'userspace') {
      setActiveView('userspace');
    }
  };

  // Auto-complete OAuth flow if user is already authenticated
  useEffect(() => {
    if (
      !oauthParams ||
      !currentUser ||
      authLoading ||
      oauthCallbackError ||
      oauthAuthorizationPromiseRef.current
    )
      return;

    const context = sessionLifecycle.capture('session');
    const completeOAuthFlow = async () => {
      try {
        const response = await apiFetch(
          '/authorize/session',
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: buildAuthorizeForm(oauthParams).toString(),
          },
          'session',
        );

        const data = await response.json();
        if (!sessionLifecycle.isCurrent(context)) return;

        if (response.ok && data.redirect_url && isResponseAuthContextCurrent(response)) {
          window.location.href = data.redirect_url;
        } else if (isResponseAuthContextCurrent(response)) {
          setOauthCallbackError({
            title: 'Authorization failed',
            runtime: true,
            ...parseAuthorizeError(data),
          });
        }
      } catch (err) {
        if (!sessionLifecycle.isCurrent(context)) return;
        setOauthCallbackError({
          title: 'Authorization failed',
          summary:
            err instanceof Error
              ? err.message
              : 'The authorization request could not be completed.',
          nextSteps: [],
          runtime: true,
        });
      }
    };

    const request = completeOAuthFlow();
    oauthAuthorizationPromiseRef.current = request;
    void request.finally(() => {
      if (oauthAuthorizationPromiseRef.current === request) {
        oauthAuthorizationPromiseRef.current = null;
      }
    });
  }, [oauthParams, currentUser, authLoading, oauthCallbackError, oauthRetry]);

  const handleLogout = logout;

  const handleRecoveryRetry = useCallback(async () => {
    // Before retry, check if user has input focused
    const activeElement = document.activeElement as HTMLElement | null;
    const hasUserInputFocus =
      activeElement instanceof HTMLInputElement ||
      activeElement instanceof HTMLTextAreaElement ||
      activeElement?.role === 'textbox' ||
      activeElement?.contentEditable === 'true';

    // Set pending flag to focus nav button after React mounts authenticated DOM
    // Only set if user doesn't have input focused
    if (!hasUserInputFocus) {
      pendingRecoveryFocusRef.current = true;
    }

    // Run the recovery - useLayoutEffect will consume the flag after DOM mounts
    await (recoveryAction === 'retry-logout' ? retryLogout() : retryBootstrap());
  }, [recoveryAction, retryLogout, retryBootstrap]);

  // Check if user is admin
  const isAdmin = currentUser?.role === 'admin';

  useEffect(() => {
    if (!currentUser || !isAdmin) {
      setOpenRouterCreditStatus(null);
      return;
    }

    let cancelled = false;
    const refreshCreditStatus = async () => {
      try {
        const status = await api.getOpenRouterCreditStatus();
        if (!cancelled) setOpenRouterCreditStatus(status);
      } catch {
        // Keep the last known result visible; a transient status request must not hide a low-credit alert.
      }
    };

    void refreshCreditStatus();
    const intervalId = window.setInterval(
      () => void refreshCreditStatus(),
      OPENROUTER_CREDIT_POLL_MS,
    );
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [currentUser, isAdmin]);

  useEffect(() => {
    if (!currentUser || !isAdmin) {
      setObservedServerBackupJob(null);
      setObservedServerRestoreJob(null);
      return;
    }

    let cancelled = false;
    void (async () => {
      try {
        const activeJobs = await api.getActiveServerBackupJobs();
        if (cancelled) {
          return;
        }
        if (activeJobs.backup_job) {
          observeServerBackupJob(activeJobs.backup_job);
        }
        if (activeJobs.restore_job) {
          observeServerRestoreJob(activeJobs.restore_job);
        }
      } catch (error) {
        console.warn('Failed to reconnect active server backup jobs:', error);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [currentUser, isAdmin, observeServerBackupJob, observeServerRestoreJob]);

  useEffect(() => {
    if (!currentUser || !isAdmin) {
      return;
    }

    if (!observedServerBackupJob && !observedServerRestoreJob) {
      return;
    }

    const interval = window.setInterval(() => {
      if (
        observedServerBackupJob &&
        !isObservedServerJobTerminal('backup', observedServerBackupJob)
      ) {
        void api
          .getServerBackupJob(observedServerBackupJob.id)
          .then((job) => {
            observeServerBackupJob(job);
          })
          .catch((error) => {
            console.warn('Failed to refresh observed backup job:', error);
          });
      }

      if (
        observedServerRestoreJob &&
        !isObservedServerJobTerminal('restore', observedServerRestoreJob)
      ) {
        void api
          .getServerRestoreJob(observedServerRestoreJob.id)
          .then((job) => {
            observeServerRestoreJob(job);
          })
          .catch((error) => {
            console.warn('Failed to refresh observed restore job:', error);
          });
      }
    }, INDEXER_ACTIVE_POLL_MS);

    return () => {
      window.clearInterval(interval);
    };
  }, [
    currentUser,
    isAdmin,
    observeServerBackupJob,
    observeServerRestoreJob,
    observedServerBackupJob,
    observedServerRestoreJob,
  ]);

  // Enforce admin-only views - redirect non-admins to chat
  useEffect(() => {
    if (currentUser && !isAdmin && activeView !== 'chat' && activeView !== 'userspace') {
      setActiveView('userspace');
    }
    // Clear fullscreen when leaving userspace
    if (activeView !== 'userspace') {
      setUserspaceFullscreen(false);
    }
    // Clear fullscreen when leaving chat
    if (activeView !== 'chat') {
      setChatFullscreen(false);
    }
  }, [currentUser, isAdmin, activeView]);

  const isChatView =
    chatEnabled && (activeView === 'chat' || (!isAdmin && activeView !== 'userspace'));
  const isUserspaceView = activeView === 'userspace';
  const isIndexerView = activeView === 'indexer';
  const lockViewportLayout = isChatView || isUserspaceView;
  const hideChrome = (isUserspaceView && userspaceFullscreen) || (isChatView && chatFullscreen);

  useEffect(() => {
    const rootClass = 'authenticated-webgl-background-active';
    const enabled = Boolean(currentUser && authenticatedWebglBackgroundEnabled);
    document.documentElement.classList.toggle(rootClass, enabled);
    document.body.classList.toggle(rootClass, enabled);

    return () => {
      document.documentElement.classList.remove(rootClass);
      document.body.classList.remove(rootClass);
    };
  }, [currentUser, authenticatedWebglBackgroundEnabled]);

  // Sync state to URL params (only sync valid views for user's role)
  // Skip URL sync during OAuth flow - we need to preserve those params until redirect
  useEffect(() => {
    if (userspaceSharedRoute) return;
    // Don't modify URL during OAuth authorization flow
    if (oauthParams) return;

    const params = new URLSearchParams(window.location.search);
    const existingConversationId = params.get('conversation');
    // Non-admins should only have user-space or chat views in URL
    const viewToSync =
      !isAdmin && activeView !== 'chat' && activeView !== 'userspace'
        ? 'userspace'
        : activeView === 'chat' && !chatEnabled
          ? 'userspace'
          : activeView;
    params.set('view', viewToSync);
    params.delete('conversation');
    if (viewToSync === 'chat') {
      if (existingConversationId && existingConversationId.trim()) {
        params.set('conversation', existingConversationId.trim());
      }
    }
    if (highlightSetting) {
      params.set('highlight', highlightSetting);
    } else {
      params.delete('highlight');
    }
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [activeView, highlightSetting, oauthParams, isAdmin, userspaceSharedRoute, chatEnabled]);

  const loadJobs = useCallback(async () => {
    try {
      const data = await api.listJobs();
      setJobs(data);
      setJobsError(null);
    } catch (err) {
      setJobsError(err instanceof Error ? err.message : 'Failed to load jobs');
    } finally {
      setJobsLoading(false);
    }
  }, []);

  const loadIndexes = useCallback(async () => {
    try {
      const data = await api.listIndexes();
      setIndexes(data);
      setIndexesError(null);
    } catch (err) {
      setIndexesError(err instanceof Error ? err.message : 'Failed to load indexes');
    } finally {
      setIndexesLoading(false);
    }
  }, []);

  const handleJobCreated = useCallback(() => {
    loadJobs();
    loadIndexes(); // Also refresh indexes to show optimistic metadata immediately
  }, [loadJobs, loadIndexes]);

  const refreshFilesystemToolIds = useCallback(async (): Promise<string[]> => {
    try {
      const allTools = await api.listToolConfigs();
      const toolIds = allTools.filter((t) => t.tool_type === 'filesystem_indexer').map((t) => t.id);
      setFilesystemToolIds(toolIds);
      setFilesystemToolIdsLoaded(true);
      return toolIds;
    } catch (err) {
      console.warn('Failed to load filesystem tools:', err);
      return [];
    }
  }, []);

  // Load filesystem tools and their jobs
  const loadFilesystemJobs = useCallback(
    async (toolIdsOverride?: string[]) => {
      try {
        const toolIds =
          toolIdsOverride ??
          (filesystemToolIdsLoaded ? filesystemToolIds : await refreshFilesystemToolIds());

        if (toolIds.length === 0) {
          setFilesystemJobs([]);
          return;
        }

        // Fetch jobs for all filesystem tools
        const allJobs: FilesystemIndexJob[] = [];
        await Promise.all(
          toolIds.map(async (toolId) => {
            try {
              const jobs = await api.getFilesystemJobs(toolId);
              allJobs.push(...jobs);
            } catch (err) {
              console.warn(`Failed to fetch jobs for ${toolId}:`, err);
            }
          }),
        );
        setFilesystemJobs(allJobs);
      } catch (err) {
        console.warn('Failed to load filesystem jobs:', err);
      }
    },
    [filesystemToolIds, filesystemToolIdsLoaded, refreshFilesystemToolIds],
  );

  const handleFilesystemToolsChanged = useCallback(async () => {
    const toolIds = await refreshFilesystemToolIds();
    await loadFilesystemJobs(toolIds);
  }, [loadFilesystemJobs, refreshFilesystemToolIds]);

  const handleCancelFilesystemJob = useCallback(
    async (toolId: string, jobId: string) => {
      await api.cancelFilesystemJob(toolId, jobId);
      await loadFilesystemJobs();
    },
    [loadFilesystemJobs],
  );

  // Load schema indexing jobs
  const loadSchemaJobs = useCallback(async () => {
    try {
      const jobs = await api.listSchemaJobs();
      setSchemaJobs(jobs);
    } catch (err) {
      console.warn('Failed to load schema jobs:', err);
    }
  }, []);

  const handleCancelSchemaJob = useCallback(
    async (toolId: string, jobId: string) => {
      await api.cancelSchemaIndexJob(toolId, jobId);
      await loadSchemaJobs();
    },
    [loadSchemaJobs],
  );

  // Load PDM indexing jobs
  const loadPdmJobs = useCallback(async () => {
    try {
      const jobs = await api.listPdmJobs();
      setPdmJobs(jobs);
    } catch (err) {
      console.warn('Failed to load PDM jobs:', err);
    }
  }, []);

  const handleCancelPdmJob = useCallback(
    async (toolId: string, jobId: string) => {
      await api.cancelPdmIndexJob(toolId, jobId);
      await loadPdmJobs();
    },
    [loadPdmJobs],
  );

  const loadUserSpaceCodeJobs = useCallback(async () => {
    try {
      const jobs = await api.listUserSpaceCodeIndexJobs();
      setUserspaceCodeJobs(jobs);
    } catch (err) {
      console.warn('Failed to load User Space code index jobs:', err);
    }
  }, []);

  // Initial load for indexer data: only when indexer view is visible.
  useEffect(() => {
    if (currentUser && isAdmin && isIndexerView) {
      loadJobs();
      loadIndexes();
      loadFilesystemJobs();
      loadSchemaJobs();
      loadPdmJobs();
      loadUserSpaceCodeJobs();
    }
  }, [
    currentUser,
    isAdmin,
    isIndexerView,
    loadJobs,
    loadIndexes,
    loadFilesystemJobs,
    loadSchemaJobs,
    loadPdmJobs,
    loadUserSpaceCodeJobs,
  ]);

  // Auto-refresh only while hidden User Space code index jobs are active.
  useEffect(() => {
    if (!currentUser || !isAdmin || !isIndexerView) return;

    const hasActiveUserSpaceCodeJob = userspaceCodeJobs.some(
      (j) => j.status === 'pending' || j.status === 'indexing',
    );

    if (!hasActiveUserSpaceCodeJob) return;

    const interval = setInterval(() => {
      loadUserSpaceCodeJobs();
    }, INDEXER_ACTIVE_POLL_MS);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, isIndexerView, userspaceCodeJobs, loadUserSpaceCodeJobs]);

  // Auto-refresh only while filesystem jobs are active.
  useEffect(() => {
    if (!currentUser || !isAdmin || !isIndexerView) return;

    const hasActiveFilesystemJob = filesystemJobs.some(
      (j) => j.status === 'pending' || j.status === 'indexing',
    );

    if (!hasActiveFilesystemJob) return;

    const interval = setInterval(() => {
      loadFilesystemJobs();
    }, INDEXER_ACTIVE_POLL_MS);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, isIndexerView, filesystemJobs, loadFilesystemJobs]);

  // Auto-refresh only while schema jobs are active.
  useEffect(() => {
    if (!currentUser || !isAdmin || !isIndexerView) return;

    const hasActiveSchemaJob = schemaJobs.some(
      (j) => j.status === 'pending' || j.status === 'indexing',
    );

    if (!hasActiveSchemaJob) return;

    const interval = setInterval(() => {
      loadSchemaJobs();
    }, INDEXER_ACTIVE_POLL_MS);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, isIndexerView, schemaJobs, loadSchemaJobs]);

  // Auto-refresh only while PDM jobs are active.
  useEffect(() => {
    if (!currentUser || !isAdmin || !isIndexerView) return;

    const hasActivePdmJob = pdmJobs.some((j) => j.status === 'pending' || j.status === 'indexing');

    if (!hasActivePdmJob) return;

    const interval = setInterval(() => {
      loadPdmJobs();
    }, INDEXER_ACTIVE_POLL_MS);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, isIndexerView, pdmJobs, loadPdmJobs]);

  // Auto-refresh only while document upload/git jobs are active.
  useEffect(() => {
    if (!currentUser || !isAdmin || !isIndexerView) return;

    const hasActiveJobs = jobs.some((j) => j.status === 'pending' || j.status === 'processing');

    if (!hasActiveJobs) return;

    const interval = setInterval(() => {
      loadJobs();
      loadIndexes();
    }, INDEXER_ACTIVE_POLL_MS);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, isIndexerView, jobs, loadJobs, loadIndexes]);

  if (userspaceSharedRoute) {
    if (!authLoading) sharedRouteMountedRef.current = true;
    // Block until the initial auth check has settled so PublicSharedChatView
    // mounts with a stable currentUser snapshot. Otherwise a null→user
    // transition during initial auth resolution would be misread as a fresh
    // sign-in and trigger an unwanted redirect to the authenticated view.
    if (authLoading && !sharedRouteMountedRef.current) {
      if (oauthParams || oauthCallbackError) {
        return (
          <LoginGradientShell className="auth-loading" aria-live="polite">
            <div className="spinner"></div>
            <p>Loading...</p>
          </LoginGradientShell>
        );
      }
      return (
        <div className="auth-loading">
          <div className="spinner"></div>
          <p>Loading...</p>
        </div>
      );
    }
    const showSharedRecovery =
      authPhase === 'unavailable' || (authPhase === 'logging-out' && recoveryBusy);
    const sharedRecoveryLabel =
      recoveryAction === 'retry-logout'
        ? 'Retry sign out'
        : recoveryAction === 'retry-bootstrap'
          ? 'Try connecting again'
          : 'Check session again';
    return (
      <>
        <div id="public-shared-route">
          <Suspense fallback={<RouteViewFallback />}>
            {userspaceSharedRoute.mode === 'token' ? (
              <LazyPublicSharedChatView
                shareToken={userspaceSharedRoute.token}
                currentUser={currentUser}
                authStatus={authStatus}
                serverName={serverName}
                onLoginSuccess={handleLoginSuccess}
                onLogout={handleLogout}
              />
            ) : (
              <LazyPublicSharedChatView
                ownerUsername={userspaceSharedRoute.ownerUsername}
                shareSlug={userspaceSharedRoute.shareSlug}
                currentUser={currentUser}
                authStatus={authStatus}
                serverName={serverName}
                onLoginSuccess={handleLoginSuccess}
                onLogout={handleLogout}
              />
            )}
          </Suspense>
        </div>
        {showSharedRecovery && (
          <div
            id="public-shared-auth-recovery-banner"
            className="login-status"
            role="region"
            aria-label={`Authentication recovery: ${sharedRecoveryLabel}`}
            aria-busy={recoveryBusy}
            style={{
              position: 'fixed',
              top: 'var(--space-lg)',
              left: '50%',
              transform: 'translateX(-50%)',
              width: 'calc(100vw - 2rem)',
              maxWidth: '32rem',
              maxHeight: 'calc(100vh - (2 * var(--space-lg)))',
              overflowY: 'auto',
              zIndex: 1000,
              padding: 'var(--space-lg)',
              color: 'var(--color-text-primary)',
              background: 'var(--color-surface)',
              border: '1px solid var(--color-error-border)',
              borderRadius: 'var(--radius-lg)',
              boxShadow: 'var(--shadow-xl)',
            }}
          >
            <p role="alert">{refreshError || 'Unable to check the session.'}</p>
            <div aria-live="polite" aria-atomic="true">
              {recoveryBusy ? 'Attempting recovery...' : null}
            </div>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={recoveryBusy}
              aria-busy={recoveryBusy}
              onClick={() => void handleRecoveryRetry()}
            >
              {sharedRecoveryLabel}
            </button>
          </div>
        )}
      </>
    );
  }

  // Show loading state while checking auth
  if (authLoading) {
    if (oauthParams || oauthCallbackError) {
      return (
        <LoginGradientShell className="auth-loading" aria-live="polite">
          <div className="spinner"></div>
          <p>Loading...</p>
        </LoginGradientShell>
      );
    }
    return (
      <div className="auth-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  // Handle OAuth callback errors (bad redirect_uri, unsupported response_type, etc.)
  if (oauthCallbackError) {
    return (
      <OAuthCallbackError
        title={oauthCallbackError.title}
        summary={oauthCallbackError.summary}
        nextSteps={oauthCallbackError.nextSteps}
        busy={false}
        onRetry={
          oauthCallbackError.runtime
            ? () => {
                setOauthCallbackError(null);
                setOauthRetry((retry) => retry + 1);
              }
            : undefined
        }
        onBack={
          oauthCallbackError.runtime
            ? () => {
                setOauthCallbackError(null);
                setOauthParams(null);
                window.history.replaceState({}, '', `${window.location.pathname}?view=userspace`);
                setActiveView('userspace');
              }
            : undefined
        }
      />
    );
  }

  // Handle OAuth authorization flow
  if (oauthParams) {
    // If user is authenticated, show authorizing state (auto-completing)
    if (currentUser) {
      return (
        <LoginGradientShell className="auth-loading" aria-live="polite">
          <div className="spinner"></div>
          <p>Authorizing...</p>
        </LoginGradientShell>
      );
    }
    // Not authenticated - show OAuth login page
    return <OAuthLoginPage params={oauthParams} serverName={serverName} />;
  }

  if (authPhase === 'unavailable') {
    return (
      <LoginGradientShell>
        <AuthRecoveryState
          action={recoveryAction}
          busy={recoveryBusy}
          message={refreshError || 'Unable to connect to the server.'}
          onRetry={() => void handleRecoveryRetry()}
        />
      </LoginGradientShell>
    );
  }

  // Show login page if not authenticated
  if (!currentUser) {
    return (
      <LoginPage
        authStatus={authStatus!}
        onLoginSuccess={handleLoginSuccess}
        serverName={serverName}
        initialError={refreshError}
      />
    );
  }

  const warningToMessage = (w: ConfigurationWarning): string =>
    w.recommendation ? `${w.message} ${w.recommendation}` : w.message;
  const encryptionKeyErrorMessages = configurationWarnings
    .filter((w) => w.category === 'encryption')
    .map(warningToMessage);
  const encryptionBackupMessages = configurationWarnings
    .filter((w) => w.category === 'encryption_backup')
    .map(warningToMessage);
  const otherConfigurationWarnings = configurationWarnings.filter(
    (w) => w.category !== 'encryption' && w.category !== 'encryption_backup',
  );
  const openRouterCreditWarning =
    openRouterCreditStatus &&
    (openRouterCreditStatus.state === 'low' || openRouterCreditStatus.state === 'exhausted')
      ? `${
          openRouterCreditStatus.warning ||
          `OpenRouter credits are ${openRouterCreditStatus.state}.`
        }${openRouterCreditStatus.stale ? ' Credit status is stale.' : ''}`
      : null;

  return (
    <AvailableModelsProvider>
      <div
        data-workbench-shell="authenticated"
        className={`app-shell${lockViewportLayout ? ' app-shell-locked' : ''}${authenticatedWebglBackgroundEnabled ? ' app-shell-webgl-background' : ''}`}
      >
        {effectiveWebglEnabled ? (
          <WebGLGradient
            className="app-background-gradient"
            fullscreen
            ignorePointerSelector=".topnav, .container, .modal, .modal-overlay, [role='dialog'], button, input, textarea, select, a"
            onBatteryStatusChange={setWebglBackgroundPausedForBattery}
          />
        ) : null}
        {authenticatedWebglBackgroundEnabled &&
        currentUser &&
        !hideChrome &&
        !webglBackgroundPausedForBattery ? (
          <button
            className="webgl-motion-toggle"
            data-active={effectiveWebglEnabled}
            onClick={toggleWebglBackground}
            aria-label={
              effectiveWebglEnabled ? 'Pause motion background' : 'Play motion background'
            }
            title={effectiveWebglEnabled ? 'Pause motion background' : 'Play motion background'}
          >
            <Waves size={14} />
          </button>
        ) : null}
        <div id="workbench-shell-stack">
          <nav
            id="workbench-topnav"
            className="topnav"
            data-workbench-surface="topnav"
            style={hideChrome ? { display: 'none' } : undefined}
          >
            <span className="topnav-brand">
              <BrandName name={serverName} />
            </span>
            <button
              type="button"
              className="topnav-overflow-trigger"
              aria-label="Toggle navigation"
              aria-controls="workbench-topnav-links"
              aria-expanded={isNavOverflowOpen}
              onClick={() => setIsNavOverflowOpen((open) => !open)}
            >
              <ThemeChromeIcon fallback={<MoreHorizontal size={16} />} codicon="ellipsis" />
            </button>
            <div
              id="workbench-topnav-links"
              className={`topnav-links${isNavOverflowOpen ? ' is-open' : ''}`}
            >
              {chatEnabled && (
                <button
                  ref={firstNavButtonRef}
                  type="button"
                  className={`topnav-link ${activeView === 'chat' ? 'active' : ''}`}
                  onClick={() => handleViewSelect('chat')}
                >
                  Chat
                </button>
              )}
              <button
                ref={!chatEnabled ? firstNavButtonRef : null}
                type="button"
                className={`topnav-link ${activeView === 'userspace' ? 'active' : ''}`}
                onClick={() => handleViewSelect('userspace')}
              >
                Workspace
              </button>
              {isAdmin && (
                <>
                  <button
                    type="button"
                    className={`topnav-link ${activeView === 'indexer' ? 'active' : ''}`}
                    onClick={() => handleViewSelect('indexer')}
                  >
                    Indexer
                  </button>
                  <button
                    type="button"
                    className={`topnav-link ${activeView === 'tools' ? 'active' : ''}`}
                    onClick={() => handleViewSelect('tools')}
                  >
                    Tools
                  </button>
                  <button
                    type="button"
                    className={`topnav-link ${activeView === 'users' ? 'active' : ''}`}
                    onClick={() => handleViewSelect('users')}
                  >
                    Users
                  </button>
                  <button
                    type="button"
                    className={`topnav-link ${activeView === 'settings' ? 'active' : ''}`}
                    onClick={() => handleViewSelect('settings')}
                  >
                    Settings
                  </button>
                </>
              )}
            </div>
            <div className="topnav-actions" data-workbench-surface="topnav-actions">
              <MemoryStatus />
              <UserMenu
                user={currentUser}
                onLogout={handleLogout}
                defaultThemePack={authStatus?.default_theme_pack}
              />
            </div>
          </nav>
          <div id="workbench-warning-stack">
            <SecurityBanner
              authStatus={authStatus}
              isAdmin={isAdmin}
              hidden={hideChrome}
              onNavigateToSettings={(highlightTarget) => {
                if (isAdmin) {
                  setHighlightSetting(highlightTarget || 'api_key_info');
                  setActiveView('settings');
                }
              }}
            />
            <WarningsBanner
              title="Encryption Key Error"
              warnings={encryptionKeyErrorMessages}
              dismissKey={ENCRYPTION_KEY_ERROR_DISMISS_KEY}
              hidden={hideChrome || !isAdmin}
            />
            <WarningsBanner
              title="Back Up Your Encryption Key"
              warnings={encryptionBackupMessages}
              dismissKey={ENCRYPTION_BACKUP_REMINDER_DISMISS_KEY}
              persistDismiss
              hidden={hideChrome || !isAdmin || encryptionBackupReminderDismissed}
              action={
                isAdmin
                  ? {
                      label: 'Open backup settings',
                      onClick: () => {
                        setHighlightSetting(SERVER_BACKUP_RESTORE_HIGHLIGHT);
                        setActiveView('settings');
                      },
                    }
                  : undefined
              }
            />
            <ConfigurationBanner
              warnings={otherConfigurationWarnings}
              isAdmin={isAdmin}
              hidden={hideChrome}
              onNavigateToSettings={() => {
                if (isAdmin) {
                  setHighlightSetting('embedding_config');
                  setActiveView('settings');
                }
              }}
            />
            <WarningsBanner
              title="OpenRouter Credit Alert"
              warnings={openRouterCreditWarning ? [openRouterCreditWarning] : []}
              compact
              hidden={hideChrome || !isAdmin}
              action={
                isAdmin
                  ? {
                      label: 'Open credit settings',
                      onClick: () => {
                        setHighlightSetting('openrouter-credit-monitor');
                        setActiveView('settings');
                      },
                    }
                  : undefined
              }
            />
            <WarningsBanner
              title={previewWarning?.title || 'Userspace Preview Setup'}
              warnings={previewWarning?.warnings || []}
              dismissKey={previewWarning?.dismiss_key}
              compact
              hidden={hideChrome || !isAdmin}
            />
          </div>
          <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
          <div className="container">
            {activeView === 'userspace' ? (
              <div
                id="workbench-userspace-route"
                className="userspace-page-container"
                data-workbench-route-root="userspace"
              >
                <Suspense fallback={<RouteViewFallback />}>
                  <LazyUserSpacePanel
                    currentUser={currentUser}
                    userspaceGenerationEnabled={userspaceGenerationEnabled}
                    debugMode={Boolean(authStatus?.debug_mode)}
                    openWorkspaceRequest={workspaceOpenRequest}
                    onFullscreenChange={setUserspaceFullscreen}
                    onPreviewWarningChange={setPreviewWarning}
                    onNavigateToTools={(section) => {
                      setHighlightToolsSection(section ?? null);
                      setActiveView('tools');
                    }}
                  />
                </Suspense>
              </div>
            ) : isChatView ? (
              <div
                id="workbench-chat-route"
                className={`chat-page-container${chatFullscreen ? ' chat-page-fullscreen' : ''}`}
                data-workbench-route-root="chat"
              >
                <Suspense fallback={<RouteViewFallback />}>
                  <LazyChatPage
                    key={chatOpenRequest ? `chat-open-${chatOpenRequest.requestId}` : 'chat-main'}
                    currentUser={currentUser}
                    debugMode={Boolean(authStatus?.debug_mode)}
                    initialConversationId={chatOpenRequest?.conversationId ?? initialConversationId}
                    chatCompactionThresholdPercent={
                      authStatus?.chat_compaction_threshold_percent ?? 80
                    }
                    chatAutoCompactionThresholdPercent={
                      authStatus?.chat_auto_compaction_threshold_percent ?? 99
                    }
                    onFullscreenChange={setChatFullscreen}
                  />
                </Suspense>
              </div>
            ) : activeView === 'settings' ? (
              <div id="workbench-settings-route" data-workbench-route-root="settings">
                <Suspense fallback={<RouteViewFallback />}>
                  <LazySettingsPanel
                    currentUser={currentUser}
                    onServerNameChange={handleServerNameChange}
                    onAuthenticatedWebglBackgroundChange={setAuthenticatedWebglBackgroundEnabled}
                    onChatCompactionThresholdChange={handleChatCompactionThresholdChange}
                    onChatAutoCompactionThresholdChange={handleChatAutoCompactionThresholdChange}
                    onSettingsSaved={handleSettingsSaved}
                    highlightSetting={highlightSetting}
                    onHighlightComplete={() => setHighlightSetting(null)}
                    authStatus={authStatus}
                    onEncryptedArtifactDelivered={handleEncryptedArtifactDelivered}
                    onServerBackupJobObserved={observeServerBackupJob}
                    onServerRestoreJobObserved={observeServerRestoreJob}
                    onServerOperationError={handleServerOperationError}
                  />
                </Suspense>
              </div>
            ) : activeView === 'tools' ? (
              <div id="workbench-tools-route" data-workbench-route-root="tools">
                <Suspense fallback={<RouteViewFallback />}>
                  <LazyToolsPanel
                    onSchemaJobTriggered={loadSchemaJobs}
                    schemaJobs={schemaJobs}
                    highlightSection={highlightToolsSection}
                    onHighlightComplete={() => setHighlightToolsSection(null)}
                  />
                </Suspense>
              </div>
            ) : activeView === 'users' ? (
              <div id="workbench-users-route" data-workbench-route-root="users">
                <Suspense fallback={<RouteViewFallback />}>
                  <LazyUsersPanel
                    currentUser={currentUser}
                    onOpenWorkspace={handleOpenWorkspaceFromUsers}
                    onOpenChat={chatEnabled ? handleOpenChatFromUsers : undefined}
                    onGenerationPolicyUpdated={async (updatedUser) => {
                      if (updatedUser.id !== currentUser.id) return;
                      await refresh('policy-save');
                    }}
                  />
                </Suspense>
              </div>
            ) : (
              <div id="workbench-indexer-route" data-workbench-route-root="indexer">
                <Suspense fallback={<RouteViewFallback />}>
                  <LazyIndexerAdminView
                    indexes={indexes}
                    jobs={jobs}
                    indexesLoading={indexesLoading}
                    indexesError={indexesError}
                    jobsLoading={jobsLoading}
                    jobsError={jobsError}
                    filesystemJobs={filesystemJobs}
                    schemaJobs={schemaJobs}
                    pdmJobs={pdmJobs}
                    userspaceCodeJobs={userspaceCodeJobs}
                    aggregateSearch={aggregateSearch}
                    embeddingDimensions={embeddingDimensions}
                    onLoadIndexes={loadIndexes}
                    onJobCreated={handleJobCreated}
                    onNavigateToSettings={() => {
                      setHighlightSetting('sequential_index_loading');
                      handleViewSelect('settings');
                    }}
                    onToolsChanged={handleFilesystemToolsChanged}
                    onFilesystemJobsChanged={loadFilesystemJobs}
                    onJobsChanged={loadJobs}
                    onSchemaJobsChanged={loadSchemaJobs}
                    onPdmJobsChanged={loadPdmJobs}
                    onUserSpaceCodeJobsChanged={loadUserSpaceCodeJobs}
                    onCancelFilesystemJob={handleCancelFilesystemJob}
                    onCancelSchemaJob={handleCancelSchemaJob}
                    onCancelPdmJob={handleCancelPdmJob}
                  />
                </Suspense>
              </div>
            )}
          </div>
        </div>
      </div>
    </AvailableModelsProvider>
  );
}
