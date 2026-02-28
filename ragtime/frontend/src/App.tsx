import { useState, useEffect, useCallback } from 'react';
import { api } from '@/api';
import { JobsTable, IndexesList, FilesystemIndexPanel, SettingsPanel, ToolsPanel, ChatPanel, UserSpacePanel, LoginPage, OAuthLoginPage, MemoryStatus, UserMenu, SecurityBanner, ConfigurationBanner } from '@/components';
import type { IndexJob, IndexInfo, User, AuthStatus, FilesystemIndexJob, SchemaIndexJob, PdmIndexJob, ToolConfig, ConfigurationWarning } from '@/types';
import type { OAuthParams } from '@/components';
import '@/styles/global.css';

type ViewType = 'chat' | 'userspace' | 'indexer' | 'tools' | 'settings';
type UserSpaceSharedRoute =
  | { mode: 'token'; token: string }
  | { mode: 'slug'; ownerUsername: string; shareSlug: string }
  | null;

function getInitialView(): ViewType {
  const params = new URLSearchParams(window.location.search);
  const view = params.get('view');
  if (view === 'settings') return 'settings';
  if (view === 'tools') return 'tools';
  if (view === 'indexer') return 'indexer';
  if (view === 'userspace') return 'userspace';
  return 'chat';
}

function getInitialHighlight(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get('highlight');
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
    };
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

function getUserSpaceSharedProxyUrl(route: UserSpaceSharedRoute): string {
  if (!route) {
    return '/';
  }
  if (route.mode === 'token') {
    return `/shared/${encodeURIComponent(route.token)}`;
  }
  return `/${encodeURIComponent(route.ownerUsername)}/${encodeURIComponent(route.shareSlug)}`;
}

export function App() {
  // Auth state
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  // OAuth flow state - capture on mount
  const [oauthParams] = useState<OAuthParams | null>(() => {
    const params = getOAuthParams();
    return params;
  });
  const [userspaceSharedRoute] = useState<UserSpaceSharedRoute>(getUserSpaceSharedRoute);

  // App state
  const [activeView, setActiveView] = useState<ViewType>(getInitialView);
  const [highlightSetting, setHighlightSetting] = useState<string | null>(getInitialHighlight);
  const [serverName, setServerName] = useState<string>('Ragtime');
  const [jobs, setJobs] = useState<IndexJob[]>([]);
  const [indexes, setIndexes] = useState<IndexInfo[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [indexesLoading, setIndexesLoading] = useState(true);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [indexesError, setIndexesError] = useState<string | null>(null);

  // Filesystem indexer state
  const [_filesystemTools, setFilesystemTools] = useState<ToolConfig[]>([]);
  const [filesystemJobs, setFilesystemJobs] = useState<FilesystemIndexJob[]>([]);
  const [aggregateSearch, setAggregateSearch] = useState(true);
  const [embeddingDimensions, setEmbeddingDimensions] = useState<number | null>(null);

  // Schema indexer state
  const [schemaJobs, setSchemaJobs] = useState<SchemaIndexJob[]>([]);

  // PDM indexer state
  const [pdmJobs, setPdmJobs] = useState<PdmIndexJob[]>([]);

  // Configuration warnings state
  const [configurationWarnings, setConfigurationWarnings] = useState<ConfigurationWarning[]>([]);

  // Userspace fullscreen state
  const [userspaceFullscreen, setUserspaceFullscreen] = useState(false);
  const [chatFullscreen, setChatFullscreen] = useState(false);

  // Load server name and embedding dimensions from settings
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const { settings, configuration_warnings } = await api.getSettings();
        if (settings.server_name) {
          setServerName(settings.server_name);
          document.title = settings.server_name;
        }
        // Also load aggregate_search setting
        setAggregateSearch(settings.aggregate_search ?? true);
        // Load embedding dimensions for memory calculation
        setEmbeddingDimensions(settings.embedding_dimensions ?? null);
        // Store configuration warnings
        setConfigurationWarnings(configuration_warnings ?? []);
      } catch {
        // Ignore errors, use default name
      }
    };
    loadSettings();
  }, []);

  // Callback to update server name from SettingsPanel
  const handleServerNameChange = useCallback((name: string) => {
    setServerName(name);
    document.title = name;
  }, []);

  // Check authentication status on mount
  useEffect(() => {
    if (userspaceSharedRoute) {
      setAuthLoading(false);
      return;
    }

    const checkAuth = async () => {
      try {
        const status = await api.getAuthStatus();
        setAuthStatus(status);

        // Only try to get current user if we might be authenticated
        // This avoids unnecessary 401 errors in the console
        try {
          const user = await api.getCurrentUser();
          setCurrentUser(user);
        } catch {
          // Not authenticated, that's fine - show login page
          setCurrentUser(null);
        }
      } catch (err) {
        console.error('Failed to check auth status:', err);
        // If we can't check auth, assume not authenticated
        setAuthStatus({
          authenticated: false,
          ldap_configured: false,
          local_admin_enabled: true,
          debug_mode: false,
          api_key_configured: false,
          session_cookie_secure: false,
          allowed_origins_open: true,
        });
      } finally {
        setAuthLoading(false);
      }
    };

    checkAuth();
  }, [userspaceSharedRoute]);

  const handleLoginSuccess = (user: User) => {
    setCurrentUser(user);
    // If non-admin tried to access admin view via URL, redirect to chat
    if (user.role !== 'admin' && activeView !== 'chat' && activeView !== 'userspace') {
      setActiveView('userspace');
    }
  };

  // Auto-complete OAuth flow if user is already authenticated
  useEffect(() => {
    if (!oauthParams || !currentUser || authLoading) return;

    const completeOAuthFlow = async () => {
      try {
        const formData = new URLSearchParams();
        formData.append('client_id', oauthParams.client_id);
        formData.append('redirect_uri', oauthParams.redirect_uri);
        formData.append('response_type', oauthParams.response_type);
        formData.append('code_challenge', oauthParams.code_challenge);
        formData.append('code_challenge_method', oauthParams.code_challenge_method);
        formData.append('state', oauthParams.state);

        const response = await fetch('/authorize/session', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
          },
          body: formData.toString(),
          credentials: 'include', // Include session cookie
        });

        const data = await response.json();

        if (response.ok && data.redirect_url) {
          window.location.href = data.redirect_url;
        } else {
          // Session invalid or error - user needs to re-login
          console.error('OAuth flow: Session auth failed', data);
          // Clear the user so login page shows
          setCurrentUser(null);
        }
      } catch (err) {
        console.error('OAuth flow: Failed to complete', err);
        // Clear the user so login page shows
        setCurrentUser(null);
      }
    };

    completeOAuthFlow();
  }, [oauthParams, currentUser, authLoading]);

  const handleLogout = async () => {
    try {
      await api.logout();
    } catch {
      // Ignore logout errors
    }
    setCurrentUser(null);
  };

  // Check if user is admin
  const isAdmin = currentUser?.role === 'admin';

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

  const isChatView = activeView === 'chat' || (!isAdmin && activeView !== 'userspace');
  const isUserspaceView = activeView === 'userspace';
  const lockViewportLayout = isChatView || isUserspaceView;
  const hideChrome = (isUserspaceView && userspaceFullscreen) || (isChatView && chatFullscreen);

  // Sync state to URL params (only sync valid views for user's role)
  // Skip URL sync during OAuth flow - we need to preserve those params until redirect
  useEffect(() => {
    if (userspaceSharedRoute) return;
    // Don't modify URL during OAuth authorization flow
    if (oauthParams) return;

    const params = new URLSearchParams();
    // Non-admins should only have user-space or chat views in URL
    const viewToSync = (!isAdmin && activeView !== 'chat' && activeView !== 'userspace') ? 'userspace' : activeView;
    params.set('view', viewToSync);
    if (highlightSetting) {
      params.set('highlight', highlightSetting);
    }
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [activeView, highlightSetting, oauthParams, isAdmin, userspaceSharedRoute]);

  useEffect(() => {
    if (!userspaceSharedRoute) {
      return;
    }
    const target = getUserSpaceSharedProxyUrl(userspaceSharedRoute);
    const current = `${window.location.pathname}${window.location.search}`;
    if (target !== current) {
      window.location.replace(target);
    }
  }, [userspaceSharedRoute]);

  if (userspaceSharedRoute) {
    return <div className="chat-status-text">Redirecting to shared preview…</div>;
  }

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

  // Load filesystem tools and their jobs
  const loadFilesystemJobs = useCallback(async () => {
    try {
      const allTools = await api.listToolConfigs();
      const fsTools = allTools.filter(t => t.tool_type === 'filesystem_indexer');
      setFilesystemTools(fsTools);

      // Fetch jobs for all filesystem tools
      const allJobs: FilesystemIndexJob[] = [];
      await Promise.all(fsTools.map(async (tool) => {
        try {
          const jobs = await api.getFilesystemJobs(tool.id);
          allJobs.push(...jobs);
        } catch (err) {
          console.warn(`Failed to fetch jobs for ${tool.id}:`, err);
        }
      }));
      setFilesystemJobs(allJobs);
    } catch (err) {
      console.warn('Failed to load filesystem tools:', err);
    }
  }, []);

  const handleCancelFilesystemJob = useCallback(async (toolId: string, jobId: string) => {
    await api.cancelFilesystemJob(toolId, jobId);
    await loadFilesystemJobs();
  }, [loadFilesystemJobs]);

  // Load schema indexing jobs
  const loadSchemaJobs = useCallback(async () => {
    try {
      const jobs = await api.listSchemaJobs();
      setSchemaJobs(jobs);
    } catch (err) {
      console.warn('Failed to load schema jobs:', err);
    }
  }, []);

  const handleCancelSchemaJob = useCallback(async (toolId: string, jobId: string) => {
    await api.cancelSchemaIndexJob(toolId, jobId);
    await loadSchemaJobs();
  }, [loadSchemaJobs]);

  // Load PDM indexing jobs
  const loadPdmJobs = useCallback(async () => {
    try {
      const jobs = await api.listPdmJobs();
      setPdmJobs(jobs);
    } catch (err) {
      console.warn('Failed to load PDM jobs:', err);
    }
  }, []);

  const handleCancelPdmJob = useCallback(async (toolId: string, jobId: string) => {
    await api.cancelPdmIndexJob(toolId, jobId);
    await loadPdmJobs();
  }, [loadPdmJobs]);

  // Initial load (only when authenticated and admin for indexer data)
  useEffect(() => {
    if (currentUser && isAdmin) {
      loadJobs();
      loadIndexes();
      loadFilesystemJobs();
      loadSchemaJobs();
      loadPdmJobs();
    }
  }, [currentUser, isAdmin, loadJobs, loadIndexes, loadFilesystemJobs, loadSchemaJobs, loadPdmJobs]);

  // Auto-refresh: fast polling when filesystem jobs are active, slow background refresh otherwise
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActiveFilesystemJob = filesystemJobs.some(
      j => j.status === 'pending' || j.status === 'indexing'
    );

    const pollInterval = hasActiveFilesystemJob ? 2000 : 30000;

    const interval = setInterval(() => {
      loadFilesystemJobs();
    }, pollInterval);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, filesystemJobs, loadFilesystemJobs]);

  // Auto-refresh: fast polling when schema jobs are active, slow background refresh otherwise
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActiveSchemaJob = schemaJobs.some(
      j => j.status === 'pending' || j.status === 'indexing'
    );

    const pollInterval = hasActiveSchemaJob ? 2000 : 30000;

    const interval = setInterval(() => {
      loadSchemaJobs();
    }, pollInterval);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, schemaJobs, loadSchemaJobs]);

  // Auto-refresh: fast polling when PDM jobs are active, slow background refresh otherwise
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActivePdmJob = pdmJobs.some(
      j => j.status === 'pending' || j.status === 'indexing'
    );

    const pollInterval = hasActivePdmJob ? 2000 : 30000;

    const interval = setInterval(() => {
      loadPdmJobs();
    }, pollInterval);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, pdmJobs, loadPdmJobs]);

  // Auto-refresh: fast polling when jobs are active, slow background refresh otherwise
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActiveJobs = jobs.some((j) => j.status === 'pending' || j.status === 'processing');

    // Fast polling (2s) when jobs are active, slow polling (30s) for background updates
    const pollInterval = hasActiveJobs ? 2000 : 30000;

    const interval = setInterval(() => {
      loadJobs();
      if (hasActiveJobs) {
        loadIndexes();
      }
    }, pollInterval);

    return () => clearInterval(interval);
  }, [currentUser, isAdmin, jobs, loadJobs, loadIndexes]);

  // Show loading state while checking auth
  if (authLoading) {
    return (
      <div className="auth-loading">
        <div className="spinner"></div>
        <p>Loading...</p>
      </div>
    );
  }

  // Handle OAuth authorization flow
  if (oauthParams) {
    // If user is authenticated, show authorizing state (auto-completing)
    if (currentUser) {
      return (
        <div className="auth-loading">
          <div className="spinner"></div>
          <p>Authorizing...</p>
        </div>
      );
    }
    // Not authenticated - show OAuth login page
    return (
      <OAuthLoginPage
        params={oauthParams}
        serverName={serverName}
      />
    );
  }

  // Show login page if not authenticated
  if (!currentUser) {
    return (
      <LoginPage
        authStatus={authStatus || {
          authenticated: false,
          ldap_configured: false,
          local_admin_enabled: true,
          debug_mode: false,
          api_key_configured: false,
          session_cookie_secure: false,
          allowed_origins_open: true,
        }}
        onLoginSuccess={handleLoginSuccess}
        serverName={serverName}
      />
    );
  }

  return (
    <div className={`app-shell${lockViewportLayout ? ' app-shell-locked' : ''}`}>
      <nav className="topnav" style={hideChrome ? { display: 'none' } : undefined}>
        <span className="topnav-brand">{serverName}</span>
        <div className="topnav-links">
          <button
            className={`topnav-link ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveView('chat')}
          >
            Chat
          </button>
          <button
            className={`topnav-link ${activeView === 'userspace' ? 'active' : ''}`}
            onClick={() => setActiveView('userspace')}
          >
            Workspace
          </button>
          {isAdmin && (
            <>
              <button
                className={`topnav-link ${activeView === 'indexer' ? 'active' : ''}`}
                onClick={() => setActiveView('indexer')}
              >
                Indexer
              </button>
              <button
                className={`topnav-link ${activeView === 'tools' ? 'active' : ''}`}
                onClick={() => setActiveView('tools')}
              >
                Tools
              </button>
              <button
                className={`topnav-link ${activeView === 'settings' ? 'active' : ''}`}
                onClick={() => setActiveView('settings')}
              >
                Settings
              </button>
            </>
          )}
        </div>
        <div className="topnav-actions">
          <MemoryStatus />
          <UserMenu user={currentUser} onLogout={handleLogout} />
        </div>
      </nav>
      <SecurityBanner
        authStatus={authStatus}
        isAdmin={isAdmin}
        hidden={hideChrome}
        onNavigateToSettings={() => {
          if (isAdmin) {
            setHighlightSetting('api_key_info');
            setActiveView('settings');
          }
        }}
      />
      <ConfigurationBanner
        warnings={configurationWarnings}
        isAdmin={isAdmin}
        hidden={hideChrome}
        onNavigateToSettings={() => {
          if (isAdmin) {
            setHighlightSetting('embedding_config');
            setActiveView('settings');
          }
        }}
      />
      <div className="container">

      {activeView === 'userspace' ? (
        <div className="userspace-page-container">
          <UserSpacePanel currentUser={currentUser} onFullscreenChange={setUserspaceFullscreen} />
        </div>
      ) : isChatView ? (
        <div className="chat-page-container">
          <ChatPanel currentUser={currentUser} onFullscreenChange={setChatFullscreen} />
        </div>
      ) : activeView === 'settings' ? (
        <SettingsPanel
          onServerNameChange={handleServerNameChange}
          highlightSetting={highlightSetting}
          onHighlightComplete={() => setHighlightSetting(null)}
          authStatus={authStatus}
        />
      ) : activeView === 'tools' ? (
        <ToolsPanel onSchemaJobTriggered={loadSchemaJobs} schemaJobs={schemaJobs} />
      ) : (
        <>
          {/* Document Indexes (FAISS) */}
          <IndexesList
            indexes={indexes}
            jobs={jobs}
            loading={indexesLoading}
            error={indexesError}
            onDelete={loadIndexes}
            onToggle={loadIndexes}
            onDescriptionUpdate={loadIndexes}
            onJobCreated={handleJobCreated}
            aggregateSearch={aggregateSearch}
            embeddingDimensions={embeddingDimensions}
            onNavigateToSettings={() => {
              setHighlightSetting('sequential_index_loading');
              setActiveView('settings');
            }}
          />

          {/* Filesystem Indexes (pgvector) */}
          <FilesystemIndexPanel
            onJobsChanged={loadFilesystemJobs}
            embeddingDimensions={embeddingDimensions}
          />

          {/* Jobs Table */}
          <JobsTable
            jobs={jobs}
            filesystemJobs={filesystemJobs}
            schemaJobs={schemaJobs}
            pdmJobs={pdmJobs}
            loading={jobsLoading}
            error={jobsError}
            onJobsChanged={loadJobs}
            onFilesystemJobsChanged={loadFilesystemJobs}
            onSchemaJobsChanged={loadSchemaJobs}
            onPdmJobsChanged={loadPdmJobs}
            onCancelFilesystemJob={handleCancelFilesystemJob}
            onCancelSchemaJob={handleCancelSchemaJob}
            onCancelPdmJob={handleCancelPdmJob}
          />
        </>
      )}
      </div>
    </div>
  );
}
