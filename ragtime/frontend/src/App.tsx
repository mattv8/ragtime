import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/api';
import { JobsTable, IndexesList, FilesystemIndexPanel, SettingsPanel, ToolsPanel, ChatPanel, LoginPage, MemoryStatus, UserMenu, SecurityBanner } from '@/components';
import type { IndexJob, IndexInfo, User, AuthStatus, FilesystemIndexJob, SchemaIndexJob, PdmIndexJob, ToolConfig, AppSettings } from '@/types';
import '@/styles/global.css';

type ViewType = 'chat' | 'indexer' | 'tools' | 'settings';

function getInitialView(): ViewType {
  const params = new URLSearchParams(window.location.search);
  const view = params.get('view');
  if (view === 'settings') return 'settings';
  if (view === 'tools') return 'tools';
  if (view === 'indexer') return 'indexer';
  return 'chat';
}

function getInitialHighlight(): string | null {
  const params = new URLSearchParams(window.location.search);
  return params.get('highlight');
}

export function App() {
  // Auth state
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

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
  const filesystemPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [aggregateSearch, setAggregateSearch] = useState(true);
  const [embeddingDimensions, setEmbeddingDimensions] = useState<number | null>(null);

  // Schema indexer state
  const [schemaJobs, setSchemaJobs] = useState<SchemaIndexJob[]>([]);
  const schemaPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // PDM indexer state
  const [pdmJobs, setPdmJobs] = useState<PdmIndexJob[]>([]);
  const pdmPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load server name and embedding dimensions from settings
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const settings: AppSettings = await api.getSettings();
        if (settings.server_name) {
          setServerName(settings.server_name);
          document.title = settings.server_name;
        }
        // Also load aggregate_search setting
        setAggregateSearch(settings.aggregate_search ?? true);
        // Load embedding dimensions for memory calculation
        setEmbeddingDimensions(settings.embedding_dimensions ?? null);
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
  }, []);

  const handleLoginSuccess = (user: User) => {
    setCurrentUser(user);
    // If non-admin tried to access admin view via URL, redirect to chat
    if (user.role !== 'admin' && activeView !== 'chat') {
      setActiveView('chat');
    }
  };

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
    if (currentUser && !isAdmin && activeView !== 'chat') {
      setActiveView('chat');
    }
  }, [currentUser, isAdmin, activeView]);

  // Sync state to URL params (only sync valid views for user's role)
  useEffect(() => {
    const params = new URLSearchParams();
    // Non-admins should only have 'chat' in URL
    const viewToSync = (!isAdmin && activeView !== 'chat') ? 'chat' : activeView;
    params.set('view', viewToSync);
    if (highlightSetting) {
      params.set('highlight', highlightSetting);
    }
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [activeView, highlightSetting]);

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
  }, [loadJobs]);

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

  // Fast polling for active filesystem jobs
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActiveFilesystemJob = filesystemJobs.some(
      j => j.status === 'pending' || j.status === 'indexing'
    );

    if (hasActiveFilesystemJob) {
      filesystemPollRef.current = setInterval(loadFilesystemJobs, 2000);
    } else {
      if (filesystemPollRef.current) {
        clearInterval(filesystemPollRef.current);
        filesystemPollRef.current = null;
      }
    }

    return () => {
      if (filesystemPollRef.current) {
        clearInterval(filesystemPollRef.current);
      }
    };
  }, [currentUser, isAdmin, filesystemJobs, loadFilesystemJobs]);

  // Fast polling for active schema jobs
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActiveSchemaJob = schemaJobs.some(
      j => j.status === 'pending' || j.status === 'indexing'
    );

    if (hasActiveSchemaJob) {
      schemaPollRef.current = setInterval(loadSchemaJobs, 2000);
    } else {
      if (schemaPollRef.current) {
        clearInterval(schemaPollRef.current);
        schemaPollRef.current = null;
      }
    }

    return () => {
      if (schemaPollRef.current) {
        clearInterval(schemaPollRef.current);
      }
    };
  }, [currentUser, isAdmin, schemaJobs, loadSchemaJobs]);

  // Fast polling for active PDM jobs
  useEffect(() => {
    if (!currentUser || !isAdmin) return;

    const hasActivePdmJob = pdmJobs.some(
      j => j.status === 'pending' || j.status === 'indexing'
    );

    if (hasActivePdmJob) {
      pdmPollRef.current = setInterval(loadPdmJobs, 2000);
    } else {
      if (pdmPollRef.current) {
        clearInterval(pdmPollRef.current);
        pdmPollRef.current = null;
      }
    }

    return () => {
      if (pdmPollRef.current) {
        clearInterval(pdmPollRef.current);
      }
    };
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
    <>
      <nav className="topnav">
        <span className="topnav-brand">{serverName}</span>
        <div className="topnav-links">
          <button
            className={`topnav-link ${activeView === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveView('chat')}
          >
            Chat
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
        onNavigateToSettings={() => {
          if (isAdmin) {
            setHighlightSetting('api_key_info');
            setActiveView('settings');
          }
        }}
      />
      <div className="container">

      {/* Non-admins can only see chat - this is the final safeguard */}
      {activeView === 'chat' || !isAdmin ? (
        <ChatPanel currentUser={currentUser} />
      ) : activeView === 'settings' ? (
        <SettingsPanel
          onServerNameChange={handleServerNameChange}
          highlightSetting={highlightSetting}
          onHighlightComplete={() => setHighlightSetting(null)}
        />
      ) : activeView === 'tools' ? (
        <ToolsPanel onSchemaJobTriggered={loadSchemaJobs} />
      ) : (
        <>
          {/* Document Indexes (FAISS) */}
          <IndexesList
            indexes={indexes}
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
    </>
  );
}
