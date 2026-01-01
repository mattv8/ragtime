import { useState, useEffect, useCallback, useRef } from 'react';
import { api } from '@/api';
import { JobsTable, IndexesList, FilesystemIndexPanel, SettingsPanel, ToolsPanel, ChatPanel, LoginPage } from '@/components';
import type { IndexJob, IndexInfo, User, AuthStatus, FilesystemIndexJob, ToolConfig, AppSettings } from '@/types';
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

export function App() {
  // Auth state
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  // App state
  const [activeView, setActiveView] = useState<ViewType>(getInitialView);
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

  // Load server name from settings
  useEffect(() => {
    const loadServerName = async () => {
      try {
        const settings: AppSettings = await api.getSettings();
        if (settings.server_name) {
          setServerName(settings.server_name);
          document.title = settings.server_name;
        }
      } catch {
        // Ignore errors, use default name
      }
    };
    loadServerName();
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
        });
      } finally {
        setAuthLoading(false);
      }
    };

    checkAuth();
  }, []);

  const handleLoginSuccess = (user: User) => {
    setCurrentUser(user);
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

  // Sync state to URL params
  useEffect(() => {
    const params = new URLSearchParams();
    params.set('view', activeView);
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [activeView]);

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

  // Initial load (only when authenticated and admin for indexer data)
  useEffect(() => {
    if (currentUser && isAdmin) {
      loadJobs();
      loadIndexes();
      loadFilesystemJobs();
    }
  }, [currentUser, isAdmin, loadJobs, loadIndexes, loadFilesystemJobs]);

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
        authStatus={authStatus || { authenticated: false, ldap_configured: false, local_admin_enabled: true, debug_mode: false }}
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
        <div className="topnav-user">
          <span className="topnav-username">
            {currentUser.display_name || currentUser.username}
            {isAdmin && <span className="admin-badge">Admin</span>}
          </span>
          <button className="topnav-link logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </nav>
      <div className="container">

      {activeView === 'chat' ? (
        <ChatPanel currentUser={currentUser} />
      ) : activeView === 'settings' ? (
        <SettingsPanel onServerNameChange={handleServerNameChange} />
      ) : activeView === 'tools' ? (
        <ToolsPanel />
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
          />

          {/* Filesystem Indexes (pgvector) */}
          <FilesystemIndexPanel onJobsChanged={loadFilesystemJobs} />

          {/* Jobs Table */}
          <JobsTable
            jobs={jobs}
            filesystemJobs={filesystemJobs}
            loading={jobsLoading}
            error={jobsError}
            onJobsChanged={loadJobs}
            onFilesystemJobsChanged={loadFilesystemJobs}
            onCancelFilesystemJob={handleCancelFilesystemJob}
          />
        </>
      )}
      </div>
    </>
  );
}
