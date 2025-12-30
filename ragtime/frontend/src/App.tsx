import { useState, useEffect, useCallback } from 'react';
import { api } from '@/api';
import { UploadForm, GitForm, JobsTable, IndexesList, SettingsPanel, ToolsPanel, ChatPanel, LoginPage } from '@/components';
import type { IndexJob, IndexInfo, User, AuthStatus } from '@/types';
import '@/styles/global.css';

type SourceType = 'upload' | 'git';
type ViewType = 'chat' | 'indexer' | 'tools' | 'settings';

function getInitialView(): ViewType {
  const params = new URLSearchParams(window.location.search);
  const view = params.get('view');
  if (view === 'settings') return 'settings';
  if (view === 'tools') return 'tools';
  if (view === 'indexer') return 'indexer';
  return 'chat';
}

function getInitialSource(): SourceType {
  const params = new URLSearchParams(window.location.search);
  const source = params.get('source');
  return source === 'git' ? 'git' : 'upload';
}

export function App() {
  // Auth state
  const [authStatus, setAuthStatus] = useState<AuthStatus | null>(null);
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [authLoading, setAuthLoading] = useState(true);

  // App state
  const [activeView, setActiveView] = useState<ViewType>(getInitialView);
  const [activeSource, setActiveSource] = useState<SourceType>(getInitialSource);
  const [jobs, setJobs] = useState<IndexJob[]>([]);
  const [indexes, setIndexes] = useState<IndexInfo[]>([]);
  const [jobsLoading, setJobsLoading] = useState(true);
  const [indexesLoading, setIndexesLoading] = useState(true);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [indexesError, setIndexesError] = useState<string | null>(null);

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
    if (activeView === 'indexer') {
      params.set('source', activeSource);
    }
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }, [activeView, activeSource]);

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

  // Initial load (only when authenticated and admin for indexer data)
  useEffect(() => {
    if (currentUser && isAdmin) {
      loadJobs();
      loadIndexes();
    }
  }, [currentUser, isAdmin, loadJobs, loadIndexes]);

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
        authStatus={authStatus || { authenticated: false, ldap_configured: false, local_admin_enabled: true }}
        onLoginSuccess={handleLoginSuccess}
      />
    );
  }

  return (
    <>
      <nav className="topnav">
        <span className="topnav-brand">Ragtime</span>
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
        <SettingsPanel />
      ) : activeView === 'tools' ? (
        <ToolsPanel />
      ) : (
        <>
          {/* Indexes List */}
          <IndexesList
            indexes={indexes}
            loading={indexesLoading}
            error={indexesError}
            onDelete={loadIndexes}
            onToggle={loadIndexes}
          />

          {/* Create Index Card */}
          <div className="card">
            <div className="card-header">
              <h2>Create New Index</h2>
              <button
                className="link-btn"
                onClick={() => setActiveSource(activeSource === 'upload' ? 'git' : 'upload')}
              >
                {activeSource === 'upload' ? 'Use Git Repository' : 'Upload Archive'}
              </button>
            </div>

            {activeSource === 'upload' && <UploadForm onJobCreated={handleJobCreated} />}
            {activeSource === 'git' && <GitForm onJobCreated={handleJobCreated} />}
          </div>

          {/* Jobs Table */}
          <JobsTable
            jobs={jobs}
            loading={jobsLoading}
            error={jobsError}
            onJobsChanged={loadJobs}
          />
        </>
      )}
      </div>
    </>
  );
}
