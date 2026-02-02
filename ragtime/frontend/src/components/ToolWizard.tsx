import { useState, useEffect, useRef, useCallback } from 'react';
import { api } from '@/api';
import type { DockerContainer, DockerNetwork } from '@/api';
import type {
  ToolConfig,
  ToolType,
  CreateToolConfigRequest,
  PostgresConnectionConfig,
  MysqlConnectionConfig,
  MssqlConnectionConfig,
  OdooShellConnectionConfig,
  SSHShellConnectionConfig,
  FilesystemConnectionConfig,
  FilesystemMountType,
  VectorStoreType,
  ConnectionConfig,
  MountInfo,
  DirectoryEntry,
  FileTypeStats,
  SolidworksPdmConnectionConfig,
} from '@/types';
import { TOOL_TYPE_INFO, MOUNT_TYPE_INFO } from '@/types';
import { DisabledPopover } from './Popover';
import { Icon, getToolIconType } from './Icon';

// System mounts to filter out from the "Available Mounts" display
// These are internal container mounts not useful for user filesystem indexing
const SYSTEM_MOUNT_PATTERNS = [
  '/root/.ssh',
  '/docker-scripts',
  '/docker-entrypoint',
  '/ragtime/ragtime',
  '/ragtime/requirements.txt',
  '/ragtime/prisma',
  '/ragtime/data',
  '/ragtime/scripts',
  '/data',
  '/var/run/docker.sock',
  '/etc/localtime',
  '/etc/timezone',
  '/app/node_modules',
];

function isSystemMount(containerPath: string): boolean {
  return SYSTEM_MOUNT_PATTERNS.some(pattern =>
    containerPath === pattern || containerPath.startsWith(pattern + '/')
  );
}

// =============================================================================
// Reusable Directory Browser (with path filtering and breadcrumbs)
// =============================================================================

interface DirectoryBrowserProps {
  currentPath: string; // relative path from root ('' is root)
  entries: DirectoryEntry[];
  loading: boolean;
  error?: string | null;
  onNavigate: (path: string) => void; // path relative to root
  onGoUp: () => void;
  onSelect: (path: string) => void;
}

function DirectoryBrowser({ currentPath, entries, loading, error, onNavigate, onGoUp, onSelect }: DirectoryBrowserProps) {
  const [pathInput, setPathInput] = useState('');

  // Clear filter input when navigating to a new directory
  useEffect(() => {
    setPathInput('');
  }, [currentPath]);

  // Filter entries based on current input (only the segment before any "/")
  const filterText = pathInput.split('/')[0];
  const filteredEntries = entries.filter(e => {
    if (!filterText) return true;
    return e.name.toLowerCase().startsWith(filterText.toLowerCase());
  });

  // Handle filter changes and implicit navigation when typing "dir/"
  const handleFilterChange = (value: string) => {
    if (value.endsWith('/')) {
      const dirName = value.slice(0, -1);
      const matchingDir = entries.find(e => e.is_dir && e.name.toLowerCase() === dirName.toLowerCase());
      if (matchingDir) {
        onNavigate(matchingDir.path);
        setPathInput('');
        return;
      }
    }

    if (value.includes('/')) {
      const segments = value.split('/');
      const first = segments[0];
      const rest = segments.slice(1).join('/');
      const matchingDir = entries.find(e => e.is_dir && e.name.toLowerCase() === first.toLowerCase());
      if (matchingDir) {
        onNavigate(matchingDir.path);
        setPathInput(rest);
        return;
      }
    }

    setPathInput(value);
  };

  const handlePathInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const firstDir = filteredEntries.find(e => e.is_dir);
      if (firstDir) {
        onNavigate(firstDir.path);
        setPathInput(''); // Clear filter after navigation
      }
    }
  };

  // Breadcrumb navigation
  const segments = currentPath.split('/').filter(Boolean);

  return (
    <div className="mount-browser">
      <div className="browser-header">
        <button
          type="button"
          className="btn btn-sm"
          onClick={onGoUp}
          disabled={!currentPath || loading}
        >
          ..
        </button>
        <div className="browser-path-wrapper">
          <span className="browser-path-breadcrumbs">
            <span className="breadcrumb-segment">
              <button type="button" className="breadcrumb-btn" onClick={() => onNavigate('')}>
                /
              </button>
            </span>
            {segments.map((segment, idx) => {
              const pathToSegment = segments.slice(0, idx + 1).join('/');
              const isLast = idx === segments.length - 1;
              return (
                <span key={idx} className="breadcrumb-segment">
                  <button
                    type="button"
                    className="breadcrumb-btn"
                    onClick={() => onNavigate(pathToSegment)}
                  >
                    {segment}
                  </button>
                  {!isLast && <span className="breadcrumb-sep">/</span>}
                </span>
              );
            })}
          </span>
          <input
            type="text"
            className="browser-path-input"
            value={pathInput}
            onChange={(e) => handleFilterChange(e.target.value)}
            onKeyDown={handlePathInputKeyDown}
            placeholder="Filter or type dir/"
          />
        </div>
        <button
          type="button"
          className="btn btn-sm btn-primary"
          onClick={() => onSelect(currentPath)}
          disabled={loading}
        >
          Select
        </button>
      </div>

      {error && <div className="browser-error">{error}</div>}

      {loading ? (
        <div className="browser-loading">Loading...</div>
      ) : (
        <div className="browser-entries">
          {filteredEntries.filter(e => e.is_dir).map((entry) => (
            <button
              key={entry.path}
              type="button"
              className="browser-entry"
              onClick={() => onNavigate(entry.path)}
            >
              <span className="entry-icon"><Icon name="folder" size={16} /></span>
              <span className="entry-name">{entry.name}</span>
            </button>
          ))}
          {filteredEntries.filter(e => !e.is_dir).slice(0, 3).map((entry) => (
            <div key={entry.path} className="browser-entry file">
              <span className="entry-icon"><Icon name="file" size={16} /></span>
              <span className="entry-name">{entry.name}</span>
            </div>
          ))}
          {filteredEntries.filter(e => !e.is_dir).length > 3 && (
            <div className="browser-more">
              +{filteredEntries.filter(e => !e.is_dir).length - 3} more files
            </div>
          )}
          {filteredEntries.length === 0 && !loading && (
            <div className="browser-empty">Empty directory</div>
          )}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// Filesystem Browser Component
// =============================================================================

interface FilesystemBrowserProps {
  currentPath: string;
  onSelectPath: (path: string) => void;
}

function FilesystemBrowser({ currentPath, onSelectPath }: FilesystemBrowserProps) {
  const [mounts, setMounts] = useState<MountInfo[]>([]);
  const [entries, setEntries] = useState<DirectoryEntry[]>([]);
  const [browsePath, setBrowsePath] = useState<string>('');
  const [pathInput, setPathInput] = useState<string>(''); // User-typed path for filtering
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dockerExample, setDockerExample] = useState<string>('');
  const [showMountGuide, setShowMountGuide] = useState(false);
  const [expandedMount, setExpandedMount] = useState<string | null>(null); // Which mount is being browsed
  const initializedRef = useRef(false);
  const pathInputRef = useRef<HTMLInputElement>(null);

  // Load mounts on first render
  useEffect(() => {
    if (initializedRef.current) return;
    initializedRef.current = true;

    const loadMounts = async () => {
      try {
        const result = await api.discoverMounts();
        const userMounts = result.mounts.filter(m => !isSystemMount(m.container_path));
        setMounts(userMounts);
        setDockerExample(result.docker_compose_example);
        // If editing with existing path, expand that mount
        if (currentPath) {
          const matchingMount = userMounts.find(m => currentPath.startsWith(m.container_path));
          if (matchingMount) {
            setExpandedMount(matchingMount.container_path);
            setBrowsePath(currentPath);
            setPathInput(''); // Filter starts empty
          }
        }
      } catch (err) {
        console.error('Failed to discover mounts:', err);
      }
    };
    loadMounts();
  }, [currentPath]);

  // Browse current path when expanded
  const browseCurrent = useCallback(async () => {
    if (!browsePath || !expandedMount) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.browseFilesystem(browsePath);
      if (result.error) {
        setError(result.error);
        setEntries([]);
      } else {
        const filteredEntries = result.entries.filter(e => !isSystemMount(e.path));
        setEntries(filteredEntries);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Browse failed');
    } finally {
      setLoading(false);
    }
  }, [browsePath, expandedMount]);

  useEffect(() => {
    browseCurrent();
  }, [browseCurrent]);

  // Helper to get relative path from mount
  const getRelativePath = (fullPath: string, mountPath: string): string => {
    if (fullPath === mountPath) return '';
    return fullPath.slice(mountPath.length).replace(/^\//, '');
  };

  const handleExpandMount = (mountPath: string) => {
    if (expandedMount === mountPath) {
      setExpandedMount(null); // Collapse if clicking same mount
    } else {
      setExpandedMount(mountPath);
      setBrowsePath(mountPath);
      setPathInput(''); // Start with empty relative path
    }
  };

  const handleNavigate = (path: string) => {
    setBrowsePath(path);
    setPathInput(''); // Clear filter when navigating
  };

  // Handle filter input changes - navigate when "/" is typed after a matching dir
  const handleFilterChange = async (value: string) => {
    // Check if user typed a "/" which means they want to navigate into a dir
    if (value.endsWith('/')) {
      const dirName = value.slice(0, -1); // Remove trailing slash
      const matchingDir = entries.find(
        e => e.is_dir && e.name.toLowerCase() === dirName.toLowerCase()
      );
      if (matchingDir) {
        // Navigate into the directory
        setBrowsePath(matchingDir.path);
        setPathInput(''); // Clear filter for new directory
        return;
      }
    }

    // Check for path segments - if there's a "/" in the middle, navigate through
    if (value.includes('/')) {
      const segments = value.split('/');
      const firstSegment = segments[0];
      const restOfPath = segments.slice(1).join('/');

      const matchingDir = entries.find(
        e => e.is_dir && e.name.toLowerCase() === firstSegment.toLowerCase()
      );
      if (matchingDir) {
        // Navigate into first segment and set remaining as filter
        setBrowsePath(matchingDir.path);
        setPathInput(restOfPath);
        return;
      }
    }

    setPathInput(value);
  };

  // Filter entries based on current input (only the part before any "/")
  const filterText = pathInput.split('/')[0];
  const filteredEntries = entries.filter(e => {
    if (!filterText) return true;
    return e.name.toLowerCase().startsWith(filterText.toLowerCase());
  });

  // Handle Enter key to navigate to first matching directory
  const handlePathInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const firstDir = filteredEntries.find(e => e.is_dir);
      if (firstDir) {
        handleNavigate(firstDir.path);
      }
    }
  };

  const handleGoUp = () => {
    const parts = browsePath.split('/').filter(Boolean);
    if (parts.length > 1) {
      parts.pop();
      const newPath = '/' + parts.join('/');
      // Don't go above the mount point
      if (expandedMount && newPath.startsWith(expandedMount.replace(/\/$/, '').split('/').slice(0, -1).join('/') || '/')) {
        setBrowsePath(newPath);
        setPathInput(''); // Clear filter when navigating
      }
    }
  };

  const handleSelect = (path: string) => {
    onSelectPath(path);
    setExpandedMount(null); // Collapse after selection
  };

  // If we have a selection, show compact view
  if (currentPath && !expandedMount && !showMountGuide) {
    return (
      <div className="filesystem-browser">
        <div className="selected-path-display">
          <span className="selected-label">Path:</span>
          <span className="selected-value">{currentPath}</span>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => {
              const matchingMount = mounts.find(m => currentPath.startsWith(m.container_path));
              if (matchingMount) {
                setExpandedMount(matchingMount.container_path);
                setBrowsePath(currentPath);
              } else if (mounts.length > 0) {
                setExpandedMount(mounts[0].container_path);
                setBrowsePath(mounts[0].container_path);
              }
            }}
          >
            Change
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="filesystem-browser">
      {/* Header with guide link */}
      <div className="mounts-header">
        <span className="mounts-title">Select a Docker Mount:</span>
        <button
          type="button"
          className="btn-link"
          onClick={() => setShowMountGuide(!showMountGuide)}
        >
          {showMountGuide ? 'Back to Mounts' : 'How to Add a Mount'}
        </button>
      </div>

      {/* Mount Guide */}
      {showMountGuide && (
        <div className="mount-guide">
          <h4>How to Add a Volume Mount</h4>
          <pre className="code-block">{dockerExample}</pre>
          <p className="field-help">
            After modifying docker-compose.yml, restart the container with:
            <code>docker compose -f docker/docker-compose.dev.yml restart ragtime</code>
          </p>
        </div>
      )}

      {/* Mount list with expandable browser */}
      {!showMountGuide && mounts.length > 0 && (
        <div className="mounts-accordion">
          {mounts.map((mount, i) => (
            <div key={i} className={`mount-item ${expandedMount === mount.container_path ? 'expanded' : ''}`}>
              <button
                type="button"
                className="mount-header"
                onClick={() => handleExpandMount(mount.container_path)}
              >
                <span className="mount-icon">{expandedMount === mount.container_path ? '▼' : '▶'}</span>
                <span className="mount-path">{mount.container_path}</span>
                {mount.read_only && <span className="ro-badge">Read Only</span>}
                {currentPath?.startsWith(mount.container_path) && (
                  <span className="current-badge">Current</span>
                )}
              </button>

              {/* Expanded browser for this mount */}
              {expandedMount === mount.container_path && (
                <div className="mount-browser">
                  <div className="browser-header">
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={handleGoUp}
                      disabled={browsePath === mount.container_path || loading}
                    >
                      ..
                    </button>
                    <div className="browser-path-wrapper">
                      {browsePath !== mount.container_path && (() => {
                        const relativePath = getRelativePath(browsePath, mount.container_path);
                        const segments = relativePath.split('/').filter(Boolean);
                        return (
                          <span className="browser-path-breadcrumbs">
                            {segments.map((segment, idx) => {
                              // Build the path up to this segment
                              const pathToSegment = mount.container_path + '/' + segments.slice(0, idx + 1).join('/');
                              return (
                                <span key={idx} className="breadcrumb-segment">
                                  <button
                                    type="button"
                                    className="breadcrumb-btn"
                                    onClick={() => handleNavigate(pathToSegment)}
                                  >
                                    {segment}
                                  </button>
                                  <span className="breadcrumb-sep">/</span>
                                </span>
                              );
                            })}
                          </span>
                        );
                      })()}
                      <input
                        ref={pathInputRef}
                        type="text"
                        className="browser-path-input"
                        value={pathInput}
                        onChange={(e) => handleFilterChange(e.target.value)}
                        onKeyDown={handlePathInputKeyDown}
                        placeholder="Filter..."
                      />
                    </div>
                    <button
                      type="button"
                      className="btn btn-sm btn-primary"
                      onClick={() => handleSelect(browsePath)}
                    >
                      Select
                    </button>
                  </div>

                  {error && <div className="browser-error">{error}</div>}

                  {loading ? (
                    <div className="browser-loading">Loading...</div>
                  ) : (
                    <div className="browser-entries">
                      {filteredEntries.filter(e => e.is_dir).map((entry) => (
                        <button
                          key={entry.path}
                          type="button"
                          className="browser-entry"
                          onClick={() => handleNavigate(entry.path)}
                        >
                          <span className="entry-icon"><Icon name="folder" size={16} /></span>
                          <span className="entry-name">{entry.name}</span>
                        </button>
                      ))}
                      {filteredEntries.filter(e => !e.is_dir).slice(0, 3).map((entry) => (
                        <div key={entry.path} className="browser-entry file">
                          <span className="entry-icon"><Icon name="file" size={16} /></span>
                          <span className="entry-name">{entry.name}</span>
                        </div>
                      ))}
                      {filteredEntries.filter(e => !e.is_dir).length > 3 && (
                        <div className="browser-more">
                          +{filteredEntries.filter(e => !e.is_dir).length - 3} more files
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* No mounts warning */}
      {!showMountGuide && mounts.length === 0 && (
        <div className="no-mounts-warning">
          <p>No Docker volumes found. Add a volume mount to your docker-compose.yml to index files.</p>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => setShowMountGuide(true)}
          >
            Show Mount Guide
          </button>
        </div>
      )}
    </div>
  );
}


// =============================================================================
// SSH Filesystem Browser Component
// =============================================================================

interface SSHFilesystemBrowserProps {
  currentPath: string;
  onSelectPath: (path: string) => void;
  sshConfig: SSHShellConnectionConfig;
}

function SSHFilesystemBrowser({ currentPath, onSelectPath, sshConfig }: SSHFilesystemBrowserProps) {
  const [entries, setEntries] = useState<DirectoryEntry[]>([]);
  const [browsePath, setBrowsePath] = useState<string>(currentPath || '/');
  const [pathInput, setPathInput] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(!currentPath || currentPath === '/'); // Start expanded if no selection

  const browseCurrent = useCallback(async () => {
    if (!browsePath) return;
    setLoading(true);
    setError(null);
    try {
      const result = await api.browseSSHFilesystem(sshConfig, browsePath);
      if (result.error) {
        setError(result.error);
        setEntries([]);
      } else {
        setEntries(result.entries || []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Browse failed');
    } finally {
      setLoading(false);
    }
  }, [browsePath, sshConfig]);

  useEffect(() => {
    browseCurrent();
  }, [browseCurrent]);

  const handleNavigate = (path: string) => {
    setBrowsePath(path || '/');
    setPathInput(''); // Clear filter when navigating
  };

  const handleGoUp = () => {
    if (!browsePath || browsePath === '/') return;
    const parts = browsePath.split('/').filter(Boolean);
    parts.pop();
    const newPath = '/' + parts.join('/') || '/';
    setBrowsePath(newPath);
    setPathInput('');
  };

  const handleSelect = (path: string) => {
    onSelectPath(path);
    setIsExpanded(false); // Collapse after selection
  };

  // Filter entries based on input
  const filterText = pathInput.split('/')[0];
  const filteredEntries = entries.filter(e => {
    if (!filterText) return true;
    return e.name.toLowerCase().startsWith(filterText.toLowerCase());
  });

  // Handle filter changes and implicit navigation when typing "dir/"
  const handleFilterChange = (value: string) => {
    if (value.endsWith('/')) {
      const dirName = value.slice(0, -1);
      const matchingDir = entries.find(e => e.is_dir && e.name.toLowerCase() === dirName.toLowerCase());
      if (matchingDir) {
        handleNavigate(matchingDir.path);
        return;
      }
    }
    if (value.includes('/')) {
      const segments = value.split('/');
      const first = segments[0];
      const matchingDir = entries.find(e => e.is_dir && e.name.toLowerCase() === first.toLowerCase());
      if (matchingDir) {
        handleNavigate(matchingDir.path);
        return;
      }
    }
    setPathInput(value);
  };

  const handlePathInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      const firstDir = filteredEntries.find(e => e.is_dir);
      if (firstDir) {
        handleNavigate(firstDir.path);
      }
    }
  };

  // Get path segments for breadcrumbs
  const segments = browsePath.split('/').filter(Boolean);

  // Display path for the accordion header
  const displayPath = currentPath && currentPath !== '/' ? currentPath : browsePath;

  return (
    <div className="filesystem-browser">
      <div className="mounts-accordion">
        <div className={`mount-item ${isExpanded ? 'expanded' : ''}`}>
          <button
            type="button"
            className="mount-header"
            onClick={() => {
              if (!isExpanded) {
                setBrowsePath(currentPath || '/');
              }
              setIsExpanded(!isExpanded);
            }}
          >
            <span className="mount-icon">{isExpanded ? '▼' : '▶'}</span>
            <span className="mount-path">{displayPath}</span>
            {currentPath && currentPath !== '/' && (
              <span className="current-badge">Selected</span>
            )}
          </button>

          {/* Expanded browser */}
          {isExpanded && (
            <div className="mount-browser">
              <div className="browser-header">
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={handleGoUp}
                  disabled={browsePath === '/' || loading}
                >
                  ..
                </button>
                <div className="browser-path-wrapper">
                  <span className="browser-path-breadcrumbs">
                    <span className="breadcrumb-segment">
                      <button type="button" className="breadcrumb-btn" onClick={() => handleNavigate('/')}>
                        /
                      </button>
                    </span>
                    {segments.map((segment, idx) => {
                      const pathToSegment = '/' + segments.slice(0, idx + 1).join('/');
                      const isLast = idx === segments.length - 1;
                      return (
                        <span key={idx} className="breadcrumb-segment">
                          <button
                            type="button"
                            className="breadcrumb-btn"
                            onClick={() => handleNavigate(pathToSegment)}
                          >
                            {segment}
                          </button>
                          {!isLast && <span className="breadcrumb-sep">/</span>}
                        </span>
                      );
                    })}
                  </span>
                  <input
                    type="text"
                    className="browser-path-input"
                    value={pathInput}
                    onChange={(e) => handleFilterChange(e.target.value)}
                    onKeyDown={handlePathInputKeyDown}
                    placeholder="Filter..."
                  />
                </div>
                <button
                  type="button"
                  className="btn btn-sm btn-primary"
                  onClick={() => handleSelect(browsePath)}
                  disabled={loading}
                >
                  Select
                </button>
              </div>

              {error && <div className="browser-error">{error}</div>}

              {loading ? (
                <div className="browser-loading">Loading...</div>
              ) : (
                <div className="browser-entries">
                  {filteredEntries.filter(e => e.is_dir).map((entry) => (
                    <button
                      key={entry.path}
                      type="button"
                      className="browser-entry"
                      onClick={() => handleNavigate(entry.path)}
                    >
                      <span className="entry-icon"><Icon name="folder" size={16} /></span>
                      <span className="entry-name">{entry.name}</span>
                    </button>
                  ))}
                  {filteredEntries.length === 0 && !loading && (
                    <div className="browser-empty">Empty directory</div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// =============================================================================
// NFS Browser Component
// =============================================================================

interface NFSBrowserProps {
  host: string;
  selectedExport: string;
  selectedPath: string;
  onHostChange: (host: string) => void;
  onSelectPath: (exportPath: string, relativePath: string) => void;
}

function NFSBrowser({ host, selectedExport, selectedPath, onHostChange, onSelectPath }: NFSBrowserProps) {
  const [exports, setExports] = useState<import('@/types').NFSExport[]>([]);
  const [entries, setEntries] = useState<DirectoryEntry[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [browseError, setBrowseError] = useState<string | null>(null);
  const [expandedExport, setExpandedExport] = useState<string | null>(null);
  const [browsePath, setBrowsePath] = useState<string>('');
  const [manualPath, setManualPath] = useState<string>('');
  const [hostInput, setHostInput] = useState(host || '');

  // Discover exports when host changes
  const handleDiscover = async () => {
    if (!hostInput.trim()) return;
    setDiscovering(true);
    setError(null);
    setExports([]);

    try {
      const result = await api.discoverNfsExports(hostInput.trim());
      if (result.success) {
        setExports(result.exports);
        onHostChange(hostInput.trim());
        if (result.exports.length === 0) {
          setError('No exports found on this server');
        }
      } else {
        setError(result.error || 'Discovery failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setDiscovering(false);
    }
  };

  // Browse export contents
  const browseExport = useCallback(async (exportPath: string, path: string = '') => {
    if (!host) return;
    setBrowsing(true);
    setBrowseError(null);

    try {
      const result = await api.browseNfsExport(host, exportPath, path);
      if (result.error) {
        // Allow manual path entry on any browse error
        setBrowseError(result.error);
        setEntries([]);
      } else {
        setEntries(result.entries);
        setBrowseError(null);
      }
    } catch (err) {
      setBrowseError(err instanceof Error ? err.message : 'Browse failed');
    } finally {
      setBrowsing(false);
    }
  }, [host]);

  const handleExpandExport = (exportPath: string) => {
    if (expandedExport === exportPath) {
      setExpandedExport(null);
    } else {
      setExpandedExport(exportPath);
      setBrowsePath('');
      browseExport(exportPath, '');
    }
  };

  const handleNavigate = (path: string) => {
    setBrowsePath(path);
    if (expandedExport) {
      browseExport(expandedExport, path);
    }
  };

  const handleGoUp = () => {
    const parts = browsePath.split('/').filter(Boolean);
    if (parts.length > 0) {
      parts.pop();
      const newPath = parts.join('/');
      handleNavigate(newPath);
    }
  };

  const handleSelect = () => {
    if (expandedExport) {
      onSelectPath(expandedExport, browsePath);
    }
  };

  // Show compact view if already selected
  if (selectedExport && !expandedExport) {
    return (
      <div className="filesystem-browser">
        <div className="selected-path-display">
          <span className="selected-label">NFS:</span>
          <span className="selected-value">{host}:{selectedExport}{selectedPath ? `/${selectedPath}` : ''}</span>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => {
              setExpandedExport(selectedExport);
              setBrowsePath(selectedPath);
              browseExport(selectedExport, selectedPath);
            }}
          >
            Change
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="filesystem-browser">
      {/* Host input */}
      <div className="connection-test-row" style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          value={hostInput}
          onChange={(e) => setHostInput(e.target.value)}
          placeholder="nfs-server.local"
          onKeyDown={(e) => e.key === 'Enter' && handleDiscover()}
        />
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleDiscover}
          disabled={discovering || !hostInput.trim()}
        >
          {discovering ? 'Discovering...' : 'Discover Exports'}
        </button>
      </div>

      {error && <div className="browser-error">{error}</div>}

      {/* Export list */}
      {exports.length > 0 && (
        <div className="mounts-accordion">
          {exports.map((exp, i) => (
            <div key={i} className={`mount-item ${expandedExport === exp.export_path ? 'expanded' : ''}`}>
              <button
                type="button"
                className="mount-header"
                onClick={() => handleExpandExport(exp.export_path)}
              >
                <span className="mount-icon">{expandedExport === exp.export_path ? '▼' : '▶'}</span>
                <span className="mount-path">{exp.export_path}</span>
                <span className="export-hosts">{exp.allowed_hosts}</span>
              </button>

              {expandedExport === exp.export_path && (
                <div className="mount-browser">
                  {browseError ? (
                    <div className="manual-path-entry">
                      <div className="browser-warning" style={{ marginBottom: '0.75rem' }}>
                        {browseError}
                      </div>
                      <div className="connection-test-row">
                        <input
                          type="text"
                          value={manualPath}
                          onChange={(e) => setManualPath(e.target.value)}
                          placeholder="/ or /subdir/path"
                          style={{ flex: 1 }}
                        />
                        <button
                          type="button"
                          className="btn btn-sm btn-primary"
                          onClick={() => onSelectPath(exp.export_path, manualPath.replace(/^\/+/, ''))}
                        >
                          Select Path
                        </button>
                      </div>
                    </div>
                  ) : (
                    <DirectoryBrowser
                      currentPath={browsePath}
                      entries={entries}
                      loading={browsing}
                      error={browseError}
                      onNavigate={handleNavigate}
                      onGoUp={handleGoUp}
                      onSelect={handleSelect}
                    />
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// =============================================================================
// SMB Browser Component
// =============================================================================

interface SMBBrowserProps {
  host: string;
  user: string;
  password: string;
  domain: string;
  selectedShare: string;
  selectedPath: string;
  onHostChange: (host: string) => void;
  onCredentialsChange: (user: string, password: string, domain: string) => void;
  onSelectPath: (share: string, relativePath: string) => void;
}

function SMBBrowser({
  host, user, password, domain,
  selectedShare, selectedPath,
  onHostChange, onCredentialsChange, onSelectPath
}: SMBBrowserProps) {
  const [shares, setShares] = useState<import('@/types').SMBShare[]>([]);
  const [entries, setEntries] = useState<DirectoryEntry[]>([]);
  const [discovering, setDiscovering] = useState(false);
  const [browsing, setBrowsing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expandedShare, setExpandedShare] = useState<string | null>(null);
  const [browsePath, setBrowsePath] = useState<string>('');

  const [hostInput, setHostInput] = useState(host || '');
  const [userInput, setUserInput] = useState(user || '');
  const [passwordInput, setPasswordInput] = useState(password || '');
  const [domainInput, setDomainInput] = useState(domain || '');

  // Discover shares
  const handleDiscover = async () => {
    if (!hostInput.trim()) return;
    setDiscovering(true);
    setError(null);
    setShares([]);

    try {
      const result = await api.discoverSmbShares(hostInput.trim(), userInput, passwordInput, domainInput);
      if (result.success) {
        setShares(result.shares);
        onHostChange(hostInput.trim());
        onCredentialsChange(userInput, passwordInput, domainInput);
        if (result.shares.length === 0) {
          setError('No shares found (or no access with provided credentials)');
        }
      } else {
        setError(result.error || 'Discovery failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setDiscovering(false);
    }
  };

  // Browse share contents - use local state for fresh values after discovery
  const browseShare = useCallback(async (shareName: string, path: string = '') => {
    const currentHost = hostInput.trim() || host;
    if (!currentHost) return;
    setBrowsing(true);
    setError(null);

    try {
      const result = await api.browseSmbShare(currentHost, shareName, path, userInput || user, passwordInput || password, domainInput || domain);
      if (result.error) {
        setError(result.error);
        setEntries([]);
      } else {
        setEntries(result.entries);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Browse failed');
    } finally {
      setBrowsing(false);
    }
  }, [host, user, password, domain, hostInput, userInput, passwordInput, domainInput]);

  const handleExpandShare = (shareName: string) => {
    if (expandedShare === shareName) {
      setExpandedShare(null);
    } else {
      setExpandedShare(shareName);
      setBrowsePath('');
      browseShare(shareName, '');
    }
  };

  const handleNavigate = (path: string) => {
    setBrowsePath(path);
    if (expandedShare) {
      browseShare(expandedShare, path);
    }
  };

  const handleGoUp = () => {
    const parts = browsePath.split('/').filter(Boolean);
    if (parts.length > 0) {
      parts.pop();
      const newPath = parts.join('/');
      handleNavigate(newPath);
    }
  };

  const handleSelect = (path: string) => {
    if (expandedShare) {
      onSelectPath(expandedShare, path);
      // Collapse the browser to show compact view
      setExpandedShare(null);
    }
  };

  // Compact view if already selected
  if (selectedShare && !expandedShare) {
    return (
      <div className="filesystem-browser">
        <div className="selected-path-display">
          <span className="selected-label">SMB:</span>
          <span className="selected-value">//{host}/{selectedShare}{selectedPath ? `/${selectedPath}` : ''}</span>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => {
              setExpandedShare(selectedShare);
              setBrowsePath(selectedPath);
              browseShare(selectedShare, selectedPath);
            }}
          >
            Change
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="filesystem-browser">
      {/* Connection inputs - Row 1: Host, Username */}
      <div className="form-row" style={{ marginBottom: '0.5rem' }}>
        <div className="form-group" style={{ flex: 2, marginBottom: 0 }}>
          <input
            type="text"
            value={hostInput}
            onChange={(e) => setHostInput(e.target.value)}
            placeholder="fileserver.local"
          />
        </div>
        <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            placeholder="Username"
          />
        </div>
      </div>
      {/* Row 2: Password, Domain */}
      <div className="form-row" style={{ marginBottom: '0.5rem' }}>
        <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
          <input
            type="password"
            value={passwordInput}
            onChange={(e) => setPasswordInput(e.target.value)}
            placeholder="Password"
          />
        </div>
        <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
          <input
            type="text"
            value={domainInput}
            onChange={(e) => setDomainInput(e.target.value)}
            placeholder="Domain (optional)"
          />
        </div>
      </div>
      {/* Row 3: Discover button */}
      <div style={{ marginBottom: '1rem' }}>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={handleDiscover}
          disabled={discovering || !hostInput.trim()}
        >
          {discovering ? 'Discovering...' : 'Discover Shares'}
        </button>
      </div>

      {error && <div className="browser-error">{error}</div>}

      {/* Share list */}
      {shares.length > 0 && (
        <div className="mounts-accordion">
          {shares.map((share, i) => (
            <div key={i} className={`mount-item ${expandedShare === share.name ? 'expanded' : ''}`}>
              <button
                type="button"
                className="mount-header"
                onClick={() => handleExpandShare(share.name)}
              >
                <span className="mount-icon">{expandedShare === share.name ? '▼' : '▶'}</span>
                <span className="mount-path">{share.name}</span>
                {share.comment && <span className="share-comment">{share.comment}</span>}
              </button>

              {expandedShare === share.name && (
                <DirectoryBrowser
                  currentPath={browsePath}
                  entries={entries}
                  loading={browsing}
                  error={error}
                  onNavigate={handleNavigate}
                  onGoUp={handleGoUp}
                  onSelect={handleSelect}
                />
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}


// =============================================================================
// Shared Docker connection panel props
// =============================================================================

interface DockerConnectionPanelProps {
  // State
  dockerContainers: DockerContainer[];
  dockerNetworks: DockerNetwork[];
  currentNetwork: string | null;
  currentContainer: string | null;
  loadingDocker: boolean;
  connectingNetwork: boolean;
  // Current values
  selectedNetwork: string;
  selectedContainer: string;
  // Handlers
  onDiscoverDocker: () => void;
  onConnectNetwork: (e: React.MouseEvent, networkName: string) => void;
  onNetworkChange: (network: string) => void;
  onContainerChange: (container: string) => void;
  // Configuration
  containerFilter?: (container: DockerContainer) => boolean;
  containerLabel?: (container: DockerContainer) => string;
  containerCountLabel: string;
  containerHelpText: string;
  fallbackPlaceholder: string;
}

// Reusable Docker connection panel component
function DockerConnectionPanel({
  dockerContainers,
  dockerNetworks,
  currentNetwork,
  currentContainer,
  loadingDocker,
  connectingNetwork,
  selectedNetwork,
  selectedContainer,
  onDiscoverDocker,
  onConnectNetwork,
  onNetworkChange,
  onContainerChange,
  containerFilter = () => true,
  containerLabel = (c) => c.name,
  containerCountLabel,
  containerHelpText,
  fallbackPlaceholder,
}: DockerConnectionPanelProps) {
  const filteredContainers = dockerContainers.filter(containerFilter);
  const networkContainers = dockerContainers.filter(
    c => currentNetwork && c.networks.includes(currentNetwork) && c.name !== currentContainer
  );

  return (
    <>
      {/* Discover Docker button */}
      <div className="form-group">
        <label>Discover Docker Environment</label>
        <button
          type="button"
          className="btn btn-secondary"
          onClick={onDiscoverDocker}
          disabled={loadingDocker}
          style={{ width: '100%' }}
        >
          {loadingDocker ? 'Scanning...' : dockerContainers.length > 0 ? 'Refresh' : 'Discover Containers'}
        </button>
        <p className="field-help">
          {dockerContainers.length > 0
            ? `Found ${filteredContainers.length} ${containerCountLabel} across ${dockerNetworks.length} network(s).`
            : `Scan for Docker containers running ${containerCountLabel.replace(/\(s\)$/, '')}.`}
        </p>
      </div>

      {/* Network selection (after discovery) */}
      {dockerNetworks.length > 0 && (
        <div className="form-group">
          <label>Docker Network</label>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <select
              value={selectedNetwork}
              onChange={(e) => onNetworkChange(e.target.value)}
              style={{ flex: 1 }}
            >
              <option value="">Select network...</option>
              {dockerNetworks.map(n => (
                <option key={n.name} value={n.name}>
                  {n.name} ({n.containers.length})
                  {n.name === currentNetwork ? ' - connected' : ''}
                </option>
              ))}
            </select>
            {selectedNetwork && selectedNetwork !== currentNetwork && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={(e) => onConnectNetwork(e, selectedNetwork)}
                disabled={connectingNetwork}
              >
                {connectingNetwork ? '...' : 'Connect'}
              </button>
            )}
          </div>
          <p className="field-help">
            {currentNetwork
              ? `Connected to ${currentNetwork}.`
              : 'Select a network and click Connect to access containers.'}
          </p>
        </div>
      )}

      {/* Container selection (after connected) */}
      {currentNetwork && (
        <div className="form-group">
          <label>Container Name</label>
          {networkContainers.length > 0 ? (
            <select
              value={selectedContainer}
              onChange={(e) => onContainerChange(e.target.value)}
            >
              <option value="">Select container...</option>
              {networkContainers.map(c => (
                <option key={c.name} value={c.name}>
                  {containerLabel(c)}
                </option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={selectedContainer}
              onChange={(e) => onContainerChange(e.target.value)}
              placeholder="No containers found on this network"
            />
          )}
          <p className="field-help">{containerHelpText}</p>
        </div>
      )}

      {/* Fallback: Manual container name if no networks discovered */}
      {dockerNetworks.length === 0 && (
        <div className="form-group">
          <label>Container Name</label>
          <input
            type="text"
            value={selectedContainer}
            onChange={(e) => onContainerChange(e.target.value)}
            placeholder={fallbackPlaceholder}
          />
          <p className="field-help">
            Enter the Docker container name manually, or click Discover above to find containers.
          </p>
        </div>
      )}
    </>
  );
}

// =============================================================================
// Reusable SSH Authentication Panel
// =============================================================================

interface SSHAuthConfig {
  host: string;
  port: number;
  user: string;
  key_path?: string;
  key_content?: string;
  public_key?: string;
  key_passphrase?: string;
  password?: string;
}

type SSHAuthMode = 'generate' | 'upload' | 'path' | 'password';

interface SSHAuthPanelProps {
  config: SSHAuthConfig;
  onConfigChange: (config: SSHAuthConfig) => void;
  authMode: SSHAuthMode;
  onAuthModeChange: (mode: SSHAuthMode) => void;
  generatingKey: boolean;
  onGenerateKey: () => void;
  keyCopied: boolean;
  onCopyPublicKey: () => void;
  toolName?: string;
  showHostPort?: boolean;  // Whether to show host/port fields (true for generic SSH, false for Odoo which shows them separately)
}

function SSHAuthPanel({
  config,
  onConfigChange,
  authMode,
  onAuthModeChange,
  generatingKey,
  onGenerateKey,
  keyCopied,
  onCopyPublicKey,
  toolName: _toolName = 'ragtime',
  showHostPort = false,
}: SSHAuthPanelProps) {
  return (
    <div className="ssh-auth-panel">
      {showHostPort && (
        <>
          <div className="form-row" style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: '3 1 200px', minWidth: '150px' }}>
              <label>SSH Host *</label>
              <input
                type="text"
                value={config.host || ''}
                onChange={(e) => onConfigChange({ ...config, host: e.target.value })}
                placeholder="server.example.com"
              />
            </div>
            <div className="form-group" style={{ flex: '1 1 80px', minWidth: '80px', maxWidth: '120px' }}>
              <label>SSH Port</label>
              <input
                type="number"
                value={config.port || 22}
                onChange={(e) => onConfigChange({ ...config, port: parseInt(e.target.value) || 22 })}
                min={1}
                max={65535}
              />
            </div>
            <div className="form-group" style={{ flex: '2 1 120px', minWidth: '100px' }}>
              <label>SSH User *</label>
              <input
                type="text"
                value={config.user || ''}
                onChange={(e) => onConfigChange({ ...config, user: e.target.value })}
                placeholder="ubuntu"
              />
            </div>
          </div>
        </>
      )}

      {/* SSH Authentication Method */}
      <div className="ssh-key-section">
        <label>SSH Authentication</label>
        <div className="ssh-key-tabs">
          <button
            type="button"
            className={`ssh-key-tab ${authMode === 'generate' ? 'active' : ''}`}
            onClick={() => onAuthModeChange('generate')}
          >
            Generate Key
          </button>
          <button
            type="button"
            className={`ssh-key-tab ${authMode === 'upload' ? 'active' : ''}`}
            onClick={() => onAuthModeChange('upload')}
          >
            Paste Key
          </button>
          <button
            type="button"
            className={`ssh-key-tab ${authMode === 'path' ? 'active' : ''}`}
            onClick={() => onAuthModeChange('path')}
          >
            File Path
          </button>
          <button
            type="button"
            className={`ssh-key-tab ${authMode === 'password' ? 'active' : ''}`}
            onClick={() => onAuthModeChange('password')}
          >
            Password
          </button>
        </div>

        {authMode === 'generate' && (
          <div className="ssh-key-panel">
            <p className="field-help">
              Generate a new SSH keypair. The private key will be stored securely with this tool configuration.
              Copy the public key to the remote server's <code>~/.ssh/authorized_keys</code>.
            </p>
            <div className="form-group" style={{ marginTop: '0.5rem' }}>
              <label>Key Passphrase (optional)</label>
              <input
                type="password"
                value={config.key_passphrase || ''}
                onChange={(e) => onConfigChange({ ...config, key_passphrase: e.target.value })}
                placeholder="Leave blank for no passphrase"
              />
              <p className="field-help">
                {config.key_content
                  ? 'Enter a new passphrase to regenerate the key, or leave blank for no passphrase.'
                  : 'Optionally encrypt the private key with a passphrase.'}
              </p>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={onGenerateKey}
              disabled={generatingKey}
              style={{ marginTop: '0.5rem' }}
            >
              {generatingKey ? 'Generating...' : (config.key_content ? 'Regenerate SSH Keypair' : 'Generate SSH Keypair')}
            </button>
            {config.public_key && (
              <div className="public-key-display" style={{ marginTop: '1rem' }}>
                <label>Public Key (add to remote server):</label>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                  <textarea
                    readOnly
                    value={config.public_key}
                    rows={3}
                    style={{ flex: 1, fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
                  />
                  <button
                    type="button"
                    className={`btn ${keyCopied ? 'btn-success' : 'btn-secondary'}`}
                    onClick={onCopyPublicKey}
                    title="Copy to clipboard"
                    style={keyCopied ? { backgroundColor: '#28a745', borderColor: '#28a745', color: 'white' } : undefined}
                  >
                    {keyCopied ? <><Icon name="check" size={14} /> Copied!</> : 'Copy'}
                  </button>
                </div>
              </div>
            )}
            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label>SSH Password (optional)</label>
              <input
                type="password"
                value={config.password || ''}
                onChange={(e) => onConfigChange({ ...config, password: e.target.value })}
                placeholder="Leave blank if not required"
              />
              <p className="field-help">Some servers require both SSH key and password. Fill in both if needed.</p>
            </div>
          </div>
        )}

        {authMode === 'upload' && (
          <div className="ssh-key-panel">
            <div className="form-group">
              <label>Private Key Content</label>
              <textarea
                value={config.key_content || ''}
                onChange={(e) => onConfigChange({ ...config, key_content: e.target.value, key_path: '' })}
                placeholder="-----BEGIN RSA PRIVATE KEY-----&#10;...&#10;-----END RSA PRIVATE KEY-----"
                rows={6}
                style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
              />
              <p className="field-help">Paste your SSH private key content here.</p>
            </div>
            <div className="form-group">
              <label>Key Passphrase (optional)</label>
              <input
                type="password"
                value={config.key_passphrase || ''}
                onChange={(e) => onConfigChange({ ...config, key_passphrase: e.target.value })}
                placeholder="Leave blank if key is not encrypted"
              />
              <p className="field-help">If your private key is encrypted with a passphrase, enter it here.</p>
            </div>
            {config.public_key && (
              <div className="form-group">
                <label>Public Key (for reference):</label>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
                  <textarea
                    readOnly
                    value={config.public_key}
                    rows={2}
                    style={{ flex: 1, fontFamily: 'var(--font-mono)', fontSize: '0.8rem' }}
                  />
                  <button
                    type="button"
                    className={`btn ${keyCopied ? 'btn-success' : 'btn-secondary'}`}
                    onClick={onCopyPublicKey}
                    style={keyCopied ? { backgroundColor: '#28a745', borderColor: '#28a745', color: 'white' } : undefined}
                  >
                    {keyCopied ? <><Icon name="check" size={14} /> Copied!</> : 'Copy'}
                  </button>
                </div>
              </div>
            )}
            <div className="form-group">
              <label>SSH Password (optional)</label>
              <input
                type="password"
                value={config.password || ''}
                onChange={(e) => onConfigChange({ ...config, password: e.target.value })}
                placeholder="Leave blank if not required"
              />
              <p className="field-help">Some servers require both SSH key and password. Fill in both if needed.</p>
            </div>
          </div>
        )}

        {authMode === 'path' && (
          <div className="ssh-key-panel">
            <div className="form-group">
              <label>SSH Key File Path</label>
              <input
                type="text"
                value={config.key_path || ''}
                onChange={(e) => onConfigChange({ ...config, key_path: e.target.value, key_content: '' })}
                placeholder="/root/.ssh/id_rsa"
              />
              <p className="field-help">
                Path to SSH private key file inside the ragtime container.
                Host keys from ~/.ssh are mounted at /root/.ssh/
              </p>
            </div>
            <div className="form-group">
              <label>Key Passphrase (optional)</label>
              <input
                type="password"
                value={config.key_passphrase || ''}
                onChange={(e) => onConfigChange({ ...config, key_passphrase: e.target.value })}
                placeholder="Leave blank if key is not encrypted"
              />
              <p className="field-help">If your private key is encrypted with a passphrase, enter it here.</p>
            </div>
            <div className="form-group">
              <label>SSH Password (optional)</label>
              <input
                type="password"
                value={config.password || ''}
                onChange={(e) => onConfigChange({ ...config, password: e.target.value })}
                placeholder="Leave blank if not required"
              />
              <p className="field-help">Some servers require both SSH key and password. Fill in both if needed.</p>
            </div>
          </div>
        )}

        {authMode === 'password' && (
          <div className="ssh-key-panel">
            <div className="form-group">
              <label>SSH Password</label>
              <input
                type="password"
                value={config.password || ''}
                onChange={(e) => onConfigChange({ ...config, password: e.target.value, key_path: '', key_content: '' })}
                placeholder="Enter SSH password"
              />
              <p className="field-help">Use password-only authentication (no SSH key).</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// SSH Tunnel Panel - Reusable component for database SSH tunnel configuration
// =============================================================================

interface SSHTunnelPanelProps {
  enabled: boolean;
  onEnabledChange: (enabled: boolean) => void;
  config: {
    ssh_tunnel_host?: string;
    ssh_tunnel_port?: number;
    ssh_tunnel_user?: string;
    ssh_tunnel_password?: string;
    ssh_tunnel_key_path?: string;
    ssh_tunnel_key_content?: string;
    ssh_tunnel_key_passphrase?: string;
    ssh_tunnel_public_key?: string;
  };
  onConfigChange: (config: SSHTunnelPanelProps['config']) => void;
  databaseLabel?: string; // e.g., "MySQL", "PostgreSQL"
  authMode: SSHAuthMode;
  onAuthModeChange: (mode: SSHAuthMode) => void;
  generatingKey: boolean;
  onGenerateKey: () => void;
  keyCopied: boolean;
  onCopyPublicKey: () => void;
  toolName?: string;
}

function SSHTunnelPanel({
  enabled,
  onEnabledChange,
  config,
  onConfigChange,
  databaseLabel = 'database',
  authMode,
  onAuthModeChange,
  generatingKey,
  onGenerateKey,
  keyCopied,
  onCopyPublicKey,
  toolName = 'tunnel',
}: SSHTunnelPanelProps) {
  return (
    <div className="ssh-tunnel-section" style={{ marginBottom: '1rem' }}>
      <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer', marginBottom: '0.5rem' }}>
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onEnabledChange(e.target.checked)}
          style={{ width: 'auto', margin: 0 }}
        />
        <span style={{ fontWeight: 500 }}>Connect via SSH Tunnel</span>
      </label>
      <p className="field-help" style={{ marginTop: 0, marginBottom: enabled ? '1rem' : 0 }}>
        Use an SSH tunnel to connect to {databaseLabel} servers that are not directly accessible (e.g., behind a firewall or bound to localhost).
      </p>

      {enabled && (
        <>
          {/* SSH Server Connection - Host, Port, User in one row */}
          <div className="form-row" style={{ display: 'flex', flexDirection: 'row', flexWrap: 'wrap' }}>
            <div className="form-group" style={{ flex: '3 1 200px', minWidth: '150px' }}>
              <label>SSH Host *</label>
              <input
                type="text"
                value={config.ssh_tunnel_host || ''}
                onChange={(e) => onConfigChange({ ...config, ssh_tunnel_host: e.target.value })}
                placeholder="ssh.example.com"
              />
            </div>
            <div className="form-group" style={{ flex: '1 1 80px', minWidth: '80px', maxWidth: '120px' }}>
              <label>SSH Port</label>
              <input
                type="number"
                value={config.ssh_tunnel_port || 22}
                onChange={(e) => onConfigChange({ ...config, ssh_tunnel_port: parseInt(e.target.value) || 22 })}
                min={1}
                max={65535}
              />
            </div>
            <div className="form-group" style={{ flex: '2 1 120px', minWidth: '100px' }}>
              <label>SSH User *</label>
              <input
                type="text"
                value={config.ssh_tunnel_user || ''}
                onChange={(e) => onConfigChange({ ...config, ssh_tunnel_user: e.target.value })}
                placeholder="root"
              />
            </div>
          </div>

          {/* SSH Authentication - reuse SSHAuthPanel */}
          <SSHAuthPanel
            config={{
              host: config.ssh_tunnel_host || '',
              port: config.ssh_tunnel_port || 22,
              user: config.ssh_tunnel_user || '',
              key_path: config.ssh_tunnel_key_path || '',
              key_content: config.ssh_tunnel_key_content || '',
              public_key: config.ssh_tunnel_public_key || '',
              key_passphrase: config.ssh_tunnel_key_passphrase || '',
              password: config.ssh_tunnel_password || '',
            }}
            onConfigChange={(sshAuthConfig) => onConfigChange({
              ...config,
              ssh_tunnel_key_path: sshAuthConfig.key_path || '',
              ssh_tunnel_key_content: sshAuthConfig.key_content || '',
              ssh_tunnel_public_key: sshAuthConfig.public_key || '',
              ssh_tunnel_key_passphrase: sshAuthConfig.key_passphrase || '',
              ssh_tunnel_password: sshAuthConfig.password || '',
            })}
            authMode={authMode}
            onAuthModeChange={onAuthModeChange}
            generatingKey={generatingKey}
            onGenerateKey={onGenerateKey}
            keyCopied={keyCopied}
            onCopyPublicKey={onCopyPublicKey}
            toolName={toolName}
            showHostPort={false}
          />
        </>
      )}
    </div>
  );
}

interface ToolWizardProps {
  existingTool: ToolConfig | null;
  onClose: () => void;
  onSave: () => void;
  defaultToolType?: ToolType;
  /** When true, renders without card wrapper and header (for embedding in other panels) */
  embedded?: boolean;
}

type WizardStep = 'type' | 'connection' | 'pdm_filtering' | 'execution_constraints' | 'description' | 'options' | 'review';

// Base steps - pdm_filtering is dynamically inserted for solidworks_pdm tools
const BASE_WIZARD_STEPS: WizardStep[] = ['type', 'connection', 'description', 'options', 'review'];
const PDM_WIZARD_STEPS: WizardStep[] = ['type', 'connection', 'pdm_filtering', 'description', 'options', 'review'];
// SSH tools combine options into the execution_constraints step
const SSH_WIZARD_STEPS: WizardStep[] = ['type', 'connection', 'execution_constraints', 'description', 'review'];
// Odoo tools show options before description for logical flow
const ODOO_WIZARD_STEPS: WizardStep[] = ['type', 'connection', 'options', 'description', 'review'];
// Filesystem indexers don't need execution options (read-only background job)
const FILESYSTEM_WIZARD_STEPS: WizardStep[] = ['type', 'connection', 'description', 'review'];

function getStepTitle(step: WizardStep): string {
  switch (step) {
    case 'type':
      return 'Select Tool Type';
    case 'connection':
      return 'Configure Connection';
    case 'pdm_filtering':
      return 'Document Filtering';
    case 'execution_constraints':
      return 'Execution Options';
    case 'description':
      return 'Add Description';
    case 'options':
      return 'Execution Options';
    case 'review':
      return 'Review & Save';
  }
}

export function ToolWizard({ existingTool, onClose, onSave, defaultToolType, embedded = false }: ToolWizardProps) {
  const isEditing = existingTool !== null;
  const progressRef = useRef<HTMLDivElement>(null);

  // Form state - use defaultToolType if provided
  const [toolType, setToolType] = useState<ToolType>(existingTool?.tool_type || defaultToolType || 'ssh_shell');

  // Get the appropriate wizard steps based on tool type
  const getWizardSteps = useCallback((): WizardStep[] => {
    let steps = BASE_WIZARD_STEPS;
    if (toolType === 'solidworks_pdm') steps = PDM_WIZARD_STEPS;
    else if (toolType === 'ssh_shell') steps = SSH_WIZARD_STEPS;
    else if (toolType === 'odoo_shell') steps = ODOO_WIZARD_STEPS;
    else if (toolType === 'filesystem_indexer') steps = FILESYSTEM_WIZARD_STEPS;

    // Skip description step in edit mode (handled inline)
    if (isEditing) {
      return steps.filter(step => step !== 'description');
    }
    return steps;
  }, [toolType, isEditing]);

  // Wizard state - skip type selection if defaultToolType is provided
  const skipTypeStep = !isEditing && defaultToolType !== undefined;
  const [currentStep, setCurrentStep] = useState<WizardStep>(
    isEditing ? 'connection' : (skipTypeStep ? 'connection' : 'type')
  );
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string; details?: unknown } | null>(null);
  const [error, setError] = useState<string | null>(null);

  // SQL tools dropdown state
  const [sqlToolsExpanded, setSqlToolsExpanded] = useState(false);

  // Auto-scroll active step into view
  useEffect(() => {
    const activeStep = progressRef.current?.querySelector('.wizard-step.active');
    if (activeStep) {
      activeStep.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }, [currentStep]);
  const [name, setName] = useState(existingTool?.name || '');
  const [description, setDescription] = useState(existingTool?.description || '');
  const [maxResults, setMaxResults] = useState(existingTool?.max_results || 100);
  const [timeoutValue, setTimeoutValue] = useState(existingTool?.timeout || 30);
  const [allowWrite, setAllowWrite] = useState(existingTool?.allow_write || false);

  // Connection config state
  const [pgConnectionMode, setPgConnectionMode] = useState<'direct' | 'container'>(
    existingTool?.tool_type === 'postgres' && (existingTool.connection_config as PostgresConnectionConfig).container
      ? 'container'
      : 'direct'
  );
  const [postgresConfig, setPostgresConfig] = useState<PostgresConnectionConfig>(
    existingTool?.tool_type === 'postgres'
      ? (existingTool.connection_config as PostgresConnectionConfig)
      : { host: '', port: 5432, user: '', password: '', database: '', container: '', docker_network: '' }
  );

  // MSSQL config state
  const [mssqlConfig, setMssqlConfig] = useState<MssqlConnectionConfig>(
    existingTool?.tool_type === 'mssql'
      ? (existingTool.connection_config as MssqlConnectionConfig)
      : { host: '', port: 1433, user: '', password: '', database: '' }
  );

  // MySQL/MariaDB config state
  const [mysqlConnectionMode, setMysqlConnectionMode] = useState<'direct' | 'container'>(
    existingTool?.tool_type === 'mysql' && (existingTool.connection_config as MysqlConnectionConfig).container
      ? 'container'
      : 'direct'
  );
  const [mysqlConfig, setMysqlConfig] = useState<MysqlConnectionConfig>(
    existingTool?.tool_type === 'mysql'
      ? (existingTool.connection_config as MysqlConnectionConfig)
      : { host: '', port: 3306, user: '', password: '', database: '', container: '', docker_network: '' }
  );

  const [odooConnectionMode, setOdooConnectionMode] = useState<'docker' | 'ssh'>(
    existingTool?.tool_type === 'odoo_shell' && (existingTool.connection_config as OdooShellConnectionConfig).mode === 'ssh'
      ? 'ssh'
      : 'docker'
  );
  const [odooConfig, setOdooConfig] = useState<OdooShellConnectionConfig>(
    existingTool?.tool_type === 'odoo_shell'
      ? (existingTool.connection_config as OdooShellConnectionConfig)
      : { mode: 'docker', container: '', database: 'odoo', docker_network: '', config_path: '', ssh_host: '', ssh_port: 22, ssh_user: '', ssh_key_path: '', ssh_key_content: '', ssh_public_key: '', ssh_key_passphrase: '', ssh_password: '', odoo_bin_path: '', working_directory: '', run_as_user: '' }
  );

  // Docker discovery state
  const [dockerContainers, setDockerContainers] = useState<DockerContainer[]>([]);
  const [dockerNetworks, setDockerNetworks] = useState<DockerNetwork[]>([]);
  const [currentNetwork, setCurrentNetwork] = useState<string | null>(null);
  const [currentContainer, setCurrentContainer] = useState<string | null>(null);
  const [loadingDocker, setLoadingDocker] = useState(false);
  const [connectingNetwork, setConnectingNetwork] = useState(false);

  // PostgreSQL database discovery state
  const [discoveredDatabases, setDiscoveredDatabases] = useState<string[]>([]);
  const [discoveringDatabases, setDiscoveringDatabases] = useState(false);
  const [databaseDiscoveryError, setDatabaseDiscoveryError] = useState<string | null>(null);

  // MSSQL database discovery state
  const [mssqlDiscoveredDatabases, setMssqlDiscoveredDatabases] = useState<string[]>([]);
  const [mssqlDiscoveringDatabases, setMssqlDiscoveringDatabases] = useState(false);
  const [mssqlDatabaseDiscoveryError, setMssqlDatabaseDiscoveryError] = useState<string | null>(null);

  // MySQL/MariaDB database discovery state
  const [mysqlDiscoveredDatabases, setMysqlDiscoveredDatabases] = useState<string[]>([]);
  const [mysqlDiscoveringDatabases, setMysqlDiscoveringDatabases] = useState(false);
  const [mysqlDatabaseDiscoveryError, setMysqlDatabaseDiscoveryError] = useState<string | null>(null);

  const [sshConfig, setSshConfig] = useState<SSHShellConnectionConfig>(
    existingTool?.tool_type === 'ssh_shell'
      ? (existingTool.connection_config as SSHShellConnectionConfig)
      : { host: '', port: 22, user: '', key_path: '', key_content: '', public_key: '', key_passphrase: '', password: '', command_prefix: '' }
  );

  // Filesystem Indexer config state
  const [filesystemConfig, setFilesystemConfig] = useState<FilesystemConnectionConfig>(() => {
    if (existingTool?.tool_type === 'filesystem_indexer') {
      const existing = existingTool.connection_config as Partial<FilesystemConnectionConfig>;
      return {
        mount_type: existing.mount_type ?? 'docker_volume',
        base_path: existing.base_path ?? '',
        volume_name: existing.volume_name ?? '',
        smb_host: existing.smb_host ?? '',
        smb_share: existing.smb_share ?? '',
        smb_user: existing.smb_user ?? '',
        smb_password: existing.smb_password ?? '',
        smb_domain: existing.smb_domain ?? '',
        nfs_host: existing.nfs_host ?? '',
        nfs_export: existing.nfs_export ?? '',
        nfs_options: existing.nfs_options ?? 'ro,noatime',
        index_name: existing.index_name ?? '',
        file_patterns: existing.file_patterns ?? ['**/*.txt', '**/*.md', '**/*.pdf', '**/*.docx', '**/*.xlsx', '**/*.pptx', '**/*.py', '**/*.json', '**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.gif', '**/*.bmp', '**/*.tiff', '**/*.webp'],
        exclude_patterns: existing.exclude_patterns ?? ['**/node_modules/**', '**/__pycache__/**', '**/venv/**', '**/.git/**', '**/.*', '**/*.cloud', '**/*.icloud'],
        recursive: existing.recursive ?? true,
        chunk_size: existing.chunk_size ?? 1000,
        chunk_overlap: existing.chunk_overlap ?? 200,
        max_file_size_mb: existing.max_file_size_mb ?? 10,
        max_total_files: existing.max_total_files ?? 10000,
        ocr_mode: existing.ocr_mode ?? 'disabled',
        ocr_vision_model: existing.ocr_vision_model,
        vector_store_type: existing.vector_store_type ?? 'pgvector',
        reindex_interval_hours: existing.reindex_interval_hours ?? 24,
        last_indexed_at: existing.last_indexed_at ?? null,
      };
    }
    return {
      mount_type: 'docker_volume',
      base_path: '',
      volume_name: '',
      smb_host: '',
      smb_share: '',
      smb_user: '',
      smb_password: '',
      smb_domain: '',
      nfs_host: '',
      nfs_export: '',
      nfs_options: 'ro,noatime',
      index_name: '',
      file_patterns: ['**/*.txt', '**/*.md', '**/*.pdf', '**/*.docx', '**/*.xlsx', '**/*.pptx', '**/*.py', '**/*.json', '**/*.png', '**/*.jpg', '**/*.jpeg', '**/*.gif', '**/*.bmp', '**/*.tiff', '**/*.webp'],
      exclude_patterns: ['**/node_modules/**', '**/__pycache__/**', '**/venv/**', '**/.git/**', '**/.*', '**/*.cloud', '**/*.icloud'],
      recursive: true,
      chunk_size: 1000,
      chunk_overlap: 200,
      max_file_size_mb: 10,
      max_total_files: 10000,
      ocr_mode: 'disabled',
      ocr_vision_model: undefined,
      vector_store_type: 'pgvector',
      reindex_interval_hours: 24,
      last_indexed_at: null,
    };
  });

  // SolidWorks PDM config state
  const [pdmConfig, setPdmConfig] = useState<SolidworksPdmConnectionConfig>(
    existingTool?.tool_type === 'solidworks_pdm'
      ? (existingTool.connection_config as SolidworksPdmConnectionConfig)
      : {
          host: '',
          port: 1433,
          user: '',
          password: '',
          database: '',
          file_extensions: ['.SLDPRT', '.SLDASM', '.SLDDRW'],
          exclude_deleted: true,
          variable_names: ['Description', 'Material', 'Revision', 'Status'],
          include_bom: true,
          include_folder_path: true,
          include_configurations: true,
          max_documents: null,
          last_indexed_at: null,
        }
  );

  // PDM discovery state
  const [pdmDiscoveringDatabases, setPdmDiscoveringDatabases] = useState(false);
  const [pdmDiscoveredDatabases, setPdmDiscoveredDatabases] = useState<string[]>([]);
  const [pdmDatabaseDiscoveryError, setPdmDatabaseDiscoveryError] = useState<string | null>(null);
  const [pdmDiscoveringSchema, setPdmDiscoveringSchema] = useState(false);
  const [pdmDiscoveredExtensions, setPdmDiscoveredExtensions] = useState<string[]>([]);
  const [pdmDiscoveredVariables, setPdmDiscoveredVariables] = useState<string[]>([]);
  const [pdmDocumentCount, setPdmDocumentCount] = useState<number>(0);
  const [pdmSchemaDiscoveryError, setPdmSchemaDiscoveryError] = useState<string | null>(null);
  const [pdmExtensionFilter, setPdmExtensionFilter] = useState('');
  const [pdmVariableFilter, setPdmVariableFilter] = useState('');

  // Ollama availability for OCR mode
  const [ollamaAvailable, setOllamaAvailable] = useState(false);

  // Fetch settings to check if Ollama OCR is configured
  useEffect(() => {
    if (toolType === 'filesystem_indexer' || defaultToolType === 'filesystem_indexer') {
      api.getSettings()
        .then(({ settings }) => {
          setOllamaAvailable(settings.default_ocr_mode === 'ollama');
        })
        .catch((err) => {
          console.warn('Failed to fetch settings for Ollama check:', err);
          setOllamaAvailable(false);
        });
    }
  }, [toolType, defaultToolType]);

  // Auto-generate name from filesystem path basename if not already set
  useEffect(() => {
    if (!isEditing && toolType === 'filesystem_indexer' && filesystemConfig.base_path && !name) {
      const pathParts = filesystemConfig.base_path.split('/').filter(Boolean);
      const basename = pathParts[pathParts.length - 1] || 'filesystem';
      const generatedName = basename.charAt(0).toUpperCase() + basename.slice(1);
      setName(generatedName);
    }
  }, [filesystemConfig.base_path, toolType, isEditing, name]);

  // Auto-discover PDM schema when entering the pdm_filtering step
  const handleDiscoverPdmSchemaRef = useRef<(() => Promise<void>) | null>(null);
  useEffect(() => {
    if (currentStep === 'pdm_filtering' && pdmDiscoveredExtensions.length === 0 && !pdmDiscoveringSchema && handleDiscoverPdmSchemaRef.current) {
      handleDiscoverPdmSchemaRef.current();
    }
  }, [currentStep, pdmDiscoveredExtensions.length, pdmDiscoveringSchema]);

  // Filesystem analysis state
  const [_fsAnalysisJobId, setFsAnalysisJobId] = useState<string | null>(null);
  const [fsAnalysisJob, setFsAnalysisJob] = useState<import('@/types').FilesystemAnalysisJob | null>(null);
  const [fsAnalyzing, setFsAnalyzing] = useState(false);
  const [fsAnalysisExpanded, setFsAnalysisExpanded] = useState(false);
  const [fsExclusionsApplied, setFsExclusionsApplied] = useState(false);
  const [fsAdvancedOpen, setFsAdvancedOpen] = useState(false);

  // Container capabilities state (for showing/hiding SMB/NFS options)
  const [containerCapabilities, setContainerCapabilities] = useState<import('@/types').ContainerCapabilitiesResponse | null>(null);
  const [loadingCapabilities, setLoadingCapabilities] = useState(false);

  // Fetch container capabilities on mount (only for filesystem indexer)
  useEffect(() => {
    if (toolType === 'filesystem_indexer' || defaultToolType === 'filesystem_indexer') {
      setLoadingCapabilities(true);
      api.checkContainerCapabilities()
        .then(setContainerCapabilities)
        .catch((err) => {
          console.warn('Failed to check container capabilities:', err);
          // Default to no mount capability if check fails
          setContainerCapabilities({
            privileged: false,
            has_sys_admin: false,
            can_mount: false,
            message: 'Failed to check container capabilities',
          });
        })
        .finally(() => setLoadingCapabilities(false));
    }
  }, [toolType, defaultToolType]);

  // SSH Key management state
  const [sshKeyMode, setSshKeyMode] = useState<'generate' | 'upload' | 'path' | 'password'>(
    (() => {
      // Determine initial mode based on existing config
      const config = existingTool?.connection_config as (OdooShellConnectionConfig | SSHShellConnectionConfig) | undefined;
      if (config) {
        if ('ssh_key_content' in config && config.ssh_key_content) return 'upload';
        if ('key_content' in config && config.key_content) return 'upload';
        if ('ssh_key_path' in config && config.ssh_key_path) return 'path';
        if ('key_path' in config && config.key_path) return 'path';
        if ('ssh_password' in config && config.ssh_password) return 'password';
        if ('password' in config && config.password) return 'password';
      }
      return 'generate';
    })()
  );
  const [generatingKey, setGeneratingKey] = useState(false);
  const [_generatedPublicKey, setGeneratedPublicKey] = useState<string | null>(null);
  const [keyCopied, setKeyCopied] = useState(false);

  // SSH Tunnel auth mode state (shared across database tools)
  const [sshTunnelAuthMode, setSshTunnelAuthMode] = useState<SSHAuthMode>(
    (() => {
      // Determine initial mode based on existing config
      const config = existingTool?.connection_config as { ssh_tunnel_key_content?: string; ssh_tunnel_key_path?: string; ssh_tunnel_password?: string } | undefined;
      if (config) {
        if (config.ssh_tunnel_key_content) return 'upload';
        if (config.ssh_tunnel_key_path) return 'path';
        if (config.ssh_tunnel_password) return 'password';
      }
      return 'generate';
    })()
  );
  const [sshTunnelGeneratingKey, setSshTunnelGeneratingKey] = useState(false);
  const [sshTunnelKeyCopied, setSshTunnelKeyCopied] = useState(false);

  const getConnectionConfig = (): ConnectionConfig => {
    switch (toolType) {
      case 'postgres':
        return postgresConfig;
      case 'mysql':
        return mysqlConfig;
      case 'mssql':
        return mssqlConfig;
      case 'odoo_shell':
        return { ...odooConfig, mode: odooConnectionMode };
      case 'ssh_shell':
        return sshConfig;
      case 'filesystem_indexer':
        // Use the tool name as the index_name (Step 3 Name field)
        return { ...filesystemConfig, index_name: name || filesystemConfig.index_name };
      case 'solidworks_pdm':
        return pdmConfig;
    }
  };

  // Shared Docker discovery handlers for PostgreSQL container mode and Odoo Docker mode
  const handleDiscoverDocker = async () => {
    setLoadingDocker(true);
    try {
      const result = await api.discoverDocker();
      if (result.success) {
        setDockerContainers(result.containers);
        setDockerNetworks(result.networks);
        setCurrentNetwork(result.current_network);
        setCurrentContainer(result.current_container);

        // Auto-select first relevant container if none selected
        if (toolType === 'postgres' && !postgresConfig.container) {
          const firstPg = result.containers.find(c => c.image.toLowerCase().includes('postgres') && c.name !== result.current_container);
          if (firstPg) {
            setPostgresConfig({
              ...postgresConfig,
              container: firstPg.name,
              docker_network: firstPg.networks[0] || ''
            });
          }
        } else if (toolType === 'odoo_shell' && !odooConfig.container) {
          const firstOdoo = result.containers.find(c => c.has_odoo && c.name !== result.current_container);
          if (firstOdoo) {
            setOdooConfig({
              ...odooConfig,
              container: firstOdoo.name,
              docker_network: firstOdoo.networks[0] || ''
            });
          }
        }
      }
    } catch (err) {
      console.error('Docker discovery failed:', err);
    } finally {
      setLoadingDocker(false);
    }
  };

  const handleConnectNetwork = async (e: React.MouseEvent, networkName: string) => {
    e.preventDefault();
    e.stopPropagation();
    setConnectingNetwork(true);
    try {
      const result = await api.connectToNetwork(networkName);
      if (result.success) {
        setCurrentNetwork(networkName);
        // Update the appropriate config
        if (toolType === 'postgres') {
          setPostgresConfig({ ...postgresConfig, docker_network: networkName });
        } else if (toolType === 'odoo_shell') {
          setOdooConfig({ ...odooConfig, docker_network: networkName });
        }
      }
    } catch (err) {
      console.error('Network connection failed:', err);
    } finally {
      setConnectingNetwork(false);
    }
  };

  // PostgreSQL database discovery handler
  const handleDiscoverDatabases = async () => {
    if (!postgresConfig.host || !postgresConfig.user || !postgresConfig.password) {
      setDatabaseDiscoveryError('Host, user, and password are required to discover databases');
      return;
    }

    setDiscoveringDatabases(true);
    setDatabaseDiscoveryError(null);
    setDiscoveredDatabases([]);

    try {
      const result = await api.discoverPostgresDatabases({
        host: postgresConfig.host,
        port: postgresConfig.port || 5432,
        user: postgresConfig.user,
        password: postgresConfig.password,
        // SSH tunnel fields
        ssh_tunnel_enabled: postgresConfig.ssh_tunnel_enabled,
        ssh_tunnel_host: postgresConfig.ssh_tunnel_host,
        ssh_tunnel_port: postgresConfig.ssh_tunnel_port,
        ssh_tunnel_user: postgresConfig.ssh_tunnel_user,
        ssh_tunnel_password: postgresConfig.ssh_tunnel_password,
        ssh_tunnel_key_path: postgresConfig.ssh_tunnel_key_path,
        ssh_tunnel_key_content: postgresConfig.ssh_tunnel_key_content,
        ssh_tunnel_key_passphrase: postgresConfig.ssh_tunnel_key_passphrase,
      });

      if (result.success) {
        setDiscoveredDatabases(result.databases);
        // Auto-select first database if none selected
        if (result.databases.length > 0 && !postgresConfig.database) {
          setPostgresConfig({ ...postgresConfig, database: result.databases[0] });
        }
      } else {
        setDatabaseDiscoveryError(result.error || 'Discovery failed');
      }
    } catch (err) {
      setDatabaseDiscoveryError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setDiscoveringDatabases(false);
    }
  };

  // MSSQL database discovery handler
  const handleDiscoverMssqlDatabases = async () => {
    if (!mssqlConfig.host || !mssqlConfig.user || !mssqlConfig.password) {
      setMssqlDatabaseDiscoveryError('Host, user, and password are required to discover databases');
      return;
    }

    setMssqlDiscoveringDatabases(true);
    setMssqlDatabaseDiscoveryError(null);
    setMssqlDiscoveredDatabases([]);

    try {
      const result = await api.discoverMssqlDatabases({
        host: mssqlConfig.host,
        port: mssqlConfig.port || 1433,
        user: mssqlConfig.user,
        password: mssqlConfig.password,
        // SSH tunnel fields
        ssh_tunnel_enabled: mssqlConfig.ssh_tunnel_enabled,
        ssh_tunnel_host: mssqlConfig.ssh_tunnel_host,
        ssh_tunnel_port: mssqlConfig.ssh_tunnel_port,
        ssh_tunnel_user: mssqlConfig.ssh_tunnel_user,
        ssh_tunnel_password: mssqlConfig.ssh_tunnel_password,
        ssh_tunnel_key_path: mssqlConfig.ssh_tunnel_key_path,
        ssh_tunnel_key_content: mssqlConfig.ssh_tunnel_key_content,
        ssh_tunnel_key_passphrase: mssqlConfig.ssh_tunnel_key_passphrase,
      });

      if (result.success) {
        setMssqlDiscoveredDatabases(result.databases);
        // Auto-select first database if none selected
        if (result.databases.length > 0 && !mssqlConfig.database) {
          setMssqlConfig({ ...mssqlConfig, database: result.databases[0] });
        }
      } else {
        setMssqlDatabaseDiscoveryError(result.error || 'Discovery failed');
      }
    } catch (err) {
      setMssqlDatabaseDiscoveryError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setMssqlDiscoveringDatabases(false);
    }
  };

  // MySQL/MariaDB database discovery handler
  const handleDiscoverMysqlDatabases = async () => {
    // For container mode, we just need the container name
    if (mysqlConnectionMode === 'container') {
      if (!mysqlConfig.container) {
        setMysqlDatabaseDiscoveryError('Container name is required to discover databases');
        return;
      }
    } else {
      // For direct mode, need host, user, password
      if (!mysqlConfig.host || !mysqlConfig.user || !mysqlConfig.password) {
        setMysqlDatabaseDiscoveryError('Host, user, and password are required to discover databases');
        return;
      }
    }

    setMysqlDiscoveringDatabases(true);
    setMysqlDatabaseDiscoveryError(null);
    setMysqlDiscoveredDatabases([]);

    try {
      const result = await api.discoverMysqlDatabases(
        mysqlConnectionMode === 'container'
          ? {
              container: mysqlConfig.container,
              docker_network: mysqlConfig.docker_network,
            }
          : {
              host: mysqlConfig.host,
              port: mysqlConfig.port || 3306,
              user: mysqlConfig.user,
              password: mysqlConfig.password,
              // SSH tunnel fields
              ssh_tunnel_enabled: mysqlConfig.ssh_tunnel_enabled,
              ssh_tunnel_host: mysqlConfig.ssh_tunnel_host,
              ssh_tunnel_port: mysqlConfig.ssh_tunnel_port,
              ssh_tunnel_user: mysqlConfig.ssh_tunnel_user,
              ssh_tunnel_password: mysqlConfig.ssh_tunnel_password,
              ssh_tunnel_key_path: mysqlConfig.ssh_tunnel_key_path,
              ssh_tunnel_key_content: mysqlConfig.ssh_tunnel_key_content,
              ssh_tunnel_key_passphrase: mysqlConfig.ssh_tunnel_key_passphrase,
            }
      );

      if (result.success) {
        setMysqlDiscoveredDatabases(result.databases);
        // Auto-select first database if none selected
        if (result.databases.length > 0 && !mysqlConfig.database) {
          setMysqlConfig({ ...mysqlConfig, database: result.databases[0] });
        }
      } else {
        setMysqlDatabaseDiscoveryError(result.error || 'Discovery failed');
      }
    } catch (err) {
      setMysqlDatabaseDiscoveryError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setMysqlDiscoveringDatabases(false);
    }
  };

  // SSH Key generation handler
  const handleGenerateSSHKey = async () => {
    setGeneratingKey(true);
    setError(null);
    try {
      // Get passphrase from the appropriate config
      const passphrase = toolType === 'odoo_shell'
        ? odooConfig.ssh_key_passphrase
        : sshConfig.key_passphrase;
      const result = await api.generateSSHKeypair(name || 'ragtime', passphrase || undefined);
      // Store the keys in the appropriate config
      if (toolType === 'odoo_shell') {
        setOdooConfig({
          ...odooConfig,
          ssh_key_content: result.private_key,
          ssh_public_key: result.public_key,
          ssh_key_path: '', // Clear path when using content
          // Keep the passphrase that was set before generation
        });
      } else if (toolType === 'ssh_shell') {
        setSshConfig({
          ...sshConfig,
          key_content: result.private_key,
          public_key: result.public_key,
          key_path: '', // Clear path when using content
          // Keep the passphrase that was set before generation
        });
      }
      setGeneratedPublicKey(result.public_key);
      // Stay on generate tab to show the public key for copying
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate SSH keypair');
    } finally {
      setGeneratingKey(false);
    }
  };

  // Copy public key to clipboard
  const handleCopyPublicKey = async () => {
    const pubKey = toolType === 'odoo_shell' ? odooConfig.ssh_public_key : sshConfig.public_key;
    if (pubKey) {
      await navigator.clipboard.writeText(pubKey);
      setKeyCopied(true);
      setTimeout(() => setKeyCopied(false), 2000);
    }
  };

  // SSH Tunnel key generation handler (for database tools)
  const handleGenerateTunnelSSHKey = async () => {
    setSshTunnelGeneratingKey(true);
    setError(null);
    try {
      // Get passphrase and config based on current tool type
      let passphrase: string | undefined;
      if (toolType === 'postgres') {
        passphrase = postgresConfig.ssh_tunnel_key_passphrase;
      } else if (toolType === 'mysql') {
        passphrase = mysqlConfig.ssh_tunnel_key_passphrase;
      } else if (toolType === 'mssql') {
        passphrase = mssqlConfig.ssh_tunnel_key_passphrase;
      } else if (toolType === 'solidworks_pdm') {
        passphrase = pdmConfig.ssh_tunnel_key_passphrase;
      }

      const result = await api.generateSSHKeypair(name || 'tunnel', passphrase || undefined);

      // Store the keys in the appropriate config
      if (toolType === 'postgres') {
        setPostgresConfig({
          ...postgresConfig,
          ssh_tunnel_key_content: result.private_key,
          ssh_tunnel_public_key: result.public_key,
          ssh_tunnel_key_path: '',
        });
      } else if (toolType === 'mysql') {
        setMysqlConfig({
          ...mysqlConfig,
          ssh_tunnel_key_content: result.private_key,
          ssh_tunnel_public_key: result.public_key,
          ssh_tunnel_key_path: '',
        });
      } else if (toolType === 'mssql') {
        setMssqlConfig({
          ...mssqlConfig,
          ssh_tunnel_key_content: result.private_key,
          ssh_tunnel_public_key: result.public_key,
          ssh_tunnel_key_path: '',
        });
      } else if (toolType === 'solidworks_pdm') {
        setPdmConfig({
          ...pdmConfig,
          ssh_tunnel_key_content: result.private_key,
          ssh_tunnel_public_key: result.public_key,
          ssh_tunnel_key_path: '',
        });
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate SSH keypair');
    } finally {
      setSshTunnelGeneratingKey(false);
    }
  };

  // Copy SSH tunnel public key to clipboard
  const handleCopyTunnelPublicKey = async () => {
    let pubKey: string | undefined;
    if (toolType === 'postgres') {
      pubKey = postgresConfig.ssh_tunnel_public_key;
    } else if (toolType === 'mysql') {
      pubKey = mysqlConfig.ssh_tunnel_public_key;
    } else if (toolType === 'mssql') {
      pubKey = mssqlConfig.ssh_tunnel_public_key;
    } else if (toolType === 'solidworks_pdm') {
      pubKey = pdmConfig.ssh_tunnel_public_key;
    }
    if (pubKey) {
      await navigator.clipboard.writeText(pubKey);
      setSshTunnelKeyCopied(true);
      setTimeout(() => setSshTunnelKeyCopied(false), 2000);
    }
  };

  const wizardSteps = getWizardSteps();
  const getCurrentStepIndex = () => wizardSteps.indexOf(currentStep);

  const canNavigateToStep = (stepIndex: number): boolean => {
    // Can always go back to previous steps
    if (stepIndex <= getCurrentStepIndex()) return true;
    // Can only go forward one step at a time, and current step must be valid
    if (stepIndex === getCurrentStepIndex() + 1 && canProceed()) return true;
    return false;
  };

  const goToStep = (step: WizardStep) => {
    const stepIndex = wizardSteps.indexOf(step);
    if (canNavigateToStep(stepIndex)) {
      setCurrentStep(step);
      setTestResult(null);
      setError(null);
    }
  };

  const goToNextStep = () => {
    const currentIndex = getCurrentStepIndex();
    if (currentIndex < wizardSteps.length - 1) {
      setCurrentStep(wizardSteps[currentIndex + 1]);
      setTestResult(null);
      setError(null);
    }
  };

  const goToPreviousStep = () => {
    const currentIndex = getCurrentStepIndex();
    if (currentIndex > 0) {
      // Skip type step if editing
      const prevIndex = currentIndex - 1;
      if (isEditing && wizardSteps[prevIndex] === 'type') {
        return;
      }
      setCurrentStep(wizardSteps[prevIndex]);
      setTestResult(null);
      setError(null);
    }
  };

  const handleTestConnection = async () => {
    setTesting(true);
    setTestResult(null);
    setError(null);

    try {
      // Auto-save when editing an existing tool before testing
      if (isEditing && existingTool) {
        await api.updateToolConfig(existingTool.id, {
          name,
          description,
          connection_config: getConnectionConfig(),
          max_results: maxResults,
          timeout: timeoutValue,
          allow_write: allowWrite,
        });
      }

      const result = await api.testToolConnection({
        tool_type: toolType,
        connection_config: getConnectionConfig(),
      });
      setTestResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Test failed');
    } finally {
      setTesting(false);
    }
  };

  // Track the created tool ID for new tools that get auto-saved during analysis
  const [createdToolId, setCreatedToolId] = useState<string | null>(null);

  // Filesystem analysis handlers
  const handleStartFilesystemAnalysis = async () => {
    // Auto-generate name from path basename if not provided
    let toolName = name.trim();
    if (!toolName && filesystemConfig.base_path) {
      const pathParts = filesystemConfig.base_path.split('/').filter(Boolean);
      const basename = pathParts[pathParts.length - 1] || 'filesystem';
      toolName = basename.charAt(0).toUpperCase() + basename.slice(1);
      setName(toolName);
    }

    setFsAnalyzing(true);
    setFsAnalysisJob(null);
    setFsExclusionsApplied(false);
    setError(null);

    try {
      let toolId = existingTool?.id || createdToolId;

      if (toolId) {
        // Update existing tool config
        await api.updateToolConfig(toolId, {
          name: toolName,
          description,
          connection_config: getConnectionConfig(),
          max_results: maxResults,
          timeout: timeoutValue,
          allow_write: allowWrite,
        });
      } else {
        // Auto-create the tool first
        const request: CreateToolConfigRequest = {
          name: toolName,
          tool_type: toolType,
          description,
          connection_config: getConnectionConfig(),
          max_results: maxResults,
          timeout: timeoutValue,
          allow_write: allowWrite,
        };
        const created = await api.createToolConfig(request);
        toolId = created.id;
        setCreatedToolId(toolId);
      }

      // Start analysis
      const job = await api.startFilesystemAnalysis(toolId);
      setFsAnalysisJobId(job.id);
      setFsAnalysisJob(job);

      // Poll for completion
      pollFilesystemAnalysis(toolId, job.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start analysis');
      setFsAnalyzing(false);
    }
  };

  const pollFilesystemAnalysis = async (toolId: string, jobId: string) => {
    try {
      const job = await api.getFilesystemAnalysisJob(toolId, jobId);
      setFsAnalysisJob(job);

      if (job.status === 'completed' || job.status === 'failed') {
        setFsAnalyzing(false);
        if (job.status === 'completed' && job.result) {
          setFsAnalysisExpanded(true);
        }
      } else {
        // Continue polling
        setTimeout(() => pollFilesystemAnalysis(toolId, jobId), 500);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get analysis status');
      setFsAnalyzing(false);
    }
  };

  const handleApplyFsExclusions = () => {
    if (!fsAnalysisJob?.result?.suggested_exclusions) return;

    const currentExclusions = filesystemConfig.exclude_patterns || [];
    const newExclusions = fsAnalysisJob.result.suggested_exclusions.filter(
      (pattern: string) => !currentExclusions.includes(pattern)
    );

    if (newExclusions.length > 0) {
      setFilesystemConfig({
        ...filesystemConfig,
        exclude_patterns: [...currentExclusions, ...newExclusions],
      });
    }
    setFsExclusionsApplied(true);
    setFsAdvancedOpen(true); // Expand advanced section to show applied exclusions
  };

  const handleSave = async () => {
    setSaving(true);
    setError(null);

    try {
      let savedToolId = existingTool?.id || createdToolId;
      const connectionConfig = getConnectionConfig();

      if (savedToolId) {
        // Update existing tool (or tool that was auto-created during analysis)
        await api.updateToolConfig(savedToolId, {
          name,
          description,
          connection_config: connectionConfig,
          max_results: maxResults,
          timeout: timeoutValue,
          allow_write: allowWrite,
        });
      } else {
        const request: CreateToolConfigRequest = {
          name,
          tool_type: toolType,
          description,
          connection_config: connectionConfig,
          max_results: maxResults,
          timeout: timeoutValue,
          allow_write: allowWrite,
        };
        const created = await api.createToolConfig(request);
        savedToolId = created.id;
      }

      // Trigger schema indexing for SQL tools if enabled
      if ((toolType === 'postgres' || toolType === 'mssql' || toolType === 'mysql') && savedToolId) {
        const config = connectionConfig as { schema_index_enabled?: boolean };
        const wasEnabled = (existingTool?.connection_config as { schema_index_enabled?: boolean } | undefined)?.schema_index_enabled;

        // Trigger indexing if schema_index_enabled is newly enabled or was already enabled (force refresh on save)
        if (config.schema_index_enabled && !wasEnabled) {
          try {
            await api.triggerSchemaIndex(savedToolId);
          } catch (schemaErr) {
            // Don't fail the save if schema indexing fails to start
            console.error('Failed to start schema indexing:', schemaErr);
          }
        }
      }

      // Trigger PDM document indexing for SolidWorks PDM tools
      if (toolType === 'solidworks_pdm' && savedToolId) {
        try {
          await api.triggerPdmIndex(savedToolId);
        } catch (pdmErr) {
          // Don't fail the save if PDM indexing fails to start
          console.error('Failed to start PDM indexing:', pdmErr);
        }
      }

      onSave();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
      setSaving(false);
    }
  };

  const canProceed = (): boolean => {
    switch (currentStep) {
      case 'type':
        return true;
      case 'connection':
        return validateConnection();
      case 'execution_constraints':
        // Optional, but if entering manually must be valid?
        // For now just allow proceeding (empty = no constraints or root)
        return true;
      case 'pdm_filtering':
        // Must have at least one file extension selected
        return (pdmConfig.file_extensions?.length ?? 0) > 0;
      case 'description':
        return name.trim().length > 0;
      case 'options':
        return true;
      case 'review':
        return true;
    }
  };

  const validateConnection = (): boolean => {
    switch (toolType) {
      case 'postgres':
        // Either host or container must be specified
        return Boolean((postgresConfig.host && postgresConfig.user) || postgresConfig.container);
      case 'mysql':
        // Either host or container must be specified
        return Boolean((mysqlConfig.host && mysqlConfig.user) || mysqlConfig.container);
      case 'mssql':
        // Host, user, password, and database are required
        return Boolean(mssqlConfig.host && mssqlConfig.user && mssqlConfig.password && mssqlConfig.database);
      case 'odoo_shell':
        // For Docker mode, need container. For SSH mode, need host and user.
        if (odooConnectionMode === 'ssh') {
          const hasAuth = Boolean(
            odooConfig.ssh_key_content ||
            odooConfig.ssh_key_path ||
            odooConfig.ssh_password
          );
          return Boolean(odooConfig.ssh_host && odooConfig.ssh_user && hasAuth);
        }
        return Boolean(odooConfig.container);
      case 'ssh_shell':
        const hasSshAuth = Boolean(
          sshConfig.key_content ||
          sshConfig.key_path ||
          sshConfig.password
        );
        return Boolean(sshConfig.host && sshConfig.user && hasSshAuth);
      case 'filesystem_indexer':
        // Validate based on mount type
        if (filesystemConfig.mount_type === 'docker_volume') {
          return Boolean(filesystemConfig.base_path);
        } else if (filesystemConfig.mount_type === 'smb') {
          return Boolean(filesystemConfig.smb_host && filesystemConfig.smb_share);
        } else if (filesystemConfig.mount_type === 'nfs') {
          return Boolean(filesystemConfig.nfs_host && filesystemConfig.nfs_export);
        }
        return Boolean(filesystemConfig.base_path);
      case 'solidworks_pdm':
        // Host, user, password, and database are required
        return Boolean(pdmConfig.host && pdmConfig.user && pdmConfig.password && pdmConfig.database);
    }
  };

  const renderTypeSelection = () => {
    // Define SQL tool types that should be grouped
    const sqlToolTypes: ToolType[] = ['postgres', 'mysql', 'mssql'];
    const isSqlTool = (type: ToolType) => sqlToolTypes.includes(type);
    const nonSqlTools = (Object.entries(TOOL_TYPE_INFO) as [ToolType, typeof TOOL_TYPE_INFO[ToolType]][])
      .filter(([type]) => !isSqlTool(type));
    const sqlTools = (Object.entries(TOOL_TYPE_INFO) as [ToolType, typeof TOOL_TYPE_INFO[ToolType]][])
      .filter(([type]) => isSqlTool(type));

    // Check if any SQL tool is currently selected
    const sqlToolSelected = isSqlTool(toolType);

    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Select the type of tool connection you want to add:
        </p>
        <div className="tool-type-selection">
          {/* SQL Tools Dropdown Group */}
          <div
            className={`tool-type-group ${sqlToolsExpanded || sqlToolSelected ? 'expanded' : ''}`}
            onMouseEnter={() => setSqlToolsExpanded(true)}
            onMouseLeave={() => setSqlToolsExpanded(false)}
          >
            <div className={`tool-type-group-header ${sqlToolSelected ? 'selected' : ''}`}>
              <span className="tool-type-option-icon">
                <Icon name="database" size={24} />
              </span>
              <span className="tool-type-option-name">SQL Databases</span>
              <span className="tool-type-option-desc">PostgreSQL, MySQL/MariaDB, MSSQL/SQL Server</span>
              <span className="tool-type-group-chevron">
                <Icon name={sqlToolsExpanded || sqlToolSelected ? 'chevron-up' : 'chevron-down'} size={16} />
              </span>
            </div>
            {(sqlToolsExpanded || sqlToolSelected) && (
              <div className="tool-type-group-items">
                {sqlTools.map(([type, info]) => (
                  <button
                    key={type}
                    type="button"
                    className={`tool-type-option tool-type-option-nested ${toolType === type ? 'selected' : ''}`}
                    onClick={() => setToolType(type)}
                  >
                    <span className="tool-type-option-icon">
                      <Icon name={getToolIconType(info.icon)} size={20} />
                    </span>
                    <span className="tool-type-option-name">{info.name}</span>
                    <span className="tool-type-option-desc">{info.description}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Non-SQL Tools */}
          {nonSqlTools.map(([type, info]) => (
            <button
              key={type}
              type="button"
              className={`tool-type-option ${toolType === type ? 'selected' : ''}`}
              onClick={() => setToolType(type)}
            >
              <span className="tool-type-option-icon">
                <Icon name={getToolIconType(info.icon)} size={24} />
              </span>
              <span className="tool-type-option-name">{info.name}</span>
              <span className="tool-type-option-desc">{info.description}</span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderPostgresConnection = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Choose your connection method:
        </p>

        <div className="connection-tabs">
          <button
            type="button"
            className={`connection-tab ${pgConnectionMode === 'direct' ? 'active' : ''}`}
            onClick={() => {
              setPgConnectionMode('direct');
              setPostgresConfig({ ...postgresConfig, container: '' });
            }}
          >
            Direct Connection
          </button>
          <button
            type="button"
            className={`connection-tab ${pgConnectionMode === 'container' ? 'active' : ''}`}
            onClick={() => {
              setPgConnectionMode('container');
              setPostgresConfig({ ...postgresConfig, host: '', user: '', password: '' });
            }}
          >
            Docker Container
          </button>
        </div>

        {pgConnectionMode === 'direct' ? (
          <div className="connection-panel">
            {/* SSH Tunnel Option - shown first for clear context */}
            <SSHTunnelPanel
              enabled={postgresConfig.ssh_tunnel_enabled ?? false}
              onEnabledChange={(enabled) => {
                // When enabling tunnel, default host to localhost if empty
                const updates: Partial<typeof postgresConfig> = { ssh_tunnel_enabled: enabled };
                if (enabled && !postgresConfig.host) {
                  updates.host = 'localhost';
                }
                setPostgresConfig({ ...postgresConfig, ...updates });
              }}
              config={{
                ssh_tunnel_host: postgresConfig.ssh_tunnel_host,
                ssh_tunnel_port: postgresConfig.ssh_tunnel_port,
                ssh_tunnel_user: postgresConfig.ssh_tunnel_user,
                ssh_tunnel_password: postgresConfig.ssh_tunnel_password,
                ssh_tunnel_key_path: postgresConfig.ssh_tunnel_key_path,
                ssh_tunnel_key_content: postgresConfig.ssh_tunnel_key_content,
                ssh_tunnel_key_passphrase: postgresConfig.ssh_tunnel_key_passphrase,
                ssh_tunnel_public_key: postgresConfig.ssh_tunnel_public_key,
              }}
              onConfigChange={(tunnelConfig) => setPostgresConfig({ ...postgresConfig, ...tunnelConfig })}
              databaseLabel="PostgreSQL"
              authMode={sshTunnelAuthMode}
              onAuthModeChange={setSshTunnelAuthMode}
              generatingKey={sshTunnelGeneratingKey}
              onGenerateKey={handleGenerateTunnelSSHKey}
              keyCopied={sshTunnelKeyCopied}
              onCopyPublicKey={handleCopyTunnelPublicKey}
              toolName={name || 'postgres'}
            />

            {/* Database Host/Port - context changes based on tunnel mode */}
            <div className="form-row">
              <div className="form-group">
                <label>{postgresConfig.ssh_tunnel_enabled ? 'Database Host (on SSH server)' : 'Host'} *</label>
                <input
                  type="text"
                  value={postgresConfig.host || ''}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, host: e.target.value })}
                  placeholder={postgresConfig.ssh_tunnel_enabled ? 'localhost' : 'db.example.com'}
                />
                {postgresConfig.ssh_tunnel_enabled && (
                  <p className="field-help">Usually "localhost" - the database host as seen from the SSH server</p>
                )}
              </div>
              <div className="form-group form-group-small">
                <label>Port</label>
                <input
                  type="number"
                  value={postgresConfig.port || 5432}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, port: parseInt(e.target.value) || 5432 })}
                  min={1}
                  max={65535}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>User *</label>
                <input
                  type="text"
                  value={postgresConfig.user || ''}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, user: e.target.value })}
                  placeholder="postgres"
                />
              </div>
              <div className="form-group">
                <label>Password *</label>
                <input
                  type="password"
                  value={postgresConfig.password || ''}
                  onChange={(e) => setPostgresConfig({ ...postgresConfig, password: e.target.value })}
                  placeholder="********"
                />
              </div>
            </div>
            <div className="form-group">
              <label>Database</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {discoveredDatabases.length > 0 ? (
                  <select
                    value={postgresConfig.database || ''}
                    onChange={(e) => setPostgresConfig({ ...postgresConfig, database: e.target.value })}
                    style={{ flex: 1 }}
                  >
                    <option value="">Select database...</option>
                    {discoveredDatabases.map(db => (
                      <option key={db} value={db}>{db}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={postgresConfig.database || ''}
                    onChange={(e) => setPostgresConfig({ ...postgresConfig, database: e.target.value })}
                    placeholder="mydb"
                    style={{ flex: 1 }}
                  />
                )}
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleDiscoverDatabases}
                  disabled={discoveringDatabases || (!postgresConfig.ssh_tunnel_enabled && !postgresConfig.host) || !postgresConfig.user || !postgresConfig.password}
                  title="Discover available databases"
                >
                  {discoveringDatabases ? 'Discovering...' : 'Discover'}
                </button>
              </div>
              <p className="field-help">
                {databaseDiscoveryError ? (
                  <span style={{ color: '#dc3545' }}>{databaseDiscoveryError}</span>
                ) : discoveredDatabases.length > 0 ? (
                  `Found ${discoveredDatabases.length} database(s). Select one or type manually.`
                ) : (
                  'Enter credentials, then click Discover to find available databases.'
                )}
              </p>
            </div>
          </div>
        ) : pgConnectionMode === 'container' ? (
          <div className="connection-panel">
            <DockerConnectionPanel
              dockerContainers={dockerContainers}
              dockerNetworks={dockerNetworks}
              currentNetwork={currentNetwork}
              currentContainer={currentContainer}
              loadingDocker={loadingDocker}
              connectingNetwork={connectingNetwork}
              selectedNetwork={postgresConfig.docker_network || ''}
              selectedContainer={postgresConfig.container || ''}
              onDiscoverDocker={handleDiscoverDocker}
              onConnectNetwork={handleConnectNetwork}
              onNetworkChange={(network) => setPostgresConfig({ ...postgresConfig, docker_network: network })}
              onContainerChange={(container) => setPostgresConfig({ ...postgresConfig, container })}
              containerFilter={(c) => c.image.toLowerCase().includes('postgres')}
              containerLabel={(c) => `${c.name}${c.image.toLowerCase().includes('postgres') ? ' (PostgreSQL)' : ''}`}
              containerCountLabel="PostgreSQL container(s)"
              containerHelpText="Uses container's POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB environment variables."
              fallbackPlaceholder="my-postgres-container"
            />

            <div className="form-group">
              <label>Database (optional override)</label>
              <input
                type="text"
                value={postgresConfig.database || ''}
                onChange={(e) => setPostgresConfig({ ...postgresConfig, database: e.target.value })}
                placeholder="Leave empty to use POSTGRES_DB"
              />
            </div>
          </div>
        ) : null}

        {/* Schema Indexing Section */}
        <div className="schema-indexing-section" style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
          <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Schema Indexing</h4>
          <p className="field-help" style={{ marginBottom: '1rem' }}>
            Index database schema for faster AI-powered queries. Requires an embedding provider to be configured.
          </p>

          <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={postgresConfig.schema_index_enabled ?? false}
              onChange={(e) => setPostgresConfig({
                ...postgresConfig,
                schema_index_enabled: e.target.checked
              })}
              style={{ width: 'auto', margin: 0 }}
            />
            <span>Enable schema indexing</span>
          </label>

          {postgresConfig.schema_index_enabled && (
            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label>Re-index interval (hours)</label>
              <input
                type="number"
                value={postgresConfig.schema_index_interval_hours ?? 24}
                onChange={(e) => setPostgresConfig({
                  ...postgresConfig,
                  schema_index_interval_hours: parseInt(e.target.value) || 24
                })}
                min={1}
                max={168}
                style={{ maxWidth: '120px' }}
              />
              <p className="field-help">
                How often to automatically re-index the database schema (1-168 hours).
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMssqlConnection = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Connect to a Microsoft SQL Server or Azure SQL database.
        </p>

        <div className="connection-panel">
          {/* SSH Tunnel Option - shown first for clear context */}
          <SSHTunnelPanel
            enabled={mssqlConfig.ssh_tunnel_enabled ?? false}
            onEnabledChange={(enabled) => {
              // When enabling tunnel, default host to localhost if empty
              const updates: Partial<typeof mssqlConfig> = { ssh_tunnel_enabled: enabled };
              if (enabled && !mssqlConfig.host) {
                updates.host = 'localhost';
              }
              setMssqlConfig({ ...mssqlConfig, ...updates });
              setMssqlDiscoveredDatabases([]);
            }}
            config={{
              ssh_tunnel_host: mssqlConfig.ssh_tunnel_host,
              ssh_tunnel_port: mssqlConfig.ssh_tunnel_port,
              ssh_tunnel_user: mssqlConfig.ssh_tunnel_user,
              ssh_tunnel_password: mssqlConfig.ssh_tunnel_password,
              ssh_tunnel_key_path: mssqlConfig.ssh_tunnel_key_path,
              ssh_tunnel_key_content: mssqlConfig.ssh_tunnel_key_content,
              ssh_tunnel_key_passphrase: mssqlConfig.ssh_tunnel_key_passphrase,
              ssh_tunnel_public_key: mssqlConfig.ssh_tunnel_public_key,
            }}
            onConfigChange={(tunnelConfig) => setMssqlConfig({ ...mssqlConfig, ...tunnelConfig })}
            databaseLabel="SQL Server"
            authMode={sshTunnelAuthMode}
            onAuthModeChange={setSshTunnelAuthMode}
            generatingKey={sshTunnelGeneratingKey}
            onGenerateKey={handleGenerateTunnelSSHKey}
            keyCopied={sshTunnelKeyCopied}
            onCopyPublicKey={handleCopyTunnelPublicKey}
            toolName={name || 'mssql'}
          />

          {/* Database Host/Port - context changes based on tunnel mode */}
          <div className="form-row">
            <div className="form-group">
              <label>{mssqlConfig.ssh_tunnel_enabled ? 'Database Host (on SSH server)' : 'Host'} *</label>
              <input
                type="text"
                value={mssqlConfig.host || ''}
                onChange={(e) => {
                  setMssqlConfig({ ...mssqlConfig, host: e.target.value });
                  setMssqlDiscoveredDatabases([]);
                }}
                placeholder={mssqlConfig.ssh_tunnel_enabled ? 'localhost' : 'server.database.windows.net'}
              />
              {mssqlConfig.ssh_tunnel_enabled && (
                <p className="field-help">Usually "localhost" - the SQL Server host as seen from the SSH server</p>
              )}
            </div>
            <div className="form-group form-group-small">
              <label>Port</label>
              <input
                type="number"
                value={mssqlConfig.port || 1433}
                onChange={(e) => setMssqlConfig({ ...mssqlConfig, port: parseInt(e.target.value) || 1433 })}
                min={1}
                max={65535}
              />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>User *</label>
              <input
                type="text"
                value={mssqlConfig.user || ''}
                onChange={(e) => {
                  setMssqlConfig({ ...mssqlConfig, user: e.target.value });
                  setMssqlDiscoveredDatabases([]);
                }}
                placeholder="sa or domain\\user"
              />
            </div>
            <div className="form-group">
              <label>Password *</label>
              <input
                type="password"
                value={mssqlConfig.password || ''}
                onChange={(e) => {
                  setMssqlConfig({ ...mssqlConfig, password: e.target.value });
                  setMssqlDiscoveredDatabases([]);
                }}
                placeholder="********"
              />
            </div>
          </div>
          <div className="form-group">
            <label>Database</label>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              {mssqlDiscoveredDatabases.length > 0 ? (
                <select
                  value={mssqlConfig.database || ''}
                  onChange={(e) => setMssqlConfig({ ...mssqlConfig, database: e.target.value })}
                  style={{ flex: 1 }}
                >
                  <option value="">Select database...</option>
                  {mssqlDiscoveredDatabases.map(db => (
                    <option key={db} value={db}>{db}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={mssqlConfig.database || ''}
                  onChange={(e) => setMssqlConfig({ ...mssqlConfig, database: e.target.value })}
                  placeholder="master"
                  style={{ flex: 1 }}
                />
              )}
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={handleDiscoverMssqlDatabases}
                disabled={mssqlDiscoveringDatabases || (!mssqlConfig.ssh_tunnel_enabled && !mssqlConfig.host) || !mssqlConfig.user || !mssqlConfig.password}
                title="Discover available databases"
              >
                {mssqlDiscoveringDatabases ? 'Discovering...' : 'Discover'}
              </button>
            </div>
            <p className="field-help">
              {mssqlDatabaseDiscoveryError ? (
                <span style={{ color: '#dc3545' }}>{mssqlDatabaseDiscoveryError}</span>
              ) : mssqlDiscoveredDatabases.length > 0 ? (
                `Found ${mssqlDiscoveredDatabases.length} database(s). Select one or type manually.`
              ) : (
                'Enter credentials, then click Discover to find available databases.'
              )}
            </p>
          </div>
        </div>

        {/* Schema Indexing Section */}
        <div className="schema-indexing-section" style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
          <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Schema Indexing</h4>
          <p className="field-help" style={{ marginBottom: '1rem' }}>
            Index database schema for faster AI-powered queries. Requires an embedding provider to be configured.
          </p>

          <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={mssqlConfig.schema_index_enabled ?? false}
              onChange={(e) => setMssqlConfig({
                ...mssqlConfig,
                schema_index_enabled: e.target.checked
              })}
              style={{ width: 'auto', margin: 0 }}
            />
            <span>Enable schema indexing</span>
          </label>

          {mssqlConfig.schema_index_enabled && (
            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label>Re-index interval (hours)</label>
              <input
                type="number"
                value={mssqlConfig.schema_index_interval_hours ?? 24}
                onChange={(e) => setMssqlConfig({
                  ...mssqlConfig,
                  schema_index_interval_hours: parseInt(e.target.value) || 24
                })}
                min={1}
                max={168}
                style={{ maxWidth: '120px' }}
              />
              <p className="field-help">
                How often to automatically re-index the database schema (1-168 hours).
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderMysqlConnection = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Choose your connection method:
        </p>

        <div className="connection-tabs">
          <button
            type="button"
            className={`connection-tab ${mysqlConnectionMode === 'direct' ? 'active' : ''}`}
            onClick={() => {
              setMysqlConnectionMode('direct');
              setMysqlConfig({ ...mysqlConfig, container: '' });
              setMysqlDiscoveredDatabases([]);
            }}
          >
            Direct Connection
          </button>
          <button
            type="button"
            className={`connection-tab ${mysqlConnectionMode === 'container' ? 'active' : ''}`}
            onClick={() => {
              setMysqlConnectionMode('container');
              setMysqlConfig({ ...mysqlConfig, host: '', user: '', password: '' });
              setMysqlDiscoveredDatabases([]);
            }}
          >
            Docker Container
          </button>
        </div>

        {mysqlConnectionMode === 'direct' ? (
          <div className="connection-panel">
            {/* SSH Tunnel Option - shown first for clear context */}
            <SSHTunnelPanel
              enabled={mysqlConfig.ssh_tunnel_enabled ?? false}
              onEnabledChange={(enabled) => {
                // When enabling tunnel, default host to localhost if empty
                const updates: Partial<typeof mysqlConfig> = { ssh_tunnel_enabled: enabled };
                if (enabled && !mysqlConfig.host) {
                  updates.host = 'localhost';
                }
                setMysqlConfig({ ...mysqlConfig, ...updates });
                setMysqlDiscoveredDatabases([]);
              }}
              config={{
                ssh_tunnel_host: mysqlConfig.ssh_tunnel_host,
                ssh_tunnel_port: mysqlConfig.ssh_tunnel_port,
                ssh_tunnel_user: mysqlConfig.ssh_tunnel_user,
                ssh_tunnel_password: mysqlConfig.ssh_tunnel_password,
                ssh_tunnel_key_path: mysqlConfig.ssh_tunnel_key_path,
                ssh_tunnel_key_content: mysqlConfig.ssh_tunnel_key_content,
                ssh_tunnel_key_passphrase: mysqlConfig.ssh_tunnel_key_passphrase,
                ssh_tunnel_public_key: mysqlConfig.ssh_tunnel_public_key,
              }}
              onConfigChange={(tunnelConfig) => setMysqlConfig({ ...mysqlConfig, ...tunnelConfig })}
              databaseLabel="MySQL/MariaDB"
              authMode={sshTunnelAuthMode}
              onAuthModeChange={setSshTunnelAuthMode}
              generatingKey={sshTunnelGeneratingKey}
              onGenerateKey={handleGenerateTunnelSSHKey}
              keyCopied={sshTunnelKeyCopied}
              onCopyPublicKey={handleCopyTunnelPublicKey}
              toolName={name || 'mysql'}
            />

            {/* Database Host/Port - context changes based on tunnel mode */}
            <div className="form-row">
              <div className="form-group">
                <label>{mysqlConfig.ssh_tunnel_enabled ? 'Database Host (on SSH server)' : 'Host'} *</label>
                <input
                  type="text"
                  value={mysqlConfig.host || ''}
                  onChange={(e) => {
                    setMysqlConfig({ ...mysqlConfig, host: e.target.value });
                    setMysqlDiscoveredDatabases([]);
                  }}
                  placeholder={mysqlConfig.ssh_tunnel_enabled ? 'localhost' : 'db.example.com'}
                />
                {mysqlConfig.ssh_tunnel_enabled && (
                  <p className="field-help">Usually "localhost" - the MySQL host as seen from the SSH server</p>
                )}
              </div>
              <div className="form-group form-group-small">
                <label>Port</label>
                <input
                  type="number"
                  value={mysqlConfig.port || 3306}
                  onChange={(e) => setMysqlConfig({ ...mysqlConfig, port: parseInt(e.target.value) || 3306 })}
                  min={1}
                  max={65535}
                />
              </div>
            </div>
            <div className="form-row">
              <div className="form-group">
                <label>User *</label>
                <input
                  type="text"
                  value={mysqlConfig.user || ''}
                  onChange={(e) => {
                    setMysqlConfig({ ...mysqlConfig, user: e.target.value });
                    setMysqlDiscoveredDatabases([]);
                  }}
                  placeholder="root"
                />
              </div>
              <div className="form-group">
                <label>Password *</label>
                <input
                  type="password"
                  value={mysqlConfig.password || ''}
                  onChange={(e) => {
                    setMysqlConfig({ ...mysqlConfig, password: e.target.value });
                    setMysqlDiscoveredDatabases([]);
                  }}
                  placeholder="********"
                />
              </div>
            </div>
            <div className="form-group">
              <label>Database</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {mysqlDiscoveredDatabases.length > 0 ? (
                  <select
                    value={mysqlConfig.database || ''}
                    onChange={(e) => setMysqlConfig({ ...mysqlConfig, database: e.target.value })}
                    style={{ flex: 1 }}
                  >
                    <option value="">Select database...</option>
                    {mysqlDiscoveredDatabases.map(db => (
                      <option key={db} value={db}>{db}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={mysqlConfig.database || ''}
                    onChange={(e) => setMysqlConfig({ ...mysqlConfig, database: e.target.value })}
                    placeholder="mydb"
                    style={{ flex: 1 }}
                  />
                )}
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleDiscoverMysqlDatabases}
                  disabled={mysqlDiscoveringDatabases || (!mysqlConfig.ssh_tunnel_enabled && !mysqlConfig.host) || !mysqlConfig.user || !mysqlConfig.password}
                  title="Discover available databases"
                >
                  {mysqlDiscoveringDatabases ? 'Discovering...' : 'Discover'}
                </button>
              </div>
              <p className="field-help">
                {mysqlDatabaseDiscoveryError ? (
                  <span style={{ color: '#dc3545' }}>{mysqlDatabaseDiscoveryError}</span>
                ) : mysqlDiscoveredDatabases.length > 0 ? (
                  `Found ${mysqlDiscoveredDatabases.length} database(s). Select one or type manually.`
                ) : (
                  'Enter credentials, then click Discover to find available databases.'
                )}
              </p>
            </div>
          </div>
        ) : mysqlConnectionMode === 'container' ? (
          <div className="connection-panel">
            <DockerConnectionPanel
              dockerContainers={dockerContainers}
              dockerNetworks={dockerNetworks}
              currentNetwork={currentNetwork}
              currentContainer={currentContainer}
              loadingDocker={loadingDocker}
              connectingNetwork={connectingNetwork}
              selectedNetwork={mysqlConfig.docker_network || ''}
              selectedContainer={mysqlConfig.container || ''}
              onDiscoverDocker={handleDiscoverDocker}
              onConnectNetwork={handleConnectNetwork}
              onNetworkChange={(network) => setMysqlConfig({ ...mysqlConfig, docker_network: network })}
              onContainerChange={(container) => {
                setMysqlConfig({ ...mysqlConfig, container });
                setMysqlDiscoveredDatabases([]);
              }}
              containerFilter={(c) => c.image.toLowerCase().includes('mysql') || c.image.toLowerCase().includes('mariadb')}
              containerLabel={(c) => {
                const isMysql = c.image.toLowerCase().includes('mysql');
                const isMariadb = c.image.toLowerCase().includes('mariadb');
                return `${c.name}${isMysql ? ' (MySQL)' : isMariadb ? ' (MariaDB)' : ''}`;
              }}
              containerCountLabel="MySQL/MariaDB container(s)"
              containerHelpText="Uses container's MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE (or MYSQL_ROOT_PASSWORD for root) environment variables."
              fallbackPlaceholder="my-mysql-container"
            />

            <div className="form-group">
              <label>Database</label>
              <div style={{ display: 'flex', gap: '0.5rem' }}>
                {mysqlDiscoveredDatabases.length > 0 ? (
                  <select
                    value={mysqlConfig.database || ''}
                    onChange={(e) => setMysqlConfig({ ...mysqlConfig, database: e.target.value })}
                    style={{ flex: 1 }}
                  >
                    <option value="">Select database...</option>
                    {mysqlDiscoveredDatabases.map(db => (
                      <option key={db} value={db}>{db}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={mysqlConfig.database || ''}
                    onChange={(e) => setMysqlConfig({ ...mysqlConfig, database: e.target.value })}
                    placeholder="Leave empty to use MYSQL_DATABASE"
                    style={{ flex: 1 }}
                  />
                )}
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={handleDiscoverMysqlDatabases}
                  disabled={mysqlDiscoveringDatabases || !mysqlConfig.container}
                  title="Discover available databases"
                >
                  {mysqlDiscoveringDatabases ? 'Discovering...' : 'Discover'}
                </button>
              </div>
              <p className="field-help">
                {mysqlDatabaseDiscoveryError ? (
                  <span style={{ color: '#dc3545' }}>{mysqlDatabaseDiscoveryError}</span>
                ) : mysqlDiscoveredDatabases.length > 0 ? (
                  `Found ${mysqlDiscoveredDatabases.length} database(s). Select one or type manually.`
                ) : (
                  'Select a container, then click Discover to find available databases.'
                )}
              </p>
            </div>
          </div>
        ) : null}

        {/* Schema Indexing Section */}
        <div className="schema-indexing-section" style={{ marginTop: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
          <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Schema Indexing</h4>
          <p className="field-help" style={{ marginBottom: '1rem' }}>
            Index database schema for faster AI-powered queries. Requires an embedding provider to be configured.
          </p>

          <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={mysqlConfig.schema_index_enabled ?? false}
              onChange={(e) => setMysqlConfig({
                ...mysqlConfig,
                schema_index_enabled: e.target.checked
              })}
              style={{ width: 'auto', margin: 0 }}
            />
            <span>Enable schema indexing</span>
          </label>

          {mysqlConfig.schema_index_enabled && (
            <div className="form-group" style={{ marginTop: '1rem' }}>
              <label>Re-index interval (hours)</label>
              <input
                type="number"
                value={mysqlConfig.schema_index_interval_hours ?? 24}
                onChange={(e) => setMysqlConfig({
                  ...mysqlConfig,
                  schema_index_interval_hours: parseInt(e.target.value) || 24
                })}
                min={1}
                max={168}
                style={{ maxWidth: '120px' }}
              />
              <p className="field-help">
                How often to automatically re-index the database schema (1-168 hours).
              </p>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Handler for PDM database discovery
  const handleDiscoverPdmDatabases = async () => {
    if (!pdmConfig.host || !pdmConfig.user || !pdmConfig.password) {
      setPdmDatabaseDiscoveryError('Host, user, and password are required to discover databases');
      return;
    }

    setPdmDiscoveringDatabases(true);
    setPdmDatabaseDiscoveryError(null);
    setPdmDiscoveredDatabases([]);

    try {
      const result = await api.discoverMssqlDatabases({
        host: pdmConfig.host,
        port: pdmConfig.port || 1433,
        user: pdmConfig.user,
        password: pdmConfig.password,
        // SSH tunnel fields
        ssh_tunnel_enabled: pdmConfig.ssh_tunnel_enabled,
        ssh_tunnel_host: pdmConfig.ssh_tunnel_host,
        ssh_tunnel_port: pdmConfig.ssh_tunnel_port,
        ssh_tunnel_user: pdmConfig.ssh_tunnel_user,
        ssh_tunnel_password: pdmConfig.ssh_tunnel_password,
        ssh_tunnel_key_path: pdmConfig.ssh_tunnel_key_path,
        ssh_tunnel_key_content: pdmConfig.ssh_tunnel_key_content,
        ssh_tunnel_key_passphrase: pdmConfig.ssh_tunnel_key_passphrase,
      });

      if (result.success) {
        setPdmDiscoveredDatabases(result.databases);
      } else {
        setPdmDatabaseDiscoveryError(result.error || 'Failed to discover databases');
      }
    } catch (err) {
      setPdmDatabaseDiscoveryError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setPdmDiscoveringDatabases(false);
    }
  };

  // Handler for PDM schema discovery (extensions and variables)
  const handleDiscoverPdmSchema = async () => {
    if (!pdmConfig.host || !pdmConfig.user || !pdmConfig.password || !pdmConfig.database) {
      setPdmSchemaDiscoveryError('Database connection must be configured first');
      return;
    }

    setPdmDiscoveringSchema(true);
    setPdmSchemaDiscoveryError(null);

    try {
      const result = await api.discoverPdmSchema({
        host: pdmConfig.host,
        port: pdmConfig.port || 1433,
        user: pdmConfig.user,
        password: pdmConfig.password,
        database: pdmConfig.database,
        // SSH tunnel fields
        ssh_tunnel_enabled: pdmConfig.ssh_tunnel_enabled,
        ssh_tunnel_host: pdmConfig.ssh_tunnel_host,
        ssh_tunnel_port: pdmConfig.ssh_tunnel_port,
        ssh_tunnel_user: pdmConfig.ssh_tunnel_user,
        ssh_tunnel_password: pdmConfig.ssh_tunnel_password,
        ssh_tunnel_key_path: pdmConfig.ssh_tunnel_key_path,
        ssh_tunnel_key_content: pdmConfig.ssh_tunnel_key_content,
        ssh_tunnel_key_passphrase: pdmConfig.ssh_tunnel_key_passphrase,
      });

      if (result.success) {
        setPdmDiscoveredExtensions(result.file_extensions);
        setPdmDiscoveredVariables(result.variable_names);
        setPdmDocumentCount(result.document_count);

        // Auto-select common SolidWorks extensions if not already configured
        if (!pdmConfig.file_extensions || pdmConfig.file_extensions.length === 0) {
          const defaultExts = ['.SLDPRT', '.SLDASM', '.SLDDRW'];
          const availableDefaults = defaultExts.filter(ext =>
            result.file_extensions.some(e => e.toUpperCase() === ext.toUpperCase())
          );
          if (availableDefaults.length > 0) {
            setPdmConfig(prev => ({ ...prev, file_extensions: availableDefaults }));
          }
        }
      } else {
        setPdmSchemaDiscoveryError(result.error || 'Failed to discover schema');
      }
    } catch (err) {
      setPdmSchemaDiscoveryError(err instanceof Error ? err.message : 'Discovery failed');
    } finally {
      setPdmDiscoveringSchema(false);
    }
  };

  // Set the ref so useEffect can call this function
  handleDiscoverPdmSchemaRef.current = handleDiscoverPdmSchema;

  const renderPdmConnection = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Connect to a SolidWorks PDM database to index engineering documents.
        </p>

        <div className="connection-panel">
          <h4 style={{ marginTop: '0', marginBottom: '0.75rem' }}>Database Connection</h4>

          {/* SSH Tunnel Option - shown first for clear context */}
          <SSHTunnelPanel
            enabled={pdmConfig.ssh_tunnel_enabled ?? false}
            onEnabledChange={(enabled) => {
              // When enabling tunnel, default host to localhost if empty
              const updates: Partial<typeof pdmConfig> = { ssh_tunnel_enabled: enabled };
              if (enabled && !pdmConfig.host) {
                updates.host = 'localhost';
              }
              setPdmConfig({ ...pdmConfig, ...updates });
              setPdmDiscoveredDatabases([]);
            }}
            config={{
              ssh_tunnel_host: pdmConfig.ssh_tunnel_host,
              ssh_tunnel_port: pdmConfig.ssh_tunnel_port,
              ssh_tunnel_user: pdmConfig.ssh_tunnel_user,
              ssh_tunnel_password: pdmConfig.ssh_tunnel_password,
              ssh_tunnel_key_path: pdmConfig.ssh_tunnel_key_path,
              ssh_tunnel_key_content: pdmConfig.ssh_tunnel_key_content,
              ssh_tunnel_key_passphrase: pdmConfig.ssh_tunnel_key_passphrase,
              ssh_tunnel_public_key: pdmConfig.ssh_tunnel_public_key,
            }}
            onConfigChange={(tunnelConfig) => setPdmConfig({ ...pdmConfig, ...tunnelConfig })}
            databaseLabel="SQL Server"
            authMode={sshTunnelAuthMode}
            onAuthModeChange={setSshTunnelAuthMode}
            generatingKey={sshTunnelGeneratingKey}
            onGenerateKey={handleGenerateTunnelSSHKey}
            keyCopied={sshTunnelKeyCopied}
            onCopyPublicKey={handleCopyTunnelPublicKey}
            toolName={name || 'pdm'}
          />

          {/* Database Host/Port - context changes based on tunnel mode */}
          <div className="form-row">
            <div className="form-group">
              <label>{pdmConfig.ssh_tunnel_enabled ? 'Database Host (on SSH server)' : 'Host'} *</label>
              <input
                type="text"
                value={pdmConfig.host || ''}
                onChange={(e) => {
                  setPdmConfig({ ...pdmConfig, host: e.target.value });
                  setPdmDiscoveredDatabases([]);
                }}
                placeholder={pdmConfig.ssh_tunnel_enabled ? 'localhost' : 'server.database.windows.net'}
              />
              {pdmConfig.ssh_tunnel_enabled && (
                <p className="field-help">Usually "localhost" - the SQL Server host as seen from the SSH server</p>
              )}
            </div>
            <div className="form-group form-group-small">
              <label>Port</label>
              <input
                type="number"
                value={pdmConfig.port || 1433}
                onChange={(e) => setPdmConfig({ ...pdmConfig, port: parseInt(e.target.value) || 1433 })}
                min={1}
                max={65535}
              />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>User *</label>
              <input
                type="text"
                value={pdmConfig.user || ''}
                onChange={(e) => {
                  setPdmConfig({ ...pdmConfig, user: e.target.value });
                  setPdmDiscoveredDatabases([]);
                }}
                placeholder="sa or domain\\user"
              />
            </div>
            <div className="form-group">
              <label>Password *</label>
              <input
                type="password"
                value={pdmConfig.password || ''}
                onChange={(e) => {
                  setPdmConfig({ ...pdmConfig, password: e.target.value });
                  setPdmDiscoveredDatabases([]);
                }}
                placeholder="********"
              />
            </div>
          </div>

          {/* Database discovery and selection */}
          <div className="form-group">
            <label>Database</label>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'flex-start' }}>
              {pdmDiscoveredDatabases.length > 0 ? (
                <select
                  value={pdmConfig.database || ''}
                  onChange={(e) => setPdmConfig({ ...pdmConfig, database: e.target.value })}
                  style={{ flex: 1 }}
                >
                  <option value="">-- Select Database --</option>
                  {pdmDiscoveredDatabases.map(db => (
                    <option key={db} value={db}>{db}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="text"
                  value={pdmConfig.database || ''}
                  onChange={(e) => setPdmConfig({ ...pdmConfig, database: e.target.value })}
                  placeholder="PDM_Vault"
                  style={{ flex: 1 }}
                />
              )}
              <button
                type="button"
                className="btn btn-sm"
                onClick={handleDiscoverPdmDatabases}
                disabled={pdmDiscoveringDatabases || (!pdmConfig.ssh_tunnel_enabled && !pdmConfig.host) || !pdmConfig.user || !pdmConfig.password}
                style={{ whiteSpace: 'nowrap', padding: '12px 16px' }}
              >
                {pdmDiscoveringDatabases ? 'Discovering...' : 'Discover'}
              </button>
            </div>
            {pdmDatabaseDiscoveryError && (
              <p className="field-help" style={{ color: 'var(--error-color)' }}>
                {pdmDatabaseDiscoveryError}
              </p>
            )}
            {pdmDiscoveredDatabases.length > 0 && (
              <p className="field-help">
                Found {pdmDiscoveredDatabases.length} database(s).
              </p>
            )}
          </div>
        </div>
      </div>
    );
  };

  const renderPdmFiltering = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Configure which documents and metadata to index from the PDM vault.
          {pdmDocumentCount > 0 && (
            <span style={{ marginLeft: '0.5rem', fontWeight: 500 }}>
              (Found {pdmDocumentCount.toLocaleString()} total documents)
            </span>
          )}
        </p>

        {pdmDiscoveringSchema ? (
          <div className="discovering-panel" style={{ padding: '2rem', textAlign: 'center' }}>
            <p>Discovering PDM schema...</p>
          </div>
        ) : pdmSchemaDiscoveryError ? (
          <div className="error-panel" style={{ padding: '1rem', backgroundColor: 'rgba(255,0,0,0.1)', borderRadius: '4px', marginBottom: '1rem' }}>
            <p style={{ color: 'var(--error-color)', margin: 0 }}>{pdmSchemaDiscoveryError}</p>
            <button
              type="button"
              className="btn btn-sm"
              onClick={handleDiscoverPdmSchema}
              style={{ marginTop: '0.5rem' }}
            >
              Retry Discovery
            </button>
          </div>
        ) : (
          <>
            {/* File Extensions Section */}
            <div className="pdm-section" style={{ marginBottom: '1.5rem' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>File Extensions</h4>
              <p className="field-help" style={{ marginBottom: '0.75rem' }}>
                PDM indexing captures metadata (filenames, variables, BOM) - not file contents.
                PDM vaults typically store many file types beyond SolidWorks files.
              </p>

              <div style={{ marginBottom: '0.75rem' }}>
                <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={(pdmConfig.file_extensions || []).length > 0}
                    onChange={(e) => {
                      if (e.target.checked) {
                        // Enable filtering - default to SolidWorks CAD files only
                        setPdmConfig({ ...pdmConfig, file_extensions: ['.SLDPRT', '.SLDASM', '.SLDDRW'] });
                      } else {
                        // Disable filtering - index all file types
                        setPdmConfig({ ...pdmConfig, file_extensions: [] });
                      }
                    }}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Limit to specific extensions</span>
                  <span className="field-help" style={{ marginLeft: '0.5rem', fontStyle: 'italic' }}>
                    {(pdmConfig.file_extensions || []).length > 0
                      ? `(${pdmConfig.file_extensions?.length} selected)`
                      : '(indexing all file types)'}
                  </span>
                </label>
              </div>

              {(pdmConfig.file_extensions || []).length > 0 && pdmDiscoveredExtensions.length > 0 && (
                <>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                    <input
                      type="text"
                      value={pdmExtensionFilter}
                      onChange={(e) => setPdmExtensionFilter(e.target.value)}
                      placeholder="Filter extensions..."
                      style={{ maxWidth: '200px', margin: 0, padding: '8px 12px' }}
                    />
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setPdmConfig({ ...pdmConfig, file_extensions: [...pdmDiscoveredExtensions] })}
                      style={{ padding: '8px 16px' }}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setPdmConfig({ ...pdmConfig, file_extensions: [] })}
                      style={{ padding: '8px 16px' }}
                    >
                      Select None
                    </button>
                  </div>
                  <div className="extension-grid" style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                    maxHeight: '200px',
                    overflow: 'auto',
                    padding: '0.5rem',
                    border: '1px solid var(--border-color)',
                    borderRadius: '4px',
                    backgroundColor: 'var(--bg-secondary)'
                  }}>
                    {pdmDiscoveredExtensions
                      .filter(ext => ext.toLowerCase().includes(pdmExtensionFilter.toLowerCase()))
                      .map(ext => {
                        const isSelected = (pdmConfig.file_extensions || []).some(
                          e => e.toUpperCase() === ext.toUpperCase()
                        );
                        return (
                          <label
                            key={ext}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              cursor: 'pointer',
                              padding: '0.25rem 0.5rem',
                              borderRadius: '4px',
                              backgroundColor: isSelected ? 'var(--primary-color-light)' : 'var(--bg-primary)',
                              border: '1px solid var(--border-color)',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={(e) => {
                                const current = pdmConfig.file_extensions || [];
                                if (e.target.checked) {
                                  setPdmConfig({ ...pdmConfig, file_extensions: [...current, ext] });
                                } else {
                                  setPdmConfig({
                                    ...pdmConfig,
                                    file_extensions: current.filter(x => x.toUpperCase() !== ext.toUpperCase())
                                  });
                                }
                              }}
                              style={{ width: 'auto', margin: 0 }}
                            />
                            <span style={{ fontSize: '0.9rem' }}>{ext}</span>
                          </label>
                        );
                      })}
                  </div>
                </>
              )}

              <div style={{ marginTop: '0.75rem' }}>
                <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={pdmConfig.exclude_deleted ?? true}
                    onChange={(e) => setPdmConfig({ ...pdmConfig, exclude_deleted: e.target.checked })}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Exclude deleted documents</span>
                </label>
              </div>
            </div>

            {/* Variable Names Section */}
            <div className="pdm-section" style={{ marginBottom: '1.5rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>PDM Variables</h4>
              <p className="field-help" style={{ marginBottom: '0.75rem' }}>
                Variables are custom properties stored in PDM for each document.
              </p>

              <div style={{ marginBottom: '0.75rem' }}>
                <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={(pdmConfig.variable_names || []).length > 0}
                    onChange={(e) => {
                      if (e.target.checked) {
                        // Enable filtering with common PDM variables as default
                        setPdmConfig({ ...pdmConfig, variable_names: ['Description', 'Revision', 'Material', 'Status'] });
                      } else {
                        // Disable filtering - extract all variables
                        setPdmConfig({ ...pdmConfig, variable_names: [] });
                      }
                    }}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Limit to specific variables</span>
                  <span className="field-help" style={{ marginLeft: '0.5rem', fontStyle: 'italic' }}>
                    {(pdmConfig.variable_names || []).length > 0
                      ? `(${pdmConfig.variable_names?.length} selected)`
                      : '(extracting all variables)'}
                  </span>
                </label>
              </div>

              {(pdmConfig.variable_names || []).length > 0 && pdmDiscoveredVariables.length > 0 && (
                <>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem', flexWrap: 'wrap' }}>
                    <input
                      type="text"
                      value={pdmVariableFilter}
                      onChange={(e) => setPdmVariableFilter(e.target.value)}
                      placeholder="Filter variables..."
                      style={{ maxWidth: '200px', margin: 0, padding: '8px 12px' }}
                    />
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setPdmConfig({ ...pdmConfig, variable_names: [...pdmDiscoveredVariables] })}
                      style={{ padding: '8px 16px' }}
                    >
                      Select All
                    </button>
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={() => setPdmConfig({ ...pdmConfig, variable_names: [] })}
                      style={{ padding: '8px 16px' }}
                    >
                      Select None
                    </button>
                  </div>
                  <div className="variable-grid" style={{
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                    maxHeight: '250px',
                    overflow: 'auto',
                    padding: '0.5rem',
                    border: '1px solid var(--border-color)',
                    borderRadius: '4px',
                    backgroundColor: 'var(--bg-secondary)'
                  }}>
                    {pdmDiscoveredVariables
                      .filter(varName => varName.toLowerCase().includes(pdmVariableFilter.toLowerCase()))
                      .map(varName => {
                        const isSelected = (pdmConfig.variable_names || []).includes(varName);
                        return (
                          <label
                            key={varName}
                            style={{
                              display: 'flex',
                              alignItems: 'center',
                              gap: '0.5rem',
                              cursor: 'pointer',
                              padding: '0.25rem 0.5rem',
                              borderRadius: '4px',
                              backgroundColor: isSelected ? 'var(--primary-color-light)' : 'var(--bg-primary)',
                              border: '1px solid var(--border-color)',
                              whiteSpace: 'nowrap'
                            }}
                          >
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={(e) => {
                                const current = pdmConfig.variable_names || [];
                                if (e.target.checked) {
                                  setPdmConfig({ ...pdmConfig, variable_names: [...current, varName] });
                                } else {
                                  setPdmConfig({
                                    ...pdmConfig,
                                    variable_names: current.filter(x => x !== varName)
                                  });
                                }
                              }}
                              style={{ width: 'auto', margin: 0 }}
                            />
                            <span style={{ fontSize: '0.9rem' }}>{varName}</span>
                          </label>
                        );
                      })}
                  </div>
                </>
              )}
            </div>

            {/* Metadata Options Section */}
            <div className="pdm-section" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Indexing Options</h4>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={pdmConfig.include_bom ?? true}
                    onChange={(e) => setPdmConfig({ ...pdmConfig, include_bom: e.target.checked })}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Include BOM (Bill of Materials)</span>
                </label>

                <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={pdmConfig.include_folder_path ?? true}
                    onChange={(e) => setPdmConfig({ ...pdmConfig, include_folder_path: e.target.checked })}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Include folder path</span>
                </label>

                <label className="toggle-container" style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={pdmConfig.include_configurations ?? true}
                    onChange={(e) => setPdmConfig({ ...pdmConfig, include_configurations: e.target.checked })}
                    style={{ width: 'auto', margin: 0 }}
                  />
                  <span>Include configurations</span>
                </label>
              </div>

              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label>Max Documents (optional)</label>
                <input
                  type="number"
                  value={pdmConfig.max_documents ?? ''}
                  onChange={(e) => setPdmConfig({
                    ...pdmConfig,
                    max_documents: e.target.value ? parseInt(e.target.value) : null
                  })}
                  placeholder="No limit"
                  min={1}
                  style={{ maxWidth: '150px' }}
                />
                <p className="field-help">
                  Limit the number of documents to index. Useful for testing. Leave empty to index all matching documents.
                </p>
              </div>
            </div>
          </>
        )}
      </div>
    );
  };

  const renderOdooConnection = () => {
    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Connect to an Odoo instance via Docker container or SSH.
        </p>

        {/* Connection mode tabs */}
        <div className="connection-tabs">
          <button
            type="button"
            className={`connection-tab ${odooConnectionMode === 'docker' ? 'active' : ''}`}
            onClick={() => setOdooConnectionMode('docker')}
          >
            Docker Container
          </button>
          <button
            type="button"
            className={`connection-tab ${odooConnectionMode === 'ssh' ? 'active' : ''}`}
            onClick={() => setOdooConnectionMode('ssh')}
          >
            SSH Shell
          </button>
        </div>

        {odooConnectionMode === 'docker' ? (
          <div className="connection-panel">
            <DockerConnectionPanel
              dockerContainers={dockerContainers}
              dockerNetworks={dockerNetworks}
              currentNetwork={currentNetwork}
              currentContainer={currentContainer}
              loadingDocker={loadingDocker}
              connectingNetwork={connectingNetwork}
              selectedNetwork={odooConfig.docker_network || ''}
              selectedContainer={odooConfig.container || ''}
              onDiscoverDocker={handleDiscoverDocker}
              onConnectNetwork={handleConnectNetwork}
              onNetworkChange={(network) => setOdooConfig({ ...odooConfig, docker_network: network })}
              onContainerChange={(container) => setOdooConfig({ ...odooConfig, container })}
              containerFilter={(c) => c.has_odoo}
              containerLabel={(c) => `${c.name}${c.has_odoo ? ' (Odoo)' : ''}`}
              containerCountLabel="Odoo container(s)"
              containerHelpText="The Docker container running the Odoo server."
              fallbackPlaceholder="odoo-server"
            />

            {/* Database and config path (always show for Docker mode) */}
            <div className="form-row">
              <div className="form-group">
                <label>Database Name</label>
                <input
                  type="text"
                  value={odooConfig.database || 'odoo'}
                  onChange={(e) => setOdooConfig({ ...odooConfig, database: e.target.value })}
                  placeholder="odoo"
                />
                <p className="field-help">
                  The Odoo database to connect to.
                </p>
              </div>
              <div className="form-group">
                <label>Config Path (optional)</label>
                <input
                  type="text"
                  value={odooConfig.config_path || ''}
                  onChange={(e) => setOdooConfig({ ...odooConfig, config_path: e.target.value })}
                  placeholder="Leave empty to use container defaults"
                />
                <p className="field-help">
                  Path to odoo.conf inside the container.
                </p>
              </div>
            </div>
          </div>
        ) : odooConnectionMode === 'ssh' ? (
          <div className="connection-panel">
            {/* SSH Connection Settings using reusable component */}
            <div className="form-row">
              <div className="form-group" style={{ flex: 2 }}>
                <label>SSH Host</label>
                <input
                  type="text"
                  value={odooConfig.ssh_host || ''}
                  onChange={(e) => setOdooConfig({ ...odooConfig, ssh_host: e.target.value })}
                  placeholder="odoo.example.com"
                />
                <p className="field-help">Hostname or IP address of the Odoo server.</p>
              </div>
              <div className="form-group" style={{ flex: 1 }}>
                <label>SSH Port</label>
                <input
                  type="number"
                  value={odooConfig.ssh_port || 22}
                  onChange={(e) => setOdooConfig({ ...odooConfig, ssh_port: parseInt(e.target.value) || 22 })}
                  placeholder="22"
                />
              </div>
            </div>

            <div className="form-group">
              <label>SSH User</label>
              <input
                type="text"
                value={odooConfig.ssh_user || ''}
                onChange={(e) => setOdooConfig({ ...odooConfig, ssh_user: e.target.value })}
                placeholder="root"
              />
              <p className="field-help">User to connect as via SSH.</p>
            </div>

            {/* SSH Authentication - using reusable component */}
            <SSHAuthPanel
              config={{
                host: odooConfig.ssh_host || '',
                port: odooConfig.ssh_port || 22,
                user: odooConfig.ssh_user || '',
                key_path: odooConfig.ssh_key_path,
                key_content: odooConfig.ssh_key_content,
                public_key: odooConfig.ssh_public_key,
                key_passphrase: odooConfig.ssh_key_passphrase,
                password: odooConfig.ssh_password,
              }}
              onConfigChange={(sshAuthConfig) => setOdooConfig({
                ...odooConfig,
                ssh_key_path: sshAuthConfig.key_path || '',
                ssh_key_content: sshAuthConfig.key_content || '',
                ssh_public_key: sshAuthConfig.public_key || '',
                ssh_key_passphrase: sshAuthConfig.key_passphrase || '',
                ssh_password: sshAuthConfig.password || '',
              })}
              authMode={sshKeyMode}
              onAuthModeChange={setSshKeyMode}
              generatingKey={generatingKey}
              onGenerateKey={handleGenerateSSHKey}
              keyCopied={keyCopied}
              onCopyPublicKey={handleCopyPublicKey}
              toolName={name || 'odoo'}
              showHostPort={false}
            />

            {/* Odoo-specific settings for SSH mode */}
            <div className="form-row" style={{ marginTop: '1rem' }}>
              <div className="form-group">
                <label>Run As User (optional)</label>
                <input
                  type="text"
                  value={odooConfig.run_as_user || ''}
                  onChange={(e) => setOdooConfig({ ...odooConfig, run_as_user: e.target.value })}
                  placeholder="odoo"
                />
                <p className="field-help">User to run odoo-bin as (sudo -u).</p>
              </div>
              <div className="form-group">
                <label>Database Name</label>
                <input
                  type="text"
                  value={odooConfig.database || 'odoo'}
                  onChange={(e) => setOdooConfig({ ...odooConfig, database: e.target.value })}
                  placeholder="odoo"
                />
                <p className="field-help">The Odoo database to connect to.</p>
              </div>
            </div>

            <div className="form-row">
              <div className="form-group">
                <label>Config Path (optional)</label>
                <input
                  type="text"
                  value={odooConfig.config_path || ''}
                  onChange={(e) => setOdooConfig({ ...odooConfig, config_path: e.target.value })}
                  placeholder="/etc/odoo/odoo.conf"
                />
                <p className="field-help">Path to odoo.conf on the remote server.</p>
              </div>
              <div className="form-group">
                <label>Odoo Binary Path</label>
                <input
                  type="text"
                  value={odooConfig.odoo_bin_path || ''}
                  onChange={(e) => setOdooConfig({ ...odooConfig, odoo_bin_path: e.target.value })}
                  placeholder="venv/bin/python3 src/odoo-bin"
                />
                <p className="field-help">Path to odoo-bin executable (relative to working dir).</p>
              </div>
            </div>

            <div className="form-group">
              <label>Working Directory (optional)</label>
              <input
                type="text"
                value={odooConfig.working_directory || ''}
                onChange={(e) => setOdooConfig({ ...odooConfig, working_directory: e.target.value })}
                placeholder="/var/odoo/staging-odoo.example.com"
              />
              <p className="field-help">Directory to cd into before running Odoo shell.</p>
            </div>
          </div>
        ) : null}
      </div>
    );
  };

  const renderSSHConnection = () => (
    <div className="wizard-content">
      <p className="wizard-help">
        Connect to a remote server via SSH to run shell commands.
      </p>

      {/* SSH Connection using reusable component with host/port */}
      <div className="form-row">
        <div className="form-group" style={{ flex: 2 }}>
          <label>SSH Host</label>
          <input
            type="text"
            value={sshConfig.host}
            onChange={(e) => setSshConfig({ ...sshConfig, host: e.target.value })}
            placeholder="server.example.com"
          />
        </div>
        <div className="form-group" style={{ flex: 1 }}>
          <label>Port</label>
          <input
            type="number"
            value={sshConfig.port || 22}
            onChange={(e) => setSshConfig({ ...sshConfig, port: parseInt(e.target.value) || 22 })}
            min={1}
            max={65535}
          />
        </div>
      </div>

      <div className="form-group">
        <label>SSH User</label>
        <input
          type="text"
          value={sshConfig.user}
          onChange={(e) => setSshConfig({ ...sshConfig, user: e.target.value })}
          placeholder="ubuntu"
        />
      </div>

      {/* SSH Authentication - using reusable component */}
      <SSHAuthPanel
        config={{
          host: sshConfig.host,
          port: sshConfig.port || 22,
          user: sshConfig.user,
          key_path: sshConfig.key_path,
          key_content: sshConfig.key_content,
          public_key: sshConfig.public_key,
          key_passphrase: sshConfig.key_passphrase,
          password: sshConfig.password,
        }}
        onConfigChange={(sshAuthConfig) => setSshConfig({
          ...sshConfig,
          key_path: sshAuthConfig.key_path || '',
          key_content: sshAuthConfig.key_content || '',
          public_key: sshAuthConfig.public_key || '',
          key_passphrase: sshAuthConfig.key_passphrase || '',
          password: sshAuthConfig.password || '',
        })}
        authMode={sshKeyMode}
        onAuthModeChange={setSshKeyMode}
        generatingKey={generatingKey}
        onGenerateKey={handleGenerateSSHKey}
        keyCopied={keyCopied}
        onCopyPublicKey={handleCopyPublicKey}
        toolName={name || 'ssh'}
        showHostPort={false}
      />

      <div className="form-group" style={{ marginTop: '1rem' }}>
        <label>Command Prefix (optional)</label>
        <input
          type="text"
          value={sshConfig.command_prefix || ''}
          onChange={(e) => setSshConfig({ ...sshConfig, command_prefix: e.target.value })}
          placeholder="cd /app && source venv/bin/activate && "
        />
        <p className="field-help">
          Commands to run before each tool command (e.g., activate virtualenv).
        </p>
      </div>
    </div>
  );

  const renderFilesystemConnection = () => {
    // Filter mount types based on container capabilities
    // SMB and NFS require privileged mode or CAP_SYS_ADMIN
    const availableMountTypes = (Object.keys(MOUNT_TYPE_INFO) as FilesystemMountType[]).filter((type) => {
      if (type === 'smb' || type === 'nfs') {
        // Show SMB/NFS only if container has mount capabilities
        return containerCapabilities?.can_mount ?? false;
      }
      return true; // Always show docker_volume
    });

    return (
    <div className="connection-panel">
      {/* Mount Type Selection */}
      <div className="form-group">
        <label>Mount Type</label>
        {loadingCapabilities ? (
          <div className="mount-type-tabs">
            <span className="loading-text">Checking container capabilities...</span>
          </div>
        ) : (
        <div className="mount-type-tabs">
          {availableMountTypes.map((type) => (
            <button
              key={type}
              type="button"
              className={`mount-type-tab ${filesystemConfig.mount_type === type ? 'active' : ''}`}
              onClick={() => {
                // Clear path and mount-type-specific fields when switching
                if (filesystemConfig.mount_type !== type) {
                  setFilesystemConfig({
                    ...filesystemConfig,
                    mount_type: type,
                    base_path: '',
                    // Clear other mount type fields to avoid mixing
                    smb_host: type === 'smb' ? filesystemConfig.smb_host : '',
                    smb_share: type === 'smb' ? filesystemConfig.smb_share : '',
                    smb_user: type === 'smb' ? filesystemConfig.smb_user : '',
                    smb_password: type === 'smb' ? filesystemConfig.smb_password : '',
                    smb_domain: type === 'smb' ? filesystemConfig.smb_domain : '',
                    nfs_host: type === 'nfs' ? filesystemConfig.nfs_host : '',
                    nfs_export: type === 'nfs' ? filesystemConfig.nfs_export : '',
                  });
                }
              }}
            >
              {MOUNT_TYPE_INFO[type].name}
              {MOUNT_TYPE_INFO[type].recommended && <span className="recommended-badge">Recommended</span>}
            </button>
          ))}
        </div>
        )}
        <p className="field-help">{MOUNT_TYPE_INFO[filesystemConfig.mount_type]?.description ?? 'Select a mount type'}</p>
        {/* Show message when SMB/NFS are not available */}
        {!loadingCapabilities && containerCapabilities && !containerCapabilities.can_mount && (
          <p className="field-help info-message" style={{ marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            SMB/NFS options are hidden because the container lacks mount privileges.
            To enable, uncomment <code>privileged: true</code> and <code>cap_add: SYS_ADMIN</code> in docker-compose.yml and restart.
          </p>
        )}
      </div>

      {/* Docker Volume Settings */}
      {filesystemConfig.mount_type === 'docker_volume' && (
        <FilesystemBrowser
          currentPath={filesystemConfig.base_path}
          onSelectPath={(path) => setFilesystemConfig({ ...filesystemConfig, base_path: path })}
        />
      )}

      {/* SMB Settings */}
      {filesystemConfig.mount_type === 'smb' && (
        <>
          <div className="form-group">
            <label>SMB Share</label>
            <p className="field-help" style={{ marginBottom: '0.5rem' }}>
              Enter server hostname and credentials, then click Discover to browse available shares.
            </p>
            <SMBBrowser
              host={filesystemConfig.smb_host || ''}
              user={filesystemConfig.smb_user || ''}
              password={filesystemConfig.smb_password || ''}
              domain={filesystemConfig.smb_domain || ''}
              selectedShare={filesystemConfig.smb_share || ''}
              selectedPath={filesystemConfig.base_path?.replace(/^\/mnt\/smb\/?/, '') || ''}
              onHostChange={(host) => setFilesystemConfig(prev => ({ ...prev, smb_host: host }))}
              onCredentialsChange={(user, password, domain) => setFilesystemConfig(prev => ({
                ...prev,
                smb_user: user,
                smb_password: password,
                smb_domain: domain
              }))}
              onSelectPath={(share, path) => setFilesystemConfig(prev => ({
                ...prev,
                smb_share: share,
                base_path: path ? `/${path}` : '/'
              }))}
            />
          </div>
        </>
      )}

      {/* NFS Settings */}
      {filesystemConfig.mount_type === 'nfs' && (
        <>
          <div className="form-group">
            <label>NFS Export</label>
            <p className="field-help" style={{ marginBottom: '0.5rem' }}>
              Enter NFS server hostname, then click Discover to browse available exports.
            </p>
            <NFSBrowser
              host={filesystemConfig.nfs_host || ''}
              selectedExport={filesystemConfig.nfs_export || ''}
              selectedPath={filesystemConfig.base_path?.replace(/^\/mnt\/nfs\/?/, '') || ''}
              onHostChange={(host) => setFilesystemConfig({ ...filesystemConfig, nfs_host: host })}
              onSelectPath={(exportPath, path) => setFilesystemConfig({
                ...filesystemConfig,
                nfs_export: exportPath,
                base_path: path ? `/${path}` : '/'
              })}
            />
          </div>
          <div className="form-group">
            <label>Mount Options</label>
            <input
              type="text"
              value={filesystemConfig.nfs_options || 'ro,noatime'}
              onChange={(e) => setFilesystemConfig({ ...filesystemConfig, nfs_options: e.target.value })}
              placeholder="ro,noatime"
            />
            <p className="field-help">NFS mount options (ro = read-only recommended).</p>
          </div>
        </>
      )}

      <div className="analysis-section" style={{ marginTop: '1.5rem' }}>
        <h4 style={{ marginBottom: '1rem' }}>Pre-Index Analysis</h4>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '0.75rem' }}>
          <DisabledPopover
            disabled={!fsAnalyzing && !filesystemConfig.base_path}
            message="Select a path above first"
            position="top"
          >
            <button
              type="button"
              className="btn"
              onClick={handleStartFilesystemAnalysis}
              disabled={fsAnalyzing || !filesystemConfig.base_path}
            >
              {fsAnalyzing ? 'Analyzing...' : 'Analyze Filesystem'}
            </button>
          </DisabledPopover>

          {fsAnalysisJob && fsAnalyzing && (
            <span className="analysis-progress">
              {fsAnalysisJob.status === 'scanning' && (
                <>Scanning: {fsAnalysisJob.files_scanned} files found ({fsAnalysisJob.dirs_scanned}/{fsAnalysisJob.total_dirs_to_scan} dirs)</>
              )}
              {fsAnalysisJob.status === 'analyzing' && 'Generating suggestions...'}
              {fsAnalysisJob.current_directory && (
                <span style={{ opacity: 0.7, marginLeft: '0.5rem' }}>
                  {fsAnalysisJob.current_directory}
                </span>
              )}
            </span>
          )}
        </div>

        {fsAnalysisJob?.status === 'failed' && (
          <div className="analysis-error" style={{ color: 'var(--error)', marginBottom: '1rem' }}>
            Analysis failed: {fsAnalysisJob.error_message}
          </div>
        )}

        {fsAnalysisJob?.status === 'completed' && fsAnalysisJob.result && (
          <details open={fsAnalysisExpanded} onToggle={(e) => setFsAnalysisExpanded((e.target as HTMLDetailsElement).open)}>
            <summary style={{ cursor: 'pointer', fontWeight: 500, marginBottom: '0.5rem' }}>
              Analysis Results ({fsAnalysisJob.result.total_files} files, {fsAnalysisJob.result.total_size_mb} MB, ~{fsAnalysisJob.result.estimated_chunks} chunks)
            </summary>

            <div className="analysis-results" style={{ padding: '1rem', backgroundColor: 'var(--panel-bg)', borderRadius: '4px', marginTop: '0.5rem' }}>
              {/* Warnings */}
              {fsAnalysisJob.result.warnings.length > 0 && (
                <div className="analysis-warnings" style={{ marginBottom: '1rem' }}>
                  <strong>Warnings:</strong>
                  <ul style={{ margin: '0.5rem 0 0 1.5rem', padding: 0 }}>
                    {fsAnalysisJob.result.warnings.map((warning: string, i: number) => (
                      <li key={i} style={{ marginBottom: '0.25rem', color: 'var(--warning, #e09f3e)' }}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* File Type Stats Table */}
              <div style={{ marginBottom: '1rem' }}>
                <strong>File Types:</strong>
                <table style={{ width: '100%', marginTop: '0.5rem', fontSize: '0.875rem', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border)' }}>
                      <th style={{ padding: '0.5rem' }}>Extension</th>
                      <th style={{ padding: '0.5rem', textAlign: 'right' }}>Files</th>
                      <th style={{ padding: '0.5rem', textAlign: 'right' }}>Size</th>
                      <th style={{ padding: '0.5rem', textAlign: 'right' }}>Est. Chunks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {fsAnalysisJob.result.file_type_stats.slice(0, 10).map((stat: FileTypeStats) => (
                      <tr key={stat.extension} style={{ borderBottom: '1px solid var(--border-light, #333)' }}>
                        <td style={{ padding: '0.5rem' }}>{stat.extension}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'right' }}>{stat.file_count}</td>
                        <td style={{ padding: '0.5rem', textAlign: 'right' }}>{(stat.total_size_bytes / 1024 / 1024).toFixed(2)} MB</td>
                        <td style={{ padding: '0.5rem', textAlign: 'right' }}>{stat.estimated_chunks}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {fsAnalysisJob.result.file_type_stats.length > 10 && (
                  <p style={{ fontSize: '0.85rem', opacity: 0.7, marginTop: '0.5rem' }}>
                    ... and {fsAnalysisJob.result.file_type_stats.length - 10} more file types
                  </p>
                )}
              </div>

              {/* Suggested Exclusions */}
              {fsAnalysisJob.result.suggested_exclusions.length > 0 && !fsExclusionsApplied && (
                <div
                  style={{
                    marginBottom: '16px',
                    background: 'rgba(59, 130, 246, 0.1)',
                    border: '1px solid rgba(59, 130, 246, 0.3)',
                    borderRadius: '8px',
                    padding: '12px',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <strong style={{ color: '#60a5fa' }}>Suggested Exclusions:</strong>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ fontSize: '0.8rem', padding: '4px 8px' }}
                      onClick={handleApplyFsExclusions}
                    >
                      Apply All
                    </button>
                  </div>
                  <code style={{ fontSize: '0.85rem', color: '#888' }}>
                    {fsAnalysisJob.result.suggested_exclusions.join(', ')}
                  </code>
                </div>
              )}

              {fsExclusionsApplied && (
                <div
                  style={{
                    marginBottom: '16px',
                    background: 'rgba(34, 197, 94, 0.1)',
                    border: '1px solid rgba(34, 197, 94, 0.3)',
                    borderRadius: '8px',
                    padding: '12px',
                  }}
                >
                  <span style={{ color: '#22c55e' }}>
                    Suggested exclusions applied. Re-run analysis to update estimates.
                  </span>
                </div>
              )}

              <div style={{ marginTop: '1rem', fontSize: '0.85rem', opacity: 0.7 }}>
                Analysis completed in {fsAnalysisJob.result.analysis_duration_seconds}s ({fsAnalysisJob.result.directories_scanned} directories scanned)
              </div>
            </div>
          </details>
        )}

        {/* OCR Mode and Vector Store - always visible */}
        <div className="form-row" style={{ marginTop: '1rem' }}>
          <div className="form-group" style={{ flex: 1 }}>
            <label>OCR Mode</label>
            <select
              value={filesystemConfig.ocr_mode || 'disabled'}
              onChange={(e) => setFilesystemConfig({ ...filesystemConfig, ocr_mode: e.target.value as 'disabled' | 'tesseract' | 'ollama' })}
            >
              <option value="disabled">Disabled - Skip image files</option>
              <option value="tesseract">Tesseract - Fast traditional OCR</option>
              {ollamaAvailable && (
                <option value="ollama">Ollama Vision - Semantic OCR (uses global settings)</option>
              )}
            </select>
            <p className="field-help">
              {filesystemConfig.ocr_mode === 'ollama'
                ? 'Uses global OCR vision model setting for semantic text extraction.'
                : filesystemConfig.ocr_mode === 'tesseract'
                ? 'Uses Tesseract for fast basic text extraction from images.'
                : 'Image files (PNG, JPG, etc.) will be skipped during indexing.'}
            </p>
          </div>

          <div className="form-group" style={{ flex: 1 }}>
            <label>Vector Store</label>
            <select
              value={filesystemConfig.vector_store_type || 'pgvector'}
              onChange={(e) => setFilesystemConfig({ ...filesystemConfig, vector_store_type: e.target.value as VectorStoreType })}
              disabled={isEditing}
            >
              <option value="pgvector">pgvector (PostgreSQL)</option>
              <option value="faiss">FAISS (In-memory)</option>
            </select>
            <p className="field-help">
              {filesystemConfig.vector_store_type === 'faiss'
                ? 'FAISS stores embeddings in memory with disk persistence. Faster searches but uses more RAM. Good for smaller indexes.'
                : 'pgvector stores embeddings in PostgreSQL. Persistent and scalable. Recommended for larger indexes.'}
              {isEditing && ' (Cannot change after creation)'}
            </p>
          </div>
        </div>

        {/* Advanced Indexing & Safety - appears after analysis results so "Apply All" can reveal updated exclusions */}
        <details open={fsAdvancedOpen} onToggle={(e) => setFsAdvancedOpen((e.target as HTMLDetailsElement).open)} style={{ marginTop: '1rem' }}>
          <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '0.5rem' }}>
            Advanced Indexing & Safety
          </summary>
          <div className="form-row" style={{ marginTop: '0.5rem' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>File Patterns</label>
              <input
                type="text"
                value={(filesystemConfig.file_patterns || []).join(', ')}
                onChange={(e) => setFilesystemConfig({
                  ...filesystemConfig,
                  file_patterns: e.target.value.split(',').map(p => p.trim()).filter(Boolean)
                })}
                placeholder="**/*.txt, **/*.md, **/*.pdf"
              />
              <p className="field-help">Glob patterns for files to include.</p>
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Exclude Patterns</label>
              <input
                type="text"
                value={(filesystemConfig.exclude_patterns || []).join(', ')}
                onChange={(e) => setFilesystemConfig({
                  ...filesystemConfig,
                  exclude_patterns: e.target.value.split(',').map(p => p.trim()).filter(Boolean)
                })}
                placeholder="**/node_modules/**, **/.git/**"
              />
              <p className="field-help">Glob patterns to exclude.</p>
            </div>
          </div>

          <div className="form-row" style={{ marginTop: '0.75rem' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Chunk Size</label>
              <input
                type="number"
                value={filesystemConfig.chunk_size || 1000}
                onChange={(e) => setFilesystemConfig({ ...filesystemConfig, chunk_size: parseInt(e.target.value) || 1000 })}
                min={100}
                max={4000}
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Chunk Overlap</label>
              <input
                type="number"
                value={filesystemConfig.chunk_overlap || 200}
                onChange={(e) => setFilesystemConfig({ ...filesystemConfig, chunk_overlap: parseInt(e.target.value) || 200 })}
                min={0}
                max={1000}
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>
                <input
                  type="checkbox"
                  checked={filesystemConfig.recursive !== false}
                  onChange={(e) => setFilesystemConfig({ ...filesystemConfig, recursive: e.target.checked })}
                  style={{ marginRight: '0.5rem' }}
                />
                Recursive
              </label>
            </div>
          </div>

          <div className="cloud-sync-warning" style={{ marginTop: '0.75rem' }}>
            <strong>Note:</strong> Indexing cloud-synced folders (OneDrive, Dropbox, Google Drive)
            may trigger downloads of "online-only" files. Consider indexing local copies or
            specific subfolders to avoid unwanted downloads.
          </div>

          <div className="form-row" style={{ marginTop: '0.75rem' }}>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Max File Size (MB)</label>
              <input
                type="number"
                value={filesystemConfig.max_file_size_mb || 10}
                onChange={(e) => setFilesystemConfig({ ...filesystemConfig, max_file_size_mb: parseInt(e.target.value) || 10 })}
                min={1}
                max={100}
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Max Total Files</label>
              <input
                type="number"
                value={filesystemConfig.max_total_files || 10000}
                onChange={(e) => setFilesystemConfig({ ...filesystemConfig, max_total_files: parseInt(e.target.value) || 10000 })}
                min={1}
                max={100000}
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Re-index Interval (hours)</label>
              <input
                type="number"
                value={filesystemConfig.reindex_interval_hours || 24}
                onChange={(e) => setFilesystemConfig({ ...filesystemConfig, reindex_interval_hours: parseInt(e.target.value) || 24 })}
                min={0}
                max={8760}
              />
              <p className="field-help">0 = manual only</p>
            </div>
          </div>

        </details>
      </div>
    </div>
  );
  };

  const renderConnectionConfig = () => {
    const content = (() => {
      switch (toolType) {
        case 'postgres':
          return renderPostgresConnection();
        case 'mysql':
          return renderMysqlConnection();
        case 'mssql':
          return renderMssqlConnection();
        case 'odoo_shell':
          return renderOdooConnection();
        case 'ssh_shell':
          return renderSSHConnection();
        case 'filesystem_indexer':
          return renderFilesystemConnection();
        case 'solidworks_pdm':
          return renderPdmConnection();
      }
    })();

    return (
      <>
        {content}
        {/* Only show test button for tools that need connection testing (not filesystem) */}
        {toolType !== 'filesystem_indexer' && (
          <div className="wizard-test-section">
            <button
              type="button"
              className={`btn ${testResult?.success ? 'btn-connected' : ''}`}
              onClick={handleTestConnection}
              disabled={testing || !validateConnection()}
            >
              {testing ? 'Testing...' : testResult?.success ? 'Connected' : 'Test Connection'}
            </button>
            {testResult && (
              <div className={`test-result-container ${testResult.success ? 'success' : 'error'}`}>
                <span className={`test-result ${testResult.success ? 'success' : 'error'}`}>
                  {testResult.message}
                </span>
                {!testResult.success && testResult.details !== undefined && testResult.details !== null && (
                  <details className="test-error-details" style={{ marginTop: '0.5rem' }}>
                    <summary style={{ cursor: 'pointer', fontSize: '0.85rem', color: '#666' }}>
                      Show error details
                    </summary>
                    <pre style={{
                      marginTop: '0.5rem',
                      padding: '0.75rem',
                      backgroundColor: '#1e1e1e',
                      color: '#f8f8f2',
                      borderRadius: '4px',
                      fontSize: '0.75rem',
                      overflow: 'auto',
                      maxHeight: '200px',
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word'
                    }}>
                      {typeof testResult.details === 'string'
                        ? testResult.details
                        : String(JSON.stringify(testResult.details, null, 2))}
                    </pre>
                  </details>
                )}
              </div>
            )}
          </div>
        )}
      </>
    );
  };

  const renderExecutionConstraints = () => {
    if (toolType !== 'ssh_shell') return null;

    // Check if we have enough config to connect
    const canConnect = sshConfig.host && sshConfig.user && (sshConfig.password || sshConfig.key_content || sshConfig.key_path);

    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Configure execution limits, security options, and optionally constrain the AI agent to a specific directory.
        </p>

        <div className="form-row">
          <div className="form-group">
            <label>Timeout (seconds)</label>
            <input
              type="number"
              value={timeoutValue}
              onChange={(e) => setTimeoutValue(parseInt(e.target.value) || 30)}
              min={1}
              max={300}
            />
            <p className="field-help">
              Maximum time to wait for a command to complete.
            </p>
          </div>
        </div>

        <fieldset>
          <legend>Security</legend>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={allowWrite}
              onChange={(e) => setAllowWrite(e.target.checked)}
            />
            Allow write operations
          </label>
          {allowWrite && (
            <p className="warning-text">
              Warning: Write operations are enabled. The AI will be able to modify files and run destructive commands.
            </p>
          )}
        </fieldset>

        <div className="form-group" style={{ marginTop: '1rem' }}>
           <label>Working Directory (optional)</label>
           {!canConnect ? (
              <>
                 <input
                    type="text"
                    value={sshConfig.working_directory || ''}
                    onChange={e => setSshConfig({...sshConfig, working_directory: e.target.value})}
                    placeholder="/var/www/html"
                 />
                 <p className="field-help warning" style={{color: '#f0ad4e'}}>
                    Enter directory manually. Configure authentication in the previous step to enable file browsing.
                 </p>
              </>
           ) : (
              <>
                 <SSHFilesystemBrowser
                    currentPath={sshConfig.working_directory || '/'}
                    onSelectPath={(path) => setSshConfig({...sshConfig, working_directory: path})}
                    sshConfig={sshConfig}
                 />
                 <p className="field-help">
                    Constrain the AI agent to a specific directory. It will not be able to interact with files outside this path.
                 </p>
              </>
           )}
        </div>
      </div>
    );
  };

  const renderDescription = () => (
    <div className="wizard-content">
      <p className="wizard-help">
        Give this tool a name and description. The description is provided to the AI model
        to help it understand when and how to use this tool.
      </p>

      <div className="form-group">
        <label>Name *</label>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="e.g., Production Database, Staging Odoo"
        />
        <p className="field-help">
          A short, descriptive name for this tool instance. A tool-safe name will be derived from this name.
        </p>
      </div>

      <div className="form-group">
        <label>Description for AI</label>
        <textarea
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe what data is available through this connection and when the AI should use it. For example: 'Production PostgreSQL database containing sales orders, customer data, and inventory. Use for real-time business queries.'"
          rows={4}
        />
        <p className="field-help">
          This description is included in the system prompt to help the AI understand
          what this tool is for and when to use it. Be specific about the data available.
        </p>
      </div>
    </div>
  );

  const renderOptions = () => {
    // SQL-based tools need Max Results for query limiting
    const showMaxResults = ['postgres', 'mysql', 'mssql', 'solidworks_pdm'].includes(toolType);

    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Configure limits and security options for this tool.
        </p>

        <div className="form-row">
          {showMaxResults && (
            <div className="form-group">
              <label>Max Results</label>
              <input
                type="number"
                value={maxResults}
                onChange={(e) => setMaxResults(parseInt(e.target.value) || 100)}
                min={1}
                max={1000}
              />
              <p className="field-help">
                Maximum number of rows/results returned per query.
              </p>
            </div>
          )}

          <div className="form-group">
            <label>Timeout (seconds)</label>
            <input
              type="number"
              value={timeoutValue}
              onChange={(e) => setTimeoutValue(parseInt(e.target.value) || 30)}
              min={1}
              max={300}
            />
            <p className="field-help">
              Maximum time to wait for a query to complete.
            </p>
          </div>
        </div>

        <fieldset>
          <legend>Security</legend>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={allowWrite}
              onChange={(e) => setAllowWrite(e.target.checked)}
            />
            Allow write operations (INSERT/UPDATE/DELETE)
          </label>
          {allowWrite && (
            <p className="warning-text">
              Warning: Write operations are enabled. The AI will be able to modify data.
            </p>
          )}
        </fieldset>
      </div>
    );
  };

  const renderReview = () => {
    const typeInfo = TOOL_TYPE_INFO[toolType];

    return (
      <div className="wizard-content">
        <p className="wizard-help">
          Review your tool configuration before saving.
        </p>

        <div className="review-section">
          <h4>Tool Type</h4>
          <p>
            <span className="review-icon">
              <Icon name={getToolIconType(typeInfo?.icon)} size={20} />
            </span>
            {typeInfo?.name}
          </p>
        </div>

        <div className="review-section">
          <h4>Name</h4>
          <p>{name}</p>
        </div>

        {description && (
          <div className="review-section">
            <h4>Description</h4>
            <p className="review-description">{description}</p>
          </div>
        )}

        <div className="review-section">
          <h4>Connection</h4>
          <pre className="review-config">
            {JSON.stringify(getConnectionConfig(), null, 2)}
          </pre>
        </div>

        <div className="review-section">
          <h4>Options</h4>
          <ul>
            <li>Max results: {maxResults}</li>
            <li>Timeout: {timeoutValue}s</li>
            <li>Write operations: {allowWrite ? 'Enabled' : 'Disabled'}</li>
          </ul>
        </div>
      </div>
    );
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 'type':
        return renderTypeSelection();
      case 'connection':
        return renderConnectionConfig();
      case 'pdm_filtering':
        return renderPdmFiltering();
      case 'execution_constraints':
        return renderExecutionConstraints();
      case 'description':
        return renderDescription();
      case 'options':
        return renderOptions();
      case 'review':
        return renderReview();
    }
  };

  return (
    <div className={`card wizard-card ${embedded ? 'embedded' : ''}`}>
      {!embedded && (
        <div className="wizard-header">
          <h2>{isEditing ? 'Edit Tool' : 'Add Tool'}</h2>
          <button type="button" className="close-btn" onClick={onClose}>
            <Icon name="close" size={18} />
          </button>
        </div>
      )}

      {/* Progress indicator */}
      <div className="wizard-progress" ref={progressRef}>
        {wizardSteps.map((step, index) => {
          if (isEditing && step === 'type') return null;
          const stepIndex = wizardSteps.indexOf(step);
          const isNavigable = canNavigateToStep(stepIndex);
          return (
            <button
              key={step}
              type="button"
              className={`wizard-step ${currentStep === step ? 'active' : ''} ${
                getCurrentStepIndex() > index ? 'completed' : ''
              } ${isNavigable ? 'navigable' : ''}`}
              onClick={() => goToStep(step)}
              disabled={!isNavigable}
            >
              <span className="step-number">{isEditing ? index : index + 1}</span>
              <span className="step-title">{getStepTitle(step)}</span>
            </button>
          );
        })}
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="wizard-body">{renderStepContent()}</div>

      <div className="wizard-footer">
        <button
          type="button"
          className="btn btn-secondary"
          onClick={getCurrentStepIndex() === 0 || (isEditing && currentStep === 'connection') ? onClose : goToPreviousStep}
        >
          {getCurrentStepIndex() === 0 || (isEditing && currentStep === 'connection') ? 'Cancel' : 'Back'}
        </button>

        {currentStep === 'review' ? (
          <button
            type="button"
            className="btn"
            onClick={handleSave}
            disabled={saving}
          >
            {saving ? 'Saving...' : isEditing ? 'Save Changes' : 'Create Tool'}
          </button>
        ) : (
          <button
            type="button"
            className="btn"
            onClick={goToNextStep}
            disabled={!canProceed()}
          >
            Continue
          </button>
        )}
      </div>
    </div>
  );
}
