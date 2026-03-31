import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import type { BrowseResponse, DirectoryEntry } from '@/types';

import { Icon } from './Icon';

interface ConstrainedPathBrowserProps {
  currentPath: string;
  rootPath: string;
  onSelectPath: (path: string) => void;
  onBrowsePath: (path: string) => Promise<BrowseResponse>;
  rootLabel?: string;
  emptyMessage?: string;
  defaultExpanded?: boolean;
  cacheKey?: string;
  showFiles?: boolean;
  stagedDirectories?: string[];
  onStageDirectory?: (path: string) => void;
  canSelectPath?: (path: string) => boolean;
  cannotSelectPathMessage?: string;
}

interface BrowserDirectoryEntry extends DirectoryEntry {
  isPending?: boolean;
}

function normalizeBrowserPath(value: string): string {
  const normalizedParts: string[] = [];
  for (const part of (value || '/').replace(/\\/g, '/').split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      normalizedParts.pop();
      continue;
    }
    normalizedParts.push(part);
  }
  return '/' + normalizedParts.join('/');
}

function getImmediateChildName(parentPath: string, childPath: string): string | null {
  const normalizedParent = normalizeBrowserPath(parentPath || '/');
  const normalizedChild = normalizeBrowserPath(childPath || '/');
  if (normalizedParent === normalizedChild) {
    return null;
  }
  const prefix = normalizedParent === '/' ? '/' : `${normalizedParent}/`;
  if (!normalizedChild.startsWith(prefix)) {
    return null;
  }
  const remainder = normalizedChild.slice(prefix.length);
  if (!remainder || remainder.includes('/')) {
    return null;
  }
  return remainder;
}

export function ConstrainedPathBrowser({
  currentPath,
  rootPath,
  onSelectPath,
  onBrowsePath,
  rootLabel,
  emptyMessage = 'Empty directory',
  defaultExpanded,
  cacheKey,
  showFiles = true,
  stagedDirectories = [],
  onStageDirectory,
  canSelectPath,
  cannotSelectPathMessage,
}: ConstrainedPathBrowserProps) {
  const normalizedRootPath = normalizeBrowserPath(rootPath || '/');
  const clampToRootPath = useCallback((value: string): string => {
    const normalized = normalizeBrowserPath(value || normalizedRootPath || '/');
    if (normalizedRootPath === '/') {
      return normalized;
    }
    if (normalized === normalizedRootPath || normalized.startsWith(`${normalizedRootPath}/`)) {
      return normalized;
    }
    return normalizedRootPath;
  }, [normalizedRootPath]);

  const [entries, setEntries] = useState<DirectoryEntry[]>([]);
  const [browsePath, setBrowsePath] = useState<string>(
    clampToRootPath(currentPath || normalizedRootPath || '/')
  );
  const [pathInput, setPathInput] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pendingDirectoryError, setPendingDirectoryError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(
    defaultExpanded ?? (!currentPath || normalizeBrowserPath(currentPath) === normalizedRootPath)
  );
  const onBrowsePathRef = useRef(onBrowsePath);
  const cacheRef = useRef(new Map<string, BrowseResponse>());
  const requestIdRef = useRef(0);

  useEffect(() => {
    onBrowsePathRef.current = onBrowsePath;
  }, [onBrowsePath]);

  useEffect(() => {
    cacheRef.current.clear();
  }, [cacheKey, normalizedRootPath]);

  useEffect(() => {
    setBrowsePath(clampToRootPath(currentPath || normalizedRootPath || '/'));
  }, [clampToRootPath, currentPath, normalizedRootPath]);

  const normalizedStagedDirectories = useMemo(() => {
    const nextPaths: string[] = [];
    const seen = new Set<string>();

    for (const rawPath of stagedDirectories) {
      const normalizedPath = normalizeBrowserPath(rawPath || '/');
      if (clampToRootPath(normalizedPath) !== normalizedPath) {
        continue;
      }
      if (normalizedPath === normalizedRootPath || seen.has(normalizedPath)) {
        continue;
      }
      seen.add(normalizedPath);
      nextPaths.push(normalizedPath);
    }

    return nextPaths.sort((left, right) => left.localeCompare(right));
  }, [clampToRootPath, normalizedRootPath, stagedDirectories]);

  const isVirtualBrowsePath = normalizedStagedDirectories.includes(browsePath);

  const clearPendingDirectoryError = useCallback(() => {
    setPendingDirectoryError(null);
  }, []);

  useEffect(() => {
    if (!isExpanded) {
      clearPendingDirectoryError();
    }
  }, [clearPendingDirectoryError, isExpanded]);

  const browseCurrent = useCallback(async () => {
    const nextPath = clampToRootPath(browsePath || normalizedRootPath || '/');
    if (!nextPath) return;

    if (isVirtualBrowsePath) {
      setLoading(false);
      setError(null);
      setEntries([]);
      return;
    }

    const cached = cacheRef.current.get(nextPath);
    if (cached) {
      setError(cached.error ?? null);
      setEntries(cached.entries || []);
      return;
    }

    const requestId = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      const result = await onBrowsePathRef.current(nextPath);
      if (requestId !== requestIdRef.current) {
        return;
      }
      cacheRef.current.set(nextPath, result);
      setError(result.error ?? null);
      setEntries(result.error ? [] : (result.entries || []));
    } catch (err) {
      if (requestId !== requestIdRef.current) {
        return;
      }
      setError(err instanceof Error ? err.message : 'Browse failed');
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [browsePath, clampToRootPath, isVirtualBrowsePath, normalizedRootPath]);

  useEffect(() => {
    if (!isExpanded) return;
    void browseCurrent();
  }, [browseCurrent, cacheKey, isExpanded]);

  const handleNavigate = (path: string) => {
    setBrowsePath(clampToRootPath(path || normalizedRootPath || '/'));
    setPathInput('');
  };

  const handleGoUp = () => {
    if (!browsePath || browsePath === '/' || browsePath === normalizedRootPath) return;
    const parts = browsePath.split('/').filter(Boolean);
    parts.pop();
    const nextPath = '/' + parts.join('/');
    setBrowsePath(clampToRootPath(nextPath || '/'));
    setPathInput('');
  };

  const handleSelect = (path: string) => {
    if (canSelectPath && !canSelectPath(path)) {
      return;
    }
    onSelectPath(path);
    setIsExpanded(false);
  };

  const isSelectableBrowsePath = canSelectPath ? canSelectPath(browsePath) : true;

  const stagedDirectoryEntries = useMemo<BrowserDirectoryEntry[]>(() => {
    if (normalizedStagedDirectories.length === 0) {
      return [];
    }

    const existingPaths = new Set(
      entries
        .filter((entry) => entry.is_dir)
        .map((entry) => normalizeBrowserPath(entry.path))
    );

    const nextEntries: BrowserDirectoryEntry[] = [];
    for (const stagedPath of normalizedStagedDirectories) {
      const childName = getImmediateChildName(browsePath, stagedPath);
      if (!childName || existingPaths.has(stagedPath)) {
        continue;
      }
      nextEntries.push({
        name: childName,
        path: stagedPath,
        is_dir: true,
        isPending: true,
      });
    }

    return nextEntries.sort((left, right) => left.name.localeCompare(right.name));
  }, [browsePath, entries, normalizedStagedDirectories]);

  const visibleDirectoryEntries = useMemo<BrowserDirectoryEntry[]>(() => {
    const nextEntries: BrowserDirectoryEntry[] = [
      ...entries.filter((entry) => entry.is_dir).map((entry) => ({ ...entry, isPending: false })),
      ...stagedDirectoryEntries,
    ];
    nextEntries.sort((left, right) => {
      if (!!left.isPending !== !!right.isPending) {
        return left.isPending ? -1 : 1;
      }
      return left.name.localeCompare(right.name);
    });
    return nextEntries;
  }, [entries, stagedDirectoryEntries]);

  const visibleFileEntries = useMemo(
    () => entries.filter((entry) => !entry.is_dir),
    [entries]
  );

  const filterText = pathInput.split('/')[0];
  const filteredDirectoryEntries = visibleDirectoryEntries.filter((entry) => {
    if (!filterText) return true;
    return entry.name.toLowerCase().startsWith(filterText.toLowerCase());
  });
  const filteredFileEntries = visibleFileEntries.filter((entry) => {
    if (!filterText) return true;
    return entry.name.toLowerCase().startsWith(filterText.toLowerCase());
  });

  const handleFilterChange = (value: string) => {
    if (pendingDirectoryError) {
      setPendingDirectoryError(null);
    }
    if (value.endsWith('/')) {
      const dirName = value.slice(0, -1);
      const matchingDir = entries.find(
        (entry) => entry.is_dir && entry.name.toLowerCase() === dirName.toLowerCase()
      );
      if (matchingDir) {
        handleNavigate(matchingDir.path);
        return;
      }
    }
    if (value.includes('/')) {
      const firstSegment = value.split('/')[0];
      const matchingDir = entries.find(
        (entry) => entry.is_dir && entry.name.toLowerCase() === firstSegment.toLowerCase()
      );
      if (matchingDir) {
        handleNavigate(matchingDir.path);
        return;
      }
    }
    setPathInput(value);
  };

  const handlePathInputKeyDown = (event: React.KeyboardEvent) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    const firstDir = filteredDirectoryEntries[0];
    if (firstDir) {
      handleNavigate(firstDir.path);
    }
  };

  const handleCreateDirectory = () => {
    if (!onStageDirectory) {
      return;
    }

    const normalizedName = pathInput.trim().replace(/\\/g, '/');
    if (!normalizedName) {
      setPendingDirectoryError('Type a folder name in the filter box first');
      return;
    }
    if (normalizedName === '.' || normalizedName === '..' || normalizedName.includes('/')) {
      setPendingDirectoryError('Use a single folder name');
      return;
    }

    const nextPath = normalizeBrowserPath(
      browsePath === '/' ? `/${normalizedName}` : `${browsePath}/${normalizedName}`
    );
    if (clampToRootPath(nextPath) !== nextPath) {
      setPendingDirectoryError('Folder must stay within the current root');
      return;
    }

    const alreadyExists = visibleDirectoryEntries.some((entry) => normalizeBrowserPath(entry.path) === nextPath)
      || visibleFileEntries.some((entry) => normalizeBrowserPath(entry.path) === nextPath);
    if (alreadyExists) {
      setPendingDirectoryError('That folder already exists');
      return;
    }

    onStageDirectory(nextPath);
    onSelectPath(nextPath);
    setBrowsePath(nextPath);
    setPathInput('');
    setPendingDirectoryError(null);
  };

  const isRestrictedToRoot = normalizedRootPath !== '/';
  const relativeBrowsePath = isRestrictedToRoot
    ? browsePath.slice(normalizedRootPath.length)
    : browsePath;
  const segments = relativeBrowsePath.split('/').filter(Boolean);
  const breadcrumbRootLabel = rootLabel || normalizedRootPath || '/';
  const selectedOrBrowsePath = currentPath || browsePath;
  const displayPath = selectedOrBrowsePath === normalizedRootPath && rootLabel
    ? rootLabel
    : selectedOrBrowsePath;

  return (
    <div className="filesystem-browser">
      <div className="mounts-accordion">
        <div className={`mount-item ${isExpanded ? 'expanded' : ''}`}>
          <button
            type="button"
            className="mount-header"
            onClick={() => {
              if (!isExpanded) {
                setBrowsePath(clampToRootPath(currentPath || normalizedRootPath || '/'));
              }
              setIsExpanded(!isExpanded);
            }}
          >
            <span className="mount-icon">{isExpanded ? '▼' : '▶'}</span>
            <span className="mount-path">{displayPath}</span>
            {currentPath && <span className="current-badge">Selected</span>}
          </button>

          {isExpanded && (
            <div className="mount-browser">
              <div className="browser-header">
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={handleGoUp}
                  disabled={browsePath === '/' || browsePath === normalizedRootPath || loading}
                >
                  ..
                </button>
                <div className="browser-path-wrapper">
                  <span className="browser-path-breadcrumbs">
                    <span className="breadcrumb-segment">
                      <button
                        type="button"
                        className="breadcrumb-btn"
                        onClick={() => handleNavigate(normalizedRootPath || '/')}
                      >
                        {breadcrumbRootLabel}
                      </button>
                    </span>
                    {segments.length > 0 && !breadcrumbRootLabel.endsWith('/') && <span className="breadcrumb-sep">/</span>}
                    {segments.map((segment, idx) => {
                      const pathToSegment = isRestrictedToRoot
                        ? `${normalizedRootPath}/${segments.slice(0, idx + 1).join('/')}`
                        : '/' + segments.slice(0, idx + 1).join('/');
                      const isLast = idx === segments.length - 1;
                      return (
                        <span key={`${segment}-${idx}`} className="breadcrumb-segment">
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
                </div>
              </div>
              <div className="browser-toolbar">
                <input
                  type="text"
                  className="browser-filter-input"
                  value={pathInput}
                  onChange={(event) => handleFilterChange(event.target.value)}
                  onKeyDown={handlePathInputKeyDown}
                  placeholder="Filter or Create..."
                />
                <div className="browser-actions">
                  {onStageDirectory && (
                    <button
                      type="button"
                      className="btn btn-sm"
                      onClick={handleCreateDirectory}
                      disabled={loading || !!error}
                    >
                      New folder
                    </button>
                  )}
                  <button
                    type="button"
                    className="btn btn-sm btn-primary"
                    onClick={() => handleSelect(browsePath)}
                    disabled={loading || !isSelectableBrowsePath}
                    title={!isSelectableBrowsePath ? cannotSelectPathMessage : undefined}
                  >
                    Select
                  </button>
                </div>
              </div>

              {error && <div className="browser-error">{error}</div>}
              {pendingDirectoryError && <div className="browser-error">{pendingDirectoryError}</div>}

              {loading ? (
                <div className="browser-loading">Loading...</div>
              ) : (
                <div className="browser-entries">
                  {filteredDirectoryEntries.map((entry) => (
                    <button
                      key={entry.path}
                      type="button"
                      className={`browser-entry${entry.isPending ? ' pending' : ''}`}
                      onClick={() => handleNavigate(entry.path)}
                    >
                      <span className="entry-icon"><Icon name="folder" size={16} /></span>
                      <span className="entry-name">{entry.name}</span>
                      {entry.isPending && <span className="browser-entry-badge">New</span>}
                    </button>
                  ))}
                  {showFiles && filteredFileEntries.slice(0, 3).map((entry) => (
                    <div key={entry.path} className="browser-entry file">
                      <span className="entry-icon"><Icon name="file" size={16} /></span>
                      <span className="entry-name">{entry.name}</span>
                    </div>
                  ))}
                  {showFiles && filteredFileEntries.length > 3 && (
                    <div className="browser-more">
                      +{filteredFileEntries.length - 3} more files
                    </div>
                  )}
                  {filteredDirectoryEntries.length === 0 && (!showFiles || filteredFileEntries.length === 0) && !loading && (
                    <div className="browser-empty">{emptyMessage}</div>
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