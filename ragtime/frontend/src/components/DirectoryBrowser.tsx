import { useEffect, useState } from 'react';

import type { DirectoryEntry } from '@/types';
import { Icon } from './Icon';

export interface DirectoryBrowserProps {
  currentPath: string;
  entries: DirectoryEntry[];
  loading: boolean;
  error?: string | null;
  onNavigate: (path: string) => void;
  onGoUp: () => void;
  onSelect: (path: string) => void;
}

export function DirectoryBrowser({ currentPath, entries, loading, error, onNavigate, onGoUp, onSelect }: DirectoryBrowserProps) {
  const [pathInput, setPathInput] = useState('');

  useEffect(() => {
    setPathInput('');
  }, [currentPath]);

  const filterText = pathInput.split('/')[0];
  const filteredEntries = entries.filter((entry) => {
    if (!filterText) return true;
    return entry.name.toLowerCase().startsWith(filterText.toLowerCase());
  });

  const handleFilterChange = (value: string) => {
    if (value.endsWith('/')) {
      const dirName = value.slice(0, -1);
      const matchingDir = entries.find((entry) => entry.is_dir && entry.name.toLowerCase() === dirName.toLowerCase());
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
      const matchingDir = entries.find((entry) => entry.is_dir && entry.name.toLowerCase() === first.toLowerCase());
      if (matchingDir) {
        onNavigate(matchingDir.path);
        setPathInput(rest);
        return;
      }
    }

    setPathInput(value);
  };

  const handlePathInputKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      const firstDir = filteredEntries.find((entry) => entry.is_dir);
      if (firstDir) {
        onNavigate(firstDir.path);
        setPathInput('');
      }
    }
  };

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
            {segments.map((segment, index) => {
              const pathToSegment = segments.slice(0, index + 1).join('/');
              const isLast = index === segments.length - 1;
              return (
                <span key={index} className="breadcrumb-segment">
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
            onChange={(event) => handleFilterChange(event.target.value)}
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
          {filteredEntries.filter((entry) => entry.is_dir).map((entry) => (
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
          {filteredEntries.filter((entry) => !entry.is_dir).slice(0, 3).map((entry) => (
            <div key={entry.path} className="browser-entry file">
              <span className="entry-icon"><Icon name="file" size={16} /></span>
              <span className="entry-name">{entry.name}</span>
            </div>
          ))}
          {filteredEntries.filter((entry) => !entry.is_dir).length > 3 && (
            <div className="browser-more">
              +{filteredEntries.filter((entry) => !entry.is_dir).length - 3} more files
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
