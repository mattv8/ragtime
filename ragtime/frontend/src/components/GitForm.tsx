import { useState, useCallback, useEffect } from 'react';
import { api } from '@/api';
import type { IndexJob } from '@/types';

interface GitFormProps {
  onJobCreated: () => void;
}

type StatusType = 'info' | 'success' | 'error' | null;

export function GitForm({ onJobCreated }: GitFormProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: null,
    message: '',
  });

  // Git form state
  const [gitUrl, setGitUrl] = useState('');
  const [gitToken, setGitToken] = useState('');
  const [isPrivateRepo, setIsPrivateRepo] = useState(false);
  const [branches, setBranches] = useState<string[]>([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);

  /** Parse GitHub/GitLab URL to extract host, owner and repo */
  const parseGitUrl = useCallback((url: string): { host: 'github' | 'gitlab' | null; owner: string; repo: string } | null => {
    // GitHub HTTPS: https://github.com/owner/repo.git or https://github.com/owner/repo
    const githubHttpsMatch = url.match(/github\.com\/([^/]+)\/([^/.]+)/);
    if (githubHttpsMatch) {
      return { host: 'github', owner: githubHttpsMatch[1], repo: githubHttpsMatch[2] };
    }
    // GitHub SSH: git@github.com:owner/repo.git
    const githubSshMatch = url.match(/git@github\.com:([^/]+)\/([^/.]+)/);
    if (githubSshMatch) {
      return { host: 'github', owner: githubSshMatch[1], repo: githubSshMatch[2] };
    }
    // GitLab HTTPS: https://gitlab.com/owner/repo.git or https://gitlab.com/owner/repo
    const gitlabHttpsMatch = url.match(/gitlab\.com\/([^/]+)\/([^/.]+)/);
    if (gitlabHttpsMatch) {
      return { host: 'gitlab', owner: gitlabHttpsMatch[1], repo: gitlabHttpsMatch[2] };
    }
    // GitLab SSH: git@gitlab.com:owner/repo.git
    const gitlabSshMatch = url.match(/git@gitlab\.com:([^/]+)\/([^/.]+)/);
    if (gitlabSshMatch) {
      return { host: 'gitlab', owner: gitlabSshMatch[1], repo: gitlabSshMatch[2] };
    }
    return null;
  }, []);

  /** Fetch branches from GitHub or GitLab API */
  const fetchBranches = useCallback(async (url: string, token?: string) => {
    const parsed = parseGitUrl(url);
    if (!parsed) {
      setBranches([]);
      setBranchError(null);
      return;
    }

    setLoadingBranches(true);
    setBranchError(null);

    try {
      let apiUrl: string;
      const headers: HeadersInit = {};

      if (parsed.host === 'github') {
        apiUrl = `https://api.github.com/repos/${parsed.owner}/${parsed.repo}/branches?per_page=100`;
        headers.Accept = 'application/vnd.github.v3+json';
        if (token) {
          headers.Authorization = `token ${token}`;
        }
      } else {
        // GitLab API
        const projectPath = encodeURIComponent(`${parsed.owner}/${parsed.repo}`);
        apiUrl = `https://gitlab.com/api/v4/projects/${projectPath}/repository/branches?per_page=100`;
        if (token) {
          headers['PRIVATE-TOKEN'] = token;
        }
      }

      const response = await fetch(apiUrl, { headers });

      if (!response.ok) {
        if (response.status === 404) {
          setBranchError(token ? 'Repository not found or token lacks access' : 'Repository not found or is private');
        } else if (response.status === 401) {
          setBranchError('Invalid or expired token');
        } else {
          setBranchError(`API error: ${response.status}`);
        }
        setBranches([]);
        return;
      }

      const data = await response.json();
      const branchNames = data.map((b: { name: string }) => b.name);
      setBranches(branchNames);

      // Auto-select default branch if available
      if (branchNames.length > 0 && !selectedBranch) {
        const defaultBranch = branchNames.includes('main') ? 'main' : branchNames.includes('master') ? 'master' : branchNames[0];
        setSelectedBranch(defaultBranch);
      }
    } catch {
      setBranchError('Failed to fetch branches');
      setBranches([]);
    } finally {
      setLoadingBranches(false);
    }
  }, [parseGitUrl, selectedBranch]);

  // Fetch branches when URL changes (for public repos) or when token is provided (for private repos)
  useEffect(() => {
    if (!gitUrl) {
      setBranches([]);
      setSelectedBranch('');
      setBranchError(null);
      return;
    }

    // Debounce the fetch
    const timer = setTimeout(() => {
      if (isPrivateRepo) {
        // For private repos, only fetch if token is provided
        if (gitToken && gitToken.length >= 10) {
          fetchBranches(gitUrl, gitToken);
        } else {
          setBranches([]);
          setBranchError(null);
        }
      } else {
        // For public repos, fetch without token
        fetchBranches(gitUrl);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [gitUrl, gitToken, isPrivateRepo, fetchBranches]);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;

    // Derive index name from repo URL
    const parsed = parseGitUrl(gitUrl);
    if (!parsed) {
      setStatus({ type: 'error', message: 'Invalid Git URL format' });
      return;
    }
    const name = parsed.repo.toLowerCase().replace(/[^a-z0-9_-]/g, '-');
    const filePatterns = (form.elements.namedItem('file_patterns') as HTMLInputElement).value || '**/*';
    const excludePatterns = (form.elements.namedItem('exclude_patterns') as HTMLInputElement).value || '';

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Starting git clone...' });

    try {
      const job: IndexJob = await api.indexFromGit({
        name,
        git_url: gitUrl,
        git_branch: selectedBranch || 'main',
        git_token: isPrivateRepo ? gitToken : undefined,
        config: {
          name,
          description: '',  // Auto-generated during indexing
          file_patterns: filePatterns.split(',').map((s) => s.trim()),
          exclude_patterns: excludePatterns.split(',').map((s) => s.trim()),
        },
      });
      setStatus({ type: 'success', message: `Job started - ID: ${job.id} - Status: ${job.status}` });
      form.reset();
      // Reset git form state
      setGitUrl('');
      setGitToken('');
      setIsPrivateRepo(false);
      setBranches([]);
      setSelectedBranch('');
      setBranchError(null);
      onJobCreated();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Request failed'}` });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="row">
        <div className="form-group" style={{ flex: 2 }}>
          <label>Git URL *</label>
          <input
            type="text"
            value={gitUrl}
            onChange={(e) => setGitUrl(e.target.value)}
            placeholder="https://github.com/user/repo.git"
            required
          />
        </div>
        <div className="form-group" style={{ flex: 1 }}>
          <label>
            Branch
            {loadingBranches && <span style={{ marginLeft: '0.5rem', color: '#888', fontSize: '0.85em' }}>(loading...)</span>}
          </label>
          {branches.length > 0 ? (
            <select
              value={selectedBranch}
              onChange={(e) => setSelectedBranch(e.target.value)}
              style={{ width: '100%' }}
            >
              {branches.map((branch) => (
                <option key={branch} value={branch}>{branch}</option>
              ))}
            </select>
          ) : (
            <input
              type="text"
              value={selectedBranch}
              onChange={(e) => setSelectedBranch(e.target.value)}
              placeholder={branchError ? 'Enter branch name' : 'main'}
            />
          )}
          {branchError && (
            <small style={{ color: '#f87171', fontSize: '0.85em', display: 'block', marginTop: '0.25rem' }}>
              {branchError}
            </small>
          )}
        </div>
      </div>

      <p className="field-help" style={{ marginBottom: '16px' }}>
        Index name will be derived from the repository name. A description will be auto-generated from the indexed content. You can edit both later.
      </p>

      <div className="form-group" style={{ marginBottom: isPrivateRepo ? '0.5rem' : undefined }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={isPrivateRepo}
            onChange={(e) => {
              setIsPrivateRepo(e.target.checked);
              if (!e.target.checked) {
                setGitToken('');
                // Re-fetch branches without token for public repos
                if (gitUrl) {
                  fetchBranches(gitUrl);
                }
              }
            }}
            style={{ width: 'auto', margin: 0 }}
          />
          Private repository (requires authentication)
        </label>
      </div>

      {isPrivateRepo && (
        <div className="form-group" style={{ marginLeft: '1.5rem', borderLeft: '2px solid #444', paddingLeft: '1rem', marginBottom: '1rem' }}>
          <label>GitHub Personal Access Token *</label>
          <input
            type="password"
            value={gitToken}
            onChange={(e) => setGitToken(e.target.value)}
            placeholder="ghp_xxxx..."
            autoComplete="off"
            required={isPrivateRepo}
          />
          <small style={{ color: '#888', fontSize: '0.85em', display: 'block', marginTop: '0.25rem' }}>
            <a
              href="https://github.com/settings/tokens/new"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: '#60a5fa' }}
            >
              Generate a new token
            </a>
            {' '}with "repo" scope for private repository access. Token is used once for cloning and never stored.
          </small>
        </div>
      )}

      <div className="row">
        <div className="form-group">
          <label>File Patterns (comma-separated)</label>
          <input
            type="text"
            name="file_patterns"
            defaultValue="**/*"
            placeholder="e.g., **/*.py,**/*.md,**/*.xml"
          />
        </div>
        <div className="form-group">
          <label>Exclude Patterns</label>
          <input
            type="text"
            name="exclude_patterns"
            defaultValue=""
            placeholder="e.g., **/test/**,**/__pycache__/**"
          />
        </div>
      </div>

      <button type="submit" className="btn" disabled={isLoading}>
        {isLoading ? 'Cloning...' : 'Clone & Index'}
      </button>

      {status.type && (
        <div className={`status-message ${status.type}`}>{status.message}</div>
      )}
    </form>
  );
}
