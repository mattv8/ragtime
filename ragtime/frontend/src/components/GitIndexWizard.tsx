import { useCallback, useEffect, useState } from 'react';
import { api } from '@/api';
import type { IndexAnalysisResult, IndexJob } from '@/types';

type StatusType = 'info' | 'success' | 'error' | null;
type WizardStep = 'input' | 'analyzing' | 'review' | 'indexing';

interface GitIndexWizardProps {
  onJobCreated?: () => void;
  onCancel?: () => void;
  onAnalysisStart?: () => void;
  onAnalysisComplete?: () => void;
}

// Default file patterns to include all files, and placeholder hints for UI
const DEFAULT_FILE_PATTERNS = '**/*';
const PLACEHOLDER_FILE_PATTERNS = 'e.g. **/*.py, **/*.md (default: all files)';
const PLACEHOLDER_EXCLUDE_PATTERNS = 'e.g. **/node_modules/**, **/__pycache__/**';
const GITHUB_TOKEN_PREFIXES = ['ghp_', 'gho_', 'ghu_', 'ghs_', 'ghr_', 'github_pat_'];
const GITLAB_TOKEN_PREFIXES = ['glpat-', 'glptt-', 'gldt-', 'glsoat-'];

export function GitIndexWizard({ onJobCreated, onCancel, onAnalysisStart, onAnalysisComplete }: GitIndexWizardProps) {
  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: null,
    message: '',
  });
  const [wizardStep, setWizardStep] = useState<WizardStep>('input');
  const [analysisResult, setAnalysisResult] = useState<IndexAnalysisResult | null>(null);

  const [gitUrl, setGitUrl] = useState('');
  const [gitToken, setGitToken] = useState('');
  const [isPrivateRepo, setIsPrivateRepo] = useState(false);
  const [branches, setBranches] = useState<string[]>([]);
  const [selectedBranch, setSelectedBranch] = useState('');
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);

  const [filePatterns, setFilePatterns] = useState(DEFAULT_FILE_PATTERNS);
  const [excludePatterns, setExcludePatterns] = useState('');
  const [chunkSize, setChunkSize] = useState(1000);
  const [chunkOverlap, setChunkOverlap] = useState(200);
  const [maxFileSizeKb, setMaxFileSizeKb] = useState(500);
  const [exclusionsApplied, setExclusionsApplied] = useState(false);
  const [patternsExpanded, setPatternsExpanded] = useState(false);

  const resetState = useCallback(() => {
    setIsLoading(false);
    setStatus({ type: null, message: '' });
    setWizardStep('input');
    setAnalysisResult(null);
    setGitUrl('');
    setGitToken('');
    setIsPrivateRepo(false);
    setBranches([]);
    setSelectedBranch('');
    setBranchError(null);
    setFilePatterns(DEFAULT_FILE_PATTERNS);
    setExcludePatterns('');
    setChunkSize(1000);
    setChunkOverlap(200);
    setMaxFileSizeKb(500);
    setExclusionsApplied(false);
    setPatternsExpanded(false);
  }, []);

  const detectProviderFromToken = useCallback((token: string): 'github' | 'gitlab' | null => {
    if (GITHUB_TOKEN_PREFIXES.some((prefix) => token.startsWith(prefix))) {
      return 'github';
    }
    if (GITLAB_TOKEN_PREFIXES.some((prefix) => token.startsWith(prefix))) {
      return 'gitlab';
    }
    return null;
  }, []);

  const parseGitUrl = useCallback(
    (
      url: string,
      token?: string,
    ): { host: 'github' | 'gitlab' | 'generic'; hostUrl: string; owner: string; repo: string } | null => {
      if (!url || typeof url !== 'string') {
        return null;
      }

      const httpsMatch = url.match(/^https?:\/\/([^\/]+)\/([^\/]+)\/([^\/]+?)(\.git)?$/);
      if (httpsMatch) {
        const [, hostUrl, owner, repo] = httpsMatch;
        let host: 'github' | 'gitlab' | 'generic' = 'generic';
        if (hostUrl === 'github.com') {
          host = 'github';
        } else if (hostUrl === 'gitlab.com' || hostUrl.includes('gitlab')) {
          host = 'gitlab';
        } else if (token) {
          const detectedProvider = detectProviderFromToken(token);
          if (detectedProvider) {
            host = detectedProvider;
          }
        }
        return { host, hostUrl, owner, repo };
      }

      const sshMatch = url.match(/^git@([^:]+):([^\/]+)\/([^\/]+?)(\.git)?$/);
      if (sshMatch) {
        const [, hostUrl, owner, repo] = sshMatch;
        let host: 'github' | 'gitlab' | 'generic' = 'generic';
        if (hostUrl === 'github.com') {
          host = 'github';
        } else if (hostUrl === 'gitlab.com' || hostUrl.includes('gitlab')) {
          host = 'gitlab';
        } else if (token) {
          const detectedProvider = detectProviderFromToken(token);
          if (detectedProvider) {
            host = detectedProvider;
          }
        }
        return { host, hostUrl, owner, repo };
      }

      return null;
    },
    [detectProviderFromToken],
  );

  const fetchBranches = useCallback(
    async (url: string, token?: string, silent404 = false) => {
      const parsed = parseGitUrl(url, token);
      if (!parsed) {
        setBranches([]);
        setBranchError(null);
        return;
      }

      const isPublicGitHub = parsed.hostUrl === 'github.com';
      const isPublicGitLab = parsed.hostUrl === 'gitlab.com';

      if (!isPublicGitHub && !isPublicGitLab) {
        setBranches([]);
        setBranchError(null);
        return;
      }

      setLoadingBranches(true);
      setBranchError(null);

      try {
        let apiUrl: string;
        const headers: HeadersInit = {};

        if (isPublicGitHub) {
          apiUrl = `https://api.github.com/repos/${parsed.owner}/${parsed.repo}/branches?per_page=100`;
          headers.Accept = 'application/vnd.github.v3+json';
          if (token) {
            headers.Authorization = `token ${token}`;
          }
        } else {
          const projectPath = encodeURIComponent(`${parsed.owner}/${parsed.repo}`);
          apiUrl = `https://gitlab.com/api/v4/projects/${projectPath}/repository/branches?per_page=100`;
          if (token) {
            headers['PRIVATE-TOKEN'] = token;
          }
        }

        const response = await fetch(apiUrl, { headers });

        if (!response.ok) {
          if (response.status === 404) {
            if (!silent404) {
              setBranchError(token ? 'Repository not found or token lacks access' : 'Repository not found or is private');
            }
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

        if (branchNames.length > 0 && !selectedBranch) {
          const defaultBranch = branchNames.includes('main')
            ? 'main'
            : branchNames.includes('master')
              ? 'master'
              : branchNames[0];
          setSelectedBranch(defaultBranch);
        }
      } catch {
        if (!silent404) {
          setBranchError('Failed to fetch branches');
        }
        setBranches([]);
      } finally {
        setLoadingBranches(false);
      }
    },
    [parseGitUrl, selectedBranch],
  );

  useEffect(() => {
    if (!gitUrl) {
      setBranches([]);
      setSelectedBranch('');
      setBranchError(null);
      return;
    }

    const timer = setTimeout(() => {
      if (isPrivateRepo && gitToken && gitToken.length >= 10) {
        fetchBranches(gitUrl, gitToken, false);
      } else if (!isPrivateRepo) {
        fetchBranches(gitUrl, undefined, true);
      } else {
        setBranches([]);
        setBranchError(null);
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [fetchBranches, gitToken, gitUrl, isPrivateRepo]);

  const handleAnalyze = async () => {
    if (!gitUrl) {
      setStatus({ type: 'error', message: 'Please enter a Git URL' });
      return;
    }

    const parsed = parseGitUrl(gitUrl, isPrivateRepo ? gitToken : undefined);
    if (!parsed) {
      setStatus({ type: 'error', message: 'Invalid Git URL format' });
      return;
    }

    setWizardStep('analyzing');
    setIsLoading(true);
    setStatus({ type: 'info', message: 'Analyzing repository (this may take a minute)...' });
    onAnalysisStart?.();

    try {
      const result = await api.analyzeRepository({
        git_url: gitUrl,
        git_branch: selectedBranch || 'main',
        git_token: isPrivateRepo ? gitToken : undefined,
        file_patterns: filePatterns.split(',').map((s) => s.trim()).filter(Boolean),
        exclude_patterns: excludePatterns.split(',').map((s) => s.trim()).filter(Boolean),
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        max_file_size_kb: maxFileSizeKb,
      });
      setAnalysisResult(result);
      setWizardStep('review');
      setStatus({ type: null, message: '' });
    } catch (err) {
      setStatus({ type: 'error', message: `Analysis failed: ${err instanceof Error ? err.message : 'Request failed'}` });
      setWizardStep('input');
    } finally {
      setIsLoading(false);
      onAnalysisComplete?.();
    }
  };

  const applySuggestedExclusions = () => {
    if (!analysisResult?.suggested_exclusions.length) {
      return;
    }

    const currentExcludes = excludePatterns.split(',').map((s) => s.trim()).filter(Boolean);
    const newExcludes = [...new Set([...currentExcludes, ...analysisResult.suggested_exclusions])];
    setExcludePatterns(newExcludes.join(','));
    setExclusionsApplied(true);
    setPatternsExpanded(true);
  };

  const handleReanalyze = async () => {
    setExclusionsApplied(false);
    setWizardStep('analyzing');
    await handleAnalyze();
  };

  const handleStartIndexing = async () => {
    const parsed = parseGitUrl(gitUrl, isPrivateRepo ? gitToken : undefined);
    if (!parsed) {
      setStatus({ type: 'error', message: 'Invalid Git URL format' });
      return;
    }
    const name = parsed.repo.toLowerCase().replace(/[^a-z0-9_-]/g, '-');

    setWizardStep('indexing');
    setIsLoading(true);
    setStatus({ type: 'info', message: 'Starting git clone and indexing...' });

    try {
      const job: IndexJob = await api.indexFromGit({
        name,
        git_url: gitUrl,
        git_branch: selectedBranch || 'main',
        git_token: isPrivateRepo ? gitToken : undefined,
        config: {
          name,
          description: '',
          file_patterns: filePatterns.split(',').map((s) => s.trim()).filter(Boolean),
          exclude_patterns: excludePatterns.split(',').map((s) => s.trim()).filter(Boolean),
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          max_file_size_kb: maxFileSizeKb,
        },
      });
      const successMessage = `Job started - ID: ${job.id}`;
      resetState();
      setStatus({ type: 'success', message: successMessage });
      onJobCreated?.();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Request failed'}` });
      setWizardStep('review');
    } finally {
      setIsLoading(false);
    }
  };

  const handleBack = () => {
    setWizardStep('input');
    setAnalysisResult(null);
    setStatus({ type: null, message: '' });
    setExclusionsApplied(false);
  };

  const handleCancel = () => {
    resetState();
    onCancel?.();
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (wizardStep === 'input' || wizardStep === 'analyzing') {
    return (
      <div>
        <div className="row">
          <div className="form-group" style={{ flex: 2 }}>
            <label>Git URL *</label>
            <input
              type="text"
              value={gitUrl}
              onChange={(e) => setGitUrl(e.target.value)}
              placeholder="https://github.com/user/repo.git or https://your-git-server.com/user/repo.git"
              disabled={isLoading}
            />
          </div>
          <div className="form-group" style={{ flex: 1 }}>
            <label>
              Branch
              {loadingBranches && (
                <span style={{ marginLeft: '0.5rem', color: '#888', fontSize: '0.85em' }}>(loading...)</span>
              )}
            </label>
            {branches.length > 0 ? (
              <select
                value={selectedBranch}
                onChange={(e) => setSelectedBranch(e.target.value)}
                style={{ width: '100%' }}
                disabled={isLoading}
              >
                {branches.map((branch) => (
                  <option key={branch} value={branch}>
                    {branch}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={selectedBranch}
                onChange={(e) => setSelectedBranch(e.target.value)}
                placeholder={branchError ? 'Enter branch name' : 'main'}
                disabled={isLoading}
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
          Index name will be derived from the repository name. Click "Analyze" to preview the index before creating.
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
                  if (gitUrl) {
                    fetchBranches(gitUrl);
                  }
                }
              }}
              style={{ width: 'auto', margin: 0 }}
              disabled={isLoading}
            />
            Private repository (requires authentication)
          </label>
        </div>

        {isPrivateRepo && (
          <div
            className="form-group"
            style={{ marginLeft: '1.5rem', borderLeft: '2px solid #444', paddingLeft: '1rem', marginBottom: '1rem' }}
          >
            <label>Personal Access Token *</label>
            <input
              type="password"
              value={gitToken}
              onChange={(e) => setGitToken(e.target.value)}
              placeholder="ghp_xxxx... or glpat-xxxx..."
              autoComplete="off"
              disabled={isLoading}
            />
            <small style={{ color: '#888', fontSize: '0.85em', display: 'block', marginTop: '0.25rem' }}>
              Required for private repositories. Token is used for cloning only and never stored.
            </small>
          </div>
        )}

        <details style={{ marginBottom: '16px' }}>
          <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Options</summary>
          <div className="row">
            <div className="form-group">
              <label>File Patterns (comma-separated)</label>
              <input
                type="text"
                value={filePatterns}
                onChange={(e) => setFilePatterns(e.target.value)}
                placeholder={PLACEHOLDER_FILE_PATTERNS}
                disabled={isLoading}
              />
            </div>
            <div className="form-group">
              <label>Exclude Patterns</label>
              <input
                type="text"
                value={excludePatterns}
                onChange={(e) => setExcludePatterns(e.target.value)}
                placeholder={PLACEHOLDER_EXCLUDE_PATTERNS}
                disabled={isLoading}
              />
            </div>
          </div>
          <div className="row">
            <div className="form-group">
              <label>Chunk Size</label>
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value, 10) || 1000)}
                min={100}
                max={4000}
                disabled={isLoading}
              />
            </div>
            <div className="form-group">
              <label>Chunk Overlap</label>
              <input
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(parseInt(e.target.value, 10) || 200)}
                min={0}
                max={1000}
                disabled={isLoading}
              />
            </div>
            <div className="form-group">
              <label>Max File Size (KB)</label>
              <input
                type="number"
                value={maxFileSizeKb}
                onChange={(e) => setMaxFileSizeKb(parseInt(e.target.value, 10) || 500)}
                min={10}
                max={10000}
                disabled={isLoading}
              />
              <small style={{ color: '#888', fontSize: '0.8rem' }}>Files larger than this are skipped</small>
            </div>
          </div>
        </details>

        <div className="wizard-actions">
          {onCancel && (
            <button type="button" className="btn btn-secondary" onClick={handleCancel} disabled={isLoading}>
              Cancel
            </button>
          )}
          <button type="button" className="btn" onClick={handleAnalyze} disabled={isLoading || !gitUrl}>
            {isLoading ? 'Analyzing...' : 'Analyze Repository'}
          </button>
        </div>

        {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
      </div>
    );
  }

  if (wizardStep === 'review' && analysisResult) {
    return (
      <div>
        <h4 style={{ marginBottom: '16px' }}>
          Analysis Results for: {gitUrl.split('/').pop()?.replace('.git', '')}
        </h4>

        {analysisResult.warnings.length > 0 && (
          <div
            style={{
              background: 'rgba(251, 191, 36, 0.1)',
              border: '1px solid rgba(251, 191, 36, 0.3)',
              borderRadius: '8px',
              padding: '12px',
              marginBottom: '16px',
            }}
          >
            <strong style={{ color: '#fbbf24' }}>Warnings:</strong>
            <ul style={{ margin: '8px 0 0 0', paddingLeft: '20px' }}>
              {analysisResult.warnings.map((warning, i) => (
                <li key={i} style={{ color: '#fbbf24', fontSize: '0.9rem' }}>
                  {warning}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px', marginBottom: '16px' }}>
          <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--accent)' }}>
              {analysisResult.total_files.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Files</div>
          </div>
          <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--accent)' }}>
              {analysisResult.total_size_mb.toLocaleString()} MB
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Source Size</div>
          </div>
          <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
            <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--accent)' }}>
              {analysisResult.estimated_chunks.toLocaleString()}
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Est. Chunks</div>
          </div>
          <div style={{ background: 'var(--bg-tertiary)', padding: '12px', borderRadius: '8px', textAlign: 'center' }}>
            <div
              style={{
                fontSize: '1.5rem',
                fontWeight: 'bold',
                color: analysisResult.estimated_index_size_mb > 500 ? '#f87171' : 'var(--accent)',
              }}
            >
              {analysisResult.estimated_index_size_mb.toLocaleString()} MB
            </div>
            <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Est. Index Size</div>
          </div>
        </div>

        {analysisResult.file_type_stats.length > 0 && (
          <div style={{ marginBottom: '16px' }}>
            <h5 style={{ marginBottom: '8px' }}>File Types (by estimated chunks)</h5>
            <div
              style={{ maxHeight: '200px', overflowY: 'auto', background: 'var(--bg-tertiary)', borderRadius: '8px', padding: '8px' }}
            >
              <table style={{ width: '100%', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ textAlign: 'left', color: 'var(--text-secondary)' }}>
                    <th style={{ padding: '4px 8px' }}>Extension</th>
                    <th style={{ padding: '4px 8px' }}>Files</th>
                    <th style={{ padding: '4px 8px' }}>Size</th>
                    <th style={{ padding: '4px 8px' }}>Est. Chunks</th>
                  </tr>
                </thead>
                <tbody>
                  {analysisResult.file_type_stats.slice(0, 15).map((ft) => (
                    <tr key={ft.extension}>
                      <td style={{ padding: '4px 8px', fontFamily: 'monospace' }}>{ft.extension}</td>
                      <td style={{ padding: '4px 8px' }}>{ft.file_count}</td>
                      <td style={{ padding: '4px 8px' }}>{formatBytes(ft.total_size_bytes)}</td>
                      <td style={{ padding: '4px 8px' }}>{ft.estimated_chunks.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {analysisResult.suggested_exclusions.length > 0 && !exclusionsApplied && (
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
                onClick={applySuggestedExclusions}
              >
                Apply All
              </button>
            </div>
            <code style={{ fontSize: '0.85rem', color: '#888' }}>
              {analysisResult.suggested_exclusions.join(', ')}
            </code>
          </div>
        )}

        {exclusionsApplied && (
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
              Suggested exclusions applied. Click "Re-analyze" to update estimates.
            </span>
          </div>
        )}

        <details
          style={{ marginBottom: '16px' }}
          open={patternsExpanded}
          onToggle={(e) => setPatternsExpanded((e.target as HTMLDetailsElement).open)}
        >
          <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>
            Edit Patterns & Settings
          </summary>
          <div className="row">
            <div className="form-group">
              <label>File Patterns</label>
              <input
                type="text"
                value={filePatterns}
                onChange={(e) => setFilePatterns(e.target.value)}
                disabled={isLoading}
              />
            </div>
            <div className="form-group">
              <label>Exclude Patterns</label>
              <input
                type="text"
                value={excludePatterns}
                onChange={(e) => setExcludePatterns(e.target.value)}
                disabled={isLoading}
              />
            </div>
          </div>
          <div className="row">
            <div className="form-group">
              <label>Chunk Size</label>
              <input
                type="number"
                value={chunkSize}
                onChange={(e) => setChunkSize(parseInt(e.target.value, 10) || 1000)}
                min={100}
                max={4000}
                disabled={isLoading}
              />
            </div>
            <div className="form-group">
              <label>Chunk Overlap</label>
              <input
                type="number"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(parseInt(e.target.value, 10) || 200)}
                min={0}
                max={1000}
                disabled={isLoading}
              />
            </div>
            <div className="form-group">
              <label>Max File Size (KB)</label>
              <input
                type="number"
                value={maxFileSizeKb}
                onChange={(e) => setMaxFileSizeKb(parseInt(e.target.value, 10) || 500)}
                min={10}
                max={10000}
                disabled={isLoading}
              />
            </div>
          </div>
          <button type="button" className="btn btn-secondary" onClick={handleReanalyze} disabled={isLoading} style={{ marginTop: '8px' }}>
            {isLoading ? 'Re-analyzing...' : 'Re-analyze'}
          </button>
        </details>

        <div className="wizard-actions">
          <button type="button" className="btn btn-secondary" onClick={handleBack} disabled={isLoading}>
            Back
          </button>
          {onCancel && (
            <button type="button" className="btn btn-secondary" onClick={handleCancel} disabled={isLoading}>
              Cancel
            </button>
          )}
          <button type="button" className="btn" onClick={handleStartIndexing} disabled={isLoading}>
            {isLoading ? 'Starting...' : 'Start Indexing'}
          </button>
        </div>

        {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
      </div>
    );
  }

  return (
    <div style={{ textAlign: 'center', padding: '40px' }}>
      <div style={{ fontSize: '1.2rem', marginBottom: '16px' }}>Starting indexing job...</div>
      {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
    </div>
  );
}
