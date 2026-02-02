import { useCallback, useEffect, useState } from 'react';
import { api } from '@/api';
import type { CommitHistoryInfo, IndexAnalysisResult, IndexJob, IndexInfo, VectorStoreType } from '@/types';
import { AnalysisStats } from './AnalysisStats';
import { IndexConfigFields } from './IndexConfigFields';
import { OcrVectorStoreFields } from './OcrVectorStoreFields';

type StatusType = 'info' | 'success' | 'error' | null;
type WizardStep = 'input' | 'analyzing' | 'review' | 'indexing';

/**
 * Compute clone timeout from depth using exponential scaling.
 * Matches backend logic in service.py: 5 min (shallow) -> 120 min (full).
 * Grows slowly at first, then rapidly as depth increases.
 */
function computeCloneTimeout(depth: number): number {
  const minTimeout = 5;
  const maxTimeout = 120;

  // depth=0 means full history
  if (depth === 0) return maxTimeout;

  // depth=1 means shallow clone
  if (depth === 1) return minTimeout;

  const maxDepth = 1001;  // Slider full + sentinel
  const effectiveDepth = Math.min(depth, maxDepth);
  // Use power curve (exponent > 1) for slow-then-fast growth
  const factor = Math.pow(effectiveDepth / maxDepth, 2.5);
  const timeout = minTimeout + (maxTimeout - minTimeout) * factor;

  return Math.round(Math.max(minTimeout, Math.min(maxTimeout, timeout)));
}

/**
 * Interpolates a date for a given depth from commit history samples.
 * Returns a human-readable description like "~6 months of history".
 */
function getDepthDateEstimate(depth: number, commitHistory: CommitHistoryInfo | undefined): string | null {
  if (!commitHistory?.samples || commitHistory.samples.length < 2) return null;
  if (depth === 0) return null;  // Full history - use oldest_date directly
  if (depth === 1) return null;  // Shallow - no history

  const samples = commitHistory.samples;
  const newest = samples.find((s) => s.depth === 0);
  if (!newest) return null;

  const newestDate = new Date(newest.date);
  let estimatedDate: Date | null = null;

  // Find the two samples that bracket the requested depth
  for (let i = 0; i < samples.length - 1; i++) {
    const lower = samples[i];
    const upper = samples[i + 1];
    if (depth >= lower.depth && depth <= upper.depth) {
      // Linear interpolation between the two sample dates
      const ratio = (depth - lower.depth) / (upper.depth - lower.depth);
      const lowerDate = new Date(lower.date);
      const upperDate = new Date(upper.date);
      const interpolatedMs = lowerDate.getTime() + ratio * (upperDate.getTime() - lowerDate.getTime());
      estimatedDate = new Date(interpolatedMs);
      break;
    }
  }

  // If depth is beyond our last sample, extrapolate or use oldest_date
  if (!estimatedDate && depth > samples[samples.length - 1].depth) {
    if (commitHistory.oldest_date) {
      estimatedDate = new Date(commitHistory.oldest_date);
    } else {
      // Extrapolate from last two samples
      const last = samples[samples.length - 1];
      const prev = samples[samples.length - 2];
      if (last && prev) {
        const ratio = (depth - prev.depth) / (last.depth - prev.depth);
        const prevDate = new Date(prev.date);
        const lastDate = new Date(last.date);
        const interpolatedMs = prevDate.getTime() + ratio * (lastDate.getTime() - prevDate.getTime());
        estimatedDate = new Date(interpolatedMs);
      }
    }
  }

  if (!estimatedDate) return null;

  // Calculate time difference and format nicely
  const diffMs = newestDate.getTime() - estimatedDate.getTime();
  const diffDays = Math.round(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays < 7) return `~${diffDays} days of history`;
  if (diffDays < 30) return `~${Math.round(diffDays / 7)} weeks of history`;
  if (diffDays < 365) return `~${Math.round(diffDays / 30)} months of history`;
  const years = (diffDays / 365).toFixed(1);
  return `~${years} years of history`;
}

interface GitIndexWizardProps {
  onJobCreated?: () => void;
  onCancel?: () => void;
  onAnalysisStart?: () => void;
  onAnalysisComplete?: () => void;
  /** When provided, wizard operates in edit mode for an existing git index */
  editIndex?: IndexInfo;
  /** Called when config is saved in edit mode (without triggering re-index) */
  onConfigSaved?: () => void;
  /** Called when user wants to navigate to settings */
  onNavigateToSettings?: () => void;
  /** If set, vector store type is locked (for consistency with existing indexes) */
  existingVectorStoreType?: VectorStoreType | null;
}

// Default file patterns to include all files
const DEFAULT_FILE_PATTERNS = '**/*';

export function GitIndexWizard({ onJobCreated, onCancel, onAnalysisStart, onAnalysisComplete, editIndex, onConfigSaved, onNavigateToSettings, existingVectorStoreType }: GitIndexWizardProps) {
  const isEditMode = !!editIndex;

  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: null,
    message: '',
  });
  const [wizardStep, setWizardStep] = useState<WizardStep>('input');
  const [analysisResult, setAnalysisResult] = useState<IndexAnalysisResult | null>(null);

  const [gitUrl, setGitUrl] = useState(editIndex?.source || '');
  const [gitToken, setGitToken] = useState('');
  const [isPrivateRepo, setIsPrivateRepo] = useState(false);
  const [hasStoredToken, setHasStoredToken] = useState(editIndex?.has_stored_token || false);
  const [storedTokenValid, setStoredTokenValid] = useState(true);  // Assume valid until proven otherwise
  const [checkingVisibility, setCheckingVisibility] = useState(false);
  const [branches, setBranches] = useState<string[]>([]);
  const [selectedBranch, setSelectedBranch] = useState(editIndex?.git_branch || '');
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);

  // Initialize from editIndex config_snapshot if available
  const configSnapshot = editIndex?.config_snapshot;
  const [filePatterns, setFilePatterns] = useState(
    configSnapshot?.file_patterns?.join(', ') || DEFAULT_FILE_PATTERNS
  );
  const [excludePatterns, setExcludePatterns] = useState(
    configSnapshot?.exclude_patterns?.join(', ') || ''
  );
  const [chunkSize, setChunkSize] = useState(configSnapshot?.chunk_size || 1000);
  const [chunkOverlap, setChunkOverlap] = useState(configSnapshot?.chunk_overlap || 200);
  const [maxFileSizeKb, setMaxFileSizeKb] = useState(configSnapshot?.max_file_size_kb || 500);
  const [ocrMode, setOcrMode] = useState<'disabled' | 'tesseract' | 'ollama'>(configSnapshot?.ocr_mode || 'disabled');
  const [ocrVisionModel, setOcrVisionModel] = useState(configSnapshot?.ocr_vision_model || '');
  const [vectorStoreType, setVectorStoreType] = useState<VectorStoreType>(existingVectorStoreType ?? editIndex?.vector_store_type ?? 'faiss');
  const [ollamaAvailable, setOllamaAvailable] = useState(false);  // Whether Ollama is configured as LLM provider
  const [gitCloneTimeoutMinutes, setGitCloneTimeoutMinutes] = useState(configSnapshot?.git_clone_timeout_minutes || 5);
  const [gitHistoryDepth, setGitHistoryDepth] = useState(configSnapshot?.git_history_depth || 1);
  const [reindexIntervalHours, setReindexIntervalHours] = useState(configSnapshot?.reindex_interval_hours || 0);
  const [timeoutManuallySet, setTimeoutManuallySet] = useState(false);  // Track if user overrode timeout
  const [exclusionsApplied, setExclusionsApplied] = useState(false);
  const [patternsExpanded, setPatternsExpanded] = useState(isEditMode);  // Expand by default in edit mode
  const [description, setDescription] = useState(editIndex?.description || '');
  const [indexName, setIndexName] = useState(editIndex?.display_name || editIndex?.name || '');

  // Fetch settings to check if Ollama OCR is configured
  useEffect(() => {
    const loadSettings = async () => {
      try {
        const { settings } = await api.getSettings();
        setOllamaAvailable(settings.default_ocr_mode === 'ollama');
      } catch {
        // If settings can't be loaded, Ollama is not available
        setOllamaAvailable(false);
      }
    };
    loadSettings();
  }, []);

  // Auto-update timeout when depth changes (unless user manually overrode it)
  useEffect(() => {
    if (!timeoutManuallySet) {
      const computed = computeCloneTimeout(gitHistoryDepth);
      setGitCloneTimeoutMinutes(computed);
    }
  }, [gitHistoryDepth, timeoutManuallySet]);

  // Sync state when editIndex changes (for when modal reopens with different index)
  useEffect(() => {
    if (editIndex) {
      setIndexName(editIndex.display_name || editIndex.name || '');
      setGitUrl(editIndex.source || '');
      setSelectedBranch(editIndex.git_branch || '');
      setDescription(editIndex.description || '');
      setHasStoredToken(editIndex.has_stored_token || false);
      setStoredTokenValid(true);  // Reset to assume valid
      const snapshot = editIndex.config_snapshot;
      if (snapshot) {
        setFilePatterns(snapshot.file_patterns?.join(', ') || DEFAULT_FILE_PATTERNS);
        setExcludePatterns(snapshot.exclude_patterns?.join(', ') || '');
        setChunkSize(snapshot.chunk_size || 1000);
        setChunkOverlap(snapshot.chunk_overlap || 200);
        setMaxFileSizeKb(snapshot.max_file_size_kb || 500);
        setOcrMode(snapshot.ocr_mode || 'disabled');
        setOcrVisionModel(snapshot.ocr_vision_model || '');
        setReindexIntervalHours(snapshot.reindex_interval_hours || 0);

        // Use nullish coalescing - 0 is a valid depth (full history)
        const loadedDepth = snapshot.git_history_depth ?? 1;
        const loadedTimeout = snapshot.git_clone_timeout_minutes ?? 5;
        const expectedTimeout = computeCloneTimeout(loadedDepth);

        setGitHistoryDepth(loadedDepth);
        setGitCloneTimeoutMinutes(loadedTimeout);
        // If timeout differs from what would be auto-computed, mark as manually set
        setTimeoutManuallySet(loadedTimeout !== expectedTimeout);
      } else {
        setFilePatterns(DEFAULT_FILE_PATTERNS);
        setExcludePatterns('');
        setChunkSize(1000);
        setChunkOverlap(200);
        setMaxFileSizeKb(500);
        setOcrMode('disabled');
        setOcrVisionModel('');
        setReindexIntervalHours(0);
        setGitCloneTimeoutMinutes(5);
        setGitHistoryDepth(1);
        setTimeoutManuallySet(false);
      }
      setPatternsExpanded(true);
    }
  }, [editIndex]);

  // Check repo visibility in edit mode to detect if repo became private
  useEffect(() => {
    if (!isEditMode || !editIndex?.source) return;

    const checkVisibility = async () => {
      setCheckingVisibility(true);
      try {
        const result = await api.checkRepoVisibility({
          git_url: editIndex.source!,
          index_name: editIndex.name,
        });

        if (result.visibility === 'private') {
          setIsPrivateRepo(true);
          setHasStoredToken(result.has_stored_token);
          setStoredTokenValid(!result.needs_token);
          if (result.needs_token) {
            setBranchError(result.message);
          }
        } else if (result.visibility === 'public') {
          setIsPrivateRepo(false);
          setBranchError(null);
        }
        // For 'error' or 'not_found', keep current state
      } catch {
        // Silently fail - don't break the UI
      } finally {
        setCheckingVisibility(false);
      }
    };

    checkVisibility();
  }, [isEditMode, editIndex?.source, editIndex?.name]);

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
    setOcrMode('disabled');
    setOcrVisionModel('');
    setVectorStoreType('faiss');
    setExclusionsApplied(false);
    setPatternsExpanded(false);
  }, []);

  /**
   * Parse a Git URL to extract the repository name.
   * Used for generating index names and URL validation.
   */
  const parseGitUrl = useCallback(
    (url: string): { repo: string } | null => {
      if (!url || typeof url !== 'string') {
        return null;
      }

      // HTTPS format: https://github.com/owner/repo.git
      const httpsMatch = url.match(/^https?:\/\/[^\/]+\/[^\/]+\/([^\/]+?)(\.git)?$/);
      if (httpsMatch) {
        return { repo: httpsMatch[1] };
      }

      // SSH format: git@github.com:owner/repo.git
      const sshMatch = url.match(/^git@[^:]+:[^\/]+\/([^\/]+?)(\.git)?$/);
      if (sshMatch) {
        return { repo: sshMatch[1] };
      }

      return null;
    },
    [],
  );

  const fetchBranches = useCallback(
    async (url: string, token?: string, silent404 = false) => {
      if (!url) {
        setBranches([]);
        setBranchError(null);
        return;
      }

      setLoadingBranches(true);
      setBranchError(null);

      try {
        const result = await api.fetchBranches({
          git_url: url,
          git_token: token || undefined,
          index_name: editIndex?.name,
        });

        if (result.error) {
          if (!silent404 && result.error) {
            setBranchError(result.error);
          }
          setBranches([]);
          return;
        }

        const branchNames = result.branches;
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
    [editIndex?.name, selectedBranch],
  );

  // Fetch branches on mount in edit mode (if we have a git URL)
  useEffect(() => {
    if (isEditMode && editIndex?.source) {
      // Immediately try to fetch branches for the existing URL
      fetchBranches(editIndex.source, undefined, true);
    }
  }, [isEditMode, editIndex?.source, fetchBranches]);

  useEffect(() => {
    if (!gitUrl) {
      setBranches([]);
      if (!isEditMode) {
        setSelectedBranch('');
      }
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
  }, [fetchBranches, gitToken, gitUrl, isPrivateRepo, isEditMode]);

  const handleAnalyze = async () => {
    if (!gitUrl) {
      setStatus({ type: 'error', message: 'Please enter a Git URL' });
      return;
    }

    const parsed = parseGitUrl(gitUrl);
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
        ocr_mode: ocrMode,
        ocr_vision_model: ocrVisionModel || undefined,
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
    const parsed = parseGitUrl(gitUrl);
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
          vector_store_type: vectorStoreType,
          ocr_mode: ocrMode,
          ocr_vision_model: ocrVisionModel || undefined,
          git_clone_timeout_minutes: gitCloneTimeoutMinutes,
          git_history_depth: gitHistoryDepth,
          reindex_interval_hours: reindexIntervalHours,
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

  /**
   * Save config changes in edit mode (does not trigger re-index)
   */
  const handleSaveConfig = async () => {
    if (!editIndex) return;

    setIsLoading(true);
    setStatus({ type: 'info', message: 'Saving configuration...' });

    try {
      // Track the current name for API calls (may change if renamed)
      let currentName = editIndex.name;

      // If name has changed, rename the index first
      // The backend will automatically convert the name to a safe identifier
      // Compare against display_name (human-readable) not the safe tool name
      const trimmedName = indexName.trim();
      const originalDisplayName = editIndex.display_name || editIndex.name;
      if (trimmedName && trimmedName !== originalDisplayName) {
        setStatus({ type: 'info', message: 'Renaming index...' });
        const renameResult = await api.renameIndex(editIndex.name, trimmedName);
        currentName = renameResult.new_name;
        // Update to display_name for the UI, not the safe tool name
        setIndexName(renameResult.display_name);
      }

      // Update description (using the potentially new name)
      await api.updateIndexDescription(currentName, description);

      // Update config
      const updated = await api.updateIndexConfig(currentName, {
        git_branch: selectedBranch || undefined,
        file_patterns: filePatterns.split(',').map((s) => s.trim()).filter(Boolean),
        exclude_patterns: excludePatterns.split(',').map((s) => s.trim()).filter(Boolean),
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        max_file_size_kb: maxFileSizeKb,
        ocr_mode: ocrMode,
        ocr_vision_model: ocrVisionModel || undefined,
        git_clone_timeout_minutes: gitCloneTimeoutMinutes,
        git_history_depth: gitHistoryDepth,
        reindex_interval_hours: reindexIntervalHours,
      });
      // Reflect saved config locally so the UI shows persisted values
      const snap = updated?.config_snapshot;
      if (snap) {
        setGitHistoryDepth(snap.git_history_depth ?? gitHistoryDepth);
        setGitCloneTimeoutMinutes(snap.git_clone_timeout_minutes ?? gitCloneTimeoutMinutes);
        setReindexIntervalHours(snap.reindex_interval_hours ?? reindexIntervalHours);
        setFilePatterns(snap.file_patterns?.join(', ') || filePatterns);
        setExcludePatterns(snap.exclude_patterns?.join(', ') || excludePatterns);
        setChunkSize(snap.chunk_size ?? chunkSize);
        setChunkOverlap(snap.chunk_overlap ?? chunkOverlap);
        setMaxFileSizeKb(snap.max_file_size_kb ?? maxFileSizeKb);
        setOcrMode(snap.ocr_mode ?? ocrMode);
        setOcrVisionModel(snap.ocr_vision_model ?? ocrVisionModel);
      }

      const wasRenamed = currentName !== editIndex.name;
      const savedMessage = wasRenamed
        ? `Index renamed to "${indexName}" and configuration saved. Click "Pull & Re-index" to apply changes.`
        : 'Configuration saved. Click "Pull & Re-index" to apply changes.';
      setStatus({ type: 'success', message: savedMessage });
      onConfigSaved?.();
    } catch (err) {
      setStatus({ type: 'error', message: `Error: ${err instanceof Error ? err.message : 'Save failed'}` });
    } finally {
      setIsLoading(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // Edit mode: show simplified config editor
  if (isEditMode && wizardStep === 'input') {
    return (
      <div>
        <h4 style={{ marginBottom: '12px' }}>Edit Index Configuration</h4>
        <p className="field-help" style={{ marginBottom: '16px' }}>
          Update settings for the next time you click "Pull & Re-index". Changes will not take effect until you re-index.
        </p>

        <div style={{ marginBottom: '16px', padding: '12px', background: 'var(--bg-tertiary)', borderRadius: '8px', fontSize: '13px' }}>
          <div><strong>Source:</strong> {editIndex.source}</div>
        </div>

        <div className="form-group">
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
        </div>

        <OcrVectorStoreFields
          isLoading={isLoading}
          ocrMode={ocrMode}
          setOcrMode={setOcrMode}
          ollamaAvailable={ollamaAvailable}
          vectorStoreType={vectorStoreType}
          setVectorStoreType={setVectorStoreType}
        />

        <details style={{ marginBottom: '16px' }}>
          <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Options</summary>
          <IndexConfigFields
            isLoading={isLoading}
            filePatterns={filePatterns}
            setFilePatterns={setFilePatterns}
            excludePatterns={excludePatterns}
            setExcludePatterns={setExcludePatterns}
            chunkSize={chunkSize}
            setChunkSize={setChunkSize}
            chunkOverlap={chunkOverlap}
            setChunkOverlap={setChunkOverlap}
            maxFileSizeKb={maxFileSizeKb}
            setMaxFileSizeKb={setMaxFileSizeKb}
            gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
            setGitCloneTimeoutMinutes={setGitCloneTimeoutMinutes}
            setTimeoutManuallySet={setTimeoutManuallySet}
            reindexIntervalHours={reindexIntervalHours}
            setReindexIntervalHours={setReindexIntervalHours}
            gitHistoryDepth={gitHistoryDepth}
            setGitHistoryDepth={setGitHistoryDepth}
          />
        </details>

        <div className="wizard-actions" style={{ marginTop: '16px' }}>
          {onCancel && (
            <button type="button" className="btn btn-secondary" onClick={handleCancel} disabled={isLoading}>
              Cancel
            </button>
          )}
          <button type="button" className="btn" onClick={handleSaveConfig} disabled={isLoading}>
            {isLoading ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>

        {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
      </div>
    );
  }

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
              disabled={isLoading || checkingVisibility}
            />
            Private repository (requires authentication)
            {checkingVisibility && <span style={{ color: '#888', fontSize: '0.85em' }}>(checking...)</span>}
          </label>
        </div>

        {isPrivateRepo && (
          <div
            className="form-group"
            style={{ marginLeft: '1.5rem', borderLeft: '2px solid #444', paddingLeft: '1rem', marginBottom: '1rem' }}
          >
            {/* Show stored token status in edit mode */}
            {isEditMode && hasStoredToken && storedTokenValid && (
              <div style={{ marginBottom: '12px', padding: '12px', background: 'rgba(34, 197, 94, 0.1)', borderRadius: '8px', border: '1px solid rgba(34, 197, 94, 0.3)' }}>
                <span style={{ color: '#22c55e' }}>Token stored - will use existing credentials.</span>
                <button
                  type="button"
                  onClick={() => setStoredTokenValid(false)}
                  style={{ marginLeft: '12px', padding: '4px 8px', fontSize: '12px', background: 'transparent', border: '1px solid #666', borderRadius: '4px', color: '#888', cursor: 'pointer' }}
                >
                  Update Token
                </button>
              </div>
            )}

            {/* Show warning if stored token is invalid */}
            {isEditMode && hasStoredToken && !storedTokenValid && (
              <div style={{ marginBottom: '12px', padding: '12px', background: 'rgba(251, 191, 36, 0.1)', borderRadius: '8px', border: '1px solid rgba(251, 191, 36, 0.3)' }}>
                <span style={{ color: '#fbbf24' }}>Stored token no longer works - please provide a new token.</span>
              </div>
            )}

            {/* Show token input if needed */}
            {(!isEditMode || !hasStoredToken || !storedTokenValid) && (
              <>
                <label>Personal Access Token {!isEditMode ? '*' : ''}</label>
                <input
                  type="password"
                  value={gitToken}
                  onChange={(e) => setGitToken(e.target.value)}
                  placeholder="ghp_xxxx... or glpat-xxxx..."
                  autoComplete="off"
                  disabled={isLoading}
                />
                <small style={{ color: '#888', fontSize: '0.85em', display: 'block', marginTop: '0.25rem' }}>
                  {isEditMode
                    ? 'Provide a new token to update stored credentials.'
                    : 'Required for private repositories. Token is stored securely for automatic re-indexing.'}
                </small>
              </>
            )}
          </div>
        )}

        <OcrVectorStoreFields
          isLoading={isLoading}
          ocrMode={ocrMode}
          setOcrMode={setOcrMode}
          ollamaAvailable={ollamaAvailable}
          vectorStoreType={vectorStoreType}
          setVectorStoreType={setVectorStoreType}
          vectorStoreDisabled={!!existingVectorStoreType}
        />

        <details style={{ marginBottom: '16px' }}>
          <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>Advanced Options</summary>
          <IndexConfigFields
            isLoading={isLoading}
            filePatterns={filePatterns}
            setFilePatterns={setFilePatterns}
            excludePatterns={excludePatterns}
            setExcludePatterns={setExcludePatterns}
            chunkSize={chunkSize}
            setChunkSize={setChunkSize}
            chunkOverlap={chunkOverlap}
            setChunkOverlap={setChunkOverlap}
            maxFileSizeKb={maxFileSizeKb}
            setMaxFileSizeKb={setMaxFileSizeKb}
            gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
            setGitCloneTimeoutMinutes={setGitCloneTimeoutMinutes}
            setTimeoutManuallySet={setTimeoutManuallySet}
            reindexIntervalHours={reindexIntervalHours}
            setReindexIntervalHours={setReindexIntervalHours}
            gitHistoryDepth={gitHistoryDepth}
            setGitHistoryDepth={setGitHistoryDepth}
          />
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

        <AnalysisStats result={analysisResult} onNavigateToSettings={onNavigateToSettings} />

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
                      <td style={{ padding: '4px 8px', fontFamily: 'var(--font-mono)' }}>{ft.extension}</td>
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

        <OcrVectorStoreFields
          isLoading={isLoading}
          ocrMode={ocrMode}
          setOcrMode={setOcrMode}
          ollamaAvailable={ollamaAvailable}
          vectorStoreType={vectorStoreType}
          setVectorStoreType={setVectorStoreType}
          vectorStoreDisabled={!!existingVectorStoreType}
        />

        <details
          style={{ marginBottom: '16px' }}
          open={patternsExpanded}
          onToggle={(e) => setPatternsExpanded((e.target as HTMLDetailsElement).open)}
        >
          <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px' }}>
            Advanced Options
          </summary>

          <IndexConfigFields
            isLoading={isLoading}
            filePatterns={filePatterns}
            setFilePatterns={setFilePatterns}
            excludePatterns={excludePatterns}
            setExcludePatterns={setExcludePatterns}
            chunkSize={chunkSize}
            setChunkSize={setChunkSize}
            chunkOverlap={chunkOverlap}
            setChunkOverlap={setChunkOverlap}
            maxFileSizeKb={maxFileSizeKb}
            setMaxFileSizeKb={setMaxFileSizeKb}
            gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
            setGitCloneTimeoutMinutes={setGitCloneTimeoutMinutes}
            setTimeoutManuallySet={setTimeoutManuallySet}
            reindexIntervalHours={reindexIntervalHours}
            setReindexIntervalHours={setReindexIntervalHours}
            gitHistoryDepth={gitHistoryDepth}
            setGitHistoryDepth={setGitHistoryDepth}
            commitHistory={analysisResult.commit_history}
            getDepthDateEstimate={getDepthDateEstimate}
          />

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
