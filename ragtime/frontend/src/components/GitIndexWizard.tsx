import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from '@/api';
import type {
  GitWebhookConfig,
  GitWebhookEnableResponse,
  IndexAnalysisResult,
  IndexJob,
  IndexInfo,
  OcrMode,
  OcrProvider,
  VectorStoreType,
} from '@/types';
import { AnalysisStats } from './AnalysisStats';
import { OcrVectorStoreFields, OCR_PROVIDER_LABELS } from './OcrVectorStoreFields';
import { FileTypeStatsTable } from './FileTypeStatsTable';
import { SuggestedExclusionsBanner } from './SuggestedExclusionsBanner';
import { WarningsBanner } from './WarningsBanner';
import { ReindexIntervalSelect } from './ReindexIntervalSelect';
import { defaultScheduleStartMinute, defaultScheduleTimezone } from './ScheduleStartTimeInput';
import { GitWebhookSettings } from './GitWebhookSettings';
import { GitHistoryDepthAndAdvancedOptions } from './GitHistoryDepthAndAdvancedOptions';

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

  const maxDepth = 1001; // Slider full + sentinel
  const effectiveDepth = Math.min(depth, maxDepth);
  // Use power curve (exponent > 1) for slow-then-fast growth
  const factor = Math.pow(effectiveDepth / maxDepth, 2.5);
  const timeout = minTimeout + (maxTimeout - minTimeout) * factor;

  return Math.round(Math.max(minTimeout, Math.min(maxTimeout, timeout)));
}

const INDEX_JOB_POLL_INTERVAL_MS = 1000;

function isIndexJobTerminal(job: IndexJob | null): boolean {
  if (!job) return false;
  return job.status === 'completed' || job.status === 'failed';
}

function formatIndexJobPhase(job: IndexJob): string {
  const labels: Record<IndexJob['phase'], string> = {
    preparing: 'Preparing',
    cloning: 'Cloning repository',
    scanning: 'Scanning files',
    loading: 'Loading files',
    chunking: 'Chunking',
    embedding: 'Embedding',
    finalizing: 'Finalizing',
    completed: 'Completed',
    failed: 'Failed',
    cancelled: 'Cancelled',
  };
  return labels[job.phase] || 'Processing';
}

/**
 * Display percentage for the job. During the clone phase the overall
 * indexing percent is still 0, so surface the dedicated clone progress.
 */
function getIndexJobDisplayPercent(job: IndexJob): number {
  if (job.status === 'completed') return 100;
  if (job.phase === 'cloning' && typeof job.clone_progress === 'number') {
    return Math.max(0, Math.min(100, Math.round(job.clone_progress * 100)));
  }
  return Math.max(0, Math.min(100, Math.round(job.progress_percent)));
}

function getWebhookErrorMessage(prefix: string, err: unknown): string {
  return `${prefix}: ${err instanceof Error ? err.message : 'Request failed'}`;
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

export function GitIndexWizard({
  onJobCreated,
  onCancel,
  onAnalysisStart,
  onAnalysisComplete,
  editIndex,
  onConfigSaved,
  onNavigateToSettings,
  existingVectorStoreType,
}: GitIndexWizardProps) {
  const isEditMode = !!editIndex;

  const [isLoading, setIsLoading] = useState(false);
  const [status, setStatus] = useState<{ type: StatusType; message: string }>({
    type: null,
    message: '',
  });
  const [wizardStep, setWizardStep] = useState<WizardStep>('input');
  const [analysisResult, setAnalysisResult] = useState<IndexAnalysisResult | null>(null);
  const [indexingJob, setIndexingJob] = useState<IndexJob | null>(null);
  const notifiedJobCreatedRef = useRef(false);
  const webhookRequestRef = useRef(0);

  const [gitUrl, setGitUrl] = useState(editIndex?.source || '');
  const [gitToken, setGitToken] = useState('');
  const [isPrivateRepo, setIsPrivateRepo] = useState(false);
  const [hasStoredToken, setHasStoredToken] = useState(editIndex?.has_stored_token || false);
  const [storedTokenValid, setStoredTokenValid] = useState(true); // Assume valid until proven otherwise
  const [checkingVisibility, setCheckingVisibility] = useState(false);
  const [branches, setBranches] = useState<string[]>([]);
  const [selectedBranch, setSelectedBranch] = useState(editIndex?.git_branch || '');
  const [loadingBranches, setLoadingBranches] = useState(false);
  const [branchError, setBranchError] = useState<string | null>(null);

  // Initialize from editIndex config_snapshot if available
  const configSnapshot = editIndex?.config_snapshot;
  const [filePatterns, setFilePatterns] = useState(
    configSnapshot?.file_patterns?.join(', ') || DEFAULT_FILE_PATTERNS,
  );
  const [excludePatterns, setExcludePatterns] = useState(
    configSnapshot?.exclude_patterns?.join(', ') || '',
  );
  const [chunkSize, setChunkSize] = useState(configSnapshot?.chunk_size || 1000);
  const [chunkOverlap, setChunkOverlap] = useState(configSnapshot?.chunk_overlap || 200);
  const [maxFileSizeKb, setMaxFileSizeKb] = useState(configSnapshot?.max_file_size_kb || 500);
  const [ocrMode, setOcrMode] = useState<OcrMode>(configSnapshot?.ocr_mode || 'disabled');
  const [ocrProvider, setOcrProvider] = useState<OcrProvider | null>(
    configSnapshot?.ocr_provider ?? null,
  );
  const [ocrVisionModel, setOcrVisionModel] = useState(configSnapshot?.ocr_vision_model || '');
  const [vectorStoreType, setVectorStoreType] = useState<VectorStoreType>(
    existingVectorStoreType ?? editIndex?.vector_store_type ?? 'faiss',
  );
  const [visionOcrAvailable] = useState(true);
  const [defaultOcrProviderLabel, setDefaultOcrProviderLabel] = useState<string | undefined>(
    undefined,
  );
  const [defaultOcrVisionModelLabel, setDefaultOcrVisionModelLabel] = useState<string | undefined>(
    undefined,
  );
  const [gitCloneTimeoutMinutes, setGitCloneTimeoutMinutes] = useState(
    configSnapshot?.git_clone_timeout_minutes || 5,
  );
  const [gitHistoryDepth, setGitHistoryDepth] = useState(configSnapshot?.git_history_depth ?? 1);
  const [reindexIntervalHours, setReindexIntervalHours] = useState(
    configSnapshot?.reindex_interval_hours || 0,
  );
  const [reindexStartMinute, setReindexStartMinute] = useState<number | null>(
    configSnapshot?.reindex_start_minute ?? null,
  );
  const [reindexTimezone, setReindexTimezone] = useState<string | null>(
    configSnapshot?.reindex_timezone ?? null,
  );
  const [timeoutManuallySet, setTimeoutManuallySet] = useState(false); // Track if user overrode timeout
  const [exclusionsApplied, setExclusionsApplied] = useState(false);
  const [patternsExpanded, setPatternsExpanded] = useState(isEditMode); // Expand by default in edit mode
  const [description, setDescription] = useState(editIndex?.description || '');
  const [indexName, setIndexName] = useState(editIndex?.display_name || editIndex?.name || '');
  const [configureWebhookAfterCreate, setConfigureWebhookAfterCreate] = useState(false);
  const [webhookConfig, setWebhookConfig] = useState<GitWebhookConfig | null>(null);
  const [revealedWebhookSecret, setRevealedWebhookSecret] = useState<string | null>(null);
  const [webhookError, setWebhookError] = useState<string | null>(null);
  const [webhookRequestState, setWebhookRequestState] = useState<'idle' | 'loading' | 'mutating'>(
    'idle',
  );

  useEffect(() => {
    return () => {
      webhookRequestRef.current += 1;
    };
  }, []);

  const loadWebhookState = useCallback(async (name: string) => {
    const requestId = webhookRequestRef.current + 1;
    webhookRequestRef.current = requestId;
    setWebhookRequestState('loading');
    setWebhookError(null);

    try {
      const nextConfig = await api.getIndexWebhook(name);
      if (requestId !== webhookRequestRef.current) {
        return;
      }
      if (nextConfig.enabled) {
        setReindexIntervalHours(0);
        setReindexStartMinute(null);
        setReindexTimezone(null);
      }
      setWebhookConfig(nextConfig);
    } catch (err) {
      if (requestId !== webhookRequestRef.current) {
        return;
      }
      setWebhookError(getWebhookErrorMessage('Failed to load webhook settings', err));
    } finally {
      if (requestId === webhookRequestRef.current) {
        setWebhookRequestState('idle');
      }
    }
  }, []);

  const applyWebhookMutation = useCallback(
    async (
      runMutation: () => Promise<void | GitWebhookConfig | GitWebhookEnableResponse>,
      fallbackConfig?: GitWebhookConfig,
    ): Promise<boolean> => {
      const requestId = webhookRequestRef.current + 1;
      webhookRequestRef.current = requestId;
      setWebhookRequestState('mutating');
      setWebhookError(null);

      try {
        const result = await runMutation();
        if (requestId !== webhookRequestRef.current) {
          return false;
        }

        let reflectedLocally = false;

        if (result) {
          setWebhookConfig({
            enabled: result.enabled,
            paused: result.paused,
            webhook_url: result.webhook_url,
            provider: result.provider,
            branch: result.branch,
            created_at: result.created_at,
          });
          setRevealedWebhookSecret('secret' in result ? result.secret || null : null);
          reflectedLocally = true;
        } else if (fallbackConfig) {
          setWebhookConfig(fallbackConfig);
          setRevealedWebhookSecret(null);
          reflectedLocally = true;
        }
        if (requestId !== webhookRequestRef.current) {
          return false;
        }
        return reflectedLocally;
      } catch (err) {
        if (requestId !== webhookRequestRef.current) {
          return false;
        }
        setWebhookError(getWebhookErrorMessage('Webhook setup failed', err));
        return false;
      } finally {
        if (requestId === webhookRequestRef.current) {
          setWebhookRequestState('idle');
        }
      }
    },
    [],
  );

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
      setStoredTokenValid(true); // Reset to assume valid
      const snapshot = editIndex.config_snapshot;
      if (snapshot) {
        setFilePatterns(snapshot.file_patterns?.join(', ') || DEFAULT_FILE_PATTERNS);
        setExcludePatterns(snapshot.exclude_patterns?.join(', ') || '');
        setChunkSize(snapshot.chunk_size || 1000);
        setChunkOverlap(snapshot.chunk_overlap || 200);
        setMaxFileSizeKb(snapshot.max_file_size_kb || 500);
        setOcrMode(snapshot.ocr_mode || 'disabled');
        setOcrProvider(snapshot.ocr_provider ?? null);
        setOcrVisionModel(snapshot.ocr_vision_model || '');
        setReindexIntervalHours(snapshot.reindex_interval_hours || 0);
        setReindexStartMinute(snapshot.reindex_start_minute ?? null);
        setReindexTimezone(snapshot.reindex_timezone ?? null);

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
        setOcrProvider(null);
        setOcrVisionModel('');
        setReindexIntervalHours(0);
        setReindexStartMinute(null);
        setReindexTimezone(null);
        setGitCloneTimeoutMinutes(5);
        setGitHistoryDepth(1);
        setTimeoutManuallySet(false);
      }
      setPatternsExpanded(true);
      setConfigureWebhookAfterCreate(false);
      setWebhookConfig(null);
      setRevealedWebhookSecret(null);
      setWebhookError(null);
      setWebhookRequestState('idle');
      webhookRequestRef.current += 1;
    }
  }, [editIndex]);

  useEffect(() => {
    if (!isEditMode || editIndex?.source_type !== 'git' || !editIndex?.name) {
      return;
    }

    void loadWebhookState(editIndex.name);
  }, [editIndex?.name, editIndex?.source_type, isEditMode, loadWebhookState]);

  // Fetch global OCR default settings to show in helptext
  useEffect(() => {
    api
      .getSettings()
      .then((response) => {
        const settings = response.settings;
        if (settings.default_ocr_provider) {
          setDefaultOcrProviderLabel(
            OCR_PROVIDER_LABELS[settings.default_ocr_provider] || settings.default_ocr_provider,
          );
        }
        if (settings.default_ocr_vision_model) {
          setDefaultOcrVisionModelLabel(settings.default_ocr_vision_model);
        }
      })
      .catch(() => {
        // Silently fail - helptext will fall back to generic message
      });
  }, []);

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
    setIndexingJob(null);
    notifiedJobCreatedRef.current = false;
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
    setOcrProvider(null);
    setOcrVisionModel('');
    setVectorStoreType('faiss');
    setReindexIntervalHours(0);
    setReindexStartMinute(null);
    setReindexTimezone(null);
    setExclusionsApplied(false);
    setPatternsExpanded(false);
    setConfigureWebhookAfterCreate(false);
    setWebhookConfig(null);
    setRevealedWebhookSecret(null);
    setWebhookError(null);
    setWebhookRequestState('idle');
    webhookRequestRef.current += 1;
  }, []);

  /**
   * Parse a Git URL to extract the repository name.
   * Used for generating index names and URL validation.
   */
  const parseGitUrl = useCallback((url: string): { repo: string } | null => {
    if (!url || typeof url !== 'string') {
      return null;
    }

    // HTTPS format: https://github.com/owner/repo.git
    const httpsMatch = url.match(/^https?:\/\/[^/]+\/[^/]+\/([^/]+?)(\.git)?$/);
    if (httpsMatch) {
      return { repo: httpsMatch[1] };
    }

    // SSH format: git@github.com:owner/repo.git
    const sshMatch = url.match(/^git@[^:]+:[^/]+\/([^/]+?)(\.git)?$/);
    if (sshMatch) {
      return { repo: sshMatch[1] };
    }

    return null;
  }, []);

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

  // Poll the indexing job so the wizard can show live clone/indexing progress.
  const indexingJobId = indexingJob?.id ?? null;
  const indexingJobStatus = indexingJob?.status ?? null;
  useEffect(() => {
    if (!indexingJobId || indexingJobStatus === 'completed' || indexingJobStatus === 'failed') {
      return;
    }
    const jobId = indexingJobId;
    let cancelled = false;
    let pollInFlight = false;

    const pollJob = async () => {
      if (pollInFlight) return;
      pollInFlight = true;
      try {
        const nextJob = await api.getJob(jobId);
        if (cancelled) return;
        setIndexingJob(nextJob);
        if (nextJob.status === 'completed') {
          setStatus({ type: 'success', message: `Indexing complete: ${nextJob.name}` });
        } else if (nextJob.status === 'failed') {
          setStatus({ type: 'error', message: nextJob.error_message || 'Indexing failed.' });
        }
      } catch (err) {
        if (!cancelled) {
          setStatus({
            type: 'error',
            message: `Failed to refresh job: ${err instanceof Error ? err.message : 'Request failed'}`,
          });
        }
      } finally {
        pollInFlight = false;
      }
    };

    void pollJob();
    const intervalId = window.setInterval(() => {
      void pollJob();
    }, INDEX_JOB_POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [indexingJobId, indexingJobStatus]);

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
    const name = parsed.repo.toLowerCase().replace(/[^a-z0-9_-]/g, '-');

    setWizardStep('analyzing');
    setIsLoading(true);
    setStatus({ type: 'info', message: 'Analyzing repository (this may take a minute)...' });
    onAnalysisStart?.();

    try {
      const result = await api.analyzeRepository({
        index_name: name,
        git_url: gitUrl,
        git_branch: selectedBranch || 'main',
        git_token: isPrivateRepo ? gitToken : undefined,
        file_patterns: filePatterns
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        exclude_patterns: excludePatterns
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        max_file_size_kb: maxFileSizeKb,
        ocr_mode: ocrMode,
        ocr_provider: ocrMode === 'vision' ? ocrProvider : null,
        ocr_vision_model: ocrVisionModel || undefined,
      });
      setAnalysisResult(result);
      setWizardStep('review');
      setStatus({ type: null, message: '' });
    } catch (err) {
      setStatus({
        type: 'error',
        message: `Analysis failed: ${err instanceof Error ? err.message : 'Request failed'}`,
      });
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

    const currentExcludes = excludePatterns
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean);
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
    notifiedJobCreatedRef.current = false;
    setIndexingJob(null);
    setStatus({ type: 'info', message: 'Starting git clone and indexing...' });
    setWebhookError(null);
    setRevealedWebhookSecret(null);

    try {
      const job: IndexJob = await api.indexFromGit({
        name,
        git_url: gitUrl,
        git_branch: selectedBranch || 'main',
        git_token: isPrivateRepo ? gitToken : undefined,
        config: {
          name,
          description: '',
          file_patterns: filePatterns
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean),
          exclude_patterns: excludePatterns
            .split(',')
            .map((s) => s.trim())
            .filter(Boolean),
          chunk_size: chunkSize,
          chunk_overlap: chunkOverlap,
          max_file_size_kb: maxFileSizeKb,
          vector_store_type: vectorStoreType,
          ocr_mode: ocrMode,
          ocr_provider: ocrMode === 'vision' ? ocrProvider : null,
          ocr_vision_model: ocrVisionModel || undefined,
          git_clone_timeout_minutes: gitCloneTimeoutMinutes,
          git_history_depth: gitHistoryDepth,
          reindex_interval_hours: reindexIntervalHours,
          reindex_start_minute:
            reindexIntervalHours > 0 ? (reindexStartMinute ?? defaultScheduleStartMinute()) : null,
          reindex_timezone:
            reindexIntervalHours > 0 ? (reindexTimezone ?? defaultScheduleTimezone()) : null,
        },
      });
      // Keep the job in state and poll it so the wizard shows live
      // clone + indexing progress instead of dismissing immediately.
      setIndexingJob(job);
      notifiedJobCreatedRef.current = true;
      onJobCreated?.();
      setStatus({ type: 'info', message: 'Cloning repository...' });

      if (configureWebhookAfterCreate) {
        await applyWebhookMutation(() => api.enableIndexWebhook(name));
      } else {
        setWebhookConfig(null);
      }
    } catch (err) {
      setStatus({
        type: 'error',
        message: `Error: ${err instanceof Error ? err.message : 'Request failed'}`,
      });
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

  const handleClearFields = async () => {
    if (isLoading) return;

    const parsedForClear = parseGitUrl(gitUrl);
    const clearTokenIndexName =
      isEditMode && editIndex && hasStoredToken
        ? editIndex.name
        : !isEditMode && analysisResult && isPrivateRepo && gitToken.trim() && parsedForClear
          ? parsedForClear.repo.toLowerCase().replace(/[^a-z0-9_-]/g, '-')
          : null;

    setStatus({ type: null, message: '' });
    setWizardStep('input');
    setAnalysisResult(null);
    setIndexingJob(null);
    notifiedJobCreatedRef.current = false;
    setGitUrl(isEditMode ? editIndex?.source || '' : '');
    setGitToken('');
    setIsPrivateRepo(false);
    setStoredTokenValid(true);
    setBranches([]);
    setSelectedBranch('');
    setLoadingBranches(false);
    setBranchError(null);
    setFilePatterns(DEFAULT_FILE_PATTERNS);
    setExcludePatterns('');
    setChunkSize(1000);
    setChunkOverlap(200);
    setMaxFileSizeKb(500);
    setOcrMode('disabled');
    setOcrProvider(null);
    setOcrVisionModel('');
    setVectorStoreType(existingVectorStoreType ?? editIndex?.vector_store_type ?? 'faiss');
    setGitHistoryDepth(1);
    setGitCloneTimeoutMinutes(5);
    setTimeoutManuallySet(false);
    setReindexIntervalHours(0);
    setReindexStartMinute(null);
    setReindexTimezone(null);
    setExclusionsApplied(false);
    setPatternsExpanded(false);
    setConfigureWebhookAfterCreate(false);
    setWebhookConfig(null);
    setRevealedWebhookSecret(null);
    setWebhookError(null);
    setWebhookRequestState('idle');
    webhookRequestRef.current += 1;

    if (clearTokenIndexName) {
      setIsLoading(true);
      setStatus({ type: 'info', message: 'Clearing fields and removing stored token...' });
      try {
        await api.updateIndexConfig(clearTokenIndexName, { git_token: null });
        setHasStoredToken(false);
        setStatus({ type: 'success', message: 'Fields cleared and stored token removed.' });
        onConfigSaved?.();
      } catch (err) {
        setStatus({
          type: 'error',
          message: `Fields cleared, but stored token could not be removed: ${err instanceof Error ? err.message : 'Request failed'}`,
        });
      } finally {
        setIsLoading(false);
      }
      return;
    }

    setHasStoredToken(false);
    setStatus({ type: 'success', message: 'Fields cleared.' });
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
      const trimmedToken = gitToken.trim();
      const updated = await api.updateIndexConfig(currentName, {
        git_branch: selectedBranch || undefined,
        git_token: trimmedToken || undefined,
        file_patterns: filePatterns
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        exclude_patterns: excludePatterns
          .split(',')
          .map((s) => s.trim())
          .filter(Boolean),
        chunk_size: chunkSize,
        chunk_overlap: chunkOverlap,
        max_file_size_kb: maxFileSizeKb,
        ocr_mode: ocrMode,
        ocr_provider: ocrMode === 'vision' ? ocrProvider : null,
        ocr_vision_model: ocrVisionModel || undefined,
        git_clone_timeout_minutes: gitCloneTimeoutMinutes,
        git_history_depth: gitHistoryDepth,
        reindex_interval_hours: reindexIntervalHours,
        reindex_start_minute:
          reindexIntervalHours > 0 ? (reindexStartMinute ?? defaultScheduleStartMinute()) : null,
        reindex_timezone:
          reindexIntervalHours > 0 ? (reindexTimezone ?? defaultScheduleTimezone()) : null,
      });
      // Reflect saved config locally so the UI shows persisted values
      const snap = updated?.config_snapshot;
      if (snap) {
        setGitHistoryDepth(snap.git_history_depth ?? gitHistoryDepth);
        setGitCloneTimeoutMinutes(snap.git_clone_timeout_minutes ?? gitCloneTimeoutMinutes);
        setReindexIntervalHours(snap.reindex_interval_hours ?? reindexIntervalHours);
        setReindexStartMinute(snap.reindex_start_minute ?? null);
        setReindexTimezone(snap.reindex_timezone ?? null);
        setFilePatterns(snap.file_patterns?.join(', ') || filePatterns);
        setExcludePatterns(snap.exclude_patterns?.join(', ') || excludePatterns);
        setChunkSize(snap.chunk_size ?? chunkSize);
        setChunkOverlap(snap.chunk_overlap ?? chunkOverlap);
        setMaxFileSizeKb(snap.max_file_size_kb ?? maxFileSizeKb);
        setOcrMode(snap.ocr_mode ?? ocrMode);
        setOcrProvider(snap.ocr_provider !== undefined ? snap.ocr_provider : ocrProvider);
        setOcrVisionModel(snap.ocr_vision_model ?? ocrVisionModel);
      }

      const wasRenamed = currentName !== editIndex.name;
      const savedMessage = wasRenamed
        ? `Index renamed to "${indexName}" and configuration saved. Click "Pull & Re-index" to apply changes.`
        : 'Configuration saved. Click "Pull & Re-index" to apply changes.';
      setStatus({ type: 'success', message: savedMessage });
      if (trimmedToken) {
        setGitToken('');
        setHasStoredToken(true);
        setStoredTokenValid(true);
      }
      onConfigSaved?.();
    } catch (err) {
      setStatus({
        type: 'error',
        message: `Error: ${err instanceof Error ? err.message : 'Save failed'}`,
      });
    } finally {
      setIsLoading(false);
    }
  };

  const currentWebhookIndexName = isEditMode
    ? (editIndex?.name ?? null)
    : (indexingJob?.name ?? null);
  const isWebhookLoading = webhookRequestState === 'loading';
  const shouldRenderWebhookSection = isWebhookLoading || !!webhookError || !!webhookConfig?.enabled;

  const handleEnableWebhook = useCallback(async (): Promise<boolean> => {
    if (!currentWebhookIndexName) {
      return false;
    }
    return applyWebhookMutation(() => api.enableIndexWebhook(currentWebhookIndexName));
  }, [applyWebhookMutation, currentWebhookIndexName]);

  const handleRotateWebhookSecret = useCallback(async () => {
    if (!currentWebhookIndexName) {
      return;
    }
    await applyWebhookMutation(() => api.rotateIndexWebhookSecret(currentWebhookIndexName));
  }, [applyWebhookMutation, currentWebhookIndexName]);

  const handleDisableWebhook = useCallback(async (): Promise<boolean> => {
    if (!currentWebhookIndexName || !webhookConfig) {
      return false;
    }
    return applyWebhookMutation(
      async () => {
        await api.disableIndexWebhook(currentWebhookIndexName);
      },
      {
        ...webhookConfig,
        enabled: false,
        paused: false,
        webhook_url: null,
        created_at: null,
      },
    );
  }, [applyWebhookMutation, currentWebhookIndexName, webhookConfig]);

  const handlePauseWebhook = useCallback(async () => {
    if (!currentWebhookIndexName) {
      return;
    }
    await applyWebhookMutation(() => api.pauseIndexWebhook(currentWebhookIndexName));
  }, [applyWebhookMutation, currentWebhookIndexName]);

  const handleResumeWebhook = useCallback(async () => {
    if (!currentWebhookIndexName) {
      return;
    }
    await applyWebhookMutation(() => api.resumeIndexWebhook(currentWebhookIndexName));
  }, [applyWebhookMutation, currentWebhookIndexName]);

  const handlePullNow = useCallback(async () => {
    if (!currentWebhookIndexName) {
      return;
    }
    setIsLoading(true);
    setWebhookError(null);
    setStatus({ type: 'info', message: 'Cloning repository...' });
    try {
      const job = await api.reindexFromGit(currentWebhookIndexName, gitToken.trim() || undefined);
      setIndexingJob(job);
      setWizardStep('indexing');
      notifiedJobCreatedRef.current = true;
      onJobCreated?.();
    } catch (err) {
      setStatus({
        type: 'error',
        message: `Error: ${err instanceof Error ? err.message : 'Request failed'}`,
      });
    } finally {
      setIsLoading(false);
    }
  }, [currentWebhookIndexName, gitToken, onJobCreated]);

  const handleWebhookDeliveryChange = useCallback(
    async (enabled: boolean): Promise<boolean> => {
      if (!enabled) {
        if (!currentWebhookIndexName) {
          setConfigureWebhookAfterCreate(false);
          return true;
        }
        const disabled = await handleDisableWebhook();
        if (disabled) {
          setConfigureWebhookAfterCreate(false);
          setReindexStartMinute(null);
          setReindexTimezone(null);
        }
        return disabled;
      }

      if (!currentWebhookIndexName) {
        setConfigureWebhookAfterCreate(true);
        setReindexIntervalHours(0);
        setReindexStartMinute(null);
        setReindexTimezone(null);
        return true;
      }

      const webhookEnabled = await handleEnableWebhook();
      if (webhookEnabled) {
        setConfigureWebhookAfterCreate(false);
        setReindexIntervalHours(0);
        setReindexStartMinute(null);
        setReindexTimezone(null);
      }
      return webhookEnabled;
    },
    [currentWebhookIndexName, handleDisableWebhook, handleEnableWebhook],
  );

  const webhookSection = shouldRenderWebhookSection ? (
    <div className="git-index-webhook-section">
      {isWebhookLoading && (
        <div className="git-index-webhook-loading">Loading webhook settings...</div>
      )}
      {webhookConfig?.enabled && (
        <GitWebhookSettings
          config={webhookConfig}
          revealedSecret={revealedWebhookSecret}
          disabled={
            isEditMode
              ? isLoading || webhookRequestState === 'mutating'
              : webhookRequestState === 'mutating'
          }
          onRotate={() => void handleRotateWebhookSecret()}
          onPause={() => void handlePauseWebhook()}
          onResume={() => void handleResumeWebhook()}
        />
      )}
      {webhookError && <div className="status-message error">{webhookError}</div>}
    </div>
  ) : null;

  // Edit mode: show simplified config editor
  if (isEditMode && wizardStep === 'input') {
    return (
      <div>
        <h4 style={{ marginBottom: '12px' }}>Edit Index Configuration</h4>
        <p className="field-help" style={{ marginBottom: '16px' }}>
          Update settings for the next time you click "Pull & Re-index". Changes will not take
          effect until you re-index.
        </p>

        <div
          style={{
            marginBottom: '16px',
            padding: '12px',
            background: 'var(--color-bg-tertiary)',
            borderRadius: '8px',
            fontSize: '13px',
          }}
        >
          <div>
            <strong>Source:</strong> {editIndex.source}
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '16px',
            marginBottom: '16px',
            alignItems: 'flex-start',
          }}
        >
          <div className="form-group" style={{ flex: '1 1 160px', minWidth: '160px', margin: 0 }}>
            <label>
              Branch
              {loadingBranches && (
                <span
                  style={{
                    marginLeft: '0.5rem',
                    color: 'var(--color-text-muted)',
                    fontSize: '0.85em',
                  }}
                >
                  (loading...)
                </span>
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
          <ReindexIntervalSelect
            value={reindexIntervalHours}
            onChange={(nextInterval) => {
              setReindexIntervalHours(nextInterval);
              setConfigureWebhookAfterCreate(false);
            }}
            webhookDeliveryEnabled={
              currentWebhookIndexName
                ? Boolean(webhookConfig?.enabled)
                : configureWebhookAfterCreate
            }
            onWebhookDeliveryChange={handleWebhookDeliveryChange}
            startMinute={reindexStartMinute}
            timezone={reindexTimezone}
            onStartMinuteChange={setReindexStartMinute}
            onTimezoneChange={setReindexTimezone}
            disabled={isLoading || webhookRequestState === 'mutating'}
            style={{ flex: '1 1 300px' }}
            action={
              currentWebhookIndexName ? (
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    void handlePullNow();
                  }}
                  disabled={
                    isLoading ||
                    webhookRequestState === 'mutating' ||
                    (indexingJob !== null && !isIndexJobTerminal(indexingJob))
                  }
                >
                  Pull now
                </button>
              ) : undefined
            }
          />
        </div>

        {webhookSection}

        <OcrVectorStoreFields
          isLoading={isLoading}
          ocrMode={ocrMode}
          setOcrMode={setOcrMode}
          ocrProvider={ocrProvider}
          setOcrProvider={setOcrProvider}
          ocrVisionModel={ocrVisionModel}
          setOcrVisionModel={setOcrVisionModel}
          visionOcrAvailable={visionOcrAvailable}
          defaultOcrProviderLabel={defaultOcrProviderLabel}
          defaultOcrVisionModelLabel={defaultOcrVisionModelLabel}
          vectorStoreType={vectorStoreType}
          setVectorStoreType={setVectorStoreType}
          vectorStoreDisabled={true}
        />

        <GitHistoryDepthAndAdvancedOptions
          gitHistoryDepth={gitHistoryDepth}
          onGitHistoryDepthChange={setGitHistoryDepth}
          gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
          onGitCloneTimeoutMinutesChange={setGitCloneTimeoutMinutes}
          filePatterns={filePatterns}
          onFilePattersChange={setFilePatterns}
          excludePatterns={excludePatterns}
          onExcludePattersChange={setExcludePatterns}
          chunkSize={chunkSize}
          onChunkSizeChange={setChunkSize}
          chunkOverlap={chunkOverlap}
          onChunkOverlapChange={setChunkOverlap}
          maxFileSizeKb={maxFileSizeKb}
          onMaxFileSizeKbChange={setMaxFileSizeKb}
          isLoading={isLoading}
          onTimeoutManuallySet={setTimeoutManuallySet}
        />

        <div className="wizard-actions" style={{ marginTop: '16px' }}>
          {onCancel && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleCancel}
              disabled={isLoading}
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleClearFields}
            disabled={isLoading}
          >
            Clear Fields
          </button>
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
        <div className="form-group" style={{ marginBottom: '16px' }}>
          <label>Git URL *</label>
          <input
            type="text"
            value={gitUrl}
            onChange={(e) => setGitUrl(e.target.value)}
            placeholder="https://github.com/user/repo.git or https://your-git-server.com/user/repo.git"
            disabled={isLoading}
          />
        </div>

        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '16px',
            marginBottom: '16px',
            alignItems: 'flex-start',
          }}
        >
          <div className="form-group" style={{ flex: '1 1 160px', minWidth: '160px', margin: 0 }}>
            <label>
              Branch
              {loadingBranches && (
                <span
                  style={{
                    marginLeft: '0.5rem',
                    color: 'var(--color-text-muted)',
                    fontSize: '0.85em',
                  }}
                >
                  (loading...)
                </span>
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
              <small
                style={{
                  color: 'var(--color-error)',
                  fontSize: '0.85em',
                  display: 'block',
                  marginTop: '0.25rem',
                }}
              >
                {branchError}
              </small>
            )}
          </div>
          <ReindexIntervalSelect
            value={reindexIntervalHours}
            onChange={(nextInterval) => {
              setReindexIntervalHours(nextInterval);
              setConfigureWebhookAfterCreate(false);
            }}
            webhookDeliveryEnabled={
              currentWebhookIndexName
                ? Boolean(webhookConfig?.enabled)
                : configureWebhookAfterCreate
            }
            onWebhookDeliveryChange={handleWebhookDeliveryChange}
            startMinute={reindexStartMinute}
            timezone={reindexTimezone}
            onStartMinuteChange={setReindexStartMinute}
            onTimezoneChange={setReindexTimezone}
            disabled={isLoading || webhookRequestState === 'mutating'}
            style={{ flex: '1 1 300px' }}
          />
        </div>

        <p className="field-help" style={{ marginBottom: '16px' }}>
          Index name will be derived from the repository name. Click "Analyze" to preview the index
          before creating.
        </p>

        <div className="form-group" style={{ marginBottom: isPrivateRepo ? '0.5rem' : undefined }}>
          <label
            style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer' }}
          >
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
            {checkingVisibility && (
              <span style={{ color: 'var(--color-text-muted)', fontSize: '0.85em' }}>
                (checking...)
              </span>
            )}
          </label>
        </div>

        {isPrivateRepo && (
          <div
            className="form-group"
            style={{
              marginLeft: '1.5rem',
              borderLeft: '2px solid var(--color-border-strong)',
              paddingLeft: '1rem',
              marginBottom: '1rem',
            }}
          >
            {/* Show stored token status in edit mode */}
            {isEditMode && hasStoredToken && storedTokenValid && (
              <div
                style={{
                  marginBottom: '12px',
                  padding: '12px',
                  background: 'var(--color-success-light)',
                  borderRadius: '8px',
                  border: '1px solid var(--color-success-border)',
                }}
              >
                <span style={{ color: 'var(--color-success)' }}>
                  Token stored - will use existing credentials.
                </span>
                <button
                  type="button"
                  onClick={() => setStoredTokenValid(false)}
                  style={{
                    marginLeft: '12px',
                    padding: '4px 8px',
                    fontSize: '12px',
                    background: 'transparent',
                    border: '1px solid var(--color-border-strong)',
                    borderRadius: '4px',
                    color: 'var(--color-text-muted)',
                    cursor: 'pointer',
                  }}
                >
                  Update Token
                </button>
              </div>
            )}

            {/* Show warning if stored token is invalid */}
            {isEditMode && hasStoredToken && !storedTokenValid && (
              <div
                style={{
                  marginBottom: '12px',
                  padding: '12px',
                  background: 'var(--color-warning-light)',
                  borderRadius: '8px',
                  border: '1px solid var(--color-warning-border)',
                }}
              >
                <span style={{ color: 'var(--color-warning)' }}>
                  Stored token no longer works - please provide a new token.
                </span>
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
                <small
                  style={{
                    color: 'var(--color-text-muted)',
                    fontSize: '0.85em',
                    display: 'block',
                    marginTop: '0.25rem',
                  }}
                >
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
          ocrProvider={ocrProvider}
          setOcrProvider={setOcrProvider}
          ocrVisionModel={ocrVisionModel}
          setOcrVisionModel={setOcrVisionModel}
          visionOcrAvailable={visionOcrAvailable}
          defaultOcrProviderLabel={defaultOcrProviderLabel}
          defaultOcrVisionModelLabel={defaultOcrVisionModelLabel}
          vectorStoreType={vectorStoreType}
          setVectorStoreType={setVectorStoreType}
          vectorStoreDisabled={!!existingVectorStoreType}
        />

        <GitHistoryDepthAndAdvancedOptions
          gitHistoryDepth={gitHistoryDepth}
          onGitHistoryDepthChange={setGitHistoryDepth}
          gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
          onGitCloneTimeoutMinutesChange={setGitCloneTimeoutMinutes}
          filePatterns={filePatterns}
          onFilePattersChange={setFilePatterns}
          excludePatterns={excludePatterns}
          onExcludePattersChange={setExcludePatterns}
          chunkSize={chunkSize}
          onChunkSizeChange={setChunkSize}
          chunkOverlap={chunkOverlap}
          onChunkOverlapChange={setChunkOverlap}
          maxFileSizeKb={maxFileSizeKb}
          onMaxFileSizeKbChange={setMaxFileSizeKb}
          isLoading={isLoading}
          onTimeoutManuallySet={setTimeoutManuallySet}
        />

        <div className="wizard-actions">
          {onCancel && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleCancel}
              disabled={isLoading}
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleClearFields}
            disabled={isLoading}
          >
            Clear Fields
          </button>
          <button
            type="button"
            className="btn"
            onClick={handleAnalyze}
            disabled={isLoading || !gitUrl}
          >
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

        <WarningsBanner warnings={analysisResult.warnings} />

        <AnalysisStats result={analysisResult} onNavigateToSettings={onNavigateToSettings} />

        {analysisResult.file_type_stats.length > 0 && (
          <div style={{ marginBottom: '16px' }}>
            <h5 style={{ marginBottom: '8px' }}>File Types (by estimated chunks)</h5>
            <div
              style={{
                maxHeight: '200px',
                overflowY: 'auto',
                background: 'var(--color-bg-tertiary)',
                borderRadius: '8px',
                padding: '8px',
              }}
            >
              <FileTypeStatsTable stats={analysisResult.file_type_stats} maxRows={15} />
            </div>
          </div>
        )}

        <SuggestedExclusionsBanner
          exclusions={analysisResult.suggested_exclusions}
          applied={exclusionsApplied}
          onApply={applySuggestedExclusions}
        />

        <OcrVectorStoreFields
          isLoading={isLoading}
          ocrMode={ocrMode}
          setOcrMode={setOcrMode}
          ocrProvider={ocrProvider}
          setOcrProvider={setOcrProvider}
          ocrVisionModel={ocrVisionModel}
          setOcrVisionModel={setOcrVisionModel}
          visionOcrAvailable={visionOcrAvailable}
          defaultOcrProviderLabel={defaultOcrProviderLabel}
          defaultOcrVisionModelLabel={defaultOcrVisionModelLabel}
          vectorStoreType={vectorStoreType}
          setVectorStoreType={setVectorStoreType}
          vectorStoreDisabled={!!existingVectorStoreType}
        />

        {/* Auto Re-index Interval - visible outside Advanced Options */}
        <ReindexIntervalSelect
          value={reindexIntervalHours}
          onChange={(nextInterval) => {
            setReindexIntervalHours(nextInterval);
            setConfigureWebhookAfterCreate(false);
          }}
          webhookDeliveryEnabled={
            currentWebhookIndexName ? Boolean(webhookConfig?.enabled) : configureWebhookAfterCreate
          }
          onWebhookDeliveryChange={handleWebhookDeliveryChange}
          startMinute={reindexStartMinute}
          timezone={reindexTimezone}
          onStartMinuteChange={setReindexStartMinute}
          onTimezoneChange={setReindexTimezone}
          disabled={isLoading || webhookRequestState === 'mutating'}
          style={{ marginBottom: '16px', maxWidth: '300px' }}
        />

        <GitHistoryDepthAndAdvancedOptions
          gitHistoryDepth={gitHistoryDepth}
          onGitHistoryDepthChange={setGitHistoryDepth}
          gitCloneTimeoutMinutes={gitCloneTimeoutMinutes}
          onGitCloneTimeoutMinutesChange={setGitCloneTimeoutMinutes}
          filePatterns={filePatterns}
          onFilePattersChange={setFilePatterns}
          excludePatterns={excludePatterns}
          onExcludePattersChange={setExcludePatterns}
          chunkSize={chunkSize}
          onChunkSizeChange={setChunkSize}
          chunkOverlap={chunkOverlap}
          onChunkOverlapChange={setChunkOverlap}
          maxFileSizeKb={maxFileSizeKb}
          onMaxFileSizeKbChange={setMaxFileSizeKb}
          isLoading={isLoading}
          onTimeoutManuallySet={setTimeoutManuallySet}
          commitHistory={analysisResult.commit_history}
          patternsExpanded={patternsExpanded}
          onPatternsExpandedChange={setPatternsExpanded}
          advancedOptionsFooterButton={
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleReanalyze}
              disabled={isLoading}
              style={{ marginTop: '8px' }}
            >
              {isLoading ? 'Re-analyzing...' : 'Re-analyze'}
            </button>
          }
        />

        <div className="wizard-actions">
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleBack}
            disabled={isLoading}
          >
            Back
          </button>
          {onCancel && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={handleCancel}
              disabled={isLoading}
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary"
            onClick={handleClearFields}
            disabled={isLoading}
          >
            Clear Fields
          </button>
          <button type="button" className="btn" onClick={handleStartIndexing} disabled={isLoading}>
            {isLoading ? 'Starting...' : 'Start Indexing'}
          </button>
        </div>

        {status.type && <div className={`status-message ${status.type}`}>{status.message}</div>}
      </div>
    );
  }

  const jobDone = isIndexJobTerminal(indexingJob);
  const jobFailed = indexingJob?.status === 'failed';
  const displayPercent = indexingJob ? getIndexJobDisplayPercent(indexingJob) : 0;
  const progressSummary = indexingJob
    ? indexingJob.phase === 'loading' || indexingJob.phase === 'scanning'
      ? `${indexingJob.processed_files}/${indexingJob.total_files} files`
      : indexingJob.phase === 'chunking'
        ? `${indexingJob.processed_chunks}/${indexingJob.total_chunks} documents`
        : indexingJob.phase !== 'cloning' && indexingJob.total_files > 0
          ? `${indexingJob.processed_files}/${indexingJob.total_files} files${indexingJob.total_chunks > 0 ? ` · ${indexingJob.processed_chunks}/${indexingJob.total_chunks} chunks` : ''}`
          : null
    : null;

  return (
    <div style={{ padding: '24px' }}>
      {!indexingJob ? (
        <div style={{ textAlign: 'center', padding: '16px' }}>
          <div style={{ fontSize: '1.2rem', marginBottom: '16px' }}>Starting indexing job...</div>
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '16px' }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 600 }}>
            {jobFailed
              ? 'Indexing failed'
              : jobDone
                ? 'Indexing complete'
                : formatIndexJobPhase(indexingJob)}
          </div>

          {!jobFailed && (
            <div>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '0.85rem',
                  marginBottom: '6px',
                }}
              >
                <span>
                  {indexingJob.phase === 'cloning'
                    ? 'Cloning repository'
                    : formatIndexJobPhase(indexingJob)}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>{displayPercent}%</span>
              </div>
              <div
                style={{
                  height: 8,
                  borderRadius: 999,
                  background: 'var(--color-bg-tertiary)',
                  overflow: 'hidden',
                }}
              >
                <div
                  style={{
                    height: '100%',
                    width: `${displayPercent}%`,
                    background: 'var(--color-accent)',
                    transition: 'width 200ms ease',
                  }}
                />
              </div>
              {progressSummary && (
                <small
                  style={{
                    color: 'var(--color-text-muted)',
                    fontSize: '0.8rem',
                    display: 'block',
                    marginTop: '6px',
                  }}
                >
                  {progressSummary}
                </small>
              )}
            </div>
          )}

          {jobFailed && indexingJob.error_message && (
            <div className="status-message error">{indexingJob.error_message}</div>
          )}

          {webhookSection}

          {jobDone && (
            <div className="wizard-actions">
              <button type="button" className="btn" onClick={handleCancel}>
                Done
              </button>
            </div>
          )}
        </div>
      )}

      {status.type === 'error' && (
        <div className={`status-message ${status.type}`}>{status.message}</div>
      )}
    </div>
  );
}
