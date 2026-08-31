import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { GitWebhookConfig, GitWebhookEnableResponse, IndexInfo } from '@/types';
import { deferred } from '@/testHelpers/deferred';

import { GitIndexWizard } from './GitIndexWizard';

const apiMock = vi.hoisted(() => ({
  getSettings: vi.fn(),
  checkRepoVisibility: vi.fn(),
  fetchBranches: vi.fn(),
  analyzeRepository: vi.fn(),
  indexFromGit: vi.fn(),
  getJob: vi.fn(),
  cancelJob: vi.fn(),
  getIndexWebhook: vi.fn(),
  enableIndexWebhook: vi.fn(),
  pauseIndexWebhook: vi.fn(),
  resumeIndexWebhook: vi.fn(),
  rotateIndexWebhookSecret: vi.fn(),
  disableIndexWebhook: vi.fn(),
  reindexFromGit: vi.fn(),
  renameIndex: vi.fn(),
  updateIndexDescription: vi.fn(),
  updateIndexConfig: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('./AnalysisStats', () => ({
  AnalysisStats: () => <div>analysis stats</div>,
}));

vi.mock('./IndexConfigFields', () => ({
  IndexConfigFields: () => <div>config fields</div>,
}));

vi.mock('./OcrVectorStoreFields', () => ({
  OcrVectorStoreFields: () => <div>ocr fields</div>,
  OCR_PROVIDER_LABELS: {},
}));

vi.mock('./FileTypeStatsTable', () => ({
  FileTypeStatsTable: () => <div>file type stats</div>,
}));

vi.mock('./SuggestedExclusionsBanner', () => ({
  SuggestedExclusionsBanner: () => null,
}));

vi.mock('./WarningsBanner', () => ({
  WarningsBanner: () => null,
}));

vi.mock('./GitWebhookSettings', () => ({
  GitWebhookSettings: ({
    config,
    revealedSecret,
    disabled,
    onRotate,
    onPause,
    onResume,
  }: {
    config: GitWebhookConfig;
    revealedSecret: string | null;
    disabled: boolean;
    onRotate: () => void;
    onPause: () => void;
    onResume: () => void;
  }) => (
    <div data-testid="git-webhook-settings-mock">
      <div>{`webhook:${config.provider}:${config.branch}:${config.enabled ? 'enabled' : 'disabled'}:${config.paused ? 'paused' : 'active'}`}</div>
      {revealedSecret && <div>{revealedSecret}</div>}
      {config.enabled && (
        <>
          <button type="button" onClick={onRotate} disabled={disabled}>
            Rotate secret
          </button>
          {config.paused ? (
            <button type="button" onClick={onResume} disabled={disabled}>
              Resume webhook
            </button>
          ) : (
            <button type="button" onClick={onPause} disabled={disabled}>
              Pause webhook
            </button>
          )}
        </>
      )}
    </div>
  ),
}));

const analysisResult = {
  estimated_documents: 12,
  estimated_chunks: 24,
  warnings: [],
  file_type_stats: [],
  suggested_exclusions: [],
  commit_history: null,
};

const startingJob = {
  id: 'job-1',
  name: 'repo',
  status: 'processing',
  phase: 'cloning',
  progress_percent: 0,
  clone_progress: 0.1,
  total_files: 10,
  processed_files: 0,
  total_chunks: 0,
  processed_chunks: 0,
  error_message: null,
  created_at: '2026-07-15T00:00:00Z',
  started_at: '2026-07-15T00:00:01Z',
  completed_at: null,
};

const existingGitIndex: IndexInfo = {
  name: 'repo',
  display_name: 'Repo',
  path: '/indexes/repo',
  size_mb: 1,
  document_count: 10,
  chunk_count: 20,
  description: '',
  enabled: true,
  search_weight: 1,
  source_type: 'git',
  source: 'https://github.com/example/repo.git',
  git_branch: 'main',
  has_stored_token: false,
  config_snapshot: {
    file_patterns: ['**/*'],
    exclude_patterns: [],
    chunk_size: 1000,
    chunk_overlap: 200,
    max_file_size_kb: 500,
    ocr_mode: 'disabled',
    ocr_provider: null,
    ocr_vision_model: '',
    git_clone_timeout_minutes: 5,
    git_history_depth: 1,
    reindex_interval_hours: 0,
    reindex_start_minute: null,
    reindex_timezone: null,
  },
  created_at: '2026-07-15T00:00:00Z',
  last_modified: '2026-07-15T00:00:00Z',
  git_repo_size_mb: 1,
  has_git_history: true,
  vector_store_type: 'faiss',
};

const secondGitIndex: IndexInfo = {
  ...existingGitIndex,
  name: 'repo-two',
  display_name: 'Repo Two',
  source: 'https://github.com/example/repo-two.git',
};

const disabledWebhookConfig: GitWebhookConfig = {
  enabled: false,
  paused: false,
  webhook_url: null,
  provider: 'github',
  branch: 'main',
  created_at: null,
};

const enabledWebhookConfig: GitWebhookConfig = {
  enabled: true,
  paused: false,
  webhook_url: 'https://ragtime.example/webhooks/git/repo',
  provider: 'github',
  branch: 'main',
  created_at: '2026-07-16T12:00:00Z',
};

const enabledWebhookWithSecret: GitWebhookEnableResponse = {
  ...enabledWebhookConfig,
  secret: 'secret-once',
};

async function completeAnalysis() {
  fireEvent.change(
    screen.getByPlaceholderText(
      'https://github.com/user/repo.git or https://your-git-server.com/user/repo.git',
    ),
    { target: { value: 'https://github.com/example/repo.git' } },
  );
  fireEvent.click(screen.getByRole('button', { name: 'Analyze Repository' }));
  await waitFor(() => expect(screen.getByRole('button', { name: 'Start Indexing' })).toBeDefined());
}

beforeEach(() => {
  apiMock.getSettings.mockResolvedValue({ settings: {} });
  apiMock.checkRepoVisibility.mockResolvedValue({
    visibility: 'public',
    has_stored_token: false,
    needs_token: false,
    message: '',
  });
  apiMock.fetchBranches.mockResolvedValue({ branches: ['main'] });
  apiMock.analyzeRepository.mockResolvedValue(analysisResult);
  apiMock.indexFromGit.mockResolvedValue(startingJob);
  apiMock.getJob.mockResolvedValue({
    ...startingJob,
    clone_progress: 0.2,
  });
  apiMock.getIndexWebhook.mockResolvedValue(disabledWebhookConfig);
  apiMock.enableIndexWebhook.mockResolvedValue(enabledWebhookWithSecret);
  apiMock.pauseIndexWebhook.mockResolvedValue({ ...enabledWebhookConfig, paused: true });
  apiMock.resumeIndexWebhook.mockResolvedValue(enabledWebhookConfig);
  apiMock.rotateIndexWebhookSecret.mockResolvedValue({
    ...enabledWebhookConfig,
    secret: 'rotated-secret',
  });
  apiMock.disableIndexWebhook.mockResolvedValue(undefined);
  apiMock.reindexFromGit.mockResolvedValue(startingJob);
  apiMock.renameIndex.mockResolvedValue({ new_name: 'repo', display_name: 'Repo' });
  apiMock.updateIndexDescription.mockResolvedValue(undefined);
  apiMock.updateIndexConfig.mockResolvedValue({
    config_snapshot: existingGitIndex.config_snapshot,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.useRealTimers();
});

describe('GitIndexWizard', () => {
  it('notifies immediately after creating the git job and starts polling it', async () => {
    const onJobCreated = vi.fn();

    render(<GitIndexWizard onJobCreated={onJobCreated} />);

    fireEvent.change(
      screen.getByPlaceholderText(
        'https://github.com/user/repo.git or https://your-git-server.com/user/repo.git',
      ),
      { target: { value: 'https://github.com/example/repo.git' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Repository' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Start Indexing' })).toBeDefined(),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await waitFor(() => {
      expect(onJobCreated).toHaveBeenCalledTimes(1);
      expect(apiMock.getJob).toHaveBeenCalledWith('job-1');
    });
  });

  it('polls at the configured interval without churn and does not cancel the backend job on unmount', async () => {
    const processingJob = {
      ...startingJob,
      clone_progress: 0.3,
    };
    apiMock.getJob.mockResolvedValue(processingJob);

    const onJobCreated = vi.fn();

    const { unmount } = render(<GitIndexWizard onJobCreated={onJobCreated} />);

    fireEvent.change(
      screen.getByPlaceholderText(
        'https://github.com/user/repo.git or https://your-git-server.com/user/repo.git',
      ),
      { target: { value: 'https://github.com/example/repo.git' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Repository' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Start Indexing' })).toBeDefined(),
    );

    vi.useFakeTimers();
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await act(async () => {
      await Promise.resolve();
    });

    expect(apiMock.getJob).toHaveBeenCalledTimes(1);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(3100);
    });

    expect(apiMock.getJob.mock.calls.length).toBeLessThanOrEqual(4);

    unmount();

    expect(apiMock.cancelJob).not.toHaveBeenCalled();
    expect(onJobCreated).toHaveBeenCalledTimes(1);
  });

  it('hides informational indexing banners and labels chunking progress as documents', async () => {
    apiMock.indexFromGit.mockResolvedValue({
      ...startingJob,
      phase: 'chunking',
      progress_percent: 40,
      total_files: 10,
      processed_files: 10,
      total_chunks: 24,
      processed_chunks: 3,
    });

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    fireEvent.change(
      screen.getByPlaceholderText(
        'https://github.com/user/repo.git or https://your-git-server.com/user/repo.git',
      ),
      { target: { value: 'https://github.com/example/repo.git' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Repository' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Start Indexing' })).toBeDefined(),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await waitFor(() => {
      expect(screen.queryByText('Cloning repository...')).toBeNull();
      expect(screen.getByText('3/24 documents')).not.toBeNull();
      expect(screen.queryByText('3/10 files')).toBeNull();
    });
  });

  it('shows the persisted scanning phase label from the job payload', async () => {
    apiMock.indexFromGit.mockResolvedValue({
      ...startingJob,
      phase: 'scanning',
      progress_percent: 12,
      total_files: 0,
      processed_files: 0,
      total_chunks: 0,
      processed_chunks: 0,
    });

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    fireEvent.change(
      screen.getByPlaceholderText(
        'https://github.com/user/repo.git or https://your-git-server.com/user/repo.git',
      ),
      { target: { value: 'https://github.com/example/repo.git' } },
    );
    fireEvent.click(screen.getByRole('button', { name: 'Analyze Repository' }));
    await waitFor(() =>
      expect(screen.getByRole('button', { name: 'Start Indexing' })).toBeDefined(),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await waitFor(() => {
      expect(screen.getAllByText('Scanning files').length).toBeGreaterThan(0);
      expect(screen.getByText('12%')).toBeTruthy();
    });
  });

  it('loads webhook configuration in edit mode and reveals a new secret after enable', async () => {
    const user = userEvent.setup();
    apiMock.getIndexWebhook.mockResolvedValue(disabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    await waitFor(() => {
      expect(apiMock.getIndexWebhook).toHaveBeenCalledWith('repo');
    });

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');

    await waitFor(() => {
      expect(apiMock.enableIndexWebhook).toHaveBeenCalledWith('repo');
    });
    expect(await screen.findByText('secret-once')).toBeTruthy();
  });

  it('rotates the webhook secret in edit mode and refreshes deliveries', async () => {
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Rotate secret' }));

    await waitFor(() => {
      expect(apiMock.rotateIndexWebhookSecret).toHaveBeenCalledWith('repo');
    });
    expect(await screen.findByText('rotated-secret')).toBeTruthy();
  });

  it('pauses and resumes the webhook in edit mode without leaving webhook cadence', async () => {
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Pause webhook' }));

    await waitFor(() => {
      expect(apiMock.pauseIndexWebhook).toHaveBeenCalledWith('repo');
    });
    await waitFor(() => {
      expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe(
        'webhook',
      );
    });
    expect(screen.getByText('webhook:github:main:enabled:paused')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Resume webhook' }));
    await waitFor(() => {
      expect(apiMock.resumeIndexWebhook).toHaveBeenCalledWith('repo');
    });
    expect(screen.getByText('webhook:github:main:enabled:active')).toBeTruthy();
  });

  it('shows pull now beside cadence for manual, scheduled, active, and paused webhook states', async () => {
    render(<GitIndexWizard editIndex={existingGitIndex} />);

    const scheduledWebhookIndex: IndexInfo = {
      ...existingGitIndex,
      config_snapshot: {
        ...existingGitIndex.config_snapshot!,
        reindex_interval_hours: 24,
        reindex_start_minute: 120,
        reindex_timezone: 'UTC',
      },
    };
    const { rerender, container } = render(<GitIndexWizard editIndex={existingGitIndex} />);

    await within(container).findByRole('button', { name: 'Pull now' });
    expect(within(container).getByRole('button', { name: 'Pull now' })).toBeTruthy();

    rerender(<GitIndexWizard editIndex={scheduledWebhookIndex} />);
    expect(await within(container).findByRole('button', { name: 'Pull now' })).toBeTruthy();

    cleanup();
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);
    const activeRender = render(<GitIndexWizard editIndex={existingGitIndex} />);
    expect(
      await within(activeRender.container).findByRole('button', { name: 'Pull now' }),
    ).toBeTruthy();

    cleanup();
    apiMock.getIndexWebhook.mockResolvedValue({ ...enabledWebhookConfig, paused: true });
    const pausedRender = render(<GitIndexWizard editIndex={existingGitIndex} />);
    expect(
      await within(pausedRender.container).findByRole('button', { name: 'Pull now' }),
    ).toBeTruthy();
  });

  it('pulls now by reusing git reindex and moves to indexing progress', async () => {
    const onJobCreated = vi.fn();
    render(<GitIndexWizard editIndex={existingGitIndex} onJobCreated={onJobCreated} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Pull now' }));

    await waitFor(() => {
      expect(apiMock.reindexFromGit).toHaveBeenCalledWith('repo', undefined);
    });
    expect(onJobCreated).toHaveBeenCalledTimes(1);
    expect((await screen.findAllByText('Cloning repository')).length).toBeGreaterThan(0);
  });

  it('invalidates an in-flight webhook mutation when edit mode switches to a different index', async () => {
    const user = userEvent.setup();
    const enableWebhook = deferred<GitWebhookEnableResponse>();
    apiMock.getIndexWebhook
      .mockResolvedValueOnce(disabledWebhookConfig)
      .mockResolvedValueOnce({ ...disabledWebhookConfig, branch: 'develop' });
    apiMock.enableIndexWebhook.mockImplementationOnce(() => enableWebhook.promise);

    const { rerender } = render(<GitIndexWizard editIndex={existingGitIndex} />);

    await screen.findByRole('combobox', { name: 'Auto Re-index Interval' });
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');

    rerender(<GitIndexWizard editIndex={secondGitIndex} />);
    await waitFor(() => expect(apiMock.getIndexWebhook).toHaveBeenCalledWith('repo-two'));

    enableWebhook.resolve(enabledWebhookWithSecret);
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.queryByText('secret-once')).toBeNull();
    expect(screen.queryByText('webhook:github:main:enabled')).toBeNull();
    expect(screen.queryByTestId('git-webhook-settings-mock')).toBeNull();
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe('0');
  });

  it('shows an initial webhook load error even when no config is available yet', async () => {
    apiMock.getIndexWebhook.mockRejectedValue(new Error('load failed'));

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    expect(await screen.findByText(/failed to load webhook settings: load failed/i)).toBeTruthy();
  });

  it('saves manual cadence after loading an enabled webhook over a persisted anchored schedule', async () => {
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);

    const scheduledWebhookIndex: IndexInfo = {
      ...existingGitIndex,
      config_snapshot: {
        ...existingGitIndex.config_snapshot!,
        reindex_interval_hours: 24,
        reindex_start_minute: 180,
        reindex_timezone: 'America/Denver',
      },
    };

    const user = userEvent.setup();
    render(<GitIndexWizard editIndex={scheduledWebhookIndex} />);

    await screen.findByText('webhook:github:main:enabled:active');
    await user.click(screen.getByRole('button', { name: 'Save Configuration' }));

    await waitFor(() => {
      expect(apiMock.updateIndexConfig).toHaveBeenCalledWith(
        'repo',
        expect.objectContaining({
          reindex_interval_hours: 0,
          reindex_start_minute: null,
          reindex_timezone: null,
        }),
      );
    });
  });

  it('keeps webhook delivery selected in the create input step before analysis', async () => {
    const user = userEvent.setup();

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    fireEvent.change(
      screen.getByPlaceholderText(
        'https://github.com/user/repo.git or https://your-git-server.com/user/repo.git',
      ),
      { target: { value: 'https://github.com/example/repo.git' } },
    );

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');

    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe(
      'webhook',
    );
  });

  it('selects webhook delivery and enables it after creating a new index', async () => {
    const user = userEvent.setup();
    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await waitFor(() => expect(apiMock.indexFromGit).toHaveBeenCalled());
    expect(apiMock.indexFromGit.mock.calls[0][0].config.reindex_interval_hours).toBe(0);
    expect(apiMock.indexFromGit.mock.calls[0][0].config.reindex_start_minute).toBeNull();
    expect(apiMock.enableIndexWebhook).toHaveBeenCalledWith('repo');
    expect(await screen.findByText('secret-once')).toBeTruthy();
  });

  it('keeps the indexing job active when webhook setup fails after creation', async () => {
    const user = userEvent.setup();
    apiMock.enableIndexWebhook.mockRejectedValue(new Error('webhook failed'));

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();

    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await waitFor(() => {
      expect(apiMock.indexFromGit).toHaveBeenCalled();
      expect(apiMock.enableIndexWebhook).toHaveBeenCalledWith('repo');
    });
    expect(await screen.findByText(/webhook failed/i)).toBeTruthy();
    expect(screen.queryByText('Indexing failed')).toBeNull();
    expect(screen.getAllByText('Cloning repository').length).toBeGreaterThan(0);
  });

  it('shows create-mode webhook setup errors without hiding a successful indexing state', async () => {
    const user = userEvent.setup();
    apiMock.enableIndexWebhook.mockRejectedValue(new Error('setup failed'));

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), 'webhook');
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    expect(await screen.findByText(/webhook setup failed: setup failed/i)).toBeTruthy();
    expect(screen.queryByTestId('git-webhook-settings-mock')).toBeNull();
    expect(screen.queryByText('Indexing failed')).toBeNull();
    expect(screen.getAllByText('Cloning repository').length).toBeGreaterThan(0);
  });

  it('confirms and disables an existing index webhook before selecting manual re-indexing', async () => {
    const user = userEvent.setup();
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    await screen.findByRole('combobox', { name: 'Auto Re-index Interval' });
    await user.selectOptions(screen.getByLabelText('Auto Re-index Interval'), '0');
    await user.click(screen.getByRole('button', { name: 'Disable webhook and continue' }));

    await waitFor(() => expect(apiMock.disableIndexWebhook).toHaveBeenCalledWith('repo'));
    expect((screen.getByLabelText('Auto Re-index Interval') as HTMLSelectElement).value).toBe('0');
  });
});
