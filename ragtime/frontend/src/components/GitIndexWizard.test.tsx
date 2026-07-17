import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type {
  GitWebhookConfig,
  GitWebhookDelivery,
  GitWebhookEnableResponse,
  IndexInfo,
} from '@/types';

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
  rotateIndexWebhookSecret: vi.fn(),
  disableIndexWebhook: vi.fn(),
  listIndexWebhookDeliveries: vi.fn(),
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

vi.mock('./ReindexIntervalSelect', () => ({
  ReindexIntervalSelect: () => <div>reindex interval</div>,
}));

vi.mock('./GitWebhookSettings', () => ({
  GitWebhookSettings: ({
    config,
    revealedSecret,
    deliveries,
    disabled,
    onEnable,
    onRotate,
    onDisable,
  }: {
    config: GitWebhookConfig;
    revealedSecret: string | null;
    deliveries: GitWebhookDelivery[];
    disabled: boolean;
    onEnable: () => void;
    onRotate: () => void;
    onDisable: () => void;
  }) => (
    <div data-testid="git-webhook-settings-mock">
      <div>{`webhook:${config.provider}:${config.branch}:${config.enabled ? 'enabled' : 'disabled'}`}</div>
      <div>{`deliveries:${deliveries.length}`}</div>
      {deliveries[0] && <div>{`delivery-id:${deliveries[0].id}`}</div>}
      {revealedSecret && <div>{revealedSecret}</div>}
      {config.enabled ? (
        <>
          <button type="button" onClick={onRotate} disabled={disabled}>
            Rotate secret
          </button>
          <button type="button" onClick={onDisable} disabled={disabled}>
            Disable webhook
          </button>
        </>
      ) : (
        <button type="button" onClick={onEnable} disabled={disabled}>
          Enable push webhook
        </button>
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
  webhook_url: null,
  provider: 'github',
  branch: 'main',
  created_at: null,
};

const enabledWebhookConfig: GitWebhookConfig = {
  enabled: true,
  webhook_url: 'https://ragtime.example/webhooks/git/repo',
  provider: 'github',
  branch: 'main',
  created_at: '2026-07-16T12:00:00Z',
};

const enabledWebhookWithSecret: GitWebhookEnableResponse = {
  ...enabledWebhookConfig,
  secret: 'secret-once',
};

const webhookDeliveries: GitWebhookDelivery[] = [
  {
    id: 'delivery-1',
    event_name: 'push',
    branch: 'main',
    head_commit: 'abc123',
    status: 'completed',
    message: 'done',
    received_at: '2026-07-16T12:00:00Z',
    started_at: '2026-07-16T12:00:01Z',
    completed_at: '2026-07-16T12:00:02Z',
  },
];

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

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
  apiMock.rotateIndexWebhookSecret.mockResolvedValue({
    ...enabledWebhookConfig,
    secret: 'rotated-secret',
  });
  apiMock.disableIndexWebhook.mockResolvedValue(undefined);
  apiMock.listIndexWebhookDeliveries.mockResolvedValue(webhookDeliveries);
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
    apiMock.getIndexWebhook.mockResolvedValue(disabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    await waitFor(() => {
      expect(apiMock.getIndexWebhook).toHaveBeenCalledWith('repo');
      expect(apiMock.listIndexWebhookDeliveries).toHaveBeenCalledWith('repo', 10);
    });

    fireEvent.click(await screen.findByRole('button', { name: 'Enable push webhook' }));

    await waitFor(() => {
      expect(apiMock.enableIndexWebhook).toHaveBeenCalledWith('repo');
    });
    expect(await screen.findByText('secret-once')).toBeTruthy();
    expect(apiMock.listIndexWebhookDeliveries).toHaveBeenCalledTimes(2);
  });

  it('rotates the webhook secret in edit mode and refreshes deliveries', async () => {
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Rotate secret' }));

    await waitFor(() => {
      expect(apiMock.rotateIndexWebhookSecret).toHaveBeenCalledWith('repo');
    });
    expect(await screen.findByText('rotated-secret')).toBeTruthy();
    expect(apiMock.listIndexWebhookDeliveries).toHaveBeenCalledTimes(2);
  });

  it('disables the webhook in edit mode and refreshes deliveries', async () => {
    apiMock.getIndexWebhook.mockResolvedValue(enabledWebhookConfig);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Disable webhook' }));

    await waitFor(() => {
      expect(apiMock.disableIndexWebhook).toHaveBeenCalledWith('repo');
    });
    expect(await screen.findByRole('button', { name: 'Enable push webhook' })).toBeTruthy();
    expect(apiMock.listIndexWebhookDeliveries).toHaveBeenCalledTimes(2);
  });

  it('preserves the visible webhook configuration when a delivery refresh fails after enable', async () => {
    apiMock.getIndexWebhook.mockResolvedValue(disabledWebhookConfig);
    apiMock.listIndexWebhookDeliveries
      .mockResolvedValueOnce(webhookDeliveries)
      .mockRejectedValueOnce(new Error('refresh failed'));

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    fireEvent.click(await screen.findByRole('button', { name: 'Enable push webhook' }));

    expect(await screen.findByText('secret-once')).toBeTruthy();
    expect(screen.getByText('webhook:github:main:enabled')).toBeTruthy();
    expect(screen.getByText(/refresh failed/i)).toBeTruthy();
  });

  it('keeps the latest webhook deliveries when a stale edit load resolves after a mutation refresh', async () => {
    const initialDeliveries = deferred<GitWebhookDelivery[]>();
    const mutationDeliveries = deferred<GitWebhookDelivery[]>();

    apiMock.getIndexWebhook.mockResolvedValue(disabledWebhookConfig);
    apiMock.listIndexWebhookDeliveries
      .mockImplementationOnce(() => initialDeliveries.promise)
      .mockImplementationOnce(() => mutationDeliveries.promise);

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    await screen.findByRole('button', { name: 'Enable push webhook' });
    fireEvent.click(screen.getByRole('button', { name: 'Enable push webhook' }));

    mutationDeliveries.resolve([{ ...webhookDeliveries[0], id: 'delivery-new' }]);
    expect(await screen.findByText('delivery-id:delivery-new')).toBeTruthy();

    initialDeliveries.resolve([{ ...webhookDeliveries[0], id: 'delivery-stale' }]);

    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.queryByText('delivery-id:delivery-stale')).toBeNull();
    expect(screen.getByText('delivery-id:delivery-new')).toBeTruthy();
  });

  it('invalidates an in-flight webhook mutation when edit mode switches to a different index', async () => {
    const enableWebhook = deferred<GitWebhookEnableResponse>();
    apiMock.getIndexWebhook
      .mockResolvedValueOnce(disabledWebhookConfig)
      .mockResolvedValueOnce({ ...disabledWebhookConfig, branch: 'develop' });
    apiMock.listIndexWebhookDeliveries
      .mockResolvedValueOnce(webhookDeliveries)
      .mockResolvedValueOnce([{ ...webhookDeliveries[0], id: 'delivery-two' }]);
    apiMock.enableIndexWebhook.mockImplementationOnce(() => enableWebhook.promise);

    const { rerender } = render(<GitIndexWizard editIndex={existingGitIndex} />);

    await screen.findByRole('button', { name: 'Enable push webhook' });
    fireEvent.click(screen.getByRole('button', { name: 'Enable push webhook' }));

    rerender(<GitIndexWizard editIndex={secondGitIndex} />);
    await waitFor(() => expect(apiMock.getIndexWebhook).toHaveBeenCalledWith('repo-two'));

    enableWebhook.resolve(enabledWebhookWithSecret);
    await act(async () => {
      await Promise.resolve();
    });

    expect(screen.queryByText('secret-once')).toBeNull();
    expect(screen.queryByText('webhook:github:main:enabled')).toBeNull();
    expect(screen.getByText('webhook:github:develop:disabled')).toBeTruthy();
  });

  it('shows an initial webhook load error even when no config is available yet', async () => {
    apiMock.getIndexWebhook.mockRejectedValue(new Error('load failed'));

    render(<GitIndexWizard editIndex={existingGitIndex} />);

    expect(await screen.findByText(/failed to load webhook settings: load failed/i)).toBeTruthy();
  });

  it('enables a requested webhook after optimistic index metadata exists', async () => {
    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();

    fireEvent.click(
      screen.getByRole('checkbox', { name: 'Configure a push webhook after creation' }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    await waitFor(() => expect(apiMock.indexFromGit).toHaveBeenCalled());
    expect(apiMock.enableIndexWebhook).toHaveBeenCalledWith('repo');
    expect(await screen.findByText('secret-once')).toBeTruthy();
  });

  it('keeps the indexing job active when webhook setup fails after creation', async () => {
    apiMock.enableIndexWebhook.mockRejectedValue(new Error('webhook failed'));

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();

    fireEvent.click(
      screen.getByRole('checkbox', { name: 'Configure a push webhook after creation' }),
    );
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
    apiMock.enableIndexWebhook.mockRejectedValue(new Error('setup failed'));

    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();
    fireEvent.click(
      screen.getByRole('checkbox', { name: 'Configure a push webhook after creation' }),
    );
    fireEvent.click(screen.getByRole('button', { name: 'Start Indexing' }));

    expect(await screen.findByText(/webhook setup failed: setup failed/i)).toBeTruthy();
    expect(screen.queryByTestId('git-webhook-settings-mock')).toBeNull();
    expect(screen.queryByText('Indexing failed')).toBeNull();
    expect(screen.getAllByText('Cloning repository').length).toBeGreaterThan(0);
  });

  it('uses an explicitly associated full-width create-mode webhook checkbox section', async () => {
    render(<GitIndexWizard onJobCreated={vi.fn()} />);

    await completeAnalysis();

    const checkbox = screen.getByRole('checkbox', {
      name: 'Configure a push webhook after creation',
    });
    const section = checkbox.closest('.git-index-webhook-checkbox-section');
    const label = screen.getByText('Configure a push webhook after creation').closest('label');

    expect(section).toBeTruthy();
    expect(checkbox.getAttribute('id')).toBe('git-index-webhook-after-create');
    expect(label?.getAttribute('for')).toBe('git-index-webhook-after-create');
  });
});
