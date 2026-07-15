import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { GitIndexWizard } from './GitIndexWizard';

const apiMock = vi.hoisted(() => ({
  getSettings: vi.fn(),
  checkRepoVisibility: vi.fn(),
  fetchBranches: vi.fn(),
  analyzeRepository: vi.fn(),
  indexFromGit: vi.fn(),
  getJob: vi.fn(),
  cancelJob: vi.fn(),
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
});
