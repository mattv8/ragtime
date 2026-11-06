import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { JobsTable } from './JobsTable';
import type { IndexJob } from '@/types';

vi.mock('@/api', () => ({
  api: {
    cancelJob: vi.fn(),
    retryJob: vi.fn(),
    cancelUserSpaceCodeIndexJob: vi.fn(),
    retryFilesystemJob: vi.fn(),
    retrySchemaJob: vi.fn(),
    retryPdmJob: vi.fn(),
  },
}));

vi.mock('@/hooks/useIndexResourceStatus', () => ({
  useIndexResourceStatus: () => ({
    stale: false,
    data: {
      jobs: [{ job_id: 'job-waiting', state: 'waiting', reason: 'memory_headroom' }],
    },
  }),
}));

afterEach(() => {
  cleanup();
});

describe('JobsTable', () => {
  it('uses the persisted chunking phase with document-unit stats for document jobs', () => {
    const job: IndexJob = {
      id: 'job-1',
      name: 'Large upload',
      status: 'processing',
      phase: 'chunking',
      progress_percent: 34,
      total_files: 528,
      processed_files: 528,
      total_chunks: 1573,
      processed_chunks: 300,
      error_message: 'Chunking documents...',
      created_at: '2026-07-15T12:00:00Z',
      started_at: '2026-07-15T12:00:10Z',
      completed_at: null,
    };

    const { container } = render(<JobsTable jobs={[job]} loading={false} error={null} />);

    expect(screen.getByText('Chunking')).toBeTruthy();
    expect(screen.getByText('300/1,573 documents')).toBeTruthy();
    expect(container.querySelector('.progress-fill')?.getAttribute('style')).toContain(
      'width: 34%',
    );
  });

  it('does not let a chunking status message override a persisted embedding phase', () => {
    const job: IndexJob = {
      id: 'job-2',
      name: 'Git repo',
      status: 'processing',
      phase: 'embedding',
      progress_percent: 81,
      total_files: 528,
      processed_files: 528,
      total_chunks: 1573,
      processed_chunks: 300,
      error_message: 'Chunking documents...',
      created_at: '2026-07-15T12:00:00Z',
      started_at: '2026-07-15T12:00:10Z',
      completed_at: null,
    };

    render(<JobsTable jobs={[job]} loading={false} error={null} />);

    expect(screen.getByText('Embedding')).toBeTruthy();
    expect(screen.getByText('300/1,573 chunks')).toBeTruthy();
    expect(screen.queryByText('300/1,573 documents')).toBeNull();
  });

  it('adds a resource wait explanation without replacing the persisted phase', () => {
    const job: IndexJob = {
      id: 'job-waiting',
      name: 'Git repo',
      status: 'processing',
      phase: 'embedding',
      progress_percent: 81,
      total_files: 1,
      processed_files: 1,
      total_chunks: 2,
      processed_chunks: 1,
      error_message: null,
      created_at: '2026-07-15T12:00:00Z',
      started_at: '2026-07-15T12:00:10Z',
      completed_at: null,
    };
    render(<JobsTable jobs={[job]} loading={false} error={null} />);
    expect(screen.getByText('Embedding')).toBeTruthy();
    expect(screen.getByText('Waiting for memory headroom')).toBeTruthy();
  });

  it('keeps a completed job visible as warned and opens its bound warning details', async () => {
    const user = userEvent.setup();
    const job: IndexJob = {
      id: 'job-warned',
      name: 'Monthly report',
      status: 'completed',
      phase: 'completed',
      progress_percent: 100,
      total_files: 1,
      processed_files: 1,
      total_chunks: 12,
      processed_chunks: 12,
      error_message: null,
      created_at: '2026-07-15T12:00:00Z',
      started_at: '2026-07-15T12:00:10Z',
      completed_at: '2026-07-15T12:01:00Z',
      warnings: {
        count: 2,
        items: [{ source: 'reports/multi-page.pdf', category: 'timeout' }],
      },
    };

    render(<JobsTable jobs={[job]} loading={false} error={null} />);

    const warningButton = screen.getByRole('button', {
      name: 'Completed with warnings: Monthly report. View 2 sources.',
    });
    expect(warningButton).toBeTruthy();
    expect(screen.queryByText('1 files, 12 chunks')).toBeNull();

    await user.click(warningButton);

    const dialog = screen.getByRole('dialog', { name: 'Job warnings' });
    expect(dialog).toBeTruthy();
    expect(screen.getByText('reports/multi-page.pdf')).toBeTruthy();
    expect(screen.getByText('OCR did not finish before the time limit.')).toBeTruthy();
    expect(screen.queryByText(/OCR pages skipped/i)).toBeNull();
  });

  it('keeps warning detail classification and focus correct as a job changes or disappears', async () => {
    const user = userEvent.setup();
    const activeJob: IndexJob = {
      id: 'job-transition',
      name: 'Source index',
      status: 'processing',
      phase: 'embedding',
      progress_percent: 50,
      total_files: 1,
      processed_files: 1,
      total_chunks: 4,
      processed_chunks: 2,
      error_message: 'Embedding',
      created_at: '2026-07-15T12:00:00Z',
      started_at: '2026-07-15T12:00:10Z',
      completed_at: null,
      warnings: { count: 1, items: [{ source: 'manual.pdf', category: 'unknown' }] },
    };
    const { rerender } = render(<JobsTable jobs={[activeJob]} loading={false} error={null} />);

    await user.click(screen.getByRole('button', { name: '1 source skipped during OCR so far' }));
    expect(screen.getByRole('dialog', { name: 'Job warnings' })).toBeTruthy();
    expect(screen.queryByText('Error Message')).toBeNull();
    expect(screen.getByText('OCR could not process this source.')).toBeTruthy();

    rerender(
      <JobsTable
        jobs={[
          {
            ...activeJob,
            status: 'completed',
            phase: 'completed',
            completed_at: '2026-07-15T12:01:00Z',
          },
        ]}
        loading={false}
        error={null}
      />,
    );
    await user.click(screen.getByRole('button', { name: 'Close job details' }));
    expect(document.activeElement).toBe(
      screen.getByRole('button', { name: 'Completed with warnings: Source index. View 1 source.' }),
    );

    await user.click(
      screen.getByRole('button', { name: 'Completed with warnings: Source index. View 1 source.' }),
    );
    rerender(<JobsTable jobs={[]} loading={false} error={null} />);
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('heading', { name: 'Indexing Jobs' })),
    );
    expect(screen.queryByRole('dialog')).toBeNull();
  });
});
