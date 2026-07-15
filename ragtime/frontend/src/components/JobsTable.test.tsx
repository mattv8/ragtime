import { cleanup, render, screen } from '@testing-library/react';
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
});
