import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { IndexesList } from './IndexesList';
import type { IndexInfo, IndexJob } from '@/types';

const apiMock = vi.hoisted(() => ({
  getSettings: vi.fn(),
  getHealth: vi.fn(),
}));

const gitWizardPropsMock = vi.hoisted(() => ({
  current: null as null | Record<string, unknown>,
}));
const uploadFormPropsMock = vi.hoisted(() => ({
  current: null as null | Record<string, unknown>,
}));
const importFaissPropsMock = vi.hoisted(() => ({
  current: null as null | Record<string, unknown>,
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('./GitIndexWizard', () => ({
  GitIndexWizard: (props: Record<string, unknown>) => {
    gitWizardPropsMock.current = props;
    return (
      <button type="button" onClick={() => (props.onJobCreated as (() => void) | undefined)?.()}>
        trigger git job
      </button>
    );
  },
}));

vi.mock('./UploadForm', () => ({
  UploadForm: (props: Record<string, unknown>) => {
    uploadFormPropsMock.current = props;
    return (
      <button type="button" onClick={() => (props.onJobCreated as (() => void) | undefined)?.()}>
        trigger upload job
      </button>
    );
  },
}));

vi.mock('./ImportFaissForm', () => ({
  ImportFaissForm: (props: Record<string, unknown>) => {
    importFaissPropsMock.current = props;
    return (
      <button type="button" onClick={() => (props.onImported as (() => void) | undefined)?.()}>
        trigger import job
      </button>
    );
  },
}));

vi.mock('./IndexCard', () => ({
  IndexCard: (props: Record<string, unknown>) => (
    <div data-testid={`index-card-${String(props.title)}`}>
      <div>{String(props.title)}</div>
      <input
        type="checkbox"
        aria-label={`${String(props.title)} toggle`}
        checked={Boolean(props.enabled)}
        disabled={Boolean(props.toggleDisabled)}
        readOnly
      />
      <div>
        {(props.onEditTitle as (() => void) | undefined) ? 'title editable' : 'title locked'}
      </div>
      <div>
        {(props.onEditDescription as (() => void) | undefined)
          ? 'description editable'
          : 'description locked'}
      </div>
      <div>{props.actions as React.ReactNode}</div>
      <div>{props.metaPills as React.ReactNode}</div>
    </div>
  ),
}));

vi.mock('./DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({ title }: Record<string, unknown>) => <button>{String(title)}</button>,
}));

vi.mock('./AnimatedCreateButton', () => ({
  AnimatedCreateButton: ({ isExpanded, onClick, label }: Record<string, unknown>) => (
    <button type="button" onClick={onClick as () => void}>
      {String(label)} {isExpanded ? 'close' : 'open'}
    </button>
  ),
}));

vi.mock('./IndexingPill', () => ({
  IndexingPill: ({ activeJob }: Record<string, unknown>) =>
    activeJob ? <span>{`IndexingPill:${String((activeJob as IndexJob).phase)}`}</span> : null,
}));

function makeIndex(overrides: Partial<IndexInfo> = {}): IndexInfo {
  return {
    name: 'repo',
    display_name: 'Repo',
    path: '/tmp/repo',
    size_mb: 12,
    document_count: 4,
    chunk_count: 9,
    description: 'Indexed repository',
    enabled: true,
    search_weight: 1,
    source_type: 'git',
    source: 'https://github.com/example/repo.git',
    git_branch: 'main',
    has_stored_token: false,
    config_snapshot: null,
    created_at: '2026-07-15T00:00:00Z',
    last_modified: '2026-07-15T00:00:00Z',
    git_repo_size_mb: null,
    has_git_history: false,
    ...overrides,
  };
}

function makeJob(overrides: Partial<IndexJob> = {}): IndexJob {
  return {
    id: 'job-1',
    name: 'repo',
    status: 'processing',
    phase: 'chunking',
    progress_percent: 40,
    total_files: 10,
    processed_files: 10,
    total_chunks: 24,
    processed_chunks: 3,
    error_message: null,
    created_at: '2026-07-15T00:00:00Z',
    started_at: '2026-07-15T00:00:05Z',
    completed_at: null,
    ...overrides,
  };
}

beforeEach(() => {
  apiMock.getSettings.mockResolvedValue({
    settings: {
      archive_max_total_size_bytes: null,
      archive_max_file_count: null,
    },
  });
  apiMock.getHealth.mockResolvedValue({ index_details: [] });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  gitWizardPropsMock.current = null;
  uploadFormPropsMock.current = null;
  importFaissPropsMock.current = null;
});

describe('IndexesList create wizard handlers', () => {
  it('keeps the git wizard open for git jobs but still closes for upload and import flows', async () => {
    const user = userEvent.setup();
    const onJobCreated = vi.fn();

    render(
      <IndexesList
        indexes={[]}
        jobs={[]}
        loading={false}
        error={null}
        onDelete={vi.fn()}
        onJobCreated={onJobCreated}
      />,
    );

    await user.click(screen.getByRole('button', { name: /Add Document Index open/i }));
    await user.click(screen.getByRole('button', { name: 'trigger git job' }));

    expect(onJobCreated).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: 'trigger git job' })).not.toBeNull();

    await user.click(screen.getByRole('button', { name: 'Upload Archive' }));
    await user.click(screen.getByRole('button', { name: 'trigger upload job' }));

    expect(onJobCreated).toHaveBeenCalledTimes(2);
    expect(screen.queryByRole('button', { name: 'trigger upload job' })).toBeNull();

    await user.click(screen.getByRole('button', { name: /Add Document Index open/i }));
    await user.click(screen.getByRole('button', { name: 'Import FAISS' }));
    await user.click(screen.getByRole('button', { name: 'trigger import job' }));

    expect(onJobCreated).toHaveBeenCalledTimes(3);
    expect(screen.queryByRole('button', { name: 'trigger import job' })).toBeNull();
  });
});

describe('IndexesList active and terminal git card controls', () => {
  it('locks active git cards to edit-only controls while indexing', async () => {
    const user = userEvent.setup();

    render(
      <IndexesList
        indexes={[makeIndex()]}
        jobs={[makeJob()]}
        loading={false}
        error={null}
        onDelete={vi.fn()}
        aggregateSearch={false}
      />,
    );

    expect(screen.getByRole('button', { name: 'Edit' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Download' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Pull & Re-index' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Delete index' })).toBeNull();
    expect((screen.getByLabelText('Repo toggle') as HTMLInputElement).disabled).toBe(true);
    expect(screen.getByText('title locked')).toBeTruthy();
    expect(screen.getByText('description locked')).toBeTruthy();
    expect(screen.getByText('IndexingPill:chunking')).toBeTruthy();

    await user.click(screen.getByText('Weight: 1.0'));
    expect(screen.queryByText('Search Weight: repo')).toBeNull();
  });

  it('restores retry and incomplete state for failed git cards', () => {
    render(
      <IndexesList
        indexes={[makeIndex()]}
        jobs={[
          makeJob({ status: 'failed', phase: 'failed', completed_at: '2026-07-15T00:01:00Z' }),
        ]}
        loading={false}
        error={null}
        onDelete={vi.fn()}
        aggregateSearch={false}
      />,
    );

    expect(screen.getByText('Incomplete')).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Delete index' })).toBeTruthy();
    expect((screen.getByLabelText('Repo toggle') as HTMLInputElement).disabled).toBe(false);
    expect(screen.getByText('title editable')).toBeTruthy();
    expect(screen.getByText('description editable')).toBeTruthy();
  });

  it('restores normal actions and edits after git indexing completes', async () => {
    const user = userEvent.setup();

    render(
      <IndexesList
        indexes={[makeIndex()]}
        jobs={[
          makeJob({
            status: 'completed',
            phase: 'completed',
            progress_percent: 100,
            processed_chunks: 24,
            completed_at: '2026-07-15T00:01:00Z',
          }),
        ]}
        loading={false}
        error={null}
        onDelete={vi.fn()}
        aggregateSearch={false}
      />,
    );

    expect(screen.getByRole('button', { name: 'Edit' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Pull & Re-index' })).toBeTruthy();
    expect(screen.getByRole('button', { name: 'Delete index' })).toBeTruthy();
    expect((screen.getByLabelText('Repo toggle') as HTMLInputElement).disabled).toBe(false);
    expect(screen.getByText('title editable')).toBeTruthy();
    expect(screen.getByText('description editable')).toBeTruthy();

    await user.click(screen.getByText('Weight: 1.0'));
    expect(screen.getByText('Search Weight: repo')).toBeTruthy();
  });
});
