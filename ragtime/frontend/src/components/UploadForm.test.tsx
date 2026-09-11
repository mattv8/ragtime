import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { IndexAnalysisResult } from '@/types';
import { UploadForm } from './UploadForm';

const apiMock = vi.hoisted(() => ({
  getSettings: vi.fn(),
  analyzeUpload: vi.fn(),
  uploadAndIndex: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('./shared/FileDropZone', () => ({
  FileDropZone: ({ onFileSelected }: { onFileSelected: (file: File) => void }) => (
    <button
      type="button"
      onClick={() => onFileSelected(new File(['x'], 'my-archive.zip', { type: 'application/zip' }))}
    >
      Select archive
    </button>
  ),
}));

vi.mock('./DescriptionField', () => ({ DescriptionField: () => null }));
vi.mock('./AnalysisStats', () => ({ AnalysisStats: () => null }));
vi.mock('./IndexConfigFields', () => ({ IndexConfigFields: () => null }));
vi.mock('./OcrVectorStoreFields', () => ({
  OcrVectorStoreFields: () => null,
  OCR_PROVIDER_LABELS: {},
}));
vi.mock('./FileTypeStatsTable', () => ({ FileTypeStatsTable: () => null }));
vi.mock('./SuggestedExclusionsBanner', () => ({ SuggestedExclusionsBanner: () => null }));
vi.mock('./WarningsBanner', () => ({ WarningsBanner: () => null }));

const analysisResult = {
  warnings: [],
  suggested_exclusions: [],
  file_type_stats: [],
} as unknown as IndexAnalysisResult;

async function selectAndAnalyzeArchive(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Select archive' }));
  await user.click(screen.getByRole('button', { name: 'Analyze Archive' }));
}

beforeEach(() => {
  apiMock.getSettings.mockResolvedValue({ settings: {} });
  apiMock.analyzeUpload.mockResolvedValue(analysisResult);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('UploadForm', () => {
  it('keeps the recovery target name after file selection', async () => {
    const user = userEvent.setup();

    render(<UploadForm onJobCreated={vi.fn()} initialName="odev_proj" />);
    await selectAndAnalyzeArchive(user);

    expect(
      ((await screen.findByPlaceholderText('e.g., odoo-17, my-codebase')) as HTMLInputElement)
        .value,
    ).toBe('odev_proj');
  });

  it('derives the index name from the filename without initialName', async () => {
    const user = userEvent.setup();

    render(<UploadForm onJobCreated={vi.fn()} />);
    await selectAndAnalyzeArchive(user);

    expect(
      ((await screen.findByPlaceholderText('e.g., odoo-17, my-codebase')) as HTMLInputElement)
        .value,
    ).toBe('my-archive');
  });

  it('preserves the recovery target name after indexing resets the form', async () => {
    const user = userEvent.setup();
    apiMock.uploadAndIndex.mockResolvedValue({ id: 'j1', status: 'pending' });

    render(<UploadForm onJobCreated={vi.fn()} initialName="odev_proj" />);
    await selectAndAnalyzeArchive(user);
    await user.click(screen.getByRole('button', { name: 'Create Index' }));
    await selectAndAnalyzeArchive(user);

    expect(
      ((await screen.findByPlaceholderText('e.g., odoo-17, my-codebase')) as HTMLInputElement)
        .value,
    ).toBe('odev_proj');
  });
});
