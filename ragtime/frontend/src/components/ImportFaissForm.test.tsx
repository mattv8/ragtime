import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ImportFaissForm } from './ImportFaissForm';

const apiMock = vi.hoisted(() => ({ importFaissIndex: vi.fn() }));

vi.mock('@/api', () => ({ api: apiMock }));

vi.mock('./shared/FileDropZone', () => ({
  FileDropZone: ({ onFileSelected }: { onFileSelected: (file: File) => void }) => (
    <button type="button" onClick={() => onFileSelected(new File(['index'], 'example.zip'))}>
      Select archive
    </button>
  ),
}));

const importedIndex = {
  name: 'example',
  display_name: 'Example',
  description: 'Imported documents',
  document_count: 2,
  chunk_count: 4,
  size_bytes: 128,
  source_type: 'upload',
  vector_store_type: 'faiss',
  message: 'FAISS index imported.',
  loaded: true,
  load_error: null,
};

async function importArchive(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByRole('button', { name: 'Select archive' }));
  await user.click(screen.getByRole('button', { name: 'Import FAISS Index' }));
  await user.click(screen.getByRole('button', { name: 'I Understand, Import' }));
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('ImportFaissForm', () => {
  it('shows an error alert and preserves the result when persistence succeeds but loading fails', async () => {
    const user = userEvent.setup();
    const onImported = vi.fn();
    apiMock.importFaissIndex.mockResolvedValue({
      ...importedIndex,
      loaded: false,
      load_error: 'Index dimension 768 does not match configured embedding dimension 1024.',
    });

    render(<ImportFaissForm onImported={onImported} onCancel={vi.fn()} />);
    await importArchive(user);

    expect((await screen.findByRole('alert')).textContent).toContain(
      'saved but is unavailable for search',
    );
    expect(screen.getByRole('alert').textContent).toContain(
      'does not match configured embedding dimension',
    );
    expect(screen.getByText('Saved but not loaded')).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Import FAISS Index' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Close' })).toBeTruthy();
    expect(onImported).toHaveBeenCalledWith(expect.objectContaining({ loaded: false }));
  });

  it('shows the existing successful loaded result', async () => {
    const user = userEvent.setup();
    apiMock.importFaissIndex.mockResolvedValue(importedIndex);

    render(<ImportFaissForm onCancel={vi.fn()} />);
    await importArchive(user);

    expect(await screen.findByText('Imported')).toBeTruthy();
    expect(screen.queryByRole('alert')).toBeNull();
  });
});
