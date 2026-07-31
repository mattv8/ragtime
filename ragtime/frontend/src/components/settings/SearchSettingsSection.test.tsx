import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { SearchSettingsSection } from './SearchSettingsSection';
import type { AppSettings, UpdateSettingsRequest } from '@/types';

afterEach(() => {
  cleanup();
});

function buildSettings(overrides: Partial<AppSettings> = {}): AppSettings {
  return {
    faiss_search_concurrency_mode: 'per_index',
    aggregate_search: true,
    search_results_k: 5,
    search_use_mmr: true,
    search_mmr_lambda: 0.5,
    chunking_use_tokens: true,
    ivfflat_lists: 100,
    archive_max_total_size_bytes: 5_368_709_120,
    archive_max_file_count: 100000,
    ...overrides,
  } as AppSettings;
}

function renderSection({
  formData = {},
  settings = buildSettings(),
  searchSaving = false,
  handleSaveSearch = vi.fn(),
}: {
  formData?: UpdateSettingsRequest;
  settings?: AppSettings | null;
  searchSaving?: boolean;
  handleSaveSearch?: () => void | Promise<void>;
} = {}) {
  function Wrapper(): JSX.Element {
    const [currentFormData, setCurrentFormData] = useState<UpdateSettingsRequest>(formData);

    return (
      <SearchSettingsSection
        open
        onToggle={() => {}}
        formData={currentFormData}
        settings={settings}
        setFormData={setCurrentFormData}
        handleSaveSearch={handleSaveSearch}
        searchSaving={searchSaving}
      />
    );
  }

  return render(<Wrapper />);
}

describe('SearchSettingsSection', () => {
  it('renders the FAISS concurrency switch card with the default per-index copy', () => {
    const { container } = renderSection();

    expect(screen.getByRole('button', { name: 'Search Configuration' })).toBeTruthy();
    expect(screen.getByLabelText('Global FAISS search gate')).toBeTruthy();
    expect(document.getElementById('setting-faiss_search_concurrency_mode')).toBeTruthy();
    expect(document.getElementById('search-faiss-concurrency-mode')).toBeTruthy();
    expect(
      screen.getByText(
        'Serialize searches per index while allowing different indexes to search concurrently (default).',
      ),
    ).toBeTruthy();
    expect(
      screen.getByText(
        'Use global mode when concurrent vector searches affect server responsiveness. Changes apply to new searches after saving.',
      ),
    ).toBeTruthy();
    expect(container.querySelector('.search-settings-switch-card')).toBeTruthy();
    expect((screen.getByLabelText('Global FAISS search gate') as HTMLInputElement).checked).toBe(
      false,
    );
  });

  it('switches to global mode copy when the toggle is enabled', async () => {
    const user = userEvent.setup();
    renderSection();

    await user.click(screen.getByLabelText('Global FAISS search gate'));

    expect((screen.getByLabelText('Global FAISS search gate') as HTMLInputElement).checked).toBe(
      true,
    );
    expect(
      screen.getByText(
        'Serialize all FAISS searches through one process-wide gate for maximum server isolation.',
      ),
    ).toBeTruthy();
  });

  it('calls save once and shows the saving state', async () => {
    const user = userEvent.setup();
    const handleSaveSearch = vi.fn();
    const { rerender } = renderSection({ handleSaveSearch });

    await user.click(screen.getByRole('button', { name: 'Save Search Configuration' }));
    expect(handleSaveSearch).toHaveBeenCalledTimes(1);

    rerender(
      <SearchSettingsSection
        open
        onToggle={() => {}}
        formData={{}}
        settings={buildSettings()}
        setFormData={vi.fn()}
        handleSaveSearch={handleSaveSearch}
        searchSaving
      />,
    );

    expect((screen.getByRole('button', { name: 'Saving...' }) as HTMLButtonElement).disabled).toBe(
      true,
    );
  });
});
