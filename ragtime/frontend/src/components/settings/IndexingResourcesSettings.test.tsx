import { fireEvent, render, screen } from '@testing-library/react';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { IndexingResourcesSettings } from './IndexingResourcesSettings';
import type { AppSettings, UpdateSettingsRequest } from '@/types';

describe('IndexingResourcesSettings', () => {
  it('preserves legacy positive resource caps and can switch the budget to Auto', () => {
    function Harness(): JSX.Element {
      const [formData, setFormData] = useState<UpdateSettingsRequest>({});
      return (
        <IndexingResourcesSettings
          settings={
            {
              chunking_max_workers: 4,
              chunking_max_batch_size: 100,
              indexing_memory_budget_mb: 512,
              sequential_index_loading: false,
            } as AppSettings
          }
          formData={formData}
          setFormData={setFormData}
        />
      );
    }
    render(<Harness />);

    expect(screen.getByDisplayValue('4')).toBeTruthy();
    expect(screen.getByDisplayValue('100')).toBeTruthy();
    fireEvent.change(screen.getByLabelText('Indexing memory budget mode'), {
      target: { value: 'auto' },
    });
    expect(screen.getByLabelText('Indexing memory budget mode')).toHaveProperty('value', 'auto');
  });

  it('shows stale status without replacing resource controls', () => {
    render(
      <IndexingResourcesSettings
        settings={null}
        formData={{ indexing_memory_budget_mb: 0 }}
        setFormData={vi.fn()}
        stale
        status={{
          sampled_at: '2026-01-01T00:00:00Z',
          stale: false,
          memory_source: 'system',
          system_available_bytes: 1,
          container_limit_bytes: null,
          container_usage_bytes: null,
          application_rss_bytes: 1,
          effective_budget_bytes: 1,
          committed_bytes: 0,
          effective_cpu_capacity: 1,
          event_loop_lag_ms: 0,
          worker_limit: 4,
          worker_target: 2,
          workers_active: 1,
          workers_live: 2,
          active_jobs: 1,
          waiting_jobs: 1,
          limiting_reason: 'memory_headroom',
          jobs: [],
        }}
      />,
    );
    expect(screen.getByText('Live scheduler status (stale)')).toBeTruthy();
    expect(screen.getByText('Waiting for memory headroom.')).toBeTruthy();
  });
});
