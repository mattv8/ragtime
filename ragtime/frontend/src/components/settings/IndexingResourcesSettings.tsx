import type { Dispatch, SetStateAction } from 'react';
import type {
  AppSettings,
  IndexResourceReason,
  IndexResourceStatus,
  UpdateSettingsRequest,
} from '@/types';

export interface IndexingResourcesSettingsProps {
  settings: AppSettings | null;
  formData: UpdateSettingsRequest;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  status?: IndexResourceStatus | null;
  stale?: boolean;
}

const reasonLabels: Record<IndexResourceReason, string> = {
  none: 'No scheduler limit is currently active.',
  memory_headroom: 'Waiting for memory headroom.',
  memory_budget: 'Limited by the indexing memory budget.',
  cpu_capacity: 'Limited by available CPU capacity.',
  provider_limit: 'Limited by the embedding provider.',
  user_limit: 'Limited by the configured ceiling.',
  memory_pressure: 'Paused while memory pressure settles.',
  metrics_unavailable: 'Using conservative scheduling while metrics are unavailable.',
};

function resourceValue(
  formData: UpdateSettingsRequest,
  settings: AppSettings | null,
  field: 'indexing_memory_budget_mb' | 'chunking_max_workers' | 'chunking_max_batch_size',
): number {
  return formData[field] ?? settings?.[field] ?? 0;
}

export function IndexingResourcesSettings({
  settings,
  formData,
  setFormData,
  status,
  stale = false,
}: IndexingResourcesSettingsProps): JSX.Element {
  const memoryBudget = resourceValue(formData, settings, 'indexing_memory_budget_mb');
  const workers = resourceValue(formData, settings, 'chunking_max_workers');
  const batchSize = resourceValue(formData, settings, 'chunking_max_batch_size');
  const sequential =
    formData.sequential_index_loading ?? settings?.sequential_index_loading ?? false;

  return (
    <fieldset data-indexing-resources-settings="true">
      <legend>Indexing Resources</legend>
      <p className="fieldset-help">
        Allocation adapts to available CPU and memory so indexing preserves application headroom.
        The scheduling budget is not an operating-system hard memory limit.
      </p>
      <div className="form-group" id="setting-indexing_memory_budget_mb">
        <label>Indexing memory budget</label>
        <select
          aria-label="Indexing memory budget mode"
          value={memoryBudget === 0 ? 'auto' : 'custom'}
          onChange={(event) =>
            setFormData((current) => ({
              ...current,
              indexing_memory_budget_mb:
                event.target.value === 'auto' ? 0 : Math.max(256, memoryBudget || 256),
            }))
          }
        >
          <option value="auto">Auto</option>
          <option value="custom">Custom</option>
        </select>
        {memoryBudget !== 0 && (
          <input
            aria-label="Indexing memory budget MiB"
            type="number"
            min={256}
            max={1048576}
            value={memoryBudget}
            onChange={(event) => {
              const value = Number.parseInt(event.target.value, 10);
              setFormData((current) => ({
                ...current,
                indexing_memory_budget_mb: Number.isFinite(value)
                  ? Math.max(256, Math.min(1048576, value))
                  : 256,
              }));
            }}
          />
        )}
        <p className="field-help">
          {memoryBudget === 0 ? 'Auto adapts to measured headroom.' : 'Custom ceiling in MiB.'}
        </p>
      </div>
      <details id="setting-indexing_resources_advanced">
        <summary className="settings-advanced-summary">Advanced settings</summary>
        <div className="form-row">
          <div className="form-group" style={{ flex: 1 }} id="setting-chunking_max_workers">
            <label>Maximum chunking workers across all indexes</label>
            <input
              aria-label="Maximum chunking workers across all indexes"
              type="number"
              min={0}
              max={16}
              value={workers}
              onChange={(event) =>
                setFormData((current) => ({
                  ...current,
                  chunking_max_workers: Math.max(
                    0,
                    Math.min(16, Number.parseInt(event.target.value, 10) || 0),
                  ),
                }))
              }
            />
            <p className="field-help">
              0 = Auto. Positive values are one aggregate cap, not a per-job cap.
            </p>
          </div>
          <div className="form-group" style={{ flex: 1 }} id="setting-chunking_max_batch_size">
            <label>Maximum documents per chunking batch</label>
            <input
              aria-label="Maximum documents per chunking batch"
              type="number"
              min={0}
              max={500}
              value={batchSize}
              onChange={(event) =>
                setFormData((current) => ({
                  ...current,
                  chunking_max_batch_size: Math.max(
                    0,
                    Math.min(500, Number.parseInt(event.target.value, 10) || 0),
                  ),
                }))
              }
            />
            <p className="field-help">
              0 = Auto. Positive values preserve the document-count ceiling.
            </p>
          </div>
        </div>
        <div className="form-group" id="setting-sequential_index_loading">
          <label>Index loading</label>
          <select
            value={sequential ? 'sequential' : 'auto'}
            onChange={(event) =>
              setFormData((current) => ({
                ...current,
                sequential_index_loading: event.target.value === 'sequential',
              }))
            }
          >
            <option value="auto">Auto</option>
            <option value="sequential">Sequential</option>
          </select>
          <p className="field-help">
            Sequential loading reduces transient overlap; already-loaded indexes remain resident.
          </p>
        </div>
      </details>
      {status && (
        <div className="form-group" data-indexing-resource-status="true">
          <strong>Live scheduler status{stale ? ' (stale)' : ''}</strong>
          <p className="field-help">
            Configured workers: {workers || 'Auto'} · Target: {status.worker_target} · Active:{' '}
            {status.workers_active} · Live: {status.workers_live}
          </p>
          <p className="field-help">{reasonLabels[status.limiting_reason]}</p>
        </div>
      )}
    </fieldset>
  );
}
