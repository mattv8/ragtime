import type { Dispatch, SetStateAction } from 'react';
import type { AppSettings, UpdateSettingsRequest } from '@/types';
import type { SettingsAccordionSectionId } from './settingsAccordionState';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import { formatBytes } from '@/utils';

interface SearchSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  settings: AppSettings | null;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  handleSaveSearch: () => void | Promise<void>;
  searchSaving: boolean;
}

export function SearchSettingsSection(props: SearchSettingsSectionProps): JSX.Element {
  const { open, onToggle, formData, settings, setFormData, handleSaveSearch, searchSaving } = props;

  return (
    <SettingsAccordionSection
      id="search"
      title="Search Configuration"
      open={open}
      onToggle={onToggle}
    >
      <fieldset>
        <legend>Search Configuration</legend>
        <p className="fieldset-help">
          Configure how vector search behaves across your indexed knowledge bases.
        </p>

        <div className="form-group">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={formData.aggregate_search ?? settings?.aggregate_search ?? true}
              onChange={(e) => setFormData({ ...formData, aggregate_search: e.target.checked })}
              style={{ marginRight: '0.5rem' }}
            />
            <span>Aggregate search results (single tool)</span>
          </label>
          <p className="field-help">
            <strong>Enabled (default):</strong> A single <code>search_knowledge</code> tool searches
            all indexes. Results are combined and the AI receives context from all sources.
            <br />
            <strong>Disabled:</strong> Creates separate <code>search_&lt;index_name&gt;</code> tools
            for each index. The AI can choose which specific index to search, giving it granular
            control. Use this when you have distinct knowledge bases (e.g., code vs. docs) and want
            the AI to target searches.
          </p>
        </div>

        {/* Advanced Search Settings */}
        <details style={{ marginBottom: '16px' }} id="setting-search_advanced">
          <summary className="settings-advanced-summary">Advanced Settings</summary>

          <div className="form-group">
            <label>Results per Search (k)</label>
            <input
              type="number"
              min={1}
              max={100}
              value={formData.search_results_k ?? settings?.search_results_k ?? 5}
              onChange={(e) =>
                setFormData({
                  ...formData,
                  search_results_k: Math.max(1, Math.min(100, parseInt(e.target.value, 10) || 5)),
                })
              }
            />
            <p className="field-help">
              Document chunks retrieved per query (k). Lower (3-5) is faster; higher (10-20) gives
              more context but costs more tokens.
            </p>
          </div>

          <div className="form-row">
            <div className="form-group" style={{ flex: 1 }}>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={formData.search_use_mmr ?? settings?.search_use_mmr ?? true}
                  onChange={(e) => setFormData({ ...formData, search_use_mmr: e.target.checked })}
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Use MMR (Max Marginal Relevance)</span>
              </label>
              <p className="field-help">
                Reduces near-duplicate results by balancing relevance with diversity.
              </p>

              {(formData.search_use_mmr ?? settings?.search_use_mmr ?? true) && (
                <div style={{ marginTop: '0.5rem' }}>
                  <label>
                    MMR Diversity/Relevance (lambda:{' '}
                    {formData.search_mmr_lambda ?? settings?.search_mmr_lambda ?? 0.5})
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.1}
                    value={formData.search_mmr_lambda ?? settings?.search_mmr_lambda ?? 0.5}
                    onChange={(e) =>
                      setFormData({ ...formData, search_mmr_lambda: parseFloat(e.target.value) })
                    }
                    style={{ width: '100%' }}
                  />
                  <p className="field-help">
                    <strong>0 = Max diversity</strong> |<strong> 1 = Max relevance</strong>. Default
                    0.5 provides a good balance.
                  </p>
                </div>
              )}
            </div>

            <div className="form-group" style={{ flex: 1 }}>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={formData.chunking_use_tokens ?? settings?.chunking_use_tokens ?? true}
                  onChange={(e) =>
                    setFormData({ ...formData, chunking_use_tokens: e.target.checked })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Token-based chunking</span>
              </label>
              <p className="field-help">
                Use token-based chunking instead of character-based for more accurate chunk sizes
                aligned with model tokenization.
              </p>

              <div style={{ marginTop: '0.5rem' }}>
                <label>IVFFlat Lists (pgvector)</label>
                <input
                  type="number"
                  min={10}
                  max={1000}
                  value={formData.ivfflat_lists ?? settings?.ivfflat_lists ?? 100}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      ivfflat_lists: Math.max(
                        10,
                        Math.min(1000, parseInt(e.target.value, 10) || 100),
                      ),
                    })
                  }
                />
                <p className="field-help">
                  pgvector index parameter. Higher = faster queries for large datasets. Recommended:
                  sqrt(num embeddings). Default: 100.
                </p>
              </div>
            </div>
          </div>

          {/* Archive Extraction Limits */}
          <div
            className="form-row"
            style={{
              marginTop: '1rem',
              borderTop: '1px solid var(--color-border)',
              paddingTop: '1rem',
            }}
          >
            <div className="form-group" style={{ flex: 1 }}>
              <label>Archive Max Size</label>
              <p className="field-help" style={{ marginTop: 0 }}>
                Maximum total uncompressed size of an uploaded index or workspace archive.
              </p>
              {(() => {
                const currentVal =
                  formData.archive_max_total_size_bytes ??
                  settings?.archive_max_total_size_bytes ??
                  5_368_709_120;

                return (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <input
                        type="range"
                        min={104_857_600}
                        max={536_870_912_000}
                        step={104_857_600}
                        style={{ flex: 1 }}
                        value={currentVal}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            archive_max_total_size_bytes: parseInt(e.target.value, 10),
                          })
                        }
                      />
                      <span
                        style={{
                          minWidth: '84px',
                          textAlign: 'right',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        {formatBytes(currentVal)}
                      </span>
                    </div>
                    <p className="field-help">
                      Range: 100 MB to 500 GB. Applies to both upload and git-sourced index
                      archives, plus User Space workspace archive imports.
                    </p>
                  </>
                );
              })()}
            </div>

            <div className="form-group" style={{ flex: 1 }}>
              <label>Archive Max File Count</label>
              <p className="field-help" style={{ marginTop: 0 }}>
                Maximum number of entries allowed in a single extracted index or workspace archive.
              </p>
              {(() => {
                const currentVal =
                  formData.archive_max_file_count ?? settings?.archive_max_file_count ?? 100000;

                return (
                  <>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <input
                        type="range"
                        min={100}
                        max={500000}
                        step={1000}
                        style={{ flex: 1 }}
                        value={currentVal}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            archive_max_file_count: parseInt(e.target.value, 10),
                          })
                        }
                      />
                      <span
                        style={{
                          minWidth: '72px',
                          textAlign: 'right',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        {currentVal.toLocaleString()}
                      </span>
                    </div>
                    <p className="field-help">
                      Default: 100,000. Increase for large monorepos or bulk archives. Range: 100 to
                      500,000.
                    </p>
                  </>
                );
              })()}
            </div>
          </div>
        </details>

        <div className="form-actions">
          <button type="button" className="btn" onClick={handleSaveSearch} disabled={searchSaving}>
            {searchSaving ? 'Saving...' : 'Save Search Configuration'}
          </button>
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
