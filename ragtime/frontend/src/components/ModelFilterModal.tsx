import { useState, useMemo } from 'react';
import type { AvailableModel } from '@/types';
import { formatProviderDisplayName } from '@/utils/modelDisplay';

export interface ModelFilterModalProps {
  isOpen: boolean;
  title: string;
  onClose: () => void;
  /** All discovered models across providers. */
  allModels: AvailableModel[];
  modelsLoading: boolean;
  /** Currently allowed scoped model ids ("provider::id"). Empty set = all allowed. */
  selectedModels: Set<string>;
  /** Helper text shown below the filter actions. */
  allowedHelpText: string;
  /** Toggle a single model selection (handles same-id-different-provider replacement). */
  toggleModel: (model: AvailableModel) => void;
  /** Replace selection with all models. */
  selectAll: () => void;
  /** Replace selection with empty set. */
  deselectAll: () => void;
  /** Persist allowed model list. */
  onSaveAllowed: () => Promise<void> | void;
  savingAllowed?: boolean;
}

export function ModelFilterModal({
  isOpen,
  title,
  onClose,
  allModels,
  modelsLoading,
  selectedModels,
  allowedHelpText,
  toggleModel,
  selectAll,
  deselectAll,
  onSaveAllowed,
  savingAllowed,
}: ModelFilterModalProps) {
  const [filterText, setFilterText] = useState('');

  const filteredAllowed = useMemo(() => {
    const lc = filterText.toLowerCase();
    return allModels.filter(
      (m) =>
        filterText === '' ||
        m.name.toLowerCase().includes(lc) ||
        m.id.toLowerCase().includes(lc) ||
        m.provider.toLowerCase().includes(lc) ||
        (m.model_provider_label || '').toLowerCase().includes(lc) ||
        (m.model_family || m.group || '').toLowerCase().includes(lc),
    );
  }, [allModels, filterText]);

  const groupedAllowed = useMemo(() => {
    const providerOrder: string[] = [];
    const providerGroups: Record<string, Record<string, Record<string, AvailableModel[]>>> = {};
    filteredAllowed.forEach((m) => {
      if (!providerGroups[m.provider]) {
        providerGroups[m.provider] = {};
        providerOrder.push(m.provider);
      }
      const modelProvider = m.model_provider_label || 'Other';
      const family = m.model_family || m.group || 'Other';
      if (!providerGroups[m.provider][modelProvider]) providerGroups[m.provider][modelProvider] = {};
      if (!providerGroups[m.provider][modelProvider][family]) providerGroups[m.provider][modelProvider][family] = [];
      providerGroups[m.provider][modelProvider][family].push(m);
    });
    return { providerOrder, providerGroups };
  }, [filteredAllowed]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="modal-body">
          {modelsLoading ? (
            <p className="muted">Loading available models...</p>
          ) : allModels.length === 0 ? (
            <p className="muted">
              No models available. Please configure API keys or local providers and save settings first.
            </p>
          ) : (
            <>
              <div className="model-filter-search">
                <input
                  type="text"
                  placeholder="Filter models..."
                  value={filterText}
                  onChange={(e) => setFilterText(e.target.value)}
                  autoFocus
                />
              </div>
              <div className="model-filter-actions">
                <button type="button" className="btn btn-sm" onClick={selectAll}>
                  Select All
                </button>
                <button type="button" className="btn btn-sm btn-secondary" onClick={deselectAll}>
                  Deselect All
                </button>
                <span className="muted" style={{ marginLeft: 'auto' }}>
                  {selectedModels.size} of {allModels.length} selected
                </span>
              </div>
              <p className="field-help" style={{ margin: '0 0 0.5rem 0' }}>
                {allowedHelpText}
              </p>
              <div className="model-filter-list">
                {groupedAllowed.providerOrder.map((provider) => (
                  <div key={provider}>
                    <div className="family-group-header">
                      {formatProviderDisplayName(provider)}
                    </div>
                    {Object.keys(groupedAllowed.providerGroups[provider]).sort().map((modelProvider) => (
                      <div key={modelProvider} className="model-group">
                        <div className="model-group-header">{modelProvider}</div>
                        {Object.keys(groupedAllowed.providerGroups[provider][modelProvider]).sort().map((groupName) => (
                          <div key={groupName}>
                            <div className="model-group-header" style={{ paddingLeft: '1rem', fontSize: '0.78rem' }}>
                              {groupName}
                            </div>
                            {groupedAllowed.providerGroups[provider][modelProvider][groupName].map((model) => (
                              <label
                                key={`${model.provider}:${model.id}`}
                                className="model-filter-item"
                                style={{
                                  paddingLeft: '1.5rem',
                                  backgroundColor: model.is_latest ? 'rgba(0,0,0,0.03)' : undefined,
                                  fontWeight: model.is_latest ? 500 : undefined,
                                }}
                              >
                                <input
                                  type="checkbox"
                                  checked={selectedModels.has(`${model.provider}::${model.id}`)}
                                  onChange={() => toggleModel(model)}
                                />
                                <span className="model-filter-name">
                                  {model.display_name || (model.id !== model.name ? model.id : model.name)}
                                  <span
                                    style={{
                                      marginLeft: '6px',
                                      fontSize: '0.7em',
                                      padding: '1px 4px',
                                      borderRadius: '4px',
                                      background: 'var(--bg-secondary, #2d2d2d)',
                                      color: 'var(--text-muted, #888)',
                                    }}
                                  >
                                    via {formatProviderDisplayName(model.provider)}
                                  </span>
                                  {model.is_latest && (
                                    <span
                                      style={{
                                        marginLeft: '6px',
                                        fontSize: '0.7em',
                                        padding: '1px 4px',
                                        borderRadius: '4px',
                                        background: '#e0e0e0',
                                        color: '#555',
                                      }}
                                    >
                                      LATEST
                                    </span>
                                  )}
                                </span>
                              </label>
                            ))}
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>

        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn"
            onClick={() => {
              void onSaveAllowed();
            }}
            disabled={modelsLoading || allModels.length === 0 || !!savingAllowed}
          >
            {savingAllowed ? 'Saving...' : 'Save Filter'}
          </button>
        </div>
      </div>
    </div>
  );
}
