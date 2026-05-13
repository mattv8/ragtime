import { useState, useMemo } from 'react';
import type { AvailableModel, ModelProviderPrecedence } from '@/types';
import { ProviderPrecedenceTabBody } from './ProviderPrecedenceTabBody';

const PROVIDER_LABELS: Record<string, string> = {
  openai: 'OpenAI',
  anthropic: 'Anthropic',
  openrouter: 'OpenRouter',
  ollama: 'Ollama',
  github_copilot: 'GitHub Copilot',
  github_models: 'GitHub Copilot',
  llama_cpp: 'llama.cpp',
  lmstudio: 'LM Studio',
  omlx: 'OMLX',
};

export interface ModelFilterModalProps {
  isOpen: boolean;
  title: string;
  onClose: () => void;
  /** All discovered models across providers. */
  allModels: AvailableModel[];
  modelsLoading: boolean;
  /** Currently allowed scoped model ids ("provider::id"). Empty set = all allowed. */
  selectedModels: Set<string>;
  /** Provider precedence draft (controlled by parent so it can persist on save). */
  precedence: ModelProviderPrecedence;
  setPrecedence: (next: ModelProviderPrecedence) => void;
  /** Helper text shown below the filter actions on the Allowed tab. */
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
  /** Persist provider precedence. */
  onSavePrecedence: () => Promise<void> | void;
  savingPrecedence?: boolean;
}

export function ModelFilterModal({
  isOpen,
  title,
  onClose,
  allModels,
  modelsLoading,
  selectedModels,
  precedence,
  setPrecedence,
  allowedHelpText,
  toggleModel,
  selectAll,
  deselectAll,
  onSaveAllowed,
  savingAllowed,
  onSavePrecedence,
  savingPrecedence,
}: ModelFilterModalProps) {
  const [tab, setTab] = useState<'allowed' | 'precedence'>('allowed');
  const [filterText, setFilterText] = useState('');
  const [dragProvider, setDragProvider] = useState<string | null>(null);

  const filteredAllowed = useMemo(() => {
    const lc = filterText.toLowerCase();
    return allModels.filter(
      (m) =>
        filterText === '' ||
        m.name.toLowerCase().includes(lc) ||
        m.id.toLowerCase().includes(lc) ||
        m.provider.toLowerCase().includes(lc),
    );
  }, [allModels, filterText]);

  const groupedAllowed = useMemo(() => {
    const providerOrder: string[] = [];
    const providerGroups: Record<string, Record<string, AvailableModel[]>> = {};
    filteredAllowed.forEach((m) => {
      if (!providerGroups[m.provider]) {
        providerGroups[m.provider] = {};
        providerOrder.push(m.provider);
      }
      const g = m.group || 'Other';
      if (!providerGroups[m.provider][g]) providerGroups[m.provider][g] = [];
      providerGroups[m.provider][g].push(m);
    });
    return { providerOrder, providerGroups };
  }, [filteredAllowed]);

  const reorderProvider = (sourceProvider: string, targetProvider: string) => {
    if (sourceProvider === targetProvider) return;
    const next = [...precedence.providers];
    const fromIdx = next.indexOf(sourceProvider);
    const toIdx = next.indexOf(targetProvider);
    if (fromIdx < 0 || toIdx < 0) return;
    next.splice(fromIdx, 1);
    next.splice(toIdx, 0, sourceProvider);
    setPrecedence({ ...precedence, providers: next });
  };

  const setModelOverride = (modelId: string, provider: string | null) => {
    const next = { ...precedence.model_overrides };
    if (provider) next[modelId] = provider;
    else delete next[modelId];
    setPrecedence({ ...precedence, model_overrides: next });
  };

  const setFamilyOverride = (family: string, provider: string | null) => {
    const next = { ...precedence.family_overrides };
    if (provider) next[family] = provider;
    else delete next[family];
    setPrecedence({ ...precedence, family_overrides: next });
  };

  if (!isOpen) return null;

  const tabButtonStyle = (active: boolean): React.CSSProperties => ({
    background: 'transparent',
    border: 'none',
    borderBottom: active ? '2px solid var(--color-accent)' : '2px solid transparent',
    padding: '8px 16px',
    cursor: 'pointer',
    color: active ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
    fontSize: 14,
    fontWeight: 600,
    transition: 'color 0.15s, border-color 0.15s',
  });

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-medium" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header" style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
          <div style={{ display: 'flex', gap: 0, flex: 1 }}>
            <button type="button" style={tabButtonStyle(tab === 'allowed')} onClick={() => setTab('allowed')}>
              {title}
            </button>
            <button type="button" style={tabButtonStyle(tab === 'precedence')} onClick={() => setTab('precedence')}>
              Provider Precedence
            </button>
          </div>
          <button className="modal-close" onClick={onClose}>
            &times;
          </button>
        </div>

        <div className="modal-body">
          {tab === 'allowed' ? (
            modelsLoading ? (
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
                        {PROVIDER_LABELS[provider] || provider}
                      </div>
                      {Object.keys(groupedAllowed.providerGroups[provider]).map((groupName) => (
                        <div key={groupName} className="model-group">
                          <div className="model-group-header">
                            {groupName}
                          </div>
                          {groupedAllowed.providerGroups[provider][groupName].map((model) => (
                            <label
                              key={`${model.provider}:${model.id}`}
                              className="model-filter-item"
                              style={{
                                paddingLeft: '1rem',
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
                                {model.id !== model.name ? model.id : model.name}
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
                                  via {PROVIDER_LABELS[model.provider] || model.provider}
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
              </>
            )
          ) : (
            <ProviderPrecedenceTabBody
              precedence={precedence}
              allModels={allModels}
              allowedScopedKeys={selectedModels}
              modelsLoading={modelsLoading}
              dragProvider={dragProvider}
              setDragProvider={setDragProvider}
              onReorder={reorderProvider}
              onSetModelOverride={setModelOverride}
              onSetFamilyOverride={setFamilyOverride}
            />
          )}
        </div>

        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          {tab === 'allowed' ? (
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
          ) : (
            <button
              type="button"
              className="btn"
              onClick={() => {
                void onSavePrecedence();
              }}
              disabled={modelsLoading || !!savingPrecedence}
            >
              {savingPrecedence ? 'Saving...' : 'Save Precedence'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
