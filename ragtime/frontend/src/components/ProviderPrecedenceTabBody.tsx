import { useState, useMemo, type DragEvent } from 'react';
import { GripVertical } from 'lucide-react';
import type { AvailableModel, ModelProviderPrecedence } from '@/types';
import { CHAT_MODEL_PROVIDER_LABELS } from '@/utils/modelDisplay';
import { normalizeProviderAlias } from '@/utils/modelProviders';

interface ProviderPrecedenceTabBodyProps {
  precedence: ModelProviderPrecedence;
  allModels: AvailableModel[];
  /** Set of "provider::id" keys currently allowed in chat (used to gray out filtered entries). */
  allowedScopedKeys: Set<string>;
  modelsLoading: boolean;
  dragProvider: string | null;
  setDragProvider: (provider: string | null) => void;
  onReorder: (sourceProvider: string, targetProvider: string) => void;
  onSetModelOverride: (modelId: string, provider: string | null) => void;
  onSetFamilyOverride: (family: string, provider: string | null) => void;
}

function providerLabel(provider: string): string {
  const norm = normalizeProviderAlias(provider) || provider;
  return CHAT_MODEL_PROVIDER_LABELS[norm] || norm;
}

export function ProviderPrecedenceTabBody({
  precedence,
  allModels,
  allowedScopedKeys,
  modelsLoading,
  dragProvider,
  setDragProvider,
  onReorder,
  onSetModelOverride,
  onSetFamilyOverride,
}: ProviderPrecedenceTabBodyProps) {
  const [filterText, setFilterText] = useState('');

  // Group models by id (across providers) and by family group (across providers)
  // so we can show overrides only for ambiguous cases.
  const { ambiguousModels, ambiguousFamilies } = useMemo(() => {
    const byModelId = new Map<string, Set<string>>();
    const byFamily = new Map<string, Set<string>>();
    const familyExamples = new Map<string, AvailableModel>();

    for (const model of allModels) {
      const provider = normalizeProviderAlias(model.provider) || model.provider;
      if (!byModelId.has(model.id)) byModelId.set(model.id, new Set());
      byModelId.get(model.id)!.add(provider);

      const family = model.group || '';
      if (family && !family.startsWith('Other')) {
        if (!byFamily.has(family)) byFamily.set(family, new Set());
        byFamily.get(family)!.add(provider);
        if (!familyExamples.has(family)) familyExamples.set(family, model);
      }
    }

    const ambiguousModelEntries: Array<{ modelId: string; providers: string[]; example: AvailableModel }> = [];
    for (const model of allModels) {
      const providers = byModelId.get(model.id);
      if (providers && providers.size >= 2) {
        if (!ambiguousModelEntries.find((e) => e.modelId === model.id)) {
          ambiguousModelEntries.push({ modelId: model.id, providers: [...providers], example: model });
        }
      }
    }

    const ambiguousFamilyEntries: Array<{ family: string; providers: string[] }> = [];
    for (const [family, providers] of byFamily) {
      if (providers.size >= 2) {
        ambiguousFamilyEntries.push({ family, providers: [...providers] });
      }
    }
    ambiguousFamilyEntries.sort((a, b) => a.family.localeCompare(b.family));

    return {
      ambiguousModels: ambiguousModelEntries.sort((a, b) => a.modelId.localeCompare(b.modelId)),
      ambiguousFamilies: ambiguousFamilyEntries,
    };
  }, [allModels]);

  if (modelsLoading) {
    return <p className="muted">Loading available models...</p>;
  }

  if (!allModels.length) {
    return (
      <p className="muted">
        No models available. Please configure API keys or local providers and save settings first.
      </p>
    );
  }

  const handleDragStart = (provider: string) => (e: DragEvent<HTMLDivElement>) => {
    setDragProvider(provider);
    e.dataTransfer.effectAllowed = 'move';
    try { e.dataTransfer.setData('text/plain', provider); } catch { /* some browsers throw on read */ }
  };
  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
  };
  const handleDrop = (target: string) => (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const source = dragProvider || (() => { try { return e.dataTransfer.getData('text/plain'); } catch { return null; } })();
    if (source) {
      onReorder(source, target);
    }
    setDragProvider(null);
  };
  const handleDragEnd = () => setDragProvider(null);

  const filterLower = filterText.toLowerCase();

  const filteredFamilies = ambiguousFamilies.filter(({ family, providers }) =>
    !filterText ||
    family.toLowerCase().includes(filterLower) ||
    providers.some(p => providerLabel(p).toLowerCase().includes(filterLower))
  );

  const filteredModels = ambiguousModels.filter(({ modelId, providers, example }) =>
    !filterText ||
    modelId.toLowerCase().includes(filterLower) ||
    (example.group && example.group.toLowerCase().includes(filterLower)) ||
    providers.some(p => providerLabel(p).toLowerCase().includes(filterLower))
  );

  return (
    <>
      <div className="model-filter-search" style={{ marginBottom: '1rem' }}>
        <input
          type="text"
          placeholder="Filter overrides..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
        />
      </div>

      <p className="field-help" style={{ margin: '0 0 0.75rem 0' }}>
        When the same model is offered by multiple providers, ragtime uses this ordering to pick which
        provider serves it. Drag a row to change provider precedence. The exact-model and family overrides
        below take precedence over the global ordering.
      </p>

      <h4 style={{ margin: '0.75rem 0 0.5rem' }}>Provider order</h4>
      <div className="model-filter-list" style={{ maxHeight: 240 }}>
        {precedence.providers.length === 0 && (
          <p className="muted" style={{ padding: '0.5rem' }}>
            No providers discovered yet. Configure at least one LLM provider to enable precedence.
          </p>
        )}
        {precedence.providers.map((provider, idx) => (
          <div
            key={provider}
            draggable
            onDragStart={handleDragStart(provider)}
            onDragOver={handleDragOver}
            onDrop={handleDrop(provider)}
            onDragEnd={handleDragEnd}
            style={{
              display: 'grid',
              gridTemplateColumns: 'auto auto minmax(0, 1fr) auto',
              alignItems: 'center',
              gap: 8,
              padding: '8px 10px',
              borderBottom: '1px solid var(--border-color, #3c3c3c)',
              cursor: 'grab',
              opacity: dragProvider === provider ? 0.5 : 1,
              background: dragProvider && dragProvider !== provider ? 'rgba(0,0,0,0.05)' : undefined,
            }}
          >
            <GripVertical size={14} style={{ color: 'var(--text-muted, #888)' }} />
            <span style={{
              fontSize: '0.75em',
              fontWeight: 600,
              minWidth: 22,
              color: 'var(--text-muted, #888)',
            }}>
              #{idx + 1}
            </span>
            <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{providerLabel(provider)}</div>
            <span style={{ fontSize: '0.7em', color: 'var(--text-muted, #888)' }}>
              {(allModels.filter((m) => normalizeProviderAlias(m.provider) === provider).length)} models
            </span>
          </div>
        ))}
      </div>

      <h4 style={{ margin: '1rem 0 0.5rem' }}>Family overrides</h4>
      {filteredFamilies.length === 0 ? (
        <p className="muted" style={{ fontSize: '0.85em', margin: '0 0 0.5rem 0' }}>
          {filterText ? 'No matching families found.' : 'No model families are currently shared across multiple providers.'}
        </p>
      ) : (
        <div className="model-filter-list" style={{ maxHeight: 200 }}>
          {filteredFamilies.map(({ family, providers }) => {
            const current = precedence.family_overrides[family] || '';
            return (
              <div key={family} style={{
                display: 'grid',
                gridTemplateColumns: 'minmax(0, 1fr) auto',
                alignItems: 'center',
                gap: 8,
                padding: '6px 10px',
                borderBottom: '1px solid var(--border-color, #3c3c3c)',
              }}>
                <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{family}</div>
                <select
                  value={current}
                  onChange={(e) => onSetFamilyOverride(family, e.target.value || null)}
                  style={{ minWidth: 160 }}
                >
                  <option value="">Use provider order</option>
                  {providers.map((p) => (
                    <option key={p} value={p}>{providerLabel(p)}</option>
                  ))}
                </select>
              </div>
            );
          })}
        </div>
      )}

      <details style={{ marginBottom: '16px', marginTop: '1.5rem' }}>
        <summary style={{ cursor: 'pointer', color: '#60a5fa', marginBottom: '8px', fontSize: '0.9rem', fontWeight: 600 }}>Advanced Settings</summary>

        <h4 style={{ margin: '0.5rem 0 0.5rem' }}>Exact-model overrides</h4>
        {filteredModels.length === 0 ? (
          <p className="muted" style={{ fontSize: '0.85em', margin: 0 }}>
            {filterText ? 'No matching models found.' : 'No exact model ids are currently shared across multiple providers.'}
          </p>
        ) : (
          <div className="model-filter-list" style={{ maxHeight: 240 }}>
            {filteredModels.map(({ modelId, providers, example }) => {
              const current = precedence.model_overrides[modelId] || '';
              const isAllowed = providers.some((p) => allowedScopedKeys.size === 0 || allowedScopedKeys.has(`${p}::${modelId}`));
              return (
                <div key={modelId} style={{
                  display: 'grid',
                  gridTemplateColumns: 'minmax(0, 1fr) auto',
                  alignItems: 'center',
                  gap: 8,
                  padding: '6px 10px',
                  borderBottom: '1px solid var(--border-color, #3c3c3c)',
                  opacity: isAllowed ? 1 : 0.55,
                }}>
                  <div style={{ fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {modelId}
                    {example.group && (
                      <span style={{ marginLeft: 8, fontSize: '0.7em', color: 'var(--text-muted, #888)' }}>
                        ({example.group})
                      </span>
                    )}
                    {!isAllowed && (
                      <span style={{ marginLeft: 8, fontSize: '0.7em', color: 'var(--text-muted, #888)' }}>
                        filtered out of chat
                      </span>
                    )}
                  </div>
                  <select
                    value={current}
                    onChange={(e) => onSetModelOverride(modelId, e.target.value || null)}
                    style={{ minWidth: 160 }}
                  >
                    <option value="">Use family / provider order</option>
                    {providers.map((p) => (
                      <option key={p} value={p}>{providerLabel(p)}</option>
                    ))}
                  </select>
                </div>
              );
            })}
          </div>
        )}
      </details>
    </>
  );
}
