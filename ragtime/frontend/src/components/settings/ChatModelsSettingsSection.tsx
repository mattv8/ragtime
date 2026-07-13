import type { Dispatch, SetStateAction } from 'react';
import type { AvailableModel, UpdateSettingsRequest } from '@/types';
import { ModelSelector } from '../ModelSelector';
import { MiniLoadingSpinner } from '../shared/MiniLoadingSpinner';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

export interface ChatModelsSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  filteredChatModels: AvailableModel[];
  manualDefaultChatModel: string | null;
  automaticDefaultChatModel: string | null;
  chatModelsLoading: boolean;
  toScopedModelIdentifier: (model: AvailableModel) => string;
  openModelFilterModal: () => void;
  openOpenapiModelModal: () => void;
  handleSaveLlm: () => void | Promise<void>;
  llmSaving: boolean;
}

export function ChatModelsSettingsSection(props: ChatModelsSettingsSectionProps): JSX.Element {
  const {
    open,
    onToggle,
    formData,
    setFormData,
    filteredChatModels,
    manualDefaultChatModel,
    automaticDefaultChatModel,
    chatModelsLoading,
    toScopedModelIdentifier,
    openModelFilterModal,
    openOpenapiModelModal,
    handleSaveLlm,
    llmSaving,
  } = props;

  return (
    <SettingsAccordionSection id="chat-models" title="Chat Models" open={open} onToggle={onToggle}>
      <fieldset id="setting-chat_models">
        <legend>Chat Models</legend>
        <p className="fieldset-help">
          Choose which models appear in chat and which model is selected by default.
        </p>

        <div className="form-row-3">
          {/* Chat Model Filter */}
          <div className="form-group">
            <label>Chat Models</label>
            <button
              type="button"
              className="btn btn-secondary settings-control-height"
              onClick={openModelFilterModal}
            >
              Configure Chat Models
            </button>
            <p className="field-help">
              Limit which models appear in the Chat view dropdown. Includes all configured providers
              (OpenAI, Anthropic, OpenRouter, Ollama, llama.cpp, GitHub Copilot, OpenAI Codex).
            </p>
          </div>

          {/* Default Chat Model configuration */}
          <div className="form-group">
            <label>
              Default Chat Model
              {chatModelsLoading && (
                <>
                  {' '}
                  <MiniLoadingSpinner variant="icon" size={12} title="Loading models..." />
                </>
              )}
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
              <div style={{ flex: 1 }}>
                <ModelSelector
                  models={filteredChatModels}
                  selectedModelId={manualDefaultChatModel ?? automaticDefaultChatModel ?? ''}
                  onModelChange={(selectedValue) =>
                    setFormData({
                      ...formData,
                      default_chat_model: selectedValue || null,
                    })
                  }
                  getModelSelectionKey={toScopedModelIdentifier}
                  disabled={chatModelsLoading || filteredChatModels.length === 0}
                  loading={chatModelsLoading}
                  placeholder="Select default chat model"
                  variant="full"
                  triggerClassName="settings-control-height"
                />
              </div>
              {manualDefaultChatModel && (
                <button
                  type="button"
                  className="btn btn-secondary settings-control-height"
                  style={{ padding: '0 0.5rem', fontSize: '0.85em', whiteSpace: 'nowrap' }}
                  title="Reset to default model"
                  onClick={() => setFormData({ ...formData, default_chat_model: null })}
                >
                  Reset
                </button>
              )}
            </div>
            <p className="field-help">
              {manualDefaultChatModel
                ? 'Manually selected. Click Reset to use the default.'
                : 'Using the default model. Select a different model to override.'}
            </p>
          </div>

          {/* OpenAPI Models configuration */}
          <div className="form-group">
            <label>OpenAPI Models</label>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
              <label
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.4rem',
                  cursor: 'pointer',
                  fontSize: '0.9em',
                  margin: 0,
                  whiteSpace: 'nowrap',
                }}
              >
                <input
                  type="checkbox"
                  checked={formData.openapi_sync_chat_models !== false}
                  onChange={(e) =>
                    setFormData({ ...formData, openapi_sync_chat_models: e.target.checked })
                  }
                />
                Mirror Chat Models
              </label>
              {formData.openapi_sync_chat_models === false && (
                <button
                  type="button"
                  className="btn btn-secondary settings-control-height"
                  onClick={openOpenapiModelModal}
                >
                  Configure OpenAPI Models
                </button>
              )}
            </div>
            <p className="field-help">
              {formData.openapi_sync_chat_models !== false
                ? 'The /v1/models endpoint returns the same models as Chat Models above.'
                : 'Configure a separate list of models exposed via the /v1/models endpoint for external clients.'}
            </p>
          </div>
        </div>

        <div className="form-group" id="setting-available_models_cache_enabled">
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.4rem',
              cursor: 'pointer',
              fontSize: '0.9em',
              margin: 0,
              whiteSpace: 'nowrap',
            }}
          >
            <input
              type="checkbox"
              checked={formData.available_models_cache_enabled !== false}
              onChange={(e) =>
                setFormData({ ...formData, available_models_cache_enabled: e.target.checked })
              }
            />
            Cache Model Discovery
          </label>
          <p className="field-help">
            {formData.available_models_cache_enabled !== false
              ? 'Model lists from providers are cached briefly (default 30s) so repeated page loads stay fast. Saved settings changes always refresh immediately.'
              : 'Caching disabled: every request performs live provider discovery. Model pickers may load noticeably slower.'}
          </p>
        </div>

        <div className="form-actions">
          <button type="button" className="btn" onClick={handleSaveLlm} disabled={llmSaving}>
            {llmSaving ? 'Saving...' : 'Save Chat Model Settings'}
          </button>
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
