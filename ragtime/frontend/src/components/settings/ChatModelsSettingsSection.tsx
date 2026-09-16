import { useEffect, useState, type Dispatch, type SetStateAction } from 'react';
import { api } from '@/api';
import type { AvailableModel, OpenRouterCreditStatus, UpdateSettingsRequest } from '@/types';
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
  isAdmin: boolean;
  hasManagementApiKey: boolean;
}

function OpenRouterCreditMonitor({
  formData,
  setFormData,
  hasManagementApiKey,
}: Pick<ChatModelsSettingsSectionProps, 'formData' | 'setFormData'> & {
  hasManagementApiKey: boolean;
}): JSX.Element {
  const [status, setStatus] = useState<OpenRouterCreditStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void api
      .getOpenRouterCreditStatus()
      .then((nextStatus) => {
        if (!cancelled) {
          setStatus(nextStatus);
          setStatusError(null);
        }
      })
      .catch(() => {
        if (!cancelled) setStatusError('Credit status is currently unavailable.');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const stateLabel = status ? status.state.replace(/_/g, ' ') : 'checking';
  const stateIsAlert =
    status?.state === 'low' || status?.state === 'exhausted' || status?.state === 'error';

  return (
    <div className="form-group" id="setting-openrouter-credit-monitor">
      <label className="chat-toggle-control">
        <span className="toggle-switch">
          <input
            type="checkbox"
            role="switch"
            aria-label="Monitor OpenRouter credits"
            checked={formData.openrouter_credit_monitor_enabled === true}
            onChange={(event) =>
              setFormData({ ...formData, openrouter_credit_monitor_enabled: event.target.checked })
            }
          />
          <span className="toggle-slider" />
        </span>
        <span>Monitor OpenRouter credits</span>
      </label>
      <p className="field-help">
        Alert administrators when configured OpenRouter credit sources are low. This monitor never
        purchases credits or sends provider requests while disabled.
      </p>
      <div className="form-row-3">
        <div className="form-group">
          <label htmlFor="openrouter-low-credit-threshold">Low credit threshold (USD)</label>
          <input
            id="openrouter-low-credit-threshold"
            type="number"
            min="0"
            step="0.01"
            value={formData.openrouter_low_credit_threshold_usd ?? 5}
            onChange={(event) =>
              setFormData({
                ...formData,
                openrouter_low_credit_threshold_usd: Math.max(0, Number(event.target.value)),
              })
            }
          />
        </div>
        <div className="form-group">
          <label htmlFor="openrouter-management-api-key">Management API key (optional)</label>
          <input
            id="openrouter-management-api-key"
            type="password"
            autoComplete="new-password"
            placeholder="Leave blank to keep the configured key"
            value={formData.openrouter_management_api_key ?? ''}
            onChange={(event) =>
              setFormData({ ...formData, openrouter_management_api_key: event.target.value })
            }
          />
          <p className="field-help">
            {hasManagementApiKey
              ? 'A management key is configured. Leave this blank to preserve it, or clear it explicitly.'
              : 'Used only to check wallet credits; it is stored as a secret.'}
          </p>
          {hasManagementApiKey && formData.openrouter_management_api_key === undefined && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setFormData({ ...formData, openrouter_management_api_key: '' })}
            >
              Clear management API key
            </button>
          )}
        </div>
      </div>
      <p
        className={stateIsAlert ? 'userspace-error' : 'field-help'}
        role={stateIsAlert ? 'alert' : undefined}
      >
        Credit monitor status: {stateLabel}
        {status?.stale ? ' (stale)' : ''}
        {status?.warning ? `. ${status.warning}` : ''}
      </p>
      {statusError && <p className="field-help">{statusError}</p>}
    </div>
  );
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
    isAdmin,
    hasManagementApiKey,
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

          <div className="form-group" id="setting-userspace-build-model">
            <label>Builder Model</label>
            <ModelSelector
              models={filteredChatModels}
              selectedModelId={formData.userspace_build_model ?? ''}
              onModelChange={(selectedValue) =>
                setFormData({ ...formData, userspace_build_model: selectedValue || null })
              }
              getModelSelectionKey={toScopedModelIdentifier}
              disabled={chatModelsLoading || filteredChatModels.length === 0}
              loading={chatModelsLoading}
              placeholder="Use normal model defaults"
              variant="full"
              triggerClassName="settings-control-height"
            />
            <p className="field-help">
              Optional model for new User Space build tasks. It uses the live allowed model catalog
              and changes take effect only after saving below.
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
        {isAdmin && (
          <OpenRouterCreditMonitor
            formData={formData}
            setFormData={setFormData}
            hasManagementApiKey={hasManagementApiKey}
          />
        )}
      </fieldset>
    </SettingsAccordionSection>
  );
}
