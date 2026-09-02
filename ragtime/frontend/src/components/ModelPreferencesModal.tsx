import { useEffect, useMemo, useState } from 'react';
import { createPortal } from 'react-dom';

import { api } from '@/api';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';
import type { ModelPreferenceResponse, UserSpaceWorkspace } from '@/types';

import { ModelSelector } from './ModelSelector';

interface ModelPreferencesModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const WORKSPACE_PAGE_SIZE = 50;

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return 'Request failed';
}

function sortWorkspacesByName(workspaces: UserSpaceWorkspace[]): UserSpaceWorkspace[] {
  return [...workspaces].sort((a, b) => a.name.localeCompare(b.name));
}

function getSelectorValue(
  draft: string | null,
  preference: string | null,
  effective: string,
): string {
  return draft ?? preference ?? effective;
}

function getScopedModelSelectionKey(model: { provider?: string; id: string }): string {
  return model.provider ? `${model.provider}::${model.id}` : model.id;
}

export function ModelPreferencesModal({ isOpen, onClose }: ModelPreferencesModalProps) {
  const { models, loading: modelsLoading, error: modelsError } = useAvailableModels();
  const [generalPreference, setGeneralPreference] = useState<ModelPreferenceResponse | null>(null);
  const [generalDraft, setGeneralDraft] = useState<string | null>(null);
  const [generalLoading, setGeneralLoading] = useState(false);
  const [generalSaving, setGeneralSaving] = useState(false);
  const [generalError, setGeneralError] = useState<string | null>(null);

  const [workspaces, setWorkspaces] = useState<UserSpaceWorkspace[]>([]);
  const [workspacesLoading, setWorkspacesLoading] = useState(false);
  const [workspacesError, setWorkspacesError] = useState<string | null>(null);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState('');
  const [workspacePreference, setWorkspacePreference] = useState<ModelPreferenceResponse | null>(
    null,
  );
  const [workspaceDraft, setWorkspaceDraft] = useState<string | null>(null);
  const [workspaceLoading, setWorkspaceLoading] = useState(false);
  const [workspaceSaving, setWorkspaceSaving] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    let cancelled = false;

    async function loadGeneralPreference() {
      setGeneralLoading(true);
      setGeneralError(null);
      try {
        const response = await api.getModelPreferences();
        if (cancelled) return;
        setGeneralPreference(response);
        setGeneralDraft(response.user_default_chat_model ?? null);
      } catch (error) {
        if (cancelled) return;
        setGeneralError(getErrorMessage(error));
      } finally {
        if (!cancelled) {
          setGeneralLoading(false);
        }
      }
    }

    async function loadAllWorkspaces() {
      setWorkspacesLoading(true);
      setWorkspacesError(null);
      try {
        const items: UserSpaceWorkspace[] = [];
        let total = 0;
        do {
          const page = await api.listUserSpaceWorkspaces(items.length, WORKSPACE_PAGE_SIZE, false);
          items.push(...page.items);
          total = page.total;
        } while (items.length < total);
        if (cancelled) return;
        setWorkspaces(sortWorkspacesByName(items));
      } catch (error) {
        if (cancelled) return;
        setWorkspacesError(getErrorMessage(error));
      } finally {
        if (!cancelled) {
          setWorkspacesLoading(false);
        }
      }
    }

    setSelectedWorkspaceId('');
    setWorkspacePreference(null);
    setWorkspaceDraft(null);
    setWorkspaceError(null);
    void Promise.all([loadGeneralPreference(), loadAllWorkspaces()]);

    return () => {
      cancelled = true;
    };
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen || !selectedWorkspaceId) {
      return;
    }

    let cancelled = false;

    async function loadWorkspacePreference() {
      setWorkspaceLoading(true);
      setWorkspaceError(null);
      try {
        const response = await api.getModelPreferences(selectedWorkspaceId);
        if (cancelled) return;
        setWorkspacePreference(response);
        setWorkspaceDraft(response.workspace_default_chat_model ?? null);
      } catch (error) {
        if (cancelled) return;
        setWorkspacePreference(null);
        setWorkspaceDraft(null);
        setWorkspaceError(getErrorMessage(error));
      } finally {
        if (!cancelled) {
          setWorkspaceLoading(false);
        }
      }
    }

    void loadWorkspacePreference();
    return () => {
      cancelled = true;
    };
  }, [isOpen, selectedWorkspaceId]);

  const selectedWorkspace = useMemo(
    () => workspaces.find((workspace) => workspace.id === selectedWorkspaceId) ?? null,
    [selectedWorkspaceId, workspaces],
  );

  if (!isOpen) {
    return null;
  }

  const generalEffective = generalPreference?.effective_default_chat_model ?? '';
  const generalPreferenceValue = generalPreference?.user_default_chat_model ?? null;
  const generalSelectedModelId = generalPreference
    ? getSelectorValue(generalDraft, generalPreferenceValue, generalEffective)
    : '';
  const generalDirty = generalPreference
    ? generalDraft !== (generalPreference.user_default_chat_model ?? null)
    : false;

  const workspaceEffective = workspacePreference?.effective_default_chat_model ?? '';
  const workspacePreferenceValue = workspacePreference?.workspace_default_chat_model ?? null;
  const workspaceSelectedModelId = workspacePreference
    ? getSelectorValue(workspaceDraft, workspacePreferenceValue, workspaceEffective)
    : '';
  const workspaceDirty = workspacePreference
    ? workspaceDraft !== (workspacePreference.workspace_default_chat_model ?? null)
    : false;

  const catalogUnavailable = modelsLoading || models.length === 0 || Boolean(modelsError);
  const selectorsDisabled =
    modelsLoading || generalSaving || workspaceSaving || models.length === 0;

  async function saveGeneralPreference(nextModel: string | null) {
    setGeneralSaving(true);
    setGeneralError(null);
    try {
      const response = await api.updateModelPreference(nextModel);
      setGeneralPreference(response);
      setGeneralDraft(response.user_default_chat_model ?? null);
    } catch (error) {
      setGeneralError(getErrorMessage(error));
    } finally {
      setGeneralSaving(false);
    }
  }

  async function saveWorkspacePreference(nextModel: string | null) {
    if (!selectedWorkspaceId) {
      return;
    }
    setWorkspaceSaving(true);
    setWorkspaceError(null);
    try {
      const response = await api.updateModelPreference(nextModel, selectedWorkspaceId);
      setWorkspacePreference(response);
      setWorkspaceDraft(response.workspace_default_chat_model ?? null);
    } catch (error) {
      setWorkspaceError(getErrorMessage(error));
    } finally {
      setWorkspaceSaving(false);
    }
  }

  return createPortal(
    <div className="modal-overlay" onClick={onClose}>
      <div
        id="model-preferences-modal"
        className="modal-content modal-medium model-preferences-modal"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3>Preferences</h3>
          <button type="button" className="modal-close" onClick={onClose} aria-label="Close">
            &times;
          </button>
        </div>

        <div className="modal-body model-preferences-body">
          <section id="general-default-model-setting" className="model-preferences-section">
            <div className="model-preferences-section-header">
              <div>
                <h4>General default chat model</h4>
                <p className="field-help">Used for new chats unless a workspace override exists.</p>
              </div>
            </div>
            {generalError ? <div className="field-error">{generalError}</div> : null}
            {modelsError ? <div className="field-error">{modelsError}</div> : null}
            {generalLoading ? (
              <p className="field-help">Loading general preferences...</p>
            ) : generalPreference ? (
              <>
                <ModelSelector
                  models={models}
                  selectedModelId={generalSelectedModelId}
                  onModelChange={(modelId) => setGeneralDraft(modelId || null)}
                  getModelSelectionKey={getScopedModelSelectionKey}
                  disabled={selectorsDisabled}
                  loading={modelsLoading}
                  placeholder="General default chat model"
                  variant="full"
                />
                <p className="field-help model-preferences-summary">
                  {generalPreference.user_default_chat_model
                    ? 'Personal override saved.'
                    : `Inherited default: ${generalPreference.effective_default_chat_model}`}
                </p>
                <div className="modal-footer model-preferences-actions">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void saveGeneralPreference(null)}
                    disabled={generalSaving || generalPreferenceValue === null}
                    aria-label="Reset general default"
                  >
                    Reset
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => void saveGeneralPreference(generalDraft)}
                    disabled={generalSaving || catalogUnavailable || !generalDirty}
                    aria-label="Save general default"
                  >
                    {generalSaving ? 'Saving...' : 'Save'}
                  </button>
                </div>
              </>
            ) : null}
          </section>

          <section id="workspace-default-model-setting" className="model-preferences-section">
            <div className="model-preferences-section-header">
              <div>
                <h4>Workspace default chat model</h4>
                <p className="field-help">
                  Personal override for new chats created inside a selected workspace.
                </p>
              </div>
            </div>
            {workspacesError ? <div className="field-error">{workspacesError}</div> : null}
            <div className="form-group">
              <label htmlFor="model-preferences-workspace-picker">Workspace</label>
              <select
                id="model-preferences-workspace-picker"
                aria-label="Workspace"
                value={selectedWorkspaceId}
                onChange={(event) => setSelectedWorkspaceId(event.target.value)}
                disabled={workspacesLoading || workspaces.length === 0}
              >
                <option value="">Select a workspace</option>
                {workspaces.map((workspace) => (
                  <option key={workspace.id} value={workspace.id}>
                    {workspace.name}
                  </option>
                ))}
              </select>
            </div>
            {workspacesLoading ? <p className="field-help">Loading workspaces...</p> : null}
            {!workspacesLoading && workspaces.length === 0 ? (
              <p className="field-help">No workspaces available for workspace-specific defaults.</p>
            ) : null}
            {workspaceError ? <div className="field-error">{workspaceError}</div> : null}
            {selectedWorkspaceId && workspaceLoading ? (
              <p className="field-help">Loading workspace preferences...</p>
            ) : null}
            {selectedWorkspace && workspacePreference ? (
              <>
                <ModelSelector
                  models={models}
                  selectedModelId={workspaceSelectedModelId}
                  onModelChange={(modelId) => setWorkspaceDraft(modelId || null)}
                  getModelSelectionKey={getScopedModelSelectionKey}
                  disabled={selectorsDisabled}
                  loading={modelsLoading}
                  placeholder="Workspace default chat model"
                  variant="full"
                />
                <p className="field-help model-preferences-summary">
                  {workspacePreference.workspace_default_chat_model
                    ? 'Workspace override saved.'
                    : `Inherited default: ${workspacePreference.effective_default_chat_model}`}
                </p>
                <div className="modal-footer model-preferences-actions">
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => void saveWorkspacePreference(null)}
                    disabled={workspaceSaving || workspacePreferenceValue === null}
                    aria-label="Reset workspace default"
                  >
                    Reset
                  </button>
                  <button
                    type="button"
                    className="btn"
                    onClick={() => void saveWorkspacePreference(workspaceDraft)}
                    disabled={workspaceSaving || catalogUnavailable || !workspaceDirty}
                    aria-label="Save workspace default"
                  >
                    {workspaceSaving ? 'Saving...' : 'Save'}
                  </button>
                </div>
              </>
            ) : null}
          </section>
        </div>
      </div>
    </div>,
    document.body,
  );
}
