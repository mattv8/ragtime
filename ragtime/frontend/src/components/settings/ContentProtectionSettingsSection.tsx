import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';
import { AlertCircle, CheckCircle2, CircleHelp, Loader2, TestTube2 } from 'lucide-react';

import {
  contentProtectionApi,
  ContentProtectionApiError,
  requirementModeFor,
  type ContentProtectionCatalog,
  type ContentProtectionConfig,
  type ContentProtectionProfile,
  type ContentProtectionRequirementMode,
  type ContentProtectionTestResult,
  withRequirement,
} from '@/api/contentProtection';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';
import { ModelSelector } from '../ModelSelector';
import { Popover } from '../Popover';
import { SearchHighlightedText } from '../shared/SearchHighlightedText';
import { ContentProtectionProfileCard } from './ContentProtectionProfileCard';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

type SetupTab = 'model' | 'coverage' | 'profiles' | 'review';
type InspectTab = 'request' | 'content' | 'decisions';
type PreviewIdentity = `user:${string}` | 'public' | 'service';
const EMPTY_CATALOG: ContentProtectionCatalog = {
  users: [],
  groups: [],
  tools: [],
  mcp_routes: [],
  surfaces: [],
};
const SETUP_TABS: Array<{ id: SetupTab; label: string }> = [
  { id: 'model', label: 'Model' },
  { id: 'coverage', label: 'Coverage' },
  { id: 'profiles', label: 'Profiles' },
  { id: 'review', label: 'Review & save' },
];
const INSPECT_TABS: Array<{ id: InspectTab; label: string }> = [
  { id: 'request', label: 'Check a request' },
  { id: 'content', label: 'Test content' },
  { id: 'decisions', label: 'Recent decisions' },
];

function formatLatency(latency: number | undefined): string {
  return latency == null ? '' : ` · ${Math.round(latency * 1000)} ms`;
}
function cloneConfig(config: ContentProtectionConfig): ContentProtectionConfig {
  return structuredClone(config);
}
function profileSets(result: { profiles: ContentProtectionProfile[][] }): string {
  return (
    result.profiles.map((set) => set.map((profile) => profile.name).join(', ')).join(' / ') ||
    'None'
  );
}

function TabbedDialog<T extends string>({
  title,
  tabs,
  activeTab,
  setActiveTab,
  onClose,
  children,
  footer,
  hook,
  busy = false,
  error,
}: {
  title: string;
  tabs: Array<{ id: T; label: string }>;
  activeTab: T;
  setActiveTab: (tab: T) => void;
  onClose: () => void;
  children: ReactNode;
  footer: ReactNode;
  hook: string;
  busy?: boolean;
  error?: string | null;
}): JSX.Element {
  const dialogRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  const restoreFocus = useRef<HTMLElement | null>(
    document.activeElement instanceof HTMLElement ? document.activeElement : null,
  );
  useEffect(() => {
    const dialog = dialogRef.current;
    const previouslyFocused = restoreFocus.current;
    const focusable = () =>
      Array.from(
        dialog?.querySelectorAll<HTMLElement>(
          'button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])',
        ) || [],
      ).filter(
        (element) => element.tabIndex >= 0 && !element.hidden && element.offsetParent !== null,
      );
    focusable()[0]?.focus();
    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.defaultPrevented) return;
      if (event.key === 'Escape') {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== 'Tab') return;
      const items = focusable();
      if (!items.length) return;
      const first = items[0];
      const last = items[items.length - 1];
      if (!dialog?.contains(document.activeElement)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      previouslyFocused?.focus();
    };
  }, []);
  const moveTab = (event: ReactKeyboardEvent<HTMLButtonElement>, tab: T) => {
    const index = tabs.findIndex((item) => item.id === tab);
    let next = index;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    setActiveTab(tabs[next].id);
    document.getElementById(`${hook}-tab-${tabs[next].id}`)?.focus();
  };
  return createPortal(
    <div
      className="modal-overlay content-protection-modal-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (!busy && event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="modal-content content-protection-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${hook}-title`}
        data-content-protection-dialog={hook}
      >
        <div className="modal-header">
          <h3 id={`${hook}-title`}>{title}</h3>
          <button
            type="button"
            className="modal-close"
            aria-label={`Close ${title}`}
            disabled={busy}
            onClick={onClose}
          >
            ×
          </button>
        </div>
        <div className="content-protection-tabs" role="tablist" aria-label={`${title} steps`}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              id={`${hook}-tab-${tab.id}`}
              type="button"
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`${hook}-panel-${tab.id}`}
              tabIndex={activeTab === tab.id ? 0 : -1}
              disabled={busy}
              onClick={() => setActiveTab(tab.id)}
              onKeyDown={(event) => moveTab(event, tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div
          className="modal-body"
          id={`${hook}-panel-${activeTab}`}
          role="tabpanel"
          aria-labelledby={`${hook}-tab-${activeTab}`}
        >
          {error && (
            <p className="field-error" role="alert">
              {error}
            </p>
          )}
          <fieldset className="content-protection-dialog-controls" disabled={busy}>
            {children}
          </fieldset>
        </div>
        <div className="modal-footer">{footer}</div>
      </div>
    </div>,
    document.body,
  );
}

export function ContentProtectionSettingsSection({
  open,
  onToggle,
  searchQuery = '',
}: {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  searchQuery?: string;
}): JSX.Element {
  const availableModels = useAvailableModels();
  const refreshModelsRef = useRef(availableModels.refresh);
  refreshModelsRef.current = availableModels.refresh;
  const [saved, setSaved] = useState<ContentProtectionConfig | null>(null);
  const [catalog, setCatalog] = useState<ContentProtectionCatalog>(EMPTY_CATALOG);
  const [error, setError] = useState<string | null>(null);
  const [setup, setSetup] = useState<ContentProtectionConfig | null>(null);
  const [setupTab, setSetupTab] = useState<SetupTab>('model');
  const [inspectOpen, setInspectOpen] = useState(false);
  const [inspectTab, setInspectTab] = useState<InspectTab>('request');
  const [saving, setSaving] = useState(false);
  const [reloading, setReloading] = useState(false);
  const [saveConflict, setSaveConflict] = useState(false);
  const [readiness, setReadiness] = useState<ContentProtectionTestResult | null>(null);
  const [checkedModel, setCheckedModel] = useState<string | null>(null);
  const [readinessBusy, setReadinessBusy] = useState(false);
  const [readinessError, setReadinessError] = useState<string | null>(null);
  const readinessToken = useRef(0);
  const [identity, setIdentity] = useState<PreviewIdentity>('service');
  const [surface, setSurface] = useState('chat');
  const [route, setRoute] = useState('');
  const [includeTool, setIncludeTool] = useState(false);
  const [tool, setTool] = useState('');
  const [preview, setPreview] = useState<Awaited<
    ReturnType<typeof contentProtectionApi.preview>
  > | null>(null);
  const previewToken = useRef(0);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [sample, setSample] = useState('');
  const [sampleProfiles, setSampleProfiles] = useState<string[]>([]);
  const [testResult, setTestResult] = useState<ContentProtectionTestResult | null>(null);
  const [testBusy, setTestBusy] = useState(false);
  const testToken = useRef(0);
  const [decisions, setDecisions] = useState<
    Awaited<ReturnType<typeof contentProtectionApi.decisions>>['items'] | null
  >(null);
  const [decisionsBusy, setDecisionsBusy] = useState(false);
  const [decisionsError, setDecisionsError] = useState<string | null>(null);
  const decisionsRequested = useRef(false);
  const decisionsToken = useRef(0);

  const load = useCallback(async () => {
    try {
      const [config, nextCatalog] = await Promise.all([
        contentProtectionApi.getConfig(),
        contentProtectionApi.getCatalog(),
      ]);
      setSaved(config);
      setCatalog(nextCatalog);
      setError(null);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : 'Failed to load content protection settings',
      );
    }
  }, []);
  useEffect(() => {
    if (open) {
      void load();
      refreshModelsRef.current();
    }
  }, [load, open]);
  useEffect(() => {
    setSampleProfiles((current) =>
      current.filter((id) => saved?.profiles.some((profile) => profile.id === id)),
    );
    testToken.current += 1;
    setTestResult(null);
    setTestBusy(false);
  }, [saved]);
  const openSetup = (tab: SetupTab = 'model') => {
    if (!saved) return;
    setError(null);
    setSaveConflict(false);
    setReadiness(null);
    setReadinessError(null);
    setReadinessBusy(false);
    readinessToken.current += 1;
    setSetup(cloneConfig(saved));
    setSetupTab(tab);
  };
  const closeSetup = () => {
    if (!saving && !reloading) {
      readinessToken.current += 1;
      setReadinessBusy(false);
      setSetup(null);
      setError(null);
      setSaveConflict(false);
    }
  };
  const updateDraft = (change: Partial<ContentProtectionConfig>) =>
    setSetup((current) => current && { ...current, ...change });
  const changeModel = (classifier_model: string) => {
    readinessToken.current += 1;
    setReadiness(null);
    setReadinessError(null);
    setReadinessBusy(false);
    updateDraft({ classifier_model });
  };
  const save = async () => {
    if (!setup || !saved) return;
    setSaving(true);
    setError(null);
    setSaveConflict(false);
    try {
      const next = await contentProtectionApi.saveConfig(setup.revision, setup);
      setSaved(next);
      readinessToken.current += 1;
      setReadinessBusy(false);
      setSetup(null);
    } catch (caught) {
      const conflict = caught instanceof ContentProtectionApiError && caught.status === 409;
      setSaveConflict(conflict);
      setError(
        conflict
          ? 'This policy changed on the server. Reload saved settings and reconcile your draft before saving.'
          : caught instanceof Error
            ? caught.message
            : 'Failed to save content protection settings',
      );
    } finally {
      setSaving(false);
    }
  };
  const reloadSaved = async () => {
    setReloading(true);
    try {
      const [config, nextCatalog] = await Promise.all([
        contentProtectionApi.getConfig(),
        contentProtectionApi.getCatalog(),
      ]);
      setSaved(config);
      setCatalog(nextCatalog);
      setSetup(null);
      setError(null);
      setSaveConflict(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Failed to reload saved settings');
    } finally {
      setReloading(false);
    }
  };
  const checkReadiness = async () => {
    if (!setup?.classifier_model) return;
    const token = ++readinessToken.current;
    setReadiness(null);
    setReadinessError(null);
    setReadinessBusy(true);
    const model = setup.classifier_model;
    try {
      const result = await contentProtectionApi.readiness(setup);
      if (token === readinessToken.current) {
        setReadiness(result);
        setCheckedModel(result.code === 'ready' ? model : null);
      }
    } catch (caught) {
      if (token === readinessToken.current) {
        setCheckedModel((current) => (current === model ? null : current));
        setReadinessError(caught instanceof Error ? caught.message : 'Readiness check failed');
      }
    } finally {
      if (token === readinessToken.current) setReadinessBusy(false);
    }
  };
  const clearPreview = () => {
    previewToken.current += 1;
    setPreview(null);
    setPreviewBusy(false);
    setError(null);
  };
  const runPreview = async () => {
    const token = ++previewToken.current;
    setPreview(null);
    setPreviewBusy(true);
    setError(null);
    try {
      const result = await contentProtectionApi.preview({
        user_id: identity.startsWith('user:') ? identity.slice(5) : undefined,
        public: identity === 'public',
        surface,
        mcp_route: surface === 'mcp' ? route || undefined : undefined,
        tool_id: includeTool ? tool || undefined : undefined,
      });
      if (token === previewToken.current) setPreview(result);
    } catch (caught) {
      if (token === previewToken.current)
        setError(caught instanceof Error ? caught.message : 'Failed to check saved policy');
    } finally {
      if (token === previewToken.current) setPreviewBusy(false);
    }
  };
  const runTest = async () => {
    if (!saved?.classifier_model || !sample.trim()) return;
    const token = ++testToken.current;
    setTestBusy(true);
    setTestResult(null);
    try {
      const result = await contentProtectionApi.test(saved, sample, sampleProfiles);
      if (token === testToken.current) setTestResult(result);
    } catch (caught) {
      if (token === testToken.current)
        setError(caught instanceof Error ? caught.message : 'Sample test failed');
    } finally {
      if (token === testToken.current) setTestBusy(false);
    }
  };
  useEffect(() => {
    if (!inspectOpen || inspectTab !== 'decisions' || decisionsRequested.current) return;
    decisionsRequested.current = true;
    const token = ++decisionsToken.current;
    setDecisionsError(null);
    setDecisionsBusy(true);
    void contentProtectionApi
      .decisions()
      .then((result) => {
        if (token === decisionsToken.current) setDecisions(result.items);
      })
      .catch((caught) => {
        if (token === decisionsToken.current)
          setDecisionsError(
            caught instanceof Error ? caught.message : 'Failed to load recent decisions',
          );
      })
      .finally(() => {
        if (token === decisionsToken.current) setDecisionsBusy(false);
      });
  }, [inspectOpen, inspectTab]);
  const updateProfile = (id: string, change: Partial<ContentProtectionProfile>) =>
    setSetup(
      (current) =>
        current && {
          ...current,
          profiles: current.profiles.map((profile) =>
            profile.id === id ? { ...profile, ...change } : profile,
          ),
        },
    );
  const deleteProfile = (profile: ContentProtectionProfile) => {
    if (!setup) return;
    const affected = setup.group_profiles.filter((item) => item.profile_id === profile.id).length;
    if (affected) {
      setError(
        `Reassign or clear ${affected} affected group${affected === 1 ? '' : 's'} before deleting ${profile.name}.`,
      );
      return;
    }
    setSetup(
      (current) =>
        current && {
          ...current,
          profiles: current.profiles.filter((item) => item.id !== profile.id),
        },
    );
  };
  const addProfile = () =>
    setSetup(
      (current) =>
        current && {
          ...current,
          profiles: [
            ...current.profiles,
            {
              id: `profile_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`,
              name: 'New profile',
              level: 0,
              scope: 'Describe the information this profile permits.',
            },
          ],
        },
    );

  return (
    <SettingsAccordionSection
      id="content-protection"
      title="Content protection"
      open={open}
      onToggle={onToggle}
      status={saved?.enabled ? 'Enabled' : 'Disabled'}
    >
      <section
        id="settings-content-protection"
        className="content-protection-section"
        aria-label="Content protection settings"
      >
        {error && !setup && !inspectOpen && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        {!saved ? (
          <p className="field-help">Loading content protection settings…</p>
        ) : (
          <>
            <p className="content-protection-status">
              {saved.enabled
                ? 'Enabled: covered traffic is classified before release.'
                : 'Disabled: saved rules are ready for preparation but production traffic makes no classifier call.'}
            </p>
            <div className="content-protection-overview" data-content-protection-overview>
              <fieldset
                id="content-protection-model-summary"
                className="content-protection-summary"
              >
                <legend>
                  Classifier{' '}
                  <span
                    title={
                      checkedModel === saved.classifier_model && checkedModel
                        ? 'Selected model passed a check in this session'
                        : 'Open Configure model to check the selected model'
                    }
                  >
                    {checkedModel === saved.classifier_model && checkedModel ? (
                      <CheckCircle2
                        className="content-protection-status-icon is-ready"
                        aria-label="Saved model checked"
                      />
                    ) : (
                      <CircleHelp
                        className="content-protection-status-icon"
                        aria-label="Model not checked"
                      />
                    )}
                  </span>
                </legend>
                <p>{saved.classifier_model || 'No model selected'}</p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => openSetup('model')}
                >
                  Configure model
                </button>
              </fieldset>
              <fieldset
                id="content-protection-coverage-summary"
                className="content-protection-summary"
              >
                <legend>Coverage</legend>
                <p>
                  {saved.coverage_mode === 'all_supported_traffic'
                    ? 'All supported traffic'
                    : 'Selected scopes'}
                </p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => openSetup('coverage')}
                >
                  Configure coverage
                </button>
              </fieldset>
              <fieldset
                id="content-protection-profiles-summary"
                className="content-protection-summary"
              >
                <legend>Profiles</legend>
                <p>
                  {saved.profiles.length
                    ? `${saved.profiles.length} permitted information profile${saved.profiles.length === 1 ? '' : 's'}`
                    : 'No profiles configured'}
                </p>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => openSetup('profiles')}
                >
                  Configure profiles
                </button>
              </fieldset>
            </div>
            <div className="form-actions">
              <button type="button" className="btn" onClick={() => openSetup()}>
                Configure protection
              </button>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  setError(null);
                  clearPreview();
                  testToken.current += 1;
                  setTestBusy(false);
                  setTestResult(null);
                  decisionsToken.current += 1;
                  decisionsRequested.current = false;
                  setDecisions(null);
                  setDecisionsError(null);
                  setDecisionsBusy(false);
                  setInspectOpen(true);
                  setInspectTab('request');
                }}
              >
                Test & inspect
              </button>
            </div>
          </>
        )}
      </section>
      {setup && (
        <TabbedDialog
          title="Configure content protection"
          tabs={SETUP_TABS}
          activeTab={setupTab}
          setActiveTab={setSetupTab}
          onClose={closeSetup}
          hook="content-protection-setup"
          busy={saving || reloading}
          error={error}
          footer={
            <>
              {setupTab !== 'model' && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={saving || reloading}
                  onClick={() =>
                    setSetupTab(
                      SETUP_TABS[SETUP_TABS.findIndex((tab) => tab.id === setupTab) - 1].id,
                    )
                  }
                >
                  Back
                </button>
              )}
              {setupTab !== 'review' ? (
                <button
                  type="button"
                  className="btn"
                  disabled={saving || reloading}
                  onClick={() =>
                    setSetupTab(
                      SETUP_TABS[SETUP_TABS.findIndex((tab) => tab.id === setupTab) + 1].id,
                    )
                  }
                >
                  Next
                </button>
              ) : (
                <button
                  type="button"
                  className="btn"
                  disabled={saving || reloading}
                  onClick={() => void save()}
                >
                  {saving ? 'Saving…' : 'Save protection'}
                </button>
              )}
              {saveConflict && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={saving || reloading}
                  onClick={() => void reloadSaved()}
                >
                  {reloading ? 'Reloading…' : 'Reload and discard draft'}
                </button>
              )}
              <button
                type="button"
                className="btn btn-secondary"
                disabled={saving || reloading}
                onClick={closeSetup}
              >
                Cancel
              </button>
            </>
          }
        >
          {setupTab === 'model' && (
            <fieldset
              id="content-protection-classifier-fieldset"
              className="content-protection-fieldset"
            >
              <legend>
                Classifier model{' '}
                <button
                  type="button"
                  className={`content-protection-icon-action${readinessError ? ' is-error' : readiness ? ' is-ready' : ''}`}
                  title="Check selected model"
                  aria-label="Check selected model"
                  disabled={!setup.classifier_model || readinessBusy}
                  onClick={() => void checkReadiness()}
                >
                  {readinessBusy ? (
                    <Loader2 className="content-protection-spinner" aria-label="Checking model" />
                  ) : readinessError ? (
                    <AlertCircle aria-label="Model check failed" />
                  ) : readiness ? (
                    <CheckCircle2 aria-label="Model ready" />
                  ) : (
                    <CircleHelp aria-label="Model unchecked" />
                  )}
                </button>
              </legend>
              <ModelSelector
                models={availableModels.models || []}
                selectedModelId={setup.classifier_model || ''}
                onModelChange={changeModel}
                getModelSelectionKey={(model) => `${model.provider}::${model.id}`}
                loading={availableModels.loading || false}
                disabled={availableModels.loading || saving || false}
                placeholder="Select a curated model"
                variant="full"
              />
              <p className="field-help">
                {availableModels.error ||
                  (availableModels.loading
                    ? 'Loading curated classifier models…'
                    : 'The selected provider receives inspected content.')}
              </p>
              {readinessError && (
                <p className="field-help" role="status">
                  {readinessError}
                </p>
              )}
              <p className="field-help">
                Enabling protection or changing its model while enabled verifies the provider on
                save.
              </p>
            </fieldset>
          )}
          {setupTab === 'coverage' && (
            <>
              <div className="content-protection-dialog-grid">
                <div className="content-protection-coverage-field">
                  <span className="content-protection-coverage-label">
                    <label htmlFor="content-protection-coverage">Coverage</label>
                    <Popover
                      trigger="click"
                      position="bottom"
                      zIndexAboveTrigger
                      content={
                        <span>
                          All supported traffic applies classification broadly. Selected scopes adds
                          requirements only where configured. Individual user policies can require
                          or skip classification in either mode.
                        </span>
                      }
                    >
                      <button
                        type="button"
                        className="content-protection-icon-action"
                        aria-label="Coverage help"
                        title="Coverage help"
                      >
                        <CircleHelp />
                      </button>
                    </Popover>
                  </span>
                  <select
                    id="content-protection-coverage"
                    value={setup.coverage_mode}
                    onChange={(event) =>
                      updateDraft({
                        coverage_mode: event.target
                          .value as ContentProtectionConfig['coverage_mode'],
                      })
                    }
                  >
                    <option value="all_supported_traffic">All supported traffic</option>
                    <option value="selected_scopes">Selected scopes</option>
                  </select>
                </div>
              </div>
              {setup.enabled && setup.coverage_mode === 'selected_scopes' && (
                <section id="content-protection-app-areas">
                  <h4>App areas</h4>
                  <p className="field-help">
                    Requirements are additive. Choose No additional requirement to leave the
                    existing coverage decision unchanged.
                  </p>
                  <div id="content-protection-area-options" className="content-protection-rows">
                    {catalog.surfaces.map((item) => (
                      <label
                        className="content-protection-row"
                        key={item.id}
                        data-surface-id={item.id}
                      >
                        <span>{item.name}</span>
                        <select
                          aria-label={`Coverage for ${item.name}`}
                          value={requirementModeFor(setup, 'surface', item.id)}
                          onChange={(event) =>
                            updateDraft({
                              requirements: withRequirement(
                                setup,
                                'surface',
                                item.id,
                                event.target.value as ContentProtectionRequirementMode,
                              ).requirements,
                            })
                          }
                        >
                          <option value="require">Require classification</option>
                          <option value="inherit">No additional requirement</option>
                        </select>
                      </label>
                    ))}
                  </div>
                </section>
              )}
              {!setup.enabled && setup.coverage_mode === 'selected_scopes' && (
                <p className="field-help">
                  Enable protection in Review &amp; save to configure app areas.
                </p>
              )}
              <div id="content-protection-policy-links" className="content-protection-policy-links">
                <span className="field-help">Configure specific policies:</span>
                <a href="?view=users#user-policies" target="_blank" rel="noopener noreferrer">
                  <SearchHighlightedText text="User policies" query={searchQuery} />
                </a>
                <a href="?view=users#manage-groups" target="_blank" rel="noopener noreferrer">
                  <SearchHighlightedText text="Manage groups" query={searchQuery} />
                </a>
                <a href="?view=tools#tools-connections" target="_blank" rel="noopener noreferrer">
                  <SearchHighlightedText text="Tool access" query={searchQuery} />
                </a>
                <a
                  href="?view=settings#manage-mcp-routes"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  <SearchHighlightedText text="MCP routes" query={searchQuery} />
                </a>
                <span className="field-help content-protection-policy-links-note">
                  Opens in a new tab; your draft stays here.
                </span>
              </div>
            </>
          )}
          {setupTab === 'profiles' && (
            <>
              <p className="field-help">
                Profiles describe permitted information. Assign groups in Manage Groups; additional
                profiles add grants and do not narrow existing grants.
              </p>
              <div id="content-protection-profile-list" className="content-protection-profiles">
                {setup.profiles.map((profile) => {
                  const affected = setup.group_profiles.filter(
                    (item) => item.profile_id === profile.id,
                  ).length;
                  return (
                    <ContentProtectionProfileCard
                      key={profile.id}
                      profile={profile}
                      affectedGroups={affected}
                      onUpdate={updateProfile}
                      onDelete={deleteProfile}
                    />
                  );
                })}
              </div>
              <button type="button" className="btn btn-secondary" onClick={addProfile}>
                Add profile
              </button>
            </>
          )}
          {setupTab === 'review' && (
            <>
              <div className="settings-switch-card" data-content-protection-enable-switch>
                <div>
                  <strong>Enable content protection</strong>
                  <p className="field-help">
                    Turn on classification after reviewing saved coverage.
                  </p>
                </div>
                <label className="toggle-switch" aria-label="Enable content protection">
                  <input
                    type="checkbox"
                    checked={setup.enabled}
                    onChange={(event) => updateDraft({ enabled: event.target.checked })}
                  />
                  <span className="toggle-slider" />
                </label>
              </div>
              <p className="field-help">
                {setup.enabled
                  ? 'Enabled: covered traffic will be classified after you save.'
                  : 'Disabled: this saved configuration remains available for preparation.'}
              </p>
              <dl className="content-protection-review">
                <dt>Classifier</dt>
                <dd>{setup.classifier_model || 'No model selected'}</dd>
                <dt>Coverage</dt>
                <dd>
                  {setup.coverage_mode === 'all_supported_traffic'
                    ? 'All supported traffic'
                    : 'Selected scopes'}
                </dd>
                <dt>Profiles</dt>
                <dd>{setup.profiles.map((profile) => profile.name).join(', ') || 'None'}</dd>
              </dl>
            </>
          )}
        </TabbedDialog>
      )}
      {inspectOpen && (
        <TabbedDialog
          title="Test and inspect content protection"
          tabs={INSPECT_TABS}
          activeTab={inspectTab}
          setActiveTab={(tab) => {
            setError(null);
            setInspectTab(tab);
          }}
          onClose={() => {
            previewToken.current += 1;
            testToken.current += 1;
            decisionsToken.current += 1;
            setPreviewBusy(false);
            setTestBusy(false);
            setDecisionsBusy(false);
            setError(null);
            setInspectOpen(false);
          }}
          hook="content-protection-inspect"
          error={error}
          footer={
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => {
                previewToken.current += 1;
                testToken.current += 1;
                decisionsToken.current += 1;
                setPreviewBusy(false);
                setTestBusy(false);
                setDecisionsBusy(false);
                setError(null);
                setInspectOpen(false);
              }}
            >
              Close
            </button>
          }
        >
          {inspectTab === 'request' && (
            <>
              <p className="field-help">Checks use the saved policy, not an open setup draft.</p>
              <div className="content-protection-dialog-grid">
                <label>
                  Who is making the request?
                  <select
                    value={identity}
                    onChange={(event) => {
                      setIdentity(event.target.value as PreviewIdentity);
                      clearPreview();
                    }}
                  >
                    <option value="service">Service</option>
                    <option value="public">Public</option>
                    {catalog.users.map((item) => (
                      <option key={item.id} value={`user:${item.id}`}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Where does it run?
                  <select
                    value={surface}
                    onChange={(event) => {
                      setSurface(event.target.value);
                      clearPreview();
                    }}
                  >
                    {catalog.surfaces.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                    {!catalog.surfaces.some((item) => item.id === surface) && (
                      <option value={surface}>{surface}</option>
                    )}
                  </select>
                </label>
                {surface === 'mcp' && (
                  <label>
                    MCP route
                    <select
                      aria-label="MCP route"
                      value={route}
                      onChange={(event) => {
                        setRoute(event.target.value);
                        clearPreview();
                      }}
                    >
                      <option value="">No route</option>
                      {catalog.mcp_routes.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={includeTool}
                    onChange={(event) => {
                      setIncludeTool(event.target.checked);
                      clearPreview();
                    }}
                  />{' '}
                  Include a tool call
                </label>
                {includeTool && (
                  <label>
                    Tool
                    <select
                      aria-label="Tool"
                      value={tool}
                      onChange={(event) => {
                        setTool(event.target.value);
                        clearPreview();
                      }}
                    >
                      <option value="">No tool</option>
                      {catalog.tools.map((item) => (
                        <option key={item.id} value={item.id}>
                          {item.name}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={previewBusy}
                onClick={() => void runPreview()}
              >
                <TestTube2 aria-hidden="true" /> {previewBusy ? 'Checking…' : 'Check saved policy'}
              </button>
              {preview && (
                <section
                  id="content-protection-preview-result"
                  className="content-protection-result"
                  role="status"
                >
                  <strong>
                    {preview.required ? 'Classification required' : 'Classification not required'}
                  </strong>
                  <span>Profile sets: {profileSets(preview)}</span>
                  <details>
                    <summary>Technical details</summary>
                    <p>
                      Provenance:{' '}
                      {typeof preview.provenance === 'string'
                        ? preview.provenance
                        : JSON.stringify(preview.provenance)}
                    </p>
                  </details>
                </section>
              )}
            </>
          )}
          {inspectTab === 'content' && (
            <>
              <p className="field-help">
                This test uses the saved configuration. If no target profiles are selected, Standard
                fallback is used.
              </p>
              <textarea
                id="content-protection-sample"
                aria-label="Sample content"
                value={sample}
                maxLength={1048576}
                onChange={(event) => {
                  testToken.current += 1;
                  setSample(event.target.value);
                  setTestResult(null);
                  setTestBusy(false);
                  setError(null);
                }}
              />
              <fieldset>
                <legend>Target profiles</legend>
                {saved?.profiles.map((profile) => (
                  <label key={profile.id} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={sampleProfiles.includes(profile.id)}
                      onChange={() => {
                        testToken.current += 1;
                        setTestResult(null);
                        setTestBusy(false);
                        setError(null);
                        setSampleProfiles((current) =>
                          current.includes(profile.id)
                            ? current.filter((id) => id !== profile.id)
                            : [...current, profile.id],
                        );
                      }}
                    />
                    {profile.name}
                  </label>
                ))}
              </fieldset>
              <button
                type="button"
                className="btn btn-secondary"
                disabled={!saved?.classifier_model || !sample.trim() || testBusy}
                onClick={() => void runTest()}
              >
                {testBusy ? 'Testing…' : 'Test content'}
              </button>
              <p className="field-help">
                Text only: uninspectable files and payloads over 1 MiB are blocked as
                unclassifiable.
              </p>
              {testResult && (
                <p
                  id="content-protection-test-result"
                  className="content-protection-result"
                  role="status"
                >
                  {testResult.verdict || 'error'} · {testResult.code}
                  {testResult.reason ? ` · ${testResult.reason}` : ''}
                  {formatLatency(testResult.latency)}
                </p>
              )}
            </>
          )}
          {inspectTab === 'decisions' && (
            <>
              {decisionsBusy && <p className="field-help">Loading recent decisions…</p>}
              {decisionsError && (
                <p className="field-error" role="alert">
                  {decisionsError}
                </p>
              )}
              {decisions &&
                (decisions.length ? (
                  <ul className="content-protection-decisions">
                    {decisions.map((item) => (
                      <li key={item.request_id || item.id}>
                        {item.created_at || 'Unknown time'} · {item.surface || 'Unknown app area'} ·{' '}
                        {item.verdict || item.code || 'Unknown result'}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="field-help">No decision metadata available.</p>
                ))}
            </>
          )}
        </TabbedDialog>
      )}
    </SettingsAccordionSection>
  );
}
