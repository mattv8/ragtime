import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import { createPortal } from 'react-dom';
import { CheckCircle2, CircleHelp, Loader2, TestTube2 } from 'lucide-react';
import { api } from '@/api/client';
import {
  accessLevelSets,
  contentProtectionApi,
  requirementModeFor,
  withRequirement,
  type AccessLevel,
  type ContentCategory,
  type ContentProtectionCatalog,
  type ContentProtectionConfig,
  type ContentProtectionReadinessResult,
  type ContentProtectionTestResult,
} from '@/api/contentProtection';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';
import { ModelSelector } from '../ModelSelector';
import { Popover } from '../Popover';
import { SearchHighlightedText } from '../shared/SearchHighlightedText';
import { ContentProtectionAccessLevelCard } from './ContentProtectionAccessLevelCard';
import { ContentProtectionCategoryCard } from './ContentProtectionCategoryCard';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

type SetupTab = 'classifier' | 'categories' | 'access_levels' | 'coverage' | 'review';
type InspectTab = 'request' | 'content' | 'decisions';
type PreviewIdentity = `user:${string}` | 'public' | 'service';
const SETUP_TABS: { id: SetupTab; label: string }[] = [
  { id: 'classifier', label: 'Classifier' },
  { id: 'categories', label: 'Categories' },
  { id: 'access_levels', label: 'Access levels' },
  { id: 'coverage', label: 'Coverage' },
  { id: 'review', label: 'Review & save' },
];
const INSPECT_TABS: { id: InspectTab; label: string }[] = [
  { id: 'request', label: 'Check a request' },
  { id: 'content', label: 'Test content' },
  { id: 'decisions', label: 'Recent decisions' },
];
const EMPTY_CATALOG: ContentProtectionCatalog = {
  users: [],
  groups: [],
  tools: [],
  mcp_routes: [],
  surfaces: [],
};
const thresholdFor = (config: ContentProtectionConfig, category: ContentCategory) =>
  category.threshold_override ??
  { strict: 0.25, balanced: 0.5, permissive: 0.75 }[config.strictness];
const cloneConfig = (config: ContentProtectionConfig) => structuredClone(config);
const classifierSignature = (config: ContentProtectionConfig) =>
  JSON.stringify({
    backend: config.classifier.backend,
    transport: config.classifier.jev.transport,
    model: config.classifier.jev.model,
    llmModel: config.classifier.llm_model,
  });
const validationError = (config: ContentProtectionConfig): string | null => {
  if (config.classifier.backend === 'llm' && !config.classifier.llm_model)
    return 'Select a generic LLM model before saving.';
  const invalidCategory = config.categories.find(
    (category) =>
      !category.system &&
      (!category.name.trim() || !category.description.trim() || !category.denial_message.trim()),
  );
  if (invalidCategory)
    return `Complete the name, description, and denial message for ${invalidCategory.name || 'each category'}.`;
  const invalidLevel = config.access_levels.find((level) => !level.name.trim());
  return invalidLevel ? 'Each access level needs a name.' : null;
};
function Tabs<T extends string>({
  tabs,
  active,
  onChange,
  hook,
  label,
}: {
  tabs: { id: T; label: string }[];
  active: T;
  onChange: (tab: T) => void;
  hook: string;
  label: string;
}) {
  const moveTab = (event: KeyboardEvent<HTMLButtonElement>, current: T) => {
    const index = tabs.findIndex((tab) => tab.id === current);
    const next =
      event.key === 'ArrowRight'
        ? (index + 1) % tabs.length
        : event.key === 'ArrowLeft'
          ? (index - 1 + tabs.length) % tabs.length
          : event.key === 'Home'
            ? 0
            : event.key === 'End'
              ? tabs.length - 1
              : index;
    if (next === index) return;
    event.preventDefault();
    onChange(tabs[next].id);
    document.getElementById(`${hook}-tab-${tabs[next].id}`)?.focus();
  };
  return (
    <div className="content-protection-tabs" role="tablist" aria-label={label}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          id={`${hook}-tab-${tab.id}`}
          type="button"
          role="tab"
          aria-selected={active === tab.id}
          aria-controls={`${hook}-panel-${tab.id}`}
          tabIndex={active === tab.id ? 0 : -1}
          onClick={() => onChange(tab.id)}
          onKeyDown={(event) => moveTab(event, tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
function Dialog({
  title,
  children,
  onClose,
  footer,
  error,
  busy = false,
  hook,
  activeTab,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  footer: React.ReactNode;
  error: string | null;
  busy?: boolean;
  hook: string;
  activeTab: string;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  const busyRef = useRef(busy);
  closeRef.current = onClose;
  busyRef.current = busy;
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
      if (event.key === 'Escape' && !busyRef.current) {
        event.preventDefault();
        closeRef.current();
        return;
      }
      if (event.key !== 'Tab') return;
      const items = focusable();
      if (!items.length) return;
      const [first] = items;
      const last = items[items.length - 1];
      if (
        !dialog?.contains(document.activeElement) ||
        (event.shiftKey && document.activeElement === first) ||
        (!event.shiftKey && document.activeElement === last)
      ) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => {
      document.removeEventListener('keydown', onKeyDown);
      previouslyFocused?.focus();
    };
  }, []);
  return createPortal(
    <div
      className="modal-overlay content-protection-modal-overlay"
      role="presentation"
      onMouseDown={(event) => {
        if (!busyRef.current && event.target === event.currentTarget) onClose();
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
            className="modal-close"
            type="button"
            aria-label={`Close ${title}`}
            disabled={busy}
            onClick={onClose}
          >
            ×
          </button>
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
          {children}
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
  refreshKey = 0,
}: {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  searchQuery?: string;
  refreshKey?: number;
}): JSX.Element {
  const availableModels = useAvailableModels();
  const refreshModelsRef = useRef(availableModels.refresh);
  refreshModelsRef.current = availableModels.refresh;
  const [saved, setSaved] = useState<ContentProtectionConfig | null>(null);
  const [catalog, setCatalog] = useState(EMPTY_CATALOG);
  const [setup, setSetup] = useState<ContentProtectionConfig | null>(null);
  const [tab, setTab] = useState<SetupTab>('classifier');
  const [inspect, setInspect] = useState(false);
  const [inspectTab, setInspectTab] = useState<InspectTab>('request');
  const [setupError, setSetupError] = useState<string | null>(null);
  const [inspectError, setInspectError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [readiness, setReadiness] = useState<ContentProtectionReadinessResult | null>(null);
  const [readinessSignature, setReadinessSignature] = useState<string | null>(null);
  const [readinessBusy, setReadinessBusy] = useState(false);
  const [typesafeKey, setTypesafeKey] = useState('');
  const [keyBusy, setKeyBusy] = useState(false);
  const [preview, setPreview] = useState<Awaited<
    ReturnType<typeof contentProtectionApi.preview>
  > | null>(null);
  const [identity, setIdentity] = useState<PreviewIdentity>('service');
  const [surface, setSurface] = useState('chat');
  const [mcpRoute, setMcpRoute] = useState('');
  const [toolId, setToolId] = useState('');
  const [sample, setSample] = useState('');
  const [sampleLevelIds, setSampleLevelIds] = useState<string[]>([]);
  const [sampleUserId, setSampleUserId] = useState('');
  const [testResult, setTestResult] = useState<ContentProtectionTestResult | null>(null);
  const [testBusy, setTestBusy] = useState(false);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [decisions, setDecisions] = useState<
    Awaited<ReturnType<typeof contentProtectionApi.decisions>>['items'] | null
  >(null);
  const [decisionsBusy, setDecisionsBusy] = useState(false);
  const [decisionsError, setDecisionsError] = useState<string | null>(null);
  const requestToken = useRef(0);
  const readinessToken = useRef(0);
  const previewToken = useRef(0);
  const [saveConflict, setSaveConflict] = useState(false);
  const load = useCallback(async () => {
    try {
      const [config, nextCatalog] = await Promise.all([
        contentProtectionApi.getConfig(),
        contentProtectionApi.getCatalog(),
      ]);
      setSaved(config);
      setCatalog(nextCatalog);
      setLoadError(null);
    } catch (caught) {
      setLoadError(
        caught instanceof Error ? caught.message : 'Failed to load content protection settings',
      );
    }
  }, []);
  useEffect(() => {
    if (open) {
      void load();
      refreshModelsRef.current();
    }
  }, [load, open, refreshKey]);
  const update = (change: Partial<ContentProtectionConfig>) => {
    if (change.classifier) {
      readinessToken.current += 1;
      setReadiness(null);
      setReadinessSignature(null);
      setReadinessBusy(false);
    }
    setSetup((current) => current && { ...current, ...change });
  };
  const openSetup = (nextTab: SetupTab = 'classifier') => {
    if (!saved) return;
    setSetup(cloneConfig(saved));
    setTab(nextTab);
    setSetupError(null);
    setReadiness(null);
    setReadinessSignature(null);
    readinessToken.current += 1;
    setSaveConflict(false);
  };
  const save = async () => {
    if (!setup) return;
    const invalid = validationError(setup);
    if (invalid) return setSetupError(invalid);
    setSaving(true);
    try {
      const next = await contentProtectionApi.saveConfig(setup.revision, setup);
      setSaved(next);
      setSetup(null);
      setReadiness(null);
      setReadinessSignature(null);
    } catch (caught) {
      setSaveConflict(caught instanceof Error && 'status' in caught && caught.status === 409);
      setSetupError(
        caught instanceof Error && 'status' in caught && caught.status === 409
          ? 'This policy changed on the server. Reload saved settings and reconcile your draft before saving.'
          : caught instanceof Error
            ? caught.message
            : 'Failed to save content protection',
      );
    } finally {
      setSaving(false);
    }
  };
  const reloadSaved = async () => {
    if (saving) return;
    setSaving(true);
    try {
      const [config, nextCatalog] = await Promise.all([
        contentProtectionApi.getConfig(),
        contentProtectionApi.getCatalog(),
      ]);
      setSaved(config);
      setCatalog(nextCatalog);
      setSetup(null);
      setSaveConflict(false);
      setSetupError(null);
    } catch (caught) {
      setSetupError(caught instanceof Error ? caught.message : 'Failed to reload saved settings');
    } finally {
      setSaving(false);
    }
  };
  const saveKey = async (field: 'typesafe_api_key', value: string) => {
    if (!value.trim()) return;
    setKeyBusy(true);
    try {
      await api.updateSettings({ [field]: value.trim() });
      setTypesafeKey('');
      setCatalog(await contentProtectionApi.getCatalog());
      readinessToken.current += 1;
      setReadiness(null);
      setReadinessSignature(null);
    } catch (caught) {
      setSetupError(caught instanceof Error ? caught.message : 'Failed to save classifier key');
    } finally {
      setKeyBusy(false);
    }
  };
  const checkReadiness = async () => {
    if (!setup) return;
    const token = ++readinessToken.current;
    setReadiness(null);
    setReadinessSignature(null);
    setReadinessBusy(true);
    try {
      const result = await contentProtectionApi.readiness(setup);
      if (token === readinessToken.current) {
        setReadiness(result);
        setReadinessSignature(result.code === 'ready' ? classifierSignature(setup) : null);
      }
    } catch (caught) {
      if (token === readinessToken.current)
        setSetupError(caught instanceof Error ? caught.message : 'Readiness check failed');
      if (token === readinessToken.current) setReadinessSignature(null);
    } finally {
      if (token === readinessToken.current) setReadinessBusy(false);
    }
  };
  const updateCategory = (id: string, change: Partial<ContentCategory>) =>
    setSetup(
      (current) =>
        current && {
          ...current,
          categories: current.categories.map((category) =>
            category.id === id ? { ...category, ...change } : category,
          ),
        },
    );
  const deleteCategory = (category: ContentCategory) => {
    if (!setup) return;
    if (setup.access_levels.some((level) => level.granted_category_ids.includes(category.id))) {
      setSetupError(`Remove ${category.name} from access levels before deleting.`);
      return;
    }
    update({ categories: setup.categories.filter((item) => item.id !== category.id) });
  };
  const updateLevel = (id: string, change: Partial<AccessLevel>) =>
    setSetup(
      (current) =>
        current && {
          ...current,
          access_levels: current.access_levels.map((level) =>
            level.id === id ? { ...level, ...change } : level,
          ),
        },
    );
  const deleteLevel = (level: AccessLevel) => {
    if (!setup) return;
    if (level.id === setup.default_access_level_id) {
      setSetupError('Reassign the default level before deleting it.');
      return;
    }
    const mapped = setup.group_access_levels.filter(
      (item) => item.access_level_id === level.id,
    ).length;
    if (mapped) {
      setSetupError(`Unassign ${mapped} group(s) from ${level.name} before deleting.`);
      return;
    }
    update({ access_levels: setup.access_levels.filter((item) => item.id !== level.id) });
  };
  const updateGroupMapping = (groupId: string, levelId: string, checked: boolean) =>
    setSetup((current) => {
      if (!current) return current;
      const group_access_levels = current.group_access_levels.filter(
        (item) => item.group_id !== groupId || item.access_level_id !== levelId,
      );
      return {
        ...current,
        group_access_levels: checked
          ? [...group_access_levels, { group_id: groupId, access_level_id: levelId }]
          : group_access_levels,
      };
    });
  const runPreview = async () => {
    const token = ++previewToken.current;
    setInspectError(null);
    setPreviewBusy(true);
    try {
      const result = await contentProtectionApi.preview({
        user_id: identity.startsWith('user:') ? identity.slice(5) : undefined,
        public: identity === 'public',
        baseline: identity === 'service' ? 'service' : identity === 'public' ? 'public' : 'user',
        surface,
        mcp_route: mcpRoute || undefined,
        tool_id: toolId || undefined,
      });
      if (token === previewToken.current) setPreview(result);
    } catch (caught) {
      if (token === previewToken.current)
        setInspectError(caught instanceof Error ? caught.message : 'Failed to check saved policy');
    } finally {
      if (token === previewToken.current) setPreviewBusy(false);
    }
  };
  const runTest = async () => {
    if (!saved || !sample.trim()) return;
    const token = ++requestToken.current;
    setInspectError(null);
    setTestBusy(true);
    try {
      const result = await contentProtectionApi.test(
        saved,
        sample,
        sampleLevelIds,
        sampleUserId || undefined,
      );
      if (token === requestToken.current) setTestResult(result);
    } catch (caught) {
      if (token === requestToken.current)
        setInspectError(caught instanceof Error ? caught.message : 'Sample test failed');
    } finally {
      if (token === requestToken.current) setTestBusy(false);
    }
  };
  const loadDecisions = async () => {
    if (decisions || decisionsBusy) return;
    setDecisionsBusy(true);
    setDecisionsError(null);
    try {
      setDecisions((await contentProtectionApi.decisions()).items);
    } catch (caught) {
      setDecisionsError(caught instanceof Error ? caught.message : 'Failed to load decisions');
    } finally {
      setDecisionsBusy(false);
    }
  };
  useEffect(() => {
    previewToken.current += 1;
    setPreview(null);
  }, [identity, surface, mcpRoute, toolId, saved?.revision]);
  useEffect(() => {
    requestToken.current += 1;
    setTestResult(null);
  }, [sample, sampleLevelIds, sampleUserId, saved?.revision]);
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
        {!saved ? (
          loadError ? (
            <div
              className="content-protection-result"
              role="alert"
              data-content-protection-load-error
            >
              <strong>Unable to load content protection settings</strong>
              <span>{loadError}</span>
              <button className="btn btn-secondary" type="button" onClick={() => void load()}>
                Retry
              </button>
            </div>
          ) : (
            <p className="field-help">Loading content protection settings…</p>
          )
        ) : (
          <>
            <p className="content-protection-status">
              {saved.enabled
                ? 'Enabled: covered traffic is classified before release.'
                : 'Disabled: saved rules are ready for preparation.'}
            </p>
            {saved.legacy_reset && (
              <div
                className="settings-switch-card"
                data-content-protection-legacy-notice
                role="status"
              >
                <div>
                  <strong>Configuration reset</strong>
                  <p className="field-help">
                    Your previous v1 configuration (
                    {saved.legacy_was_enabled ? 'was enabled' : 'was disabled'}) was replaced with
                    disabled v2 defaults. Review and save to dismiss this notice.
                  </p>
                </div>
              </div>
            )}
            <div className="content-protection-overview" data-content-protection-overview>
              <fieldset
                id="content-protection-classifier-summary"
                className="content-protection-summary"
              >
                <legend>
                  Classifier{' '}
                  {readiness?.code === 'ready' &&
                  readinessSignature === classifierSignature(saved) ? (
                    <CheckCircle2 className="content-protection-status-icon is-ready" />
                  ) : (
                    <CircleHelp className="content-protection-status-icon" />
                  )}
                </legend>
                <p>
                  {saved.classifier.backend === 'jev'
                    ? `Jev · ${saved.classifier.jev.transport}`
                    : 'Generic LLM'}
                </p>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => openSetup('classifier')}
                >
                  Configure classifier
                </button>
              </fieldset>
              <fieldset
                id="content-protection-categories-summary"
                className="content-protection-summary"
              >
                <legend>Categories</legend>
                <p>{saved.categories.length} categories</p>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => openSetup('categories')}
                >
                  Configure categories
                </button>
              </fieldset>
              <fieldset
                id="content-protection-levels-summary"
                className="content-protection-summary"
              >
                <legend>Access levels</legend>
                <p>
                  {saved.access_levels.length} levels · {saved.group_access_levels.length} group
                  mappings
                </p>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => openSetup('access_levels')}
                >
                  Configure access levels
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
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => openSetup('coverage')}
                >
                  Configure coverage
                </button>
              </fieldset>
            </div>
            <div className="form-actions">
              <button
                className="btn btn-secondary"
                type="button"
                onClick={() => {
                  setInspect(true);
                  setInspectTab('request');
                  setInspectError(null);
                }}
              >
                Test & inspect
              </button>
            </div>
          </>
        )}
      </section>
      {setup && (
        <Dialog
          title="Configure content protection"
          error={setupError}
          onClose={() => {
            if (saving) return;
            readinessToken.current += 1;
            setReadiness(null);
            setReadinessSignature(null);
            setSetup(null);
          }}
          busy={saving}
          hook="content-protection-setup"
          activeTab={tab}
          footer={
            <>
              <button
                className="btn btn-secondary"
                type="button"
                disabled={saving || tab === 'classifier'}
                onClick={() =>
                  setTab(SETUP_TABS[SETUP_TABS.findIndex((item) => item.id === tab) - 1].id)
                }
              >
                Back
              </button>
              {tab === 'review' ? (
                <button className="btn" type="button" disabled={saving} onClick={() => void save()}>
                  {saving ? 'Saving…' : 'Save protection'}
                </button>
              ) : (
                <button
                  className="btn"
                  type="button"
                  disabled={saving}
                  onClick={() =>
                    setTab(SETUP_TABS[SETUP_TABS.findIndex((item) => item.id === tab) + 1].id)
                  }
                >
                  Next
                </button>
              )}
              {saveConflict && (
                <button
                  className="btn btn-secondary"
                  type="button"
                  disabled={saving}
                  onClick={() => void reloadSaved()}
                >
                  Reload and discard draft
                </button>
              )}
              <button
                className="btn btn-secondary"
                type="button"
                disabled={saving}
                onClick={() => {
                  readinessToken.current += 1;
                  setReadiness(null);
                  setReadinessSignature(null);
                  setSetup(null);
                }}
              >
                Cancel
              </button>
            </>
          }
        >
          <fieldset disabled={saving} className="content-protection-dialog-controls">
            <Tabs
              tabs={SETUP_TABS}
              active={tab}
              onChange={setTab}
              hook="content-protection-setup"
              label="Configure content protection steps"
            />
            {tab === 'classifier' && (
              <>
                <fieldset
                  id="content-protection-classifier-fieldset"
                  className="content-protection-fieldset"
                  data-content-protection-jev-config
                >
                  <legend>
                    <label>
                      <input
                        type="radio"
                        name="content-protection-backend"
                        checked={setup.classifier.backend === 'jev'}
                        onChange={() =>
                          update({ classifier: { ...setup.classifier, backend: 'jev' } })
                        }
                      />{' '}
                      Jev
                    </label>{' '}
                    <span className="tool-badge">Recommended</span>
                  </legend>
                  <fieldset>
                    <legend>Transport</legend>
                    {(['auto', 'typesafe', 'openrouter'] as const).map((transport) => (
                      <label key={transport}>
                        <input
                          type="radio"
                          name="jev-transport"
                          disabled={setup.classifier.backend !== 'jev'}
                          checked={setup.classifier.jev.transport === transport}
                          onChange={() =>
                            update({
                              classifier: {
                                ...setup.classifier,
                                backend: 'jev',
                                jev: { ...setup.classifier.jev, transport },
                              },
                            })
                          }
                        />{' '}
                        {transport}
                      </label>
                    ))}
                  </fieldset>
                  <label htmlFor="jev-model-input">
                    Model
                    <input
                      id="jev-model-input"
                      value={setup.classifier.jev.model}
                      onChange={(event) =>
                        update({
                          classifier: {
                            ...setup.classifier,
                            jev: { ...setup.classifier.jev, model: event.target.value },
                          },
                        })
                      }
                      onBlur={(event) => {
                        if (!event.target.value.trim())
                          update({
                            classifier: {
                              ...setup.classifier,
                              jev: { ...setup.classifier.jev, model: 'jev-latest' },
                            },
                          });
                      }}
                    />
                  </label>
                  <p className="field-help">
                    <code>jev-latest</code> tracks the current recommended version. Pin a release to
                    freeze behavior.
                  </p>
                  {(setup.classifier.jev.transport === 'auto' ||
                    setup.classifier.jev.transport === 'typesafe') && (
                    <KeyRow
                      id="content-protection-typesafe-key-row"
                      hook="content-protection-typesafe-key"
                      label="TypeSafe API key"
                      configured={catalog.classifier_status?.typesafe_key_configured}
                      value={typesafeKey}
                      onChange={setTypesafeKey}
                      onSave={() => void saveKey('typesafe_api_key', typesafeKey)}
                      busy={keyBusy}
                    />
                  )}
                  {(setup.classifier.jev.transport === 'auto' ||
                    setup.classifier.jev.transport === 'openrouter') && (
                    <div
                      id="content-protection-openrouter-key-row"
                      data-content-protection-openrouter-key
                    >
                      <p className="field-help">
                        {catalog.classifier_status?.openrouter_key_configured
                          ? 'An OpenRouter key is configured.'
                          : 'No OpenRouter key is configured.'}{' '}
                        <a
                          href="#setting-openrouter-api-key"
                          onClick={() => {
                            window.dispatchEvent(
                              new CustomEvent('highlight-settings', {
                                detail: 'setting-openrouter-api-key',
                              }),
                            );
                            readinessToken.current += 1;
                            setReadiness(null);
                            setReadinessSignature(null);
                            setSetup(null);
                          }}
                        >
                          Manage the OpenRouter key in provider settings.
                        </a>
                      </p>
                    </div>
                  )}
                  <button
                    type="button"
                    className="btn btn-secondary"
                    disabled={readinessBusy}
                    onClick={() => void checkReadiness()}
                  >
                    {readinessBusy && <Loader2 className="content-protection-spinner" />} Check
                    readiness
                  </button>
                  {readiness && (
                    <section
                      id="readiness-result"
                      className="content-protection-result"
                      role="status"
                      data-content-protection-readiness-result
                    >
                      <strong>{readiness.code === 'ready' ? 'Ready' : 'Not ready'}</strong>
                      {readiness.model && (
                        <span className="tool-badge">
                          Resolved: {readiness.model} via {readiness.transport}
                        </span>
                      )}
                      {readiness.cases?.map((item) => (
                        <span key={item.name}>
                          {item.name}: {item.verdict} ·{' '}
                          {Object.entries(item.probabilities)
                            .map(([id, probability]) => `${id} ${(probability * 100).toFixed(0)}%`)
                            .join(', ')}
                        </span>
                      ))}
                    </section>
                  )}
                </fieldset>
                <details
                  id="content-protection-llm-advanced"
                  data-content-protection-llm-advanced
                  open={setup.classifier.backend === 'llm'}
                >
                  <summary>
                    Advanced: generic LLM classifier{' '}
                    <span className="tool-badge">Not recommended</span>
                  </summary>
                  <p className="field-help">
                    Generic model probabilities are not calibrated. Retained for evaluation only.
                  </p>
                  <label>
                    <input
                      type="radio"
                      name="content-protection-backend"
                      checked={setup.classifier.backend === 'llm'}
                      onChange={() =>
                        update({ classifier: { ...setup.classifier, backend: 'llm' } })
                      }
                    />{' '}
                    Use generic LLM
                  </label>
                  {setup.classifier.backend === 'llm' && (
                    <ModelSelector
                      models={(availableModels.models || []).filter(
                        (model) => !/jev/i.test(`${model.provider} ${model.id}`),
                      )}
                      selectedModelId={setup.classifier.llm_model || ''}
                      onModelChange={(llm_model) =>
                        update({ classifier: { ...setup.classifier, llm_model } })
                      }
                      getModelSelectionKey={(model) => `${model.provider}::${model.id}`}
                      loading={availableModels.loading || false}
                      disabled={false}
                      placeholder="Select a generic model"
                      variant="full"
                    />
                  )}
                </details>
              </>
            )}
            {tab === 'categories' && (
              <>
                <p className="field-help">
                  Categories define what information is classified. System categories cannot be
                  edited or deleted.
                </p>
                <div
                  id="content-protection-category-list"
                  className="content-protection-profiles"
                  data-content-protection-category-list
                >
                  {setup.categories.map((category) => (
                    <ContentProtectionCategoryCard
                      key={category.id}
                      category={category}
                      onUpdate={updateCategory}
                      onDelete={deleteCategory}
                    />
                  ))}
                </div>
                <button
                  className="btn btn-secondary"
                  type="button"
                  disabled={setup.categories.length >= 24}
                  onClick={() =>
                    update({
                      categories: [
                        ...setup.categories,
                        {
                          id: `cat_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`,
                          name: 'New category',
                          description: 'Describe restricted content.',
                          includes: [],
                          excludes: [],
                          examples: [],
                          denial_message: 'This information is restricted.',
                          threshold_override: null,
                          system: false,
                        },
                      ],
                    })
                  }
                >
                  Add category
                </button>
              </>
            )}
            {tab === 'access_levels' && (
              <>
                <p className="field-help">
                  Access levels define what categories each audience may see.
                </p>
                <div
                  id="content-protection-level-list"
                  className="content-protection-profiles"
                  data-content-protection-level-list
                >
                  {setup.access_levels.map((level) => (
                    <ContentProtectionAccessLevelCard
                      key={level.id}
                      level={level}
                      categories={setup.categories.filter(
                        (category) => category.id !== 'rule_override',
                      )}
                      disabled={
                        level.id === setup.default_access_level_id ||
                        setup.group_access_levels.some((item) => item.access_level_id === level.id)
                      }
                      onUpdate={updateLevel}
                      onDelete={deleteLevel}
                    />
                  ))}
                </div>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() =>
                    update({
                      access_levels: [
                        ...setup.access_levels,
                        {
                          id: `level_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`,
                          name: 'New level',
                          granted_category_ids: [],
                          guidance: '',
                        },
                      ],
                    })
                  }
                >
                  Add access level
                </button>
                <fieldset
                  id="content-protection-default-level"
                  className="content-protection-fieldset"
                >
                  <legend>Default level</legend>
                  <select
                    aria-label="Default level"
                    value={setup.default_access_level_id}
                    onChange={(event) => update({ default_access_level_id: event.target.value })}
                  >
                    {setup.access_levels.map((level) => (
                      <option key={level.id} value={level.id}>
                        {level.name}
                      </option>
                    ))}
                  </select>
                </fieldset>
                <section
                  id="content-protection-group-mappings"
                  data-content-protection-group-mappings
                >
                  <h4>Group mappings</h4>
                  <p className="field-help">
                    One group may map to multiple levels; grants are unioned within each user's
                    mapped levels.
                  </p>
                  <div className="content-protection-rows">
                    {catalog.groups.map((group) => (
                      <div
                        className="content-protection-row"
                        key={group.id}
                        data-group-id={group.id}
                      >
                        <span>{group.name}</span>
                        <div role="group" aria-label={`Access levels for ${group.name}`}>
                          {setup.access_levels.map((level) => (
                            <label className="checkbox-label" key={level.id}>
                              <input
                                type="checkbox"
                                checked={setup.group_access_levels.some(
                                  (item) =>
                                    item.group_id === group.id && item.access_level_id === level.id,
                                )}
                                onChange={(event) =>
                                  updateGroupMapping(group.id, level.id, event.target.checked)
                                }
                              />{' '}
                              {level.name}
                            </label>
                          ))}
                        </div>
                      </div>
                    ))}
                    {setup.group_access_levels
                      .filter(
                        (mapping) => !catalog.groups.some((group) => group.id === mapping.group_id),
                      )
                      .map((mapping) => (
                        <div
                          className="content-protection-row"
                          key={`${mapping.group_id}:${mapping.access_level_id}`}
                          data-content-protection-unknown-group-mapping
                        >
                          <span>
                            Missing group {mapping.group_id} →{' '}
                            {setup.access_levels.find(
                              (level) => level.id === mapping.access_level_id,
                            )?.name || mapping.access_level_id}
                          </span>
                          <button
                            className="btn btn-secondary btn-sm"
                            type="button"
                            onClick={() =>
                              updateGroupMapping(mapping.group_id, mapping.access_level_id, false)
                            }
                          >
                            Remove mapping
                          </button>
                        </div>
                      ))}
                  </div>
                </section>
              </>
            )}
            {tab === 'coverage' && (
              <>
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
                      >
                        <CircleHelp />
                      </button>
                    </Popover>
                  </span>
                  <select
                    id="content-protection-coverage"
                    value={setup.coverage_mode}
                    onChange={(event) =>
                      update({
                        coverage_mode: event.target
                          .value as ContentProtectionConfig['coverage_mode'],
                      })
                    }
                  >
                    <option value="all_supported_traffic">All supported traffic</option>
                    <option value="selected_scopes">Selected scopes</option>
                  </select>
                </div>
                {setup.enabled && setup.coverage_mode === 'selected_scopes' && (
                  <section id="content-protection-app-areas">
                    <h4>App areas</h4>
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
                            update({
                              requirements: withRequirement(
                                setup,
                                'surface',
                                item.id,
                                event.target.value as 'inherit' | 'require',
                              ).requirements,
                            })
                          }
                        >
                          <option value="require">Require classification</option>
                          <option value="inherit">No additional requirement</option>
                        </select>
                      </label>
                    ))}
                  </section>
                )}
                {!setup.enabled && setup.coverage_mode === 'selected_scopes' && (
                  <p className="field-help">
                    Enable protection in Review &amp; save to configure app areas. Saved
                    requirements remain in place while protection is disabled.
                  </p>
                )}
                <div
                  id="content-protection-policy-links"
                  className="content-protection-policy-links"
                >
                  <span className="field-help">
                    Configure specific policies (opens in a new tab):
                  </span>
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
                </div>
              </>
            )}
            {tab === 'review' && (
              <>
                <Switch
                  hook="content-protection-enable-switch"
                  label="Enable content protection"
                  description="Turn on classification after reviewing saved coverage."
                  checked={setup.enabled}
                  onChange={(enabled) => update({ enabled })}
                />
                <Switch
                  hook="content-protection-advisory-switch"
                  label="Share access guidance with assistant"
                  description="Independent of enforcement; no classifier call is made for guidance alone."
                  checked={setup.share_with_assistant}
                  onChange={(share_with_assistant) => update({ share_with_assistant })}
                />
                <fieldset
                  id="content-protection-strictness"
                  className="content-protection-fieldset"
                >
                  <legend>Strictness</legend>
                  <select
                    aria-label="Strictness"
                    value={setup.strictness}
                    onChange={(event) =>
                      update({
                        strictness: event.target.value as ContentProtectionConfig['strictness'],
                      })
                    }
                  >
                    <option value="strict">Strict — deny at ≥25% probability</option>
                    <option value="balanced">Balanced — deny at ≥50% probability</option>
                    <option value="permissive">Permissive — deny at ≥75% probability</option>
                  </select>
                  <p className="field-help">
                    Thresholds are policy parameters, not security guarantees.
                  </p>
                </fieldset>
                <dl className="content-protection-review">
                  <dt>Classifier</dt>
                  <dd>
                    {setup.classifier.backend === 'jev'
                      ? `Jev · ${setup.classifier.jev.transport} · ${setup.classifier.jev.model}`
                      : `LLM · ${setup.classifier.llm_model || 'none'}`}
                  </dd>
                  <dt>Access levels</dt>
                  <dd>{setup.access_levels.map((level) => level.name).join(', ')}</dd>
                  <dt>Advisory guidance</dt>
                  <dd>{setup.share_with_assistant ? 'Shared with assistant' : 'Off'}</dd>
                </dl>
              </>
            )}
          </fieldset>
        </Dialog>
      )}
      {inspect && (
        <Dialog
          title="Test and inspect content protection"
          error={inspectError}
          onClose={() => setInspect(false)}
          hook="content-protection-inspect"
          activeTab={inspectTab}
          footer={
            <button className="btn btn-secondary" type="button" onClick={() => setInspect(false)}>
              Close
            </button>
          }
        >
          <Tabs
            tabs={INSPECT_TABS}
            active={inspectTab}
            onChange={(next) => {
              setInspectTab(next);
              if (next === 'decisions') void loadDecisions();
            }}
            hook="content-protection-inspect"
            label="Test and inspect content protection steps"
          />
          {inspectTab === 'request' && (
            <>
              <div className="content-protection-dialog-grid">
                <label>
                  Who is making the request?
                  <select
                    value={identity}
                    onChange={(event) => setIdentity(event.target.value as PreviewIdentity)}
                  >
                    <option value="service">Service</option>
                    <option value="public">Public</option>
                    {catalog.users.map((user) => (
                      <option key={user.id} value={`user:${user.id}`}>
                        {user.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Where does it run?
                  <select value={surface} onChange={(event) => setSurface(event.target.value)}>
                    {catalog.surfaces.map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  MCP route (optional)
                  <select value={mcpRoute} onChange={(event) => setMcpRoute(event.target.value)}>
                    <option value="">None</option>
                    {catalog.mcp_routes.map((route) => (
                      <option key={route.id} value={route.id}>
                        {route.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Tool (optional)
                  <select value={toolId} onChange={(event) => setToolId(event.target.value)}>
                    <option value="">None</option>
                    {catalog.tools.map((tool) => (
                      <option key={tool.id} value={tool.id}>
                        {tool.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <button
                className="btn btn-secondary"
                type="button"
                disabled={previewBusy}
                onClick={() => void runPreview()}
              >
                <TestTube2 /> Check saved policy
              </button>
              {preview && (
                <section
                  id="content-protection-preview-result"
                  className="content-protection-result"
                  role="status"
                  data-content-protection-preview-result
                >
                  <strong>{preview.required ? 'Classification required' : 'Not required'}</strong>
                  <span>Access levels: {accessLevelSets(preview)}</span>
                  <span>
                    Granted categories: {preview.granted_category_ids.join(', ') || 'None'}
                  </span>
                  {preview.share_with_assistant && (
                    <span>Advisory guidance active ({preview.guidance.length} segments)</span>
                  )}
                  <details>
                    <summary>Technical details</summary>
                    <p>
                      Policy revision: {preview.policy_revision} · Guidance revision:{' '}
                      {preview.guidance_revision}
                    </p>
                  </details>
                </section>
              )}
              {preview?.share_with_assistant && preview.prompt_fragment && (
                <section
                  id="content-protection-advisory-preview"
                  className="content-protection-fieldset content-protection-advisory-preview"
                  data-content-protection-advisory-preview
                >
                  <strong>Access guidance sent to assistant</strong>
                  <pre>{preview.prompt_fragment}</pre>
                </section>
              )}
            </>
          )}
          {inspectTab === 'content' && (
            <>
              <textarea
                id="content-protection-sample"
                aria-label="Sample content"
                value={sample}
                maxLength={1048576}
                onChange={(event) => setSample(event.target.value)}
              />
              <fieldset id="content-protection-test-levels" disabled={Boolean(sampleUserId)}>
                <legend>Target access levels</legend>
                {saved?.access_levels.map((level) => (
                  <label className="checkbox-label" key={level.id}>
                    <input
                      type="checkbox"
                      checked={sampleLevelIds.includes(level.id)}
                      onChange={() =>
                        setSampleLevelIds((current) =>
                          current.includes(level.id)
                            ? current.filter((id) => id !== level.id)
                            : [...current, level.id],
                        )
                      }
                    />{' '}
                    {level.name}
                  </label>
                ))}
              </fieldset>
              <label>
                Or test as a real user
                <select
                  value={sampleUserId}
                  onChange={(event) => setSampleUserId(event.target.value)}
                >
                  <option value="">— Use selected levels above —</option>
                  {catalog.users.map((user) => (
                    <option key={user.id} value={user.id}>
                      {user.name}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="btn btn-secondary"
                type="button"
                disabled={!sample.trim() || testBusy}
                onClick={() => void runTest()}
              >
                {testBusy ? 'Testing…' : 'Test content'}
              </button>
              {testResult && (
                <section
                  id="content-protection-test-result"
                  className="content-protection-result"
                  role="status"
                  data-content-protection-test-result
                >
                  <strong>
                    {testResult.verdict || 'error'} · {testResult.code}
                  </strong>
                  {testResult.reason && <span>{testResult.reason}</span>}
                  {testResult.model && (
                    <span className="tool-badge">
                      {testResult.model} via {testResult.transport}
                    </span>
                  )}
                  {testResult.probabilities && (
                    <details id="content-protection-test-probabilities">
                      <summary>Category probabilities</summary>
                      <dl className="content-protection-review">
                        {saved?.categories.map((category) => {
                          const value = testResult.probabilities?.[category.id];
                          return value == null ? null : (
                            <div key={category.id}>
                              <dt>{category.name}</dt>
                              <dd data-category-prob={category.id}>
                                {(value * 100).toFixed(1)}%{' '}
                                {value >= thresholdFor(saved, category) && (
                                  <span className="tool-badge">above threshold</span>
                                )}
                              </dd>
                            </div>
                          );
                        })}
                      </dl>
                    </details>
                  )}
                </section>
              )}
            </>
          )}
          {inspectTab === 'decisions' &&
            (decisionsBusy ? (
              <p className="field-help">Loading decisions…</p>
            ) : decisionsError ? (
              <div className="content-protection-result" role="alert">
                <span>{decisionsError}</span>
                <button
                  className="btn btn-secondary"
                  type="button"
                  onClick={() => void loadDecisions()}
                >
                  Retry
                </button>
              </div>
            ) : decisions?.length ? (
              <ul className="content-protection-decisions">
                {decisions.map((item) => (
                  <li key={item.request_id || item.id}>
                    {item.created_at} · {item.verdict || item.code}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="field-help">No decision metadata available.</p>
            ))}
        </Dialog>
      )}
    </SettingsAccordionSection>
  );
}
function KeyRow({
  id,
  hook,
  label,
  configured,
  value,
  onChange,
  onSave,
  busy,
}: {
  id: string;
  hook: string;
  label: string;
  configured?: boolean;
  value: string;
  onChange: (value: string) => void;
  onSave: () => void;
  busy: boolean;
}) {
  return (
    <div id={id} className="input-with-button" data-content-protection-key={hook}>
      <label htmlFor={`${hook}-input`}>{label}</label>
      <input
        id={`${hook}-input`}
        type="password"
        autoComplete="new-password"
        value={value}
        placeholder={configured ? '••••••••' : 'Enter key'}
        onChange={(event) => onChange(event.target.value)}
      />
      <button
        type="button"
        className="btn btn-secondary btn-sm"
        disabled={!value.trim() || busy}
        onClick={onSave}
      >
        {busy ? 'Saving…' : 'Save key'}
      </button>
      <p className="field-help">
        {configured
          ? 'A key is configured. Enter a new value to replace it.'
          : 'No key configured.'}
      </p>
    </div>
  );
}
function Switch({
  hook,
  label,
  checked,
  onChange,
  description,
}: {
  hook: string;
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  description: string;
}) {
  return (
    <div className="settings-switch-card" data-content-protection-switch={hook}>
      <div>
        <strong>{label}</strong>
        <p className="field-help">{description}</p>
      </div>
      <label className="toggle-switch" aria-label={label}>
        <input
          type="checkbox"
          checked={checked}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span className="toggle-slider" />
      </label>
    </div>
  );
}
