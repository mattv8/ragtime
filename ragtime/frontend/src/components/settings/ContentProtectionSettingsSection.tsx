import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import {
  contentProtectionApi,
  ContentProtectionApiError,
  type ContentProtectionCatalog,
  type ContentProtectionConfig,
  type ContentProtectionOverrideMode,
  type ContentProtectionProfile,
  type ContentProtectionRequirement,
  type ContentProtectionRequirementMode,
  type ContentProtectionTestResult,
} from '@/api/contentProtection';
import { useAvailableModels } from '@/contexts/AvailableModelsContext';
import { ModelSelector } from '../ModelSelector';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

type Tab = 'users' | 'groups' | 'tools' | 'mcp-routes' | 'surfaces';
type PreviewIdentity = `user:${string}` | 'public' | 'service';

const TABS: Array<{ id: Tab; label: string }> = [
  { id: 'users', label: 'Users' },
  { id: 'groups', label: 'Groups' },
  { id: 'tools', label: 'Tools' },
  { id: 'mcp-routes', label: 'MCP routes' },
  { id: 'surfaces', label: 'Surfaces' },
];
const EMPTY_CATALOG: ContentProtectionCatalog = {
  users: [],
  groups: [],
  tools: [],
  mcp_routes: [],
  surfaces: [],
};

function formatLatency(latency: number | undefined): string {
  return latency == null ? '' : ` · ${Math.round(latency * 1000)} ms`;
}

function requirementMode(
  config: ContentProtectionConfig,
  scope_kind: ContentProtectionRequirement['scope_kind'],
  scope_key: string,
): ContentProtectionRequirementMode {
  return (
    config.requirements.find(
      (item) => item.scope_kind === scope_kind && item.scope_key === scope_key,
    )?.mode || 'inherit'
  );
}

function setRequirement(
  config: ContentProtectionConfig,
  scope_kind: ContentProtectionRequirement['scope_kind'],
  scope_key: string,
  mode: ContentProtectionRequirementMode,
): ContentProtectionConfig {
  const requirements = config.requirements.filter(
    (item) => item.scope_kind !== scope_kind || item.scope_key !== scope_key,
  );
  if (mode === 'require') requirements.push({ scope_kind, scope_key, mode });
  return { ...config, requirements };
}

function userMode(config: ContentProtectionConfig, user_id: string): ContentProtectionOverrideMode {
  return config.user_overrides.find((item) => item.user_id === user_id)?.mode || 'inherit';
}

function setUserOverride(
  config: ContentProtectionConfig,
  user_id: string,
  mode: ContentProtectionOverrideMode,
): ContentProtectionConfig {
  const user_overrides = config.user_overrides.filter((item) => item.user_id !== user_id);
  if (mode !== 'inherit') user_overrides.push({ user_id, mode });
  return { ...config, user_overrides };
}

function previewIdentityLabel(
  identity: PreviewIdentity,
  catalog: ContentProtectionCatalog,
): string {
  if (identity === 'public') return 'Public baseline';
  if (identity === 'service') return 'Service baseline';
  return (
    catalog.users.find((user) => user.id === identity.slice('user:'.length))?.name ||
    'Selected user'
  );
}

export function ContentProtectionSettingsSection({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
}): JSX.Element {
  const {
    models,
    loading: modelsLoading,
    error: modelsError,
    refresh: refreshModels,
  } = useAvailableModels();
  const refreshModelsRef = useRef(refreshModels);
  refreshModelsRef.current = refreshModels;
  const [config, setConfig] = useState<ContentProtectionConfig | null>(null);
  const [catalog, setCatalog] = useState<ContentProtectionCatalog>(EMPTY_CATALOG);
  const [activeTab, setActiveTab] = useState<Tab>('users');
  const [filter, setFilter] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewIdentity, setPreviewIdentity] = useState<PreviewIdentity>('service');
  const [previewSurface, setPreviewSurface] = useState('chat');
  const [previewRoute, setPreviewRoute] = useState('default');
  const [previewTool, setPreviewTool] = useState('');
  const [sample, setSample] = useState('');
  const [sampleProfiles, setSampleProfiles] = useState<string[]>([]);
  const [testResult, setTestResult] = useState<ContentProtectionTestResult | null>(null);
  const [readiness, setReadiness] = useState<ContentProtectionTestResult | null>(null);
  const [decisions, setDecisions] = useState<
    Awaited<ReturnType<typeof contentProtectionApi.decisions>>['items']
  >([]);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [nextConfig, nextCatalog, nextDecisions] = await Promise.all([
        contentProtectionApi.getConfig(),
        contentProtectionApi.getCatalog(),
        contentProtectionApi.decisions(),
      ]);
      setConfig(nextConfig);
      setCatalog(nextCatalog);
      setDecisions(nextDecisions.items);
      setSampleProfiles((current) =>
        current.length ? current : nextConfig.profiles.map((profile) => profile.id),
      );
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : 'Failed to load content protection settings',
      );
    }
  }, []);

  useEffect(() => {
    if (open) void load();
  }, [load, open]);
  useEffect(() => {
    if (open) refreshModelsRef.current();
  }, [open]);
  useEffect(() => {
    if (window.location.hash === '#content-protection-groups-tab') setActiveTab('groups');
  }, [open]);

  const update = (change: Partial<ContentProtectionConfig>) =>
    setConfig((current) => current && { ...current, ...change });
  const selectedModel = config?.classifier_model || '';
  const rows = useMemo(() => {
    const query = filter.trim().toLowerCase();
    const source =
      activeTab === 'users'
        ? catalog.users
        : activeTab === 'groups'
          ? catalog.groups
          : activeTab === 'tools'
            ? catalog.tools
            : activeTab === 'mcp-routes'
              ? catalog.mcp_routes
              : catalog.surfaces;
    return query
      ? source.filter((row) => `${row.name} ${row.id}`.toLowerCase().includes(query))
      : source;
  }, [activeTab, catalog, filter]);

  const save = async () => {
    if (!config) return;
    setSaving(true);
    setError(null);
    try {
      setConfig(await contentProtectionApi.saveConfig(config.revision, config));
    } catch (caught) {
      setError(
        caught instanceof ContentProtectionApiError && caught.status === 409
          ? 'This policy changed on the server. Reload and reconcile before saving.'
          : caught instanceof Error
            ? caught.message
            : 'Failed to save content protection settings',
      );
    } finally {
      setSaving(false);
    }
  };

  const runPreview = async () => {
    setError(null);
    setPreview(null);
    const user_id = previewIdentity.startsWith('user:')
      ? previewIdentity.slice('user:'.length)
      : undefined;
    const isPublic = previewIdentity === 'public';
    try {
      const result = await contentProtectionApi.preview({
        user_id,
        surface: previewSurface,
        mcp_route: previewRoute || undefined,
        tool_id: previewTool || undefined,
        public: isPublic,
      });
      const profiles =
        result.profiles
          .map((profileSet) => profileSet.map((profile) => profile.name).join(', '))
          .join(' / ') || 'none';
      setPreview(
        `${result.required ? 'Classification required' : 'Classification not required'} · User: ${previewIdentityLabel(previewIdentity, catalog)} · Surface: ${previewSurface} · MCP route: ${previewRoute || 'none'} · Tool: ${previewTool || 'none'} · Provenance: ${typeof result.provenance === 'string' ? result.provenance : JSON.stringify(result.provenance)} · Profile sets: ${profiles}`,
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Failed to preview policy');
    }
  };

  const runTest = async () => {
    if (!config) return;
    try {
      setTestResult(await contentProtectionApi.test(config, sample, sampleProfiles));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Sample test failed');
    }
  };
  const runReadiness = async () => {
    if (!config) return;
    try {
      setReadiness(await contentProtectionApi.readiness(config));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Readiness probe failed');
    }
  };
  const setGroupProfile = (group_id: string, profile_id: string) =>
    update({
      group_profiles: [
        ...(config?.group_profiles || []).filter((item) => item.group_id !== group_id),
        ...(profile_id ? [{ group_id, profile_id }] : []),
      ],
    });
  const updateProfile = (id: string, change: Partial<ContentProtectionProfile>) =>
    update({
      profiles: (config?.profiles || []).map((profile) =>
        profile.id === id ? { ...profile, ...change } : profile,
      ),
    });
  const deleteProfile = (profile: ContentProtectionProfile) => {
    if (!config) return;
    const affected = config.group_profiles.filter(
      (mapping) => mapping.profile_id === profile.id,
    ).length;
    if (affected) {
      setError(
        `Reassign or clear ${affected} affected group${affected === 1 ? '' : 's'} before deleting ${profile.name}.`,
      );
      return;
    }
    update({ profiles: config.profiles.filter((candidate) => candidate.id !== profile.id) });
  };
  const addProfile = () => {
    if (!config) return;
    const id = `profile_${crypto.randomUUID().replace(/-/g, '').slice(0, 12)}`;
    update({
      profiles: [
        ...config.profiles,
        {
          id,
          name: 'New profile',
          level: 0,
          scope: 'Describe the information this profile permits.',
        },
      ],
    });
  };

  return (
    <SettingsAccordionSection
      id="content-protection"
      title="Content protection"
      open={open}
      onToggle={onToggle}
      status={config?.enabled ? 'Enabled' : 'Disabled'}
    >
      <section
        id="settings-content-protection"
        className="content-protection-section"
        aria-label="Content protection settings"
      >
        {error && (
          <p className="field-error" role="alert">
            {error}
          </p>
        )}
        {!config ? (
          <p className="field-help">Loading content protection settings…</p>
        ) : (
          <>
            <div id="content-protection-status-card" className="content-protection-card">
              <div className="form-group">
                <label htmlFor="content-protection-master-switch">Content protection</label>
                <label className="toggle-switch">
                  <input
                    id="content-protection-master-switch"
                    type="checkbox"
                    checked={config.enabled}
                    onChange={(event) => update({ enabled: event.target.checked })}
                  />
                  <span className="toggle-slider" />
                </label>
                <p className="field-help">
                  {config.enabled
                    ? 'Enabled. Covered traffic is classified before release.'
                    : 'Disabled. Stored rules remain dormant and production traffic makes no classifier call.'}
                </p>
              </div>
              <div className="form-group" id="content-protection-model-selector">
                <label>Classifier model</label>
                <ModelSelector
                  models={models}
                  selectedModelId={selectedModel}
                  onModelChange={(classifier_model) => update({ classifier_model })}
                  getModelSelectionKey={(model) => `${model.provider}::${model.id}`}
                  loading={modelsLoading}
                  disabled={modelsLoading}
                  placeholder="Select a curated model"
                  variant="full"
                />
                <p className="field-help">
                  {modelsError
                    ? modelsError
                    : modelsLoading
                      ? 'Loading curated classifier models…'
                      : models.length
                        ? 'The selected provider receives the content being inspected.'
                        : 'No curated classifier models are available.'}
                </p>
              </div>
              <div id="content-protection-readiness" className="form-group">
                <label>Readiness</label>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => void runReadiness()}
                  disabled={!config.classifier_model}
                >
                  Probe model
                </button>
                <p className="field-help">
                  {readiness
                    ? `${readiness.code}${formatLatency(readiness.latency)}`
                    : 'No probe run.'}
                </p>
              </div>
            </div>
            <div id="content-protection-coverage-card" className="content-protection-card">
              <div className="form-group">
                <label htmlFor="content-protection-coverage-mode">Coverage</label>
                <select
                  id="content-protection-coverage-mode"
                  value={config.coverage_mode}
                  onChange={(event) =>
                    update({
                      coverage_mode: event.target.value as ContentProtectionConfig['coverage_mode'],
                    })
                  }
                >
                  <option value="all_supported_traffic">All supported traffic</option>
                  <option value="selected_scopes">Selected scopes</option>
                </select>
                <p className="field-help">
                  Scope requirements are additive. User overrides remain editable in all-traffic
                  mode.
                </p>
              </div>
              <div
                id="content-protection-policy-preview"
                className="content-protection-preview-controls"
              >
                <h4>Preview effective policy</h4>
                <p className="field-help">
                  Uses the saved policy, not unsaved draft edits. Select the server-resolved scope
                  to inspect its coverage and provenance.
                </p>
                <label htmlFor="content-protection-preview-user">
                  User
                  <select
                    id="content-protection-preview-user"
                    value={previewIdentity}
                    onChange={(event) => setPreviewIdentity(event.target.value as PreviewIdentity)}
                  >
                    <option value="service">Service baseline</option>
                    <option value="public">Public baseline</option>
                    {catalog.users.map((user) => (
                      <option key={user.id} value={`user:${user.id}`}>
                        {user.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label htmlFor="content-protection-preview-surface">
                  Surface
                  <select
                    id="content-protection-preview-surface"
                    value={previewSurface}
                    onChange={(event) => setPreviewSurface(event.target.value)}
                  >
                    {catalog.surfaces.map((surface) => (
                      <option key={surface.id} value={surface.id}>
                        {surface.name}
                      </option>
                    ))}
                    {!catalog.surfaces.some((surface) => surface.id === previewSurface) && (
                      <option value={previewSurface}>{previewSurface}</option>
                    )}
                  </select>
                </label>
                <label htmlFor="content-protection-preview-route">
                  MCP route
                  <select
                    id="content-protection-preview-route"
                    value={previewRoute}
                    onChange={(event) => setPreviewRoute(event.target.value)}
                  >
                    <option value="">None</option>
                    {catalog.mcp_routes.map((route) => (
                      <option key={route.id} value={route.id}>
                        {route.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label htmlFor="content-protection-preview-tool">
                  Tool
                  <select
                    id="content-protection-preview-tool"
                    value={previewTool}
                    onChange={(event) => setPreviewTool(event.target.value)}
                  >
                    <option value="">None</option>
                    {catalog.tools.map((tool) => (
                      <option key={tool.id} value={tool.id}>
                        {tool.name}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => void runPreview()}
                >
                  Preview effective policy
                </button>
                {preview && (
                  <p id="content-protection-preview-result" className="field-help" role="status">
                    {preview}
                  </p>
                )}
              </div>
            </div>
            <div id="content-protection-scope-tabs" className="content-protection-card">
              <div
                className="content-protection-tabs"
                role="tablist"
                aria-label="Content protection scopes"
              >
                {TABS.map((tab) => (
                  <button
                    key={tab.id}
                    id={`content-protection-${tab.id}-tab`}
                    type="button"
                    role="tab"
                    aria-selected={activeTab === tab.id}
                    className={activeTab === tab.id ? 'is-active' : ''}
                    onClick={() => setActiveTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
              <label className="content-protection-filter">
                Search <input value={filter} onChange={(event) => setFilter(event.target.value)} />
              </label>
              <div role="tabpanel" className="content-protection-rows">
                {activeTab === 'users' ? (
                  <>
                    {rows.map((row) => (
                      <div className="content-protection-row" key={row.id} data-user-id={row.id}>
                        <span>
                          <strong>{row.name}</strong>
                          <small>
                            {config.enabled
                              ? 'Override applies at supported boundaries.'
                              : 'Dormant while master switch is off.'}
                          </small>
                        </span>
                        <select
                          aria-label={`Override for ${row.name}`}
                          value={userMode(config, row.id)}
                          onChange={(event) =>
                            update({
                              user_overrides: setUserOverride(
                                config,
                                row.id,
                                event.target.value as ContentProtectionOverrideMode,
                              ).user_overrides,
                            })
                          }
                        >
                          <option value="inherit">Inherit</option>
                          <option value="always_classify">Always classify</option>
                          <option value="never_classify">Never classify</option>
                        </select>
                      </div>
                    ))}
                    {(['anonymous', 'service', 'public'] as const).map((baseline) => (
                      <div
                        className="content-protection-row"
                        key={baseline}
                        data-baseline-identity-type={baseline}
                      >
                        <span>
                          <strong>{baseline}</strong>
                          <small>
                            Baseline identities are informational and cannot receive user overrides.
                          </small>
                        </span>
                        <span aria-label={`${baseline} overrides unavailable`}>Inherit only</span>
                      </div>
                    ))}
                  </>
                ) : (
                  rows.map((row) => {
                    const scope_kind =
                      activeTab === 'groups'
                        ? 'group'
                        : activeTab === 'tools'
                          ? 'tool'
                          : activeTab === 'mcp-routes'
                            ? 'mcp_route'
                            : 'surface';
                    return (
                      <div
                        className="content-protection-row"
                        key={row.id}
                        {...(scope_kind === 'group'
                          ? { 'data-group-id': row.id }
                          : scope_kind === 'tool'
                            ? { 'data-tool-id': row.id }
                            : scope_kind === 'mcp_route'
                              ? { 'data-mcp-route-key': row.id }
                              : { 'data-surface-id': row.id })}
                      >
                        <span>
                          <strong>{row.name}</strong>
                          <small>
                            {config.coverage_mode === 'all_supported_traffic'
                              ? 'Covered by global mode; this requirement is dormant.'
                              : 'Require adds coverage; Inherit does not exempt traffic.'}
                          </small>
                        </span>
                        <select
                          aria-label={`Coverage for ${row.name}`}
                          value={requirementMode(config, scope_kind, row.id)}
                          onChange={(event) =>
                            update({
                              requirements: setRequirement(
                                config,
                                scope_kind,
                                row.id,
                                event.target.value as ContentProtectionRequirementMode,
                              ).requirements,
                            })
                          }
                          disabled={config.coverage_mode === 'all_supported_traffic'}
                        >
                          <option value="inherit">Inherit</option>
                          <option value="require">Require</option>
                        </select>
                        {scope_kind === 'group' && (
                          <select
                            aria-label={`Profile for ${row.name}`}
                            value={
                              config.group_profiles.find((mapping) => mapping.group_id === row.id)
                                ?.profile_id || ''
                            }
                            onChange={(event) => setGroupProfile(row.id, event.target.value)}
                          >
                            <option value="">Standard baseline</option>
                            {config.profiles.map((profile) => (
                              <option key={profile.id} value={profile.id}>
                                {profile.name}
                              </option>
                            ))}
                          </select>
                        )}
                      </div>
                    );
                  })
                )}
              </div>
            </div>
            <details id="content-protection-profile-editor" className="content-protection-card">
              <summary id="content-protection-profile-editor-link">Profile definitions</summary>
              <div>
                {config.profiles.map((profile) => {
                  const affected = config.group_profiles.filter(
                    (mapping) => mapping.profile_id === profile.id,
                  ).length;
                  return (
                    <div
                      className="content-protection-profile"
                      key={profile.id}
                      data-profile-id={profile.id}
                    >
                      <input
                        aria-label={`${profile.name} name`}
                        value={profile.name}
                        onChange={(event) =>
                          updateProfile(profile.id, { name: event.target.value })
                        }
                      />
                      <input
                        aria-label={`${profile.name} level`}
                        type="number"
                        min="0"
                        max="2"
                        value={profile.level}
                        onChange={(event) =>
                          updateProfile(profile.id, { level: Number(event.target.value) })
                        }
                      />
                      <textarea
                        aria-label={`${profile.name} scope`}
                        value={profile.scope}
                        onChange={(event) =>
                          updateProfile(profile.id, { scope: event.target.value })
                        }
                      />
                      <button
                        type="button"
                        className="btn btn-sm btn-danger"
                        onClick={() => deleteProfile(profile)}
                        title={
                          affected
                            ? `${affected} group mappings must be reassigned or cleared first`
                            : `Delete ${profile.name}`
                        }
                      >
                        Delete{affected ? ` (${affected} groups)` : ''}
                      </button>
                    </div>
                  );
                })}
              </div>
              <button type="button" className="btn btn-secondary" onClick={addProfile}>
                Add profile
              </button>
            </details>
            <div id="content-protection-sample-test" className="content-protection-card">
              <label htmlFor="content-protection-sample">Synthetic sample</label>
              <textarea
                id="content-protection-sample"
                value={sample}
                maxLength={1048576}
                onChange={(event) => setSample(event.target.value)}
              />
              <fieldset>
                <legend>Target profiles</legend>
                {config.profiles.map((profile) => (
                  <label key={profile.id} className="checkbox-label">
                    <input
                      type="checkbox"
                      checked={sampleProfiles.includes(profile.id)}
                      onChange={() =>
                        setSampleProfiles((current) =>
                          current.includes(profile.id)
                            ? current.filter((id) => id !== profile.id)
                            : [...current, profile.id],
                        )
                      }
                    />
                    {profile.name}
                  </label>
                ))}
              </fieldset>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => void runTest()}
                disabled={!sample.trim()}
              >
                Test sample
              </button>
              {testResult && (
                <p className="field-help" role="status">
                  {testResult.verdict || 'error'} · {testResult.code}
                  {testResult.reason ? ` · ${testResult.reason}` : ''}
                  {formatLatency(testResult.latency)}
                </p>
              )}
              <p className="field-help">
                Text only: uninspectable files and payloads over 1 MiB are blocked as
                unclassifiable.
              </p>
            </div>
            <div id="content-protection-recent-metadata" className="content-protection-card">
              <h4>Recent decision metadata</h4>
              {decisions.length ? (
                <ul>
                  {decisions.map((decision) => (
                    <li key={decision.request_id}>
                      {decision.created_at || 'Unknown time'} · {decision.surface || 'unknown'} ·{' '}
                      {decision.direction || 'decision'} ·{' '}
                      {decision.verdict || decision.code || 'unknown'} · {decision.request_id}
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="field-help">No decision metadata available.</p>
              )}
            </div>
            <div id="content-protection-draft-actions" className="form-actions">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => void load()}
                disabled={saving}
              >
                Revert
              </button>
              <button type="button" className="btn" onClick={() => void save()} disabled={saving}>
                {saving ? 'Saving…' : 'Save'}
              </button>
            </div>
          </>
        )}
      </section>
    </SettingsAccordionSection>
  );
}
