import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  contentProtectionApi,
  requirementModeFor,
  type ContentProtectionConfig,
} from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { ContentProtectionAccessLevelCard } from '../settings/ContentProtectionAccessLevelCard';
import { useAccessLevelDrafts } from './useAccessLevelDrafts';
import { CONTENT_PROTECTION_SETTING_ID, SettingsHighlightLink } from './SettingsHighlightLink';
type PendingMapping = { levelId: string; enabled: boolean; hiddenDirtyIds: string[] };

export interface GroupProtectionPanelHandle {
  requestNavigation: (continueNavigation: () => void) => void;
  busy: () => boolean;
}

interface Props {
  group: AuthGroup;
  config: ContentProtectionConfig;
  authGroups: AuthGroup[];
  onConfigSaved: (config: ContentProtectionConfig) => void;
  onUpdateMapping: (levelId: string, enabled: boolean) => Promise<void>;
  onUpdateRequirement: (mode: 'inherit' | 'require') => Promise<void>;
  onBack: () => void;
  onBusyChange?: (busy: boolean) => void;
  hideHeader?: boolean;
  onNavigateToSetting?: (settingId: string) => void;
  toast: { success: (message: string) => void; error: (message: string) => void };
}

export const GroupProtectionPanel = forwardRef<GroupProtectionPanelHandle, Props>(
  function GroupProtectionPanel(
    {
      group,
      config,
      authGroups,
      onConfigSaved,
      onUpdateMapping,
      onUpdateRequirement,
      onBack,
      onBusyChange,
      hideHeader = false,
      onNavigateToSetting,
      toast,
    },
    ref,
  ) {
    const [mappingBusy, setMappingBusy] = useState(false);
    const [classificationBusy, setClassificationBusy] = useState(false);
    const [pendingNavigation, setPendingNavigation] = useState<(() => void) | null>(null);
    const [pendingMapping, setPendingMapping] = useState<PendingMapping | null>(null);
    const [preview, setPreview] = useState<Awaited<
      ReturnType<typeof contentProtectionApi.preview>
    > | null>(null);
    const [previewError, setPreviewError] = useState<string | null>(null);
    const [previewLoading, setPreviewLoading] = useState(false);
    const generation = useRef(0);
    const confirmRef = useRef<HTMLDivElement>(null);
    const {
      drafts,
      conflicts,
      saveErrors,
      savingLevelId,
      isDirty,
      hasDirty,
      updateDraft,
      saveLevel,
      discardLevel,
      discardAll,
      loadLatest,
    } = useAccessLevelDrafts({ config, onConfigSaved, toast });

    const mappedIds = useMemo(
      () =>
        config.group_access_levels
          .filter((item) => item.group_id === group.id)
          .map((item) => item.access_level_id),
      [config.group_access_levels, group.id],
    );
    const mappedLevels = config.access_levels.filter((level) => mappedIds.includes(level.id));
    const defaultLevel = config.access_levels.find(
      (level) => level.id === config.default_access_level_id,
    );
    const deletedDirtyLevels = Object.keys(drafts)
      .filter((id) => !config.access_levels.some((level) => level.id === id) && isDirty(id))
      .map((id) => drafts[id]);
    const editableLevels = [
      ...(mappedLevels.length ? mappedLevels : defaultLevel ? [defaultLevel] : []),
      ...deletedDirtyLevels,
    ];
    const busy = Boolean(savingLevelId || mappingBusy || classificationBusy);

    useEffect(() => onBusyChange?.(busy), [busy, onBusyChange]);

    useEffect(() => () => onBusyChange?.(false), [onBusyChange]);

    useEffect(() => {
      if (pendingNavigation) confirmRef.current?.focus();
    }, [pendingNavigation]);
    useImperativeHandle(
      ref,
      () => ({
        busy: () => busy,
        requestNavigation: (continueNavigation) => {
          if (busy) return;
          if (hasDirty) setPendingNavigation(() => continueNavigation);
          else continueNavigation();
        },
      }),
      [busy, hasDirty],
    );

    const refreshPreview = useCallback(
      async (nextConfig = config) => {
        const requestGeneration = generation.current;
        setPreviewLoading(true);
        setPreviewError(null);
        try {
          const result = await contentProtectionApi.preview({
            access_level_ids: nextConfig.group_access_levels
              .filter((item) => item.group_id === group.id)
              .map((item) => item.access_level_id),
          });
          if (generation.current === requestGeneration) setPreview(result);
        } catch (error) {
          if (generation.current === requestGeneration)
            setPreviewError(
              error instanceof Error ? error.message : 'Could not load prompt preview.',
            );
        } finally {
          if (generation.current === requestGeneration) setPreviewLoading(false);
        }
      },
      [config, group.id],
    );

    useEffect(() => {
      void refreshPreview(config);
      return () => {
        generation.current += 1;
      };
    }, [config, group.id, refreshPreview]);

    const runMapping = async (levelId: string, enabled: boolean) => {
      setMappingBusy(true);
      try {
        await onUpdateMapping(levelId, enabled);
      } finally {
        setMappingBusy(false);
      }
    };
    const toggleLevel = async (levelId: string, enabled: boolean) => {
      const nextMappedIds = enabled
        ? [...new Set([...mappedIds, levelId])]
        : mappedIds.filter((id) => id !== levelId);
      const nextVisibleIds = nextMappedIds.length
        ? nextMappedIds
        : config.default_access_level_id
          ? [config.default_access_level_id]
          : [];
      const hiddenDirtyIds = editableLevels
        .map((level) => level.id)
        .filter((id) => !nextVisibleIds.includes(id) && isDirty(id));
      if (hiddenDirtyIds.length) {
        setPendingMapping({ levelId, enabled, hiddenDirtyIds });
        return;
      }
      await runMapping(levelId, enabled);
    };
    const changeRequirement = async (mode: 'inherit' | 'require') => {
      setClassificationBusy(true);
      try {
        await onUpdateRequirement(mode);
      } finally {
        setClassificationBusy(false);
      }
    };
    const levelReach = (levelId: string) => {
      const count = config.group_access_levels.filter(
        (item) => item.access_level_id === levelId,
      ).length;
      return `(${count} group${count === 1 ? '' : 's'})`;
    };
    const controlsDisabled = busy;

    return (
      <section
        id={`group-protection-panel-${group.id}`}
        data-group-protection-panel={group.id}
        className="group-protection-panel is-open"
        aria-label={`Protection settings for ${group.display_name}`}
        aria-busy={busy}
        onKeyDown={(event) => {
          if (event.key === 'Escape' && !event.defaultPrevented) {
            event.preventDefault();
            if (!busy) {
              if (hasDirty) setPendingNavigation(() => onBack);
              else onBack();
            }
          }
        }}
      >
        {!hideHeader && (
          <header className="group-protection-panel-header" data-group-protection-header>
            <button
              type="button"
              className="btn btn-sm btn-secondary group-protection-back-btn"
              onClick={() => {
                if (hasDirty) setPendingNavigation(() => onBack);
                else onBack();
              }}
              disabled={busy}
            >
              ← Back to groups
            </button>
            <div className="group-protection-panel-title">
              <h4>{group.display_name}</h4>
              <span className={`auth-group-provider-badge auth-group-provider-${group.provider}`}>
                {group.provider === 'local_managed' ? 'Internal' : group.provider.toUpperCase()}
              </span>
              <span className="auth-group-row-title-member-count">
                {group.member_count} member{group.member_count === 1 ? '' : 's'}
              </span>
            </div>
          </header>
        )}
        {pendingNavigation && (
          <div
            ref={confirmRef}
            tabIndex={-1}
            className="tool-access-callout tool-access-callout-warning"
            role="alertdialog"
            aria-label="Unsaved changes"
            data-group-protection-dirty-confirm
          >
            <p>You have unsaved changes. Discard them?</p>
            <div className="group-protection-confirm-actions">
              <button
                type="button"
                className="btn btn-sm btn-danger"
                onClick={() => {
                  const next = pendingNavigation;
                  discardAll();
                  setPendingNavigation(null);
                  next();
                }}
              >
                Discard and continue
              </button>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => setPendingNavigation(null)}
              >
                Keep editing
              </button>
            </div>
          </div>
        )}
        {!config.enabled && !config.share_with_assistant && (
          <div
            className="tool-access-callout tool-access-callout-warning"
            role="status"
            data-group-protection-advisory
          >
            <strong>Content protection is off.</strong>
            <span>
              <SettingsHighlightLink
                settingId={CONTENT_PROTECTION_SETTING_ID}
                onNavigate={onNavigateToSetting}
              >
                Enable in Settings → Content protection
              </SettingsHighlightLink>{' '}
              to activate these mappings.
            </span>
          </div>
        )}
        <fieldset
          className="content-protection-fieldset"
          data-group-protection-classification={group.id}
          disabled={controlsDisabled}
        >
          <legend>Classification</legend>
          <label htmlFor={`gpp-classification-${group.id}`}>Require classification</label>
          <select
            id={`gpp-classification-${group.id}`}
            value={requirementModeFor(config, 'group', group.id)}
            onChange={(event) =>
              void changeRequirement(event.target.value as 'inherit' | 'require')
            }
          >
            <option value="inherit">Inherit</option>
            <option value="require">Require classification</option>
          </select>
          <fieldset
            className="content-protection-dialog-controls"
            data-group-protection-level-checkboxes={group.id}
          >
            <legend>Access levels</legend>
            {config.access_levels.map((level) => (
              <label key={level.id} className="checkbox-label">
                <input
                  id={`gpp-level-${group.id}-${level.id}`}
                  type="checkbox"
                  checked={mappedIds.includes(level.id)}
                  onChange={(event) => void toggleLevel(level.id, event.target.checked)}
                />
                {level.name}
                <span className="group-protection-level-reach">{levelReach(level.id)}</span>
              </label>
            ))}
          </fieldset>
        </fieldset>
        {pendingMapping && (
          <div
            className="tool-access-callout tool-access-callout-warning"
            role="alertdialog"
            aria-label={pendingMapping.enabled ? 'Confirm mapping' : 'Confirm unmap'}
            aria-describedby={`gpp-mapping-confirm-description-${group.id}`}
            data-group-protection-unmap-confirm
          >
            <p id={`gpp-mapping-confirm-description-${group.id}`}>
              Unsaved edits to hidden levels will be discarded.
            </p>
            <div className="group-protection-confirm-actions">
              <button
                type="button"
                className="btn btn-sm btn-danger"
                disabled={busy}
                onClick={() => {
                  const pending = pendingMapping;
                  for (const id of pending.hiddenDirtyIds) discardLevel(id);
                  setPendingMapping(null);
                  void runMapping(pending.levelId, pending.enabled);
                }}
              >
                {pendingMapping.enabled ? 'Discard and map' : 'Discard and unmap'}
              </button>
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                disabled={busy}
                onClick={() => setPendingMapping(null)}
              >
                Keep
              </button>
            </div>
          </div>
        )}
        <div data-group-protection-shared-levels>
          {editableLevels.map((level) => {
            const draft = drafts[level.id] ?? level;
            const conflict = conflicts[level.id];
            const deleted = conflict?.type === 'deleted';
            return (
              <details
                key={level.id}
                open
                className="content-protection-fieldset group-protection-shared-level"
                data-group-shared-level={level.id}
                data-dirty={isDirty(level.id) || undefined}
              >
                <summary className="group-protection-shared-level-summary">
                  {draft.name}
                  {isDirty(level.id) && (
                    <span className="group-protection-dirty-badge" aria-label="Unsaved changes">
                      ●
                    </span>
                  )}
                </summary>
                <p className="field-help group-protection-shared-notice">
                  Editing this level affects all groups mapped to it.{' '}
                  {config.group_access_levels
                    .filter((mapping) => mapping.access_level_id === level.id)
                    .map(
                      (mapping) =>
                        authGroups.find((candidate) => candidate.id === mapping.group_id)
                          ?.display_name || mapping.group_id,
                    )
                    .join(', ') || 'No mapped groups.'}
                </p>
                {level.id === config.default_access_level_id && (
                  <p className="field-help">
                    This fallback affects all verified users across the organization with no mapped
                    group. Other memberships override it; anonymous and public sessions are
                    excluded.
                  </p>
                )}
                {conflict && (
                  <div
                    className="error-banner"
                    role="alert"
                    data-group-protection-conflict={level.id}
                  >
                    {conflict.type === 'deleted' ? (
                      'This level was deleted by another admin. Your draft is retained.'
                    ) : (
                      <>
                        This level was updated by another admin. Your draft may conflict.
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => loadLatest(level.id)}
                        >
                          Load latest
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-danger"
                          onClick={() => void saveLevel(level.id, { overwrite: true })}
                        >
                          Overwrite latest
                        </button>
                      </>
                    )}
                  </div>
                )}
                {saveErrors[level.id] && (
                  <div className="error-banner" role="alert">
                    {saveErrors[level.id]}
                  </div>
                )}
                <ContentProtectionAccessLevelCard
                  level={draft}
                  categories={config.categories}
                  idPrefix={`gpp-${group.id}-`}
                  hideHeader
                  fieldsDisabled={busy || deleted}
                  onUpdate={updateDraft}
                />
                <div className="group-protection-level-actions">
                  <button
                    type="button"
                    className="btn btn-sm"
                    disabled={busy || deleted || !isDirty(level.id) || !draft.name.trim()}
                    onClick={() => void saveLevel(level.id)}
                  >
                    Save
                  </button>
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    disabled={busy}
                    onClick={() => {
                      discardLevel(level.id);
                    }}
                  >
                    Discard
                  </button>
                </div>
              </details>
            );
          })}
        </div>
        {mappedIds.length === 0 && (
          <div
            className="tool-access-callout tool-access-callout-warning"
            role="status"
            data-group-protection-default-warning
          >
            <strong>
              Using default level:{' '}
              {config.access_levels.find((level) => level.id === config.default_access_level_id)
                ?.name ?? 'default'}
              .
            </strong>{' '}
            Members of this group with no other group mappings receive the default level. Other
            group memberships override this fallback. Anonymous and public sessions are unaffected.
          </div>
        )}
        <details
          className="content-protection-advisory-preview group-protection-preview"
          data-group-protection-preview
        >
          <summary className="group-protection-preview-summary">
            Prompt preview {previewLoading && 'Loading…'}
          </summary>
          <p className="field-help group-protection-preview-disclaimer">
            Synthetic group-only scope. This preview combines the selected access levels as one
            identity. Other group memberships, audience settings, and sharing status alter the
            actual prompt each member receives.
          </p>
          {!config.share_with_assistant && (
            <p className="field-help">Sharing with assistant is off — prompt not sent.</p>
          )}
          {previewError && (
            <div className="error-banner" role="alert" data-group-protection-preview-error>
              {previewError}
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => void refreshPreview()}
              >
                Retry
              </button>
            </div>
          )}
          {preview && !previewError && (
            <div data-group-protection-preview-result>
              <pre>{preview.prompt_fragment || 'No prompt content for this configuration.'}</pre>
            </div>
          )}
        </details>
      </section>
    );
  },
);
