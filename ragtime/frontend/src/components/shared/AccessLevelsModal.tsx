import { useCallback, useEffect, useRef, useState } from 'react';
import {
  ContentProtectionApiError,
  contentProtectionApi,
  updateContentProtectionConfigSlice,
  withExistingAccessLevel,
  withGroupAccessLevel,
  type AccessLevel,
  type ContentProtectionConfig,
} from '@/api/contentProtection';
import type { AuthGroup } from '@/types';
import { generateCredentialValue } from '@/utils/credentialGenerator';
import {
  AccessLevelsMasterDetail,
  accessLevelDeleteBlockedReason,
} from './AccessLevelsMasterDetail';
import { useAccessLevelDrafts } from './useAccessLevelDrafts';
import { CONTENT_PROTECTION_SETTING_ID, SettingsHighlightLink } from './SettingsHighlightLink';

interface Props {
  open: boolean;
  onClose: () => void;
  onChanged?: () => void;
  onNavigateToSetting?: (settingId: string) => void;
  authGroups: AuthGroup[];
  toast: { success: (message: string) => void; error: (message: string) => void };
}

const newLevel = (): AccessLevel => ({
  id: `level_${generateCredentialValue(12).toLowerCase()}`,
  name: '',
  granted_category_ids: [],
  guidance: '',
});

const focusableSelector =
  'button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), summary, a[href], [tabindex="0"]';

function visibleFocusableElements(container: HTMLElement | null): HTMLElement[] {
  if (!container) return [];

  return [...container.querySelectorAll<HTMLElement>(focusableSelector)].filter((element) => {
    for (let ancestor: HTMLElement | null = element; ancestor; ancestor = ancestor.parentElement) {
      const style = window.getComputedStyle(ancestor);
      if (style.display === 'none' || style.visibility === 'hidden') return false;
      if (ancestor === container) break;
    }
    return true;
  });
}

export function AccessLevelsModal({
  open,
  onClose,
  onChanged,
  onNavigateToSetting,
  authGroups,
  toast,
}: Props): JSX.Element | null {
  const [config, setConfig] = useState<ContentProtectionConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [pendingNavigation, setPendingNavigation] = useState<(() => void) | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const generation = useRef(0);
  const openerRef = useRef<HTMLElement | null>(null);
  const modalRef = useRef<HTMLDivElement>(null);
  const confirmRef = useRef<HTMLDivElement>(null);

  const reload = useCallback(async () => {
    const request = ++generation.current;
    setError(null);
    try {
      const loaded = await contentProtectionApi.getConfig();
      if (request === generation.current) setConfig(loaded);
    } catch (loadError) {
      if (request === generation.current)
        setError(loadError instanceof Error ? loadError.message : 'Could not load access levels.');
    }
  }, []);

  useEffect(() => {
    if (open) {
      openerRef.current =
        document.activeElement instanceof HTMLElement ? document.activeElement : null;
      void reload();
    } else {
      generation.current += 1;
      setConfig(null);
      setError(null);
      setSelectedId(null);
      setQuery('');
      setPendingNavigation(null);
      setDeleting(false);
      setPreview(null);
      setPreviewError(null);
      setPreviewLoading(false);
    }
  }, [open, reload]);
  useEffect(() => {
    if (!open || !config) return;
    requestAnimationFrame(() => {
      if (modalRef.current?.contains(document.activeElement)) return;
      const search = document.querySelector<HTMLInputElement>('[data-access-levels-search]');
      if (search && search.offsetParent !== null) search.focus();
      else document.getElementById('access-level-detail-title')?.focus();
    });
  }, [config, open]);
  useEffect(() => {
    if (pendingNavigation) confirmRef.current?.focus();
  }, [pendingNavigation]);

  const closeWithFocusRestore = () => {
    onClose();
    requestAnimationFrame(() => openerRef.current?.focus());
  };
  const onShellKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape' && !event.defaultPrevented) {
      event.preventDefault();
      closeWithFocusRestore();
      return;
    }
    if (event.key !== 'Tab') return;
    const items = visibleFocusableElements(modalRef.current);
    const index = items.indexOf(document.activeElement as HTMLElement);
    if (!items.length) return;
    if (event.shiftKey && index <= 0) {
      event.preventDefault();
      items[items.length - 1]?.focus();
    } else if (!event.shiftKey && index === items.length - 1) {
      event.preventDefault();
      items[0].focus();
    }
  };

  if (!open) return null;
  if (!config) {
    return (
      <div className="modal-overlay" id="access-levels-modal-overlay">
        <div
          ref={modalRef}
          className="modal-content access-levels-modal"
          role="dialog"
          aria-modal="true"
          aria-labelledby="access-levels-modal-title"
          onKeyDown={onShellKeyDown}
        >
          <div className="modal-header">
            <h3 id="access-levels-modal-title">Manage Access Levels</h3>
            <button
              type="button"
              className="modal-close"
              aria-label="Close"
              onClick={closeWithFocusRestore}
            >
              ×
            </button>
          </div>
          {error ? (
            <div className="error-banner" role="alert" data-levels-error>
              {error}
              <button
                type="button"
                className="btn btn-sm btn-secondary"
                onClick={() => void reload()}
              >
                Retry
              </button>
            </div>
          ) : (
            <div className="auth-group-empty-state" role="status" data-levels-loading>
              Loading access levels…
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <LoadedAccessLevelsModal
      {...{
        config,
        setConfig,
        selectedId,
        setSelectedId,
        query,
        setQuery,
        pendingNavigation,
        setPendingNavigation,
        deleting,
        setDeleting,
        preview,
        setPreview,
        previewError,
        setPreviewError,
        previewLoading,
        setPreviewLoading,
        modalRef,
        confirmRef,
        onClose: closeWithFocusRestore,
        onChanged,
        onNavigateToSetting,
        authGroups,
        toast,
      }}
    />
  );
}

function LoadedAccessLevelsModal(props: {
  config: ContentProtectionConfig;
  setConfig: (config: ContentProtectionConfig) => void;
  selectedId: string | null;
  setSelectedId: (id: string | null) => void;
  query: string;
  setQuery: (query: string) => void;
  pendingNavigation: (() => void) | null;
  setPendingNavigation: (action: (() => void) | null) => void;
  deleting: boolean;
  setDeleting: (value: boolean) => void;
  preview: string | null;
  setPreview: (value: string | null) => void;
  previewError: string | null;
  setPreviewError: (value: string | null) => void;
  previewLoading: boolean;
  setPreviewLoading: (value: boolean) => void;
  modalRef: { current: HTMLDivElement | null };
  confirmRef: { current: HTMLDivElement | null };
  onClose: () => void;
  onChanged?: () => void;
  onNavigateToSetting?: (settingId: string) => void;
  authGroups: AuthGroup[];
  toast: Props['toast'];
}): JSX.Element {
  const {
    config,
    setConfig,
    selectedId,
    setSelectedId,
    query,
    setQuery,
    pendingNavigation,
    setPendingNavigation,
    deleting,
    setDeleting,
    preview,
    setPreview,
    previewError,
    setPreviewError,
    previewLoading,
    setPreviewLoading,
    modalRef,
    confirmRef,
    onClose,
    onChanged,
    onNavigateToSetting,
    authGroups,
    toast,
  } = props;
  const saveConfig = (next: ContentProtectionConfig) => {
    setConfig(next);
    onChanged?.();
  };
  const draftsApi = useAccessLevelDrafts({ config, onConfigSaved: saveConfig, toast });
  const {
    drafts,
    snapshots,
    conflicts,
    saveErrors,
    savingLevelId,
    isDirty,
    hasDirty,
    updateDraft,
    createLevel,
    saveLevel,
    discardLevel,
    discardAll,
    loadLatest,
    prunedGrantsLevelId,
  } = draftsApi;
  const previewGeneration = useRef(0);
  const [previewRetry, setPreviewRetry] = useState(0);
  useEffect(() => {
    if (!selectedId)
      setSelectedId(config.default_access_level_id || config.access_levels[0]?.id || null);
    // Intentionally select only on this editor mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const selected = selectedId
    ? (drafts[selectedId] ?? config.access_levels.find((level) => level.id === selectedId))
    : undefined;
  const savedLevelIds = new Set(config.access_levels.map((level) => level.id));
  const levels = [
    ...config.access_levels.map((level) => drafts[level.id] ?? level),
    ...Object.values(drafts).filter((level) => !savedLevelIds.has(level.id)),
  ];
  const filtered = levels.filter((level) => level.name.toLowerCase().includes(query.toLowerCase()));
  const isUnsaved = Boolean(selectedId && snapshots[selectedId] === null);
  const groupIdsForLevel = (levelId: string) =>
    config.group_access_levels
      .filter((item) => item.access_level_id === levelId)
      .map((item) => item.group_id);
  const busy = Boolean(savingLevelId || deleting);
  const groups = authGroups.map((group) => ({
    id: group.id,
    label: group.display_name,
    badge: (
      <span className={`auth-group-provider-badge auth-group-provider-${group.provider}`}>
        {group.provider === 'local_managed' ? 'Internal' : group.provider.toUpperCase()}
      </span>
    ),
  }));
  const hasDeletedConflict = Boolean(selectedId && conflicts[selectedId]?.type === 'deleted');
  const requestNavigation = (action: () => void) => {
    if (deleting || savingLevelId) return;
    if (hasDirty) setPendingNavigation(() => action);
    else action();
  };
  const restoreRow = (id: string) => {
    const selectorId = id.replace(/\\|"/g, '\\$&');
    requestAnimationFrame(() =>
      document.querySelector<HTMLButtonElement>(`[data-access-level-id="${selectorId}"]`)?.focus(),
    );
  };
  const closeDetail = () =>
    requestNavigation(() => {
      const id = selectedId;
      setSelectedId(null);
      if (id) restoreRow(id);
    });
  const closeModal = () => requestNavigation(onClose);
  const select = (id: string) => {
    if (id === selectedId) return;
    requestNavigation(() => {
      setSelectedId(id);
      requestAnimationFrame(() => document.getElementById('access-level-detail-title')?.focus());
    });
  };
  useEffect(() => {
    setPreview(null);
    setPreviewError(null);
    setPreviewLoading(false);
  }, [selectedId, setPreview, setPreviewError, setPreviewLoading]);
  const updateMapping = async (groupId: string, enabled: boolean) => {
    if (!selectedId || isUnsaved || hasDeletedConflict) return;
    try {
      const saved = await updateContentProtectionConfigSlice((fresh) => {
        withExistingAccessLevel(fresh, selectedId, () => undefined);
        return withGroupAccessLevel(fresh, groupId, selectedId, enabled);
      });
      saveConfig(saved);
    } catch (mappingError) {
      toast.error(
        mappingError instanceof Error ? mappingError.message : 'Could not update group mapping.',
      );
    }
  };
  const makeDefault = async () => {
    if (!selectedId || isUnsaved || hasDeletedConflict) return;
    try {
      saveConfig(
        await updateContentProtectionConfigSlice((fresh) => {
          withExistingAccessLevel(fresh, selectedId, () => undefined);
          return { ...fresh, default_access_level_id: selectedId };
        }),
      );
    } catch (defaultError) {
      toast.error(
        defaultError instanceof Error
          ? defaultError.message
          : 'Could not make access level the default.',
      );
    }
  };
  const deleteBlockedReason = (level: AccessLevel) => {
    if (snapshots[level.id] === null || conflicts[level.id]?.type === 'deleted') return undefined;
    return accessLevelDeleteBlockedReason(
      level.id === config.default_access_level_id,
      groupIdsForLevel(level.id).length,
    );
  };
  const forgetLevel = (levelId: string) => {
    discardLevel(levelId);
    if (selectedId === levelId) setSelectedId(null);
  };
  const deleteLevel = async (level: AccessLevel): Promise<boolean> => {
    const levelId = level.id;
    if (snapshots[levelId] === null || conflicts[levelId]?.type === 'deleted') {
      forgetLevel(levelId);
      return true;
    }
    const blockedReason = deleteBlockedReason(level);
    if (blockedReason) {
      toast.error(blockedReason);
      return false;
    }
    if (deleting || savingLevelId) return false;
    setDeleting(true);
    const remove = async (base: ContentProtectionConfig) =>
      contentProtectionApi.saveConfig(base.revision, {
        ...base,
        access_levels: base.access_levels.filter((candidate) => candidate.id !== levelId),
      });
    try {
      let saved: ContentProtectionConfig;
      try {
        saved = await remove(config);
      } catch (deleteError) {
        if (!(deleteError instanceof ContentProtectionApiError) || deleteError.status !== 409)
          throw deleteError;
        const latest = await contentProtectionApi.getConfig();
        const current = latest.access_levels.find((candidate) => candidate.id === levelId);
        if (!current) {
          saveConfig(latest);
          forgetLevel(levelId);
          toast.success('Access level deleted');
          return true;
        }
        const snapshot = snapshots[levelId];
        const conflictReason =
          current.id === latest.default_access_level_id
            ? 'This access level is now the default level.'
            : latest.group_access_levels.some((mapping) => mapping.access_level_id === levelId)
              ? 'This access level is now mapped to a group.'
              : !snapshot || JSON.stringify(current) !== JSON.stringify(snapshot)
                ? 'This access level was edited by another admin.'
                : null;
        if (conflictReason) {
          setConfig(latest);
          throw new Error(conflictReason);
        }
        saved = await remove(latest);
      }
      saveConfig(saved);
      forgetLevel(levelId);
      toast.success('Access level deleted');
      return true;
    } catch (deleteError) {
      toast.error(
        deleteError instanceof Error ? deleteError.message : 'Could not delete access level.',
      );
      return false;
    } finally {
      setDeleting(false);
    }
  };
  useEffect(() => {
    if (!selectedId || isUnsaved) return;
    const request = ++previewGeneration.current;
    setPreviewLoading(true);
    setPreviewError(null);
    void contentProtectionApi
      .preview({ access_level_ids: [selectedId] })
      .then((result) => {
        if (request === previewGeneration.current)
          setPreview(result.prompt_fragment ?? 'No prompt content for this configuration.');
      })
      .catch((loadError: unknown) => {
        if (request === previewGeneration.current)
          setPreviewError(
            loadError instanceof Error ? loadError.message : 'Could not load prompt preview.',
          );
      })
      .finally(() => {
        if (request === previewGeneration.current) setPreviewLoading(false);
      });
  }, [
    config.revision,
    isUnsaved,
    previewRetry,
    selectedId,
    setPreview,
    setPreviewError,
    setPreviewLoading,
  ]);
  const onKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape' && !event.defaultPrevented) {
      event.preventDefault();
      if (selectedId) closeDetail();
      else closeModal();
      return;
    }
    if (event.key !== 'Tab') return;
    const items = visibleFocusableElements(modalRef.current);
    if (!items.length) return;
    const index = items.indexOf(document.activeElement as HTMLElement);
    if (event.shiftKey && index <= 0) {
      event.preventDefault();
      items[items.length - 1]?.focus();
    } else if (!event.shiftKey && index === items.length - 1) {
      event.preventDefault();
      items[0].focus();
    }
  };
  return (
    <div
      className="modal-overlay"
      id="access-levels-modal-overlay"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) closeModal();
      }}
    >
      <div
        ref={modalRef}
        className={`modal-content access-levels-modal${selectedId ? ' detail-open' : ''}`}
        id="manage-access-levels-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="access-levels-modal-title"
        onKeyDown={onKeyDown}
      >
        <div className="modal-header">
          <h3 id="access-levels-modal-title">Manage Access Levels</h3>
          <button type="button" className="modal-close" aria-label="Close" onClick={closeModal}>
            ×
          </button>
        </div>
        <AccessLevelsMasterDetail
          id="access-levels"
          idPrefix="alm-"
          headingId="access-level-detail-title"
          levels={filtered}
          selectedLevel={selected}
          categories={config.categories}
          groups={groups}
          defaultLevelId={config.default_access_level_id}
          groupIdsForLevel={groupIdsForLevel}
          selectedId={selectedId}
          onSelect={select}
          onCloseDetail={closeDetail}
          onAdd={() =>
            requestNavigation(() => {
              const level = newLevel();
              createLevel(level);
              setSelectedId(level.id);
            })
          }
          onUpdate={updateDraft}
          onAddGroup={(_levelId, groupId) => void updateMapping(groupId, true)}
          onRemoveGroup={(_levelId, groupId) => void updateMapping(groupId, false)}
          onMakeDefault={() => void makeDefault()}
          onDelete={deleteLevel}
          deleteBlockedReason={deleteBlockedReason}
          confirmDelete
          busy={busy}
          fieldsDisabled={busy || hasDeletedConflict}
          levelActionsDisabled={isUnsaved || hasDeletedConflict}
          groupsDisabledHint={isUnsaved ? 'Save this level before mapping groups.' : undefined}
          startEditingLevelId={isUnsaved && selected && !selected.name ? selected.id : null}
          rowIndicator={(level) => isDirty(level.id) && <span aria-label="Unsaved changes">●</span>}
          search={
            <input
              type="text"
              className="access-levels-rail-search"
              placeholder="Search access levels"
              aria-label="Search access levels"
              data-access-levels-search
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
          }
          noMatchMessage={levels.length ? 'No matching access levels.' : undefined}
          emptyHookAttribute="data-levels-empty"
          noMatchHookAttribute="data-levels-no-match"
          rowHookAttribute="data-access-level-id"
          rootProps={{
            className: 'modal-body access-levels-modal-body',
            ...{ 'data-access-levels-modal-body': '' },
          }}
          listProps={{ className: '', ...{ 'data-access-levels-rail': '' } }}
          detailProps={{ className: '', ...{ 'data-access-level-detail': '' } }}
          detailIntro={
            !config.enabled &&
            !config.share_with_assistant && (
              <div
                className="tool-access-callout tool-access-callout-warning"
                role="status"
                data-access-protection-off
              >
                <strong>Content protection is off.</strong>
                <span>
                  <SettingsHighlightLink
                    settingId={CONTENT_PROTECTION_SETTING_ID}
                    onNavigate={
                      onNavigateToSetting
                        ? (settingId) =>
                            requestNavigation(() => {
                              onClose();
                              onNavigateToSetting(settingId);
                            })
                        : undefined
                    }
                  >
                    Enable in Settings → Content protection
                  </SettingsHighlightLink>{' '}
                  to apply these access levels.
                </span>
              </div>
            )
          }
          detailExtras={
            selected && (
              <>
                <details
                  className="content-protection-advisory-preview"
                  data-level-section="preview"
                  data-level-preview
                >
                  <summary className="access-level-preview-summary">
                    Prompt preview {previewLoading && 'Loading…'}
                  </summary>
                  <p className="field-help">
                    Synthetic scope. This reflects the last saved state while a draft is dirty.
                  </p>
                  {!config.share_with_assistant && (
                    <p className="field-help">Sharing with assistant is off — prompt not sent.</p>
                  )}
                  {isUnsaved ? (
                    <p className="field-help" data-level-preview-unsaved>
                      Save to preview.
                    </p>
                  ) : previewError ? (
                    <div className="error-banner" role="alert" data-level-preview-error>
                      {previewError}
                      <button
                        type="button"
                        className="btn btn-sm btn-secondary"
                        onClick={() => setPreviewRetry((current) => current + 1)}
                      >
                        Retry
                      </button>
                    </div>
                  ) : (
                    preview && <pre data-level-preview-result>{preview}</pre>
                  )}
                </details>
                {prunedGrantsLevelId === selected.id && (
                  <p className="field-help" role="status" data-level-pruned-grants-notice>
                    Removed grants for categories that no longer exist.
                  </p>
                )}
              </>
            )
          }
          detailFooter={
            selected && (
              <>
                {conflicts[selected.id] && (
                  <div
                    className="error-banner"
                    role="alert"
                    data-level-conflict={conflicts[selected.id].type}
                  >
                    {conflicts[selected.id].type === 'deleted' ? (
                      'This level was deleted by another admin. Your draft is retained read-only.'
                    ) : (
                      <>
                        <span>Another admin updated this level.</span>
                        <button
                          type="button"
                          className="btn btn-sm btn-secondary"
                          onClick={() => loadLatest(selected.id)}
                        >
                          Load latest
                        </button>
                        <button
                          type="button"
                          className="btn btn-sm btn-danger"
                          onClick={() => void saveLevel(selected.id, { overwrite: true })}
                        >
                          Overwrite
                        </button>
                      </>
                    )}
                  </div>
                )}
                {saveErrors[selected.id] && (
                  <div className="error-banner" role="alert">
                    {saveErrors[selected.id]}
                  </div>
                )}
                <div className="access-md-footer" data-access-level-footer>
                  <button
                    type="button"
                    className="btn btn-sm btn-secondary"
                    disabled={!isDirty(selected.id)}
                    onClick={() => {
                      discardLevel(selected.id);
                      if (isUnsaved) setSelectedId(null);
                    }}
                  >
                    Discard
                  </button>
                  <button
                    type="button"
                    className={`btn btn-sm${isDirty(selected.id) ? '' : ' btn-secondary'}`}
                    disabled={
                      !isDirty(selected.id) || !selected.name.trim() || busy || hasDeletedConflict
                    }
                    onClick={() => void saveLevel(selected.id)}
                  >
                    Save
                  </button>
                </div>
              </>
            )
          }
        />
        {pendingNavigation && (
          <div
            ref={confirmRef}
            tabIndex={-1}
            className="tool-access-callout tool-access-callout-warning"
            role="alertdialog"
            aria-label="Unsaved changes"
            data-level-dirty-confirm
          >
            <p>You have unsaved changes. Discard them?</p>
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
        )}
      </div>
    </div>
  );
}
