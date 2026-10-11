import { useEffect, useRef, useState } from 'react';

import {
  ContentProtectionApiError,
  contentProtectionApi,
  type AccessLevel,
  type ContentProtectionConfig,
} from '@/api/contentProtection';

export type AccessLevelConflict =
  | { type: 'changed'; serverLevel: AccessLevel; latestConfig: ContentProtectionConfig }
  | { type: 'deleted' };

interface UseAccessLevelDraftsOptions {
  config: ContentProtectionConfig;
  onConfigSaved: (config: ContentProtectionConfig) => void;
  toast: { success: (message: string) => void; error: (message: string) => void };
}

export interface AccessLevelDrafts {
  drafts: Record<string, AccessLevel>;
  snapshots: Record<string, AccessLevel | null>;
  conflicts: Record<string, AccessLevelConflict>;
  saveErrors: Record<string, string>;
  savingLevelId: string | null;
  prunedGrantsLevelId: string | null;
  isDirty: (id: string) => boolean;
  hasDirty: boolean;
  updateDraft: (id: string, change: Partial<AccessLevel>) => void;
  createLevel: (level: AccessLevel) => void;
  saveLevel: (id: string, options?: { overwrite?: boolean }) => Promise<void>;
  discardLevel: (id: string) => void;
  discardAll: () => void;
  loadLatest: (id: string) => void;
}

const copyLevel = (level: AccessLevel): AccessLevel => ({
  ...level,
  granted_category_ids: [...level.granted_category_ids],
});
const sameLevel = (left: AccessLevel, right: AccessLevel) =>
  JSON.stringify(left) === JSON.stringify(right);
const withoutKey = <T>(items: Record<string, T>, key: string): Record<string, T> => {
  const { [key]: _, ...remaining } = items;
  return remaining;
};

export function useAccessLevelDrafts({
  config,
  onConfigSaved,
  toast,
}: UseAccessLevelDraftsOptions): AccessLevelDrafts {
  const [drafts, setDrafts] = useState<Record<string, AccessLevel>>({});
  const [snapshots, setSnapshots] = useState<Record<string, AccessLevel | null>>({});
  const [conflicts, setConflicts] = useState<Record<string, AccessLevelConflict>>({});
  const [saveErrors, setSaveErrors] = useState<Record<string, string>>({});
  const [savingLevelId, setSavingLevelId] = useState<string | null>(null);
  const [prunedGrantsLevelId, setPrunedGrantsLevelId] = useState<string | null>(null);
  const draftsRef = useRef(drafts);
  const snapshotsRef = useRef(snapshots);
  const baseRevisions = useRef<Record<string, number>>({});
  const baseConfigs = useRef<Record<string, ContentProtectionConfig>>({});
  const savingLevelRef = useRef<string | null>(null);

  const setDraftState = (next: Record<string, AccessLevel>) => {
    draftsRef.current = next;
    setDrafts(next);
  };
  const setSnapshotState = (next: Record<string, AccessLevel | null>) => {
    snapshotsRef.current = next;
    setSnapshots(next);
  };

  useEffect(() => {
    const serverLevels = Object.fromEntries(config.access_levels.map((level) => [level.id, level]));
    const nextDrafts = { ...draftsRef.current };
    const nextSnapshots = { ...snapshotsRef.current };
    const nextConflicts = { ...conflicts };

    for (const [id, serverLevel] of Object.entries(serverLevels)) {
      const draft = nextDrafts[id];
      const snapshot = nextSnapshots[id];
      const dirty = Boolean(draft && snapshot && !sameLevel(draft, snapshot));
      if (!dirty) {
        nextDrafts[id] = copyLevel(serverLevel);
        nextSnapshots[id] = copyLevel(serverLevel);
        baseRevisions.current[id] = config.revision;
        baseConfigs.current[id] = config;
      } else if (snapshot && !sameLevel(serverLevel, snapshot)) {
        nextConflicts[id] = {
          type: 'changed',
          serverLevel: copyLevel(serverLevel),
          latestConfig: config,
        };
      }
    }
    for (const [id, draft] of Object.entries(nextDrafts)) {
      const snapshot = nextSnapshots[id];
      if (!serverLevels[id] && snapshot && !sameLevel(draft, snapshot))
        nextConflicts[id] = { type: 'deleted' };
      else if (!serverLevels[id] && snapshot) {
        delete nextDrafts[id];
        delete nextSnapshots[id];
      }
    }
    setDraftState(nextDrafts);
    setSnapshotState(nextSnapshots);
    setConflicts(nextConflicts);
    // Dirty drafts deliberately retain their captured base across config refreshes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config]);

  const isDirty = (id: string) => {
    const draft = drafts[id];
    const snapshot = snapshots[id];
    return Boolean(draft && (snapshot === null || (snapshot && !sameLevel(draft, snapshot))));
  };
  const hasDirty = Object.keys(drafts).some(isDirty);

  const updateDraft = (id: string, change: Partial<AccessLevel>) => {
    const granted_category_ids = change.granted_category_ids?.filter(
      (categoryId) => categoryId !== 'rule_override',
    );
    const next = {
      ...draftsRef.current,
      [id]: {
        ...draftsRef.current[id],
        ...change,
        ...(granted_category_ids ? { granted_category_ids } : {}),
      },
    };
    setDraftState(next);
  };

  const createLevel = (level: AccessLevel) => {
    setDraftState({ ...draftsRef.current, [level.id]: copyLevel(level) });
    setSnapshotState({ ...snapshotsRef.current, [level.id]: null });
    baseRevisions.current[level.id] = config.revision;
    baseConfigs.current[level.id] = config;
  };

  const completeSave = (saved: ContentProtectionConfig, levelId: string) => {
    const savedLevel = saved.access_levels.find((level) => level.id === levelId);
    if (!savedLevel) return;
    baseRevisions.current[levelId] = saved.revision;
    baseConfigs.current[levelId] = saved;
    setSnapshotState({ ...snapshotsRef.current, [levelId]: copyLevel(savedLevel) });
    setDraftState({ ...draftsRef.current, [levelId]: copyLevel(savedLevel) });
    setConflicts((current) => withoutKey(current, levelId));
    setSaveErrors((current) => withoutKey(current, levelId));
    onConfigSaved(saved);
  };

  const saveLevel = async (
    levelId: string,
    options: { overwrite?: boolean } = {},
  ): Promise<void> => {
    const draft = draftsRef.current[levelId];
    if (savingLevelRef.current === levelId) return;
    if (!draft || !draft.name.trim()) {
      toast.error('Access level name is required.');
      return;
    }
    savingLevelRef.current = levelId;
    setSavingLevelId(levelId);
    setSaveErrors((current) => withoutKey(current, levelId));
    const normalizedDraft = { ...draft, name: draft.name.trim() };
    const snapshot = snapshotsRef.current[levelId];
    const save = (base: ContentProtectionConfig) => {
      const access_levels = base.access_levels.some((level) => level.id === levelId)
        ? base.access_levels.map((level) => (level.id === levelId ? normalizedDraft : level))
        : [...base.access_levels, normalizedDraft];
      return contentProtectionApi.saveConfig(base.revision, { ...base, access_levels });
    };
    const setChangedConflict = (latest: ContentProtectionConfig, serverLevel: AccessLevel) => {
      setConflicts((current) => ({
        ...current,
        [levelId]: { type: 'changed', serverLevel: copyLevel(serverLevel), latestConfig: latest },
      }));
    };

    try {
      let saved: ContentProtectionConfig;
      if (options.overwrite) {
        const latest = await contentProtectionApi.getConfig();
        const serverLevel = latest.access_levels.find((level) => level.id === levelId);
        if (!serverLevel) {
          setConflicts((current) => ({ ...current, [levelId]: { type: 'deleted' } }));
          return;
        }
        saved = await save(latest);
      } else {
        const base = baseConfigs.current[levelId] ?? config;
        try {
          saved = await save({
            ...base,
            revision: baseRevisions.current[levelId] ?? base.revision,
          });
        } catch (error) {
          if (!(error instanceof ContentProtectionApiError) || error.status !== 409) throw error;
          const latest = await contentProtectionApi.getConfig();
          const serverLevel = latest.access_levels.find((level) => level.id === levelId);
          if (snapshot === null && !serverLevel) {
            const categoryIds = new Set(latest.categories.map((category) => category.id));
            normalizedDraft.granted_category_ids = normalizedDraft.granted_category_ids.filter(
              (id) => categoryIds.has(id),
            );
            if (normalizedDraft.granted_category_ids.length !== draft.granted_category_ids.length)
              setPrunedGrantsLevelId(levelId);
            saved = await save(latest);
          } else if (!serverLevel) {
            setConflicts((current) => ({ ...current, [levelId]: { type: 'deleted' } }));
            return;
          } else if (!snapshot || !sameLevel(serverLevel, snapshot)) {
            setChangedConflict(latest, serverLevel);
            return;
          } else {
            try {
              saved = await save(latest);
            } catch (retryError) {
              if (retryError instanceof ContentProtectionApiError && retryError.status === 409) {
                setChangedConflict(latest, serverLevel);
                return;
              }
              throw retryError;
            }
          }
        }
      }
      completeSave(saved, levelId);
      toast.success('Access level saved');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not save access level.';
      setSaveErrors((current) => ({ ...current, [levelId]: message }));
      toast.error(message);
    } finally {
      savingLevelRef.current = null;
      setSavingLevelId(null);
    }
  };

  const discardLevel = (id: string) => {
    const snapshot = snapshotsRef.current[id];
    if (snapshot === null) {
      setDraftState(withoutKey(draftsRef.current, id));
      setSnapshotState(withoutKey(snapshotsRef.current, id));
    } else if (snapshot && conflicts[id]?.type === 'deleted') {
      setDraftState(withoutKey(draftsRef.current, id));
      setSnapshotState(withoutKey(snapshotsRef.current, id));
    } else if (snapshot) {
      setDraftState({ ...draftsRef.current, [id]: copyLevel(snapshot) });
    }
    setConflicts((current) => withoutKey(current, id));
    setSaveErrors((current) => withoutKey(current, id));
  };

  const discardAll = () => {
    const nextDrafts = { ...draftsRef.current };
    const nextSnapshots = { ...snapshotsRef.current };
    for (const [id, snapshot] of Object.entries(nextSnapshots)) {
      if (snapshot === null) {
        delete nextDrafts[id];
        delete nextSnapshots[id];
      } else {
        nextDrafts[id] = copyLevel(snapshot);
      }
    }
    setDraftState(nextDrafts);
    setSnapshotState(nextSnapshots);
  };

  const loadLatest = (id: string) => {
    const conflict = conflicts[id];
    if (conflict?.type !== 'changed') return;
    baseRevisions.current[id] = conflict.latestConfig.revision;
    baseConfigs.current[id] = conflict.latestConfig;
    setDraftState({ ...draftsRef.current, [id]: copyLevel(conflict.serverLevel) });
    setSnapshotState({ ...snapshotsRef.current, [id]: copyLevel(conflict.serverLevel) });
    setConflicts((current) => withoutKey(current, id));
    setSaveErrors((current) => withoutKey(current, id));
  };

  return {
    drafts,
    snapshots,
    conflicts,
    saveErrors,
    savingLevelId,
    prunedGrantsLevelId,
    isDirty,
    hasDirty,
    updateDraft,
    createLevel,
    saveLevel,
    discardLevel,
    discardAll,
    loadLatest,
  };
}
