export const SETTINGS_ACCORDION_SECTION_IDS = [
  'chat',
  'agent-behavior',
  'content-protection',
  'mcp',
  'userspace',
  'llm-providers',
  'embedding',
  'authentication',
  'search',
  'appearance',
  'server-backup-restore',
  'security',
] as const;

export type SettingsAccordionSectionId = (typeof SETTINGS_ACCORDION_SECTION_IDS)[number];

export type SettingsAccordionState = Record<SettingsAccordionSectionId, boolean>;

export const DEFAULT_OPEN_SETTINGS_SECTIONS: SettingsAccordionSectionId[] = [];

export function getDefaultSettingsAccordionState(): SettingsAccordionState {
  return Object.fromEntries(
    SETTINGS_ACCORDION_SECTION_IDS.map((id) => [id, DEFAULT_OPEN_SETTINGS_SECTIONS.includes(id)]),
  ) as SettingsAccordionState;
}

export function openSettingsAccordionSections(
  current: SettingsAccordionState,
  sectionIds: Iterable<SettingsAccordionSectionId>,
): SettingsAccordionState {
  const next = { ...current };
  for (const id of sectionIds) {
    if (id in next) {
      next[id] = true;
    }
  }
  return next;
}

export function restoreSettingsAccordionState(
  snapshot: SettingsAccordionState | null | undefined,
): SettingsAccordionState {
  if (!snapshot) {
    return getDefaultSettingsAccordionState();
  }
  return { ...getDefaultSettingsAccordionState(), ...snapshot };
}

export interface SettingsAccordionViewState {
  remembered: SettingsAccordionState;
  searchOpened: SettingsAccordionSectionId[];
  manuallyToggled: SettingsAccordionSectionId[];
}

export function createSettingsAccordionViewState(
  remembered: SettingsAccordionState,
): SettingsAccordionViewState {
  return {
    remembered: { ...remembered },
    searchOpened: [],
    manuallyToggled: [],
  };
}

export function getSettingsAccordionVisibleState(
  view: SettingsAccordionViewState,
): SettingsAccordionState {
  const manuallyToggled = new Set(view.manuallyToggled);
  const visible = { ...view.remembered };

  for (const id of view.searchOpened) {
    if (!manuallyToggled.has(id)) {
      visible[id] = true;
    }
  }

  return visible;
}

export function toggleSettingsAccordionSection(
  view: SettingsAccordionViewState,
  id: SettingsAccordionSectionId,
): SettingsAccordionViewState {
  const visible = getSettingsAccordionVisibleState(view);

  return {
    ...view,
    remembered: { ...view.remembered, [id]: !visible[id] },
    manuallyToggled: addSectionId(view.manuallyToggled, id),
  };
}

export function applySettingsAccordionSearch(
  view: SettingsAccordionViewState,
  ids: Iterable<SettingsAccordionSectionId>,
): SettingsAccordionViewState {
  return {
    ...view,
    searchOpened: uniqueSectionIds(ids),
    manuallyToggled: [...view.manuallyToggled],
  };
}

export function clearSettingsAccordionSearch(
  view: SettingsAccordionViewState,
): SettingsAccordionViewState {
  return {
    ...view,
    remembered: { ...view.remembered },
    searchOpened: [],
    manuallyToggled: [],
  };
}

export function openSettingsAccordionNavigation(
  view: SettingsAccordionViewState,
  ids: Iterable<SettingsAccordionSectionId>,
): SettingsAccordionViewState {
  const openedIds = uniqueSectionIds(ids);
  const openedIdSet = new Set(openedIds);

  return {
    ...view,
    remembered: openSettingsAccordionSections(view.remembered, openedIds),
    searchOpened: [...view.searchOpened],
    manuallyToggled: view.manuallyToggled.filter((id) => !openedIdSet.has(id)),
  };
}

function addSectionId(
  ids: SettingsAccordionSectionId[],
  id: SettingsAccordionSectionId,
): SettingsAccordionSectionId[] {
  return ids.includes(id) ? [...ids] : [...ids, id];
}

function uniqueSectionIds(ids: Iterable<SettingsAccordionSectionId>): SettingsAccordionSectionId[] {
  return [...new Set(ids)];
}
