export const SETTINGS_ACCORDION_SECTION_IDS = [
  'chat-models',
  'mcp',
  'userspace',
  'llm-providers',
  'embedding',
  'authentication',
  'search',
  'appearance',
  'security',
] as const;

export type SettingsAccordionSectionId = (typeof SETTINGS_ACCORDION_SECTION_IDS)[number];

export type SettingsAccordionState = Record<SettingsAccordionSectionId, boolean>;

export const DEFAULT_OPEN_SETTINGS_SECTIONS: SettingsAccordionSectionId[] = [
  'chat-models',
  'mcp',
  'userspace',
];

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
  return { ...snapshot };
}
