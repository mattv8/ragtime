import {
  SETTINGS_ACCORDION_SECTION_IDS,
  type SettingsAccordionSectionId,
  type SettingsAccordionState,
} from '../components/settings/settingsAccordionState';
import { clearCookieValue, getCookieValue, setSessionCookieValue } from './cookies';

const SETTINGS_ACCORDION_COOKIE_PREFIX = 'settings_accordion_open_';
const OPEN_IDS_SEPARATOR = ',';

function createClosedSettingsAccordionState(): SettingsAccordionState {
  return Object.fromEntries(
    SETTINGS_ACCORDION_SECTION_IDS.map((id) => [id, false]),
  ) as SettingsAccordionState;
}

function isSettingsAccordionSectionId(value: string): value is SettingsAccordionSectionId {
  return (SETTINGS_ACCORDION_SECTION_IDS as readonly string[]).includes(value);
}

export function getSettingsAccordionCookieName(userId: string): string {
  return `${SETTINGS_ACCORDION_COOKIE_PREFIX}${encodeURIComponent(userId)}`;
}

export function getSettingsAccordionPreference(
  userId: string | null | undefined,
): SettingsAccordionState {
  const state = createClosedSettingsAccordionState();
  if (!userId) return state;

  const value = getCookieValue(getSettingsAccordionCookieName(userId));
  if (!value) return state;

  for (const candidate of value.split(OPEN_IDS_SEPARATOR)) {
    if (isSettingsAccordionSectionId(candidate)) {
      state[candidate] = true;
    }
  }
  return state;
}

export function setSettingsAccordionPreference(
  userId: string | null | undefined,
  state: SettingsAccordionState,
): void {
  if (!userId) return;

  const name = getSettingsAccordionCookieName(userId);
  const openIds = SETTINGS_ACCORDION_SECTION_IDS.filter((id) => state[id]);
  if (openIds.length === 0) {
    clearCookieValue(name);
    return;
  }
  setSessionCookieValue(name, openIds.join(OPEN_IDS_SEPARATOR));
}
