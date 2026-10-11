import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  SETTINGS_ACCORDION_SECTION_IDS,
  type SettingsAccordionSectionId,
  type SettingsAccordionState,
} from '../components/settings/settingsAccordionState';
import {
  getSettingsAccordionCookieName,
  getSettingsAccordionPreference,
  setSettingsAccordionPreference,
} from './settingsAccordionPreferences';

const COOKIE_PREFIX = 'settings_accordion_open_';

function stateWithOpen(...openIds: SettingsAccordionSectionId[]): SettingsAccordionState {
  return Object.fromEntries(
    SETTINGS_ACCORDION_SECTION_IDS.map((id) => [id, openIds.includes(id)]),
  ) as SettingsAccordionState;
}

function openIdsOf(state: SettingsAccordionState): SettingsAccordionSectionId[] {
  return SETTINGS_ACCORDION_SECTION_IDS.filter((id) => state[id]);
}

afterEach(() => {
  vi.restoreAllMocks();
  document.cookie.split('; ').forEach((entry) => {
    document.cookie = `${entry.split('=')[0]}=; path=/; max-age=0`;
  });
});

describe('settingsAccordionPreferences', () => {
  describe('getSettingsAccordionCookieName', () => {
    it('prefixes the cookie with the encoded user id', () => {
      expect(getSettingsAccordionCookieName('admin/a')).toBe(`${COOKIE_PREFIX}admin%2Fa`);
    });
  });

  describe('getSettingsAccordionPreference', () => {
    it('returns every section closed when there is no user id', () => {
      document.cookie = `${getSettingsAccordionCookieName('')}=chat; path=/`;
      expect(getSettingsAccordionPreference(null)).toEqual(stateWithOpen());
      expect(getSettingsAccordionPreference(undefined)).toEqual(stateWithOpen());
      expect(getSettingsAccordionPreference('')).toEqual(stateWithOpen());
    });

    it('returns every section closed when the user has no cookie', () => {
      expect(getSettingsAccordionPreference('new-user')).toEqual(stateWithOpen());
    });

    it('ignores the default open sections and keeps all sections closed', () => {
      document.cookie = `${getSettingsAccordionCookieName('alice')}=; path=/`;
      const state = getSettingsAccordionPreference('alice');
      expect(openIdsOf(state)).toEqual([]);
      expect(Object.keys(state)).toHaveLength(SETTINGS_ACCORDION_SECTION_IDS.length);
    });

    it('keeps only known section ids from the cookie', () => {
      document.cookie = `${getSettingsAccordionCookieName('alice')}=chat,bogus,,security,__proto__; path=/`;
      const state = getSettingsAccordionPreference('alice');
      expect(openIdsOf(state)).toEqual(['chat', 'security']);
      expect(Object.keys(state)).toHaveLength(SETTINGS_ACCORDION_SECTION_IDS.length);
    });

    it('falls back to closed sections for malformed encoding', () => {
      document.cookie = `${getSettingsAccordionCookieName('alice')}=%E0%A4%A; path=/`;
      expect(() => getSettingsAccordionPreference('alice')).not.toThrow();
      expect(getSettingsAccordionPreference('alice')).toEqual(stateWithOpen());
    });
  });

  describe('setSettingsAccordionPreference', () => {
    it('restores the open sections written for the same user', () => {
      setSettingsAccordionPreference('alice', stateWithOpen('mcp', 'security'));
      expect(openIdsOf(getSettingsAccordionPreference('alice'))).toEqual(['mcp', 'security']);
    });

    it('writes only known open ids as a comma-separated cookie value', () => {
      setSettingsAccordionPreference('alice', stateWithOpen('security', 'chat'));
      expect(document.cookie).toContain(
        `${getSettingsAccordionCookieName('alice')}=chat%2Csecurity`,
      );
    });

    it('writes a session cookie without expiration or max-age', () => {
      const setCookie = vi.spyOn(document, 'cookie', 'set');
      setSettingsAccordionPreference('alice', stateWithOpen('chat'));
      expect(setCookie).toHaveBeenCalledTimes(1);
      const written = setCookie.mock.calls[0][0];
      expect(written).not.toMatch(/max-age|expires/i);
    });

    it('keeps users independent', () => {
      setSettingsAccordionPreference('alice', stateWithOpen('chat'));
      setSettingsAccordionPreference('bob', stateWithOpen('userspace'));
      expect(openIdsOf(getSettingsAccordionPreference('alice'))).toEqual(['chat']);
      expect(openIdsOf(getSettingsAccordionPreference('bob'))).toEqual(['userspace']);
      expect(openIdsOf(getSettingsAccordionPreference('carol'))).toEqual([]);
    });

    it('deletes the cookie when every section is closed', () => {
      setSettingsAccordionPreference('alice', stateWithOpen('chat'));
      setSettingsAccordionPreference('alice', stateWithOpen());
      expect(document.cookie).not.toContain(getSettingsAccordionCookieName('alice'));
      expect(openIdsOf(getSettingsAccordionPreference('alice'))).toEqual([]);
    });

    it('does nothing when there is no user id', () => {
      setSettingsAccordionPreference(null, stateWithOpen('chat'));
      setSettingsAccordionPreference(undefined, stateWithOpen('chat'));
      setSettingsAccordionPreference('', stateWithOpen('chat'));
      expect(document.cookie).not.toContain(COOKIE_PREFIX);
    });
  });
});
