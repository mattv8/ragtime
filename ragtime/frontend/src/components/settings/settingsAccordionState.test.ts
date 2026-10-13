import { describe, expect, it } from 'vitest';
import {
  DEFAULT_OPEN_SETTINGS_SECTIONS,
  applySettingsAccordionSearch,
  clearSettingsAccordionSearch,
  createSettingsAccordionViewState,
  getDefaultSettingsAccordionState,
  getSettingsAccordionVisibleState,
  openSettingsAccordionNavigation,
  openSettingsAccordionSections,
  restoreSettingsAccordionState,
  SETTINGS_ACCORDION_SECTION_IDS,
  toggleSettingsAccordionSection,
  type SettingsAccordionSectionId,
  type SettingsAccordionState,
} from './settingsAccordionState';

describe('settingsAccordionState', () => {
  describe('SETTINGS_ACCORDION_SECTION_IDS', () => {
    it('lists all accordion section ids in order', () => {
      expect(SETTINGS_ACCORDION_SECTION_IDS).toEqual([
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
      ]);
    });
  });

  describe('DEFAULT_OPEN_SETTINGS_SECTIONS', () => {
    it('defaults every section to closed', () => {
      expect(DEFAULT_OPEN_SETTINGS_SECTIONS).toEqual([]);
    });
  });

  describe('getDefaultSettingsAccordionState', () => {
    it('closes every section', () => {
      const state = getDefaultSettingsAccordionState();
      for (const id of SETTINGS_ACCORDION_SECTION_IDS) {
        expect(state[id]).toBe(false);
      }
    });

    it('returns a new object on each call', () => {
      const a = getDefaultSettingsAccordionState();
      const b = getDefaultSettingsAccordionState();
      expect(a).not.toBe(b);
      expect(a).toEqual(b);
    });
  });

  describe('openSettingsAccordionSections', () => {
    it('opens the requested sections while preserving the rest', () => {
      const current: SettingsAccordionState = {
        chat: false,
        'agent-behavior': false,
        'content-protection': false,
        mcp: true,
        userspace: false,
        'llm-providers': true,
        embedding: false,
        authentication: false,
        search: false,
        appearance: false,
        'server-backup-restore': false,
        security: false,
      };
      const next = openSettingsAccordionSections(current, ['chat', 'userspace']);
      expect(next).toEqual({
        chat: true,
        'agent-behavior': false,
        'content-protection': false,
        mcp: true,
        userspace: true,
        'llm-providers': true,
        embedding: false,
        authentication: false,
        search: false,
        appearance: false,
        'server-backup-restore': false,
        security: false,
      });
    });

    it('does not mutate the current state', () => {
      const current = getDefaultSettingsAccordionState();
      const copy = { ...current };
      const next = openSettingsAccordionSections(current, ['appearance']);
      expect(current).toEqual(copy);
      expect(next).not.toBe(current);
    });

    it.each([
      {
        label: 'array input',
        sections: ['appearance'] as SettingsAccordionSectionId[],
        expectedOpen: 'appearance' as SettingsAccordionSectionId,
      },
      {
        label: 'set input',
        sections: new Set<SettingsAccordionSectionId>(['security']),
        expectedOpen: 'security' as SettingsAccordionSectionId,
      },
    ])('accepts section ids from any iterable (%s)', ({ sections, expectedOpen }) => {
      const current = getDefaultSettingsAccordionState();
      const next = openSettingsAccordionSections(current, sections);
      expect(next[expectedOpen]).toBe(true);
    });
  });

  describe('restoreSettingsAccordionState', () => {
    it('returns a copy of the provided snapshot when present', () => {
      const snapshot: SettingsAccordionState = {
        chat: false,
        'agent-behavior': true,
        'content-protection': false,
        mcp: false,
        userspace: false,
        'llm-providers': true,
        embedding: true,
        authentication: true,
        search: true,
        appearance: true,
        'server-backup-restore': false,
        security: true,
      };
      const restored = restoreSettingsAccordionState(snapshot);
      expect(restored).toEqual(snapshot);
      expect(restored).not.toBe(snapshot);
    });

    it.each([
      { label: 'null', snapshot: null },
      { label: 'undefined', snapshot: undefined },
    ])('returns the default state when given $label', ({ snapshot }) => {
      expect(restoreSettingsAccordionState(snapshot)).toEqual(getDefaultSettingsAccordionState());
    });

    it('adds defaults for newly introduced sections when restoring an older snapshot', () => {
      const restored = restoreSettingsAccordionState({
        userspace: false,
      } as SettingsAccordionState);
      expect(restored.chat).toBe(false);
      expect(restored.userspace).toBe(false);
    });
  });

  describe('view state', () => {
    it('shows remembered sections and opens search matches unless manually toggled', () => {
      const view = createSettingsAccordionViewState({
        ...getDefaultSettingsAccordionState(),
        chat: true,
      });
      const searched = applySettingsAccordionSearch(view, ['mcp', 'userspace']);
      const manuallyClosed = toggleSettingsAccordionSection(searched, 'mcp');

      expect(getSettingsAccordionVisibleState(manuallyClosed)).toMatchObject({
        chat: true,
        mcp: false,
        userspace: true,
      });
      expect(manuallyClosed.remembered.mcp).toBe(false);
      expect(manuallyClosed.searchOpened).toEqual(['mcp', 'userspace']);
      expect(manuallyClosed.manuallyToggled).toEqual(['mcp']);
    });

    it('uses the visible value when toggling an automatically opened section', () => {
      const searched = applySettingsAccordionSearch(
        createSettingsAccordionViewState(getDefaultSettingsAccordionState()),
        ['search'],
      );

      const toggled = toggleSettingsAccordionSection(searched, 'search');

      expect(toggled.remembered.search).toBe(false);
      expect(getSettingsAccordionVisibleState(toggled).search).toBe(false);
    });

    it('retains manual overrides across search updates and clears them with search', () => {
      const searched = applySettingsAccordionSearch(
        createSettingsAccordionViewState(getDefaultSettingsAccordionState()),
        ['mcp'],
      );
      const manuallyOpened = toggleSettingsAccordionSection(searched, 'chat');
      const updatedSearch = applySettingsAccordionSearch(manuallyOpened, ['mcp', 'security']);
      const cleared = clearSettingsAccordionSearch(updatedSearch);

      expect(getSettingsAccordionVisibleState(updatedSearch)).toMatchObject({
        chat: true,
        mcp: true,
        security: true,
      });
      expect(cleared.searchOpened).toEqual([]);
      expect(cleared.manuallyToggled).toEqual([]);
      expect(getSettingsAccordionVisibleState(cleared).chat).toBe(true);
    });

    it('remembers navigation openings and supersedes a prior manual close', () => {
      const searched = applySettingsAccordionSearch(
        createSettingsAccordionViewState(getDefaultSettingsAccordionState()),
        ['appearance'],
      );
      const manuallyClosed = toggleSettingsAccordionSection(searched, 'appearance');
      const navigated = openSettingsAccordionNavigation(manuallyClosed, ['appearance']);

      expect(navigated.remembered.appearance).toBe(true);
      expect(navigated.manuallyToggled).toEqual([]);
      expect(getSettingsAccordionVisibleState(navigated).appearance).toBe(true);
    });

    it('does not mutate inputs', () => {
      const view = createSettingsAccordionViewState(getDefaultSettingsAccordionState());
      const remembered = { ...view.remembered };

      const next = toggleSettingsAccordionSection(view, 'chat');

      expect(view.remembered).toEqual(remembered);
      expect(next).not.toBe(view);
      expect(next.remembered).not.toBe(view.remembered);
    });
  });
});
