import { describe, expect, it } from 'vitest';
import {
  DEFAULT_OPEN_SETTINGS_SECTIONS,
  getDefaultSettingsAccordionState,
  openSettingsAccordionSections,
  restoreSettingsAccordionState,
  SETTINGS_ACCORDION_SECTION_IDS,
  type SettingsAccordionSectionId,
  type SettingsAccordionState,
} from './settingsAccordionState';

describe('settingsAccordionState', () => {
  describe('SETTINGS_ACCORDION_SECTION_IDS', () => {
    it('lists all accordion section ids in order', () => {
      expect(SETTINGS_ACCORDION_SECTION_IDS).toEqual([
        'chat-models',
        'agent-behavior',
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
    it('defaults to chat-models, mcp, and userspace', () => {
      expect(DEFAULT_OPEN_SETTINGS_SECTIONS).toEqual(['chat-models', 'mcp', 'userspace']);
    });

    it('keeps agent-behavior closed by default', () => {
      expect(DEFAULT_OPEN_SETTINGS_SECTIONS).not.toContain('agent-behavior');
    });
  });

  describe('getDefaultSettingsAccordionState', () => {
    it('opens the default sections and closes all others', () => {
      const state = getDefaultSettingsAccordionState();
      for (const id of SETTINGS_ACCORDION_SECTION_IDS) {
        expect(state[id]).toBe(
          DEFAULT_OPEN_SETTINGS_SECTIONS.includes(id as SettingsAccordionSectionId),
        );
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
        'chat-models': false,
        'agent-behavior': false,
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
      const next = openSettingsAccordionSections(current, ['chat-models', 'userspace']);
      expect(next).toEqual({
        'chat-models': true,
        'agent-behavior': false,
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
        'chat-models': false,
        'agent-behavior': true,
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
  });
});
