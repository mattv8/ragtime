import { afterEach, describe, expect, it, vi } from 'vitest';
import { getCollapsedToolSections, setCollapsedToolSections } from './toolSectionPreferences';

afterEach(() => {
  vi.restoreAllMocks();
  document.cookie.split('; ').forEach((entry) => {
    document.cookie = `${entry.split('=')[0]}=; path=/; max-age=0`;
  });
});

describe('tool section preferences', () => {
  it('round trips encoded ids, prunes unknown ids and deduplicates', () => {
    const spy = vi.spyOn(document, 'cookie', 'set');
    setCollapsedToolSections('user/a', ['a,b', 'x%', 'unknown', 'a,b'], new Set(['a,b', 'x%']));
    expect(getCollapsedToolSections('user/a')).toEqual(new Set(['a,b', 'x%']));
    expect(document.cookie).toContain('tools_sections_collapsed_user%2Fa=');
    expect(spy.mock.calls[0][0]).not.toMatch(/expires|max-age/i);
  });
  it('caps stored ids at 60', () => {
    const ids = Array.from({ length: 70 }, (_, i) => `g${i}`);
    setCollapsedToolSections('alice', ids, new Set(ids));
    expect([...getCollapsedToolSections('alice')]).toEqual(ids.slice(0, 60));
  });
  it('clears empty preferences and isolates users', () => {
    setCollapsedToolSections('alice', ['a'], new Set(['a']));
    setCollapsedToolSections('bob', ['b'], new Set(['b']));
    setCollapsedToolSections('alice', [], new Set(['a']));
    expect(getCollapsedToolSections('alice')).toEqual(new Set());
    expect(document.cookie).not.toContain('tools_sections_collapsed_alice=');
    expect(getCollapsedToolSections('bob')).toEqual(new Set(['b']));
  });
  it('does nothing without a user', () => {
    const spy = vi.spyOn(document, 'cookie', 'set');
    for (const user of [null, undefined, '']) {
      setCollapsedToolSections(user, ['a'], new Set(['a']));
      expect(getCollapsedToolSections(user)).toEqual(new Set());
    }
    expect(spy).not.toHaveBeenCalled();
  });
});
