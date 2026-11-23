import { describe, expect, it } from 'vitest';
import type { PdmRightsItem, PdmRightsResponse } from '../api/solidworks';
import { resolveRightsState } from './solidworksPdmRights';

const rights = { read: true, add_files: true, check_out: true, add_folders: false };
const item = (overrides: Partial<PdmRightsItem> = {}): PdmRightsItem => ({
  path: '\\Folder',
  checked: true,
  exists: true,
  rights,
  write_ready: true,
  ...overrides,
});
const response = (
  root: PdmRightsItem | null,
  folders: PdmRightsItem[] = [],
): PdmRightsResponse => ({
  checked: true,
  user: 'user',
  root,
  folders,
  error: null,
});

describe('resolveRightsState', () => {
  it.each([
    [null, 'not_checked'],
    [item({ checked: false }), 'not_checked'],
    [item({ checked: true, exists: null }), 'not_checked'],
    [item(), 'ok'],
    [item({ exists: false, rights: null }), 'no_read'],
    [item({ rights: { ...rights, read: false } }), 'no_read'],
  ] as const)('resolves root %#', (root, expected) => {
    expect(
      resolveRightsState({ response: response(root), allowChanges: true, folders: [] }).root,
    ).toBe(expected);
  });

  it.each([
    [undefined, 'not_checked'],
    [item({ checked: false }), 'not_checked'],
    [item({ exists: false, rights: null, write_ready: false }), 'not_found'],
    [item(), 'write_ready'],
    [item({ write_ready: false }), 'missing_write'],
  ] as const)('resolves folder %# case-insensitively', (folder, expected) => {
    const state = resolveRightsState({
      response: response(item(), folder ? [folder] : []),
      allowChanges: true,
      folders: ['\\FOLDER'],
    });
    expect(state.folders).toEqual({ '\\folder': expected });
    expect(state.user).toBe('user');
  });

  it('never reports folder rights when changes are off and reports extra root rights', () => {
    const state = resolveRightsState({
      response: response(item(), [item()]),
      allowChanges: false,
      folders: ['\\Folder'],
    });
    expect(state.folders['\\folder']).toBe('not_checked');
    expect(state.extraWriteInfo).toBe(true);
  });

  it.each([
    [null, false],
    [item({ checked: false }), false],
    [item({ exists: false }), false],
    [item({ rights: { ...rights, add_files: false, check_out: false } }), false],
    [item({ rights: { ...rights, add_files: false } }), true],
    [item({ rights: { ...rights, check_out: false } }), true],
  ] as const)('calculates extra write info %#', (root, expected) => {
    expect(
      resolveRightsState({ response: response(root), allowChanges: false, folders: [] })
        .extraWriteInfo,
    ).toBe(expected);
  });

  it('returns unchecked state for a missing response', () => {
    expect(resolveRightsState({ response: null, allowChanges: true, folders: ['\\A'] })).toEqual({
      root: 'not_checked',
      folders: { '\\a': 'not_checked' },
      extraWriteInfo: false,
      user: null,
    });
  });
});
