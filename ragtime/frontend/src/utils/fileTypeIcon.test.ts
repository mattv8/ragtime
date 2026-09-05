import { describe, expect, it } from 'vitest';

import {
  EXTENSION_GLYPHS,
  SPECIAL_FILENAMES,
  getCodiconForExtension,
  getFileTypeCodicon,
} from './fileTypeIcon';

// Every glyph the mapping is allowed to emit. Confirmed present in
// @vscode/codicons/dist/codicon.css. Guards against typo'd glyph names that would
// otherwise render an empty box with no test failure.
const KNOWN_GLYPHS = new Set([
  'file',
  'file-code',
  'file-text',
  'file-media',
  'file-pdf',
  'file-zip',
  'file-binary',
  'json',
  'markdown',
  'python',
  'database',
  'folder',
]);

describe('getCodiconForExtension', () => {
  it('maps known extensions per category', () => {
    expect(getCodiconForExtension('json')).toBe('json');
    expect(getCodiconForExtension('md')).toBe('markdown');
    expect(getCodiconForExtension('py')).toBe('python');
    expect(getCodiconForExtension('sql')).toBe('database');
    expect(getCodiconForExtension('ts')).toBe('file-code');
    expect(getCodiconForExtension('png')).toBe('file-media');
    expect(getCodiconForExtension('pdf')).toBe('file-pdf');
    expect(getCodiconForExtension('zip')).toBe('file-zip');
    expect(getCodiconForExtension('exe')).toBe('file-binary');
    expect(getCodiconForExtension('txt')).toBe('file-text');
  });

  it('is case-insensitive and tolerates a leading dot', () => {
    expect(getCodiconForExtension('.TSX')).toBe('file-code');
    expect(getCodiconForExtension('JSON')).toBe('json');
    expect(getCodiconForExtension('..md')).toBe('markdown');
  });

  it('falls back to file for empty/unknown input', () => {
    expect(getCodiconForExtension('')).toBe('file');
    expect(getCodiconForExtension('.')).toBe('file');
    expect(getCodiconForExtension('qwerty')).toBe('file');
  });

  it('maps CAD extensions to file-code', () => {
    expect(getCodiconForExtension('SLDPRT')).toBe('file-code');
    expect(getCodiconForExtension('.step')).toBe('file-code');
  });
});

describe('getFileTypeCodicon', () => {
  it('resolves the extension from full paths', () => {
    expect(getFileTypeCodicon('src/components/App.tsx')).toBe('file-code');
    expect(getFileTypeCodicon('/a/b/data.json')).toBe('json');
    expect(getFileTypeCodicon('notes.md')).toBe('markdown');
  });

  it('uses only the final segment of multi-dot names', () => {
    expect(getFileTypeCodicon('foo.test.tsx')).toBe('file-code');
    expect(getFileTypeCodicon('archive.tar.gz')).toBe('file-zip');
  });

  it('is case-insensitive on the extension', () => {
    expect(getFileTypeCodicon('IMAGE.PNG')).toBe('file-media');
  });

  it('handles dot-prefixed names', () => {
    expect(getFileTypeCodicon('.env')).toBe('file-code');
    expect(getFileTypeCodicon('.eslintrc.json')).toBe('json');
    expect(getFileTypeCodicon('.eslintrc')).toBe('file');
  });

  it('matches special filenames before extension logic', () => {
    expect(getFileTypeCodicon('Dockerfile')).toBe('file-code');
    expect(getFileTypeCodicon('Makefile')).toBe('file-code');
    expect(getFileTypeCodicon('package.json')).toBe('json');
    expect(getFileTypeCodicon('README')).toBe('file-text');
    expect(getFileTypeCodicon('README.md')).toBe('markdown');
    expect(getFileTypeCodicon('LICENSE')).toBe('file');
    expect(getFileTypeCodicon('.gitignore')).toBe('file');
  });

  it('falls back to file for empty input and directory-like paths', () => {
    expect(getFileTypeCodicon('')).toBe('file');
    expect(getFileTypeCodicon('some/dir/')).toBe('file');
    expect(getFileTypeCodicon('noextension')).toBe('file');
  });

  it('only ever emits known-good codicon glyph names (exhaustive over the maps)', () => {
    // Every value in the extension map must be a real codicon glyph.
    for (const glyph of Object.values(EXTENSION_GLYPHS)) {
      expect(KNOWN_GLYPHS.has(glyph)).toBe(true);
    }
    // Every non-null value in the special-filename map must be a real codicon glyph.
    for (const glyph of Object.values(SPECIAL_FILENAMES)) {
      if (glyph != null) expect(KNOWN_GLYPHS.has(glyph)).toBe(true);
    }
    // Resolving every mapped extension through the public API stays in-set.
    for (const ext of Object.keys(EXTENSION_GLYPHS)) {
      expect(KNOWN_GLYPHS.has(getCodiconForExtension(ext))).toBe(true);
    }
    // Unknown/empty inputs fall back to a known glyph.
    for (const sample of ['a.unknownext', 'a', '', 'zzz', 'some/dir/']) {
      expect(KNOWN_GLYPHS.has(getFileTypeCodicon(sample))).toBe(true);
      expect(KNOWN_GLYPHS.has(getCodiconForExtension(sample))).toBe(true);
    }
  });
});
