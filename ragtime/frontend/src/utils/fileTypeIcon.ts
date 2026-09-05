/**
 * Maps a file (by path, name, or bare extension) to a VS Code "codicon" glyph name.
 *
 * This is intentionally separate from `codemirrorLanguage.ts` (which resolves syntax
 * highlighting): here we resolve *iconography*. Codicons are monochrome and lack
 * per-language glyphs for most languages, so many code files share `file-code`.
 *
 * Pure module (no React import) so it can be used from both JSX components and
 * imperative DOM code (e.g. RichChatInput chips).
 */

const DEFAULT_GLYPH = 'file';

// Extension -> codicon glyph. Keys are lowercase, no leading dot.
// Exported so tests can exhaustively assert every emitted glyph is a real codicon.
export const EXTENSION_GLYPHS: Record<string, string> = {
  // JSON
  json: 'json',
  jsonc: 'json',
  json5: 'json',
  jsonl: 'json',
  // Markdown
  md: 'markdown',
  markdown: 'markdown',
  mdx: 'markdown',
  // Python
  py: 'python',
  pyi: 'python',
  pyw: 'python',
  // SQL
  sql: 'database',
  // Generic code
  ts: 'file-code',
  tsx: 'file-code',
  mts: 'file-code',
  cts: 'file-code',
  js: 'file-code',
  jsx: 'file-code',
  mjs: 'file-code',
  cjs: 'file-code',
  css: 'file-code',
  scss: 'file-code',
  sass: 'file-code',
  less: 'file-code',
  html: 'file-code',
  htm: 'file-code',
  xml: 'file-code',
  vue: 'file-code',
  svelte: 'file-code',
  sh: 'file-code',
  bash: 'file-code',
  zsh: 'file-code',
  c: 'file-code',
  h: 'file-code',
  cpp: 'file-code',
  hpp: 'file-code',
  cc: 'file-code',
  go: 'file-code',
  rs: 'file-code',
  rb: 'file-code',
  php: 'file-code',
  java: 'file-code',
  kt: 'file-code',
  swift: 'file-code',
  toml: 'file-code',
  ini: 'file-code',
  cfg: 'file-code',
  conf: 'file-code',
  yml: 'file-code',
  yaml: 'file-code',
  env: 'file-code',
  // CAD (SolidWorks / PDM)
  sldprt: 'file-code',
  sldasm: 'file-code',
  slddrw: 'file-code',
  step: 'file-code',
  stp: 'file-code',
  iges: 'file-code',
  igs: 'file-code',
  // Media
  png: 'file-media',
  jpg: 'file-media',
  jpeg: 'file-media',
  gif: 'file-media',
  webp: 'file-media',
  svg: 'file-media',
  ico: 'file-media',
  bmp: 'file-media',
  avif: 'file-media',
  mp4: 'file-media',
  mov: 'file-media',
  webm: 'file-media',
  mp3: 'file-media',
  wav: 'file-media',
  // PDF
  pdf: 'file-pdf',
  // Archives
  zip: 'file-zip',
  tar: 'file-zip',
  gz: 'file-zip',
  tgz: 'file-zip',
  bz2: 'file-zip',
  '7z': 'file-zip',
  rar: 'file-zip',
  xz: 'file-zip',
  // Binary
  exe: 'file-binary',
  bin: 'file-binary',
  so: 'file-binary',
  dll: 'file-binary',
  dylib: 'file-binary',
  wasm: 'file-binary',
  o: 'file-binary',
  a: 'file-binary',
  class: 'file-binary',
  pyc: 'file-binary',
  // Text
  txt: 'file-text',
  log: 'file-text',
  csv: 'file-text',
  tsv: 'file-text',
  rtf: 'file-text',
};

// Exact basename matches (lowercased), checked before extension logic.
// Value `null` means "fall through to extension logic".
// Exported so tests can exhaustively assert every emitted glyph is a real codicon.
export const SPECIAL_FILENAMES: Record<string, string | null> = {
  dockerfile: 'file-code',
  makefile: 'file-code',
  'cmakelists.txt': 'file-code',
  license: 'file',
  licence: 'file',
  readme: 'file-text',
  '.gitignore': 'file',
  '.gitattributes': 'file',
  '.editorconfig': 'file',
  '.npmrc': 'file',
  '.dockerignore': 'file',
};

/**
 * Resolve a codicon glyph name from a bare extension (with or without a leading dot,
 * case-insensitive). Returns `DEFAULT_GLYPH` for unknown/empty input.
 */
export function getCodiconForExtension(ext: string): string {
  if (!ext) return DEFAULT_GLYPH;
  const normalized = ext.replace(/^\.+/, '').toLowerCase();
  if (!normalized) return DEFAULT_GLYPH;
  return EXTENSION_GLYPHS[normalized] ?? DEFAULT_GLYPH;
}

/**
 * Resolve a codicon glyph name from a full path or filename.
 *
 * Extension rule: substring after the LAST dot in the basename. A name whose only dot is
 * at index 0 (dot-prefixed) has no extension. Special filenames are matched first.
 */
export function getFileTypeCodicon(pathOrName: string): string {
  if (!pathOrName) return DEFAULT_GLYPH;

  // Basename: strip a trailing slash, then take the segment after the last slash.
  const trimmed = pathOrName.replace(/\/+$/, '');
  if (!trimmed) return DEFAULT_GLYPH;
  const basename = trimmed.slice(trimmed.lastIndexOf('/') + 1);
  if (!basename) return DEFAULT_GLYPH;

  const lowerBase = basename.toLowerCase();
  if (Object.prototype.hasOwnProperty.call(SPECIAL_FILENAMES, lowerBase)) {
    const mapped = SPECIAL_FILENAMES[lowerBase];
    if (mapped) return mapped;
    // null -> fall through to extension logic below.
  }

  const lastDot = basename.lastIndexOf('.');
  // No dot at all -> no extension (special filenames already handled above).
  if (lastDot < 0) return DEFAULT_GLYPH;

  // Extension = segment after the last dot (dot-prefixed names like `.env` use `env`).
  const ext = basename.slice(lastDot + 1);
  if (!ext) return DEFAULT_GLYPH; // trailing dot, e.g. "foo."
  return getCodiconForExtension(ext);
}
