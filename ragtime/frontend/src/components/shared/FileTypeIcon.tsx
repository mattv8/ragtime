import type { CSSProperties } from 'react';

import { getCodiconForExtension, getFileTypeCodicon } from '@/utils/fileTypeIcon';

export interface FileTypeIconProps {
  /** Full path or filename; used to derive the filetype glyph. */
  path?: string;
  /** Alias for a bare filename (equivalent to `path`). */
  name?: string;
  /** Extension-first callers (e.g. stats tables) that have no full name. */
  extension?: string;
  /** Render a folder glyph instead of a filetype glyph. */
  isDir?: boolean;
  /** Icon size in px (drives font-size of the codicon). Default 16. */
  size?: number;
  className?: string;
  /** Optional tooltip. The icon is decorative (aria-hidden) by default. */
  title?: string;
}

/**
 * Renders a VS Code "codicon" glyph representing a file's type. Unlike `ThemeChromeIcon`,
 * this renders in ALL theme packs (the codicon font is imported globally in icons.css).
 * Purely decorative: it always sits next to a visible filename/path, so it is aria-hidden.
 */
export function FileTypeIcon({
  path,
  name,
  extension,
  isDir = false,
  size = 16,
  className,
  title,
}: FileTypeIconProps) {
  let glyph: string;
  if (isDir) {
    glyph = 'folder';
  } else if (extension != null && extension !== '') {
    glyph = getCodiconForExtension(extension);
  } else {
    glyph = getFileTypeCodicon(path ?? name ?? '');
  }

  const classes = ['file-type-icon', 'codicon', `codicon-${glyph}`, className]
    .filter(Boolean)
    .join(' ');

  return (
    <span
      aria-hidden="true"
      className={classes}
      style={{ fontSize: `${size}px` } as CSSProperties}
      title={title}
    />
  );
}
