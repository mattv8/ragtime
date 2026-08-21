import {
  DEFAULT_THEME_PACK_ID,
  THEME_PACK_STORAGE_KEY,
  normalizeThemePackId,
  type ThemePackId,
} from './themes';

export function resolveThemePackId(
  userPack: string | null | undefined,
  globalDefault: string | null | undefined,
): ThemePackId {
  return (
    normalizeThemePackId(userPack) ?? normalizeThemePackId(globalDefault) ?? DEFAULT_THEME_PACK_ID
  );
}

export function getStoredThemePack(): ThemePackId {
  try {
    const stored = localStorage.getItem(THEME_PACK_STORAGE_KEY);
    const resolved = normalizeThemePackId(stored);
    if (resolved && stored === 'vscode') {
      localStorage.setItem(THEME_PACK_STORAGE_KEY, 'modern');
    }
    if (resolved) {
      return resolved;
    }
  } catch {
    /* localStorage unavailable */
  }
  return DEFAULT_THEME_PACK_ID;
}

export function applyThemePack(pack: ThemePackId): void {
  const root = document.documentElement;
  if (pack === DEFAULT_THEME_PACK_ID) {
    root.removeAttribute('data-theme-pack');
  } else {
    root.setAttribute('data-theme-pack', pack);
  }
}

export function setThemePack(pack: ThemePackId): void {
  applyThemePack(pack);
  try {
    if (pack === DEFAULT_THEME_PACK_ID) {
      localStorage.removeItem(THEME_PACK_STORAGE_KEY);
    } else {
      localStorage.setItem(THEME_PACK_STORAGE_KEY, pack);
    }
  } catch {
    /* localStorage unavailable */
  }
}
