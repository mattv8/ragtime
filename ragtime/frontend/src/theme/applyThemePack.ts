import {
  DEFAULT_THEME_PACK_ID,
  THEME_PACK_STORAGE_KEY,
  isThemePackId,
  type ThemePackId,
} from './themes';

export function resolveThemePackId(
  userPack: string | null | undefined,
  globalDefault: string | null | undefined,
): ThemePackId {
  if (isThemePackId(userPack)) {
    return userPack;
  }
  if (isThemePackId(globalDefault)) {
    return globalDefault;
  }
  return DEFAULT_THEME_PACK_ID;
}

export function getStoredThemePack(): ThemePackId {
  try {
    const stored = localStorage.getItem(THEME_PACK_STORAGE_KEY);
    if (isThemePackId(stored)) {
      return stored;
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
