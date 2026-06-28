export type ColorMode = 'light' | 'dark' | 'system';

export const COLOR_MODE_STORAGE_KEY = 'ragtime-theme';

export function getStoredColorMode(): ColorMode {
  try {
    const stored = localStorage.getItem(COLOR_MODE_STORAGE_KEY);
    if (stored === 'light' || stored === 'dark') {
      return stored;
    }
  } catch {
    /* localStorage unavailable */
  }
  return 'system';
}

export function applyColorMode(mode: ColorMode): void {
  const root = document.documentElement;
  if (mode === 'system') {
    root.removeAttribute('data-theme');
  } else {
    root.setAttribute('data-theme', mode);
  }
}

export function setColorMode(mode: ColorMode): void {
  applyColorMode(mode);
  try {
    if (mode === 'system') {
      localStorage.removeItem(COLOR_MODE_STORAGE_KEY);
    } else {
      localStorage.setItem(COLOR_MODE_STORAGE_KEY, mode);
    }
  } catch {
    /* localStorage unavailable */
  }
}

export function resolveEffectiveColorMode(mode: ColorMode): 'light' | 'dark' {
  if (mode === 'system') {
    return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
  }
  return mode;
}
