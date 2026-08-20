import { resolveThemePackId } from './applyThemePack';
import type { ThemePackId } from './themes';

export interface ThemeSnapshot {
  pack: ThemePackId;
  mode: 'light' | 'dark';
}

const subscribers = new Set<() => void>();

let cachedSnapshot: ThemeSnapshot | null = null;
let mutationObserver: MutationObserver | null = null;
let mediaQueryList: MediaQueryList | null = null;
let mediaQueryListener: ((event: MediaQueryListEvent) => void) | null = null;
let mediaQueryFactory: typeof window.matchMedia | null = null;

function getMediaQueryList(): MediaQueryList | null {
  if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
    return null;
  }

  if (!mediaQueryList || mediaQueryFactory !== window.matchMedia) {
    mediaQueryList = window.matchMedia('(prefers-color-scheme: light)');
    mediaQueryFactory = window.matchMedia;
  }

  return mediaQueryList;
}

function readSystemMode(): 'light' | 'dark' {
  return getMediaQueryList()?.matches ? 'light' : 'dark';
}

function readThemeSnapshot(): ThemeSnapshot {
  if (typeof document === 'undefined') {
    return { pack: 'default', mode: 'dark' };
  }

  const root = document.documentElement;
  const explicitMode = root.getAttribute('data-theme');

  return {
    pack: resolveThemePackId(root.getAttribute('data-theme-pack'), null),
    mode: explicitMode === 'light' || explicitMode === 'dark' ? explicitMode : readSystemMode(),
  };
}

function themeSnapshotsEqual(left: ThemeSnapshot | null, right: ThemeSnapshot): boolean {
  return left?.pack === right.pack && left?.mode === right.mode;
}

function cacheThemeSnapshot(nextSnapshot: ThemeSnapshot): ThemeSnapshot {
  if (themeSnapshotsEqual(cachedSnapshot, nextSnapshot) && cachedSnapshot) {
    return cachedSnapshot;
  }

  cachedSnapshot = nextSnapshot;
  return nextSnapshot;
}

function notifyIfChanged(): void {
  const nextSnapshot = readThemeSnapshot();
  if (themeSnapshotsEqual(cachedSnapshot, nextSnapshot)) {
    return;
  }

  cacheThemeSnapshot(nextSnapshot);
  for (const subscriber of subscribers) {
    subscriber();
  }
}

function ensureObservers(): void {
  if (typeof document === 'undefined' || typeof window === 'undefined' || subscribers.size === 0) {
    return;
  }

  cacheThemeSnapshot(readThemeSnapshot());

  if (!mutationObserver) {
    mutationObserver = new MutationObserver(() => {
      notifyIfChanged();
    });
    mutationObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme', 'data-theme-pack'],
    });
  }

  const nextMediaQueryList = getMediaQueryList();
  if (nextMediaQueryList && !mediaQueryListener) {
    mediaQueryListener = () => {
      notifyIfChanged();
    };
    if (typeof nextMediaQueryList.addEventListener === 'function') {
      nextMediaQueryList.addEventListener('change', mediaQueryListener);
    } else if (typeof nextMediaQueryList.addListener === 'function') {
      nextMediaQueryList.addListener(mediaQueryListener);
    }
  }
}

function teardownObservers(): void {
  if (subscribers.size > 0) {
    return;
  }

  mutationObserver?.disconnect();
  mutationObserver = null;

  if (mediaQueryList && mediaQueryListener) {
    if (typeof mediaQueryList.removeEventListener === 'function') {
      mediaQueryList.removeEventListener('change', mediaQueryListener);
    } else if (typeof mediaQueryList.removeListener === 'function') {
      mediaQueryList.removeListener(mediaQueryListener);
    }
  }

  mediaQueryListener = null;
}

export function getThemeSnapshot(): ThemeSnapshot {
  return cacheThemeSnapshot(readThemeSnapshot());
}

export function subscribeToThemeChanges(listener: () => void): () => void {
  subscribers.add(listener);
  ensureObservers();

  return () => {
    subscribers.delete(listener);
    teardownObservers();
  };
}
