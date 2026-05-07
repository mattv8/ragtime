import type { BrowseResponse } from '@/types';

export type BrowserPathDisplayMap = Record<string, string>;

export interface CloudPathDisplayOptions {
  sourceType?: string | null;
  fallbackDriveName?: string | null;
}

export function createBrowserPathDisplayMap(): BrowserPathDisplayMap {
  return { '/': '/' };
}

export function normalizeMountBrowserPath(value: string): string {
  const normalizedParts: string[] = [];
  for (const part of (value || '/').replace(/\\/g, '/').split('/')) {
    if (!part || part === '.') continue;
    if (part === '..') {
      normalizedParts.pop();
      continue;
    }
    normalizedParts.push(part);
  }
  return `/${normalizedParts.join('/')}`;
}

export function browserPathToSourcePath(browserPath: string): string {
  const normalized = normalizeMountBrowserPath(browserPath);
  return normalized === '/' ? '.' : normalized.slice(1);
}

export function sourcePathToBrowserPath(sourcePath: string): string {
  const normalized = (sourcePath || '').trim();
  if (!normalized || normalized === '.') return '/';
  return normalizeMountBrowserPath(`/${normalized}`);
}

export function browserPathToWorkspaceMountTargetPath(browserPath: string): string {
  const normalized = normalizeMountBrowserPath(browserPath);
  return normalized === '/' ? '/workspace' : `/workspace${normalized}`;
}

export function formatMountSyncPreviewPath(path: string): string {
  return path.startsWith('/') ? path : `/${path}`;
}

export function joinDisplayBrowserPath(parentPath: string, childName: string): string {
  const parent = normalizeMountBrowserPath(parentPath || '/');
  const cleanName = childName.trim().replace(/\\/g, '/').split('/').filter(Boolean).join(' ');
  if (!cleanName) return parent;
  return normalizeMountBrowserPath(parent === '/' ? `/${cleanName}` : `${parent}/${cleanName}`);
}

function cloudSourceLabel(sourceType: string | null | undefined): string {
  if (sourceType === 'microsoft_drive') return 'OneDrive';
  if (sourceType === 'google_drive') return 'Google Drive';
  return 'Cloud Drive';
}

function cleanDisplaySegment(value: string | null | undefined): string {
  return String(value || '').trim().replace(/\\/g, '/').split('/').filter(Boolean).join(' ');
}

function resolveMappedBrowserDisplayPath(
  browserPath: string,
  pathDisplayMap: BrowserPathDisplayMap | undefined,
): string | null {
  if (!pathDisplayMap) {
    return null;
  }

  const normalizedPath = normalizeMountBrowserPath(browserPath);
  if (normalizedPath === '/') {
    const mappedRoot = pathDisplayMap['/'];
    return mappedRoot ? normalizeMountBrowserPath(mappedRoot) : '/';
  }

  const segments = normalizedPath.split('/').filter(Boolean);
  for (let length = segments.length; length > 0; length -= 1) {
    const ancestorPath = `/${segments.slice(0, length).join('/')}`;
    const mappedAncestorPath = pathDisplayMap[ancestorPath];
    if (!mappedAncestorPath) {
      continue;
    }
    const suffix = segments.slice(length).join('/');
    return normalizeMountBrowserPath(suffix ? `${mappedAncestorPath}/${suffix}` : mappedAncestorPath);
  }

  return null;
}

function resolveCloudVirtualBrowserDisplayPath(
  browserPath: string,
  options: CloudPathDisplayOptions | undefined,
): string | null {
  const normalizedPath = normalizeMountBrowserPath(browserPath);
  const segments = normalizedPath.split('/').filter(Boolean);
  if (segments.length === 0) {
    return '/';
  }

  if (options?.sourceType === 'google_drive' && segments[0] === 'my-drive') {
    return normalizeMountBrowserPath(['My Drive', ...segments.slice(1)].join('/'));
  }

  if (segments[0] === 'drives' && segments.length >= 2) {
    const driveName = cleanDisplaySegment(options?.fallbackDriveName) || cloudSourceLabel(options?.sourceType);
    return normalizeMountBrowserPath([driveName, ...segments.slice(2)].join('/'));
  }

  return null;
}

export function resolveBrowserDisplayPath(
  browserPath: string,
  pathDisplayMap?: BrowserPathDisplayMap,
  options?: CloudPathDisplayOptions,
): string {
  const normalizedPath = normalizeMountBrowserPath(browserPath || '/');
  const mappedPath = resolveMappedBrowserDisplayPath(normalizedPath, pathDisplayMap);
  if (mappedPath) {
    return mappedPath;
  }
  return resolveCloudVirtualBrowserDisplayPath(normalizedPath, options) ?? normalizedPath;
}

export function resolveSourceDisplayPath(
  sourcePath: string,
  pathDisplayMap?: BrowserPathDisplayMap,
  options?: CloudPathDisplayOptions,
): string {
  return resolveBrowserDisplayPath(sourcePathToBrowserPath(sourcePath), pathDisplayMap, options);
}

export function resolveBrowserDisplaySegment(
  browserPath: string,
  fallback: string,
  pathDisplayMap?: BrowserPathDisplayMap,
  options?: CloudPathDisplayOptions,
): string {
  const segments = resolveBrowserDisplayPath(browserPath, pathDisplayMap, options).split('/').filter(Boolean);
  return segments.length > 0 ? segments[segments.length - 1] : fallback;
}

function isImmediateDisplayEntry(parentPath: string, childPath: string): boolean {
  const normalizedParent = normalizeMountBrowserPath(parentPath || '/');
  const normalizedChild = normalizeMountBrowserPath(childPath || '/');
  if (normalizedParent === normalizedChild) {
    return false;
  }

  const parentSegments = normalizedParent.split('/').filter(Boolean);
  const childSegments = normalizedChild.split('/').filter(Boolean);
  if (parentSegments.length === 0 && childSegments[0] === 'drives' && childSegments.length === 2) {
    return true;
  }
  if (parentSegments.length === 0 && childSegments[0] === 'my-drive' && childSegments.length === 1) {
    return true;
  }
  if (childSegments.length !== parentSegments.length + 1) {
    return false;
  }
  return parentSegments.every((segment, index) => childSegments[index] === segment);
}

export function mergeBrowserPathDisplayMapFromBrowseResponse(
  currentMap: BrowserPathDisplayMap,
  result: BrowseResponse,
  options?: CloudPathDisplayOptions,
): BrowserPathDisplayMap {
  const normalizedPath = normalizeMountBrowserPath(result.path);
  const parentDisplayPath = resolveBrowserDisplayPath(normalizedPath, currentMap, options);
  const nextMap: BrowserPathDisplayMap = { ...currentMap, [normalizedPath]: parentDisplayPath };

  for (const entry of result.entries || []) {
    if (!entry.is_dir) continue;
    const entryPath = normalizeMountBrowserPath(entry.path);
    nextMap[entryPath] = isImmediateDisplayEntry(normalizedPath, entryPath)
      ? joinDisplayBrowserPath(parentDisplayPath, entry.name)
      : resolveBrowserDisplayPath(entryPath, nextMap, options);
  }

  return nextMap;
}