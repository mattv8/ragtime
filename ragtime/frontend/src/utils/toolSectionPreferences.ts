import { clearCookieValue, getCookieValue, setSessionCookieValue } from './cookies';

const COOKIE_PREFIX = 'tools_sections_collapsed_';
const MAX_COLLAPSED_SECTIONS = 60;

export function getCollapsedToolSections(userId: string | null | undefined): Set<string> {
  if (!userId) return new Set();
  const value = getCookieValue(`${COOKIE_PREFIX}${encodeURIComponent(userId)}`);
  if (!value) return new Set();
  const ids = new Set<string>();
  for (const candidate of value.split(',')) {
    try {
      const id = decodeURIComponent(candidate);
      if (id) ids.add(id);
    } catch {
      // Ignore malformed entries without discarding the remaining preferences.
    }
  }
  return ids;
}

export function setCollapsedToolSections(
  userId: string | null | undefined,
  collapsedIds: Iterable<string>,
  knownIds: ReadonlySet<string>,
): void {
  if (!userId) return;
  const name = `${COOKIE_PREFIX}${encodeURIComponent(userId)}`;
  const ids = [...new Set([...collapsedIds].filter((id) => knownIds.has(id)))].slice(
    0,
    MAX_COLLAPSED_SECTIONS,
  );
  if (!ids.length) clearCookieValue(name);
  else setSessionCookieValue(name, ids.map(encodeURIComponent).join(','));
}
