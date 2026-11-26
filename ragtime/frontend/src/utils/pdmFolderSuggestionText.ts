/** Visible and announced text for the vault folder suggestion list of the approved-folders box. */

export const MAX_SHOWN_MATCHES = 100;

export interface SuggestionListing {
  status: 'idle' | 'loading' | 'ready' | 'error';
  /** Folders loaded for the parent; may be fewer than `total`. */
  loaded: number;
  total: number;
  error: { message: string; code: string } | null;
}

export interface SuggestionTextInput {
  listing: SuggestionListing;
  parent: string;
  prefix: string;
  matchCount: number;
}

const ROOT = '\\';
const TYPED_PATH_STILL_WORKS = 'You can still type a path and press Enter.';

export function joinVaultPath(parent: string, name: string): string {
  return parent === ROOT ? `${ROOT}${name}` : `${parent}${ROOT}${name}`;
}

function noMatch(parent: string, prefix: string): string {
  return `No folder in ${parent} starts with “${prefix}”. Press Enter to add ${joinVaultPath(parent, prefix)} anyway.`;
}

function errorText({ listing, parent, prefix }: SuggestionTextInput): string {
  if (listing.error?.code === 'pdm_not_found') {
    if (prefix) return noMatch(parent, prefix);
    return parent === ROOT
      ? `The PDM user can't list the vault root. ${TYPED_PATH_STILL_WORKS}`
      : `${parent} isn't in the vault, or the PDM user can't see it. ${TYPED_PATH_STILL_WORKS}`;
  }
  const reason = (listing.error?.message ?? 'unknown error').replace(/\.$/, '');
  return `Couldn't load folders in ${parent}: ${reason}. Press the down arrow or choose Try again to retry. ${TYPED_PATH_STILL_WORKS}`;
}

function readyText({ listing, parent, prefix, matchCount }: SuggestionTextInput): string | null {
  const truncated = listing.loaded < listing.total;
  if (matchCount === 0) {
    if (!prefix)
      return parent === ROOT ? 'No folders in the vault.' : `No subfolders in ${parent}.`;
    if (!truncated) return noMatch(parent, prefix);
    return `No loaded folder in ${parent} starts with “${prefix}”. Only the first ${listing.loaded} of ${listing.total} are loaded, so it may still exist. Press Enter to add ${joinVaultPath(parent, prefix)} anyway.`;
  }
  const notes: string[] = [];
  if (matchCount > MAX_SHOWN_MATCHES) {
    notes.push(
      `Showing the first ${MAX_SHOWN_MATCHES} of ${matchCount} matches. Keep typing to narrow the list.`,
    );
  }
  if (truncated) {
    notes.push(
      `Showing the first ${listing.loaded} of ${listing.total} folders in ${parent}. Keep typing to narrow the list, or type the full path and press Enter.`,
    );
  }
  return notes.length > 0 ? notes.join(' ') : null;
}

/** The status row under the suggestions; null when there is nothing to say. */
export function suggestionMessage(
  input: SuggestionTextInput & { showLoading: boolean },
): string | null {
  switch (input.listing.status) {
    case 'loading':
      return input.showLoading ? `Loading folders in ${input.parent}…` : null;
    case 'error':
      return errorText(input);
    case 'ready':
      return readyText(input);
    default:
      return null;
  }
}

/** The polite screen-reader summary of the list; `opened` when the user just opened `parent`. */
export function suggestionAnnouncement(
  input: SuggestionTextInput & { message: string | null; opened: boolean },
): string {
  const { listing, parent, matchCount, message, opened } = input;
  if (listing.status === 'error') return message ?? '';
  if (listing.status !== 'ready') return '';
  const lead = opened ? `Opened ${parent}. ` : '';
  if (matchCount === 0) return `${lead}No matching folders.`;
  const count = `${matchCount} ${matchCount === 1 ? 'folder' : 'folders'}`;
  return opened ? `${lead}${count}.` : `${count} in ${parent}.`;
}
