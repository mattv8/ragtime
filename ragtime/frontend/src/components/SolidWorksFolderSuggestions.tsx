import { ChevronRight, Folder } from 'lucide-react';
import type { PdmFolderChild } from '../api/solidworks';
import { isWithinFolder, vaultFolderKey } from '../utils/solidworksPdmLink';

export const LISTBOX_ID = 'solidworks-pdm-folder-listbox';
const OPTION_ID_PREFIX = 'solidworks-pdm-folder-option-';

export type SuggestionState =
  | 'approved'
  | 'included'
  | 'not_checked'
  | 'no_access'
  | 'can_change'
  | 'read_only';

export interface FolderSuggestion {
  folder: PdmFolderChild;
  state: SuggestionState;
  badge: string;
  tone: 'info' | 'warning' | 'success' | 'muted';
  /** Read after the folder name by screen readers; the badge itself is hidden from them. */
  description: string;
  selectable: boolean;
  canOpen: boolean;
}

/** How a vault folder relates to the approved list and the current PDM user's rights. */
export function describeSuggestion(folder: PdmFolderChild, approved: string[]): FolderSuggestion {
  const key = vaultFolderKey(folder.vault_path);
  const base = { folder, canOpen: folder.rights?.read !== false };
  if (approved.some((item) => vaultFolderKey(item) === key)) {
    return {
      ...base,
      state: 'approved',
      badge: 'Approved',
      tone: 'info',
      description: 'already approved',
      selectable: false,
    };
  }
  const parent = approved.find((item) => isWithinFolder(folder.vault_path, item));
  if (parent !== undefined) {
    return {
      ...base,
      state: 'included',
      badge: 'Included',
      tone: 'info',
      description: `included in ${parent}`,
      selectable: false,
    };
  }
  if (!folder.rights) {
    return {
      ...base,
      state: 'not_checked',
      badge: 'Not checked',
      tone: 'muted',
      description: 'rights not checked',
      selectable: true,
    };
  }
  if (!folder.rights.read) {
    return {
      ...base,
      state: 'no_access',
      badge: 'No access',
      tone: 'warning',
      description: "the PDM user can't see inside it; it can still be approved",
      selectable: true,
    };
  }
  if (folder.write_ready) {
    return {
      ...base,
      state: 'can_change',
      badge: 'Can change',
      tone: 'success',
      description: 'can add and check out',
      selectable: true,
    };
  }
  return {
    ...base,
    state: 'read_only',
    badge: 'Read only',
    tone: 'warning',
    description: 'read only for this PDM user; it can still be approved',
    selectable: true,
  };
}

/** A stable, unique element id per folder: every character outside [a-z0-9] is hex-escaped. */
export function suggestionOptionId(vaultPath: string): string {
  const escaped = vaultFolderKey(vaultPath).replace(
    /[^a-z0-9]/g,
    (char) => `_${char.charCodeAt(0).toString(16)}`,
  );
  return `${OPTION_ID_PREFIX}${escaped}`;
}

interface SolidWorksFolderSuggestionsProps {
  parent: string;
  prefix: string;
  suggestions: FolderSuggestion[];
  activeIndex: number;
  message: string | null;
  busy: boolean;
  onRetry: (() => void) | null;
  onChoose: (suggestion: FolderSuggestion) => void;
  onOpen: (suggestion: FolderSuggestion) => void;
}

/** The suggestion popup of the approved-folders combobox. Focus always stays in the input. */
export function SolidWorksFolderSuggestions({
  parent,
  prefix,
  suggestions,
  activeIndex,
  message,
  busy,
  onRetry,
  onChoose,
  onOpen,
}: SolidWorksFolderSuggestionsProps) {
  return (
    <div
      id="solidworks-pdm-folder-suggest"
      className="sw-suggest"
      aria-busy={busy || undefined}
      onMouseDown={(event) => event.preventDefault()}
    >
      {suggestions.length > 0 && (
        <>
          <p className="sw-suggest-heading" aria-hidden="true">
            Folders in <code>{parent}</code>
          </p>
          <ul
            id={LISTBOX_ID}
            role="listbox"
            aria-label={`Vault folders in ${parent}`}
            className="sw-suggest-list"
          >
            {suggestions.map((suggestion, index) => (
              <SuggestionOption
                key={vaultFolderKey(suggestion.folder.vault_path)}
                suggestion={suggestion}
                prefixLength={prefix.length}
                active={index === activeIndex}
                onChoose={onChoose}
                onOpen={onOpen}
              />
            ))}
          </ul>
        </>
      )}
      {message && (
        <p id="solidworks-pdm-folder-suggest-message" className="sw-suggest-status">
          {busy && <span className="rc-polling-indicator" aria-hidden="true" />}
          <span>{message}</span>
        </p>
      )}
      {onRetry && (
        <button
          id="solidworks-pdm-folder-suggest-retry"
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onRetry}
        >
          Try again
        </button>
      )}
    </div>
  );
}

function SuggestionOption({
  suggestion,
  prefixLength,
  active,
  onChoose,
  onOpen,
}: {
  suggestion: FolderSuggestion;
  prefixLength: number;
  active: boolean;
  onChoose: (suggestion: FolderSuggestion) => void;
  onOpen: (suggestion: FolderSuggestion) => void;
}) {
  const { folder, state, badge, tone, description, selectable, canOpen } = suggestion;
  return (
    <li
      id={suggestionOptionId(folder.vault_path)}
      role="option"
      className="sw-suggest-option"
      aria-selected={active}
      aria-disabled={selectable ? undefined : true}
      data-sw-folder={vaultFolderKey(folder.vault_path)}
      data-sw-suggest-state={state}
      onClick={() => onChoose(suggestion)}
    >
      <Folder className="sw-suggest-icon" size={16} aria-hidden="true" />
      <span className="sw-suggest-name">
        {prefixLength > 0 && <strong>{folder.name.slice(0, prefixLength)}</strong>}
        {folder.name.slice(prefixLength)}
      </span>
      <span className="sw-visually-hidden">, {description}</span>
      <span className="browser-entry-badge" data-tone={tone} aria-hidden="true">
        {badge}
      </span>
      {canOpen && (
        <span
          className="sw-suggest-drill"
          aria-hidden="true"
          title={`Open ${folder.name}`}
          onClick={(event) => {
            event.stopPropagation();
            onOpen(suggestion);
          }}
        >
          <ChevronRight size={16} />
        </span>
      )}
    </li>
  );
}
