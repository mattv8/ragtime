import { useEffect, useRef, useState, type KeyboardEvent } from 'react';
import { AlertTriangle, CheckCircle, X } from 'lucide-react';
import type { PdmFolderChild } from '../api/solidworks';
import { usePdmFolderSuggestions } from '../hooks/usePdmFolderSuggestions';
import {
  MAX_SHOWN_MATCHES,
  suggestionAnnouncement,
  suggestionMessage,
} from '../utils/pdmFolderSuggestionText';
import {
  addApprovedFolder,
  removeApprovedFolder,
  splitVaultDraft,
  vaultFolderKey,
} from '../utils/solidworksPdmLink';
import type { FolderRightsState } from '../utils/solidworksPdmRights';
import { InfoPopover } from './shared/InfoPopover';
import { PdmMessage, type PdmMessageTone } from './shared/PdmMessage';
import {
  LISTBOX_ID,
  SolidWorksFolderSuggestions,
  describeSuggestion,
  suggestionOptionId,
  type FolderSuggestion,
} from './SolidWorksFolderSuggestions';

/** Where folder suggestions come from, or why there are none. Typed paths always work. */
export type FolderSuggestionSource =
  | { toolId: string; pdmToolId: string }
  | { unavailableReason: string };

export interface SolidWorksApprovedFoldersProps {
  folders: string[];
  onChange: (next: string[]) => void;
  vaultRoot: string | null;
  suggestions: FolderSuggestionSource;
  disabled?: boolean;
  rights?: Record<string, FolderRightsState>;
}

const EXAMPLE_FOLDER = '\\13_Ragtime';
const INPUT_ID = 'solidworks-pdm-folder-input';
const FEEDBACK_ID = 'solidworks-pdm-folder-feedback';
const HINT_ID = 'solidworks-pdm-folder-hint';
const UNAVAILABLE_ID = 'solidworks-pdm-folder-suggest-unavailable';
const POPUP_SELECTOR = '#solidworks-pdm-folder-suggest';
const REMOVE_BUTTON_SELECTOR = '[data-sw-action="remove-folder"]';
const LOADING_DELAY_MS = 200;
const ANNOUNCE_DELAY_MS = 400;
const READ_ONLY_WARNING =
  "The PDM user can't add or check out files there yet, so the AI's changes there will be refused until you grant those rights in the PDM Administration tool.";

interface FolderFeedback {
  text: string;
  tone: PdmMessageTone;
}

export function approvedCountLabel(count: number): string {
  return `${count} ${count === 1 ? 'folder' : 'folders'} approved.`;
}

/** The draft that lists `parent` again, so siblings of a folder just added are one keystroke away. */
function draftFor(parent: string): string {
  return parent === '\\' ? '' : `${parent}\\`;
}

function startsWithPrefix(name: string, prefix: string): boolean {
  return name.toLowerCase().startsWith(prefix.toLowerCase());
}

/** Rights were checked and the PDM user can't add and check out files there. */
function lacksWriteRights(folder: PdmFolderChild | undefined): boolean {
  return folder !== undefined && folder.rights !== null && !folder.write_ready;
}

function caretAtEnd(input: HTMLInputElement): boolean {
  return input.selectionStart === input.value.length && input.selectionEnd === input.value.length;
}

export function SolidWorksApprovedFolders({
  folders,
  onChange,
  vaultRoot,
  suggestions: source,
  disabled,
  rights,
}: SolidWorksApprovedFoldersProps) {
  const [draft, setDraft] = useState('');
  const [feedback, setFeedback] = useState<FolderFeedback | null>(null);
  const [status, setStatus] = useState({ text: '', revision: 0 });
  const [open, setOpen] = useState(false);
  const [engaged, setEngaged] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [openedParent, setOpenedParent] = useState<string | null>(null);
  const [loadingShownFor, setLoadingShownFor] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState('');
  const comboboxRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const pendingRemoveIndex = useRef<number | null>(null);

  const vault = 'toolId' in source ? source : null;
  const available = vault !== null;
  const split = available ? splitVaultDraft(draft, vaultRoot) : null;
  const parent = split?.parent ?? null;
  const prefix = split?.prefix ?? '';
  const listing = usePdmFolderSuggestions({
    toolId: vault?.toolId ?? null,
    pdmToolId: vault?.pdmToolId ?? null,
    parent,
    enabled: available && engaged,
  });
  const matches = listing.items.filter((item) => startsWithPrefix(item.name, prefix));
  const options = matches
    .slice(0, MAX_SHOWN_MATCHES)
    .map((item) => describeSuggestion(item, folders));
  const loadingKey = listing.status === 'loading' ? parent : null;
  // A fast answer never flashes a loading row, but a folder the user just opened shows one at once.
  const showLoading =
    loadingKey !== null && (loadingShownFor === loadingKey || openedParent === loadingKey);
  const textInput =
    parent === null
      ? null
      : {
          listing: {
            status: listing.status,
            loaded: listing.items.length,
            total: listing.total,
            error: listing.error,
          },
          parent,
          prefix,
          matchCount: matches.length,
        };
  const message = textInput && suggestionMessage({ ...textInput, showLoading });
  const popupVisible =
    available && open && !disabled && parent !== null && (options.length > 0 || message !== null);
  const expanded = popupVisible && options.length > 0;
  const active = expanded ? (options[activeIndex] ?? null) : null;
  const popupAnnouncement =
    popupVisible && textInput
      ? suggestionAnnouncement({ ...textInput, message, opened: openedParent === parent })
      : '';

  // Runs once the parent has committed the shorter list, so the focus target exists in the DOM.
  useEffect(() => {
    const index = pendingRemoveIndex.current;
    if (index === null) return;
    pendingRemoveIndex.current = null;
    const buttons = removeButtons();
    const target = buttons[Math.min(index, buttons.length - 1)] ?? inputRef.current;
    target?.focus();
  }, [folders]);

  useEffect(() => {
    if (loadingKey === null) {
      setLoadingShownFor(null);
      return;
    }
    const timer = setTimeout(() => setLoadingShownFor(loadingKey), LOADING_DELAY_MS);
    return () => clearTimeout(timer);
  }, [loadingKey]);

  useEffect(() => {
    const timer = setTimeout(() => setAnnouncement(popupAnnouncement), ANNOUNCE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [popupAnnouncement]);

  // The box is often the last field on the page; keep the list above the fold and the keyboard.
  useEffect(() => {
    if (!popupVisible) return;
    comboboxRef.current
      ?.querySelector<HTMLElement>(POPUP_SELECTOR)
      ?.scrollIntoView?.({ block: 'nearest' });
  }, [popupVisible, options.length]);

  const activeOptionId = active ? suggestionOptionId(active.folder.vault_path) : null;
  useEffect(() => {
    if (activeOptionId === null) return;
    document.getElementById(activeOptionId)?.scrollIntoView?.({ block: 'nearest' });
  }, [activeOptionId]);

  function removeButtons(): HTMLButtonElement[] {
    return Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>(REMOVE_BUTTON_SELECTOR) ?? [],
    );
  }

  /** The revision re-renders the live region, so the same message is announced again. */
  function announce(text: string) {
    setStatus((previous) => ({ text, revision: previous.revision + 1 }));
  }

  function closePopup() {
    setOpen(false);
    setActiveIndex(-1);
  }

  function changeDraft(next: string) {
    setDraft(next);
    setActiveIndex(-1);
    setOpenedParent(null);
  }

  function reportAdded(added: string, notice: string | null, readOnly: boolean, count: number) {
    const lead = notice ? `${notice} ` : '';
    if (readOnly)
      setFeedback({ text: `${lead}Added ${added}. ${READ_ONLY_WARNING}`, tone: 'warning' });
    else setFeedback(notice ? { text: notice, tone: 'info' } : null);
    const rightsNote = readOnly ? " The PDM user can't change files there yet." : '';
    announce(`${lead}Added ${added}. ${approvedCountLabel(count)}${rightsNote}`);
  }

  /** Adds `input` to the list. `suggestion` is the listed folder it came from, if any. */
  function add(input: string, suggestion?: PdmFolderChild) {
    const result = addApprovedFolder(folders, input, vaultRoot);
    if (!result.ok) {
      setFeedback({ text: result.error, tone: result.info ? 'info' : 'error' });
      announce(result.error);
      return;
    }
    onChange(result.folders);
    const known =
      suggestion ??
      listing.items.find(
        (item) => vaultFolderKey(item.vault_path) === vaultFolderKey(result.added),
      );
    reportAdded(result.added, result.notice, lacksWriteRights(known), result.folders.length);
    changeDraft(suggestion && parent !== null ? draftFor(parent) : '');
    inputRef.current?.focus();
  }

  /** Approved and included rows are refused by `addApprovedFolder`, which explains why. */
  function choose(option: FolderSuggestion) {
    add(option.folder.vault_path, option.folder);
  }

  function openFolder(option: FolderSuggestion) {
    if (!option.canOpen) return;
    const path = option.folder.vault_path;
    changeDraft(`${path}\\`);
    setOpenedParent(path);
    setOpen(true);
    setFeedback(null);
    inputRef.current?.focus();
  }

  function remove(index: number) {
    const folder = folders[index];
    pendingRemoveIndex.current = index;
    const next = removeApprovedFolder(folders, folder);
    onChange(next);
    setFeedback(null);
    announce(`Removed ${folder}. ${approvedCountLabel(next.length)}`);
  }

  function moveActive(step: 1 | -1) {
    if (!expanded) {
      setOpen(true);
      setActiveIndex(step === 1 ? 0 : options.length - 1);
      return;
    }
    const count = options.length;
    // Wraps through "no active option" (-1), so the typed text can be committed again.
    setActiveIndex((current) => ((current + 1 + step + count + 1) % (count + 1)) - 1);
  }

  function focusLastChip(): boolean {
    const buttons = removeButtons();
    const last = buttons[buttons.length - 1];
    if (!last) return false;
    last.focus();
    announce(`Last folder ${folders[folders.length - 1]}. Press Enter or Backspace to remove it.`);
    return true;
  }

  const handleInputKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.nativeEvent.isComposing) return;
    const input = event.currentTarget;
    switch (event.key) {
      case 'ArrowDown':
      case 'ArrowUp':
        if (!available) return;
        event.preventDefault();
        if (
          event.key === 'ArrowDown' &&
          popupVisible &&
          listing.status === 'error' &&
          options.length === 0
        ) {
          listing.retry();
          return;
        }
        moveActive(event.key === 'ArrowDown' ? 1 : -1);
        return;
      case 'Enter':
        event.preventDefault();
        if (active) choose(active);
        else add(draft);
        return;
      case 'ArrowRight':
        if (active?.canOpen && caretAtEnd(input)) {
          event.preventDefault();
          openFolder(active);
        }
        return;
      case '\\':
      case '/':
        if (active?.canOpen) {
          event.preventDefault();
          openFolder(active);
        }
        return;
      case 'Tab':
        if (!event.shiftKey && active?.canOpen && draft !== active.folder.vault_path) {
          event.preventDefault();
          openFolder(active);
          return;
        }
        closePopup();
        return;
      case 'Escape':
        if (popupVisible) {
          event.preventDefault();
          event.stopPropagation();
          closePopup();
        } else if (draft) {
          event.preventDefault();
          event.stopPropagation();
          changeDraft('');
          setFeedback(null);
        }
        return;
      case 'Backspace':
        // A held key empties the draft; it must not go on to remove folders.
        if (event.repeat) return;
        if (draft === '' && input.selectionStart === 0 && focusLastChip()) {
          event.preventDefault();
        }
        return;
    }
  };

  const handleChipKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    const buttons = removeButtons();
    switch (event.key) {
      case 'Backspace':
      case 'Delete':
        event.preventDefault();
        if (!event.repeat) remove(index);
        return;
      case 'ArrowLeft':
        event.preventDefault();
        buttons[Math.max(index - 1, 0)]?.focus();
        return;
      case 'ArrowRight':
        event.preventDefault();
        (buttons[index + 1] ?? inputRef.current)?.focus();
        return;
      case 'Escape':
        event.preventDefault();
        event.stopPropagation();
        inputRef.current?.focus();
        return;
    }
  };

  const engage = () => {
    if (available) setEngaged(true);
  };

  const hasError = feedback?.tone === 'error';
  const helpId = available ? HINT_ID : UNAVAILABLE_ID;
  const describedBy = feedback ? `${helpId} ${FEEDBACK_ID}` : helpId;
  const placeholder = !available
    ? `Type a vault folder, like ${EXAMPLE_FOLDER}`
    : folders.length > 0
      ? 'Add another folder'
      : `Type \\ to browse, or a path like ${EXAMPLE_FOLDER}`;

  return (
    <div
      id="solidworks-pdm-folders"
      role="group"
      aria-labelledby="solidworks-pdm-folders-label"
      data-testid="solidworks-pdm-folders"
      className="form-group"
    >
      <div className="sw-field-label">
        <span id="solidworks-pdm-folders-label" className="sw-group-label">
          Approved folders
        </span>
        <InfoPopover id="solidworks-pdm-folders-help" label="About approved folders">
          Vault folders the AI may change, such as <code>{EXAMPLE_FOLDER}</code>. Each folder
          includes its subfolders. Changes anywhere else in the vault are refused. Click the field,
          press the down arrow or type <code>\</code> to browse folders, then press Enter or click a
          folder to add it. Open a folder with the arrow at the end of its row or the right arrow
          key. You can approve a folder the PDM user can't change yet. It is marked, and the AI's
          changes there are refused until you grant the rights in the PDM Administration tool.
        </InfoPopover>
      </div>
      <label htmlFor={INPUT_ID} className="sw-visually-hidden">
        Add a vault folder
      </label>
      <div id="solidworks-pdm-folder-combobox" className="sw-combobox" ref={comboboxRef}>
        <div id="solidworks-pdm-folder-box" className="sw-chip-input">
          {folders.length > 0 && (
            <ul
              id="solidworks-pdm-folder-list"
              ref={listRef}
              className="sw-chip-list"
              data-testid="solidworks-pdm-folder-list"
            >
              {folders.map((folder, index) => (
                <FolderChip
                  key={vaultFolderKey(folder)}
                  folder={folder}
                  right={rights?.[vaultFolderKey(folder)] ?? 'not_checked'}
                  disabled={disabled}
                  onRemove={() => remove(index)}
                  onKeyDown={(event) => handleChipKeyDown(event, index)}
                />
              ))}
            </ul>
          )}
          <input
            id={INPUT_ID}
            ref={inputRef}
            type="text"
            role={available ? 'combobox' : undefined}
            aria-autocomplete={available ? 'list' : undefined}
            aria-expanded={available ? expanded : undefined}
            aria-controls={expanded ? LISTBOX_ID : undefined}
            aria-activedescendant={activeOptionId ?? undefined}
            autoComplete="off"
            autoCapitalize="off"
            spellCheck={false}
            placeholder={placeholder}
            value={draft}
            disabled={disabled}
            aria-invalid={hasError}
            aria-describedby={describedBy}
            onChange={(event) => {
              changeDraft(event.target.value.replace(/\//g, '\\'));
              setFeedback(null);
              engage();
              setOpen(true);
            }}
            onKeyDown={handleInputKeyDown}
            onFocus={engage}
            onClick={() => {
              engage();
              setOpen(true);
            }}
            onBlur={closePopup}
          />
          <button
            id="solidworks-pdm-folder-add"
            type="button"
            className="btn btn-secondary btn-sm"
            aria-label="Add folder"
            disabled={disabled}
            onClick={() => add(draft)}
          >
            Add
          </button>
        </div>
        {/* Above the list, which opens below this wrapper and would otherwise cover it. */}
        {feedback && (
          <PdmMessage id={FEEDBACK_ID} tone={feedback.tone}>
            {feedback.text}
          </PdmMessage>
        )}
        {popupVisible && parent !== null && (
          <SolidWorksFolderSuggestions
            parent={parent}
            prefix={prefix}
            suggestions={options}
            activeIndex={active ? activeIndex : -1}
            message={message}
            busy={showLoading}
            onRetry={listing.status === 'error' ? listing.retry : null}
            onChoose={choose}
            onOpen={openFolder}
          />
        )}
      </div>
      {vault ? (
        <p id={HINT_ID} className="field-help">
          Click the field or type <code>\</code> to browse folders. Press Enter to add a typed path.
          Each folder includes its subfolders.
        </p>
      ) : (
        <p id={UNAVAILABLE_ID} className="field-help">
          {'unavailableReason' in source && source.unavailableReason}
        </p>
      )}
      <p
        id="solidworks-pdm-folders-status"
        role="status"
        aria-live="polite"
        className="sw-visually-hidden"
      >
        <span key={status.revision}>{status.text}</span>
      </p>
      <p
        id="solidworks-pdm-folder-suggest-status"
        role="status"
        aria-live="polite"
        className="sw-visually-hidden"
      >
        {announcement}
      </p>
    </div>
  );
}

const RIGHTS_LABEL: Partial<Record<FolderRightsState, string>> = {
  write_ready: 'Can add and check out',
  missing_write: 'Missing add or check-out rights',
  not_found: 'Not found in the vault, or hidden from this PDM user',
};

function FolderChip({
  folder,
  right,
  disabled,
  onRemove,
  onKeyDown,
}: {
  folder: string;
  right: FolderRightsState;
  disabled?: boolean;
  onRemove: () => void;
  onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) => void;
}) {
  const label = RIGHTS_LABEL[right];
  const ready = right === 'write_ready';
  return (
    <li className="sw-chip" data-sw-folder={vaultFolderKey(folder)} data-sw-rights={right}>
      <code>{folder}</code>
      {label && (
        <span
          className="sw-chip-rights"
          data-tone={ready ? 'success' : 'warning'}
          title={ready ? undefined : label}
        >
          {ready ? (
            <CheckCircle size={12} aria-hidden="true" />
          ) : (
            <AlertTriangle size={12} aria-hidden="true" />
          )}
          <span className="sw-visually-hidden">{label}</span>
        </span>
      )}
      <button
        type="button"
        className="sw-chip-remove"
        data-sw-action="remove-folder"
        aria-label={`Remove folder ${folder}`}
        disabled={disabled}
        onClick={onRemove}
        onKeyDown={onKeyDown}
      >
        <X size={12} aria-hidden="true" />
      </button>
    </li>
  );
}
