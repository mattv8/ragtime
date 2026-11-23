import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { AlertTriangle, CheckCircle, Info, X, XCircle, type LucideIcon } from 'lucide-react';
import {
  addApprovedFolder,
  removeApprovedFolder,
  vaultFolderKey,
} from '../utils/solidworksPdmLink';
import type { FolderRightsState } from '../utils/solidworksPdmRights';
import { InfoPopover } from './shared/InfoPopover';

export interface SolidWorksApprovedFoldersProps {
  folders: string[];
  onChange: (next: string[]) => void;
  vaultRoot: string | null;
  disabled?: boolean;
  rights?: Record<string, FolderRightsState>;
  onBrowse?: () => void;
  browseDisabledReason?: string | null;
}

type Tone = 'info' | 'error';
const TONE_ICON: Record<Tone, LucideIcon> = { info: Info, error: XCircle };

const EXAMPLE_FOLDER = '\\13_Ragtime';
const INPUT_ID = 'solidworks-pdm-folder-input';
const ERROR_ID = 'solidworks-pdm-folder-input-error';
const REMOVE_BUTTON_SELECTOR = '[data-sw-action="remove-folder"]';

interface FolderFeedback {
  message: string;
  info: boolean;
}

function Message({ id, tone, children }: { id: string; tone: Tone; children: ReactNode }) {
  const Icon = TONE_ICON[tone];
  return (
    <p id={id} className="sw-message" data-tone={tone}>
      <Icon className="sw-message-icon" size={14} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}

export function approvedCountLabel(count: number): string {
  return `${count} ${count === 1 ? 'folder' : 'folders'} approved.`;
}

export function SolidWorksApprovedFolders({
  folders,
  onChange,
  vaultRoot,
  disabled,
  rights,
  onBrowse,
  browseDisabledReason,
}: SolidWorksApprovedFoldersProps) {
  const [draft, setDraft] = useState('');
  const [feedback, setFeedback] = useState<FolderFeedback | null>(null);
  const [status, setStatus] = useState('');
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const pendingRemoveIndex = useRef<number | null>(null);

  // Runs once the parent has committed the shorter list, so the focus target exists in the DOM.
  useEffect(() => {
    const index = pendingRemoveIndex.current;
    if (index === null) return;
    pendingRemoveIndex.current = null;
    const buttons = Array.from(
      listRef.current?.querySelectorAll<HTMLButtonElement>(REMOVE_BUTTON_SELECTOR) ?? [],
    );
    const target = buttons[Math.min(index, buttons.length - 1)] ?? inputRef.current;
    target?.focus();
  }, [folders]);

  const add = () => {
    const result = addApprovedFolder(folders, draft, vaultRoot);
    if (!result.ok) {
      setFeedback({ message: result.error, info: result.info });
      return;
    }
    setFeedback(null);
    setDraft('');
    onChange(result.folders);
    const notice = result.notice ? `${result.notice} ` : '';
    setStatus(`${notice}Added ${result.added}. ${approvedCountLabel(result.folders.length)}`);
    inputRef.current?.focus();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    add();
  };

  const remove = (index: number) => {
    const folder = folders[index];
    pendingRemoveIndex.current = index;
    onChange(removeApprovedFolder(folders, folder));
    setStatus(`Removed ${folder}.`);
  };

  const hasError = feedback !== null && !feedback.info;
  const describedBy = feedback ? ERROR_ID : undefined;

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
          Vault-relative paths, for example <code>{EXAMPLE_FOLDER}</code>. Includes subfolders.
          Changes anywhere else in the vault are refused.
        </InfoPopover>
      </div>
      <label htmlFor={INPUT_ID} className="sw-visually-hidden">
        Folder path
      </label>
      <div id="solidworks-pdm-folder-box" className="sw-chip-input">
        {folders.length > 0 && (
          <ul
            id="solidworks-pdm-folder-list"
            ref={listRef}
            className="sw-chip-list"
            data-testid="solidworks-pdm-folder-list"
          >
            {folders.map((folder, index) => {
              const key = vaultFolderKey(folder);
              const right = rights?.[key] ?? 'not_checked';
              const label =
                right === 'write_ready'
                  ? 'Can add and check out'
                  : right === 'missing_write'
                    ? 'Missing add or check-out rights'
                    : right === 'not_found'
                      ? 'Not found in the vault, or hidden from this PDM user'
                      : null;
              return (
                <li key={key} className="sw-chip" data-sw-folder={key} data-sw-rights={right}>
                  <code>{folder}</code>
                  {label && (
                    <span
                      className="sw-chip-rights"
                      data-tone={right === 'write_ready' ? 'success' : 'warning'}
                      title={right === 'write_ready' ? undefined : label}
                    >
                      {right === 'write_ready' ? (
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
                    onClick={() => remove(index)}
                  >
                    <X size={12} aria-hidden="true" />
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <input
          id={INPUT_ID}
          ref={inputRef}
          type="text"
          placeholder="Add a vault folder and press Enter"
          value={draft}
          disabled={disabled}
          aria-invalid={hasError}
          aria-describedby={describedBy}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button
          id="solidworks-pdm-folder-add"
          type="button"
          className="btn btn-secondary btn-sm"
          aria-label="Add folder"
          disabled={disabled}
          onClick={add}
        >
          Add
        </button>
      </div>
      {onBrowse && (
        <>
          <div className="sw-actions" data-sw-actions="browse-folders">
            <button
              id="solidworks-pdm-folder-browse"
              type="button"
              className="btn btn-secondary btn-sm"
              disabled={Boolean(browseDisabledReason)}
              aria-describedby={
                browseDisabledReason ? 'solidworks-pdm-folder-browse-help' : undefined
              }
              onClick={onBrowse}
            >
              Browse vault…
            </button>
          </div>
          {browseDisabledReason && (
            <p id="solidworks-pdm-folder-browse-help" className="field-help">
              {browseDisabledReason}
            </p>
          )}
        </>
      )}
      {feedback && (
        <Message id={ERROR_ID} tone={feedback.info ? 'info' : 'error'}>
          {feedback.message}
        </Message>
      )}
      <p
        id="solidworks-pdm-folders-status"
        role="status"
        aria-live="polite"
        className="sw-visually-hidden"
      >
        {status}
      </p>
    </div>
  );
}
