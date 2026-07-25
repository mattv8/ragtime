import { ChevronDown, X } from 'lucide-react';
import { createPortal } from 'react-dom';
import { useEffect, useId, useMemo, useRef, useState, type CSSProperties } from 'react';

import { Popover } from './Popover';
import {
  SearchFilterBar,
  normalizeSearchFilterText,
  searchFilterTextMatchesQuery,
  useLocalSearchFilterState,
} from './shared/SearchFilterBar';
import type { ToolAccessEntry, ToolAccessLevel, ToolAccessPolicy } from '@/types';

export type { ToolAccessEntry, ToolAccessLevel, ToolAccessPolicy } from '@/types';

export interface ToolAccessUserOption {
  id: string;
  username: string;
  display_name: string | null;
  is_admin: boolean;
}

export interface ToolAccessGroupOption {
  id: string;
  key: string;
  display_name: string;
  provider: string;
  member_count: number;
}

interface ToolAccessEditorProps {
  policy: ToolAccessPolicy;
  userOptions: ToolAccessUserOption[];
  groupOptions: ToolAccessGroupOption[];
  disabled?: boolean;
  globalWriteEnabled?: boolean;
  autoFocusSearch?: boolean;
  onChange: (policy: ToolAccessPolicy) => void;
}

type EntrySurfaceValue = ToolAccessLevel | null;
type EntryKind = 'users' | 'groups';

type PrincipalCandidate =
  | {
      key: string;
      kind: 'users';
      principalId: string;
      label: string;
      detail: string;
      searchText: string;
      source: ToolAccessUserOption;
    }
  | {
      key: string;
      kind: 'groups';
      principalId: string;
      label: string;
      detail: string;
      searchText: string;
      source: ToolAccessGroupOption;
    };

type PrincipalRow = {
  key: string;
  kind: EntryKind;
  principalId: string;
  label: string;
  detail: string;
  isAdmin: boolean;
  isAutomaticAdmin: boolean;
  badges: string[];
  chatAccess: EntrySurfaceValue;
  workspaceAccess: EntrySurfaceValue;
  searchText: string;
};

const DEFAULT_OPTIONS: ToolAccessLevel[] = ['deny', 'read', 'read_write'];
const ENTRY_OPTIONS: Array<EntrySurfaceValue | 'inherit'> = [
  'inherit',
  'deny',
  'read',
  'read_write',
];
const ADMIN_LOCKED_EXPLANATION = 'Admins always have full access.';

function formatLevelLabel(level: EntrySurfaceValue | 'inherit'): string {
  if (level === 'inherit') return 'Inherit';
  if (level === 'read_write') return 'Read+Write';
  if (level === 'read') return 'Read';
  return 'Deny';
}

function formatUserLabel(user: ToolAccessUserOption): string {
  return user.display_name?.trim() || user.username;
}

function formatUserDetail(user: ToolAccessUserOption): string {
  return `@${user.username.replace(/^@/, '')}`;
}

function formatUserEntryDetail(entry: ToolAccessEntry, fallbackUsername?: string): string {
  const rawDetail = entry.principal_detail?.trim() || fallbackUsername || entry.principal_id;
  return `@${rawDetail.replace(/^@/, '')}`;
}

function formatGroupProvider(provider: string): string {
  if (provider === 'local_managed') return 'Internal';
  if (provider === 'ldap') return 'LDAP';
  return provider;
}

function formatGroupDetail(group: ToolAccessGroupOption): string {
  return `${formatGroupProvider(group.provider)} · ${group.member_count} member${group.member_count === 1 ? '' : 's'}`;
}

function formatPrincipalLabel(entry: ToolAccessEntry): string {
  return entry.display_name?.trim() || entry.principal_detail?.trim() || entry.principal_id;
}

function getEntryBadges(entry: ToolAccessEntry, group?: ToolAccessGroupOption): string[] {
  const badges: string[] = [];
  if (entry.orphaned) {
    badges.push('Orphaned');
  }
  if (group && group.member_count === 0) {
    badges.push('No members');
  }
  return badges;
}

function IdentityBadges({ kind, isAdmin = false }: { kind: EntryKind; isAdmin?: boolean }) {
  if (kind !== 'groups' && !isAdmin) {
    return null;
  }

  return (
    <>
      {kind === 'groups' ? <span className="tool-access-kind-badge">Group</span> : null}
      {isAdmin ? (
        <span className="tool-access-kind-badge tool-access-admin-badge">Admin</span>
      ) : null}
    </>
  );
}

function GroupCandidateMetadata({ group }: { group: ToolAccessGroupOption }) {
  const memberLabel = `${group.member_count} member${group.member_count === 1 ? '' : 's'}`;

  return (
    <span className="tool-access-group-candidate-meta">
      {group.provider === 'ldap' ? (
        <span className="tool-access-group-provider-badge">LDAP</span>
      ) : null}
      <span className="tool-access-group-member-count">{memberLabel}</span>
    </span>
  );
}

function AccessSelect({
  ariaLabel,
  value,
  options,
  disabled,
  globalWriteEnabled,
  enforceDefaultWriteCeiling = false,
  lockedExplanation,
  onChange,
}: {
  ariaLabel: string;
  value: EntrySurfaceValue;
  options: Array<EntrySurfaceValue | 'inherit'>;
  disabled: boolean;
  globalWriteEnabled: boolean;
  enforceDefaultWriteCeiling?: boolean;
  lockedExplanation?: string;
  onChange: (value: EntrySurfaceValue) => void;
}) {
  const listboxId = useId();
  const wrapRef = useRef<HTMLDivElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const [open, setOpen] = useState(false);
  const [panelStyle, setPanelStyle] = useState<CSSProperties | null>(null);
  const selectedValue = value ?? 'inherit';
  const isLocked = Boolean(lockedExplanation);
  const blocksReadWrite = enforceDefaultWriteCeiling && !globalWriteEnabled;
  const effectiveSelectedValue =
    !isLocked && blocksReadWrite && selectedValue === 'read_write' ? 'read' : selectedValue;
  const isDisabled = disabled && !isLocked;

  useEffect(() => {
    if (!open || isLocked) {
      setPanelStyle(null);
      return;
    }

    const updatePosition = () => {
      if (!wrapRef.current) {
        return;
      }
      const rect = wrapRef.current.getBoundingClientRect();
      const viewportPadding = 8;
      const width = Math.max(rect.width, 140);
      const spaceAbove = Math.max(0, rect.top - viewportPadding);
      const spaceBelow = Math.max(0, window.innerHeight - rect.bottom - viewportPadding);
      const opensUpward = spaceAbove > spaceBelow;
      const position: CSSProperties = {
        position: 'fixed',
        left: Math.min(rect.left, window.innerWidth - viewportPadding - width),
        width,
        maxHeight: opensUpward ? spaceAbove : spaceBelow,
        zIndex: 9100,
      };

      if (opensUpward) {
        position.bottom = window.innerHeight - rect.top + 4;
      } else {
        position.top = rect.bottom + 4;
      }

      setPanelStyle(position);
    };

    const handlePointerDown = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!wrapRef.current?.contains(target) && !panelRef.current?.contains(target)) {
        setOpen(false);
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };

    updatePosition();
    window.addEventListener('resize', updatePosition);
    window.addEventListener('scroll', updatePosition, true);
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('resize', updatePosition);
      window.removeEventListener('scroll', updatePosition, true);
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isLocked, open]);

  const selectOption = (nextValue: EntrySurfaceValue | 'inherit', isDisabledOption: boolean) => {
    if (isDisabledOption) {
      return;
    }
    onChange(nextValue === 'inherit' ? null : nextValue);
    setOpen(false);
    triggerRef.current?.focus();
  };

  const trigger = (
    <button
      ref={triggerRef}
      type="button"
      className={`tool-access-select${isLocked ? ' is-locked' : ''}`}
      role="combobox"
      aria-label={ariaLabel}
      aria-expanded={open}
      aria-controls={open ? listboxId : undefined}
      aria-haspopup="listbox"
      aria-disabled={isLocked ? 'true' : undefined}
      value={effectiveSelectedValue}
      disabled={isDisabled}
      onClick={() => {
        if (disabled || isLocked) {
          return;
        }
        setOpen((current) => !current);
      }}
    >
      <span className="tool-access-select-label">{formatLevelLabel(effectiveSelectedValue)}</span>
      <ChevronDown
        size={14}
        aria-hidden="true"
        className={`tool-access-select-chevron${open ? ' is-open' : ''}`}
      />
    </button>
  );

  return (
    <div className="tool-access-select-wrap" ref={wrapRef}>
      {isLocked ? <Popover content={lockedExplanation}>{trigger}</Popover> : trigger}
      {open &&
        panelStyle &&
        createPortal(
          <div
            ref={panelRef}
            id={listboxId}
            className="tool-access-select-dropdown"
            role="listbox"
            aria-label={`${ariaLabel} options`}
            style={panelStyle}
          >
            {options.map((option) => {
              const optionValue = option ?? 'inherit';
              const optionDisabled = blocksReadWrite && option === 'read_write';
              const optionButton = (
                <button
                  key={optionValue}
                  type="button"
                  className={`tool-access-select-option${effectiveSelectedValue === optionValue ? ' is-selected' : ''}${optionDisabled ? ' is-disabled' : ''}`}
                  role="option"
                  aria-selected={effectiveSelectedValue === optionValue}
                  aria-disabled={optionDisabled ? 'true' : undefined}
                  tabIndex={0}
                  onClick={() => selectOption(option, optionDisabled)}
                >
                  {formatLevelLabel(option)}
                </button>
              );

              if (optionDisabled) {
                return (
                  <Popover
                    key={optionValue}
                    content="Setting Default to Read+Write would grant everyone write access. Enable Write Access on this tool first. Specific users and groups can still be granted Read+Write below."
                    zIndexAboveTrigger
                  >
                    {optionButton}
                  </Popover>
                );
              }

              return optionButton;
            })}
          </div>,
          document.body,
        )}
    </div>
  );
}

export function ToolAccessEditor({
  policy,
  userOptions,
  groupOptions,
  disabled = false,
  globalWriteEnabled = true,
  autoFocusSearch = false,
  onChange,
}: ToolAccessEditorProps) {
  const searchState = useLocalSearchFilterState();
  const searchInputRef = useRef<HTMLInputElement>(null);
  const browseTriggerRef = useRef<HTMLButtonElement>(null);
  const candidateListRef = useRef<HTMLDivElement>(null);
  const candidateListId = useId();
  const [isBrowseOpen, setIsBrowseOpen] = useState(false);

  useEffect(() => {
    if (autoFocusSearch) {
      searchInputRef.current?.focus();
    }
  }, [autoFocusSearch]);

  const usersById = useMemo(
    () => new Map(userOptions.map((user) => [user.id, user])),
    [userOptions],
  );
  const groupsById = useMemo(
    () => new Map(groupOptions.map((group) => [group.id, group])),
    [groupOptions],
  );

  const selectedUserIds = useMemo(
    () => new Set(policy.users.map((entry) => entry.principal_id)),
    [policy.users],
  );
  const selectedGroupIds = useMemo(
    () => new Set(policy.groups.map((entry) => entry.principal_id)),
    [policy.groups],
  );
  const canBrowse =
    !disabled &&
    (userOptions.some((user) => !selectedUserIds.has(user.id)) ||
      groupOptions.some((group) => !selectedGroupIds.has(group.id)));

  const principalRows = useMemo<PrincipalRow[]>(
    () => [
      ...policy.users.map((entry) => {
        const user = usersById.get(entry.principal_id);
        const label = formatPrincipalLabel(entry);
        const detail = formatUserEntryDetail(entry, user?.username);
        return {
          key: `user-${entry.principal_id}`,
          kind: 'users' as const,
          principalId: entry.principal_id,
          label,
          detail,
          isAdmin: user?.is_admin ?? false,
          isAutomaticAdmin: false,
          badges: [],
          chatAccess: user?.is_admin ? 'read_write' : entry.chat_access,
          workspaceAccess: user?.is_admin ? 'read_write' : entry.workspace_access,
          searchText: `${label} ${detail} ${user?.username || ''}`,
        };
      }),
      ...userOptions
        .filter((user) => user.is_admin && !selectedUserIds.has(user.id))
        .map((user) => {
          const label = formatUserLabel(user);
          const detail = formatUserDetail(user);
          return {
            key: `user-${user.id}`,
            kind: 'users' as const,
            principalId: user.id,
            label,
            detail,
            isAdmin: true,
            isAutomaticAdmin: true,
            badges: [],
            chatAccess: 'read_write' as const,
            workspaceAccess: 'read_write' as const,
            searchText: `${label} ${detail} ${user.username}`,
          };
        }),
      ...policy.groups.map((entry) => {
        const group = groupsById.get(entry.principal_id);
        const label = formatPrincipalLabel(entry);
        const detail =
          entry.principal_detail?.trim() || (group ? formatGroupDetail(group) : entry.principal_id);
        const badges = getEntryBadges(entry, group);
        return {
          key: `group-${entry.principal_id}`,
          kind: 'groups' as const,
          principalId: entry.principal_id,
          label,
          detail,
          isAdmin: false,
          isAutomaticAdmin: false,
          badges,
          chatAccess: entry.chat_access,
          workspaceAccess: entry.workspace_access,
          searchText: `${label} ${detail} ${group?.key || ''} ${group ? formatGroupProvider(group.provider) : ''} ${badges.join(' ')}`,
        };
      }),
    ],
    [groupsById, policy.groups, policy.users, selectedUserIds, userOptions, usersById],
  );

  const rowFilterQueries = useMemo(() => {
    const liveInput = normalizeSearchFilterText(searchState.input);
    return [
      ...searchState.tags.map(normalizeSearchFilterText),
      ...(liveInput ? [liveInput] : []),
    ].filter(Boolean);
  }, [searchState.input, searchState.tags]);

  const filteredRows = useMemo(
    () =>
      principalRows.filter((row) => searchFilterTextMatchesQuery(row.searchText, rowFilterQueries)),
    [principalRows, rowFilterQueries],
  );

  const candidateQueries = useMemo(() => {
    const normalized = normalizeSearchFilterText(searchState.input);
    return normalized ? [normalized] : [];
  }, [searchState.input]);

  const isBrowseMode = isBrowseOpen && candidateQueries.length === 0;

  useEffect(() => {
    if (!isBrowseMode) {
      return;
    }

    const handlePointerDown = (event: MouseEvent | PointerEvent) => {
      const target = event.target as Node;
      if (
        candidateListRef.current?.contains(target) ||
        browseTriggerRef.current?.contains(target)
      ) {
        return;
      }

      setIsBrowseOpen(false);
    };

    document.addEventListener('mousedown', handlePointerDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
    };
  }, [isBrowseMode]);

  const addableCandidates = useMemo<PrincipalCandidate[]>(() => {
    if (candidateQueries.length === 0 && !isBrowseMode) {
      return [];
    }

    const users: PrincipalCandidate[] = userOptions
      .filter((user) => !selectedUserIds.has(user.id))
      .map((user) => ({
        key: `candidate-user-${user.id}`,
        kind: 'users' as const,
        principalId: user.id,
        label: formatUserLabel(user),
        detail: formatUserDetail(user),
        searchText: `${formatUserLabel(user)} ${user.username}`,
        source: user,
      }))
      .filter((candidate) => searchFilterTextMatchesQuery(candidate.searchText, candidateQueries));

    const groups: PrincipalCandidate[] = groupOptions
      .filter((group) => !selectedGroupIds.has(group.id))
      .map((group) => ({
        key: `candidate-group-${group.id}`,
        kind: 'groups' as const,
        principalId: group.id,
        label: group.display_name,
        detail: formatGroupDetail(group),
        searchText: `${group.display_name} ${group.key} ${formatGroupProvider(group.provider)} ${formatGroupDetail(group)}`,
        source: group,
      }))
      .filter((candidate) => searchFilterTextMatchesQuery(candidate.searchText, candidateQueries));

    return [...users, ...groups];
  }, [
    candidateQueries,
    groupOptions,
    isBrowseMode,
    selectedGroupIds,
    selectedUserIds,
    userOptions,
  ]);

  const hasExplicitDeny = useMemo(
    () =>
      [...policy.users, ...policy.groups].some(
        (entry) => entry.chat_access === 'deny' || entry.workspace_access === 'deny',
      ),
    [policy.groups, policy.users],
  );

  const updateEntrySurface = (
    kind: EntryKind,
    principalId: string,
    surface: 'chat_access' | 'workspace_access',
    nextValue: EntrySurfaceValue,
  ) => {
    const nextEntries = policy[kind].flatMap((entry) => {
      if (entry.principal_id !== principalId) {
        return [entry];
      }

      const updatedEntry = { ...entry, [surface]: nextValue };
      if (updatedEntry.chat_access == null && updatedEntry.workspace_access == null) {
        return [];
      }

      return [updatedEntry];
    });

    onChange({
      ...policy,
      [kind]: nextEntries,
    });
  };

  const addUser = (user: ToolAccessUserOption) => {
    onChange({
      ...policy,
      users: [
        ...policy.users,
        {
          principal_id: user.id,
          chat_access: 'read',
          workspace_access: 'read',
          display_name: formatUserLabel(user),
          principal_detail: user.username,
          orphaned: false,
        },
      ],
    });
    setIsBrowseOpen(false);
    searchState.clear();
  };

  const addGroup = (group: ToolAccessGroupOption) => {
    onChange({
      ...policy,
      groups: [
        ...policy.groups,
        {
          principal_id: group.id,
          chat_access: 'read',
          workspace_access: 'read',
          display_name: group.display_name,
          principal_detail: formatGroupDetail(group),
          orphaned: false,
        },
      ],
    });
    setIsBrowseOpen(false);
    searchState.clear();
  };

  const removeEntry = (kind: EntryKind, principalId: string) => {
    onChange({
      ...policy,
      [kind]: policy[kind].filter((entry) => entry.principal_id !== principalId),
    });
  };

  return (
    <div className="tool-access-editor">
      {hasExplicitDeny && (
        <div className="tool-access-callout tool-access-callout-warning" role="alert">
          <div>Explicit deny overrides any other grant for that surface.</div>
        </div>
      )}

      <div className="tool-access-search">
        <SearchFilterBar
          state={searchState}
          inputRef={searchInputRef}
          placeholder="Search or add users and groups..."
          ariaLabel="Search or add users and groups"
          className="tool-access-search-bar"
          disabled={disabled}
        />
        <div className="tool-access-browse-hint">
          Type to search, or{' '}
          <button
            ref={browseTriggerRef}
            type="button"
            className="btn-link"
            disabled={!canBrowse}
            aria-expanded={canBrowse && isBrowseMode ? 'true' : 'false'}
            aria-controls={canBrowse ? candidateListId : undefined}
            onClick={() => setIsBrowseOpen((open) => !open)}
          >
            browse all
          </button>{' '}
          to add users and groups.
        </div>
        {!disabled && (searchState.input.trim() || isBrowseMode) && addableCandidates.length > 0 ? (
          <div
            id={candidateListId}
            ref={candidateListRef}
            className="tool-access-search-dropdown"
            role="listbox"
            aria-label="Add users and groups"
          >
            {addableCandidates.map((candidate) => (
              <button
                key={candidate.key}
                type="button"
                className="tool-access-search-option"
                role="option"
                aria-label={`Add ${candidate.label}`}
                disabled={candidate.kind === 'users' && candidate.source.is_admin}
                title={
                  candidate.kind === 'users' && candidate.source.is_admin
                    ? 'Admins already have full access.'
                    : undefined
                }
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  if (candidate.kind === 'users') {
                    addUser(candidate.source);
                    return;
                  }
                  addGroup(candidate.source);
                }}
              >
                <span className="tool-access-search-option-title">
                  <span className="tool-access-identity">
                    <span>{candidate.label}</span>
                    <IdentityBadges
                      kind={candidate.kind}
                      isAdmin={candidate.kind === 'users' ? candidate.source.is_admin : false}
                    />
                  </span>
                </span>
                <span className="tool-access-search-option-detail">
                  {candidate.kind === 'groups' ? (
                    <GroupCandidateMetadata group={candidate.source} />
                  ) : (
                    candidate.detail
                  )}
                </span>
              </button>
            ))}
          </div>
        ) : null}
      </div>

      <div className="tool-access-table" role="table" aria-label="Tool access policy editor">
        <div className="tool-access-table-head" role="rowgroup">
          <div className="tool-access-table-row tool-access-table-row-header" role="row">
            <div className="tool-access-table-cell tool-access-principal-cell" role="columnheader">
              Who
            </div>
            <div className="tool-access-table-cell" role="columnheader">
              Chat
            </div>
            <div className="tool-access-table-cell" role="columnheader">
              Workspace
            </div>
            <div className="tool-access-table-cell tool-access-remove-cell" role="columnheader" />
          </div>
        </div>

        <div className="tool-access-table-body" role="rowgroup">
          <div className="tool-access-table-row tool-access-default-row" role="row">
            <div className="tool-access-table-cell tool-access-principal-cell" role="cell">
              <div className="tool-access-principal">
                <span className="tool-access-principal-name">Default</span>
                <span className="tool-access-principal-detail">When no override matches</span>
              </div>
            </div>
            <div className="tool-access-table-cell" role="cell">
              <span className="tool-access-mobile-label">Chat</span>
              <AccessSelect
                ariaLabel="Default chat access"
                value={policy.default_chat_access}
                options={DEFAULT_OPTIONS}
                disabled={disabled}
                globalWriteEnabled={globalWriteEnabled}
                enforceDefaultWriteCeiling
                onChange={(value) => {
                  if (value) {
                    onChange({
                      ...policy,
                      default_chat_access: value,
                    });
                  }
                }}
              />
            </div>
            <div className="tool-access-table-cell" role="cell">
              <span className="tool-access-mobile-label">Workspace</span>
              <AccessSelect
                ariaLabel="Default workspace access"
                value={policy.default_workspace_access}
                options={DEFAULT_OPTIONS}
                disabled={disabled}
                globalWriteEnabled={globalWriteEnabled}
                enforceDefaultWriteCeiling
                onChange={(value) => {
                  if (value) {
                    onChange({
                      ...policy,
                      default_workspace_access: value,
                    });
                  }
                }}
              />
            </div>
            <div className="tool-access-table-cell tool-access-remove-cell" role="cell">
              <button
                type="button"
                className="btn btn-secondary btn-sm btn-icon tool-access-default-lock-btn"
                disabled
                aria-label="Default access cannot be removed"
                title="The Default row sets fallback access for everyone and cannot be removed."
              >
                <X size={14} aria-hidden="true" />
              </button>
            </div>
          </div>

          {filteredRows.map((row) => (
            <div key={row.key} className="tool-access-table-row" role="row">
              <div className="tool-access-table-cell tool-access-principal-cell" role="cell">
                <div className="tool-access-principal">
                  <span className="tool-access-principal-name">
                    <span className="tool-access-identity">
                      <span>{row.label}</span>
                      <IdentityBadges kind={row.kind} isAdmin={row.isAdmin} />
                    </span>
                  </span>
                  <div className="tool-access-principal-meta">
                    <span className="tool-access-principal-detail">{row.detail}</span>
                    {row.badges.length > 0 ? (
                      <span className="tool-access-badges" aria-label={`${row.label} status`}>
                        {row.badges.map((badge) => (
                          <span key={badge} className="tool-access-badge">
                            {badge}
                          </span>
                        ))}
                      </span>
                    ) : null}
                  </div>
                </div>
              </div>
              <div className="tool-access-table-cell" role="cell">
                <span className="tool-access-mobile-label">Chat</span>
                <AccessSelect
                  ariaLabel={`${row.label} chat access`}
                  value={row.chatAccess}
                  options={ENTRY_OPTIONS}
                  disabled={disabled}
                  globalWriteEnabled={globalWriteEnabled}
                  lockedExplanation={row.isAdmin ? ADMIN_LOCKED_EXPLANATION : undefined}
                  onChange={(value) =>
                    updateEntrySurface(row.kind, row.principalId, 'chat_access', value)
                  }
                />
              </div>
              <div className="tool-access-table-cell" role="cell">
                <span className="tool-access-mobile-label">Workspace</span>
                <AccessSelect
                  ariaLabel={`${row.label} workspace access`}
                  value={row.workspaceAccess}
                  options={ENTRY_OPTIONS}
                  disabled={disabled}
                  globalWriteEnabled={globalWriteEnabled}
                  lockedExplanation={row.isAdmin ? ADMIN_LOCKED_EXPLANATION : undefined}
                  onChange={(value) =>
                    updateEntrySurface(row.kind, row.principalId, 'workspace_access', value)
                  }
                />
              </div>
              <div className="tool-access-table-cell tool-access-remove-cell" role="cell">
                <button
                  type="button"
                  className="btn btn-secondary btn-sm btn-icon"
                  disabled={disabled || row.isAutomaticAdmin}
                  aria-label={
                    row.isAutomaticAdmin
                      ? `${row.label} access is automatic`
                      : `Remove ${row.label}`
                  }
                  onClick={() => {
                    if (row.isAutomaticAdmin) {
                      return;
                    }
                    removeEntry(row.kind, row.principalId);
                  }}
                >
                  <X size={14} aria-hidden="true" />
                </button>
              </div>
            </div>
          ))}

          {principalRows.length === 0 ? (
            <div className="tool-access-table-empty" role="row">
              <div className="tool-access-table-empty-cell" role="cell">
                No user or group overrides. Search above to add one.
              </div>
            </div>
          ) : null}

          {principalRows.length > 0 && rowFilterQueries.length > 0 && filteredRows.length === 0 ? (
            <div className="tool-access-table-empty" role="row">
              <div className="tool-access-table-empty-cell" role="cell">
                No matching overrides.
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
