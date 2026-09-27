import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '@/api';
import {
  updateContentProtectionConfigSlice,
  userOverrideMode,
  withUserOverride,
  type ContentProtectionConfig,
  type ContentProtectionOverrideMode,
} from '@/api/contentProtection';
import type {
  AuthGroup,
  Conversation,
  ConversationSummary,
  User,
  UserSpaceWorkspace,
} from '@/types';
import { CheckboxDropdown } from '../shared/CheckboxDropdown';
import { ModalTabs, type ModalTab } from '../shared/ModalTabs';
import { UserSecurityTab } from './UserSecurityTab';
import { UserResourcesTab } from './UserResourcesTab';

type Tab = 'account' | 'policies' | 'security' | 'resources';
type DirtySection =
  | 'profile'
  | 'role'
  | 'groups'
  | 'generation'
  | 'protection'
  | 'security'
  | 'delete';
const manualIds = (user: User) => user.manual_group_ids ?? user.local_group_ids ?? [];
const profileProvider = (user: User) => user.auth_provider === 'local_managed';
const policyValue = (value: boolean | null | undefined) =>
  value == null ? 'inherit' : value ? 'enabled' : 'disabled';

interface Props {
  user: User;
  currentUser: User | null;
  authGroups: AuthGroup[];
  users: User[];
  workspaces: UserSpaceWorkspace[];
  chats: ConversationSummary[];
  chatsLoaded: boolean;
  chatsLoading: boolean;
  chatsError: string | null;
  workspaceStateById: Record<string, import('@/types').WorkspaceConversationStateSummaryItem>;
  workspaceLastMessageAtById: Record<string, string | null>;
  workspaceLastConversationById: Record<string, ConversationSummary | null>;
  workspaceMetaLoading: boolean;
  storageByWorkspaceId: Record<string, number>;
  storageFailuresByWorkspaceId: Record<string, string>;
  storageLoading: boolean;
  deletingWorkspaceIds: ReadonlySet<string>;
  contentProtectionConfig: ContentProtectionConfig | null;
  contentProtectionLoadFailed: boolean;
  onRetryContentProtectionConfig: () => Promise<void>;
  onClose: () => void;
  onUserUpdated: (user: User) => void;
  onDelete: (id: string) => Promise<void>;
  onWorkspaceDelete: (id: string) => Promise<void>;
  onWorkspaceTransfer: (id: string, owner: string) => Promise<void>;
  onOpenWorkspace: (id: string) => void;
  onOpenChat?: (id: string) => void;
  onDeleteChat: (id: string) => Promise<void>;
  onCancelChat: (id: string, task: string) => Promise<void>;
  onResourcesOpen: () => void;
  onComputeStorage: () => void;
  formatBytes: (bytes: number) => string;
  formatDateTime: (value: string | null | undefined) => string;
  getConversationContextMeta: (
    conversation: Conversation | ConversationSummary | null | undefined,
  ) => string;
  onGenerationPolicyUpdated?: (user: User) => void | Promise<void>;
  onContentProtectionConfigChange: (config: ContentProtectionConfig) => void;
}

export function UserManagementModal(props: Props) {
  const { user, currentUser } = props;
  const isSelf = currentUser?.id === user.id;
  const [tab, setTab] = useState<Tab>('account');
  const [dirty, setDirty] = useState<Partial<Record<DirtySection, boolean>>>({});
  const [confirmClose, setConfirmClose] = useState(false);
  const [deleteConfirm, setDeleteConfirm] = useState(false);
  const [busy, setBusy] = useState<Partial<Record<DirtySection, boolean>>>({});
  const [errors, setErrors] = useState<Partial<Record<DirtySection, string>>>({});
  const [notices, setNotices] = useState<Partial<Record<DirtySection, string>>>({});
  const [role, setRole] = useState(user.role);
  const [groups, setGroups] = useState(manualIds(user));
  const [name, setName] = useState(user.display_name ?? '');
  const [email, setEmail] = useState(user.email ?? '');
  const [chat, setChat] = useState(policyValue(user.chat_enabled));
  const [generation, setGeneration] = useState(policyValue(user.userspace_generation_enabled));
  const [protection, setProtection] = useState<ContentProtectionOverrideMode>(() =>
    props.contentProtectionConfig
      ? userOverrideMode(props.contentProtectionConfig, user.id)
      : 'inherit',
  );
  const opener = useRef<HTMLElement | null>(null);
  const confirmationOrigin = useRef<HTMLElement | null>(null);
  const deleteCancel = useRef<HTMLButtonElement | null>(null);
  const deleteTrigger = useRef<HTMLButtonElement | null>(null);
  const discardCancel = useRef<HTMLButtonElement | null>(null);
  const dialog = useRef<HTMLDivElement>(null);
  const hasDirty = Object.values(dirty).some(Boolean);

  const markDirty = (section: DirtySection, value = true) =>
    setDirty((current) => ({ ...current, [section]: value }));
  const sectionError = (section: DirtySection, error: unknown) => {
    setErrors((current) => ({
      ...current,
      [section]:
        error instanceof Error ? error.message : 'Save failed. Your changes are still here.',
    }));
  };
  const run = async (section: DirtySection, task: () => Promise<void>) => {
    setBusy((current) => ({ ...current, [section]: true }));
    setErrors((current) => ({ ...current, [section]: undefined }));
    try {
      await task();
      markDirty(section, false);
    } catch (error) {
      sectionError(section, error);
    } finally {
      setBusy((current) => ({ ...current, [section]: false }));
    }
  };
  const refreshUser = async (): Promise<User> => api.getUser(user.id);

  useEffect(() => {
    opener.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialog.current?.focus();
    return () => opener.current?.focus();
  }, []);
  useEffect(() => {
    if (!deleteConfirm && !confirmClose) return;
    const target = deleteConfirm ? deleteCancel.current : discardCancel.current;
    requestAnimationFrame(() => target?.focus());
  }, [confirmClose, deleteConfirm]);
  useEffect(() => {
    if (!props.contentProtectionConfig || dirty.protection) return;
    setProtection(userOverrideMode(props.contentProtectionConfig, user.id));
  }, [dirty.protection, props.contentProtectionConfig, user.id]);

  const hasPendingMutation = Object.values(busy).some(Boolean);
  const requestClose = (origin?: HTMLElement) => {
    if (hasPendingMutation) return;
    if (hasDirty) {
      confirmationOrigin.current =
        origin ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
      setConfirmClose(true);
    } else props.onClose();
  };
  const discardAndClose = () => {
    if (!hasPendingMutation) props.onClose();
  };
  const cancelConfirmation = (kind: 'delete' | 'close') => {
    if (kind === 'delete') setDeleteConfirm(false);
    else setConfirmClose(false);
    requestAnimationFrame(() =>
      (kind === 'delete' ? deleteTrigger.current : confirmationOrigin.current)?.focus(),
    );
  };
  const onConfirmationKeyDown = (
    event: React.KeyboardEvent<HTMLDivElement>,
    kind: 'delete' | 'close',
  ) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      cancelConfirmation(kind);
      return;
    }
    if (event.key !== 'Tab') return;
    event.stopPropagation();
    const focusable = Array.from(
      event.currentTarget.querySelectorAll<HTMLElement>(
        'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    );
    if (!focusable.length) return;
    const index = focusable.indexOf(document.activeElement as HTMLElement);
    if (event.shiftKey && index <= 0) {
      event.preventDefault();
      focusable[focusable.length - 1]?.focus();
    } else if (!event.shiftKey && index === focusable.length - 1) {
      event.preventDefault();
      focusable[0]?.focus();
    }
  };
  const openDeleteConfirmation = (origin?: HTMLElement) => {
    confirmationOrigin.current =
      origin ?? (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    setDeleteConfirm(true);
  };
  const onDialogKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
      requestClose();
      return;
    }
    if (event.key === 'Tab') {
      const focusable = Array.from(
        dialog.current?.querySelectorAll<HTMLElement>(
          'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        ) ?? [],
      ).filter(
        (element) => !element.hidden && element.offsetParent !== null && element.tabIndex >= 0,
      );
      if (!focusable.length) return;
      const index = focusable.indexOf(document.activeElement as HTMLElement);
      if (event.shiftKey && (index <= 0 || document.activeElement === dialog.current)) {
        event.preventDefault();
        focusable[focusable.length - 1]?.focus();
      } else if (!event.shiftKey && index === focusable.length - 1) {
        event.preventDefault();
        focusable[0]?.focus();
      }
    }
  };
  const groupOptions = useMemo(
    () =>
      props.authGroups.map((group) => ({
        id: group.id,
        label: group.display_name,
        badge: group.provider === 'ldap' ? 'LDAP' : undefined,
        disabled: group.provider === 'ldap',
        checked: (user.ldap_group_ids ?? []).includes(group.id),
      })),
    [props.authGroups, user.ldap_group_ids],
  );
  const effective = (raw: boolean | null | undefined, value: boolean | undefined) =>
    raw == null
      ? `Inherited. Server effective: ${value === undefined ? 'unknown' : value ? 'enabled' : 'disabled'}.`
      : `Override configured. Server effective: ${value === undefined ? 'unknown' : value ? 'enabled' : 'disabled'}.`;
  const renderError = (section: DirtySection) =>
    errors[section] ? (
      <p className="form-error" role="alert">
        {errors[section]}
      </p>
    ) : null;

  const modalTabs: ModalTab[] = [
    {
      id: 'account',
      label: 'Account & access',
      content: (
        <div data-user-management-section="account">
          {profileProvider(user) ? (
            <>
              <div className="form-group">
                <label htmlFor={`user-name-${user.id}`}>Display name</label>
                <input
                  id={`user-name-${user.id}`}
                  value={name}
                  onChange={(event) => {
                    setName(event.target.value);
                    markDirty('profile');
                  }}
                />
              </div>
              <div className="form-group">
                <label htmlFor={`user-email-${user.id}`}>Email</label>
                <input
                  id={`user-email-${user.id}`}
                  value={email}
                  onChange={(event) => {
                    setEmail(event.target.value);
                    markDirty('profile');
                  }}
                />
              </div>
              {renderError('profile')}
              <div className="user-management-actions">
                <button
                  className="btn btn-primary"
                  disabled={busy.profile || !dirty.profile}
                  onClick={() =>
                    void run('profile', async () => {
                      const updated = await api.updateLocalUser(user.id, {
                        display_name: name || null,
                        email: email || null,
                      });
                      props.onUserUpdated(updated);
                      setName(updated.display_name ?? updated.username);
                      setEmail(updated.email ?? '');
                    })
                  }
                >
                  Save profile
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={!dirty.profile}
                  onClick={() => {
                    setName(user.display_name ?? '');
                    setEmail(user.email ?? '');
                    markDirty('profile', false);
                  }}
                >
                  Discard
                </button>
              </div>
            </>
          ) : (
            <p className="field-help">
              Profile fields are managed by this account&apos;s identity provider.
            </p>
          )}
          {!isSelf && (
            <>
              <div className="form-group">
                <label htmlFor={`user-role-${user.id}`}>Role</label>
                <select
                  id={`user-role-${user.id}`}
                  value={role}
                  onChange={(event) => {
                    setRole(event.target.value as User['role']);
                    markDirty('role');
                  }}
                >
                  <option value="user">user</option>
                  <option value="admin">admin</option>
                </select>
                {user.auth_provider !== 'ldap' && (
                  <p className="field-help">Role reset is supported only for LDAP users.</p>
                )}
              </div>
              {renderError('role')}
              <div className="user-management-actions">
                <button
                  className="btn btn-primary"
                  disabled={busy.role || !dirty.role}
                  onClick={() =>
                    void run('role', async () => {
                      await api.updateUserRole(user.id, role);
                      props.onUserUpdated(await refreshUser());
                    })
                  }
                >
                  Save role
                </button>
                {user.auth_provider === 'ldap' && user.role_manually_set && (
                  <button
                    className="btn btn-secondary"
                    disabled={busy.role}
                    onClick={() =>
                      void run('role', async () => {
                        await api.resetUserRoleOverride(user.id);
                        props.onUserUpdated(await refreshUser());
                      })
                    }
                  >
                    Reset LDAP role override
                  </button>
                )}
                <button
                  className="btn btn-secondary"
                  disabled={!dirty.role}
                  onClick={() => {
                    setRole(user.role);
                    markDirty('role', false);
                  }}
                >
                  Discard
                </button>
              </div>
              <div className="form-group">
                <label>Manual groups</label>
                <CheckboxDropdown
                  options={groupOptions}
                  selectedIds={groups}
                  onChange={(ids) => {
                    setGroups(ids);
                    markDirty('groups');
                  }}
                  placeholder="No manual groups"
                />
                <p className="field-help">
                  Directory groups are read-only. Manual group changes are saved separately.
                </p>
              </div>
              {renderError('groups')}
              <div className="user-management-actions">
                <button
                  className="btn btn-primary"
                  disabled={busy.groups || !dirty.groups}
                  onClick={() =>
                    void run('groups', async () => {
                      props.onUserUpdated(await api.setUserGroups(user.id, { group_ids: groups }));
                    })
                  }
                >
                  Save groups
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={!dirty.groups}
                  onClick={() => {
                    setGroups(manualIds(user));
                    markDirty('groups', false);
                  }}
                >
                  Discard
                </button>
              </div>
              <div className="user-management-danger">
                <h4>Delete user</h4>
                <p>
                  Deleting this account permanently deletes owned workspaces and their database
                  rows. Chats are retained but their user ownership is cleared. Transfer or delete
                  resources first if they must be preserved.
                </p>
                {deleteConfirm ? (
                  <div
                    role="alertdialog"
                    aria-label="Confirm delete user"
                    onKeyDown={(event) => onConfirmationKeyDown(event, 'delete')}
                  >
                    <p>Delete {user.username}? This cannot be undone.</p>
                    <button
                      className="btn btn-danger"
                      disabled={busy.delete}
                      onClick={() =>
                        void run('delete', async () => {
                          await props.onDelete(user.id);
                          props.onClose();
                        })
                      }
                    >
                      Delete user
                    </button>
                    {renderError('delete')}
                    <button
                      className="btn btn-secondary"
                      disabled={busy.delete}
                      ref={deleteCancel}
                      onClick={() => cancelConfirmation('delete')}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    className="btn btn-danger"
                    ref={deleteTrigger}
                    onClick={(event) => openDeleteConfirmation(event.currentTarget)}
                  >
                    Delete user
                  </button>
                )}
              </div>
            </>
          )}
        </div>
      ),
    },
    {
      id: 'policies',
      label: 'Policies',
      content: (
        <div data-user-management-section="policies">
          <div className="form-group">
            <label htmlFor={`chat-policy-${user.id}`}>Chat generation</label>
            <select
              id={`chat-policy-${user.id}`}
              value={chat}
              onChange={(event) => {
                setChat(event.target.value);
                markDirty('generation');
              }}
            >
              <option value="inherit">Use instance default</option>
              <option value="enabled">Enabled</option>
              <option value="disabled">Disabled</option>
            </select>
            <p className="field-help">
              {effective(user.chat_enabled, user.chat_enabled_effective)}
            </p>
          </div>
          <div className="form-group">
            <label htmlFor={`generation-policy-${user.id}`}>User Space AI generation</label>
            <select
              id={`generation-policy-${user.id}`}
              value={generation}
              onChange={(event) => {
                setGeneration(event.target.value);
                markDirty('generation');
              }}
            >
              <option value="inherit">Use instance default</option>
              <option value="enabled">Enabled</option>
              <option value="disabled">Disabled</option>
            </select>
            <p className="field-help">
              {effective(
                user.userspace_generation_enabled,
                user.userspace_generation_enabled_effective,
              )}
            </p>
          </div>
          {renderError('generation')}
          <div className="user-management-actions">
            <button
              className="btn btn-primary"
              disabled={busy.generation || !dirty.generation}
              onClick={() =>
                void run('generation', async () => {
                  const updated = await api.updateUserGenerationPolicy(user.id, {
                    chat_enabled: chat === 'inherit' ? null : chat === 'enabled',
                    userspace_generation_enabled:
                      generation === 'inherit' ? null : generation === 'enabled',
                  });
                  props.onUserUpdated(updated);
                  if (isSelf) {
                    try {
                      await props.onGenerationPolicyUpdated?.(updated);
                    } catch (error) {
                      setNotices((current) => ({
                        ...current,
                        generation: `Saved, but authenticated state could not be refreshed: ${error instanceof Error ? error.message : 'unknown error'}`,
                      }));
                    }
                  }
                })
              }
            >
              Save generation settings
            </button>
            <button
              className="btn btn-secondary"
              disabled={!dirty.generation}
              onClick={() => {
                setChat(policyValue(user.chat_enabled));
                setGeneration(policyValue(user.userspace_generation_enabled));
                markDirty('generation', false);
              }}
            >
              Discard
            </button>
          </div>
          {notices.generation && (
            <p className="field-help" role="status">
              {notices.generation}
            </p>
          )}
          {props.contentProtectionLoadFailed ? (
            <div className="user-management-inline-state" role="alert">
              Content protection settings could not be loaded.{' '}
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => void props.onRetryContentProtectionConfig()}
              >
                Retry
              </button>
            </div>
          ) : props.contentProtectionConfig === null ? (
            <p className="field-help" role="status">
              Loading content protection settings…
            </p>
          ) : !props.contentProtectionConfig.enabled ? (
            <p className="field-help">Content protection is disabled for this instance.</p>
          ) : (
            <>
              <div className="form-group">
                <label htmlFor={`protection-policy-${user.id}`}>Content protection override</label>
                <select
                  id={`protection-policy-${user.id}`}
                  value={protection}
                  onChange={(event) => {
                    setProtection(event.target.value as ContentProtectionOverrideMode);
                    markDirty('protection');
                  }}
                >
                  <option value="inherit">Inherit contextual policy</option>
                  <option value="always_classify">Always classify</option>
                  <option value="never_classify">Never classify</option>
                </select>
                <p className="field-help">
                  This setting is contextual; there is no single universal effective protection
                  state.
                </p>
              </div>
              {renderError('protection')}
              <div className="user-management-actions">
                <button
                  className="btn btn-primary"
                  disabled={busy.protection || !dirty.protection}
                  onClick={() =>
                    void run('protection', async () => {
                      const updated = await updateContentProtectionConfigSlice((config) =>
                        withUserOverride(config, user.id, protection),
                      );
                      props.onContentProtectionConfigChange(updated);
                    })
                  }
                >
                  Save content protection
                </button>
                <button
                  className="btn btn-secondary"
                  disabled={!dirty.protection}
                  onClick={() => {
                    setProtection(
                      props.contentProtectionConfig
                        ? userOverrideMode(props.contentProtectionConfig, user.id)
                        : 'inherit',
                    );
                    markDirty('protection', false);
                  }}
                >
                  Discard
                </button>
              </div>
            </>
          )}
        </div>
      ),
    },
    {
      id: 'security',
      label: 'Security',
      content: (
        <div data-user-management-section="security">
          <UserSecurityTab
            user={user}
            isSelf={isSelf}
            onUserUpdated={props.onUserUpdated}
            onDirtyChange={(value) => markDirty('security', value)}
            onBusyChange={(value) => setBusy((current) => ({ ...current, security: value }))}
          />
        </div>
      ),
    },
    {
      id: 'resources',
      label: 'Resources',
      content: (
        <div data-user-management-section="resources">
          <UserResourcesTab {...props} onLoad={props.onResourcesOpen} />
        </div>
      ),
    },
  ];

  return (
    <div
      id="user-management-modal-overlay"
      className="modal-overlay user-management-overlay"
      onMouseDown={() => requestClose()}
    >
      <div
        id={`user-management-modal-${user.id}`}
        className="modal-content modal-with-tabs user-management-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`user-management-title-${user.id}`}
        tabIndex={-1}
        ref={dialog}
        onKeyDown={onDialogKeyDown}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header className="modal-header">
          <div>
            <h3 id={`user-management-title-${user.id}`}>{user.display_name || user.username}</h3>
            <p className="users-subnum user-management-identity">
              <span data-user-management-identity="username">@{user.username}</span>
              <span data-user-management-identity="provider">{user.auth_provider}</span>
              {isSelf && <span data-user-management-identity="self">You</span>}
            </p>
          </div>
          <button
            type="button"
            className="modal-close"
            aria-label="Close manage user"
            onClick={(event) => requestClose(event.currentTarget)}
          >
            &times;
          </button>
        </header>
        <div className="modal-body">
          <ModalTabs
            idPrefix={`user-management-${user.id}`}
            label="Manage user"
            tabs={modalTabs}
            activeTabId={tab}
            onChange={(nextTab) => {
              if (nextTab === 'resources' && tab !== 'resources') props.onResourcesOpen();
              setTab(nextTab as Tab);
            }}
          />
        </div>
        {confirmClose && (
          <div
            className="user-management-close-confirm"
            role="alertdialog"
            aria-label="Discard unsaved changes"
            onKeyDown={(event) => onConfirmationKeyDown(event, 'close')}
          >
            <p>Discard unsaved changes?</p>
            <button
              className="btn btn-danger"
              disabled={hasPendingMutation}
              onClick={discardAndClose}
            >
              Discard
            </button>
            <button
              className="btn btn-secondary"
              ref={discardCancel}
              onClick={() => cancelConfirmation('close')}
            >
              Keep editing
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
