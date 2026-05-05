import { useCallback, useEffect, useRef, useState } from 'react';
import { Eye, EyeOff, Pencil, Plus } from 'lucide-react';
import { api } from '@/api';
import type { AuthGroup, UserRole } from '@/types';
import { DeleteConfirmButton } from '../DeleteConfirmButton';
import { InlineCopyButton } from './InlineCopyButton';
import { Popover } from '../Popover';

type ToastActions = {
  success: (message: string, durationMs?: number) => void;
  error: (message: string, durationMs?: number) => void;
};

interface LocalUserFormState {
  username: string;
  password: string;
  display_name: string;
  email: string;
  role: UserRole;
}

interface AuthAdminModalHostProps {
  createUserOpen: boolean;
  manageGroupsOpen: boolean;
  authGroups: AuthGroup[];
  onAuthGroupsChange: (groups: AuthGroup[]) => void;
  onUsersChanged?: () => void | Promise<void>;
  onCloseCreateUser: () => void;
  onCloseManageGroups: () => void;
  toast: ToastActions;
}

const EMPTY_LOCAL_USER_FORM: LocalUserFormState = {
  username: '',
  password: '',
  display_name: '',
  email: '',
  role: 'user',
};

function generateCredentialValue(length: number): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789';
  const randomValues = new Uint8Array(length);

  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(randomValues);
  } else {
    for (let index = 0; index < randomValues.length; index += 1) {
      randomValues[index] = Math.floor(Math.random() * alphabet.length);
    }
  }

  return Array.from(randomValues, (value) => alphabet[value % alphabet.length]).join('');
}

function sortAuthGroupsByName(groups: AuthGroup[]): AuthGroup[] {
  return [...groups].sort((a, b) => a.display_name.localeCompare(b.display_name));
}

function isLocalManagedAuthGroup(group: AuthGroup): boolean {
  return group.provider === 'local_managed';
}

function getAuthGroupProviderLabel(group: AuthGroup): string {
  if (group.provider === 'ldap') return 'LDAP';
  if (group.provider === 'local_managed') return 'Internal';
  return group.provider;
}

function getProviderSortOrder(provider: string): number {
  if (provider === 'local_managed') return 0;
  if (provider === 'ldap') return 1;
  return 2;
}

type AuthGroupAccessMode = 'none' | 'logon' | 'admin';
type InlineEditField = 'display_name' | 'description';

function getAuthGroupAccessMode(group: AuthGroup): AuthGroupAccessMode {
  if (group.role === 'admin') return 'admin';
  if (group.is_logon_group) return 'logon';
  return 'none';
}

export function AuthAdminModalHost({
  createUserOpen,
  manageGroupsOpen,
  authGroups,
  onAuthGroupsChange,
  onUsersChanged,
  onCloseCreateUser,
  onCloseManageGroups,
  toast,
}: AuthAdminModalHostProps) {
  const [localUserForm, setLocalUserForm] = useState<LocalUserFormState>(EMPTY_LOCAL_USER_FORM);
  const [showLocalUserPassword, setShowLocalUserPassword] = useState(false);
  const [localUserSaving, setLocalUserSaving] = useState(false);

  const [inlineEditId, setInlineEditId] = useState<string | null>(null);
  const [inlineEditField, setInlineEditField] = useState<InlineEditField | null>(null);
  const [inlineEditValue, setInlineEditValue] = useState('');
  const [inlineEditSaving, setInlineEditSaving] = useState(false);
  const [newGroupMode, setNewGroupMode] = useState(false);
  const [newGroupName, setNewGroupName] = useState('');
  const [newGroupSaving, setNewGroupSaving] = useState(false);
  const [authGroupUpdatingId, setAuthGroupUpdatingId] = useState<string | null>(null);
  const [authGroupDeletingId, setAuthGroupDeletingId] = useState<string | null>(null);
  const inlineEditInputRef = useRef<HTMLInputElement>(null);

  const toastRef = useRef(toast);
  useEffect(() => {
    toastRef.current = toast;
  }, [toast]);

  const refreshAuthGroups = useCallback(async () => {
    try {
      const groups = await api.listAuthGroups();
      onAuthGroupsChange(groups);
    } catch (err) {
      toastRef.current.error(err instanceof Error ? err.message : 'Failed to load Group Memberships');
    }
  }, [onAuthGroupsChange]);

  useEffect(() => {
    if (manageGroupsOpen) {
      void refreshAuthGroups();
    }
  }, [manageGroupsOpen, refreshAuthGroups]);

  useEffect(() => {
    if (inlineEditId && inlineEditField && inlineEditInputRef.current) {
      inlineEditInputRef.current.focus();
      inlineEditInputRef.current.select();
    }
  }, [inlineEditField, inlineEditId]);

  const handleCancelInlineEdit = useCallback(() => {
    setInlineEditId(null);
    setInlineEditField(null);
    setInlineEditValue('');
  }, []);

  const closeCreateUserModal = useCallback(() => {
    setShowLocalUserPassword(false);
    setLocalUserForm(EMPTY_LOCAL_USER_FORM);
    onCloseCreateUser();
  }, [onCloseCreateUser]);

  const closeManageGroupsModal = useCallback(() => {
    setInlineEditId(null);
    setInlineEditField(null);
    setInlineEditValue('');
    setNewGroupMode(false);
    setNewGroupName('');
    onCloseManageGroups();
  }, [onCloseManageGroups]);

  const handleCreateLocalUser = async () => {
    setLocalUserSaving(true);
    try {
      await api.createLocalUser({
        username: localUserForm.username.trim(),
        password: localUserForm.password,
        display_name: localUserForm.display_name.trim() || null,
        email: localUserForm.email.trim() || null,
        role: localUserForm.role,
      });
      toast.success('Internal user created');
      await onUsersChanged?.();
      closeCreateUserModal();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to create internal user');
    } finally {
      setLocalUserSaving(false);
    }
  };

  const handleStartInlineEdit = useCallback((group: AuthGroup, field: InlineEditField) => {
    if (!isLocalManagedAuthGroup(group)) return;
    setInlineEditId(group.id);
    setInlineEditField(field);
    setInlineEditValue(field === 'display_name' ? group.display_name : (group.description || ''));
  }, []);

  const handleSaveInlineEdit = async () => {
    if (!inlineEditId || !inlineEditField) return;
    const group = authGroups.find((g) => g.id === inlineEditId);
    if (!group) return;
    if (inlineEditField === 'display_name' && !inlineEditValue.trim()) {
      handleCancelInlineEdit();
      return;
    }

    const nextDisplayName = inlineEditField === 'display_name' ? inlineEditValue.trim() : group.display_name;
    const nextDescription = inlineEditField === 'description' ? inlineEditValue : (group.description || '');

    if (nextDisplayName === group.display_name && nextDescription === (group.description || '')) {
      handleCancelInlineEdit();
      return;
    }

    setInlineEditSaving(true);
    try {
      const updated = await api.updateAuthGroup(inlineEditId, {
        display_name: nextDisplayName,
        description: nextDescription.trim(),
        role: group.role || null,
        is_logon_group: Boolean(group.is_logon_group),
      });
      onAuthGroupsChange(sortAuthGroupsByName(authGroups.map((g) => (g.id === updated.id ? updated : g))));
      toast.success('Auth group updated');
      handleCancelInlineEdit();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to save auth group');
      await refreshAuthGroups();
    } finally {
      setInlineEditSaving(false);
    }
  };

  const handleCreateNewGroup = async () => {
    if (!newGroupName.trim()) return;
    setNewGroupSaving(true);
    try {
      const group = await api.createAuthGroup({
        display_name: newGroupName.trim(),
        description: '',
        role: null,
        is_logon_group: false,
      });
      onAuthGroupsChange(sortAuthGroupsByName([...authGroups, group]));
      toast.success('Internal group created');
      setNewGroupMode(false);
      setNewGroupName('');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to create auth group');
    } finally {
      setNewGroupSaving(false);
    }
  };

  const handleDeleteAuthGroup = async (group: AuthGroup) => {
    if (!isLocalManagedAuthGroup(group)) {
      toast.error('LDAP-synced groups cannot be deleted manually');
      await refreshAuthGroups();
      return;
    }
    setAuthGroupDeletingId(group.id);
    try {
      await api.deleteAuthGroup(group.id);
      onAuthGroupsChange(authGroups.filter((candidate) => candidate.id !== group.id));
      if (inlineEditId === group.id) {
        handleCancelInlineEdit();
      }
      toast.success('Internal group deleted');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete internal group');
      await refreshAuthGroups();
    } finally {
      setAuthGroupDeletingId(null);
    }
  };

  const handleToggleAuthGroupAssignment = async (
    group: AuthGroup,
    updates: Partial<Pick<AuthGroup, 'role' | 'is_logon_group'>>,
  ) => {
    let nextRole = updates.role !== undefined ? updates.role : group.role;
    let nextIsLogonGroup = updates.is_logon_group !== undefined ? updates.is_logon_group : group.is_logon_group;

    // Keep assignment state mutually exclusive: admin implies non-logon and vice versa.
    if (updates.role === 'admin') {
      nextIsLogonGroup = false;
    }
    if (updates.is_logon_group === true) {
      nextRole = null;
    }

    setAuthGroupUpdatingId(group.id);
    try {
      const updated = await api.updateAuthGroup(group.id, {
        display_name: group.display_name,
        description: group.description || '',
        role: nextRole || null,
        is_logon_group: Boolean(nextIsLogonGroup),
      });
      onAuthGroupsChange(sortAuthGroupsByName(authGroups.map((candidate) => (candidate.id === updated.id ? updated : candidate))));
      toast.success('Auth group assignment updated');
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to update auth group assignment');
      await refreshAuthGroups();
    } finally {
      setAuthGroupUpdatingId(null);
    }
  };

  const handleSetAuthGroupAccessMode = async (group: AuthGroup, mode: AuthGroupAccessMode) => {
    if (mode === 'admin') {
      await handleToggleAuthGroupAssignment(group, { role: 'admin', is_logon_group: false });
      return;
    }
    if (mode === 'logon') {
      await handleToggleAuthGroupAssignment(group, { role: null, is_logon_group: true });
      return;
    }
    await handleToggleAuthGroupAssignment(group, { role: null, is_logon_group: false });
  };

  const handleInlineEditKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      handleCancelInlineEdit();
      return;
    }

    if (event.key === 'Enter') {
      event.preventDefault();
      void handleSaveInlineEdit();
    }
  };

  const groupedAuthGroups = authGroups.reduce<Record<string, AuthGroup[]>>((acc, group) => {
    const provider = group.provider || 'other';
    if (!acc[provider]) {
      acc[provider] = [];
    }
    acc[provider].push(group);
    return acc;
  }, {});

  const providerSections = Object.entries(groupedAuthGroups)
    .sort(([providerA], [providerB]) => {
      const orderA = getProviderSortOrder(providerA);
      const orderB = getProviderSortOrder(providerB);
      if (orderA !== orderB) return orderA - orderB;
      return providerA.localeCompare(providerB);
    })
    .map(([provider, groups]) => ({
      provider,
      label: getAuthGroupProviderLabel(groups[0]),
      groups: sortAuthGroupsByName(groups),
    }));

  return (
    <>
      {createUserOpen && (
        <div className="modal-overlay" onClick={closeCreateUserModal}>
          <div className="modal-content modal-medium" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3>Create Internal User</h3>
              <button className="modal-close" onClick={closeCreateUserModal}>&times;</button>
            </div>
            <div className="modal-body">
              <div className="form-row-3" style={{ gridTemplateColumns: '1fr 1.6fr 0.7fr' }}>
                <div className="form-group">
                  <label>Username</label>
                  <input type="text" value={localUserForm.username} onChange={(event) => setLocalUserForm({ ...localUserForm, username: event.target.value })} placeholder="jane.doe" autoFocus />
                </div>
                <div className="form-group">
                  <label>Password</label>
                  <div className="input-with-button">
                    <div className="settings-inline-copy-wrap local-user-password-copy-wrap" style={{ flex: 1 }}>
                      <input
                        type={showLocalUserPassword ? 'text' : 'password'}
                        value={localUserForm.password}
                        onChange={(event) => setLocalUserForm({ ...localUserForm, password: event.target.value })}
                        placeholder="At least 8 characters"
                        style={{ width: '100%', fontFamily: 'var(--font-mono)' }}
                      />
                      <InlineCopyButton
                        copyText={localUserForm.password}
                        className="settings-inline-copy"
                        disabled={!localUserForm.password}
                        title="Copy password"
                        ariaLabel="Copy password"
                        copiedTitle="Password copied"
                        copiedAriaLabel="Password copied"
                        feedbackMs={2000}
                        onCopySuccess={() => toast.success('Password copied')}
                        onCopyError={() => toast.error('Unable to copy password. Please copy it manually.')}
                      />
                      <button
                        type="button"
                        className="settings-inline-copy settings-inline-copy-secondary"
                        onClick={() => setShowLocalUserPassword(!showLocalUserPassword)}
                        title={showLocalUserPassword ? 'Hide password' : 'Show password'}
                        aria-label={showLocalUserPassword ? 'Hide password' : 'Show password'}
                      >
                        {showLocalUserPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                      </button>
                    </div>
                    <button
                      type="button"
                      className="btn btn-sm btn-secondary"
                      onClick={() => setLocalUserForm({ ...localUserForm, password: generateCredentialValue(20) })}
                    >
                      Generate
                    </button>
                  </div>
                </div>
                <div className="form-group">
                  <label>Role</label>
                  <select value={localUserForm.role} onChange={(event) => setLocalUserForm({ ...localUserForm, role: event.target.value as UserRole })}>
                    <option value="user">user</option>
                    <option value="admin">admin</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Display Name</label>
                  <input type="text" value={localUserForm.display_name} onChange={(event) => setLocalUserForm({ ...localUserForm, display_name: event.target.value })} placeholder="Jane Doe" />
                </div>
                <div className="form-group">
                  <label>Email</label>
                  <input type="email" value={localUserForm.email} onChange={(event) => setLocalUserForm({ ...localUserForm, email: event.target.value })} placeholder="jane@example.com" />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-secondary" onClick={closeCreateUserModal} disabled={localUserSaving}>Cancel</button>
              <button type="button" className="btn" onClick={handleCreateLocalUser} disabled={localUserSaving || !localUserForm.username.trim() || localUserForm.password.length < 8}>
                {localUserSaving ? 'Creating...' : 'Create Internal User'}
              </button>
            </div>
          </div>
        </div>
      )}

      {manageGroupsOpen && (
        <div className="modal-overlay" onClick={closeManageGroupsModal}>
          <div className="modal-content modal-large auth-group-manage-modal" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3>Manage Group Memberships</h3>
                <p className="auth-group-modal-subtitle">{authGroups.filter(isLocalManagedAuthGroup).length} internal, {authGroups.filter((group) => group.provider === 'ldap').length} LDAP</p>
              </div>
              <button className="modal-close" onClick={closeManageGroupsModal}>&times;</button>
            </div>
            <div className="modal-body auth-group-manage-body">
              <div className="auth-group-manage-list-panel">
                <div className="auth-group-panel-header">
                  <h4>Groups</h4>
                  {!newGroupMode && (
                    <button type="button" className="btn btn-sm btn-secondary" onClick={() => setNewGroupMode(true)}>
                      <Plus size={14} />
                      New Group
                    </button>
                  )}
                </div>
                {newGroupMode && (
                  <div className="auth-group-new-group-row">
                    <input
                      type="text"
                      className="auth-group-row-inline-input"
                      value={newGroupName}
                      onChange={(e) => setNewGroupName(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') void handleCreateNewGroup();
                        if (e.key === 'Escape') { setNewGroupMode(false); setNewGroupName(''); }
                      }}
                      placeholder="Group name"
                      disabled={newGroupSaving}
                      autoFocus
                    />
                    <button type="button" className="btn" onClick={() => void handleCreateNewGroup()} disabled={newGroupSaving || !newGroupName.trim()}>
                      {newGroupSaving ? 'Creating...' : 'Create'}
                    </button>
                    <button type="button" className="btn btn-secondary" onClick={() => { setNewGroupMode(false); setNewGroupName(''); }} disabled={newGroupSaving}>
                      Cancel
                    </button>
                  </div>
                )}

                {authGroups.length === 0 ? (
                  <div className="auth-group-empty-state">No groups created yet.</div>
                ) : (
                  <div className="auth-group-manage-list">
                    {providerSections.map((section) => (
                      <div key={section.provider} className="auth-group-provider-section">
                        <div className="model-group-header auth-group-provider-section-header">
                          <span>{section.label}</span>
                          <span className="auth-group-provider-section-count">{section.groups.length}</span>
                        </div>
                        {section.groups.map((group) => {
                          const localGroup = isLocalManagedAuthGroup(group);
                          const busy = authGroupDeletingId === group.id || authGroupUpdatingId === group.id;
                          const memberPreviews = (group.member_previews || []).filter((member) => member?.username);
                          const fallbackMemberLabels = !memberPreviews.length
                            ? ((group as AuthGroup & { member_labels?: string[] }).member_labels || [])
                            : [];
                          const accessMode = getAuthGroupAccessMode(group);
                          return (
                            <Popover
                              key={group.id}
                              trigger="hover"
                              position="right"
                              className="auth-group-row-popover-trigger"
                              content={
                                <div className="auth-group-members-popover">
                                  <div className="auth-group-members-popover-header">{group.display_name} members</div>
                                  {memberPreviews.length === 0 && fallbackMemberLabels.length === 0 ? (
                                    <div className="auth-group-members-popover-status">No members in this group.</div>
                                  ) : (
                                    <ul className="auth-group-members-popover-list">
                                      {memberPreviews.map((member) => {
                                        const displayName = member.display_name || member.username;
                                        return (
                                          <li key={`${group.id}-${member.username}`} className="auth-group-members-popover-item">
                                            <span className="auth-group-member-display-name">{displayName}</span>
                                            <span className="auth-group-member-handle">@{member.username}</span>
                                          </li>
                                        );
                                      })}
                                      {fallbackMemberLabels.map((label) => (
                                        <li key={`${group.id}-label-${label}`} className="auth-group-members-popover-item">
                                          <span className="auth-group-member-display-name">{label}</span>
                                        </li>
                                      ))}
                                    </ul>
                                  )}
                                </div>
                              }
                            >
                              <div className={`auth-group-manage-row${localGroup ? '' : ' is-synced'}${inlineEditId === group.id ? ' is-editing' : ''}`}>
                                <div className="auth-group-row-main">
                                  <div className="auth-group-row-title-wrap">
                                    {inlineEditId === group.id && inlineEditField === 'display_name' ? (
                                      <div className="inline-edit-field auth-group-inline-edit-field">
                                        <input
                                          ref={inlineEditInputRef}
                                          type="text"
                                          className="inline-edit-input auth-group-row-inline-input"
                                          value={inlineEditValue}
                                          onChange={(e) => setInlineEditValue(e.target.value)}
                                          onKeyDown={handleInlineEditKeyDown}
                                          onBlur={() => void handleSaveInlineEdit()}
                                          disabled={inlineEditSaving}
                                        />
                                      </div>
                                    ) : (
                                      <div
                                        className={`editable-field-wrapper name-wrapper auth-group-row-editable-title ${localGroup ? 'editable' : ''}`}
                                        onClick={localGroup ? () => handleStartInlineEdit(group, 'display_name') : undefined}
                                      >
                                        <div className="auth-group-row-title">{group.display_name}</div>
                                        <span className="auth-group-row-title-member-count">{group.member_count} member{group.member_count === 1 ? '' : 's'}</span>
                                        {localGroup && (
                                          <button
                                            type="button"
                                            className="inline-edit-btn"
                                            onClick={(event) => {
                                              event.stopPropagation();
                                              handleStartInlineEdit(group, 'display_name');
                                            }}
                                            title="Edit group name"
                                            aria-label={`Edit ${group.display_name} name`}
                                          >
                                            <Pencil size={12} />
                                          </button>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                  {group.manual_member_count > 0 && (
                                    <div className="auth-group-row-meta">
                                      <span>{group.manual_member_count} manual</span>
                                    </div>
                                  )}
                                  {inlineEditId === group.id && inlineEditField === 'description' ? (
                                    <div className="inline-edit-field auth-group-inline-edit-field description-edit">
                                      <input
                                        ref={inlineEditInputRef}
                                        type="text"
                                        className="inline-edit-input auth-group-row-inline-input auth-group-row-inline-desc"
                                        value={inlineEditValue}
                                        onChange={(e) => setInlineEditValue(e.target.value)}
                                        onKeyDown={handleInlineEditKeyDown}
                                        onBlur={() => void handleSaveInlineEdit()}
                                        disabled={inlineEditSaving}
                                        placeholder="Description (optional)"
                                      />
                                    </div>
                                  ) : (
                                    <div
                                      className={`editable-field-wrapper auth-group-row-editable-description ${localGroup ? 'editable' : ''}`}
                                      onClick={localGroup ? () => handleStartInlineEdit(group, 'description') : undefined}
                                    >
                                      {group.description || group.source_dn ? (
                                        <div className="auth-group-row-description" title={group.source_dn || group.description}>
                                          {group.description || group.source_dn}
                                        </div>
                                      ) : (
                                        localGroup && <div className="auth-group-row-description auth-group-row-description-placeholder">Add description</div>
                                      )}
                                      {localGroup && (
                                        <button
                                          type="button"
                                          className="inline-edit-btn"
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            handleStartInlineEdit(group, 'description');
                                          }}
                                          title="Edit group description"
                                          aria-label={`Edit ${group.display_name} description`}
                                        >
                                          <Pencil size={12} />
                                        </button>
                                      )}
                                    </div>
                                  )}
                                </div>
                                <div className="auth-group-row-side">
                                  <div className="auth-group-row-status-pills">
                                    <span className={`auth-group-provider-badge auth-group-provider-${group.provider}`}>{getAuthGroupProviderLabel(group)}</span>
                                    {group.role && <span className={`auth-group-role-badge${group.role === 'admin' ? ' auth-group-admin-badge' : ''}`}>{group.role}</span>}
                                    {group.is_logon_group && <span className="auth-group-logon-badge">Logon</span>}
                                    {localGroup && (
                                      <div className="auth-group-row-status-actions">
                                        <DeleteConfirmButton
                                          onDelete={() => { void handleDeleteAuthGroup(group); }}
                                          disabled={busy}
                                          deleting={authGroupDeletingId === group.id}
                                          className="btn btn-sm btn-danger auth-group-delete-button"
                                          title="Delete group"
                                          buttonText="Delete"
                                        />
                                      </div>
                                    )}
                                  </div>
                                  <div className="auth-group-row-actions">
                                    <div className="auth-group-access-segment" role="group" aria-label={`Access mode for ${group.display_name}`}>
                                      <button
                                        type="button"
                                        className={`auth-group-access-option${accessMode === 'none' ? ' is-active' : ''}`}
                                        disabled={busy}
                                        onClick={() => void handleSetAuthGroupAccessMode(group, 'none')}
                                        title="No role grant and not a logon group"
                                      >
                                        None
                                      </button>
                                      <button
                                        type="button"
                                        className={`auth-group-access-option${accessMode === 'logon' ? ' is-active' : ''}`}
                                        disabled={busy}
                                        onClick={() => void handleSetAuthGroupAccessMode(group, 'logon')}
                                        title="Allow members of this group to log in"
                                      >
                                        Logon
                                      </button>
                                      <button
                                        type="button"
                                        className={`auth-group-access-option${accessMode === 'admin' ? ' is-active' : ''}`}
                                        disabled={busy}
                                        onClick={() => void handleSetAuthGroupAccessMode(group, 'admin')}
                                        title="Promote members of this group to admin"
                                      >
                                        Admin
                                      </button>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            </Popover>
                          );
                        })}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
