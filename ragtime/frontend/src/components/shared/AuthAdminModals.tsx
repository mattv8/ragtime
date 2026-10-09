import { useCallback, useEffect, useRef, useState } from 'react';
import { Eye, EyeOff, Pencil, Plus, ShieldCheck } from 'lucide-react';
import { api } from '@/api';
import {
  contentProtectionApi,
  DELETED_ACCESS_LEVEL_MESSAGE,
  updateContentProtectionConfigSlice,
  withExistingAccessLevel,
  withGroupAccessLevel,
  withRequirement,
} from '@/api/contentProtection';
import type { ContentProtectionConfig } from '@/api/contentProtection';
import { generateCredentialValue } from '@/utils/credentialGenerator';
import type { AuthGroup, UserRole } from '@/types';
import { DeleteConfirmButton } from '../DeleteConfirmButton';
import { InlineCopyButton } from './InlineCopyButton';
import { Popover } from '../Popover';
import { GroupProtectionPanel, type GroupProtectionPanelHandle } from './GroupProtectionPanel';
import { AccessDetailHeader } from './AccessDetailHeader';

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
  onNavigateToSetting?: (settingId: string) => void;
  toast: ToastActions;
}

const UNAVAILABLE_GROUP_MESSAGE =
  'This group is no longer available. Protection changes are disabled; discard or close this editor.';

const EMPTY_LOCAL_USER_FORM: LocalUserFormState = {
  username: '',
  password: '',
  display_name: '',
  email: '',
  role: 'user',
};

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
  onNavigateToSetting,
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
  const [contentProtectionConfig, setContentProtectionConfig] =
    useState<ContentProtectionConfig | null>(null);
  const [contentProtectionError, setContentProtectionError] = useState<string | null>(null);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [selectedGroupSnapshot, setSelectedGroupSnapshot] = useState<AuthGroup | null>(null);
  const [protectionPanelBusy, setProtectionPanelBusy] = useState(false);
  const inlineEditInputRef = useRef<HTMLInputElement>(null);
  const protectionPanelRef = useRef<GroupProtectionPanelHandle>(null);
  const protectionTriggerRef = useRef<HTMLButtonElement | null>(null);
  const groupRailRef = useRef<HTMLDivElement>(null);
  const protectionConfigRequest = useRef(0);

  const liveSelectedGroup = selectedGroupId
    ? authGroups.find((group) => group.id === selectedGroupId)
    : undefined;
  // Keep the last known group so a dirty draft survives external removal of the group.
  const protectedGroup =
    liveSelectedGroup ??
    (selectedGroupSnapshot && selectedGroupSnapshot.id === selectedGroupId
      ? selectedGroupSnapshot
      : null);
  const protectedGroupUnavailable = Boolean(
    selectedGroupId && !liveSelectedGroup && protectedGroup,
  );

  const toastRef = useRef(toast);
  useEffect(() => {
    toastRef.current = toast;
  }, [toast]);

  const refreshAuthGroups = useCallback(async () => {
    try {
      const groups = await api.listAuthGroups();
      onAuthGroupsChange(groups);
    } catch (err) {
      toastRef.current.error(
        err instanceof Error ? err.message : 'Failed to load Group Memberships',
      );
    }
  }, [onAuthGroupsChange]);

  // Only the newest config request may write state; closing or reopening the modal invalidates older ones.
  const loadProtectionConfig = useCallback(async () => {
    const requestId = protectionConfigRequest.current + 1;
    protectionConfigRequest.current = requestId;
    setContentProtectionError(null);
    try {
      const config = await contentProtectionApi.getConfig();
      if (protectionConfigRequest.current === requestId) setContentProtectionConfig(config);
    } catch (error) {
      if (protectionConfigRequest.current === requestId) {
        setContentProtectionError(
          error instanceof Error ? error.message : 'Could not load protection config.',
        );
      }
    }
  }, []);

  useEffect(() => {
    if (!manageGroupsOpen) return undefined;
    void refreshAuthGroups();
    setContentProtectionConfig(null);
    void loadProtectionConfig();
    return () => {
      protectionConfigRequest.current += 1;
    };
  }, [manageGroupsOpen, loadProtectionConfig, refreshAuthGroups]);

  useEffect(() => {
    if (liveSelectedGroup) setSelectedGroupSnapshot(liveSelectedGroup);
  }, [liveSelectedGroup]);

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
    setSelectedGroupId(null);
    onCloseManageGroups();
  }, [onCloseManageGroups]);

  const requestPanelNavigation = useCallback((continueNavigation: () => void) => {
    if (protectionPanelRef.current)
      protectionPanelRef.current.requestNavigation(continueNavigation);
    else continueNavigation();
  }, []);
  const navigateToSetting = useCallback(
    (settingId: string) => {
      requestPanelNavigation(() => {
        closeManageGroupsModal();
        onNavigateToSetting?.(settingId);
      });
    },
    [closeManageGroupsModal, onNavigateToSetting, requestPanelNavigation],
  );

  const selectGroup = useCallback(
    (group: AuthGroup, trigger?: HTMLButtonElement) => {
      const select = () => {
        if (trigger) protectionTriggerRef.current = trigger;
        setSelectedGroupId(group.id);
        requestAnimationFrame(() =>
          document.getElementById(`group-detail-heading-${group.id}`)?.focus(),
        );
      };
      if (selectedGroupId && selectedGroupId !== group.id) requestPanelNavigation(select);
      else select();
    },
    [requestPanelNavigation, selectedGroupId],
  );

  const closeProtectionPanel = useCallback(() => {
    const previousGroupId = selectedGroupId;
    setSelectedGroupId(null);
    requestAnimationFrame(() => {
      const groupRow = [...document.querySelectorAll<HTMLElement>('[data-group-id]')].find(
        (row) => row.dataset.groupId === previousGroupId,
      );
      const rowButton =
        groupRow?.querySelector<HTMLButtonElement>('.group-protection-button') ??
        groupRow?.querySelector<HTMLButtonElement>('button');
      (rowButton ?? groupRailRef.current)?.focus();
    });
  }, [selectedGroupId]);

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
    setInlineEditValue(field === 'display_name' ? group.display_name : group.description || '');
  }, []);

  const handleSaveInlineEdit = async () => {
    if (!inlineEditId || !inlineEditField) return;
    const group = authGroups.find((g) => g.id === inlineEditId);
    if (!group) return;
    if (inlineEditField === 'display_name' && !inlineEditValue.trim()) {
      handleCancelInlineEdit();
      return;
    }

    const nextDisplayName =
      inlineEditField === 'display_name' ? inlineEditValue.trim() : group.display_name;
    const nextDescription =
      inlineEditField === 'description' ? inlineEditValue : group.description || '';

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
      onAuthGroupsChange(
        sortAuthGroupsByName(authGroups.map((g) => (g.id === updated.id ? updated : g))),
      );
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

  const deleteAuthGroupNow = async (group: AuthGroup) => {
    setAuthGroupDeletingId(group.id);
    try {
      await api.deleteAuthGroup(group.id);
      onAuthGroupsChange(authGroups.filter((candidate) => candidate.id !== group.id));
      setSelectedGroupId((current) => (current === group.id ? null : current));
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

  const handleDeleteAuthGroup = (group: AuthGroup) => {
    if (!isLocalManagedAuthGroup(group)) {
      toast.error('LDAP-synced groups cannot be deleted manually');
      void refreshAuthGroups();
      return;
    }
    // Deleting the open group must first resolve any dirty protection draft.
    if (selectedGroupId === group.id) requestPanelNavigation(() => void deleteAuthGroupNow(group));
    else void deleteAuthGroupNow(group);
  };

  const handleToggleAuthGroupAssignment = async (
    group: AuthGroup,
    updates: Partial<Pick<AuthGroup, 'role' | 'is_logon_group'>>,
  ) => {
    let nextRole = updates.role !== undefined ? updates.role : group.role;
    let nextIsLogonGroup =
      updates.is_logon_group !== undefined ? updates.is_logon_group : group.is_logon_group;

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
      onAuthGroupsChange(
        sortAuthGroupsByName(
          authGroups.map((candidate) => (candidate.id === updated.id ? updated : candidate)),
        ),
      );
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

  const handleContentProtectionRequirementChange = async (
    group: AuthGroup,
    mode: 'inherit' | 'require',
  ) => {
    if (!contentProtectionConfig) return;
    try {
      const savedConfig = await updateContentProtectionConfigSlice((config) =>
        withRequirement(config, 'group', group.id, mode),
      );
      setContentProtectionConfig(savedConfig);
      toast.success('Content protection classification updated');
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : 'Failed to update content protection classification',
      );
    }
  };

  const handleContentProtectionAccessLevelChange = async (
    group: AuthGroup,
    accessLevelId: string,
    enabled: boolean,
  ) => {
    if (!contentProtectionConfig) return;
    try {
      // The slice helper may retry after a 409; check the target on every fresh config.
      const savedConfig = await updateContentProtectionConfigSlice((config) =>
        withExistingAccessLevel(config, accessLevelId, () =>
          withGroupAccessLevel(config, group.id, accessLevelId, enabled),
        ),
      );
      setContentProtectionConfig(savedConfig);
      toast.success('Content protection access level updated');
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Failed to update content protection access level';
      toast.error(message);
      if (message === DELETED_ACCESS_LEVEL_MESSAGE) void loadProtectionConfig();
    }
  };

  const handleInlineEditKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Escape') {
      event.preventDefault();
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

  const renderGroupDetailControls = (group: AuthGroup) => {
    const localGroup = isLocalManagedAuthGroup(group);
    const busy = authGroupDeletingId === group.id || authGroupUpdatingId === group.id;
    const accessMode = getAuthGroupAccessMode(group);

    return (
      <div data-group-detail-controls>
        {localGroup && (
          <section className="group-detail-section">
            <h5>Group</h5>
            <div className="auth-group-row-title-wrap">
              {inlineEditId === group.id && inlineEditField === 'display_name' ? (
                <input
                  ref={inlineEditInputRef}
                  type="text"
                  className="auth-group-row-inline-input"
                  value={inlineEditValue}
                  onChange={(event) => setInlineEditValue(event.target.value)}
                  onKeyDown={handleInlineEditKeyDown}
                  onBlur={() => void handleSaveInlineEdit()}
                  disabled={inlineEditSaving || protectionPanelBusy}
                />
              ) : (
                <div className="editable-field-wrapper name-wrapper auth-group-row-editable-title editable">
                  <div className="auth-group-row-title">{group.display_name}</div>
                  <button
                    type="button"
                    className="inline-edit-btn"
                    onClick={() => handleStartInlineEdit(group, 'display_name')}
                    disabled={protectionPanelBusy}
                    aria-label={`Edit ${group.display_name} name`}
                  >
                    <Pencil size={12} />
                  </button>
                </div>
              )}
            </div>
            {inlineEditId === group.id && inlineEditField === 'description' ? (
              <input
                ref={inlineEditInputRef}
                type="text"
                className="auth-group-row-inline-input auth-group-row-inline-desc"
                value={inlineEditValue}
                onChange={(event) => setInlineEditValue(event.target.value)}
                onKeyDown={handleInlineEditKeyDown}
                onBlur={() => void handleSaveInlineEdit()}
                disabled={inlineEditSaving || protectionPanelBusy}
                placeholder="Description (optional)"
              />
            ) : (
              <div className="editable-field-wrapper auth-group-row-editable-description editable">
                <div className="auth-group-row-description">
                  {group.description || 'Add description'}
                </div>
                <button
                  type="button"
                  className="inline-edit-btn"
                  onClick={() => handleStartInlineEdit(group, 'description')}
                  disabled={protectionPanelBusy}
                  aria-label={`Edit ${group.display_name} description`}
                >
                  <Pencil size={12} />
                </button>
              </div>
            )}
          </section>
        )}
        <div className="group-detail-access-row">
          <span className="group-detail-access-label">Access</span>
          <div
            className="auth-group-access-segment"
            role="group"
            aria-label={`Access mode for ${group.display_name}`}
          >
            {(['none', 'logon', 'admin'] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                className={`auth-group-access-option${accessMode === mode ? ' is-active' : ''}`}
                disabled={busy || protectionPanelBusy || protectedGroupUnavailable}
                onClick={() => void handleSetAuthGroupAccessMode(group, mode)}
              >
                {mode === 'none' ? 'None' : mode === 'logon' ? 'Logon' : 'Admin'}
              </button>
            ))}
          </div>
          {localGroup && (
            <DeleteConfirmButton
              onDelete={() => void handleDeleteAuthGroup(group)}
              disabled={busy || protectionPanelBusy}
              deleting={authGroupDeletingId === group.id}
              className="btn btn-sm btn-danger auth-group-delete-button"
              title="Delete group"
              buttonText="Delete"
            />
          )}
        </div>
      </div>
    );
  };

  return (
    <>
      {createUserOpen && (
        <div className="modal-overlay" onClick={closeCreateUserModal}>
          <div className="modal-content modal-medium" onClick={(event) => event.stopPropagation()}>
            <div className="modal-header">
              <h3>Create Internal User</h3>
              <button className="modal-close" onClick={closeCreateUserModal}>
                &times;
              </button>
            </div>
            <div className="modal-body">
              <div className="form-row-3" style={{ gridTemplateColumns: '1fr 1.6fr 0.7fr' }}>
                <div className="form-group">
                  <label>Username</label>
                  <input
                    type="text"
                    value={localUserForm.username}
                    onChange={(event) =>
                      setLocalUserForm({ ...localUserForm, username: event.target.value })
                    }
                    placeholder="jane.doe"
                    autoFocus
                  />
                </div>
                <div className="form-group">
                  <label>Password</label>
                  <div className="input-with-button">
                    <div
                      className="settings-inline-copy-wrap local-user-password-copy-wrap"
                      style={{ flex: 1 }}
                    >
                      <input
                        type={showLocalUserPassword ? 'text' : 'password'}
                        value={localUserForm.password}
                        onChange={(event) =>
                          setLocalUserForm({ ...localUserForm, password: event.target.value })
                        }
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
                        onCopyError={() =>
                          toast.error('Unable to copy password. Please copy it manually.')
                        }
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
                      onClick={() =>
                        setLocalUserForm({
                          ...localUserForm,
                          password: generateCredentialValue(20),
                        })
                      }
                    >
                      Generate
                    </button>
                  </div>
                </div>
                <div className="form-group">
                  <label>Role</label>
                  <select
                    value={localUserForm.role}
                    onChange={(event) =>
                      setLocalUserForm({ ...localUserForm, role: event.target.value as UserRole })
                    }
                  >
                    <option value="user">user</option>
                    <option value="admin">admin</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Display Name</label>
                  <input
                    type="text"
                    value={localUserForm.display_name}
                    onChange={(event) =>
                      setLocalUserForm({ ...localUserForm, display_name: event.target.value })
                    }
                    placeholder="Jane Doe"
                  />
                </div>
                <div className="form-group">
                  <label>Email</label>
                  <input
                    type="email"
                    value={localUserForm.email}
                    onChange={(event) =>
                      setLocalUserForm({ ...localUserForm, email: event.target.value })
                    }
                    placeholder="jane@example.com"
                  />
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <button
                type="button"
                className="btn btn-secondary"
                onClick={closeCreateUserModal}
                disabled={localUserSaving}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn"
                onClick={handleCreateLocalUser}
                disabled={
                  localUserSaving ||
                  !localUserForm.username.trim() ||
                  localUserForm.password.length < 8
                }
              >
                {localUserSaving ? 'Creating...' : 'Create Internal User'}
              </button>
            </div>
          </div>
        </div>
      )}

      {manageGroupsOpen && (
        <div
          className="modal-overlay"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              requestPanelNavigation(
                selectedGroupId ? closeProtectionPanel : closeManageGroupsModal,
              );
            }
          }}
        >
          <div
            className={`modal-content auth-group-manage-modal${selectedGroupId ? ' detail-open' : ''}`}
            id="manage-group-memberships-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="manage-groups-title"
            onClick={(event) => event.stopPropagation()}
            onMouseDown={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
              if (event.key === 'Escape' && !event.defaultPrevented) {
                event.preventDefault();
                requestPanelNavigation(
                  selectedGroupId ? closeProtectionPanel : closeManageGroupsModal,
                );
              }
            }}
          >
            <div className="modal-header">
              <div>
                <h3 id="manage-groups-title">Manage Group Memberships</h3>
                <p className="auth-group-modal-subtitle">
                  {authGroups.filter(isLocalManagedAuthGroup).length} internal,{' '}
                  {authGroups.filter((group) => group.provider === 'ldap').length} LDAP
                </p>
              </div>
              <button
                className="modal-close"
                onClick={() => requestPanelNavigation(closeManageGroupsModal)}
              >
                &times;
              </button>
            </div>
            <div
              className="modal-body auth-group-manage-body access-md"
              data-auth-group-modal-body
              {...(selectedGroupId ? { 'data-detail-open': '' } : {})}
            >
              <div className="auth-group-manage-list-panel access-md-list" data-group-rail>
                <div className="auth-group-panel-header" data-group-rail-header>
                  <h4>Groups</h4>
                  {!newGroupMode && (
                    <button
                      type="button"
                      className="btn btn-sm btn-secondary"
                      onClick={() => setNewGroupMode(true)}
                      disabled={protectionPanelBusy}
                    >
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
                        if (e.key === 'Escape') {
                          e.preventDefault();
                          setNewGroupMode(false);
                          setNewGroupName('');
                        }
                      }}
                      placeholder="Group name"
                      disabled={newGroupSaving}
                      autoFocus
                    />
                    <button
                      type="button"
                      className="btn"
                      onClick={() => void handleCreateNewGroup()}
                      disabled={newGroupSaving || !newGroupName.trim()}
                    >
                      {newGroupSaving ? 'Creating...' : 'Create'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => {
                        setNewGroupMode(false);
                        setNewGroupName('');
                      }}
                      disabled={newGroupSaving}
                    >
                      Cancel
                    </button>
                  </div>
                )}

                {authGroups.length === 0 ? (
                  <div className="auth-group-empty-state">No groups created yet.</div>
                ) : (
                  <div
                    ref={groupRailRef}
                    className="auth-group-manage-list access-md-list-body"
                    aria-label="Groups"
                    role="list"
                    data-group-list
                    tabIndex={-1}
                  >
                    {providerSections.map((section) => (
                      <div key={section.provider} className="auth-group-provider-section">
                        <div className="model-group-header auth-group-provider-section-header">
                          <span>{section.label}</span>
                          <span className="auth-group-provider-section-count">
                            {section.groups.length}
                          </span>
                        </div>
                        {section.groups.map((group) => {
                          const localGroup = isLocalManagedAuthGroup(group);
                          const busy =
                            authGroupDeletingId === group.id || authGroupUpdatingId === group.id;
                          const memberPreviews = (group.member_previews || []).filter(
                            (member) => member?.username,
                          );
                          const fallbackMemberLabels = !memberPreviews.length
                            ? (group as AuthGroup & { member_labels?: string[] }).member_labels ||
                              []
                            : [];
                          const accessMode = getAuthGroupAccessMode(group);
                          const groupAccessLevelIds =
                            contentProtectionConfig?.group_access_levels
                              .filter((mapping) => mapping.group_id === group.id)
                              .map((mapping) => mapping.access_level_id) || [];
                          const accessLevelSummary = contentProtectionConfig
                            ? groupAccessLevelIds.length
                              ? contentProtectionConfig.access_levels
                                  .filter((level) => groupAccessLevelIds.includes(level.id))
                                  .map((level) => level.name)
                                  .join(', ')
                              : 'Default only'
                            : 'Protection unavailable';
                          if (selectedGroupId) {
                            return (
                              <div key={group.id} data-group-id={group.id} role="listitem">
                                <button
                                  type="button"
                                  id={`group-rail-row-${group.id}`}
                                  data-group-rail-row={group.id}
                                  className={`access-md-rail-row${selectedGroupId === group.id ? ' is-selected' : ''}`}
                                  aria-current={selectedGroupId === group.id ? 'true' : undefined}
                                  disabled={protectionPanelBusy}
                                  onClick={(event) => selectGroup(group, event.currentTarget)}
                                >
                                  <span
                                    className="access-md-rail-row-name"
                                    title={group.display_name}
                                  >
                                    {group.display_name}
                                  </span>
                                  <span className="access-md-rail-row-meta">
                                    <span
                                      className={`auth-group-provider-badge auth-group-provider-${group.provider}`}
                                    >
                                      {getAuthGroupProviderLabel(group)}
                                    </span>
                                    <span>
                                      {group.member_count} member
                                      {group.member_count === 1 ? '' : 's'}
                                    </span>
                                    <span className="auth-group-protection-summary access-md-rail-row-level">
                                      {accessLevelSummary}
                                    </span>
                                  </span>
                                </button>
                              </div>
                            );
                          }
                          return (
                            <Popover
                              key={group.id}
                              trigger="hover"
                              position="right"
                              className="auth-group-row-popover-trigger"
                              content={
                                <div className="auth-group-members-popover">
                                  <div className="auth-group-members-popover-header">
                                    {group.display_name} members
                                  </div>
                                  {memberPreviews.length === 0 &&
                                  fallbackMemberLabels.length === 0 ? (
                                    <div className="auth-group-members-popover-status">
                                      No members in this group.
                                    </div>
                                  ) : (
                                    <ul className="auth-group-members-popover-list">
                                      {memberPreviews.map((member) => {
                                        const displayName = member.display_name || member.username;
                                        return (
                                          <li
                                            key={`${group.id}-${member.username}`}
                                            className="auth-group-members-popover-item"
                                          >
                                            <span className="auth-group-member-display-name">
                                              {displayName}
                                            </span>
                                            <span className="auth-group-member-handle">
                                              @{member.username}
                                            </span>
                                          </li>
                                        );
                                      })}
                                      {fallbackMemberLabels.map((label) => (
                                        <li
                                          key={`${group.id}-label-${label}`}
                                          className="auth-group-members-popover-item"
                                        >
                                          <span className="auth-group-member-display-name">
                                            {label}
                                          </span>
                                        </li>
                                      ))}
                                    </ul>
                                  )}
                                </div>
                              }
                            >
                              <div
                                className={`auth-group-manage-row${localGroup ? '' : ' is-synced'}${inlineEditId === group.id ? ' is-editing' : ''}${selectedGroupId === group.id ? ' is-selected' : ''}`}
                                data-group-id={group.id}
                                aria-current={selectedGroupId === group.id ? 'true' : undefined}
                                onClick={(event) => {
                                  if (
                                    (event.target as HTMLElement).closest(
                                      'button,input,select,textarea,a',
                                    )
                                  )
                                    return;
                                  if (!protectionPanelBusy) selectGroup(group);
                                }}
                              >
                                <div className="auth-group-row-main">
                                  <div className="auth-group-row-title-wrap">
                                    {inlineEditId === group.id &&
                                    inlineEditField === 'display_name' ? (
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
                                        onClick={
                                          localGroup && !protectionPanelBusy
                                            ? () => handleStartInlineEdit(group, 'display_name')
                                            : undefined
                                        }
                                      >
                                        <div className="auth-group-row-title">
                                          {group.display_name}
                                        </div>
                                        <span className="auth-group-row-title-member-count">
                                          {group.member_count} member
                                          {group.member_count === 1 ? '' : 's'}
                                        </span>
                                        {localGroup && (
                                          <button
                                            type="button"
                                            className="inline-edit-btn"
                                            onClick={(event) => {
                                              event.stopPropagation();
                                              handleStartInlineEdit(group, 'display_name');
                                            }}
                                            disabled={protectionPanelBusy}
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
                                  <div className="auth-group-protection-summary">
                                    {accessLevelSummary}
                                  </div>
                                  {inlineEditId === group.id &&
                                  inlineEditField === 'description' ? (
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
                                      onClick={
                                        localGroup && !protectionPanelBusy
                                          ? () => handleStartInlineEdit(group, 'description')
                                          : undefined
                                      }
                                    >
                                      {group.description || group.source_dn ? (
                                        <div
                                          className="auth-group-row-description"
                                          title={group.source_dn || group.description}
                                        >
                                          {group.description || group.source_dn}
                                        </div>
                                      ) : (
                                        localGroup && (
                                          <div className="auth-group-row-description auth-group-row-description-placeholder">
                                            Add description
                                          </div>
                                        )
                                      )}
                                      {localGroup && (
                                        <button
                                          type="button"
                                          className="inline-edit-btn"
                                          onClick={(event) => {
                                            event.stopPropagation();
                                            handleStartInlineEdit(group, 'description');
                                          }}
                                          disabled={protectionPanelBusy}
                                          title="Edit group description"
                                          aria-label={`Edit ${group.display_name} description`}
                                        >
                                          <Pencil size={12} />
                                        </button>
                                      )}
                                    </div>
                                  )}
                                </div>
                                <button
                                  ref={
                                    selectedGroupId === group.id ? protectionTriggerRef : undefined
                                  }
                                  type="button"
                                  className="btn btn-sm btn-secondary group-protection-button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    selectGroup(group, event.currentTarget);
                                  }}
                                  aria-label={`Edit protection for ${group.display_name}`}
                                  aria-pressed={selectedGroupId === group.id}
                                  disabled={protectionPanelBusy}
                                >
                                  <ShieldCheck size={14} aria-hidden="true" />
                                  Access
                                </button>
                                <div className="auth-group-row-side">
                                  <div className="auth-group-row-status-pills">
                                    <span
                                      className={`auth-group-provider-badge auth-group-provider-${group.provider}`}
                                    >
                                      {getAuthGroupProviderLabel(group)}
                                    </span>
                                    {group.role && (
                                      <span
                                        className={`auth-group-role-badge${group.role === 'admin' ? ' auth-group-admin-badge' : ''}`}
                                      >
                                        {group.role}
                                      </span>
                                    )}
                                    {group.is_logon_group && (
                                      <span className="auth-group-logon-badge">Logon</span>
                                    )}
                                    {localGroup && (
                                      <div className="auth-group-row-status-actions">
                                        <DeleteConfirmButton
                                          onDelete={() => {
                                            void handleDeleteAuthGroup(group);
                                          }}
                                          disabled={busy || protectionPanelBusy}
                                          deleting={authGroupDeletingId === group.id}
                                          className="btn btn-sm btn-danger auth-group-delete-button"
                                          title="Delete group"
                                          buttonText="Delete"
                                        />
                                      </div>
                                    )}
                                  </div>
                                  <div className="auth-group-row-actions">
                                    <div
                                      className="auth-group-access-segment"
                                      role="group"
                                      aria-label={`Access mode for ${group.display_name}`}
                                    >
                                      <button
                                        type="button"
                                        className={`auth-group-access-option${accessMode === 'none' ? ' is-active' : ''}`}
                                        disabled={busy || protectionPanelBusy}
                                        onClick={() =>
                                          void handleSetAuthGroupAccessMode(group, 'none')
                                        }
                                        title="No role grant and not a logon group"
                                      >
                                        None
                                      </button>
                                      <button
                                        type="button"
                                        className={`auth-group-access-option${accessMode === 'logon' ? ' is-active' : ''}`}
                                        disabled={busy || protectionPanelBusy}
                                        onClick={() =>
                                          void handleSetAuthGroupAccessMode(group, 'logon')
                                        }
                                        title="Allow members of this group to log in"
                                      >
                                        Logon
                                      </button>
                                      <button
                                        type="button"
                                        className={`auth-group-access-option${accessMode === 'admin' ? ' is-active' : ''}`}
                                        disabled={busy || protectionPanelBusy}
                                        onClick={() =>
                                          void handleSetAuthGroupAccessMode(group, 'admin')
                                        }
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
              {selectedGroupId && protectedGroup && (
                <div className="access-md-detail" data-group-detail>
                  <AccessDetailHeader
                    headingId={`group-detail-heading-${protectedGroup.id}`}
                    title={protectedGroup.display_name}
                    meta={
                      <>
                        <span
                          className={`auth-group-provider-badge auth-group-provider-${protectedGroup.provider}`}
                        >
                          {getAuthGroupProviderLabel(protectedGroup)}
                        </span>
                        <span className="auth-group-row-title-member-count">
                          {protectedGroup.member_count} member
                          {protectedGroup.member_count === 1 ? '' : 's'}
                        </span>
                        {!isLocalManagedAuthGroup(protectedGroup) && protectedGroup.source_dn && (
                          <span className="auth-group-detail-dn" title={protectedGroup.source_dn}>
                            {protectedGroup.source_dn}
                          </span>
                        )}
                      </>
                    }
                    onClose={() => requestPanelNavigation(closeProtectionPanel)}
                  />
                  <div
                    className="access-md-detail-body"
                    data-group-detail-body
                    role="region"
                    aria-labelledby={`group-detail-heading-${protectedGroup.id}`}
                  >
                    {renderGroupDetailControls(protectedGroup)}
                    {protectedGroupUnavailable && (
                      <div
                        className="error-banner"
                        role="alert"
                        data-group-protection-unavailable={protectedGroup.id}
                      >
                        {UNAVAILABLE_GROUP_MESSAGE}
                      </div>
                    )}
                    {contentProtectionConfig ? (
                      <GroupProtectionPanel
                        key={protectedGroup.id}
                        ref={protectionPanelRef}
                        hideHeader
                        group={protectedGroup}
                        config={contentProtectionConfig}
                        authGroups={authGroups}
                        onConfigSaved={setContentProtectionConfig}
                        onUpdateMapping={
                          protectedGroupUnavailable
                            ? async () => toast.error(UNAVAILABLE_GROUP_MESSAGE)
                            : (levelId, enabled) =>
                                handleContentProtectionAccessLevelChange(
                                  protectedGroup,
                                  levelId,
                                  enabled,
                                )
                        }
                        onUpdateRequirement={
                          protectedGroupUnavailable
                            ? async () => toast.error(UNAVAILABLE_GROUP_MESSAGE)
                            : (mode) =>
                                handleContentProtectionRequirementChange(protectedGroup, mode)
                        }
                        onBack={() => requestPanelNavigation(closeProtectionPanel)}
                        onNavigateToSetting={onNavigateToSetting ? navigateToSetting : undefined}
                        onBusyChange={setProtectionPanelBusy}
                        toast={toast}
                      />
                    ) : (
                      <section
                        className="group-protection-panel"
                        role="alert"
                        data-group-protection-config-error
                      >
                        {contentProtectionError || 'Loading protection config…'}
                        {contentProtectionError && (
                          <button
                            type="button"
                            className="btn btn-sm btn-secondary"
                            onClick={() => void loadProtectionConfig()}
                          >
                            Retry
                          </button>
                        )}
                      </section>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
