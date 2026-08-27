import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';

import { ArrowLeft, Check, Pencil, Plus, Trash2, X } from 'lucide-react';

import type {
  ConversationShareAccessMode,
  ConversationShareLinkStatus,
  UserDirectoryEntry,
  UserSpaceShareAccessMode,
  UserSpaceWorkspaceShareLinkStatus,
} from '@/types';

type ShareAccessMode = UserSpaceShareAccessMode | ConversationShareAccessMode;
type ShareStatus = UserSpaceWorkspaceShareLinkStatus | ConversationShareLinkStatus;

import { LdapGroupChips, LdapGroupSelect, type LdapGroup } from '../LdapGroupSelect';
import { InlineCopyButton } from './InlineCopyButton';
import type { ShareLinkStyle } from '@/types';

interface ShareLinkModalProps {
  isOpen: boolean;
  loadingShareStatus: boolean;
  title?: string;
  shareLinkType: ShareLinkStyle;
  shareStatus: ShareStatus | null;
  shareLinks?: ShareStatus[];
  selectedShareId?: string | null;
  shareSlugDraft: string;
  shareSlugAvailable: boolean | null;
  shareAccessMode: ShareAccessMode;
  sharePasswordDraft: string;
  shareSelectableUsers: UserDirectoryEntry[];
  shareSelectedUserIdsDraft: string[];
  shareSelectedLdapGroupsDraft: string[];
  shareLdapGroupDraft: string;
  ldapDiscoveredGroups: LdapGroup[];
  loadingLdapGroups: boolean;
  shareSubdomainEnabled: boolean;
  shareSubdomainDisabledReason: string | null;
  showProtectedSubdomainNotice: boolean;
  effectiveShareUrl: string | null;
  activeShareCreatedLabel: string;
  savingShareAccess: boolean;
  sharingWorkspace: boolean;
  checkingShareSlug: boolean;
  shareHasUnsavedChanges: boolean;
  creatingShareLink?: boolean;
  updatingShareLabel?: boolean;
  deletingSelectedShareLink?: boolean;
  allowSubdomainOption?: boolean;
  shareTargetLabel?: string;
  openActionLabel?: string;
  extraAccessControls?: ReactNode;
  agentAccessSection?: ReactNode;
  onClose: () => void;
  onSelectShare?: (shareId: string) => void;
  onCreateShareLink?: () => void;
  onSaveShareLabel?: (label: string) => void;
  onDeleteSelectedShareLink?: (shareId: string) => void;
  onShareSlugChange: (value: string) => void;
  onShareAccessModeChange: (value: ShareAccessMode) => void;
  onSharePasswordDraftChange: (value: string) => void;
  onToggleShareSelectedUser: (userId: string) => void;
  onShareLdapGroupDraftChange: (value: string) => void;
  onAddShareLdapGroup: () => void;
  onRemoveShareLdapGroup: (groupDn: string) => void;
  onSaveShareAccess: () => void;
  onOpenFullPreview: () => void;
  onShareUrlInlineCopySuccess?: () => void;
  onShareUrlInlineCopyError?: (error: Error) => void;
  formatUserLabel: (user: UserDirectoryEntry, fallback: string) => string;
}

const SHARE_ACCESS_MODE_LABELS: Record<ShareAccessMode, string> = {
  token: 'Public link',
  password: 'Password',
  authenticated_users: 'Authenticated',
  selected_users: 'Selected users',
  ldap_groups: 'LDAP groups',
};

function getShareDisplayLabel(link: ShareStatus, index: number): string {
  return link.label?.trim() || `Untitled link ${index + 1}`;
}

function getScopeSummary(link: ShareStatus): string | null {
  if ('scope_anchor_message_idx' in link && typeof link.scope_anchor_message_idx === 'number') {
    return link.scope_direction === 'backward'
      ? `Shared up to message #${link.scope_anchor_message_idx + 1}`
      : `Shared from message #${link.scope_anchor_message_idx + 1}`;
  }
  return null;
}

function getShareLinkUrl(link: ShareStatus): string | null {
  if (
    link.active_share_style === 'subdomain' &&
    'subdomain_share_enabled' in link &&
    'subdomain_share_url' in link &&
    link.subdomain_share_enabled &&
    link.subdomain_share_url
  ) {
    return link.subdomain_share_url;
  }
  if (link.active_share_style === 'anonymous' && link.anonymous_share_url) {
    return link.anonymous_share_url;
  }
  return link.share_url;
}

export function ShareLinkModal({
  isOpen,
  loadingShareStatus,
  title = 'Share Workspace',
  shareLinkType,
  shareStatus,
  shareLinks = [],
  selectedShareId = null,
  shareSlugDraft,
  shareSlugAvailable,
  shareAccessMode,
  sharePasswordDraft,
  shareSelectableUsers,
  shareSelectedUserIdsDraft,
  shareSelectedLdapGroupsDraft,
  shareLdapGroupDraft,
  ldapDiscoveredGroups,
  loadingLdapGroups,
  shareSubdomainEnabled,
  shareSubdomainDisabledReason,
  showProtectedSubdomainNotice,
  effectiveShareUrl,
  activeShareCreatedLabel,
  savingShareAccess,
  sharingWorkspace,
  checkingShareSlug,
  shareHasUnsavedChanges,
  creatingShareLink = false,
  updatingShareLabel = false,
  deletingSelectedShareLink = false,
  allowSubdomainOption = true,
  shareTargetLabel = 'workspace',
  openActionLabel = 'Open Preview',
  extraAccessControls,
  agentAccessSection,
  onClose,
  onSelectShare,
  onCreateShareLink,
  onSaveShareLabel,
  onDeleteSelectedShareLink,
  onShareSlugChange,
  onShareAccessModeChange,
  onSharePasswordDraftChange,
  onToggleShareSelectedUser,
  onShareLdapGroupDraftChange,
  onAddShareLdapGroup,
  onRemoveShareLdapGroup,
  onSaveShareAccess,
  onOpenFullPreview,
  onShareUrlInlineCopySuccess,
  onShareUrlInlineCopyError,
  formatUserLabel,
}: ShareLinkModalProps) {
  const [view, setView] = useState<'list' | 'edit'>('list');
  const [deleteConfirmShareId, setDeleteConfirmShareId] = useState<string | null>(null);
  const [shareLabelDraft, setShareLabelDraft] = useState('');
  const [isEditingShareTitle, setIsEditingShareTitle] = useState(false);
  const pendingEditAfterCreateRef = useRef(false);
  const previousSelectedShareIdRef = useRef<string | null>(null);
  const previousCreatingShareLinkRef = useRef(creatingShareLink);

  const availableShareLinks = useMemo(
    () => (shareLinks.length > 0 ? shareLinks : shareStatus?.id ? [shareStatus] : []),
    [shareLinks, shareStatus],
  );

  const selectedShare = useMemo(() => {
    if (!selectedShareId) {
      return shareStatus;
    }
    return availableShareLinks.find((link) => link.id === selectedShareId) ?? shareStatus;
  }, [availableShareLinks, selectedShareId, shareStatus]);

  useEffect(() => {
    if (!isOpen) {
      setView('list');
      setDeleteConfirmShareId(null);
      setShareLabelDraft('');
      setIsEditingShareTitle(false);
      pendingEditAfterCreateRef.current = false;
      previousSelectedShareIdRef.current = null;
    }
  }, [isOpen]);

  useEffect(() => {
    if (
      deleteConfirmShareId &&
      !availableShareLinks.some((link) => link.id === deleteConfirmShareId)
    ) {
      setDeleteConfirmShareId(null);
    }
    if (view === 'edit' && !selectedShare) {
      setView('list');
    }
  }, [availableShareLinks, deleteConfirmShareId, selectedShare, view]);

  useEffect(() => {
    setShareLabelDraft(selectedShare?.label?.trim() || '');
    setIsEditingShareTitle(false);
  }, [selectedShare?.id, selectedShare?.label]);

  useEffect(() => {
    const wasCreatingShareLink = previousCreatingShareLinkRef.current;
    if (wasCreatingShareLink && !creatingShareLink && pendingEditAfterCreateRef.current) {
      if (
        selectedShare?.id &&
        (selectedShare.id !== previousSelectedShareIdRef.current ||
          !previousSelectedShareIdRef.current)
      ) {
        setView('edit');
      }
      pendingEditAfterCreateRef.current = false;
      previousSelectedShareIdRef.current = null;
    }
    previousCreatingShareLinkRef.current = creatingShareLink;
  }, [creatingShareLink, selectedShare?.id]);

  if (!isOpen) {
    return null;
  }

  const selectedShareLabel = selectedShare?.label?.trim() || '';
  const selectedShareIndex = selectedShare?.id
    ? availableShareLinks.findIndex((link) => link.id === selectedShare.id)
    : -1;
  const selectedShareDisplayLabel =
    selectedShareLabel ||
    (selectedShareIndex >= 0 ? `Untitled link ${selectedShareIndex + 1}` : 'Untitled link');
  const selectedShareScopeSummary = selectedShare ? getScopeSummary(selectedShare) : null;
  const canSaveShareLabel = shareLabelDraft.trim() !== (selectedShare?.label?.trim() || '');
  const listBusy = creatingShareLink || deletingSelectedShareLink || updatingShareLabel;
  const isEditView = view === 'edit' && Boolean(selectedShare);
  const linkIdentityLocked = Boolean(selectedShare?.has_share_link);

  const handleSaveShareLabel = () => {
    if (!canSaveShareLabel || updatingShareLabel) {
      setIsEditingShareTitle(false);
      return;
    }
    setIsEditingShareTitle(false);
    onSaveShareLabel?.(shareLabelDraft.trim());
  };

  const handleShareLabelKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      handleSaveShareLabel();
      event.currentTarget.blur();
    }
    if (event.key === 'Escape') {
      event.preventDefault();
      setShareLabelDraft(selectedShare?.label?.trim() || '');
      setIsEditingShareTitle(false);
    }
  };

  const handleBackToList = () => {
    if (selectedShare?.id && shareHasUnsavedChanges) {
      onSelectShare?.(selectedShare.id);
    }
    setView('list');
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content userspace-share-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div className="userspace-share-modal-title">
            {isEditView && (
              <button
                type="button"
                className="userspace-share-header-back-btn"
                onClick={handleBackToList}
                title="Back to all links"
                aria-label="Back to all links"
              >
                <ArrowLeft size={14} />
              </button>
            )}
            <h3>{title}</h3>
          </div>
          <button className="modal-close" onClick={onClose}>
            &times;
          </button>
        </div>
        <div className="modal-body">
          {loadingShareStatus ? (
            <p className="userspace-muted">Loading share settings...</p>
          ) : isEditView && selectedShare ? (
            <>
              <div className="userspace-share-link-pane">
                <div className="userspace-share-edit-header">
                  <div className="userspace-share-edit-heading">
                    {isEditingShareTitle ? (
                      <input
                        className="userspace-share-title-input"
                        type="text"
                        value={shareLabelDraft}
                        onChange={(event) => setShareLabelDraft(event.target.value)}
                        onKeyDown={handleShareLabelKeyDown}
                        onBlur={handleSaveShareLabel}
                        placeholder="Untitled link"
                        disabled={updatingShareLabel}
                        autoFocus
                      />
                    ) : (
                      <button
                        type="button"
                        className="userspace-share-edit-title"
                        onClick={() => setIsEditingShareTitle(true)}
                        title="Edit label"
                      >
                        <span>{selectedShareDisplayLabel}</span>
                        <Pencil size={13} />
                      </button>
                    )}
                    <span className="userspace-share-meta">
                      {activeShareCreatedLabel}
                      {selectedShareScopeSummary ? ` · ${selectedShareScopeSummary}` : ''}
                    </span>
                  </div>
                </div>

                <div className="userspace-share-edit-grid userspace-share-edit-grid-single">
                  <div className="userspace-share-access-row">
                    <label htmlFor="userspace-share-access-mode" className="userspace-share-label">
                      Access mode
                    </label>
                    <select
                      id="userspace-share-access-mode"
                      value={shareAccessMode}
                      onChange={(event) =>
                        onShareAccessModeChange(event.target.value as ShareAccessMode)
                      }
                      disabled={savingShareAccess || sharingWorkspace}
                    >
                      <option value="token">Tokenized public link</option>
                      <option value="password">Password protected</option>
                      <option value="authenticated_users">Any authenticated user</option>
                      <option value="selected_users">Selected users only</option>
                      <option value="ldap_groups">Selected LDAP groups only</option>
                    </select>
                  </div>
                </div>

                {(!allowSubdomainOption || shareLinkType !== 'subdomain') && (
                  <div className="userspace-share-access-row">
                    <label htmlFor="userspace-share-slug" className="userspace-share-label">
                      Custom slug
                    </label>
                    <div className="userspace-share-slug-row">
                      <input
                        id="userspace-share-slug"
                        value={shareSlugDraft}
                        onChange={(event) => onShareSlugChange(event.target.value)}
                        placeholder="custom_slug"
                        autoComplete="off"
                        disabled={linkIdentityLocked}
                      />
                    </div>
                    {shareSlugAvailable !== null && (
                      <div
                        className={`userspace-share-meta ${shareSlugAvailable ? '' : 'userspace-error'}`}
                      >
                        {shareSlugAvailable ? 'Slug is available' : 'Slug is unavailable'}
                      </div>
                    )}
                  </div>
                )}

                {linkIdentityLocked && (
                  <div className="userspace-share-meta">
                    Link URL and style are locked after creation. Delete and recreate this link to
                    change them.
                  </div>
                )}

                {selectedShare.has_share_link &&
                  allowSubdomainOption &&
                  !shareSubdomainEnabled &&
                  shareSubdomainDisabledReason && (
                    <div className="userspace-share-meta">{shareSubdomainDisabledReason}</div>
                  )}

                {showProtectedSubdomainNotice && (
                  <div className="userspace-share-warning-banner" role="alert">
                    Warning: if this workspace has already been unlocked in this browser, opening
                    the subdomain link again may not prompt you to login. Protection is still
                    enforced for new sessions and other browsers.
                  </div>
                )}

                {selectedShare.has_share_link && effectiveShareUrl ? (
                  <>
                    <label htmlFor="userspace-share-url" className="userspace-share-label">
                      Active share URL
                    </label>
                    <div className="userspace-share-url-copy-wrap">
                      <input id="userspace-share-url" value={effectiveShareUrl} readOnly />
                      <InlineCopyButton
                        copyText={effectiveShareUrl}
                        className="userspace-share-inline-copy"
                        title="Copy share URL"
                        ariaLabel="Copy share URL"
                        copiedTitle="Share URL copied"
                        copiedAriaLabel="Share URL copied"
                        iconSize={12}
                        onCopySuccess={onShareUrlInlineCopySuccess}
                        onCopyError={onShareUrlInlineCopyError}
                      />
                    </div>
                  </>
                ) : (
                  <p className="userspace-muted">
                    No active share link for this {shareTargetLabel}.
                  </p>
                )}
              </div>

              <div className="userspace-share-controls">
                {shareAccessMode === 'password' && (
                  <div className="userspace-share-access-row">
                    <label htmlFor="userspace-share-password" className="userspace-share-label">
                      Share password {selectedShare.has_password ? '(set)' : '(required)'}
                    </label>
                    <input
                      id="userspace-share-password"
                      type="password"
                      value={sharePasswordDraft}
                      onChange={(event) => onSharePasswordDraftChange(event.target.value)}
                      placeholder={
                        selectedShare.has_password
                          ? 'Enter new password to update'
                          : 'Enter password'
                      }
                      autoComplete="new-password"
                    />
                  </div>
                )}

                {shareAccessMode === 'selected_users' && (
                  <div className="userspace-share-access-row">
                    <label className="userspace-share-label">Allowed users</label>
                    <div className="userspace-share-user-grid">
                      {shareSelectableUsers.map((user) => (
                        <label key={user.id} className="userspace-share-user-option">
                          <input
                            type="checkbox"
                            checked={shareSelectedUserIdsDraft.includes(user.id)}
                            onChange={() => onToggleShareSelectedUser(user.id)}
                          />
                          <span>{formatUserLabel(user, user.id)}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                )}

                {shareAccessMode === 'ldap_groups' && (
                  <div className="userspace-share-access-row">
                    <label className="userspace-share-label">Allowed LDAP groups</label>
                    <div className="userspace-share-slug-row">
                      {ldapDiscoveredGroups.length > 0 ? (
                        <LdapGroupSelect
                          value={shareLdapGroupDraft}
                          onChange={onShareLdapGroupDraftChange}
                          groups={ldapDiscoveredGroups}
                          excludedDns={shareSelectedLdapGroupsDraft}
                          emptyOptionLabel="Select an LDAP group..."
                        />
                      ) : (
                        <input
                          value={shareLdapGroupDraft}
                          onChange={(event) => onShareLdapGroupDraftChange(event.target.value)}
                          placeholder="cn=group,ou=groups,dc=example,dc=com"
                          autoComplete="off"
                        />
                      )}
                      <button
                        className="btn btn-secondary"
                        onClick={onAddShareLdapGroup}
                        type="button"
                      >
                        Add Group
                      </button>
                    </div>
                    {loadingLdapGroups ? (
                      <p className="userspace-share-meta">Loading LDAP groups…</p>
                    ) : ldapDiscoveredGroups.length > 0 ? (
                      <p className="userspace-share-meta">
                        Groups are discovered from the configured LDAP base domain.
                      </p>
                    ) : (
                      <p className="userspace-share-meta">
                        Could not auto-discover LDAP groups. Enter group DN manually.
                      </p>
                    )}
                    <LdapGroupChips
                      selectedDns={shareSelectedLdapGroupsDraft}
                      groups={ldapDiscoveredGroups}
                      onRemove={onRemoveShareLdapGroup}
                    />
                  </div>
                )}

                {extraAccessControls}
              </div>
            </>
          ) : (
            <>
              <div className="userspace-share-link-pane">
                <div className="jobs-table-wrapper userspace-share-links-table-wrapper">
                  <table className="jobs-table userspace-share-links-table">
                    <thead>
                      <tr>
                        <th>Label</th>
                        <th>Access</th>
                        <th>Click Count</th>
                        <th>URL</th>
                        <th>Created</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {availableShareLinks.length === 0 ? (
                        <tr>
                          <td colSpan={6} className="userspace-share-links-empty-cell">
                            No share links yet for this {shareTargetLabel}.
                          </td>
                        </tr>
                      ) : (
                        availableShareLinks.map((link, index) => {
                          const isConfirmingDelete = deleteConfirmShareId === link.id;
                          const scopeSummary = getScopeSummary(link);
                          const shareUrl = getShareLinkUrl(link);

                          return (
                            <tr key={link.id}>
                              <td>
                                <div
                                  className="userspace-share-link-cell-primary"
                                  title={getShareDisplayLabel(link, index)}
                                >
                                  {getShareDisplayLabel(link, index)}
                                </div>
                                {scopeSummary && (
                                  <div className="userspace-share-link-cell-secondary">
                                    {scopeSummary}
                                  </div>
                                )}
                              </td>
                              <td>
                                <span className="userspace-share-access-badge">
                                  {SHARE_ACCESS_MODE_LABELS[link.share_access_mode]}
                                </span>
                              </td>
                              <td>{link.public_hit_count ?? 0}</td>
                              <td>
                                {shareUrl ? (
                                  <span title={shareUrl} className="userspace-share-table-url-text">
                                    {shareUrl}
                                  </span>
                                ) : (
                                  <span className="userspace-share-link-cell-secondary">-</span>
                                )}
                              </td>
                              <td>
                                {link.created_at
                                  ? new Date(link.created_at).toLocaleDateString()
                                  : '-'}
                              </td>
                              <td>
                                <div className="actions-cell userspace-share-table-actions">
                                  {isConfirmingDelete ? (
                                    <>
                                      <button
                                        type="button"
                                        className="action-btn action-btn-confirm"
                                        onClick={() => onDeleteSelectedShareLink?.(link.id)}
                                        disabled={deletingSelectedShareLink}
                                        title="Confirm delete"
                                        aria-label="Confirm delete"
                                      >
                                        <Check size={12} />
                                      </button>
                                      <button
                                        type="button"
                                        className="action-btn userspace-share-table-action"
                                        onClick={() => setDeleteConfirmShareId(null)}
                                        disabled={deletingSelectedShareLink}
                                        title="Cancel delete"
                                        aria-label="Cancel delete"
                                      >
                                        <X size={12} />
                                      </button>
                                    </>
                                  ) : (
                                    <>
                                      {shareUrl && (
                                        <InlineCopyButton
                                          copyText={shareUrl}
                                          className="action-btn userspace-share-copy-action"
                                          title="Copy share URL"
                                          ariaLabel="Copy share URL"
                                          copiedTitle="Share URL copied"
                                          copiedAriaLabel="Share URL copied"
                                          iconSize={12}
                                          onCopySuccess={onShareUrlInlineCopySuccess}
                                          onCopyError={onShareUrlInlineCopyError}
                                        />
                                      )}
                                      <button
                                        type="button"
                                        className="action-btn userspace-share-table-action"
                                        onClick={() => {
                                          onSelectShare?.(link.id);
                                          setDeleteConfirmShareId(null);
                                          setView('edit');
                                        }}
                                        disabled={listBusy}
                                        title="Edit link"
                                        aria-label="Edit link"
                                      >
                                        <Pencil size={12} />
                                      </button>
                                      <button
                                        type="button"
                                        className="action-btn userspace-share-table-action"
                                        onClick={() => {
                                          onSelectShare?.(link.id);
                                          setDeleteConfirmShareId(link.id);
                                        }}
                                        disabled={listBusy}
                                        title="Delete link"
                                        aria-label="Delete link"
                                      >
                                        <Trash2 size={12} />
                                      </button>
                                    </>
                                  )}
                                </div>
                              </td>
                            </tr>
                          );
                        })
                      )}
                    </tbody>
                  </table>
                </div>
              </div>
              <div className="userspace-share-actions userspace-share-actions-single">
                <button
                  type="button"
                  className="btn btn-secondary"
                  onClick={() => {
                    pendingEditAfterCreateRef.current = true;
                    previousSelectedShareIdRef.current = selectedShareId;
                    onCreateShareLink?.();
                  }}
                  disabled={listBusy || loadingShareStatus}
                >
                  <Plus size={14} />
                  <span>{creatingShareLink ? 'Creating...' : 'New Link'}</span>
                </button>
              </div>
              {agentAccessSection}
            </>
          )}
        </div>
        {isEditView && (
          <div className="modal-footer userspace-share-modal-footer">
            <div className="userspace-share-actions userspace-share-actions-edit">
              <button
                className="btn btn-secondary"
                onClick={onSaveShareAccess}
                disabled={
                  loadingShareStatus ||
                  savingShareAccess ||
                  sharingWorkspace ||
                  checkingShareSlug ||
                  (Boolean(selectedShare?.has_share_link) && !shareHasUnsavedChanges)
                }
              >
                {savingShareAccess ? 'Saving Access...' : 'Save Access'}
              </button>
              <button
                className="btn btn-secondary"
                onClick={onOpenFullPreview}
                disabled={
                  loadingShareStatus ||
                  sharingWorkspace ||
                  checkingShareSlug ||
                  savingShareAccess ||
                  shareHasUnsavedChanges
                }
              >
                {openActionLabel}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
