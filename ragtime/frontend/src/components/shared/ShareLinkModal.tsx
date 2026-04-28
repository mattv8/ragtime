import { useEffect, useMemo, useState, type ReactNode } from 'react';

import { Check, Pencil, Plus, Trash2, X } from 'lucide-react';

import type {
  ConversationShareAccessMode,
  ConversationShareLinkStatus,
  User,
  UserSpaceShareAccessMode,
  UserSpaceWorkspaceShareLinkStatus,
} from '@/types';

type ShareAccessMode = UserSpaceShareAccessMode | ConversationShareAccessMode;
type ShareStatus = UserSpaceWorkspaceShareLinkStatus | ConversationShareLinkStatus;

import { LdapGroupSelect } from '../LdapGroupSelect';
import { InlineCopyButton } from './InlineCopyButton';

interface ShareLinkModalProps {
  isOpen: boolean;
  loadingShareStatus: boolean;
  title?: string;
  shareLinkType: 'named' | 'anonymous' | 'subdomain';
  shareStatus: ShareStatus | null;
  shareLinks?: ShareStatus[];
  selectedShareId?: string | null;
  shareSlugDraft: string;
  shareSlugAvailable: boolean | null;
  shareAccessMode: ShareAccessMode;
  sharePasswordDraft: string;
  shareSelectableUsers: User[];
  shareSelectedUserIdsDraft: string[];
  shareSelectedLdapGroupsDraft: string[];
  shareLdapGroupDraft: string;
  ldapDiscoveredGroups: Array<{ dn: string; name: string }>;
  loadingLdapGroups: boolean;
  shareSubdomainEnabled: boolean;
  shareSubdomainDisabledReason: string | null;
  showProtectedSubdomainNotice: boolean;
  effectiveShareUrl: string | null;
  activeShareCreatedLabel: string;
  savingShareAccess: boolean;
  sharingWorkspace: boolean;
  revokingShareLink: boolean;
  rotatingShareLink: boolean;
  checkingShareSlug: boolean;
  shareHasUnsavedChanges: boolean;
  shareCopied: boolean;
  creatingShareLink?: boolean;
  updatingShareLabel?: boolean;
  deletingSelectedShareLink?: boolean;
  allowSubdomainOption?: boolean;
  shareTargetLabel?: string;
  openActionLabel?: string;
  extraAccessControls?: ReactNode;
  onClose: () => void;
  onSelectShare?: (shareId: string) => void;
  onCreateShareLink?: () => void;
  onSaveShareLabel?: (label: string) => void;
  onDeleteSelectedShareLink?: (shareId: string) => void;
  onShareSlugChange: (value: string) => void;
  onShareLinkTypeChange: (value: 'named' | 'anonymous' | 'subdomain') => void;
  onShareAccessModeChange: (value: ShareAccessMode) => void;
  onSharePasswordDraftChange: (value: string) => void;
  onToggleShareSelectedUser: (userId: string) => void;
  onShareLdapGroupDraftChange: (value: string) => void;
  onAddShareLdapGroup: () => void;
  onRemoveShareLdapGroup: (groupDn: string) => void;
  onSaveShareAccess: () => void;
  onCopyShareLink: () => void;
  onOpenFullPreview: () => void;
  onRotateShareLink: () => void;
  onRevokeShareLink: () => void;
  onShareUrlInlineCopySuccess?: () => void;
  onShareUrlInlineCopyError?: (error: Error) => void;
  formatUserLabel: (user: User, fallback: string) => string;
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
  revokingShareLink,
  rotatingShareLink,
  checkingShareSlug,
  shareHasUnsavedChanges,
  shareCopied,
  creatingShareLink = false,
  updatingShareLabel = false,
  deletingSelectedShareLink = false,
  allowSubdomainOption = true,
  shareTargetLabel = 'workspace',
  openActionLabel = 'Open Preview',
  extraAccessControls,
  onClose,
  onSelectShare,
  onCreateShareLink,
  onSaveShareLabel,
  onDeleteSelectedShareLink,
  onShareSlugChange,
  onShareLinkTypeChange,
  onShareAccessModeChange,
  onSharePasswordDraftChange,
  onToggleShareSelectedUser,
  onShareLdapGroupDraftChange,
  onAddShareLdapGroup,
  onRemoveShareLdapGroup,
  onSaveShareAccess,
  onCopyShareLink,
  onOpenFullPreview,
  onRotateShareLink,
  onRevokeShareLink,
  onShareUrlInlineCopySuccess,
  onShareUrlInlineCopyError,
  formatUserLabel,
}: ShareLinkModalProps) {
  const [isShareMenuOpen, setIsShareMenuOpen] = useState(false);
  const [editingShareId, setEditingShareId] = useState<string | null>(null);
  const [deleteConfirmShareId, setDeleteConfirmShareId] = useState<string | null>(null);
  const [shareLabelDraft, setShareLabelDraft] = useState('');

  const availableShareLinks = shareLinks.length > 0
    ? shareLinks
    : (shareStatus?.id ? [shareStatus] : []);

  const selectedShare = useMemo(() => {
    if (!selectedShareId) {
      return shareStatus;
    }
    return availableShareLinks.find((link) => link.id === selectedShareId) ?? shareStatus;
  }, [availableShareLinks, selectedShareId, shareStatus]);

  useEffect(() => {
    if (!isOpen) {
      setIsShareMenuOpen(false);
      setEditingShareId(null);
      setDeleteConfirmShareId(null);
      setShareLabelDraft('');
    }
  }, [isOpen]);

  useEffect(() => {
    if (editingShareId && !availableShareLinks.some((link) => link.id === editingShareId)) {
      setEditingShareId(null);
      setShareLabelDraft('');
    }
    if (deleteConfirmShareId && !availableShareLinks.some((link) => link.id === deleteConfirmShareId)) {
      setDeleteConfirmShareId(null);
    }
  }, [availableShareLinks, deleteConfirmShareId, editingShareId]);

  if (!isOpen) {
    return null;
  }

  const selectedShareLabel = selectedShare?.label?.trim() || '';
  const selectedShareDisplayLabel = selectedShareLabel || 'Untitled link';
  const selectedShareScopeSummary = selectedShare && 'scope_anchor_message_idx' in selectedShare
    && typeof selectedShare.scope_anchor_message_idx === 'number'
    ? selectedShare.scope_direction === 'backward'
      ? `Shared up to message #${selectedShare.scope_anchor_message_idx + 1}`
      : `Shared from message #${selectedShare.scope_anchor_message_idx + 1}`
    : null;
  const revokeActionLabel = availableShareLinks.length > 1 ? 'Revoke All' : 'Revoke Link';

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content modal-small userspace-share-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{title}</h3>
          <button className="modal-close" onClick={onClose}>&times;</button>
        </div>
        <div className="modal-body">
          {loadingShareStatus ? (
            <p className="userspace-muted">Loading share settings...</p>
          ) : (
            <>
              <div className="userspace-share-link-pane">
                <div className="userspace-share-access-row">
                  <label htmlFor="userspace-share-link-trigger" className="userspace-share-label">Share links</label>
                  <div className="model-selector model-selector-full userspace-share-link-picker">
                    <button
                      id="userspace-share-link-trigger"
                      type="button"
                      className="model-selector-trigger model-selector-trigger-full userspace-share-link-trigger"
                      onClick={() => setIsShareMenuOpen((open) => !open)}
                      aria-haspopup="listbox"
                      aria-expanded={isShareMenuOpen}
                      disabled={creatingShareLink || updatingShareLabel || deletingSelectedShareLink}
                    >
                      <span className="model-selector-text">
                        {selectedShare?.id ? selectedShareDisplayLabel : 'No links yet'}
                      </span>
                      <span className="model-selector-arrow">▾</span>
                    </button>

                    {isShareMenuOpen && (
                      <div className="model-selector-dropdown userspace-share-link-dropdown">
                        <div className="model-selector-dropdown-inner" role="listbox" aria-label="Share links">
                          {availableShareLinks.length === 0 ? (
                            <div className="userspace-share-link-empty">No links yet</div>
                          ) : (
                            availableShareLinks.map((link, index) => {
                              const isSelected = link.id === selectedShareId;
                              const isEditing = editingShareId === link.id;
                              const isConfirmingDelete = deleteConfirmShareId === link.id;
                              const linkLabel = link.label?.trim() || `Untitled link ${index + 1}`;

                              return (
                                <div
                                  key={link.id}
                                  className={`model-selector-item userspace-share-link-item ${isSelected ? 'is-selected' : ''}`}
                                >
                                  {isEditing ? (
                                    <div className="userspace-share-link-inline-edit">
                                      <input
                                        type="text"
                                        className="userspace-share-link-rename-input"
                                        value={shareLabelDraft}
                                        onChange={(event) => setShareLabelDraft(event.target.value)}
                                        onKeyDown={(event) => {
                                          if (event.key === 'Enter') {
                                            event.preventDefault();
                                            onSaveShareLabel?.(shareLabelDraft.trim() || '');
                                          }
                                          if (event.key === 'Escape') {
                                            event.preventDefault();
                                            setEditingShareId(null);
                                            setShareLabelDraft(link.label?.trim() || '');
                                          }
                                        }}
                                        autoFocus
                                      />
                                    </div>
                                  ) : (
                                    <button
                                      type="button"
                                      role="option"
                                      aria-selected={isSelected}
                                      className="userspace-share-link-select-btn"
                                      onClick={() => {
                                        onSelectShare?.(link.id);
                                        setEditingShareId(null);
                                        setDeleteConfirmShareId(null);
                                        setIsShareMenuOpen(false);
                                      }}
                                    >
                                      <span className="model-selector-item-name">{linkLabel}</span>
                                    </button>
                                  )}

                                  <div className="userspace-item-actions" style={{ opacity: 1 }}>
                                    {isConfirmingDelete ? (
                                      <>
                                        <button
                                          className="chat-action-btn confirm-delete"
                                          onClick={() => onDeleteSelectedShareLink?.(link.id)}
                                          title="Confirm"
                                          disabled={deletingSelectedShareLink}
                                          type="button"
                                        >
                                          <Check size={12} />
                                        </button>
                                        <button
                                          className="chat-action-btn cancel-delete"
                                          onClick={() => setDeleteConfirmShareId(null)}
                                          title="Cancel"
                                          disabled={deletingSelectedShareLink}
                                          type="button"
                                        >
                                          <X size={12} />
                                        </button>
                                      </>
                                    ) : isEditing ? (
                                      <>
                                        <button
                                          className="chat-action-btn confirm-delete"
                                          onClick={() => onSaveShareLabel?.(shareLabelDraft.trim() || '')}
                                          title="Save label"
                                          disabled={updatingShareLabel}
                                          type="button"
                                        >
                                          <Check size={12} />
                                        </button>
                                        <button
                                          className="chat-action-btn cancel-delete"
                                          onClick={() => {
                                            setEditingShareId(null);
                                            setShareLabelDraft(link.label?.trim() || '');
                                          }}
                                          title="Cancel"
                                          disabled={updatingShareLabel}
                                          type="button"
                                        >
                                          <X size={12} />
                                        </button>
                                      </>
                                    ) : (
                                      <>
                                        <button
                                          className="chat-action-btn"
                                          onClick={() => {
                                            onSelectShare?.(link.id);
                                            setShareLabelDraft(link.label?.trim() || '');
                                            setEditingShareId(link.id);
                                            setDeleteConfirmShareId(null);
                                          }}
                                          title="Rename link"
                                          type="button"
                                        >
                                          <Pencil size={12} />
                                        </button>
                                        <button
                                          className="chat-action-btn"
                                          onClick={() => {
                                            onSelectShare?.(link.id);
                                            setDeleteConfirmShareId(link.id);
                                            setEditingShareId(null);
                                          }}
                                          title="Delete link"
                                          type="button"
                                        >
                                          <Trash2 size={12} />
                                        </button>
                                      </>
                                    )}
                                  </div>
                                </div>
                              );
                            })
                          )}
                        </div>
                        <button
                          className="userspace-share-link-create-btn"
                          onClick={() => {
                            setIsShareMenuOpen(false);
                            onCreateShareLink?.();
                          }}
                          disabled={creatingShareLink || updatingShareLabel || deletingSelectedShareLink}
                          type="button"
                        >
                          <Plus size={12} />
                          <span>{creatingShareLink ? 'Creating...' : 'New Link'}</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>

                {(!allowSubdomainOption || shareLinkType !== 'subdomain') && (
                  <>
                    <label htmlFor="userspace-share-slug" className="userspace-share-label">Custom slug</label>
                    <div className="userspace-share-slug-row">
                      <input
                        id="userspace-share-slug"
                        value={shareSlugDraft}
                        onChange={(event) => onShareSlugChange(event.target.value)}
                        placeholder="custom_slug"
                        autoComplete="off"
                      />
                    </div>
                    {shareSlugAvailable !== null && (
                      <div className={`userspace-share-meta ${shareSlugAvailable ? '' : 'userspace-error'}`}>
                        {shareSlugAvailable ? 'Slug is available' : 'Slug is unavailable'}
                      </div>
                    )}
                  </>
                )}

                {shareStatus?.has_share_link && (
                  <div className="userspace-share-link-type-toggle">
                    <label className="userspace-share-radio-option">
                      <input
                        type="radio"
                        name="shareLinkType"
                        value="named"
                        checked={shareLinkType === 'named'}
                        onChange={() => onShareLinkTypeChange('named')}
                      />
                      Named
                    </label>
                    <label className="userspace-share-radio-option">
                      <input
                        type="radio"
                        name="shareLinkType"
                        value="anonymous"
                        checked={shareLinkType === 'anonymous'}
                        onChange={() => onShareLinkTypeChange('anonymous')}
                      />
                      Anonymous
                    </label>
                    {allowSubdomainOption && (
                      <label className="userspace-share-radio-option">
                        <input
                          type="radio"
                          name="shareLinkType"
                          value="subdomain"
                          checked={shareLinkType === 'subdomain'}
                          onChange={() => onShareLinkTypeChange('subdomain')}
                          disabled={!shareSubdomainEnabled}
                        />
                        Subdomain
                      </label>
                    )}
                  </div>
                )}

                {shareStatus?.has_share_link && allowSubdomainOption && !shareSubdomainEnabled && shareSubdomainDisabledReason && (
                  <div className="userspace-share-meta">
                    {shareSubdomainDisabledReason}
                  </div>
                )}

                {showProtectedSubdomainNotice && (
                  <div className="userspace-share-warning-banner" role="alert">
                    Warning: if this workspace has already been unlocked in this browser, opening the subdomain link again may not prompt you to login. Protection is still enforced for new sessions and other browsers.
                  </div>
                )}

                {shareStatus?.has_share_link && effectiveShareUrl ? (
                  <>
                    <label htmlFor="userspace-share-url" className="userspace-share-label">Active share URL</label>
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
                    <div className="userspace-share-meta">
                      {activeShareCreatedLabel}
                    </div>
                    {selectedShareScopeSummary && (
                      <div className="userspace-share-meta">
                        {selectedShareScopeSummary}
                      </div>
                    )}
                  </>
                ) : (
                  <p className="userspace-muted">No active share link for this {shareTargetLabel}.</p>
                )}
              </div>

              <div className="userspace-share-controls">
                <div className="userspace-share-access-row">
                  <label htmlFor="userspace-share-access-mode" className="userspace-share-label">Access mode</label>
                  <select
                    id="userspace-share-access-mode"
                    value={shareAccessMode}
                    onChange={(event) => onShareAccessModeChange(event.target.value as ShareAccessMode)}
                    disabled={savingShareAccess || sharingWorkspace || revokingShareLink}
                  >
                    <option value="token">Tokenized public link</option>
                    <option value="password">Password protected</option>
                    <option value="authenticated_users">Any authenticated user</option>
                    <option value="selected_users">Selected users only</option>
                    <option value="ldap_groups">Selected LDAP groups only</option>
                  </select>
                </div>

                {shareAccessMode === 'password' && (
                  <div className="userspace-share-access-row">
                    <label htmlFor="userspace-share-password" className="userspace-share-label">
                      Share password {shareStatus?.has_password ? '(set)' : '(required)'}
                    </label>
                    <input
                      id="userspace-share-password"
                      type="password"
                      value={sharePasswordDraft}
                      onChange={(event) => onSharePasswordDraftChange(event.target.value)}
                      placeholder={shareStatus?.has_password ? 'Enter new password to update' : 'Enter password'}
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
                      <button className="btn btn-secondary" onClick={onAddShareLdapGroup} type="button">Add Group</button>
                    </div>
                    {loadingLdapGroups ? (
                      <p className="userspace-share-meta">Loading LDAP groups…</p>
                    ) : ldapDiscoveredGroups.length > 0 ? (
                      <p className="userspace-share-meta">Groups are discovered from the configured LDAP base domain.</p>
                    ) : (
                      <p className="userspace-share-meta">Could not auto-discover LDAP groups. Enter group DN manually.</p>
                    )}
                    {shareSelectedLdapGroupsDraft.length > 0 && (
                      <div className="userspace-share-group-list">
                        {shareSelectedLdapGroupsDraft.map((groupDn) => (
                          <div key={groupDn} className="userspace-share-group-item">
                            <span>{groupDn}</span>
                            <button
                              className="btn btn-secondary"
                              onClick={() => onRemoveShareLdapGroup(groupDn)}
                              type="button"
                            >
                              Remove
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {extraAccessControls}
              </div>
            </>
          )}
        </div>
        <div className="modal-footer userspace-share-modal-footer">
          <div className="userspace-share-actions userspace-share-actions-single">
            <button
              className="btn btn-secondary"
              onClick={onSaveShareAccess}
              disabled={savingShareAccess || sharingWorkspace || revokingShareLink || checkingShareSlug}
            >
              {savingShareAccess ? 'Saving Access...' : 'Save Access'}
            </button>
          </div>
          <div className="userspace-share-actions">
            <button
              className="btn btn-secondary"
              onClick={onCopyShareLink}
              disabled={sharingWorkspace || revokingShareLink || checkingShareSlug || savingShareAccess || shareHasUnsavedChanges}
            >
              {shareCopied ? 'Copied' : 'Copy Link'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={onOpenFullPreview}
              disabled={sharingWorkspace || revokingShareLink || checkingShareSlug || savingShareAccess || shareHasUnsavedChanges}
            >
              {openActionLabel}
            </button>
            <button
              className="btn btn-secondary"
              onClick={onRotateShareLink}
              disabled={sharingWorkspace || revokingShareLink || checkingShareSlug || savingShareAccess}
            >
              {rotatingShareLink ? 'Rotating...' : 'Rotate Link'}
            </button>
            <button
              className="btn btn-secondary"
              onClick={onRevokeShareLink}
              disabled={revokingShareLink || sharingWorkspace || checkingShareSlug || savingShareAccess || !shareStatus?.has_share_link}
            >
              {revokingShareLink ? 'Revoking...' : revokeActionLabel}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}