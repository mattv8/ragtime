import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api } from '@/api';
import type {
  ConversationShareAccessMode,
  ConversationShareLinkStatus,
  ConversationShareRole,
  User,
  UserDirectoryEntry,
} from '@/types';
import {
  areSameNormalizedStringArrays,
  normalizeShareSlugInput,
  normalizeUniqueStrings,
} from '@/utils';

import { ChatPanel } from './ChatPanel';
import { ShareLinkModal } from './shared/ShareLinkModal';
import { ToastContainer, useToast } from './shared/Toast';

interface ChatPageProps {
  currentUser: User;
  debugMode?: boolean;
  initialConversationId?: string | null;
  onFullscreenChange?: (isFullscreen: boolean) => void;
}

export function ChatPage({ currentUser, debugMode = false, initialConversationId, onFullscreenChange }: ChatPageProps) {
  const [toasts, toastActions] = useToast();
  const showSuccessToast = toastActions.success;
  const showErrorToast = toastActions.error;
  const dismissToast = toastActions.dismiss;
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [shareModalOpen, setShareModalOpen] = useState(false);
  const [loadingShareStatus, setLoadingShareStatus] = useState(false);
  const [shareLinks, setShareLinks] = useState<ConversationShareLinkStatus[]>([]);
  const [selectedShareId, setSelectedShareId] = useState<string | null>(null);
  const [shareStatus, setShareStatus] = useState<ConversationShareLinkStatus | null>(null);
  const [shareLinkType, setShareLinkType] = useState<'named' | 'anonymous' | 'subdomain'>('named');
  const [shareSlugDraft, setShareSlugDraft] = useState('');
  const [shareSlugAvailable, setShareSlugAvailable] = useState<boolean | null>(null);
  const [shareSlugEdited, setShareSlugEdited] = useState(false);
  const [shareAccessMode, setShareAccessMode] = useState<ConversationShareAccessMode>('token');
  const [sharePasswordDraft, setSharePasswordDraft] = useState('');
  const [shareSelectedUserIdsDraft, setShareSelectedUserIdsDraft] = useState<string[]>([]);
  const [shareSelectedLdapGroupsDraft, setShareSelectedLdapGroupsDraft] = useState<string[]>([]);
  const [shareLdapGroupDraft, setShareLdapGroupDraft] = useState('');
  const [shareSelectableUsers, setShareSelectableUsers] = useState<UserDirectoryEntry[]>([]);
  const [sharingConversation, setSharingConversation] = useState(false);
  const [savingShareAccess, setSavingShareAccess] = useState(false);
  const [savingShareLabel, setSavingShareLabel] = useState(false);
  const [revokingShareLink, setRevokingShareLink] = useState(false);
  const [deletingSelectedShareLink, setDeletingSelectedShareLink] = useState(false);
  const [rotatingShareLink, setRotatingShareLink] = useState(false);
  const [autoCreateShareLinkAttempted, setAutoCreateShareLinkAttempted] = useState(false);
  const [checkingShareSlug, setCheckingShareSlug] = useState(false);
  const [shareCopied, setShareCopied] = useState(false);
  const [grantedRoleDraft, setGrantedRoleDraft] = useState<ConversationShareRole>('viewer');
  const [shareAnchorMessageIdx, setShareAnchorMessageIdx] = useState<number | null>(null);
  const [shareScopeDirection, setShareScopeDirection] = useState<'forward' | 'backward'>('forward');
  const shareSlugCheckRequestRef = useRef(0);

  const formatUserLabel = useCallback((user?: Pick<User, 'username' | 'display_name'> | null, fallbackId?: string) => {
    const username = user?.username?.trim() || fallbackId?.trim() || 'unknown';
    const displayName = user?.display_name?.trim();
    if (displayName && displayName !== username) {
      return `${displayName} (@${username})`;
    }
    return `@${username}`;
  }, []);

  const applyShareStatus = useCallback((status: ConversationShareLinkStatus) => {
    setShareStatus(status);
    setShareSlugDraft(status.share_slug || '');
    setShareSlugEdited(false);
    setShareAccessMode(status.share_access_mode);
    setShareSelectedUserIdsDraft(status.selected_user_ids || []);
    setShareSelectedLdapGroupsDraft(status.selected_ldap_groups || []);
    setSharePasswordDraft('');
    setGrantedRoleDraft(status.granted_role || 'viewer');
    setShareAnchorMessageIdx(status.scope_anchor_message_idx ?? null);
    setShareScopeDirection(status.scope_direction ?? 'forward');
    setShareSlugAvailable(null);
  }, []);

  const syncShareSelection = useCallback((
    conversationId: string,
    links: ConversationShareLinkStatus[],
    ownerUsername: string,
    preferredShareId?: string | null,
  ) => {
    const nextSelected = (
      (preferredShareId ? links.find((candidate) => candidate.id === preferredShareId) : null)
      || (selectedShareId ? links.find((candidate) => candidate.id === selectedShareId) : null)
      || links[0]
      || null
    );

    setShareLinks(links);
    setSelectedShareId(nextSelected?.id ?? null);

    applyShareStatus(nextSelected ?? {
      id: '',
      conversation_id: conversationId,
      has_share_link: false,
      owner_username: ownerUsername,
      label: null,
      share_slug: null,
      share_token: null,
      share_url: null,
      anonymous_share_url: null,
      created_at: null,
      share_access_mode: 'token',
      selected_user_ids: [],
      selected_ldap_groups: [],
      has_password: false,
      granted_role: 'viewer',
      scope_anchor_message_idx: null,
      scope_direction: null,
    });
  }, [applyShareStatus, selectedShareId]);

  const loadShareStatus = useCallback(async (conversationId: string, preferredShareId?: string | null) => {
    setLoadingShareStatus(true);
    try {
      const response = await api.listConversationShareLinks(conversationId);
      syncShareSelection(
        conversationId,
        response.links,
        response.owner_username,
        preferredShareId,
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to load conversation share settings';
      showErrorToast(message);
    } finally {
      setLoadingShareStatus(false);
    }
  }, [showErrorToast, syncShareSelection]);

  useEffect(() => {
    if (!shareModalOpen || !activeConversationId) {
      return;
    }
    void loadShareStatus(activeConversationId);
  }, [activeConversationId, loadShareStatus, shareModalOpen]);

  useEffect(() => {
    if (!shareModalOpen) {
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const users = await api.listUsersDirectory();
        if (!cancelled) {
          setShareSelectableUsers(users);
        }
      } catch {
        if (!cancelled) {
          setShareSelectableUsers([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [shareModalOpen]);

  useEffect(() => {
    if (!shareModalOpen || !activeConversationId) {
      return;
    }
    if (!shareSlugEdited) {
      shareSlugCheckRequestRef.current += 1;
      setShareSlugAvailable(null);
      setCheckingShareSlug(false);
      return;
    }
    const normalizedSlug = normalizeShareSlugInput(shareSlugDraft);
    if (!normalizedSlug) {
      shareSlugCheckRequestRef.current += 1;
      setShareSlugAvailable(null);
      setCheckingShareSlug(false);
      return;
    }
    const requestId = shareSlugCheckRequestRef.current + 1;
    shareSlugCheckRequestRef.current = requestId;
    const handle = window.setTimeout(() => {
      setCheckingShareSlug(true);
      void api.checkConversationShareSlugAvailability(
        activeConversationId,
        normalizedSlug,
        shareStatus?.id || undefined,
      ).then((availability) => {
        if (shareSlugCheckRequestRef.current !== requestId) {
          return;
        }
        setShareSlugAvailable(availability.available);
      }).catch(() => {
        if (shareSlugCheckRequestRef.current !== requestId) {
          return;
        }
        setShareSlugAvailable(null);
      }).finally(() => {
        if (shareSlugCheckRequestRef.current !== requestId) {
          return;
        }
        setCheckingShareSlug(false);
      });
    }, 250);
    return () => window.clearTimeout(handle);
  }, [activeConversationId, shareModalOpen, shareSlugDraft, shareSlugEdited, shareStatus?.id]);

  const activeShareCreatedLabel = useMemo(() => {
    if (!shareStatus?.created_at) {
      return 'Share link inactive';
    }
    const createdDate = new Date(shareStatus.created_at);
    return `Created ${createdDate.toLocaleString()}`;
  }, [shareStatus?.created_at]);

  const effectiveShareUrl = useMemo(() => {
    if (!shareStatus?.has_share_link) {
      return null;
    }
    if (shareLinkType === 'anonymous') {
      return shareStatus.anonymous_share_url || shareStatus.share_url;
    }
    return shareStatus.share_url;
  }, [shareLinkType, shareStatus]);

  const scopedShareUrl = effectiveShareUrl;

  const shareHasUnsavedChanges = useMemo(() => {
    if (!shareStatus) {
      return false;
    }
    const normalizedDraftSlug = normalizeShareSlugInput(shareSlugDraft);
    const normalizedStatusSlug = normalizeShareSlugInput(shareStatus.share_slug || '');
    const selectedUsersChanged = !areSameNormalizedStringArrays(
      shareSelectedUserIdsDraft,
      shareStatus.selected_user_ids || [],
    );
    const selectedGroupsChanged = !areSameNormalizedStringArrays(
      shareSelectedLdapGroupsDraft,
      shareStatus.selected_ldap_groups || [],
    );
    const scopeAnchorChanged = shareAnchorMessageIdx !== (shareStatus.scope_anchor_message_idx ?? null);
    const scopeDirectionChanged = shareAnchorMessageIdx !== null
      && shareScopeDirection !== (shareStatus.scope_direction ?? 'forward');
    return normalizedDraftSlug !== normalizedStatusSlug
      || shareAccessMode !== shareStatus.share_access_mode
      || grantedRoleDraft !== shareStatus.granted_role
      || selectedUsersChanged
      || selectedGroupsChanged
      || scopeAnchorChanged
      || scopeDirectionChanged
      || (shareAccessMode === 'password' && sharePasswordDraft.trim().length > 0);
  }, [
    grantedRoleDraft,
    shareAccessMode,
    shareAnchorMessageIdx,
    shareLdapGroupDraft,
    sharePasswordDraft,
    shareScopeDirection,
    shareSelectedLdapGroupsDraft,
    shareSelectedUserIdsDraft,
    shareSlugDraft,
    shareStatus,
  ]);

  const resetCopiedState = useCallback(() => {
    setShareCopied(true);
    window.setTimeout(() => setShareCopied(false), 1500);
  }, []);

  const handleOpenShareModal = useCallback(() => {
    if (!activeConversationId) {
      return;
    }
    setAutoCreateShareLinkAttempted(false);
    setShareSlugEdited(false);
    setShareModalOpen(true);
  }, [activeConversationId]);

  const handleSelectShare = useCallback((shareId: string) => {
    const selected = shareLinks.find((candidate) => candidate.id === shareId);
    if (!selected) {
      return;
    }
    setSelectedShareId(shareId);
    applyShareStatus(selected);
  }, [applyShareStatus, shareLinks]);

  const handleShareConversationAtMessage = useCallback(async (messageIdx: number) => {
    if (!activeConversationId) {
      return;
    }
    setSharingConversation(true);
    try {
      // Create a new scoped share link bound to this anchor on the server. The
      // scope is enforced by the API (no URL params can override it).
      const link = await api.createConversationShareLink(activeConversationId, {
        scope_anchor_message_idx: messageIdx,
        scope_direction: 'forward',
        label: `Up to message ${messageIdx + 1}`,
      });
      setAutoCreateShareLinkAttempted(true);
      await loadShareStatus(activeConversationId, link.id);
      setShareModalOpen(true);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create scoped share link';
      showErrorToast(message);
    } finally {
      setSharingConversation(false);
    }
  }, [activeConversationId, loadShareStatus, showErrorToast]);

  const ensureShareLink = useCallback(async (rotateToken = false) => {
    if (!activeConversationId) {
      return null;
    }
    setSharingConversation(true);
    try {
      if (rotateToken) {
        // Legacy "rotate token" semantics: revoke existing primary then create a fresh one.
        await api.revokeConversationShareLink(activeConversationId);
      }
      const link = await api.createConversationShareLink(activeConversationId, {});
      await loadShareStatus(activeConversationId, link.id);
      return link;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create conversation share link';
      showErrorToast(message);
      return null;
    } finally {
      setSharingConversation(false);
    }
  }, [activeConversationId, loadShareStatus, showErrorToast]);

  useEffect(() => {
    if (!shareModalOpen || !activeConversationId) {
      return;
    }
    if (loadingShareStatus || sharingConversation || autoCreateShareLinkAttempted) {
      return;
    }
    if (shareLinks.length > 0) {
      return;
    }

    setAutoCreateShareLinkAttempted(true);
    void ensureShareLink(false);
  }, [
    activeConversationId,
    autoCreateShareLinkAttempted,
    ensureShareLink,
    loadingShareStatus,
    shareModalOpen,
    shareLinks.length,
    sharingConversation,
  ]);

  const handleSaveShareAccess = useCallback(async () => {
    if (!activeConversationId) {
      return;
    }
    setSavingShareAccess(true);
    try {
      const ensuredLink = !shareStatus?.id ? await ensureShareLink(false) : null;
      const currentShareId = shareStatus?.id || ensuredLink?.id;
      if (!currentShareId) {
        return;
      }

      const normalizedSlug = normalizeShareSlugInput(shareSlugDraft);
      if (normalizedSlug && normalizedSlug !== normalizeShareSlugInput(shareStatus?.share_slug || '')) {
        await api.updateConversationShareSlug(activeConversationId, currentShareId, normalizedSlug);
      }

      const scopeAnchor = shareAnchorMessageIdx;
      const scopeDirection = scopeAnchor !== null ? shareScopeDirection : null;
      const scopeChanged = scopeAnchor !== (shareStatus?.scope_anchor_message_idx ?? null)
        || scopeDirection !== (shareStatus?.scope_direction ?? null);
      if (scopeChanged) {
        await api.updateConversationShareLinkMetadata(activeConversationId, currentShareId, {
          label: shareStatus?.label ?? null,
          scope_anchor_message_idx: scopeAnchor,
          scope_direction: scopeDirection,
        });
      }

      const selectedUserIds = normalizeUniqueStrings(shareSelectedUserIdsDraft);
      const selectedLdapGroups = normalizeUniqueStrings(shareSelectedLdapGroupsDraft);
      const hasAccessChanges =
        shareAccessMode !== (shareStatus?.share_access_mode ?? 'token')
        || grantedRoleDraft !== (shareStatus?.granted_role ?? 'viewer')
        || !areSameNormalizedStringArrays(selectedUserIds, shareStatus?.selected_user_ids || [])
        || !areSameNormalizedStringArrays(selectedLdapGroups, shareStatus?.selected_ldap_groups || [])
        || (shareAccessMode === 'password' && Boolean(sharePasswordDraft.trim()));

      if (hasAccessChanges) {
        await api.updateConversationShareAccess(activeConversationId, currentShareId, {
          share_access_mode: shareAccessMode,
          password: shareAccessMode === 'password' ? sharePasswordDraft : undefined,
          selected_user_ids: selectedUserIds,
          selected_ldap_groups: selectedLdapGroups,
          granted_role: grantedRoleDraft,
        });
      }

      await loadShareStatus(activeConversationId, currentShareId);
      setSharePasswordDraft('');
      showSuccessToast('Conversation sharing updated');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update conversation sharing';
      showErrorToast(message);
    } finally {
      setSavingShareAccess(false);
    }
  }, [
    activeConversationId,
    ensureShareLink,
    grantedRoleDraft,
    loadShareStatus,
    shareAccessMode,
    shareAnchorMessageIdx,
    sharePasswordDraft,
    shareScopeDirection,
    shareSelectedLdapGroupsDraft,
    shareSelectedUserIdsDraft,
    shareSlugDraft,
    shareStatus,
    showErrorToast,
    showSuccessToast,
  ]);

  const handleCreateShareLink = useCallback(async () => {
    if (!activeConversationId) {
      return;
    }
    setSharingConversation(true);
    try {
      const link = await api.createConversationShareLink(activeConversationId, {});
      await loadShareStatus(activeConversationId, link.id);
      showSuccessToast('Conversation share link created');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to create conversation share link';
      showErrorToast(message);
    } finally {
      setSharingConversation(false);
    }
  }, [activeConversationId, loadShareStatus, showErrorToast, showSuccessToast]);

  const handleSaveShareLabel = useCallback(async (label: string) => {
    if (!activeConversationId || !shareStatus?.id) {
      return;
    }
    setSavingShareLabel(true);
    try {
      await api.updateConversationShareLinkMetadata(activeConversationId, shareStatus.id, {
        label,
        scope_anchor_message_idx: shareStatus.scope_anchor_message_idx ?? null,
        scope_direction: shareStatus.scope_direction ?? null,
      });
      await loadShareStatus(activeConversationId, shareStatus.id);
      showSuccessToast('Conversation share label updated');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to update conversation share label';
      showErrorToast(message);
    } finally {
      setSavingShareLabel(false);
    }
  }, [activeConversationId, loadShareStatus, shareStatus, showErrorToast, showSuccessToast]);

  const handleDeleteSelectedShareLink = useCallback(async (shareId: string) => {
    if (!activeConversationId || !shareId) {
      return;
    }
    setDeletingSelectedShareLink(true);
    try {
      await api.deleteConversationShareLink(activeConversationId, shareId);
      await loadShareStatus(activeConversationId);
      showSuccessToast('Conversation share link deleted');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to delete conversation share link';
      showErrorToast(message);
    } finally {
      setDeletingSelectedShareLink(false);
    }
  }, [activeConversationId, loadShareStatus, showErrorToast, showSuccessToast]);

  const handleCopyShareLink = useCallback(async () => {
    let targetUrl = effectiveShareUrl;
    if (!targetUrl) {
      const link = await ensureShareLink(false);
      if (!link) {
        return;
      }
      targetUrl = shareLinkType === 'anonymous'
        ? (link.anonymous_share_url || link.share_url)
        : link.share_url;
    }
    try {
      await navigator.clipboard.writeText(targetUrl);
      resetCopiedState();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to copy share link';
      showErrorToast(message);
    }
  }, [effectiveShareUrl, ensureShareLink, resetCopiedState, shareLinkType, showErrorToast]);

  const handleOpenShareLink = useCallback(async () => {
    let targetUrl = effectiveShareUrl;
    if (!targetUrl) {
      const link = await ensureShareLink(false);
      if (!link) {
        return;
      }
      targetUrl = shareLinkType === 'anonymous'
        ? (link.anonymous_share_url || link.share_url)
        : link.share_url;
    }
    try {
      // Force the preview to load through the current browser origin so that in
      // dev (Vite at :8001) we get hot-reload instead of falling back to the
      // backend port (:8000) that the server-minted share URL points to. The
      // public share path itself (e.g. /<owner>/<slug> or /shared/<token>) is
      // identical between the dev proxy and the production server.
      const minted = new URL(targetUrl, window.location.origin);
      const previewUrl = new URL(
        `${minted.pathname}${minted.search}${minted.hash}`,
        window.location.origin,
      );
      window.open(previewUrl.toString(), '_blank', 'noopener,noreferrer');
    } catch {
      // Fallback for malformed URLs; keep existing behavior.
      window.open(targetUrl, '_blank', 'noopener,noreferrer');
    }
  }, [effectiveShareUrl, ensureShareLink, shareLinkType]);

  const handleRotateShareLink = useCallback(async () => {
    if (!activeConversationId) {
      return;
    }
    setRotatingShareLink(true);
    try {
      await ensureShareLink(true);
      showSuccessToast('Conversation share link rotated');
    } finally {
      setRotatingShareLink(false);
    }
  }, [activeConversationId, ensureShareLink, showSuccessToast]);

  const handleRevokeShareLink = useCallback(async () => {
    if (!activeConversationId) {
      return;
    }
    setRevokingShareLink(true);
    try {
      await api.revokeConversationShareLink(activeConversationId);
      await loadShareStatus(activeConversationId);
      showSuccessToast('Conversation share link revoked');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to revoke conversation share link';
      showErrorToast(message);
    } finally {
      setRevokingShareLink(false);
    }
  }, [activeConversationId, loadShareStatus, showErrorToast, showSuccessToast]);

  const extraAccessControls = (
    <>
      <div className="userspace-share-access-row">
        <label htmlFor="conversation-share-role" className="userspace-share-label">Chat access</label>
        <select
          id="conversation-share-role"
          value={grantedRoleDraft}
          onChange={(event) => setGrantedRoleDraft(event.target.value as ConversationShareRole)}
        >
          <option value="viewer">View only</option>
          <option value="editor">Can participate</option>
        </select>
      </div>
      {shareAnchorMessageIdx !== null && (
        <div className="userspace-share-access-row">
          <label htmlFor="conversation-share-scope" className="userspace-share-label">
            Share scope (anchored at message #{shareAnchorMessageIdx + 1})
          </label>
          <div className="userspace-share-slug-row">
            <select
              id="conversation-share-scope"
              value={shareScopeDirection}
              onChange={(event) => setShareScopeDirection(event.target.value as 'forward' | 'backward')}
            >
              <option value="forward">From this message onwards</option>
              <option value="backward">Up to and including this message</option>
            </select>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setShareAnchorMessageIdx(null)}
              title="Share entire conversation"
            >
              Share entire chat
            </button>
          </div>
          <p className="userspace-share-meta">
            This share link is bound to the selected scope, so recipients only see the selected portion of this conversation.
          </p>
        </div>
      )}
    </>
  );

  return (
    <>
      <ChatPanel
        currentUser={currentUser}
        debugMode={debugMode}
        initialConversationId={initialConversationId}
        onFullscreenChange={onFullscreenChange}
        onActiveConversationChange={setActiveConversationId}
        onOpenShareModal={handleOpenShareModal}
        canShareConversation={Boolean(activeConversationId)}
        onShareConversationAtMessage={handleShareConversationAtMessage}
      />
      <ShareLinkModal
        isOpen={shareModalOpen}
        loadingShareStatus={loadingShareStatus}
        title="Share Conversation"
        shareLinkType={shareLinkType}
        shareStatus={shareStatus}
        shareLinks={shareLinks}
        selectedShareId={selectedShareId}
        shareSlugDraft={shareSlugDraft}
        shareSlugAvailable={shareSlugAvailable}
        shareAccessMode={shareAccessMode}
        sharePasswordDraft={sharePasswordDraft}
        shareSelectableUsers={shareSelectableUsers}
        shareSelectedUserIdsDraft={shareSelectedUserIdsDraft}
        shareSelectedLdapGroupsDraft={shareSelectedLdapGroupsDraft}
        shareLdapGroupDraft={shareLdapGroupDraft}
        ldapDiscoveredGroups={[]}
        loadingLdapGroups={false}
        shareSubdomainEnabled={false}
        shareSubdomainDisabledReason={null}
        showProtectedSubdomainNotice={false}
        effectiveShareUrl={scopedShareUrl}
        activeShareCreatedLabel={activeShareCreatedLabel}
        savingShareAccess={savingShareAccess}
        sharingWorkspace={sharingConversation}
        revokingShareLink={revokingShareLink}
        rotatingShareLink={rotatingShareLink}
        checkingShareSlug={checkingShareSlug}
        shareHasUnsavedChanges={shareHasUnsavedChanges}
        shareCopied={shareCopied}
        creatingShareLink={sharingConversation}
        updatingShareLabel={savingShareLabel}
        deletingSelectedShareLink={deletingSelectedShareLink}
        allowSubdomainOption={false}
        shareTargetLabel="conversation"
        openActionLabel="Open Link"
        extraAccessControls={extraAccessControls}
        onClose={() => setShareModalOpen(false)}
        onSelectShare={handleSelectShare}
        onCreateShareLink={() => { void handleCreateShareLink(); }}
        onSaveShareLabel={(label) => { void handleSaveShareLabel(label); }}
        onDeleteSelectedShareLink={(shareId) => { void handleDeleteSelectedShareLink(shareId); }}
        onShareSlugChange={(value) => {
          setShareSlugEdited(true);
          setShareSlugAvailable(null);
          setShareSlugDraft(normalizeShareSlugInput(value));
        }}
        onShareLinkTypeChange={setShareLinkType}
        onShareAccessModeChange={(value) => setShareAccessMode(value as ConversationShareAccessMode)}
        onSharePasswordDraftChange={setSharePasswordDraft}
        onToggleShareSelectedUser={(userId) => {
          setShareSelectedUserIdsDraft((previous) => (
            previous.includes(userId)
              ? previous.filter((candidate) => candidate !== userId)
              : [...previous, userId]
          ));
        }}
        onShareLdapGroupDraftChange={setShareLdapGroupDraft}
        onAddShareLdapGroup={() => {
          const normalizedGroup = shareLdapGroupDraft.trim();
          if (!normalizedGroup) {
            return;
          }
          setShareSelectedLdapGroupsDraft((previous) => normalizeUniqueStrings([...previous, normalizedGroup]));
          setShareLdapGroupDraft('');
        }}
        onRemoveShareLdapGroup={(groupDn) => {
          setShareSelectedLdapGroupsDraft((previous) => previous.filter((candidate) => candidate !== groupDn));
        }}
        onSaveShareAccess={handleSaveShareAccess}
        onCopyShareLink={() => { void handleCopyShareLink(); }}
        onOpenFullPreview={() => { void handleOpenShareLink(); }}
        onRotateShareLink={() => { void handleRotateShareLink(); }}
        onRevokeShareLink={() => { void handleRevokeShareLink(); }}
        onShareUrlInlineCopySuccess={resetCopiedState}
        onShareUrlInlineCopyError={(error) => showErrorToast(error.message)}
        formatUserLabel={formatUserLabel}
      />
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </>
  );
}