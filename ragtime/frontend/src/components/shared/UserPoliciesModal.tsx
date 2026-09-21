import { useState } from 'react';
import {
  updateContentProtectionConfigSlice,
  withUserOverride,
  type ContentProtectionConfig,
  type ContentProtectionOverrideMode,
} from '@/api/contentProtection';
import type { User } from '@/types';

interface UserPoliciesModalProps {
  user: User;
  isSelf: boolean;
  actionLoading: boolean;
  contentProtectionAvailable: boolean;
  contentProtectionEnabled: boolean;
  contentProtectionMode: ContentProtectionOverrideMode;
  onClose: () => void;
  onHostedChatPolicyChange: (userId: string, value: string) => Promise<void>;
  onContentProtectionConfigChange: (config: ContentProtectionConfig) => void;
  onSuccess: (message: string) => void;
  onError: (message: string) => void;
}

export function UserPoliciesModal({
  user,
  isSelf,
  actionLoading,
  contentProtectionAvailable,
  contentProtectionEnabled,
  contentProtectionMode,
  onClose,
  onHostedChatPolicyChange,
  onContentProtectionConfigChange,
  onSuccess,
  onError,
}: UserPoliciesModalProps) {
  const [contentProtectionSaving, setContentProtectionSaving] = useState(false);

  const hostedChatMode =
    user.hosted_chat_enabled === null || user.hosted_chat_enabled === undefined
      ? 'inherit'
      : user.hosted_chat_enabled
        ? 'enabled'
        : 'disabled';

  const handleContentProtectionChange = async (mode: ContentProtectionOverrideMode) => {
    setContentProtectionSaving(true);
    try {
      const saved = await updateContentProtectionConfigSlice((config) =>
        withUserOverride(config, user.id, mode),
      );
      onContentProtectionConfigChange(saved);
      onSuccess('Content protection policy updated');
    } catch (error) {
      onError(
        error instanceof Error ? error.message : 'Failed to update content protection policy',
      );
    } finally {
      setContentProtectionSaving(false);
    }
  };

  return (
    <div
      id="user-policies-modal-overlay"
      className="modal-overlay"
      data-user-policies-modal="overlay"
      onClick={onClose}
    >
      <div
        id={`user-policies-modal-${user.id}`}
        className="modal-content modal-small"
        data-user-policies-modal="content"
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3>User policies: {user.display_name || user.username}</h3>
          <button className="modal-close" onClick={onClose} aria-label="Close user policies">
            &times;
          </button>
        </div>
        <div className="modal-body" data-user-policies-modal="body">
          {!isSelf && (
            <div className="form-group" data-user-policy="hosted-chat">
              <label htmlFor={`hosted-chat-policy-${user.id}`}>Hosted chat</label>
              <select
                id={`hosted-chat-policy-${user.id}`}
                value={hostedChatMode}
                disabled={actionLoading}
                onChange={(event) => void onHostedChatPolicyChange(user.id, event.target.value)}
              >
                <option value="inherit">inherit</option>
                <option value="enabled">enabled</option>
                <option value="disabled">disabled</option>
              </select>
            </div>
          )}
          {contentProtectionAvailable && (
            <div className="form-group" data-user-policy="content-protection">
              <label htmlFor={`content-protection-policy-${user.id}`}>Content protection</label>
              <select
                id={`content-protection-policy-${user.id}`}
                value={contentProtectionMode}
                disabled={!contentProtectionEnabled || contentProtectionSaving}
                title={
                  contentProtectionEnabled
                    ? undefined
                    : 'Content protection is disabled in Settings.'
                }
                onChange={(event) =>
                  void handleContentProtectionChange(
                    event.target.value as ContentProtectionOverrideMode,
                  )
                }
              >
                <option value="inherit">inherit</option>
                <option value="always_classify">always classify</option>
                <option value="never_classify">never classify</option>
              </select>
              {!contentProtectionEnabled && (
                <p className="field-help">Content protection is disabled in Settings.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
