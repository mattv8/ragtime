import { useEffect, useRef, useState } from 'react';
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
  onGenerationPolicyChange: (
    userId: string,
    policy: 'chat_enabled' | 'userspace_generation_enabled',
    value: string,
  ) => Promise<void>;
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
  onGenerationPolicyChange,
  onContentProtectionConfigChange,
  onSuccess,
  onError,
}: UserPoliciesModalProps) {
  const [contentProtectionSaving, setContentProtectionSaving] = useState(false);
  const dialogRef = useRef<HTMLDivElement>(null);

  const chatMode =
    user.chat_enabled === null || user.chat_enabled === undefined
      ? 'inherit'
      : user.chat_enabled
        ? 'enabled'
        : 'disabled';
  const userspaceGenerationMode =
    user.userspace_generation_enabled === null || user.userspace_generation_enabled === undefined
      ? 'inherit'
      : user.userspace_generation_enabled
        ? 'enabled'
        : 'disabled';

  const getEffectivePolicyHelp = (
    rawValue: boolean | null | undefined,
    effectiveValue: boolean | undefined,
    label: string,
  ): string => {
    if (effectiveValue === undefined) {
      return 'Effective status is unavailable until the policy refreshes.';
    }
    if (rawValue === null || rawValue === undefined) {
      return `Inherited. Effective: ${effectiveValue ? 'Enabled' : 'Disabled'}.`;
    }
    if (rawValue && !effectiveValue) {
      return `Enabled for this user, but Disabled effectively: this instance disables ${label}.`;
    }
    return `Effective: ${effectiveValue ? 'Enabled' : 'Disabled'}.`;
  };

  useEffect(() => {
    const previouslyFocused =
      document.activeElement instanceof HTMLElement ? document.activeElement : null;
    dialogRef.current?.focus();
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };
    document.addEventListener('keydown', handleEscape);
    return () => {
      document.removeEventListener('keydown', handleEscape);
      previouslyFocused?.focus();
    };
  }, [onClose]);

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
        ref={dialogRef}
        className="modal-content modal-small"
        data-user-policies-modal="content"
        role="dialog"
        aria-modal="true"
        aria-labelledby={`user-policies-modal-title-${user.id}`}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id={`user-policies-modal-title-${user.id}`}>
            User policies: {user.display_name || user.username}
          </h3>
          <button className="modal-close" onClick={onClose} aria-label="Close user policies">
            &times;
          </button>
        </div>
        <div className="modal-body" data-user-policies-modal="body">
          {!isSelf && (
            <>
              <div className="form-group" data-user-policy="chat-generation">
                <label htmlFor={`chat-generation-policy-${user.id}`}>Chat</label>
                <select
                  id={`chat-generation-policy-${user.id}`}
                  value={chatMode}
                  disabled={actionLoading}
                  onChange={(event) =>
                    void onGenerationPolicyChange(user.id, 'chat_enabled', event.target.value)
                  }
                >
                  <option value="inherit">inherit</option>
                  <option value="enabled">enabled</option>
                  <option value="disabled">disabled</option>
                </select>
                <p className="field-help">
                  {getEffectivePolicyHelp(user.chat_enabled, user.chat_enabled_effective, 'Chat')}
                </p>
              </div>
              <div className="form-group" data-user-policy="userspace-generation">
                <label htmlFor={`userspace-generation-policy-${user.id}`}>
                  User Space AI generation
                </label>
                <select
                  id={`userspace-generation-policy-${user.id}`}
                  value={userspaceGenerationMode}
                  disabled={actionLoading}
                  onChange={(event) =>
                    void onGenerationPolicyChange(
                      user.id,
                      'userspace_generation_enabled',
                      event.target.value,
                    )
                  }
                >
                  <option value="inherit">inherit</option>
                  <option value="enabled">enabled</option>
                  <option value="disabled">disabled</option>
                </select>
                <p className="field-help">
                  {getEffectivePolicyHelp(
                    user.userspace_generation_enabled,
                    user.userspace_generation_enabled_effective,
                    'User Space AI generation',
                  )}
                </p>
              </div>
            </>
          )}
          {contentProtectionAvailable && contentProtectionEnabled && (
            <div className="form-group" data-user-policy="content-protection">
              <label htmlFor={`content-protection-policy-${user.id}`}>Content protection</label>
              <select
                id={`content-protection-policy-${user.id}`}
                value={contentProtectionMode}
                disabled={contentProtectionSaving}
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
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
