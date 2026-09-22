import { useEffect, useId, useState } from 'react';
import { X } from 'lucide-react';
import {
  contentProtectionApi,
  updateContentProtectionConfigSlice,
  withRequirement,
  type ContentProtectionConfig,
  type ContentProtectionRequirementMode,
} from '@/api/contentProtection';
import { ToastContainer, useToast } from './shared/Toast';

import {
  ToolAccessEditor,
  type ToolAccessGroupOption,
  type ToolAccessPolicy,
  type ToolAccessUserOption,
} from './ToolAccessEditor';

interface ToolAccessModalProps {
  open: boolean;
  toolName: string;
  policy: ToolAccessPolicy | null;
  userOptions: ToolAccessUserOption[];
  groupOptions: ToolAccessGroupOption[];
  loading?: boolean;
  saving?: boolean;
  disabled?: boolean;
  globalWriteEnabled?: boolean;
  onChange: (policy: ToolAccessPolicy) => void;
  onSave: (policy: ToolAccessPolicy) => void | Promise<void>;
  onContentProtectionModeChange?: (toolId: string, mode: ContentProtectionRequirementMode) => void;
  onClose: () => void;
}

export function ToolAccessModal({
  open,
  toolName,
  policy,
  userOptions,
  groupOptions,
  loading = false,
  saving = false,
  disabled = false,
  globalWriteEnabled = true,
  onChange,
  onSave,
  onContentProtectionModeChange,
  onClose,
}: ToolAccessModalProps) {
  const titleId = useId();
  const [contentProtectionConfig, setContentProtectionConfig] =
    useState<ContentProtectionConfig | null>(null);
  const [contentProtectionSaving, setContentProtectionSaving] = useState(false);
  const [toasts, toast] = useToast();

  useEffect(() => {
    if (!open) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose, open]);

  useEffect(() => {
    if (!open) {
      setContentProtectionConfig(null);
      return;
    }

    let active = true;
    void contentProtectionApi
      .getConfig()
      .then((config) => {
        if (active) setContentProtectionConfig(config);
      })
      .catch(() => {
        if (active) setContentProtectionConfig(null);
      });
    return () => {
      active = false;
    };
  }, [open]);

  if (!open) {
    return null;
  }

  const editorDisabled = disabled || loading || saving || policy == null;
  const contentProtectionMode =
    contentProtectionConfig && policy
      ? contentProtectionConfig.requirements.some(
          (requirement) =>
            requirement.scope_kind === 'tool' &&
            requirement.scope_key === policy.tool_id &&
            requirement.mode === 'require',
        )
        ? 'require'
        : 'inherit'
      : 'inherit';

  const handleContentProtectionModeChange = async (mode: ContentProtectionRequirementMode) => {
    if (!policy || contentProtectionSaving) return;

    setContentProtectionSaving(true);
    try {
      const savedConfig = await updateContentProtectionConfigSlice((config) =>
        withRequirement(config, 'tool', policy.tool_id, mode),
      );
      setContentProtectionConfig(savedConfig);
      onContentProtectionModeChange?.(policy.tool_id, mode);
      toast.success('Content protection updated');
    } catch (error) {
      toast.error(error instanceof Error ? error.message : 'Failed to update content protection');
    } finally {
      setContentProtectionSaving(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        className="modal-content modal-large tool-access-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="modal-header">
          <h3 id={titleId}>Tool Access - {toolName}</h3>
          <button type="button" className="modal-close" aria-label="Close" onClick={onClose}>
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className="modal-body">
          {loading || policy == null ? (
            <p className="field-help" style={{ margin: 0 }}>
              Loading access policy...
            </p>
          ) : (
            <ToolAccessEditor
              policy={policy}
              userOptions={userOptions}
              groupOptions={groupOptions}
              disabled={editorDisabled}
              globalWriteEnabled={globalWriteEnabled}
              autoFocusSearch
              onChange={onChange}
            />
          )}
          {policy != null && contentProtectionConfig?.enabled && (
            <section id="tool-content-protection" data-tool-content-protection>
              <label htmlFor="tool-content-protection-mode">Content protection</label>
              <select
                id="tool-content-protection-mode"
                data-tool-content-protection-mode
                value={contentProtectionMode}
                disabled={disabled || contentProtectionSaving}
                onChange={(event) =>
                  void handleContentProtectionModeChange(
                    event.target.value as ContentProtectionRequirementMode,
                  )
                }
              >
                <option value="inherit">Inherit</option>
                <option value="require">Require classification</option>
              </select>
              {contentProtectionConfig.coverage_mode === 'all_supported_traffic' && (
                <p className="field-help">
                  Coverage is currently All supported traffic; scope requirements apply when
                  coverage is Selected scopes.
                </p>
              )}
            </section>
          )}
        </div>
        <div className="modal-footer">
          <button type="button" className="btn btn-secondary" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="btn btn-primary"
            disabled={policy == null || loading || saving || disabled}
            onClick={() => {
              if (policy) {
                void onSave(policy);
              }
            }}
          >
            {saving ? 'Saving...' : 'Save Access'}
          </button>
        </div>
        <ToastContainer toasts={toasts} onDismiss={toast.dismiss} />
      </div>
    </div>
  );
}
