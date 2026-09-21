import type { Dispatch, SetStateAction } from 'react';

import type { UpdateSettingsRequest } from '@/types';

import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

export interface HostedExecutionSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  handleSaveHostedExecution: () => void | Promise<void>;
  hostedExecutionSaving: boolean;
}

export function HostedExecutionSettingsSection(
  props: HostedExecutionSettingsSectionProps,
): JSX.Element {
  const {
    open,
    onToggle,
    formData,
    setFormData,
    handleSaveHostedExecution,
    hostedExecutionSaving,
  } = props;
  const hostedChatEnabled = formData.hosted_chat_enabled !== false;

  return (
    <SettingsAccordionSection
      id="hosted-execution"
      title="Hosted Execution"
      open={open}
      onToggle={onToggle}
    >
      <fieldset id="setting-hosted_execution">
        <legend>Hosted Execution</legend>
        <p className="fieldset-help">
          Control whether Ragtime-hosted generation is available by default.
        </p>

        <div className="form-group settings-switch-card" id="setting-hosted_chat_enabled">
          <div className="settings-switch-copy">
            <label htmlFor="hosted-execution-enabled" className="settings-switch-title">
              Enable hosted chat
            </label>
            <p className="field-help">
              {hostedChatEnabled
                ? 'Hosted chat is enabled for users without an individual override.'
                : 'Hosted execution is unavailable globally. Individual user overrides cannot re-enable it.'}
            </p>
          </div>

          <label className="toggle-switch settings-switch-toggle">
            <input
              id="hosted-execution-enabled"
              type="checkbox"
              checked={hostedChatEnabled}
              onChange={(event) =>
                setFormData((current) => ({
                  ...current,
                  hosted_chat_enabled: event.target.checked,
                }))
              }
            />
            <span className="toggle-slider" />
          </label>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="btn"
            onClick={handleSaveHostedExecution}
            disabled={hostedExecutionSaving}
          >
            {hostedExecutionSaving ? 'Saving...' : 'Save Hosted Execution'}
          </button>
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
