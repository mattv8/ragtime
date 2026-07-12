import type { Dispatch, SetStateAction } from 'react';
import type { AppSettings, UpdateSettingsRequest } from '@/types';
import type { SettingsAccordionSectionId } from './settingsAccordionState';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import { PasswordRequirementsChecklist } from '../shared/PasswordRequirementsChecklist';
import { getExportPasswordPolicy } from '@/utils/exportPasswordPolicy';

interface SecuritySettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  settings: AppSettings | null;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  handleSaveSecurity: () => void | Promise<void>;
  securitySaving: boolean;
  exportPolicyPreview: string;
  setExportPolicyPreview: Dispatch<SetStateAction<string>>;
}

export function SecuritySettingsSection(props: SecuritySettingsSectionProps): JSX.Element {
  const {
    open,
    onToggle,
    formData,
    settings,
    setFormData,
    handleSaveSecurity,
    securitySaving,
    exportPolicyPreview,
    setExportPolicyPreview,
  } = props;

  return (
    <SettingsAccordionSection id="security" title="Security" open={open} onToggle={onToggle}>
      <fieldset id="setting-security">
        <legend>Security</legend>
        <p className="fieldset-help">
          Set the minimum password strength required when exporting encrypted tool configurations.
          These rules are enforced by the server; the export dialog also checks them live so users
          get immediate feedback.
        </p>

        <div className="form-row">
          <div>
            <div className="form-group">
              <label>Minimum Password Length</label>
              <input
                type="number"
                min={1}
                max={128}
                value={
                  formData.export_password_min_length ?? settings?.export_password_min_length ?? 12
                }
                onChange={(e) =>
                  setFormData({
                    ...formData,
                    export_password_min_length: Math.max(
                      1,
                      Math.min(128, parseInt(e.target.value, 10) || 1),
                    ),
                  })
                }
              />
              <p className="field-help">
                Number of characters an export password must contain. Range: 1 to 128. Default: 12.
              </p>
            </div>

            <div className="form-group">
              <label htmlFor="export-policy-preview">Test a password against this policy</label>
              <input
                id="export-policy-preview"
                type="text"
                value={exportPolicyPreview}
                onChange={(e) => setExportPolicyPreview(e.target.value)}
                placeholder="Type a sample password"
                autoComplete="off"
              />
              <PasswordRequirementsChecklist
                password={exportPolicyPreview}
                policy={getExportPasswordPolicy({
                  export_password_min_length:
                    formData.export_password_min_length ??
                    settings?.export_password_min_length ??
                    12,
                  export_password_require_uppercase:
                    formData.export_password_require_uppercase ??
                    settings?.export_password_require_uppercase ??
                    true,
                  export_password_require_lowercase:
                    formData.export_password_require_lowercase ??
                    settings?.export_password_require_lowercase ??
                    true,
                  export_password_require_number:
                    formData.export_password_require_number ??
                    settings?.export_password_require_number ??
                    true,
                  export_password_require_special:
                    formData.export_password_require_special ??
                    settings?.export_password_require_special ??
                    true,
                })}
              />
            </div>
          </div>

          <div>
            <div className="form-group">
              <label>Complexity requirements</label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={
                    formData.export_password_require_uppercase ??
                    settings?.export_password_require_uppercase ??
                    true
                  }
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      export_password_require_uppercase: e.target.checked,
                    })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Require an uppercase letter</span>
              </label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={
                    formData.export_password_require_lowercase ??
                    settings?.export_password_require_lowercase ??
                    true
                  }
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      export_password_require_lowercase: e.target.checked,
                    })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Require a lowercase letter</span>
              </label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={
                    formData.export_password_require_number ??
                    settings?.export_password_require_number ??
                    true
                  }
                  onChange={(e) =>
                    setFormData({ ...formData, export_password_require_number: e.target.checked })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Require a number</span>
              </label>
            </div>

            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={
                    formData.export_password_require_special ??
                    settings?.export_password_require_special ??
                    true
                  }
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      export_password_require_special: e.target.checked,
                    })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Require a special character</span>
              </label>
            </div>
          </div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="btn"
            onClick={handleSaveSecurity}
            disabled={securitySaving}
          >
            {securitySaving ? 'Saving...' : 'Save Security Settings'}
          </button>
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
