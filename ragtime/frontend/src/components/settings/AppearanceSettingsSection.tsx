import type { CSSProperties, Dispatch, SetStateAction } from 'react';
import { Check } from 'lucide-react';
import type { AppSettings, UpdateSettingsRequest } from '@/types';
import type { ThemePackId } from '@/theme';
import { THEME_PACKS } from '@/theme';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

type AppearanceCardStyle = CSSProperties & {
  '--appearance-card-background'?: string;
  '--appearance-card-surface'?: string;
  '--appearance-card-primary'?: string;
  '--appearance-card-text'?: string;
};

interface AppearanceSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  settings: AppSettings | null;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  defaultThemePack: ThemePackId;
  setDefaultThemePack: Dispatch<SetStateAction<ThemePackId>>;
  isAdmin: boolean;
  highlightSetting?: string | null;
  handleSaveBranding: () => void | Promise<void>;
  brandingSaving: boolean;
}

export function AppearanceSettingsSection(props: AppearanceSettingsSectionProps): JSX.Element {
  const {
    open,
    onToggle,
    formData,
    settings,
    setFormData,
    defaultThemePack,
    setDefaultThemePack,
    isAdmin,
    highlightSetting,
    handleSaveBranding,
    brandingSaving,
  } = props;

  return (
    <SettingsAccordionSection id="appearance" title="Appearance" open={open} onToggle={onToggle}>
      <fieldset
        id="setting-appearance"
        className={highlightSetting === 'appearance' ? 'highlight-setting' : ''}
      >
        <legend>Appearance</legend>
        <p className="fieldset-help">
          Choose the instance-wide default theme and customize server branding. The default theme
          applies app-wide for users who have not picked their own theme from the user menu. Each
          theme has matching light and dark modes.
        </p>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '1.5rem',
            alignItems: 'start',
          }}
        >
          <div className="form-group">
            <label>Default theme</label>
            <div
              className="appearance-theme-grid"
              id="appearance-theme-grid"
              data-workbench-boundary="appearance-theme-grid"
            >
              {THEME_PACKS.map((pack) => {
                const selected = pack.id === defaultThemePack;
                const previewStyle: AppearanceCardStyle = {
                  '--appearance-card-background': pack.swatches.background,
                  '--appearance-card-surface': pack.swatches.surface,
                  '--appearance-card-primary': pack.swatches.primary,
                  '--appearance-card-text': pack.swatches.text,
                };

                return (
                  <button
                    type="button"
                    key={pack.id}
                    className="appearance-theme-card"
                    aria-pressed={selected}
                    disabled={!isAdmin}
                    onClick={() => setDefaultThemePack(pack.id)}
                    data-theme-pack-card={pack.id}
                    data-appearance-theme-card={pack.id}
                    style={previewStyle}
                  >
                    <span className="appearance-theme-card-header">
                      <span className="appearance-theme-card-name">{pack.label}</span>
                      {selected && (
                        <span className="appearance-theme-card-check">
                          <Check size={18} />
                        </span>
                      )}
                    </span>
                    <span className="appearance-theme-card-preview" aria-hidden="true">
                      <span className="appearance-theme-card-preview-header">
                        <span className="appearance-theme-card-preview-dot" />
                        <span className="appearance-theme-card-preview-dot" />
                        <span className="appearance-theme-card-preview-dot" />
                      </span>
                      <span className="appearance-theme-card-surface">
                        <span className="appearance-theme-card-accent" />
                        <span className="appearance-theme-card-preview-copy appearance-theme-card-preview-copy-strong">
                          {pack.label}
                        </span>
                        <span className="appearance-theme-card-preview-copy">
                          {pack.description}
                        </span>
                        <span className="appearance-swatches">
                          <span
                            className="appearance-swatch"
                            style={{ background: pack.swatches.background }}
                          />
                          <span
                            className="appearance-swatch"
                            style={{ background: pack.swatches.surface }}
                          />
                          <span
                            className="appearance-swatch"
                            style={{ background: pack.swatches.primary }}
                          />
                          <span
                            className="appearance-swatch appearance-swatch-sample"
                            style={{
                              background: pack.swatches.surface,
                              color: pack.swatches.text,
                              fontFamily: pack.headingFontPreview,
                            }}
                          >
                            Aa
                          </span>
                        </span>
                      </span>
                    </span>
                    <span
                      className="appearance-swatches appearance-swatches-compact"
                      aria-hidden="true"
                    >
                      <span
                        className="appearance-swatch"
                        style={{ background: pack.swatches.background }}
                      />
                      <span
                        className="appearance-swatch"
                        style={{ background: pack.swatches.surface }}
                      />
                      <span
                        className="appearance-swatch"
                        style={{ background: pack.swatches.primary }}
                      />
                      <span
                        className="appearance-swatch appearance-swatch-sample"
                        style={{
                          background: pack.swatches.surface,
                          color: pack.swatches.text,
                          fontFamily: pack.headingFontPreview,
                        }}
                      >
                        Aa
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
            <p className="field-help">
              Set your own theme from the user menu in the top-right corner; that personal choice
              overrides this default.
            </p>
          </div>

          <div
            id="setting-server_branding"
            className={highlightSetting === 'server_branding' ? 'highlight-setting' : ''}
          >
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                gap: '1rem',
                alignItems: 'start',
              }}
            >
              <div className="form-group">
                <label>Server Name</label>
                <input
                  type="text"
                  value={formData.server_name ?? settings?.server_name ?? 'Ragtime'}
                  onChange={(e) => setFormData({ ...formData, server_name: e.target.value })}
                  placeholder="Ragtime"
                />
              </div>

              <div className="form-group">
                <label
                  className="chat-toggle-control"
                  style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
                >
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={
                        formData.authenticated_webgl_background_enabled ??
                        settings?.authenticated_webgl_background_enabled ??
                        true
                      }
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          authenticated_webgl_background_enabled: e.target.checked,
                        })
                      }
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <span>Animated Background After Login</span>
                </label>
                <p className="field-help">
                  Show the WebGL gradient behind authenticated app pages. Disable this to use the
                  static theme background after login.
                </p>
              </div>

              <div className="form-group">
                <label
                  className="chat-toggle-control"
                  style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
                >
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={
                        formData.openapi_model_prefix_enabled ??
                        settings?.openapi_model_prefix_enabled ??
                        true
                      }
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          openapi_model_prefix_enabled: e.target.checked,
                        })
                      }
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <span>Prefix API Model Names</span>
                </label>
                <p className="field-help">
                  Add the server name before models listed by the OpenAI-compatible API.
                </p>
              </div>

              <div className="form-group">
                <label
                  className="chat-toggle-control"
                  style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
                >
                  <label className="toggle-switch">
                    <input
                      type="checkbox"
                      checked={
                        formData.show_tool_card_footer_actions ??
                        settings?.show_tool_card_footer_actions ??
                        false
                      }
                      onChange={(e) =>
                        setFormData({
                          ...formData,
                          show_tool_card_footer_actions: e.target.checked,
                        })
                      }
                    />
                    <span className="toggle-slider"></span>
                  </label>
                  <span>Show Tool Card Action Buttons</span>
                </label>
                <p className="field-help">
                  Display action buttons directly on tool cards. When disabled, actions stay in the
                  right-click menu.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="form-actions">
          {isAdmin && (
            <button
              type="button"
              className="btn"
              disabled={brandingSaving}
              onClick={handleSaveBranding}
            >
              {brandingSaving ? 'Saving...' : 'Save Appearance'}
            </button>
          )}
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
