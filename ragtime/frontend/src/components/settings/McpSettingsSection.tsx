import type { Dispatch, SetStateAction } from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { InlineCopyButton } from '../shared/InlineCopyButton';
import { LdapGroupChips, LdapGroupSelect, type LdapGroup } from '../LdapGroupSelect';
import { SettingsAccordionSection } from './SettingsAccordionSection';
import type { AppSettings, UpdateSettingsRequest } from '@/types';
import type { SettingsAccordionSectionId } from './settingsAccordionState';

export interface McpSettingsSectionProps {
  open: boolean;
  onToggle: (id: SettingsAccordionSectionId) => void;
  formData: UpdateSettingsRequest;
  settings: AppSettings | null;
  setFormData: Dispatch<SetStateAction<UpdateSettingsRequest>>;
  ldapConfigured: boolean;
  ldapDiscoveredGroups: LdapGroup[];
  showMcpPassword: boolean;
  setShowMcpPassword: Dispatch<SetStateAction<boolean>>;
  mcpError: string | null;
  setMcpError: Dispatch<SetStateAction<string | null>>;
  mcpSaving: boolean;
  handleSaveMcp: () => void | Promise<void>;
  setShowMcpRoutesPanel: Dispatch<SetStateAction<boolean>>;
  toast: { success: (message: string) => void };
  generateMcpClientId: () => string;
  generateMcpSecret: () => string;
}

export function McpSettingsSection(props: McpSettingsSectionProps): JSX.Element {
  const {
    open,
    onToggle,
    formData,
    settings,
    setFormData,
    ldapConfigured,
    ldapDiscoveredGroups,
    showMcpPassword,
    setShowMcpPassword,
    mcpError,
    setMcpError,
    mcpSaving,
    handleSaveMcp,
    setShowMcpRoutesPanel,
    toast,
    generateMcpClientId,
    generateMcpSecret,
  } = props;
  const authEnabled = formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth ?? false;
  const authMethod =
    formData.mcp_default_route_auth_method ?? settings?.mcp_default_route_auth_method ?? 'oauth2';
  const selectedMcpAllowedGroup =
    formData.mcp_default_route_allowed_group ?? settings?.mcp_default_route_allowed_group ?? '';
  const hasConfiguredOAuthFallbackPassword =
    authMethod === 'oauth2' &&
    ((settings?.has_mcp_default_password && formData.mcp_default_route_password !== '') ||
      Boolean(formData.mcp_default_route_password));

  return (
    <SettingsAccordionSection id="mcp" title="MCP Configuration" open={open} onToggle={onToggle}>
      <fieldset>
        <legend>MCP Configuration</legend>
        <p className="fieldset-help">
          Configure Model Context Protocol (MCP) access and authentication settings.
        </p>

        <div className="form-group">
          <label
            className="chat-toggle-control"
            style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}
          >
            <label className="toggle-switch">
              <input
                type="checkbox"
                checked={formData.mcp_enabled ?? settings?.mcp_enabled ?? false}
                onChange={(e) => setFormData({ ...formData, mcp_enabled: e.target.checked })}
              />
              <span className="toggle-slider"></span>
            </label>
            <span>Enable MCP Server</span>
          </label>
          <p className="field-help">
            When enabled, the MCP server endpoints (<code>/mcp</code> and custom routes) will be
            active. Disable to prevent all MCP access.
          </p>
        </div>

        {/* Only show other MCP settings when enabled */}
        {(formData.mcp_enabled ?? settings?.mcp_enabled ?? false) && (
          <>
            <div className="form-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={
                    formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth ?? false
                  }
                  onChange={(e) =>
                    setFormData({ ...formData, mcp_default_route_auth: e.target.checked })
                  }
                  style={{ marginRight: '0.5rem' }}
                />
                <span>Require authentication for default /mcp route</span>
              </label>
              <p className="field-help">
                When enabled, the default <code>/mcp</code> endpoint requires authentication.
                {authMethod === 'oauth2'
                  ? ' MCP clients authenticate through OAuth2 sign-in, and you can optionally allow MCP-Password as a fallback.'
                  : authMethod === 'client_credentials'
                    ? ' MCP clients must authenticate with a client ID and client secret, either via HTTP Basic or the per-route token endpoint.'
                    : settings?.has_mcp_default_password
                      ? ' A password is configured - MCP clients should use this password as the Bearer token.'
                      : ' Set a password below to enable password-based authentication.'}
              </p>
            </div>

            {/* Auth method selection - always show when auth is enabled. LDAP-only OAuth2 is conditional. */}
            {authEnabled && (
              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label>Authentication Method</label>
                <div
                  style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginTop: '0.5rem' }}
                >
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="mcp_auth_method"
                      value="oauth2"
                      checked={authMethod === 'oauth2'}
                      onChange={() =>
                        setFormData({ ...formData, mcp_default_route_auth_method: 'oauth2' })
                      }
                    />
                    <span>OAuth2</span>
                  </label>
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="mcp_auth_method"
                      value="password"
                      checked={authMethod === 'password'}
                      onChange={() =>
                        setFormData({ ...formData, mcp_default_route_auth_method: 'password' })
                      }
                    />
                    <span>Password</span>
                  </label>
                  <label className="radio-label">
                    <input
                      type="radio"
                      name="mcp_auth_method"
                      value="client_credentials"
                      checked={authMethod === 'client_credentials'}
                      onChange={() =>
                        setFormData({
                          ...formData,
                          mcp_default_route_auth_method: 'client_credentials',
                        })
                      }
                    />
                    <span>Client Credentials</span>
                  </label>
                </div>
                <p className="field-help">
                  {authMethod === 'oauth2'
                    ? 'Local and LDAP users can sign in through OAuth2. Optionally configure MCP-Password below as a fallback header.'
                    : authMethod === 'client_credentials'
                      ? 'MCP clients authenticate with client_id/client_secret over HTTP Basic, or exchange them at the token endpoint for a short-lived Bearer token.'
                      : 'MCP clients use a static password as the Bearer token or MCP-Password header.'}
                </p>
              </div>
            )}

            {/* LDAP Group restriction - only for OAuth2 auth method */}
            {authEnabled && ldapConfigured && authMethod === 'oauth2' && (
              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label htmlFor="mcp-allowed-group">Allowed LDAP Group (Optional)</label>
                <div style={{ maxWidth: '500px' }}>
                  <LdapGroupSelect
                    id="mcp-allowed-group"
                    value={
                      formData.mcp_default_route_allowed_group ??
                      settings?.mcp_default_route_allowed_group ??
                      ''
                    }
                    onChange={(value) =>
                      setFormData({
                        ...formData,
                        mcp_default_route_allowed_group: value || null,
                      })
                    }
                    groups={ldapDiscoveredGroups}
                    emptyOptionLabel="Any authenticated LDAP user"
                  />
                  <LdapGroupChips
                    selectedDns={selectedMcpAllowedGroup ? [selectedMcpAllowedGroup] : []}
                    groups={ldapDiscoveredGroups}
                    onRemove={() =>
                      setFormData({
                        ...formData,
                        mcp_default_route_allowed_group: null,
                      })
                    }
                  />
                </div>
                <p className="field-help">
                  Restrict access to members of a specific LDAP group. Leave empty to allow all
                  authenticated LDAP users.
                </p>
                {selectedMcpAllowedGroup && hasConfiguredOAuthFallbackPassword && (
                  <div className="field-warning">MCP-Password bypasses this group restriction.</div>
                )}
              </div>
            )}

            {authEnabled && authMethod === 'client_credentials' && (
              <>
                <div className="form-group" style={{ marginTop: '1rem' }}>
                  <label htmlFor="mcp-client-id">Client ID</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div
                      className="settings-inline-copy-wrap"
                      style={{ flex: 1, maxWidth: '400px' }}
                    >
                      <input
                        type="text"
                        id="mcp-client-id"
                        placeholder="cid-..."
                        value={
                          formData.mcp_default_route_client_id ??
                          settings?.mcp_default_route_client_id ??
                          ''
                        }
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            mcp_default_route_client_id: e.target.value,
                          })
                        }
                        style={{ width: '100%', fontFamily: 'var(--font-mono)' }}
                      />
                      <InlineCopyButton
                        copyText={
                          formData.mcp_default_route_client_id ??
                          settings?.mcp_default_route_client_id ??
                          ''
                        }
                        className="settings-inline-copy"
                        disabled={
                          !(
                            formData.mcp_default_route_client_id ??
                            settings?.mcp_default_route_client_id ??
                            ''
                          )
                        }
                        title="Copy client ID"
                        ariaLabel="Copy client ID"
                        copiedTitle="Client ID copied"
                        copiedAriaLabel="Client ID copied"
                        feedbackMs={2000}
                        onCopySuccess={() => toast.success('Client ID copied')}
                        onCopyError={() =>
                          setMcpError('Unable to copy client-id. Please copy it manually.')
                        }
                      />
                    </div>
                    <button
                      type="button"
                      className="btn btn-small btn-secondary"
                      onClick={() =>
                        setFormData({
                          ...formData,
                          mcp_default_route_client_id: generateMcpClientId(),
                        })
                      }
                    >
                      Generate Client ID
                    </button>
                  </div>
                  <p className="field-help">
                    Public identifier for MCP clients. Use this with the client secret for HTTP
                    Basic auth or token exchange.
                  </p>
                </div>

                <div className="form-group" style={{ marginTop: '1rem' }}>
                  <label htmlFor="mcp-client-secret">Client Secret</label>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div
                      className="settings-inline-copy-wrap"
                      style={{ flex: 1, maxWidth: '400px' }}
                    >
                      <input
                        type={showMcpPassword ? 'text' : 'password'}
                        id="mcp-client-secret"
                        placeholder={
                          settings?.has_mcp_default_password
                            ? '••••••••'
                            : 'Enter client secret (min 8 characters)'
                        }
                        value={formData.mcp_default_route_password ?? ''}
                        onChange={(e) =>
                          setFormData({
                            ...formData,
                            mcp_default_route_password: e.target.value,
                          })
                        }
                        style={{ width: '100%', fontFamily: 'var(--font-mono)' }}
                      />
                      <InlineCopyButton
                        copyText={formData.mcp_default_route_password ?? ''}
                        className="settings-inline-copy"
                        disabled={!(formData.mcp_default_route_password ?? '')}
                        title="Copy client secret"
                        ariaLabel="Copy client secret"
                        copiedTitle="Client secret copied"
                        copiedAriaLabel="Client secret copied"
                        feedbackMs={2000}
                        onCopySuccess={() => toast.success('Client secret copied')}
                        onCopyError={() =>
                          setMcpError('Unable to copy secret. Please copy it manually.')
                        }
                      />
                      <button
                        type="button"
                        className="settings-inline-copy settings-inline-copy-secondary"
                        onClick={() => setShowMcpPassword(!showMcpPassword)}
                        title={showMcpPassword ? 'Hide client secret' : 'Show client secret'}
                        aria-label={showMcpPassword ? 'Hide client secret' : 'Show client secret'}
                      >
                        {showMcpPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                      </button>
                    </div>
                    <button
                      type="button"
                      className="btn btn-small btn-secondary"
                      onClick={() =>
                        setFormData({
                          ...formData,
                          mcp_default_route_password: generateMcpSecret(),
                        })
                      }
                    >
                      Generate Password
                    </button>
                    {settings?.has_mcp_default_password && (
                      <button
                        type="button"
                        className="btn btn-small btn-secondary"
                        onClick={() => setFormData({ ...formData, mcp_default_route_password: '' })}
                        title="Clear client secret (submit empty to remove)"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  <p className="field-help">
                    {settings?.has_mcp_default_password
                      ? 'Client secret is set. Leave blank to keep the current secret, or enter a new one to rotate it. Clear and save to remove client credentials protection.'
                      : 'Set a client secret for MCP clients. Minimum 8 characters.'}
                  </p>
                  {window.location.protocol === 'http:' && (
                    <div className="field-warning">
                      <strong>Security:</strong> You are accessing over HTTP. Client credentials
                      will be transmitted in plaintext. Consider using HTTPS via a reverse proxy for
                      production deployments.
                    </div>
                  )}
                  {mcpError && <p className="field-error">{mcpError}</p>}
                </div>
              </>
            )}

            {/* Warning when auth is disabled */}
            {!(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) && (
              <div className="field-warning">
                <strong>Security Notice:</strong> The <code>/mcp</code> endpoint is currently open
                without authentication. Anyone with network access can invoke your configured tools.
                Consider enabling authentication if this server is accessible beyond localhost or a
                trusted network.
              </div>
            )}

            {/* Password for default MCP route - only show for password auth method */}
            {authEnabled && (authMethod === 'password' || authMethod === 'oauth2') && (
              <div className="form-group" style={{ marginTop: '1rem' }}>
                <label htmlFor="mcp-password">
                  {authMethod === 'oauth2' ? 'MCP Password Fallback (Optional)' : 'MCP Password'}
                </label>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div className="settings-inline-copy-wrap" style={{ flex: 1, maxWidth: '400px' }}>
                    <input
                      type={showMcpPassword ? 'text' : 'password'}
                      id="mcp-password"
                      placeholder={
                        settings?.has_mcp_default_password
                          ? '••••••••'
                          : 'Enter password (min 8 characters)'
                      }
                      value={formData.mcp_default_route_password ?? ''}
                      onChange={(e) =>
                        setFormData({ ...formData, mcp_default_route_password: e.target.value })
                      }
                      style={{ width: '100%' }}
                    />
                    <InlineCopyButton
                      copyText={formData.mcp_default_route_password ?? ''}
                      className="settings-inline-copy"
                      disabled={!(formData.mcp_default_route_password ?? '')}
                      title="Copy password"
                      ariaLabel="Copy password"
                      copiedTitle="Password copied"
                      copiedAriaLabel="Password copied"
                      feedbackMs={2000}
                      onCopySuccess={() => toast.success('Password copied')}
                      onCopyError={() =>
                        setMcpError('Unable to copy secret. Please copy it manually.')
                      }
                    />
                    <button
                      type="button"
                      className="settings-inline-copy settings-inline-copy-secondary"
                      onClick={() => setShowMcpPassword(!showMcpPassword)}
                      title={showMcpPassword ? 'Hide password' : 'Show password'}
                      aria-label={showMcpPassword ? 'Hide password' : 'Show password'}
                    >
                      {showMcpPassword ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  </div>
                  <button
                    type="button"
                    className="btn btn-small btn-secondary"
                    onClick={() =>
                      setFormData({
                        ...formData,
                        mcp_default_route_password: generateMcpSecret(),
                      })
                    }
                    title="Generate password"
                  >
                    Generate Password
                  </button>
                  {settings?.has_mcp_default_password && (
                    <button
                      type="button"
                      className="btn btn-small btn-secondary"
                      onClick={() => setFormData({ ...formData, mcp_default_route_password: '' })}
                      title="Clear password (submit empty to remove)"
                    >
                      Clear
                    </button>
                  )}
                </div>
                <p className="field-help">
                  {authMethod === 'oauth2'
                    ? settings?.has_mcp_default_password
                      ? 'Fallback password is set. Leave blank to keep current password, or enter a new one to change it. Clear and save to remove the MCP-Password fallback.'
                      : 'Optionally set an MCP-Password fallback header for clients that cannot sign in with OAuth2. Minimum 8 characters.'
                    : settings?.has_mcp_default_password
                      ? 'Password is set. Leave blank to keep current password, or enter a new one to change it. Clear and save to remove password protection.'
                      : 'Set a password that MCP clients will use as their Bearer token. Minimum 8 characters.'}
                </p>
                {window.location.protocol === 'http:' && (
                  <div className="field-warning">
                    <strong>Security:</strong> You are accessing over HTTP. MCP passwords will be
                    transmitted in plaintext. Consider using HTTPS via a reverse proxy for
                    production deployments.
                  </div>
                )}
                {mcpError && <p className="field-error">{mcpError}</p>}
              </div>
            )}

            {/* Show MCP error when password field is not visible */}
            {!(formData.mcp_default_route_auth ?? settings?.mcp_default_route_auth) && mcpError && (
              <p className="field-error" style={{ marginTop: '0.5rem' }}>
                {mcpError}
              </p>
            )}
          </>
        )}

        <div className="form-actions">
          <button type="button" className="btn" onClick={handleSaveMcp} disabled={mcpSaving}>
            {mcpSaving ? 'Saving...' : 'Save MCP Configuration'}
          </button>
          {(formData.mcp_enabled ?? settings?.mcp_enabled ?? false) && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setShowMcpRoutesPanel(true)}
            >
              Manage Custom Routes
            </button>
          )}
        </div>
      </fieldset>
    </SettingsAccordionSection>
  );
}
