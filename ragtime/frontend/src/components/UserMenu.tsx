import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { User, ChevronDown, LogOut, Moon, Sun, Monitor, Palette, Shield } from 'lucide-react';
import type { User as UserType } from '@/types';
import { api } from '@/api';
import {
  THEME_PACKS,
  type ThemePackId,
  isThemePackId,
  type ColorMode,
  getStoredColorMode,
  setColorMode,
  setThemePack,
  resolveThemePackId,
  getThemePack,
} from '@/theme';

interface UserMenuProps {
  user: UserType;
  onLogout: () => void;
  defaultThemePack?: string | null;
}

export function UserMenu({ user, onLogout, defaultThemePack }: UserMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [dropdownPosition, setDropdownPosition] = useState<{ top: number; right: number } | null>(
    null,
  );
  const menuRef = useRef<HTMLDivElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const isAdmin = user.role === 'admin';

  const computeDropdownPosition = useCallback(() => {
    if (!menuRef.current) return;
    const rect = menuRef.current.getBoundingClientRect();
    setDropdownPosition({
      top: rect.bottom + 4,
      right: window.innerWidth - rect.right,
    });
  }, []);

  // Color mode (light/dark/system) — applied per browser via the shared util.
  const [colorMode, setColorModeState] = useState<ColorMode>(() => getStoredColorMode());
  useEffect(() => {
    setColorMode(colorMode);
  }, [colorMode]);

  const [themePack, setThemePackState] = useState<ThemePackId | null>(() =>
    isThemePackId(user.theme_pack) ? user.theme_pack : null,
  );
  const [mfaModalOpen, setMfaModalOpen] = useState(false);
  const [mfaSecret, setMfaSecret] = useState('');
  const [mfaUri, setMfaUri] = useState('');
  const [mfaCode, setMfaCode] = useState('');
  const [mfaRecoveryCodes, setMfaRecoveryCodes] = useState<string[]>([]);
  const [mfaError, setMfaError] = useState<string | null>(null);
  const [mfaLoading, setMfaLoading] = useState(false);
  useEffect(() => {
    setThemePackState(isThemePackId(user.theme_pack) ? user.theme_pack : null);
  }, [user.theme_pack]);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      const target = event.target as Node;
      if (
        menuRef.current &&
        !menuRef.current.contains(target) &&
        !dropdownRef.current?.contains(target)
      ) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    if (!isOpen) {
      setDropdownPosition(null);
      return;
    }

    computeDropdownPosition();
    window.addEventListener('scroll', computeDropdownPosition, true);
    window.addEventListener('resize', computeDropdownPosition);
    return () => {
      window.removeEventListener('scroll', computeDropdownPosition, true);
      window.removeEventListener('resize', computeDropdownPosition);
    };
  }, [isOpen, computeDropdownPosition]);

  // Close on escape key
  useEffect(() => {
    function handleEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setIsOpen(false);
      }
    }
    document.addEventListener('keydown', handleEscape);
    return () => document.removeEventListener('keydown', handleEscape);
  }, []);

  const getModeIcon = () => {
    if (colorMode === 'system') return <Monitor size={16} />;
    if (colorMode === 'dark') return <Moon size={16} />;
    return <Sun size={16} />;
  };

  const getModeLabel = () => {
    if (colorMode === 'system') return 'System';
    if (colorMode === 'dark') return 'Dark';
    return 'Light';
  };

  const cycleMode = () => {
    setColorModeState((current) => {
      if (current === 'dark') return 'light';
      if (current === 'light') return 'system';
      return 'dark';
    });
  };

  const cyclePack = () => {
    const options: Array<ThemePackId | null> = [...THEME_PACKS.map((p) => p.id), null];
    const index = options.indexOf(themePack);
    const next = options[(index + 1) % options.length];
    setThemePackState(next);
    setThemePack(resolveThemePackId(next, defaultThemePack));
    api.updateMyThemePack(next).catch(() => {});
  };

  const getPackLabel = () => {
    if (themePack === null) {
      return `System (${getThemePack(resolveThemePackId(null, defaultThemePack)).label})`;
    }
    return getThemePack(themePack).label;
  };

  const openMfaSetup = async () => {
    setMfaModalOpen(true);
    setMfaError(null);
    setMfaRecoveryCodes([]);
    setMfaCode('');
    setMfaLoading(true);
    try {
      const setup = await api.startMfaEnrollment();
      setMfaSecret(setup.secret);
      setMfaUri(setup.otpauth_uri);
    } catch (err) {
      setMfaError(err instanceof Error ? err.message : 'Failed to start MFA setup');
    } finally {
      setMfaLoading(false);
    }
  };

  const completeMfaSetup = async () => {
    setMfaError(null);
    setMfaLoading(true);
    try {
      const result = await api.completeMfaEnrollment({ code: mfaCode, remember_device: true });
      setMfaRecoveryCodes(result.recovery_codes);
    } catch (err) {
      setMfaError(err instanceof Error ? err.message : 'Failed to complete MFA setup');
    } finally {
      setMfaLoading(false);
    }
  };

  return (
    <div className="user-menu" ref={menuRef}>
      <button
        className="user-menu-trigger"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup="true"
      >
        <span className="user-menu-avatar">
          <User size={16} />
        </span>
        <span className="user-menu-name">{user.display_name || user.username}</span>
        {isAdmin && <span className="admin-badge">Admin</span>}
        <ChevronDown size={14} className={`user-menu-chevron ${isOpen ? 'rotated' : ''}`} />
      </button>

      {isOpen &&
        dropdownPosition &&
        createPortal(
          <div
            ref={dropdownRef}
            className="user-menu-dropdown"
            style={{ position: 'fixed', top: dropdownPosition.top, right: dropdownPosition.right }}
          >
            <div className="user-menu-header">
              <div className="user-menu-avatar-large">
                <User size={24} />
              </div>
              <div className="user-menu-info">
                <span className="user-menu-display-name">{user.display_name || user.username}</span>
                <span className="user-menu-role">{isAdmin ? 'Administrator' : 'User'}</span>
              </div>
            </div>

            <div className="user-menu-divider" />

            <button className="user-menu-item" onClick={cyclePack}>
              <Palette size={16} />
              <span>Theme: {getPackLabel()}</span>
            </button>

            <button className="user-menu-item" onClick={cycleMode}>
              {getModeIcon()}
              <span>Mode: {getModeLabel()}</span>
            </button>

            {!user.mfa_enabled && (
              <button className="user-menu-item" onClick={() => void openMfaSetup()}>
                <Shield size={16} />
                <span>Set up authenticator MFA</span>
              </button>
            )}

            <div className="user-menu-divider" />

            <button className="user-menu-item user-menu-logout" onClick={onLogout}>
              <LogOut size={16} />
              <span>Logout</span>
            </button>
          </div>,
          document.body,
        )}
      {mfaModalOpen &&
        createPortal(
          <div className="modal-overlay" onClick={() => setMfaModalOpen(false)}>
            <div className="modal-content" onClick={(event) => event.stopPropagation()}>
              <div className="modal-header">
                <h3>Set Up Authenticator MFA</h3>
                <button className="modal-close" onClick={() => setMfaModalOpen(false)}>
                  x
                </button>
              </div>
              {mfaError && <div className="login-error">{mfaError}</div>}
              {mfaRecoveryCodes.length > 0 ? (
                <div className="form-group">
                  <p className="field-help">
                    Save these recovery codes now. They will not be shown again.
                  </p>
                  <code className="cloud-oauth-callback-code">
                    {mfaRecoveryCodes.map((code) => (
                      <div key={code}>{code}</div>
                    ))}
                  </code>
                  <button className="btn btn-primary" onClick={() => setMfaModalOpen(false)}>
                    Done
                  </button>
                </div>
              ) : (
                <div className="form-group">
                  <p className="field-help">
                    Add this account to your authenticator app, then enter a code.
                  </p>
                  {mfaLoading && !mfaSecret ? (
                    <p className="field-help">Preparing setup...</p>
                  ) : null}
                  {mfaSecret && <code className="cloud-oauth-callback-code">{mfaSecret}</code>}
                  {mfaUri && <code className="cloud-oauth-callback-code">{mfaUri}</code>}
                  <input
                    className="form-input"
                    value={mfaCode}
                    onChange={(event) => setMfaCode(event.target.value)}
                    placeholder="Verification code"
                    autoComplete="one-time-code"
                  />
                  <button
                    className="btn btn-primary"
                    disabled={mfaLoading || !mfaCode}
                    onClick={() => void completeMfaSetup()}
                  >
                    {mfaLoading ? 'Verifying...' : 'Finish setup'}
                  </button>
                </div>
              )}
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
