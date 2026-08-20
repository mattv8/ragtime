import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { User, ChevronDown, LogOut, Moon, Sun, Monitor, Palette, Shield } from 'lucide-react';
import type { User as UserType } from '@/types';
import { api } from '@/api';
import { Manage2FAModal } from './Manage2FAModal';
import { ThemeChromeIcon } from '@/components/shared/ThemeChromeIcon';
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
  const [mfaHubOpen, setMfaHubOpen] = useState(false);
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

  return (
    <div className="user-menu" ref={menuRef} data-workbench-boundary="user-menu">
      <button
        type="button"
        className="user-menu-trigger"
        onClick={() => setIsOpen(!isOpen)}
        aria-expanded={isOpen}
        aria-haspopup="true"
        aria-controls="user-menu-dropdown"
      >
        <span className="user-menu-avatar">
          <User size={16} />
        </span>
        <span className="user-menu-name">{user.display_name || user.username}</span>
        {isAdmin && <span className="admin-badge">Admin</span>}
        <ThemeChromeIcon
          className={`user-menu-chevron ${isOpen ? 'rotated' : ''}`}
          fallback={<ChevronDown size={14} />}
          codicon="chevron-down"
          size={14}
        />
      </button>

      {isOpen &&
        dropdownPosition &&
        createPortal(
          <div
            id="user-menu-dropdown"
            ref={dropdownRef}
            className="user-menu-dropdown"
            data-workbench-surface="user-menu-dropdown"
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

            <button className="user-menu-item" onClick={() => setMfaHubOpen(true)}>
              <Shield size={16} />
              <span>Manage 2FA</span>
            </button>

            <div className="user-menu-divider" />

            <button className="user-menu-item user-menu-logout" onClick={onLogout}>
              <LogOut size={16} />
              <span>Logout</span>
            </button>
          </div>,
          document.body,
        )}
      <Manage2FAModal isOpen={mfaHubOpen} onClose={() => setMfaHubOpen(false)} />
    </div>
  );
}
