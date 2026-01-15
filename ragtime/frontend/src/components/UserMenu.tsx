import { useState, useRef, useEffect } from 'react';
import { User, ChevronDown, LogOut, Moon, Sun, Monitor } from 'lucide-react';
import type { User as UserType } from '@/types';

type Theme = 'light' | 'dark' | 'system';

interface UserMenuProps {
  user: UserType;
  onLogout: () => void;
}

export function UserMenu({ user, onLogout }: UserMenuProps) {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const isAdmin = user.role === 'admin';

  // Theme state
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('ragtime-theme') as Theme | null;
    return stored || 'system';
  });

  // Apply theme changes
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'system') {
      root.removeAttribute('data-theme');
      localStorage.removeItem('ragtime-theme');
    } else {
      root.setAttribute('data-theme', theme);
      localStorage.setItem('ragtime-theme', theme);
    }
  }, [theme]);

  // Close menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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

  const getThemeIcon = () => {
    if (theme === 'system') return <Monitor size={16} />;
    if (theme === 'dark') return <Moon size={16} />;
    return <Sun size={16} />;
  };

  const getThemeLabel = () => {
    if (theme === 'system') return 'System';
    if (theme === 'dark') return 'Dark';
    return 'Light';
  };

  const cycleTheme = () => {
    setTheme(current => {
      if (current === 'dark') return 'light';
      if (current === 'light') return 'system';
      return 'dark';
    });
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
        <span className="user-menu-name">
          {user.display_name || user.username}
        </span>
        {isAdmin && <span className="admin-badge">Admin</span>}
        <ChevronDown size={14} className={`user-menu-chevron ${isOpen ? 'rotated' : ''}`} />
      </button>

      {isOpen && (
        <div className="user-menu-dropdown">
          <div className="user-menu-header">
            <div className="user-menu-avatar-large">
              <User size={24} />
            </div>
            <div className="user-menu-info">
              <span className="user-menu-display-name">
                {user.display_name || user.username}
              </span>
              <span className="user-menu-role">
                {isAdmin ? 'Administrator' : 'User'}
              </span>
            </div>
          </div>

          <div className="user-menu-divider" />

          <button className="user-menu-item" onClick={cycleTheme}>
            {getThemeIcon()}
            <span>Theme: {getThemeLabel()}</span>
          </button>

          <div className="user-menu-divider" />

          <button className="user-menu-item user-menu-logout" onClick={onLogout}>
            <LogOut size={16} />
            <span>Logout</span>
          </button>
        </div>
      )}
    </div>
  );
}
