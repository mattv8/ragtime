import { Moon, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';
import {
  type ColorMode,
  getStoredColorMode,
  setColorMode,
  resolveEffectiveColorMode,
} from '@/theme';

export function ThemeToggle() {
  const [theme, setTheme] = useState<ColorMode>(() => getStoredColorMode());

  useEffect(() => {
    setColorMode(theme);
  }, [theme]);

  const cycleTheme = () => {
    setTheme((current) => {
      if (current === 'dark') return 'light';
      if (current === 'light') return 'system';
      return 'dark';
    });
  };

  const effectiveTheme = resolveEffectiveColorMode(theme);

  return (
    <button
      className="theme-toggle"
      onClick={cycleTheme}
      aria-label={`Current theme: ${theme}. Click to change.`}
      title={`Theme: ${theme}`}
    >
      {effectiveTheme === 'dark' ? <Moon size={18} /> : <Sun size={18} />}
      {theme === 'system' && <span className="theme-toggle-indicator">auto</span>}
    </button>
  );
}
