import { Moon, Sun } from 'lucide-react';
import { useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'system';

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => {
    const stored = localStorage.getItem('ragtime-theme') as Theme | null;
    return stored || 'system';
  });

  useEffect(() => {
    const root = document.documentElement;

    if (theme === 'system') {
      // Remove data-theme to let CSS @media handle it
      root.removeAttribute('data-theme');
      localStorage.removeItem('ragtime-theme');
    } else {
      root.setAttribute('data-theme', theme);
      localStorage.setItem('ragtime-theme', theme);
    }
  }, [theme]);

  const cycleTheme = () => {
    setTheme(current => {
      if (current === 'dark') return 'light';
      if (current === 'light') return 'system';
      return 'dark';
    });
  };

  // Determine the effective theme for the icon display
  const getEffectiveTheme = (): 'light' | 'dark' => {
    if (theme === 'system') {
      return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
    }
    return theme;
  };

  const effectiveTheme = getEffectiveTheme();

  return (
    <button
      className="theme-toggle"
      onClick={cycleTheme}
      aria-label={`Current theme: ${theme}. Click to change.`}
      title={`Theme: ${theme}`}
    >
      {effectiveTheme === 'dark' ? (
        <Moon size={18} />
      ) : (
        <Sun size={18} />
      )}
      {theme === 'system' && <span className="theme-toggle-indicator">auto</span>}
    </button>
  );
}
