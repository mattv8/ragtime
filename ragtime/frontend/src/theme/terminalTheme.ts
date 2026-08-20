import type { ThemeSnapshot } from './themeSnapshot';

const _ANSI_SUFFIXES = [
  'black',
  'red',
  'green',
  'yellow',
  'blue',
  'magenta',
  'cyan',
  'white',
  'bright-black',
  'bright-red',
  'bright-green',
  'bright-yellow',
  'bright-blue',
  'bright-magenta',
  'bright-cyan',
  'bright-white',
] as const;

type AnsiSuffix = (typeof _ANSI_SUFFIXES)[number];

export interface TerminalThemeValues {
  foreground: string;
  background: string;
  cursor: string;
  cursorAccent: string;
  selectionBackground: string;
  selectionInactiveBackground: string;
  black: string;
  red: string;
  green: string;
  yellow: string;
  blue: string;
  magenta: string;
  cyan: string;
  white: string;
  brightBlack: string;
  brightRed: string;
  brightGreen: string;
  brightYellow: string;
  brightBlue: string;
  brightMagenta: string;
  brightCyan: string;
  brightWhite: string;
}

export interface TerminalThemeSnapshot {
  fontFamily: string;
  theme: TerminalThemeValues;
}

type TerminalTarget = {
  options: {
    theme?: Partial<TerminalThemeValues>;
    fontFamily?: string;
  };
};

const DARK_ANSI_FALLBACKS: Record<AnsiSuffix, string> = {
  black: '#000000',
  red: '#CD3131',
  green: '#0DBC79',
  yellow: '#E5E510',
  blue: '#2472C8',
  magenta: '#BC3FBC',
  cyan: '#11A8CD',
  white: '#E5E5E5',
  'bright-black': '#666666',
  'bright-red': '#F14C4C',
  'bright-green': '#23D18B',
  'bright-yellow': '#F5F543',
  'bright-blue': '#3B8EEA',
  'bright-magenta': '#D670D6',
  'bright-cyan': '#29B8DB',
  'bright-white': '#E5E5E5',
};

const LIGHT_ANSI_FALLBACKS: Record<AnsiSuffix, string> = {
  black: '#000000',
  red: '#CD3131',
  green: '#00BC00',
  yellow: '#949800',
  blue: '#0451A5',
  magenta: '#BC05BC',
  cyan: '#0598BC',
  white: '#555555',
  'bright-black': '#666666',
  'bright-red': '#CD3131',
  'bright-green': '#14CE14',
  'bright-yellow': '#B5BA00',
  'bright-blue': '#0451A5',
  'bright-magenta': '#BC05BC',
  'bright-cyan': '#0598BC',
  'bright-white': '#A5A5A5',
};

function readCssVar(styles: CSSStyleDeclaration, ...names: string[]): string {
  for (const name of names) {
    const value = styles.getPropertyValue(name).trim();
    if (value) {
      return value;
    }
  }
  return '';
}

function readAnsiColor(styles: CSSStyleDeclaration, suffix: AnsiSuffix, fallback: string): string {
  return readCssVar(styles, `--color-terminal-ansi-${suffix}`) || fallback;
}

export function readTerminalTheme(snapshot: ThemeSnapshot): TerminalThemeSnapshot {
  if (typeof document === 'undefined') {
    const ansiFallbacks = snapshot.mode === 'light' ? LIGHT_ANSI_FALLBACKS : DARK_ANSI_FALLBACKS;
    return {
      fontFamily:
        "var(--font-mono, 'Fira Code', 'SF Mono', Monaco, Consolas, 'Liberation Mono', monospace)",
      theme: {
        foreground: snapshot.mode === 'light' ? '#3B3B3B' : '#CCCCCC',
        background: snapshot.mode === 'light' ? '#FFFFFF' : '#181818',
        cursor: snapshot.mode === 'light' ? '#005FB8' : '#0078D4',
        cursorAccent: snapshot.mode === 'light' ? '#FFFFFF' : '#1F1F1F',
        selectionBackground:
          snapshot.mode === 'light' ? 'rgba(0, 95, 184, 0.16)' : 'rgba(0, 120, 212, 0.2)',
        selectionInactiveBackground:
          snapshot.mode === 'light' ? 'rgba(0, 95, 184, 0.1)' : 'rgba(0, 120, 212, 0.12)',
        black: ansiFallbacks.black,
        red: ansiFallbacks.red,
        green: ansiFallbacks.green,
        yellow: ansiFallbacks.yellow,
        blue: ansiFallbacks.blue,
        magenta: ansiFallbacks.magenta,
        cyan: ansiFallbacks.cyan,
        white: ansiFallbacks.white,
        brightBlack: ansiFallbacks['bright-black'],
        brightRed: ansiFallbacks['bright-red'],
        brightGreen: ansiFallbacks['bright-green'],
        brightYellow: ansiFallbacks['bright-yellow'],
        brightBlue: ansiFallbacks['bright-blue'],
        brightMagenta: ansiFallbacks['bright-magenta'],
        brightCyan: ansiFallbacks['bright-cyan'],
        brightWhite: ansiFallbacks['bright-white'],
      },
    };
  }

  const styles = getComputedStyle(document.documentElement);
  const ansiFallbacks = snapshot.mode === 'light' ? LIGHT_ANSI_FALLBACKS : DARK_ANSI_FALLBACKS;

  return {
    fontFamily:
      readCssVar(styles, '--font-mono') ||
      "'Fira Code', 'SF Mono', Monaco, Consolas, 'Liberation Mono', monospace",
    theme: {
      foreground:
        readCssVar(styles, '--color-terminal-foreground') ||
        (snapshot.mode === 'light' ? '#3B3B3B' : '#CCCCCC'),
      background:
        readCssVar(styles, '--color-terminal-surface', '--color-panel') ||
        (snapshot.mode === 'light' ? '#FFFFFF' : '#181818'),
      cursor:
        readCssVar(styles, '--color-terminal-cursor', '--color-focus') ||
        (snapshot.mode === 'light' ? '#005FB8' : '#0078D4'),
      cursorAccent:
        readCssVar(styles, '--color-editor', '--color-terminal-surface') ||
        (snapshot.mode === 'light' ? '#FFFFFF' : '#1F1F1F'),
      selectionBackground:
        readCssVar(styles, '--color-terminal-selection', '--color-primary-soft') ||
        (snapshot.mode === 'light' ? 'rgba(0, 95, 184, 0.16)' : 'rgba(0, 120, 212, 0.2)'),
      selectionInactiveBackground:
        readCssVar(styles, '--color-terminal-selection') ||
        (snapshot.mode === 'light' ? 'rgba(0, 95, 184, 0.1)' : 'rgba(0, 120, 212, 0.12)'),
      black: readAnsiColor(styles, 'black', ansiFallbacks.black),
      red: readAnsiColor(styles, 'red', ansiFallbacks.red),
      green: readAnsiColor(styles, 'green', ansiFallbacks.green),
      yellow: readAnsiColor(styles, 'yellow', ansiFallbacks.yellow),
      blue: readAnsiColor(styles, 'blue', ansiFallbacks.blue),
      magenta: readAnsiColor(styles, 'magenta', ansiFallbacks.magenta),
      cyan: readAnsiColor(styles, 'cyan', ansiFallbacks.cyan),
      white: readAnsiColor(styles, 'white', ansiFallbacks.white),
      brightBlack: readAnsiColor(styles, 'bright-black', ansiFallbacks['bright-black']),
      brightRed: readAnsiColor(styles, 'bright-red', ansiFallbacks['bright-red']),
      brightGreen: readAnsiColor(styles, 'bright-green', ansiFallbacks['bright-green']),
      brightYellow: readAnsiColor(styles, 'bright-yellow', ansiFallbacks['bright-yellow']),
      brightBlue: readAnsiColor(styles, 'bright-blue', ansiFallbacks['bright-blue']),
      brightMagenta: readAnsiColor(styles, 'bright-magenta', ansiFallbacks['bright-magenta']),
      brightCyan: readAnsiColor(styles, 'bright-cyan', ansiFallbacks['bright-cyan']),
      brightWhite: readAnsiColor(styles, 'bright-white', ansiFallbacks['bright-white']),
    },
  };
}

export function applyTerminalTheme(terminal: TerminalTarget, next: TerminalThemeSnapshot): void {
  const themeTarget = terminal.options.theme ?? {};

  Object.assign(themeTarget, next.theme);

  terminal.options.theme = themeTarget;
  terminal.options.fontFamily = next.fontFamily;
}
