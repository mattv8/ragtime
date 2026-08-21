// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';
import { EditorState } from '@codemirror/state';
import { EditorView } from '@codemirror/view';

import {
  createCodeMirrorThemeCompartment,
  createCodeMirrorThemeExtension,
  getCodeMirrorThemePalette,
  reconfigureCodeMirrorTheme,
} from './codemirrorTheme';
import { applyTerminalTheme, readTerminalTheme } from './terminalTheme';

describe('editor themes', () => {
  afterEach(() => {
    document.documentElement.removeAttribute('style');
    document.body.innerHTML = '';
  });

  it('defines Modern editor palettes, syntax colors, and the Fira Code stack', () => {
    expect(getCodeMirrorThemePalette({ pack: 'modern', mode: 'dark' })).toMatchObject({
      background: '#1f1f1f',
      gutterBackground: '#1f1f1f',
      gutterForeground: '#858585',
      selection: '#264f78',
      activeLine: '#2a2d2e',
      fontFamily: expect.stringContaining('Fira Code'),
      syntax: {
        keyword: '#569cd6',
        string: '#ce9178',
        comment: '#6a9955',
        number: '#b5cea8',
      },
    });

    expect(getCodeMirrorThemePalette({ pack: 'modern', mode: 'light' })).toMatchObject({
      background: '#ffffff',
      gutterBackground: '#f8f8f8',
      gutterForeground: '#237893',
      selection: '#add6ff',
      activeLine: '#f3f3f3',
      fontFamily: expect.stringContaining('Fira Code'),
      syntax: {
        keyword: '#0000ff',
        string: '#a31515',
        comment: '#008000',
        number: '#098658',
      },
    });
  });

  it('reconfigures a CodeMirror compartment in place without losing editor state', () => {
    const parent = document.createElement('div');
    document.body.appendChild(parent);

    const themeCompartment = createCodeMirrorThemeCompartment();
    const view = new EditorView({
      state: EditorState.create({
        doc: 'const answer = 42;\n',
        selection: { anchor: 6, head: 12 },
        extensions: [
          themeCompartment.of(createCodeMirrorThemeExtension({ pack: 'modern', mode: 'dark' })),
        ],
      }),
      parent,
    });

    expect(view.state.facet(EditorView.darkTheme)).toBe(true);

    const sameView = view;
    reconfigureCodeMirrorTheme(view, themeCompartment, { pack: 'modern', mode: 'light' });

    expect(view).toBe(sameView);
    expect(view.state.selection.main.from).toBe(6);
    expect(view.state.selection.main.to).toBe(12);
    expect(view.state.doc.toString()).toBe('const answer = 42;\n');
    expect(view.state.facet(EditorView.darkTheme)).toBe(false);

    view.destroy();
  });

  it('reads a complete terminal theme from canonical shared tokens and updates an existing theme object in place', () => {
    const rootStyle = document.documentElement.style;

    rootStyle.setProperty('--font-mono', "'Fira Code', monospace");
    rootStyle.setProperty('--color-terminal-surface', '#181818');
    rootStyle.setProperty('--color-terminal-foreground', '#cccccc');
    rootStyle.setProperty('--color-terminal-cursor', '#0078d4');
    rootStyle.setProperty('--color-terminal-selection', 'rgba(0, 120, 212, 0.2)');
    rootStyle.setProperty('--color-terminal-ansi-black', '#000000');
    rootStyle.setProperty('--color-terminal-ansi-red', '#cd3131');
    rootStyle.setProperty('--color-terminal-ansi-green', '#0dbc79');
    rootStyle.setProperty('--color-terminal-ansi-yellow', '#e5e510');
    rootStyle.setProperty('--color-terminal-ansi-blue', '#2472c8');
    rootStyle.setProperty('--color-terminal-ansi-magenta', '#bc3fbc');
    rootStyle.setProperty('--color-terminal-ansi-cyan', '#11a8cd');
    rootStyle.setProperty('--color-terminal-ansi-white', '#e5e5e5');
    rootStyle.setProperty('--color-terminal-ansi-bright-black', '#666666');
    rootStyle.setProperty('--color-terminal-ansi-bright-red', '#f14c4c');
    rootStyle.setProperty('--color-terminal-ansi-bright-green', '#23d18b');
    rootStyle.setProperty('--color-terminal-ansi-bright-yellow', '#f5f543');
    rootStyle.setProperty('--color-terminal-ansi-bright-blue', '#3b8eea');
    rootStyle.setProperty('--color-terminal-ansi-bright-magenta', '#d670d6');
    rootStyle.setProperty('--color-terminal-ansi-bright-cyan', '#29b8db');
    rootStyle.setProperty('--color-terminal-ansi-bright-white', '#e5e5e5');

    const nextTheme = readTerminalTheme({ pack: 'modern', mode: 'dark' });

    expect(nextTheme).toMatchObject({
      fontFamily: expect.stringContaining('Fira Code'),
      theme: {
        background: '#181818',
        foreground: '#cccccc',
        cursor: '#0078d4',
        selectionBackground: 'rgba(0, 120, 212, 0.2)',
        black: '#000000',
        brightWhite: '#e5e5e5',
      },
    });

    const existingThemeObject = { foreground: 'old-value' };
    const terminal = {
      options: {
        theme: existingThemeObject,
        fontFamily: 'old-font',
      },
    };

    applyTerminalTheme(terminal, nextTheme);

    expect(terminal.options.theme).toBe(existingThemeObject);
    expect(existingThemeObject).toMatchObject({
      background: '#181818',
      foreground: '#cccccc',
      brightBlue: '#3b8eea',
    });
    expect(terminal.options.fontFamily).toContain('Fira Code');
  });

  it('ignores legacy terminal alias variables when canonical values are absent', () => {
    const rootStyle = document.documentElement.style;

    rootStyle.setProperty('--terminal-foreground', '#123456');
    rootStyle.setProperty('--terminal-cursor', '#654321');
    rootStyle.setProperty('--terminal-selection-background', 'rgba(1, 2, 3, 0.4)');
    rootStyle.setProperty('--terminal-ansi-blue', '#abcdef');

    const nextTheme = readTerminalTheme({ pack: 'modern', mode: 'dark' });

    expect(nextTheme.theme.foreground).toBe('#CCCCCC');
    expect(nextTheme.theme.cursor).toBe('#0078D4');
    expect(nextTheme.theme.selectionBackground).toBe('rgba(0, 120, 212, 0.2)');
    expect(nextTheme.theme.blue).toBe('#2472C8');
  });

  it('defines token-driven User Space workbench styling without pack-specific color literals', () => {
    const css = readFileSync(join(cwd(), 'src/styles/workbench-userspace.css'), 'utf8');

    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-layout\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-toolbar\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-file-sidebar(?:\s*,|\s*\{)/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-code-editor(?:\s*,|\s*\{)/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-chat-section(?:\s*,|\s*\{)/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-preview-section(?:\s*,|\s*\{)/);
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-runtime-terminal(?:\s*,|\s*\{)/,
    );
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-status-pill\s*\{/);
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-snapshot-diff-editor-wrap\s*\{/,
    );
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-preview-card(?:\s*,|\s*\{)/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.userspace-readonly-badge\s*\{/);
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-chat-placeholder,\s*\[data-theme-pack='modern'\]\s+\.userspace-nontext-file-placeholder\s*\{/,
    );
    expect(css).toContain('@media (max-width: 768px)');
    expect(css).not.toMatch(/#[0-9a-fA-F]{3,8}/);
  });
});
