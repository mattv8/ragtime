import { Compartment, type Extension } from '@codemirror/state';
import { HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import { EditorView } from '@codemirror/view';
import { tags } from '@lezer/highlight';

import type { ThemeSnapshot } from './themeSnapshot';

const CODE_FONT_FAMILY =
  "var(--font-mono, 'Fira Code', 'SF Mono', Monaco, Consolas, 'Liberation Mono', monospace)";

interface CodeMirrorSyntaxPalette {
  keyword: string;
  string: string;
  comment: string;
  number: string;
  type: string;
  function: string;
  variable: string;
  property: string;
  regexp: string;
  meta: string;
  invalid: string;
}

export interface CodeMirrorThemePalette {
  background: string;
  foreground: string;
  gutterBackground: string;
  gutterForeground: string;
  gutterBorder: string;
  activeLine: string;
  activeLineGutter: string;
  selection: string;
  selectionMatch: string;
  cursor: string;
  tooltipBackground: string;
  tooltipBorder: string;
  searchMatch: string;
  fontFamily: string;
  syntax: CodeMirrorSyntaxPalette;
}

const VSCODE_DARK_PALETTE: CodeMirrorThemePalette = {
  background: '#1f1f1f',
  foreground: '#d4d4d4',
  gutterBackground: '#1f1f1f',
  gutterForeground: '#858585',
  gutterBorder: '#2b2b2b',
  activeLine: '#2a2d2e',
  activeLineGutter: '#2a2d2e',
  selection: '#264f78',
  selectionMatch: 'rgba(104, 104, 104, 0.35)',
  cursor: '#aeafad',
  tooltipBackground: '#252526',
  tooltipBorder: '#454545',
  searchMatch: 'rgba(234, 92, 0, 0.33)',
  fontFamily: CODE_FONT_FAMILY,
  syntax: {
    keyword: '#569cd6',
    string: '#ce9178',
    comment: '#6a9955',
    number: '#b5cea8',
    type: '#4ec9b0',
    function: '#dcdcaa',
    variable: '#9cdcfe',
    property: '#9cdcfe',
    regexp: '#d16969',
    meta: '#c586c0',
    invalid: '#f44747',
  },
};

const VSCODE_LIGHT_PALETTE: CodeMirrorThemePalette = {
  background: '#ffffff',
  foreground: '#000000',
  gutterBackground: '#f8f8f8',
  gutterForeground: '#237893',
  gutterBorder: '#e5e5e5',
  activeLine: '#f3f3f3',
  activeLineGutter: '#f3f3f3',
  selection: '#add6ff',
  selectionMatch: 'rgba(160, 160, 160, 0.3)',
  cursor: '#000000',
  tooltipBackground: '#ffffff',
  tooltipBorder: '#c8c8c8',
  searchMatch: 'rgba(234, 92, 0, 0.28)',
  fontFamily: CODE_FONT_FAMILY,
  syntax: {
    keyword: '#0000ff',
    string: '#a31515',
    comment: '#008000',
    number: '#098658',
    type: '#267f99',
    function: '#795e26',
    variable: '#001080',
    property: '#001080',
    regexp: '#811f3f',
    meta: '#af00db',
    invalid: '#cd3131',
  },
};

const GENERIC_DARK_PALETTE: CodeMirrorThemePalette = {
  background: 'var(--color-editor)',
  foreground: 'var(--color-text-primary)',
  gutterBackground: 'var(--color-editor)',
  gutterForeground: 'var(--color-text-muted)',
  gutterBorder: 'var(--color-border)',
  activeLine: 'color-mix(in srgb, var(--color-text-primary) 6%, var(--color-editor))',
  activeLineGutter: 'color-mix(in srgb, var(--color-text-primary) 6%, var(--color-editor))',
  selection: 'var(--color-primary-soft)',
  selectionMatch: 'color-mix(in srgb, var(--color-text-primary) 16%, transparent)',
  cursor: 'var(--color-focus)',
  tooltipBackground: 'var(--color-widget)',
  tooltipBorder: 'var(--color-border)',
  searchMatch: 'color-mix(in srgb, var(--color-warning) 28%, transparent)',
  fontFamily: CODE_FONT_FAMILY,
  syntax: {
    keyword: 'var(--color-primary)',
    string: 'var(--color-accent)',
    comment: 'var(--color-text-muted)',
    number: 'var(--color-success)',
    type: 'var(--color-info)',
    function: 'var(--color-text-strong)',
    variable: 'var(--color-text-primary)',
    property: 'var(--color-text-primary)',
    regexp: 'var(--color-warning)',
    meta: 'var(--color-primary)',
    invalid: 'var(--color-error)',
  },
};

const GENERIC_LIGHT_PALETTE: CodeMirrorThemePalette = {
  ...GENERIC_DARK_PALETTE,
  activeLine: 'color-mix(in srgb, var(--color-primary) 5%, var(--color-editor))',
  activeLineGutter: 'color-mix(in srgb, var(--color-primary) 5%, var(--color-editor))',
  selectionMatch: 'color-mix(in srgb, var(--color-text-primary) 10%, transparent)',
};

export function getCodeMirrorThemePalette(snapshot: ThemeSnapshot): CodeMirrorThemePalette {
  if (snapshot.pack === 'vscode') {
    return snapshot.mode === 'light' ? VSCODE_LIGHT_PALETTE : VSCODE_DARK_PALETTE;
  }

  return snapshot.mode === 'light' ? GENERIC_LIGHT_PALETTE : GENERIC_DARK_PALETTE;
}

function createSyntaxHighlightStyle(palette: CodeMirrorThemePalette): HighlightStyle {
  return HighlightStyle.define([
    {
      tag: [tags.keyword, tags.operatorKeyword, tags.controlKeyword, tags.definitionKeyword],
      color: palette.syntax.keyword,
    },
    {
      tag: [tags.comment, tags.lineComment, tags.blockComment],
      color: palette.syntax.comment,
    },
    {
      tag: [tags.string, tags.special(tags.string)],
      color: palette.syntax.string,
    },
    {
      tag: [tags.number, tags.integer, tags.float, tags.bool, tags.null],
      color: palette.syntax.number,
    },
    {
      tag: [tags.typeName, tags.className],
      color: palette.syntax.type,
    },
    {
      tag: [tags.function(tags.variableName), tags.function(tags.propertyName)],
      color: palette.syntax.function,
    },
    {
      tag: [tags.variableName],
      color: palette.syntax.variable,
    },
    {
      tag: [tags.propertyName, tags.attributeName],
      color: palette.syntax.property,
    },
    {
      tag: [tags.regexp, tags.escape],
      color: palette.syntax.regexp,
    },
    {
      tag: [tags.meta, tags.annotation],
      color: palette.syntax.meta,
    },
    {
      tag: tags.invalid,
      color: palette.syntax.invalid,
    },
  ]);
}

export function createCodeMirrorThemeExtension(snapshot: ThemeSnapshot): Extension {
  const palette = getCodeMirrorThemePalette(snapshot);

  return [
    EditorView.theme(
      {
        '&': {
          color: palette.foreground,
          backgroundColor: palette.background,
          fontFamily: palette.fontFamily,
        },
        '.cm-scroller': {
          fontFamily: palette.fontFamily,
        },
        '.cm-content, .cm-line': {
          caretColor: palette.cursor,
        },
        '.cm-gutters': {
          backgroundColor: palette.gutterBackground,
          color: palette.gutterForeground,
          borderRight: `1px solid ${palette.gutterBorder}`,
        },
        '.cm-activeLine': {
          backgroundColor: palette.activeLine,
        },
        '.cm-activeLineGutter': {
          backgroundColor: palette.activeLineGutter,
          color: palette.foreground,
        },
        '.cm-selectionBackground, .cm-content ::selection': {
          backgroundColor: palette.selection,
        },
        '&.cm-focused .cm-selectionBackground': {
          backgroundColor: palette.selection,
        },
        '.cm-cursor, .cm-dropCursor': {
          borderLeftColor: palette.cursor,
        },
        '.cm-searchMatch': {
          backgroundColor: palette.searchMatch,
          outline: '1px solid transparent',
        },
        '.cm-searchMatch.cm-searchMatch-selected': {
          backgroundColor: palette.selectionMatch,
        },
        '.cm-tooltip': {
          backgroundColor: palette.tooltipBackground,
          border: `1px solid ${palette.tooltipBorder}`,
          color: palette.foreground,
        },
        '.cm-panels': {
          backgroundColor: palette.tooltipBackground,
          color: palette.foreground,
          borderTop: `1px solid ${palette.tooltipBorder}`,
        },
      },
      { dark: snapshot.mode === 'dark' },
    ),
    syntaxHighlighting(createSyntaxHighlightStyle(palette)),
  ];
}

export function createCodeMirrorThemeCompartment(): Compartment {
  return new Compartment();
}

export function reconfigureCodeMirrorTheme(
  view: EditorView | null,
  compartment: Compartment,
  snapshot: ThemeSnapshot,
): void {
  if (!view) {
    return;
  }

  view.dispatch({
    effects: compartment.reconfigure(createCodeMirrorThemeExtension(snapshot)),
  });
}
