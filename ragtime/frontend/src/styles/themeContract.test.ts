// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const TERMINAL_ANSI_SUFFIXES = [
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

const REQUIRED_THEME_TOKENS = [
  '--color-workbench',
  '--color-chrome',
  '--color-editor',
  '--color-panel',
  '--color-widget',
  '--color-focus',
  '--color-sash-hover',
  '--color-terminal-surface',
  '--color-terminal-foreground',
  '--color-terminal-cursor',
  '--color-terminal-selection',
  ...TERMINAL_ANSI_SUFFIXES.map((suffix) => `--color-terminal-ansi-${suffix}`),
] as const;

const LEGACY_TERMINAL_TOKEN_PATTERN =
  /--terminal-(?:background|foreground|cursor|cursor-accent|selection-background|selection-inactive-background|ansi-[\w-]+)/;

function readProjectFile(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), 'utf8');
}

function getRuleBody(css: string, selector: string): string {
  const start = css.indexOf(`${selector} {`);
  expect(start).toBeGreaterThanOrEqual(0);

  const bodyStart = css.indexOf('{', start);
  let depth = 0;

  for (let index = bodyStart; index < css.length; index += 1) {
    const character = css[index];
    if (character === '{') depth += 1;
    if (character === '}') {
      depth -= 1;
      if (depth === 0) {
        return css.slice(bodyStart + 1, index);
      }
    }
  }

  throw new Error(`Unterminated CSS rule for ${selector}`);
}

function getMergedRuleProperties(css: string, selector: string): Record<string, string> {
  const blocks: string[] = [];
  let searchStart = 0;

  while (searchStart < css.length) {
    const start = css.indexOf(`${selector} {`, searchStart);
    if (start < 0) {
      break;
    }

    const bodyStart = css.indexOf('{', start);
    let depth = 0;

    for (let index = bodyStart; index < css.length; index += 1) {
      const character = css[index];
      if (character === '{') depth += 1;
      if (character === '}') {
        depth -= 1;
        if (depth === 0) {
          blocks.push(css.slice(bodyStart + 1, index));
          searchStart = index + 1;
          break;
        }
      }
    }
  }

  expect(blocks.length).toBeGreaterThan(0);
  return Object.assign({}, ...blocks.map((block) => parseCustomProperties(block)));
}

function parseCustomProperties(block: string): Record<string, string> {
  return Object.fromEntries(
    Array.from(block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g), (match) => [
      match[1],
      match[2].trim(),
    ]),
  );
}

function collectDirectVarEdges(properties: Record<string, string>): Map<string, string[]> {
  return new Map(
    Object.entries(properties).map(([property, value]) => [
      property,
      Array.from(value.matchAll(/var\((--[\w-]+)/g), (match) => match[1]),
    ]),
  );
}

function expectNoAliasCycles(properties: Record<string, string>): void {
  const edges = collectDirectVarEdges(properties);
  const visiting = new Set<string>();
  const visited = new Set<string>();

  const visit = (property: string, path: string[]) => {
    if (visited.has(property)) {
      return;
    }
    if (visiting.has(property)) {
      throw new Error(`Alias cycle detected: ${[...path, property].join(' -> ')}`);
    }

    visiting.add(property);
    for (const dependency of edges.get(property) ?? []) {
      if (edges.has(dependency)) {
        visit(dependency, [...path, property]);
      }
    }
    visiting.delete(property);
    visited.add(property);
  };

  for (const property of edges.keys()) {
    visit(property, []);
  }
}

function expectRequiredTokens(properties: Record<string, string>): void {
  for (const token of REQUIRED_THEME_TOKENS) {
    expect(properties[token], `${token} should be defined`).toBeTruthy();
  }
}

describe('theme contract', () => {
  it('defines the required theme and canonical terminal tokens for each pack and effective mode', () => {
    const themeCss = readProjectFile('src/styles/theme.css');
    const serifCss = readProjectFile('src/styles/themes/serif.css');
    const vscodeCss = readProjectFile('src/styles/themes/vscode.css');

    const defaultDark = getMergedRuleProperties(themeCss, ':root');
    const defaultLight = getMergedRuleProperties(themeCss, "[data-theme='light']");
    const defaultSystemLight = getMergedRuleProperties(
      getRuleBody(themeCss, '@media (prefers-color-scheme: light)'),
      ":root:not([data-theme='dark'])",
    );

    const serifDark = getMergedRuleProperties(serifCss, "[data-theme-pack='serif']");
    const serifLight = getMergedRuleProperties(
      serifCss,
      "[data-theme-pack='serif'][data-theme='light']",
    );
    const serifSystemLight = getMergedRuleProperties(
      getRuleBody(serifCss, '@media (prefers-color-scheme: light)'),
      "[data-theme-pack='serif']:not([data-theme='dark'])",
    );

    const vscodeDark = getMergedRuleProperties(vscodeCss, "[data-theme-pack='vscode']");
    const vscodeLight = getMergedRuleProperties(
      vscodeCss,
      "[data-theme-pack='vscode'][data-theme='light']",
    );
    const vscodeSystemLight = getMergedRuleProperties(
      getRuleBody(vscodeCss, '@media (prefers-color-scheme: light)'),
      "[data-theme-pack='vscode']:not([data-theme='dark'])",
    );

    for (const properties of [
      defaultDark,
      defaultLight,
      defaultSystemLight,
      serifDark,
      serifLight,
      serifSystemLight,
      vscodeDark,
      vscodeLight,
      vscodeSystemLight,
    ]) {
      expectRequiredTokens(properties);
    }
  });

  it('keeps explicit and system light token blocks identical for every pack', () => {
    const themeCss = readProjectFile('src/styles/theme.css');
    const serifCss = readProjectFile('src/styles/themes/serif.css');
    const vscodeCss = readProjectFile('src/styles/themes/vscode.css');

    expect(parseCustomProperties(getRuleBody(themeCss, "[data-theme='light']"))).toEqual(
      parseCustomProperties(
        getRuleBody(
          getRuleBody(themeCss, '@media (prefers-color-scheme: light)'),
          ":root:not([data-theme='dark'])",
        ),
      ),
    );

    expect(
      parseCustomProperties(getRuleBody(serifCss, "[data-theme-pack='serif'][data-theme='light']")),
    ).toEqual(
      parseCustomProperties(
        getRuleBody(
          getRuleBody(serifCss, '@media (prefers-color-scheme: light)'),
          "[data-theme-pack='serif']:not([data-theme='dark'])",
        ),
      ),
    );

    expect(
      parseCustomProperties(
        getRuleBody(vscodeCss, "[data-theme-pack='vscode'][data-theme='light']"),
      ),
    ).toEqual(
      parseCustomProperties(
        getRuleBody(
          getRuleBody(vscodeCss, '@media (prefers-color-scheme: light)'),
          "[data-theme-pack='vscode']:not([data-theme='dark'])",
        ),
      ),
    );
  });

  it('uses only canonical terminal token names and has no alias cycles in theme declarations', () => {
    const files = [
      'src/styles/theme.css',
      'src/styles/themes/serif.css',
      'src/styles/themes/vscode.css',
    ] as const;

    for (const relativePath of files) {
      const css = readProjectFile(relativePath);
      expect(css).not.toMatch(LEGACY_TERMINAL_TOKEN_PATTERN);
      expectNoAliasCycles(parseCustomProperties(css));
    }
  });

  it('uses local font assets and validates the pre-paint pack against the shared allowlist', () => {
    const html = readProjectFile('index.html');

    expect(html).not.toContain('fonts.googleapis.com');
    expect(html).not.toContain('fonts.gstatic.com');
    expect(html).toContain("var allowedThemePacks = ['default', 'vscode', 'serif'];");
    expect(html).toContain("if (allowedThemePacks.includes(pack) && pack !== 'default') {");
  });

  it('keeps sash tokens and forced-colors support in the shared workbench contract', () => {
    const themeCss = readProjectFile('src/styles/theme.css');
    const workbenchCss = readProjectFile('src/styles/workbench.css');

    expect(themeCss).toContain('--sash-size: 4px;');
    expect(themeCss).toContain('--sash-hover-size: 4px;');
    expect(themeCss).toContain('--color-sash-hover: var(--color-focus);');
    expect(workbenchCss).toContain('@media (forced-colors: active)');
    expect(workbenchCss).toContain('background: var(--color-sash-hover);');
  });

  it('defines the canonical toast z-layer token in the root theme contract', () => {
    const themeCss = readProjectFile('src/styles/theme.css');
    const defaultDark = getMergedRuleProperties(themeCss, ':root');

    expect(defaultDark['--z-toast']).toBe('1200');
  });
});
