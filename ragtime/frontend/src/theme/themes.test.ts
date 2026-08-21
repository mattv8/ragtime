// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { resolveThemePackId } from './applyThemePack';
import { THEME_PACKS, getThemePack } from './themes';

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

function parseCustomProperties(block: string): Record<string, string> {
  return Object.fromEntries(
    Array.from(block.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g), (match) => [
      match[1],
      match[2].trim(),
    ]),
  );
}

describe('theme pack registry', () => {
  it('registers the theme packs in default, modern, serif order and resolves fallbacks', () => {
    expect(THEME_PACKS.map((pack) => pack.id)).toEqual(['default', 'modern', 'serif']);
    expect(getThemePack('modern').label).toBe('Modern');
    expect(resolveThemePackId('modern', null)).toBe('modern');
    expect(resolveThemePackId('vscode', null)).toBe('modern');
    expect(resolveThemePackId(null, 'vscode')).toBe('modern');
    expect(resolveThemePackId('unknown-pack', 'unknown-pack')).toBe('default');
  });

  it('keeps the Modern explicit-light and system-light token sets identical', () => {
    const css = readFileSync(join(cwd(), 'src/styles/themes/modern.css'), 'utf8');
    const explicitLight = parseCustomProperties(
      getRuleBody(css, "[data-theme-pack='modern'][data-theme='light']"),
    );
    const mediaBody = getRuleBody(css, '@media (prefers-color-scheme: light)');
    const systemLight = parseCustomProperties(
      getRuleBody(mediaBody, "[data-theme-pack='modern']:not([data-theme='dark'])"),
    );

    expect(Object.keys(explicitLight).length).toBeGreaterThan(0);
    expect(systemLight).toEqual(explicitLight);
  });

  it('accepts vscode as a legacy alias in the pre-paint script and normalizes it', () => {
    const html = readFileSync(join(cwd(), 'index.html'), 'utf8');

    expect(html).toContain("var allowedThemePacks = ['default', 'modern', 'serif'];");
    expect(html).toContain("pack === 'vscode'");
    expect(html).toContain("pack = 'modern'");
  });
});
