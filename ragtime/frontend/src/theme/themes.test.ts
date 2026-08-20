// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { resolveThemePackId } from './applyThemePack';
import { THEME_PACKS } from './themes';

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
  it('registers the theme packs in default, vscode, serif order and resolves fallbacks', () => {
    expect(THEME_PACKS.map((pack) => pack.id)).toEqual(['default', 'vscode', 'serif']);
    expect(resolveThemePackId('vscode', null)).toBe('vscode');
    expect(resolveThemePackId('unknown-pack', 'vscode')).toBe('vscode');
    expect(resolveThemePackId('unknown-pack', 'unknown-pack')).toBe('default');
  });

  it('keeps the vscode explicit-light and system-light token sets identical', () => {
    const css = readFileSync(join(cwd(), 'src/styles/themes/vscode.css'), 'utf8');
    const explicitLight = parseCustomProperties(
      getRuleBody(css, "[data-theme-pack='vscode'][data-theme='light']"),
    );
    const mediaBody = getRuleBody(css, '@media (prefers-color-scheme: light)');
    const systemLight = parseCustomProperties(
      getRuleBody(mediaBody, "[data-theme-pack='vscode']:not([data-theme='dark'])"),
    );

    expect(Object.keys(explicitLight).length).toBeGreaterThan(0);
    expect(systemLight).toEqual(explicitLight);
  });
});
