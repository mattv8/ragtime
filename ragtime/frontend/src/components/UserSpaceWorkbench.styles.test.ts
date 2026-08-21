// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), 'utf8');
}

function splitSelectorList(selectorText: string): string[] {
  const selectors: string[] = [];
  let current = '';
  let parenDepth = 0;

  for (const character of selectorText) {
    if (character === '(') parenDepth += 1;
    if (character === ')') parenDepth -= 1;

    if (character === ',' && parenDepth === 0) {
      const trimmed = current.trim();
      if (trimmed) selectors.push(trimmed);
      current = '';
      continue;
    }

    current += character;
  }

  const trimmed = current.trim();
  if (trimmed) selectors.push(trimmed);

  return selectors;
}

function collectRuleSelectors(css: string): string[] {
  const selectors: string[] = [];
  let index = 0;

  while (index < css.length) {
    const braceIndex = css.indexOf('{', index);
    if (braceIndex === -1) break;

    const prelude = css
      .slice(index, braceIndex)
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .trim();
    let depth = 1;
    let cursor = braceIndex + 1;

    while (cursor < css.length && depth > 0) {
      if (css[cursor] === '{') depth += 1;
      if (css[cursor] === '}') depth -= 1;
      cursor += 1;
    }

    const body = css.slice(braceIndex + 1, cursor - 1);
    if (!prelude) {
      index = cursor;
      continue;
    }

    if (!prelude.startsWith('@')) {
      selectors.push(...splitSelectorList(prelude));
    } else if (body.includes('{')) {
      selectors.push(...collectRuleSelectors(body));
    }

    index = cursor;
  }

  return selectors;
}

function expectModernScopedSelectors(css: string): void {
  const selectors = collectRuleSelectors(css);

  expect(selectors.length).toBeGreaterThan(0);
  for (const selector of selectors) {
    expect(selector).toMatch(
      /^\[data-theme-pack='modern'\](?:\[data-theme='light'\]|:not\(\[data-theme='dark'\]\))?\s/,
    );
  }
}

describe('User Space workbench styles contract', () => {
  it('covers reduced motion, forced colors, focus rings, and Modern light editor depth', () => {
    const css = readSource('src/styles/workbench-userspace.css');

    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\.userspace-toolbar-tab[\s\S]*\.userspace-tree-row[\s\S]*\.userspace-preview-frame[\s\S]*\.userspace-status-overlay[\s\S]*transition:\s*none !important;/,
    );
    expect(css).toContain('@media (forced-colors: active)');
    expect(css).toMatch(
      /@media\s*\(forced-colors:\s*active\)[\s\S]*\.userspace-toolbar[\s\S]*\.userspace-file-sidebar[\s\S]*\.userspace-code-editor[\s\S]*\.userspace-preview-section[\s\S]*\.userspace-runtime-terminal-wrap[\s\S]*\.userspace-snapshot-diff-column/,
    );
    expect(css).toMatch(
      /\.userspace-layout\s*:is\(button,\s*\[role='button'\],\s*\[role='tab'\],\s*input,\s*select,\s*textarea\):focus-visible[\s\S]*outline:\s*2px solid var\(--color-focus\);[\s\S]*outline-offset:\s*2px;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\[data-theme='light'\]\s+\.userspace-code-editor/,
    );
    expect(css).toMatch(
      /@media\s*\(prefers-color-scheme:\s*light\)[\s\S]*\[data-theme-pack='modern'\]:not\(\[data-theme='dark'\]\)\s+\.userspace-code-editor/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-layout\s*\{[\s\S]*padding:\s*0;[\s\S]*max-width:\s*none;[\s\S]*margin:\s*0;[\s\S]*border:\s*none;[\s\S]*border-radius:\s*0;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-toolbar\s*\{[\s\S]*padding:\s*0\s+var\(--space-sm\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-workspace-trigger\s*\{[\s\S]*padding:\s*0\s+var\(--space-sm\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-chat-section\s*\{[\s\S]*background:\s*transparent;[\s\S]*border:\s*none;/,
    );
  });

  it('scopes every User Space workbench selector to the Modern theme pack, including stateful and responsive rules', () => {
    const css = readSource('src/styles/workbench-userspace.css');

    expectModernScopedSelectors(css);
    expect(css).toMatch(
      /@media\s*\(max-width:\s*768px\)[\s\S]*\[data-theme-pack='modern'\]\s+\.userspace-layout/,
    );
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\[data-theme-pack='modern'\]\s+\.userspace-toolbar-tab/,
    );
    expect(css).toMatch(
      /@media\s*\(forced-colors:\s*active\)[\s\S]*\[data-theme-pack='modern'\]\s+\.userspace-toolbar/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-layout\s*:is\(button,\s*\[role='button'\],\s*\[role='tab'\],\s*input,\s*select,\s*textarea\):focus-visible/,
    );
  });
});
