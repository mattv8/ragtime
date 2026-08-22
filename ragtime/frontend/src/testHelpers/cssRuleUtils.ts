import { expect } from 'vitest';

/**
 * Extracts the body of a CSS rule by selector.
 * Handles nested rules and tracks brace depth.
 */
export function getRuleBody(css: string, selector: string): string {
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

/**
 * Splits a CSS selector list into individual selectors.
 * Respects parentheses (e.g., in :is(), :where(), etc.).
 */
export function splitSelectorList(selectorText: string): string[] {
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

/**
 * Collects all rule selectors from CSS, including those in nested rules.
 * Recursively handles media queries and other at-rules.
 */
export function collectRuleSelectors(css: string): string[] {
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

/**
 * Validates that all selectors in CSS are scoped to the Modern theme pack.
 * Optionally allows light/dark theme qualifiers.
 */
export function expectModernScopedSelectors(css: string): void {
  const selectors = collectRuleSelectors(css);

  expect(selectors.length).toBeGreaterThan(0);
  for (const selector of selectors) {
    expect(selector).toMatch(
      /^\[data-theme-pack='modern'\](?:\[data-theme='light'\]|:not\(\[data-theme='dark'\]\))?\s/,
    );
  }
}
