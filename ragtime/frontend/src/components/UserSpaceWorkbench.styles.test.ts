// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { expectModernScopedSelectors } from '@/testHelpers/cssRuleUtils';

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), 'utf8');
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
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-preview-section\s*\{[\s\S]*padding:\s*0;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-preview-frame-wrap\s*\{[\s\S]*border:\s*none;[\s\S]*border-radius:\s*0;/,
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
