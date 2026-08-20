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

describe('User Space workbench styles contract', () => {
  it('covers reduced motion, forced colors, focus rings, and VS Code light editor depth', () => {
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
      /\[data-theme-pack='vscode'\]\[data-theme='light'\]\s+\.userspace-code-editor/,
    );
    expect(css).toMatch(
      /@media\s*\(prefers-color-scheme:\s*light\)[\s\S]*\[data-theme-pack='vscode'\]:not\(\[data-theme='dark'\]\)\s+\.userspace-code-editor/,
    );
  });
});
