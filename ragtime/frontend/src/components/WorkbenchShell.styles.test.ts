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

describe('Workbench shell styles contract', () => {
  it('defines the compact full-width admin/auth shell and shared accessibility coverage', () => {
    const css = readSource('src/styles/workbench-admin.css');
    const appSource = readSource('src/App.tsx');
    const userMenuSource = readSource('src/components/UserMenu.tsx');
    const usersPanelSource = readSource('src/components/UsersPanel.tsx');
    const webglSource = readSource('src/components/WebGLGradient/WebGLGradient.tsx');

    expect(css).toMatch(/\.topnav\s*\{[\s\S]*min-height:\s*var\(--workbench-titlebar-height\);/);
    expect(css).toMatch(/\.app-shell-webgl-background::before\s*\{[\s\S]*var\(--color-workbench\)/);
    expect(css).toMatch(
      /\[data-workbench-shell='authenticated'\]\s*>\s*\.container\s*\{[\s\S]*max-width:\s*none;/,
    );
    expect(css).toMatch(
      /\[data-workbench-route-root='settings'\],[\s\S]*\[data-workbench-route-root='indexer'\],[\s\S]*\[data-workbench-route-root='tools'\],[\s\S]*\[data-workbench-route-root='users'\]/,
    );
    expect(css).toMatch(/\.appearance-theme-card\s*\{[\s\S]*align-items:\s*stretch;/);
    expect(css).toMatch(
      /\.appearance-theme-card\s*\{[\s\S]*background:\s*var\([\s\S]*--appearance-card-background,/,
    );
    expect(css).toMatch(
      /\.appearance-theme-card-preview\s*\{[\s\S]*background:\s*var\(--appearance-card-background\);/,
    );
    expect(css).toMatch(
      /\.appearance-theme-card-surface\s*\{[\s\S]*background:\s*var\(--appearance-card-surface\);/,
    );
    expect(css).toMatch(
      /\.appearance-theme-card-accent\s*\{[\s\S]*background:\s*var\(--appearance-card-primary\);/,
    );
    expect(css).toMatch(
      /\.appearance-theme-card-preview-copy\s*\{[\s\S]*color:\s*var\(--appearance-card-text\);/,
    );
    expect(css).toMatch(/\.user-menu-dropdown\s*\{[\s\S]*background:\s*var\(--color-widget\);/);
    expect(css).toMatch(
      /\.topnav\s*\{[\s\S]*background:\s*var\(--color-chrome\);[\s\S]*background:\s*color-mix\(in srgb, var\(--color-chrome\) 94%, transparent\);/,
    );
    expect(css).toMatch(
      /input:not\(\[type='checkbox'\]\):not\(\[type='radio'\]\):not\(\[type='range'\]\):not\(\[type='hidden'\]\)/,
    );
    expect(css).toMatch(/\.modal(?:-content)?\s*\{[\s\S]*background:\s*var\(--color-widget\);/);
    expect(css).toMatch(/\.toast(?:-container|-viewport|)\S*[\s\S]*var\(--color-widget\)/);
    expect(css).toMatch(/\.katex(?:-display)?[\s\S]*var\(--color-editor\)/);
    expect(css).toContain('@media (max-width: 768px)');
    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toContain('@media (forced-colors: active)');
    expect(css).toContain('@media print');

    expect(appSource).toContain('data-workbench-shell="authenticated"');
    expect(appSource).toContain('data-workbench-route-root');
    expect(appSource).toContain('topnav-overflow-trigger');
    expect(appSource).not.toContain('backdrop-filter:');
    expect(userMenuSource).toContain('ThemeChromeIcon');
    expect(usersPanelSource).toContain('subscribeToThemeChanges');
    expect(usersPanelSource).not.toContain('new MutationObserver');
    expect(webglSource).toContain('subscribeToThemeChanges');
    expect(webglSource).not.toContain('new MutationObserver');

    expect(userMenuSource).not.toContain('appearance-theme-card');
  });
});
