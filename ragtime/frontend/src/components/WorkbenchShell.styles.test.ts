// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { getRuleBody } from '@/testHelpers/cssRuleUtils';

function readSource(relativePath: string): string {
  return readFileSync(join(cwd(), relativePath), 'utf8');
}

describe('Workbench shell styles contract', () => {
  it('defines the compact full-width admin/auth shell and shared accessibility coverage', () => {
    const css = readSource('src/styles/workbench-admin.css');
    const layoutCss = readSource('src/styles/layout.css');
    const appSource = readSource('src/App.tsx');
    const userMenuSource = readSource('src/components/UserMenu.tsx');
    const usersPanelSource = readSource('src/components/UsersPanel.tsx');
    const webglSource = readSource('src/components/WebGLGradient/WebGLGradient.tsx');
    const mobileCss = getRuleBody(css, '@media (max-width: 768px)');
    const reducedMotionCss = getRuleBody(css, '@media (prefers-reduced-motion: reduce)');
    const forcedColorsCss = getRuleBody(css, '@media (forced-colors: active)');
    const printCss = getRuleBody(css, '@media print');

    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.topnav\s*\{[\s\S]*min-height:\s*var\(--workbench-titlebar-height\);[\s\S]*padding:\s*0\s+var\(--space-sm\);/,
    );
    expect(css).not.toMatch(/^\.topnav\s*\{/m);
    expect(css).not.toContain('.app-shell-webgl-background::before');
    expect(css).not.toContain('.app-shell > .webgl-motion-toggle');
    expect(layoutCss).toMatch(/\.app-shell-webgl-background::before\s*\{/);
    expect(layoutCss).toMatch(/\.app-shell\s*>\s*\.webgl-motion-toggle\s*\{/);
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+#workbench-shell-stack\s*>\s*\.topnav,[\s\S]*>\s*\.container\s*\{[\s\S]*width:\s*100%;[\s\S]*max-width:\s*none;[\s\S]*margin:\s*0;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.security-banner,[\s\S]*\.warnings-banner,[\s\S]*\.config-banner\s*\{[\s\S]*width:\s*100%;[\s\S]*margin:\s*0\s+0\s+var\(--workbench-gap\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.topnav-link\.active\s*\{[\s\S]*color:\s*var\(--color-primary-text\);[\s\S]*background:\s*var\(--color-primary\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.topnav-link\.active:hover,[\s\S]*\.tab\.active:hover[\s\S]*\{[\s\S]*background:\s*var\(--color-primary-hover\);[\s\S]*color:\s*var\(--color-primary-text\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\[data-workbench-route-root='settings'\],[\s\S]*\[data-theme-pack='modern'\]\s+\[data-workbench-route-root='indexer'\],[\s\S]*\[data-theme-pack='modern'\]\s+\[data-workbench-route-root='tools'\],[\s\S]*\[data-theme-pack='modern'\]\s+\[data-workbench-route-root='users'\]/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\[data-workbench-route-root='settings'\][\s\S]*display:\s*flex;[\s\S]*flex-direction:\s*column;[\s\S]*gap:\s*var\(--workbench-gap\);/,
    );
    expect(css).toMatch(
      />\s*\*\s*\{[\s\S]*width:\s*100%;[\s\S]*max-width:\s*none;[\s\S]*margin-inline:\s*0;[\s\S]*margin-block:\s*0;/,
    );
    expect(css).not.toContain('margin: 0 auto;');
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+#workbench-indexer-route,[\s\S]*\[data-theme-pack='modern'\]\s+\[data-workbench-route-root='indexer'\][\s\S]*background:\s*var\(--color-panel\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.appearance-theme-card\s*\{[\s\S]*align-items:\s*stretch;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.appearance-theme-card\s*\{[\s\S]*background:\s*var\([\s\S]*--appearance-card-background,/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.appearance-theme-card-preview\s*\{[\s\S]*background:\s*var\(--appearance-card-background\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.appearance-theme-card-surface\s*\{[\s\S]*background:\s*var\(--appearance-card-surface\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.appearance-theme-card-accent\s*\{[\s\S]*background:\s*var\(--appearance-card-primary\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.appearance-theme-card-preview-copy\s*\{[\s\S]*color:\s*var\(--appearance-card-text\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.user-menu-dropdown\s*\{[\s\S]*background:\s*var\(--color-widget\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.topnav\s*\{[\s\S]*background:\s*var\(--color-chrome\);[\s\S]*background:\s*color-mix\(in srgb, var\(--color-chrome\) 94%, transparent\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+input:not\(\[type='checkbox'\]\):not\(\[type='radio'\]\):not\(\[type='range'\]\):not\(\[type='hidden'\]\)/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.modal(?:-content)?\s*\{[\s\S]*background:\s*var\(--color-widget\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\][\s\S]*\.toast(?:-container|-viewport|)\S*[\s\S]*var\(--color-widget\)/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\][\s\S]*\.katex(?:-display)?[\s\S]*var\(--color-editor\)/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.settings-filter-search\s*\{[\s\S]*background:\s*var\(--color-input-bg\);[\s\S]*border:\s*var\(--workbench-container-border\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.settings-filter-search\s+input\s*\{[\s\S]*background:\s*transparent !important;[\s\S]*border:\s*none !important;/,
    );
    expect(css).toContain('@media (max-width: 768px)');
    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toContain('@media (forced-colors: active)');
    expect(css).toContain('@media print');
    expect(mobileCss).toContain("[data-theme-pack='modern'] .topnav");
    expect(mobileCss).toContain('min-height: 44px;');
    expect(reducedMotionCss).toContain("[data-theme-pack='modern'] .topnav");
    expect(forcedColorsCss).toContain("[data-theme-pack='modern'] .topnav");
    expect(printCss).toContain("[data-theme-pack='modern'] body");

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

  it('provides layout-neutral stack wrappers and a bounded ordered Modern stack with neutral buttons', () => {
    const layoutCss = readSource('src/styles/layout.css');
    const adminCss = readSource('src/styles/workbench-admin.css');
    const lockedLayoutBlock = getRuleBody(layoutCss, '.app-shell-locked #workbench-shell-stack');

    expect(layoutCss).toContain('#workbench-shell-stack');
    expect(layoutCss).toContain('#workbench-warning-stack');
    expect(layoutCss).toMatch(
      /#workbench-shell-stack,\s*#workbench-warning-stack\s*\{[\s\S]*display:\s*contents;/,
    );
    expect(lockedLayoutBlock).toMatch(/flex:\s*1;[\s\S]*min-height:\s*0;/);

    expect(adminCss).toMatch(
      /\[data-theme-pack='modern'\]\s+#workbench-shell-stack\s*\{[\s\S]*display:\s*flex;[\s\S]*width:\s*100%/,
    );
    expect(adminCss).toMatch(
      /\[data-theme-pack='modern'\]\s+#workbench-warning-stack\s*\{[\s\S]*order:\s*-1;/,
    );
    expect(adminCss).toMatch(
      /\[data-theme-pack='modern'\]\s+#workbench-shell-stack\s*>\s*\.topnav,[\s\S]*>\s*\.container\s*\{[\s\S]*width:\s*100%;[\s\S]*max-width:\s*none;[\s\S]*margin:\s*0;/,
    );
    expect(adminCss).not.toMatch(
      /\[data-theme-pack='modern'\]\s+\[data-workbench-shell='authenticated'\]\s*>\s*\.container\s*\{/,
    );
    expect(adminCss).toMatch(
      /\[data-theme-pack='modern'\]\s+:where\(button:not\(\.chat-message-navigator-tick\),\s*\[role='button'\]:not\(\.chat-message-navigator-tick\)\)[\s\S]*\{[\s\S]*background:\s*transparent;[\s\S]*border-color:\s*transparent;/,
    );
    expect(adminCss).toMatch(
      /\[data-theme-pack='modern'\]\s+:where\(button:not\(\.chat-message-navigator-tick\):hover:not\(:disabled\),\s*\[role='button'\]:not\(\.chat-message-navigator-tick\):hover:not\(:disabled\)\)[\s\S]*\{[\s\S]*background:\s*var\(--color-surface-hover\);/,
    );
    expect(adminCss).toMatch(
      /:is\([\s\S]*\.active[\s\S]*\.is-selected[\s\S]*\[aria-pressed='true'\][\s\S]*\[aria-selected='true'\][\s\S]*\[data-active='true'\][\s\S]*\)[\s\S]*\{[\s\S]*background:\s*var\(--color-primary\);/,
    );
  });
});
