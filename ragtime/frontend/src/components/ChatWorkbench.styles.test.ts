// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

const cssPath = join(cwd(), 'src/styles/workbench-chat.css');
const workbenchCssPath = join(cwd(), 'src/styles/workbench.css');
const responsiveCssPath = join(cwd(), 'src/styles/responsive.css');
const chatPanelPath = join(cwd(), 'src/components/ChatPanel.tsx');
const publicSharedChatPath = join(cwd(), 'src/components/PublicSharedChatView.tsx');

function read(filePath: string): string {
  return readFileSync(filePath, 'utf8');
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

describe('Chat workbench surface contract', () => {
  it('defines token-driven chat workbench surfaces and responsive states', () => {
    const css = read(cssPath);
    const workbenchCss = read(workbenchCssPath);
    const responsiveCss = read(responsiveCssPath);
    const responsiveMax768 = getRuleBody(responsiveCss, '@media (max-width: 768px)');
    const responsiveMobileChatSidebar = getRuleBody(responsiveMax768, '.chat-sidebar');
    const responsiveMobileChatMain = getRuleBody(responsiveMax768, '.chat-main');

    expect(css).toMatch(/\.chat-page-container\s*\{[\s\S]*width:\s*100%;[\s\S]*max-width:\s*none;/);
    expect(css).toMatch(
      /\.chat-page-container\.chat-page-fullscreen\s*\{[\s\S]*position:\s*fixed;[\s\S]*inset:\s*var\(--workbench-padding\);/,
    );
    expect(css).toMatch(
      /\.chat-panel(?:,\s*\.chat-panel-shared)?\s*\{[\s\S]*background:\s*var\(--color-editor\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(/\.chat-sidebar\s*\{[\s\S]*background:\s*var\(--color-chrome\);/);
    expect(css).toMatch(/\.chat-main\s*\{[\s\S]*background:\s*var\(--color-editor\);/);
    expect(css).toMatch(
      /\.chat-header\s*\{[\s\S]*min-height:\s*var\(--workbench-titlebar-height\);/,
    );
    expect(css).toMatch(/\.chat-input-area\s*\{[\s\S]*background:\s*var\(--color-panel\);/);
    expect(css).toMatch(/\.markdown-content\s*\{[\s\S]*color:\s*var\(--color-text-primary\);/);
    expect(css).toMatch(
      /\.chat-tool-calls\s*\{[\s\S]*border-top:\s*1px solid var\(--color-border\);/,
    );
    expect(css).toMatch(/\.datatable-container\s*\{[\s\S]*background:\s*var\(--color-editor\);/);
    expect(css).toMatch(/\.chat-panel-embedded\s*\{[\s\S]*border-radius:\s*0;/);
    expect(css).toMatch(/\.chat-panel-shared\s*\{[\s\S]*width:\s*100%;/);
    expect(css).toContain('@media (max-width: 768px)');
    expect(css).toMatch(
      /@media\s*\(max-width:\s*768px\)[\s\S]*\.chat-sidebar\s*\{[\s\S]*position:\s*absolute;/,
    );
    expect(css).toMatch(
      /@media\s*\(max-width:\s*768px\)[\s\S]*\.chat-panel\s*>\s*\.resize-handle-horizontal\.chat-resize-handle,[\s\S]*\.chat-in-chat-search-resize-handle\s*\{[\s\S]*display:\s*none;/,
    );
    expect(css).toContain('@media (prefers-reduced-motion: reduce)');
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\.chat-panel[\s\S]*transition:\s*none !important;/,
    );
    expect(css).toContain('@media (forced-colors: active)');
    expect(workbenchCss).toMatch(/\.resize-handle::before\s*\{[\s\S]*pointer-events:\s*auto;/);
    expect(workbenchCss).toMatch(/\.resize-handle::after\s*\{[\s\S]*pointer-events:\s*none;/);
    expect(workbenchCss).not.toContain('.chat-in-chat-search-resize-handle');
    expect(css).toContain('.chat-in-chat-search-resize-handle-horizontal');
    expect(css).toContain('.chat-in-chat-search-resize-handle-vertical');
    expect(css).toMatch(/#chat-mobile-sidebar-toggle\s*\{[\s\S]*display:\s*none;/);
    expect(css).toMatch(
      /@media\s*\(max-width:\s*768px\)[\s\S]*#chat-mobile-sidebar-toggle\s*\{[\s\S]*display:\s*inline-flex;[\s\S]*z-index:\s*26;/,
    );
    expect(responsiveCss).not.toContain('.userspace-content > .resize-handle-horizontal');
    expect(responsiveCss).not.toContain('.userspace-left-pane > .resize-handle-vertical');
    expect(responsiveCss).not.toContain('.userspace-editor-section > .resize-handle-horizontal');
    expect(responsiveMobileChatSidebar).not.toContain('z-index');
    expect(responsiveMobileChatMain).not.toContain('z-index');
    expect(css).not.toMatch(/#[0-9a-fA-F]{3,8}/);
  });

  it('uses shared theme subscription, selective chrome icons, and stable workbench hooks', () => {
    const chatPanel = read(chatPanelPath);
    const publicSharedChat = read(publicSharedChatPath);

    expect(chatPanel).toContain('subscribeToThemeChanges');
    expect(chatPanel).toContain('ThemeChromeIcon');
    expect(chatPanel).toContain('layout-sidebar-left');
    expect(chatPanel).toContain('layout-sidebar-left-off');
    expect(chatPanel).toMatch(/codicon="refresh"/);
    expect(chatPanel).toMatch(/codicon="close"/);
    expect(chatPanel).toMatch(/codicon="ellipsis"/);
    expect(chatPanel).toMatch(/id="chat-workbench-panel"/);
    expect(chatPanel).toMatch(/id="chat-workbench-sidebar"/);
    expect(chatPanel).toMatch(/id="chat-workbench-main"/);
    expect(chatPanel).toMatch(/id="chat-workbench-header"/);
    expect(chatPanel).toMatch(/id="chat-workbench-composer"/);
    expect(chatPanel).toMatch(/id="chat-mobile-sidebar-toggle"/);
    expect(chatPanel).toMatch(/id="chat-prompt-debug-modal"/);
    expect(chatPanel).toMatch(/id="chat-compaction-review-modal"/);
    expect(publicSharedChat).toMatch(/id="public-shared-chat-view"/);
    expect(publicSharedChat).toMatch(/id="public-shared-chat-panel"/);
    expect(publicSharedChat).toMatch(/id="public-shared-chat-composer"/);
  });

  it('wires chat sidebar resizing through the drag-end commit callback', () => {
    const chatPanel = read(chatPanelPath);

    expect(chatPanel).toMatch(/const commitResizeSidebar = useCallback\(/);
    expect(chatPanel).toMatch(/onResizeEnd=\{commitResizeSidebar\}/);
  });
});
