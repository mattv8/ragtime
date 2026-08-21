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
const adminCssPath = join(cwd(), 'src/styles/workbench-admin.css');
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
      /\[data-theme-pack='modern'\]\s+\.chat-panel\s*\{[\s\S]*display:\s*flex;[\s\S]*gap:\s*var\(--workbench-gap\);[\s\S]*background:\s*transparent;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-sidebar\s*\{[\s\S]*display:\s*flex;[\s\S]*gap:\s*var\(--workbench-gap\);[\s\S]*background:\s*transparent;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-conversation-list\s*\{[\s\S]*background:\s*var\(--color-panel\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-main\s*\{[\s\S]*display:\s*flex;[\s\S]*flex-direction:\s*column;[\s\S]*gap:\s*var\(--workbench-gap\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-message-region\s*\{[\s\S]*background:\s*var\(--color-panel\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-message\.chat-message-assistant,[\s\S]*background:\s*var\(--color-panel\);/,
    );
    expect(css).toMatch(
      /\.chat-header\s*\{[\s\S]*min-height:\s*var\(--workbench-titlebar-height\);[\s\S]*padding:\s*0\s+var\(--space-sm\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-input-wrapper\s*\{[\s\S]*background:\s*var\(--color-panel\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-conversation-search-input\s*\{[\s\S]*min-height:\s*var\(--workbench-titlebar-height\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-conversation-search-input::placeholder\s*\{[\s\S]*color:\s*var\(--color-text-muted\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-show-older-btn\s*\{[\s\S]*background:\s*var\(--color-input-bg\);[\s\S]*border:\s*var\(--workbench-container-border\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    const adminCss = read(adminCssPath);
    expect(adminCss).toMatch(
      /\[data-theme-pack='modern'\]\s+:where\(button:not\(\.chat-message-navigator-tick\),\s*\[role='button'\]:not\(\.chat-message-navigator-tick\)\)/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-welcome,\s*\[data-theme-pack='modern'\]\s+\.chat-empty-state\s*\{[\s\S]*background:\s*transparent;[\s\S]*border:\s*none;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-message-navigator-rail\s*\{[\s\S]*background:\s*transparent;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-message-navigator-tick\s*\{[\s\S]*display:\s*block;/,
    );
    expect(css).toMatch(/\.markdown-content\s*\{[\s\S]*color:\s*var\(--color-text-primary\);/);
    expect(css).toMatch(
      /\.chat-tool-calls\s*\{[\s\S]*border-top:\s*var\(--workbench-container-border\);/,
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
    expect(responsiveMobileChatSidebar).not.toContain('z-index');
    expect(responsiveMobileChatMain).not.toContain('z-index');
    expect(css).not.toMatch(/#[0-9a-fA-F]{3,8}/);
  });

  it('scopes every chat workbench selector to the Modern theme pack, including media queries', () => {
    const css = read(cssPath);

    expectModernScopedSelectors(css);
    expect(css).toMatch(
      /@media\s*\(max-width:\s*768px\)[\s\S]*\[data-theme-pack='modern'\]\s+\.chat-sidebar/,
    );
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\[data-theme-pack='modern'\]\s+\.chat-panel/,
    );
    expect(css).toMatch(
      /@media\s*\(forced-colors:\s*active\)[\s\S]*\[data-theme-pack='modern'\]\s+\.chat-panel/,
    );
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

  it('unifies embedded chat control surfaces under one Modern variable', () => {
    const css = read(cssPath);

    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-panel-embedded\s*\{[\s\S]*--chat-embedded-control-surface:\s*var\(--color-panel\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-panel-embedded\s+\.chat-header\s*\{[\s\S]*background:\s*var\(--color-chrome\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-panel-embedded\s+\.chat-workspace-conversation-trigger[\s\S]*background:\s*var\(--chat-embedded-control-surface\)/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-panel-embedded\s+\.chat-input-wrapper\s*\{[\s\S]*background:\s*var\(--color-panel\);[\s\S]*border-radius:\s*var\(--workbench-surface-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-panel-embedded\s+\.chat-header-actions\s+:is\([\s\S]*?\.model-selector-trigger\)\s*\{[\s\S]*?background:\s*var\(--chat-embedded-control-surface\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-panel-embedded\s+\.chat-input-wrapper:focus-within\s*\{[\s\S]*border-color:\s*var\(--color-focus\);/,
    );
  });

  it('wires chat sidebar resizing through the drag-end commit callback', () => {
    const chatPanel = read(chatPanelPath);

    expect(chatPanel).toMatch(/const commitResizeSidebar = useCallback\(/);
    expect(chatPanel).toMatch(/onResizeEnd=\{commitResizeSidebar\}/);
  });
});
