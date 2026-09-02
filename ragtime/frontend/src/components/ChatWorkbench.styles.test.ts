// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { expectModernScopedSelectors, getRuleBody } from '@/testHelpers/cssRuleUtils';

const cssPath = join(cwd(), 'src/styles/workbench-chat.css');
const workbenchCssPath = join(cwd(), 'src/styles/workbench.css');
const responsiveCssPath = join(cwd(), 'src/styles/responsive.css');
const adminCssPath = join(cwd(), 'src/styles/workbench-admin.css');
const chatPanelPath = join(cwd(), 'src/components/ChatPanel.tsx');
const publicSharedChatPath = join(cwd(), 'src/components/PublicSharedChatView.tsx');

function read(filePath: string): string {
  return readFileSync(filePath, 'utf8');
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
    const modernErrorRule = getRuleBody(css, "[data-theme-pack='modern'] .chat-error");
    const modernToolCallsRule = getRuleBody(css, "[data-theme-pack='modern'] .chat-tool-calls");

    expect(modernErrorRule).toMatch(/margin-inline:\s*0;/);
    expect(modernToolCallsRule).not.toMatch(/border-(?:top|block-start)\s*:/);
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-branch-wrapper-assistant\s*\{[\s\S]*width:\s*80%;[\s\S]*max-width:\s*80%;[\s\S]*align-self:\s*flex-start;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.chat-branch-wrapper-assistant\s+\.chat-message-assistant,[\s\S]*width:\s*100%;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.tool-call\.tool-call-datatable,[\s\S]*background:\s*transparent;[\s\S]*border:\s*none;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.tool-call\.tool-call-chart,[\s\S]*background:\s*transparent;[\s\S]*border:\s*none;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.tool-call:not\(:has\(\.tool-call-details\)\),[\s\S]*border-color:\s*transparent;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.tool-call:has\(\.tool-call-details\),[\s\S]*border:\s*var\(--workbench-container-border\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.reasoning-block\s+\.tool-call:not\(:has\(\.tool-call-details\)\),[\s\S]*border-color:\s*transparent;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.reasoning-block\s+\.tool-call:has\(\.tool-call-details\),[\s\S]*border:\s*var\(--workbench-container-border\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.reasoning-embedded-tool\s+\.tool-call:not\(:has\(\.tool-call-details\)\)[\s\S]*margin-inline-start:\s*calc\(var\(--space-xs\)\s*\*\s*-1\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.reasoning-embedded-tool\s+\.tool-call:not\(:has\(\.tool-call-details\)\)[\s\S]*padding:\s*4px\s+var\(--space-xs\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.reasoning-embedded-tool\.chat-tool-calls\s*\{[\s\S]*margin:\s*0\.5em\s+0;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.tool-call-userspace-diff-card\s*\{[\s\S]*background:\s*var\(--color-editor\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.markdown-content\s+table\s*\{[\s\S]*background:\s*var\(--color-editor\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.markdown-content\s+\.markdown-codeblock\s+pre,[\s\S]*border:\s*none;/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.reasoning-block\s*\{[\s\S]*background:\s*var\(--color-widget\);[\s\S]*border-radius:\s*var\(--workbench-control-radius\);/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.live-data-refresh-btn\s+\.theme-chrome-icon,[\s\S]*transform:\s*translateY\(1px\);/,
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
      /@media\s*\(max-width:\s*1024px\)[\s\S]*\[data-theme-pack='modern'\]\s+\.chat-branch-wrapper-assistant\s*\{[\s\S]*width:\s*100%;[\s\S]*max-width:\s*100%;/,
    );
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
    expect(chatPanel).toMatch(
      /className="chat-branch-wrapper chat-branch-wrapper-assistant chat-branch-wrapper-streaming"/,
    );
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
