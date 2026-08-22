// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { getRuleBody } from '@/testHelpers/cssRuleUtils';

describe('ChatMessageNavigator styles contract', () => {
  it('defines the centered navigator overlay, bounded tick stack, and responsive accessibility rules', () => {
    const css = readFileSync(join(cwd(), 'src/styles/chat.css'), 'utf8');
    const navigatorItemRule = getRuleBody(css, '.chat-message-navigator-item');
    const navigatorItemTextRule = getRuleBody(css, '.chat-message-navigator-item-text');
    const navigatorTickRule = getRuleBody(css, '.chat-message-navigator-tick');

    expect(css).toMatch(
      /\.chat-message-region\s*\{[\s\S]*display:\s*flex;[\s\S]*min-height:\s*0;[\s\S]*position:\s*relative;/,
    );
    expect(css).toMatch(
      /\.chat-message-navigator\s*\{[\s\S]*position:\s*absolute;[\s\S]*right:\s*0;[\s\S]*bottom:\s*var\(--space-lg\);[\s\S]*pointer-events:\s*none;/,
    );
    expect(css).toMatch(/--chat-message-navigator-offset:\s*var\(--space-sm\);/);
    expect(css).toMatch(/--chat-message-navigator-rail-width:\s*2rem;/);
    expect(css).toMatch(
      /\.chat-message-navigator-trigger\s*\{[\s\S]*pointer-events:\s*auto;[\s\S]*top:\s*50%;[\s\S]*height:\s*min\(100%,\s*var\(--chat-message-navigator-panel-height\)\);[\s\S]*transform:\s*translateY\(-50%\);/,
    );
    expect(css).toMatch(
      /\.chat-message-navigator-rail\s*\{[\s\S]*align-items:\s*center;[\s\S]*box-shadow:\s*none;/,
    );
    expect(css).toMatch(
      /\.chat-message-navigator:hover \.chat-message-navigator-rail,[\s\S]*\.chat-message-navigator\.is-open \.chat-message-navigator-rail\s*\{[\s\S]*box-shadow:[\s\S]*var\(--shadow-sm\);/,
    );
    expect(css).toMatch(
      /\.chat-message-navigator-popover\s*\{[\s\S]*position:\s*absolute;[\s\S]*right:\s*calc\([\s\S]*var\(--chat-message-navigator-offset\)[\s\S]*var\(--chat-message-navigator-rail-width\)[\s\S]*var\(--chat-message-navigator-gap\)[\s\S]*\);/,
    );
    expect(css).toMatch(/--chat-message-navigator-panel-width:\s*min\(20rem,/);
    expect(css).toMatch(/--chat-message-navigator-panel-height:\s*18rem;/);
    expect(css).toMatch(
      /\.chat-message-navigator-popover\s*\{[\s\S]*top:\s*50%;[\s\S]*height:\s*min\(var\(--chat-message-navigator-panel-height\),\s*100%\);[\s\S]*max-height:\s*100%;[\s\S]*transform:\s*translate\(var\(--space-xs\),\s*-50%\);/,
    );
    expect(css).toMatch(
      /\.chat-message-navigator-ticks,\s*\.chat-message-navigator-list\s*\{[\s\S]*scroll-behavior:\s*auto;/,
    );
    expect(css).toMatch(
      /\.chat-message-navigator-ticks\s*\{[\s\S]*flex:\s*0\s+1\s+auto;[\s\S]*max-height:\s*100%;[\s\S]*overflow-y:\s*auto;[\s\S]*padding:\s*calc\(var\(--space-xs\)\s*\/\s*2\)\s*0;/,
    );
    expect(css).toMatch(/\.chat-message-navigator-list\s*\{[\s\S]*overflow-y:\s*auto;/);
    expect(navigatorItemRule).toMatch(/flex:\s*0\s+0\s+auto;/);
    expect(navigatorItemRule).toMatch(/display:\s*block;/);
    expect(navigatorItemRule).toMatch(/font-size:\s*var\(--text-sm\);/);
    expect(navigatorItemRule).toMatch(/white-space:\s*normal;/);
    expect(navigatorItemRule).not.toMatch(/-webkit-box-orient:/);
    expect(navigatorItemRule).not.toMatch(/-webkit-line-clamp:/);
    expect(navigatorItemRule).not.toMatch(/text-overflow:/);
    expect(navigatorItemTextRule).toMatch(/display:\s*-webkit-box;/);
    expect(navigatorItemTextRule).toMatch(/line-clamp:\s*2;/);
    expect(navigatorItemTextRule).toMatch(/-webkit-box-orient:\s*vertical;/);
    expect(navigatorItemTextRule).toMatch(/-webkit-line-clamp:\s*2;/);
    expect(navigatorItemTextRule).toMatch(/overflow:\s*hidden;/);
    expect(css).toContain('scrollbar-width: none;');
    expect(css).toMatch(
      /\.chat-message-navigator-tick\.is-active\s*\{[\s\S]*background:\s*var\(--color-primary\)/,
    );
    expect(navigatorTickRule).toMatch(/border:\s*0;/);
    expect(navigatorTickRule).toMatch(/padding:\s*0;/);
    expect(navigatorTickRule).toMatch(/cursor:\s*pointer;/);
    expect(css).toMatch(
      /\.chat-message-navigator-item\.is-active,\s*\.chat-message-navigator-item\.is-previewed\s*\{[\s\S]*(?:border-color:\s*var\(--color-primary-border\)|box-shadow:\s*inset[^;]*var\(--color-primary\))/,
    );
    expect(css).toContain('@container chat-embedded (max-width: 720px)');
    expect(css).toContain('@media (hover: none), (pointer: coarse)');
    expect(css).toMatch(
      /@media[\s\S]*\.chat-message-navigator\s*\{[\s\S]*display:\s*none\s*!important;/,
    );
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\.chat-message-navigator[\s\S]*transition:\s*none\s*!important;/,
    );
    expect(css).toMatch(
      /@media\s*\(prefers-reduced-motion:\s*reduce\)[\s\S]*\.chat-message-navigator-popover\s*\{[\s\S]*transform:\s*translateY\(-50%\);/,
    );
  });
});
