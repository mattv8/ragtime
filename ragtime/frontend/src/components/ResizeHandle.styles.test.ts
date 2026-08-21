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

describe('Resize handle style contracts', () => {
  it('restores the shared legacy resize strips and mobile suppression rules', () => {
    const componentsCss = readSource('src/styles/components.css');
    const chatCss = readSource('src/styles/chat.css');
    const responsiveCss = readSource('src/styles/responsive.css');
    const max768Css = getRuleBody(responsiveCss, '@media (max-width: 768px)');

    expect(componentsCss).toMatch(
      /\.resize-handle\s*\{[\s\S]*background:\s*var\(--color-border\);[\s\S]*transition:\s*background var\(--transition-fast\);/,
    );
    expect(componentsCss).toMatch(
      /\.resize-handle:hover,\s*\.resize-handle:active\s*\{[\s\S]*background:\s*var\(--color-primary\);/,
    );
    expect(componentsCss).toMatch(
      /\.resize-handle-horizontal::before\s*\{[\s\S]*left:\s*-5px;[\s\S]*right:\s*-5px;/,
    );
    expect(componentsCss).toMatch(
      /\.resize-handle-vertical::before\s*\{[\s\S]*top:\s*-5px;[\s\S]*bottom:\s*-5px;/,
    );
    expect(componentsCss).toMatch(
      /\.resize-handle-horizontal\s*\{[\s\S]*width:\s*4px;[\s\S]*cursor:\s*col-resize;/,
    );
    expect(componentsCss).toMatch(
      /\.resize-handle-vertical\s*\{[\s\S]*height:\s*4px;[\s\S]*cursor:\s*row-resize;/,
    );
    expect(componentsCss).toMatch(
      /\.resize-handle-collapsed\s*\{[\s\S]*display:\s*flex;[\s\S]*background:\s*var\(--color-border\);/,
    );

    expect(chatCss).toMatch(/\.chat-resize-handle\s*\{[\s\S]*align-self:\s*stretch;/);
    expect(chatCss).toMatch(
      /\.chat-in-chat-search-resize-handle\s*\{[\s\S]*position:\s*absolute;[\s\S]*border-width:\s*0 0 10px 10px;/,
    );
    expect(chatCss).toMatch(
      /\.chat-in-chat-search-resize-handle-horizontal\s*\{[\s\S]*top:\s*6px;[\s\S]*right:\s*0;[\s\S]*bottom:\s*12px;/,
    );
    expect(chatCss).toMatch(
      /\.chat-in-chat-search-resize-handle-vertical\s*\{[\s\S]*right:\s*6px;[\s\S]*bottom:\s*0;[\s\S]*left:\s*calc\(100% - 44px\);/,
    );

    expect(max768Css).toMatch(
      /\.userspace-content > \.resize-handle-horizontal,\s*\.userspace-left-pane > \.resize-handle-vertical,\s*\.userspace-editor-section > \.resize-handle-horizontal\s*\{[\s\S]*display:\s*none;/,
    );
  });

  it('gates every integrated workbench sash selector to the Modern theme pack', () => {
    const css = readSource('src/styles/workbench.css');
    const pointerCoarseCss = getRuleBody(css, '@media (pointer: coarse)');
    const max768Css = getRuleBody(css, '@media (max-width: 768px)');
    const forcedColorsCss = getRuleBody(css, '@media (forced-colors: active)');

    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle::before\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle::after\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-horizontal\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-horizontal::before\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-horizontal::after\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-vertical\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-vertical::before\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-vertical::after\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle:hover::after,/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle:active::after,/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle:focus-visible::after,/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-collapsed::after\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle:focus-visible\s*\{/);
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-collapsed\s*\{/);
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.resize-handle-collapsed\.resize-handle-horizontal\s*\{/,
    );
    expect(css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.resize-handle-collapsed\.resize-handle-vertical\s*\{/,
    );
    expect(css).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle-chevron\s*\{/);

    expect(pointerCoarseCss).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle::after\s*\{/);
    expect(pointerCoarseCss).toMatch(
      /\[data-theme-pack='modern'\]\s+\.resize-handle-collapsed::after\s*\{/,
    );
    expect(max768Css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-content > \.resize-handle-horizontal,/,
    );
    expect(max768Css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-left-pane > \.resize-handle-vertical,/,
    );
    expect(max768Css).toMatch(
      /\[data-theme-pack='modern'\]\s+\.userspace-editor-section > \.resize-handle-horizontal\s*\{/,
    );
    expect(forcedColorsCss).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle\s*\{/);
    expect(forcedColorsCss).toMatch(/\[data-theme-pack='modern'\]\s+\.resize-handle::after\s*\{/);
    expect(forcedColorsCss).toMatch(
      /\[data-theme-pack='modern'\]\s+\.resize-handle:focus-visible\s*\{/,
    );

    expect(css).not.toContain('\n.resize-handle {');
    expect(css).not.toContain('\n.resize-handle::before {');
    expect(css).not.toContain('\n.resize-handle::after {');
    expect(css).not.toContain('\n.resize-handle-horizontal {');
    expect(css).not.toContain('\n.resize-handle-vertical {');
    expect(css).not.toContain('\n.resize-handle-collapsed {');
    expect(css).not.toContain('\n.resize-handle-chevron {');
  });
});
