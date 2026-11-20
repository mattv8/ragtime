// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { getRuleBody } from '@/testHelpers/cssRuleUtils';

const css = readFileSync(join(cwd(), 'src/styles/instruction-skills.css'), 'utf8');

describe('instruction skills style contracts', () => {
  it('uses readable scoped ink for empty content and the Skills menu description', () => {
    expect(getRuleBody(css, '.instruction-skills-empty > h4')).toContain(
      'color: var(--color-text-primary);',
    );
    for (const selector of [
      '.instruction-skills-empty > p',
      '[data-instruction-skills-menu-description]',
    ]) {
      expect(getRuleBody(css, selector)).toContain('color: var(--color-text-secondary);');
    }
    const toolsPanel = readFileSync(join(cwd(), 'src/components/ToolsPanel.tsx'), 'utf8');
    expect(toolsPanel).toContain('<span data-instruction-skills-menu-description>');
  });

  it('keeps mono fields above global input specificity and preserves coarse-pointer sizing', () => {
    const selector = '#instruction-skills-dialog .instruction-skills-mono';
    expect(getRuleBody(css, selector)).toContain('font-family: var(--font-mono);');
    expect(getRuleBody(css, selector)).toContain('font-size: var(--text-sm);');
    expect(getRuleBody(getRuleBody(css, '@media (pointer: coarse)'), selector)).toContain(
      'font-size: 16px;',
    );
  });

  it('keeps the Always badge tinted above the shared tool-badge rule', () => {
    const badge = getRuleBody(css, '.tool-badge.instruction-skill-badge-always');
    expect(badge).toContain('background: var(--color-primary-light);');
    expect(badge).toContain('border-color: var(--color-primary-border);');
    expect(badge).toContain('color: var(--color-text-primary);');
  });

  it('lets Description use its two rows and removes only empty validation spacing', () => {
    expect(getRuleBody(css, '#instruction-skill-description')).toContain('min-height: 0;');
    expect(css).toContain(
      '#instruction-skill-name-error:empty,\n#instruction-skill-description-error:empty {',
    );
    expect(getRuleBody(css, '#instruction-skill-description-error:empty')).toContain(
      'display: none;',
    );
  });

  it('outlines toggle thumbs and focus in forced colors while preserving distinct positions', () => {
    const forcedColors = getRuleBody(css, '@media (forced-colors: active)');
    const toggle = '#instruction-skills-dialog .instruction-skill-row-toggle';
    const thumb = getRuleBody(forcedColors, `${toggle} .toggle-slider::before`);
    expect(thumb).toContain('border: 2px solid ButtonText;');
    expect(thumb).toContain('background: Canvas;');
    expect(thumb).toContain('transform: translateX(0);');
    const checked = getRuleBody(forcedColors, `${toggle} input:checked + .toggle-slider::before`);
    expect(checked).toContain('transform: translateX(20px);');
    expect(checked).toContain('background: Highlight;');
    expect(getRuleBody(forcedColors, `${toggle} input:focus-visible + .toggle-slider`)).toContain(
      'outline: 2px solid Highlight;',
    );
    expect(forcedColors).not.toContain('forced-color-adjust: none');
  });

  it('uses themed secondary Move buttons without losing coarse-pointer target sizes', () => {
    const dialog = readFileSync(join(cwd(), 'src/components/InstructionSkillsDialog.tsx'), 'utf8');
    for (const direction of ['up', 'down']) {
      expect(dialog).toContain(
        `className="btn btn-secondary"\n                            data-instruction-skill-move="${direction}"`,
      );
    }
    const buttons = getRuleBody(
      getRuleBody(css, '@media (pointer: coarse)'),
      '.instruction-skill-move button',
    );
    expect(buttons).toContain('min-width: 44px;');
    expect(buttons).toContain('min-height: 44px;');
  });
});
