/**
 * CSS Contract Test: External API Access Section
 *
 * Validates that the theme-native inset-section, two-column form, and portal-dialog
 * styling for External API Access is correctly implemented using theme tokens.
 */

import { describe, it, expect, beforeEach } from 'vitest';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { resolve } from 'node:path';

import { getRuleBody } from '@/testHelpers/cssRuleUtils';

describe('ExternalApiAccessSection Styles', () => {
  let componentsCss: string;
  let workbenchAdminCss: string;
  const modernModalScope =
    "[data-theme-pack='modern'] .userspace-share-modal-with-tabs .userspace-external-api-access";

  beforeEach(() => {
    // Load CSS files for static analysis
    const componentsPath = resolve(cwd(), 'src/styles/components.css');
    const workbenchPath = resolve(cwd(), 'src/styles/workbench-admin.css');
    componentsCss = readFileSync(componentsPath, 'utf-8');
    workbenchAdminCss = readFileSync(workbenchPath, 'utf-8');
  });

  const countRules = (css: string, selector: string): number =>
    (css.match(new RegExp(`(^|\\n)${selector.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&')}\\s*\\{`, 'g')) ?? [])
      .length;

  const expectDeclarations = (body: string, declarations: string[]): void => {
    declarations.forEach((declaration) => {
      expect(body).toContain(declaration);
    });
  };

  describe('base inset section contract', () => {
    it('keeps a single inset-section rule with the shared tokenized shell', () => {
      const sectionRule = getRuleBody(componentsCss, '.userspace-external-api-section');

      expect(countRules(componentsCss, '.userspace-external-api-section')).toBe(1);
      expectDeclarations(sectionRule, [
        'display: flex',
        'flex-direction: column',
        'gap: var(--space-sm)',
        'padding: var(--space-md)',
        'border: 1px solid var(--color-border)',
        'border-radius: var(--radius-md)',
        'background: var(--color-bg-secondary)',
      ]);
      expect(sectionRule).not.toContain('border-top:');
    });

    it('keeps checklist rows flat inside the inset section', () => {
      const checkboxRowRule = getRuleBody(componentsCss, '.userspace-external-api-checkbox-row');

      expectDeclarations(checkboxRowRule, [
        'display: flex',
        'gap: var(--space-sm)',
        'align-items: flex-start',
      ]);
      expect(checkboxRowRule).not.toContain('padding:');
      expect(checkboxRowRule).not.toContain('border:');
      expect(checkboxRowRule).not.toContain('border-radius:');
      expect(checkboxRowRule).not.toContain('background:');
    });

    it('removes the obsolete external-api select styling hook', () => {
      expect(countRules(componentsCss, '.userspace-external-api-select')).toBe(0);
      expect(componentsCss).not.toContain('.userspace-external-api-select:focus');
      expect(componentsCss).not.toContain('.userspace-external-api-select option');
    });

    it('removes legacy inline reveal hooks that no longer have TSX callers', () => {
      expect(componentsCss).not.toMatch(
        /\.userspace-external-api-(?:reveal|reveal-header|reveal-copy|dismiss)(?=[\s:{.,#])/,
      );
    });
  });

  describe('dialog layering contract', () => {
    it('keeps the external-api portal above the base modal layer and on the widget surface', () => {
      const backdropRule = getRuleBody(componentsCss, '.userspace-external-api-dialog-backdrop');
      const panelRule = getRuleBody(componentsCss, '.userspace-external-api-dialog-panel');
      const modalOverlayRule = getRuleBody(componentsCss, '.modal-overlay');

      expectDeclarations(backdropRule, ['z-index: calc(var(--z-modal) + 10)']);
      expectDeclarations(panelRule, [
        'z-index: calc(var(--z-modal) + 11)',
        'background: var(--color-widget)',
      ]);
      expectDeclarations(modalOverlayRule, ['z-index: var(--z-modal)']);
    });

    it('keeps the narrow modern selected-tab override after the broad selected-state rule', () => {
      const broadSelectedStateIndex = workbenchAdminCss.indexOf("[data-theme-pack='modern'] :is(");
      const narrowExternalTabIndex = workbenchAdminCss.indexOf(
        "[data-theme-pack='modern'] .userspace-external-api-dialog-tab[aria-selected='true']",
      );
      const narrowRule = getRuleBody(
        workbenchAdminCss,
        "[data-theme-pack='modern'] .userspace-external-api-dialog-tab[aria-selected='true']",
      );

      expect(broadSelectedStateIndex).toBeGreaterThanOrEqual(0);
      expect(narrowExternalTabIndex).toBeGreaterThan(broadSelectedStateIndex);
      expectDeclarations(narrowRule, [
        'background: transparent',
        'color: var(--color-text-primary)',
        'border-bottom-color: var(--color-accent)',
      ]);
    });
  });

  describe('modern flat-intro dark-pane lane', () => {
    it('keeps the modern outer external-api access surface flat on the share modal', () => {
      const outerShellRule = getRuleBody(
        workbenchAdminCss,
        "[data-theme-pack='modern'] .userspace-external-api-access",
      );

      expectDeclarations(outerShellRule, [
        'background: transparent',
        'border: none',
        'border-radius: 0',
        'padding: 0',
        'gap: var(--workbench-padding)',
      ]);
    });

    it('styles only direct functional children as dark modern panes', () => {
      const directChildRule = getRuleBody(
        workbenchAdminCss,
        "[data-theme-pack='modern'] .userspace-external-api-access > .userspace-external-api-section",
      );

      expectDeclarations(directChildRule, [
        'background: var(--color-panel)',
        'border: var(--workbench-container-border)',
        'border-radius: var(--workbench-surface-radius)',
        'padding: var(--workbench-padding)',
      ]);
      expect(directChildRule).not.toContain('--color-bg-tertiary');
    });
  });

  describe('modern modal control geometry', () => {
    it('keeps equal credential columns and scoped shrinkable field controls', () => {
      const credentialFieldsRule = getRuleBody(componentsCss, '.userspace-external-api-credential-fields');
      const fieldShrinkRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} .userspace-external-api-credential-fields > .userspace-external-api-field`,
      );
      const inputRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} .userspace-external-api-credential-fields input`,
      );

      expectDeclarations(credentialFieldsRule, [
        'display: grid',
        'grid-template-columns: minmax(0, 1fr) minmax(0, 1fr)',
      ]);
      expectDeclarations(fieldShrinkRule, ['min-width: 0']);
      expectDeclarations(inputRule, [
        'box-sizing: border-box',
        'width: 100%',
        'min-width: 0',
        'height: var(--workbench-control-height)',
        'min-height: var(--workbench-control-height)',
        'padding: 0 var(--space-sm)',
        'border-radius: var(--workbench-control-radius)',
      ]);
    });

    it('keeps desktop action buttons compact and naturally sized while preserving status pills', () => {
      const actionButtonRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} :is(.userspace-external-api-create-actions, .userspace-external-api-row-actions) > .btn`,
      );
      const rowRule = getRuleBody(workbenchAdminCss, `${modernModalScope} .userspace-external-api-row`);
      const rowHeaderRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} .userspace-external-api-row-header`,
      );
      const rowActionsRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} .userspace-external-api-row-actions`,
      );
      const statusRule = getRuleBody(componentsCss, '.userspace-external-api-status');

      expectDeclarations(actionButtonRule, [
        'width: auto',
        'height: var(--workbench-control-height)',
        'min-height: var(--workbench-control-height)',
        'padding: 0 var(--space-sm)',
        'border-radius: var(--workbench-control-radius)',
      ]);
      expectDeclarations(rowRule, ['align-items: center']);
      expectDeclarations(rowHeaderRule, ['align-items: center']);
      expectDeclarations(rowActionsRule, ['align-self: center']);
      expect(statusRule).toContain('padding: 2px 8px');
      expect(statusRule).not.toContain('height: var(--workbench-control-height)');
      expect(statusRule).not.toContain('min-height: var(--workbench-control-height)');
    });

    it('restores mobile action button stretching within the existing 640px stacking breakpoint', () => {
      const mobileActionStretchRule = new RegExp(
        String.raw`@media \(max-width: 640px\)\s*\{[\s\S]*${modernModalScope.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')} :is\(\.userspace-external-api-create-actions, \.userspace-external-api-row-actions\) > \.btn\s*\{[\s\S]*width: 100%;`,
      );

      expect(workbenchAdminCss).toMatch(mobileActionStretchRule);
    });

    it('styles credential rows as a responsive two-column grid with bottom-pinned actions', () => {
      const credentialListRule = getRuleBody(
        componentsCss,
        '.userspace-external-api-credential-list',
      );
      const credentialRowRule = getRuleBody(
        componentsCss,
        '.userspace-external-api-credential-row',
      );
      const credentialActionsRule = getRuleBody(
        componentsCss,
        '.userspace-external-api-credential-actions',
      );
      const rowMainRule = getRuleBody(componentsCss, '.userspace-external-api-row-main');

      expectDeclarations(credentialListRule, [
        'display: grid',
        'grid-template-columns: repeat(2, minmax(0, 1fr))',
        'gap: var(--space-sm)',
      ]);
      expectDeclarations(credentialRowRule, [
        'display: flex',
        'flex-direction: column',
        'min-width: 0',
        'border: 1px solid var(--color-border)',
        'border-radius: var(--radius-md)',
        'padding: var(--space-sm)',
        'background: transparent',
      ]);
      expect(componentsCss).not.toContain('.userspace-external-api-credential-row + .userspace-external-api-credential-row');
      expectDeclarations(rowMainRule, ['flex: 1']);
      expectDeclarations(credentialActionsRule, [
        'display: flex',
        'justify-content: flex-end',
        'align-items: center',
        'gap: var(--space-xs)',
        'flex-wrap: nowrap',
        'margin-top: auto',
      ]);
      expect(credentialActionsRule).not.toContain('width: 100%');
      expect(getRuleBody(componentsCss, '.userspace-external-api-status')).not.toContain(
        'height: var(--workbench-control-height)',
      );
      expect(componentsCss).toMatch(
        /@media \(max-width: 640px\)\s*\{[\s\S]*\.userspace-external-api-credential-list\s*\{[\s\S]*grid-template-columns:\s*1fr;/,
      );
    });

    it('uses modern credential row border and control tokens without forcing credential status pills to button height', () => {
      const modernCredentialRowRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} .userspace-external-api-credential-row`,
      );
      const modernCredentialActionButtonRule = getRuleBody(
        workbenchAdminCss,
        `${modernModalScope} .userspace-external-api-credential-actions > .btn`,
      );

      expectDeclarations(modernCredentialRowRule, [
        'border: var(--workbench-container-border)',
        'border-radius: var(--workbench-control-radius)',
      ]);
      expectDeclarations(modernCredentialActionButtonRule, [
        'width: auto',
        'height: var(--workbench-control-height)',
        'min-height: var(--workbench-control-height)',
      ]);
      expect(workbenchAdminCss).not.toContain(
        `${modernModalScope} .userspace-external-api-credential-actions .userspace-external-api-status`,
      );
    });

    it('reveals endpoint selection controls only on hover-capable desktop, but keeps them visible for selected, touch, and mobile layouts', () => {
      const endpointControlRule = getRuleBody(
        componentsCss,
        '.userspace-external-api-endpoint-selection-control',
      );

      expectDeclarations(endpointControlRule, [
        'display: inline-flex',
        'align-items: center',
        'justify-content: center',
        'opacity: 0',
        'visibility: hidden',
        'transition:',
      ]);
      expect(endpointControlRule).not.toContain('display: none');
      expect(componentsCss).toMatch(
        /\.userspace-external-api-row:hover\s+\.userspace-external-api-endpoint-selection-control,[\s\S]*\.userspace-external-api-row:focus-within\s+\.userspace-external-api-endpoint-selection-control,[\s\S]*\.userspace-external-api-row\.is-credential-selected\s+\.userspace-external-api-endpoint-selection-control\s*\{[\s\S]*opacity:\s*1;[\s\S]*visibility:\s*visible;/,
      );
      expect(componentsCss).toMatch(
        /@media \(hover: none\)\s*\{[\s\S]*\.userspace-external-api-endpoint-selection-control\s*\{[\s\S]*opacity:\s*1;[\s\S]*visibility:\s*visible;/,
      );
      expect(componentsCss).toMatch(
        /@media \(max-width: 640px\)\s*\{[\s\S]*\.userspace-external-api-endpoint-selection-control\s*\{[\s\S]*opacity:\s*1;[\s\S]*visibility:\s*visible;/,
      );
    });

    it('styles inline credential details as a divider-separated subregion without an extra card shell', () => {
      const credentialDetailsRule = getRuleBody(
        componentsCss,
        '#workspace-external-api-credential-details',
      );

      expectDeclarations(credentialDetailsRule, [
        'margin-top: var(--space-sm)',
        'padding-top: var(--space-sm)',
        'border-top: 1px solid var(--color-border)',
      ]);
      expect(credentialDetailsRule).not.toContain('border: 1px solid');
      expect(credentialDetailsRule).not.toContain('border-radius:');
      expect(credentialDetailsRule).not.toContain('background:');
    });
  });

  describe('existing share-workspace contracts', () => {
    it('removes duplicate top spacing from the tabbed modal body', () => {
      const tabbedModalBodyRule = getRuleBody(
        componentsCss,
        '.userspace-share-modal-with-tabs .modal-body',
      );

      expectDeclarations(tabbedModalBodyRule, ['padding-top: 0']);
    });

    it('keeps credential history list density and table overflow protections', () => {
      const listRule = getRuleBody(componentsCss, '.userspace-external-api-list');
      const tableWrapRule = getRuleBody(componentsCss, '.userspace-external-api-table-wrap');

      expectDeclarations(listRule, ['gap: 0']);
      expectDeclarations(tableWrapRule, ['overflow-x: auto']);
    });
  });
});
