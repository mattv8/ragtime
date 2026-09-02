// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { readFileSync } from 'node:fs';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { cwd } from 'node:process';
// @ts-expect-error Vitest runs in Node, but the frontend tsconfig omits Node types.
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { getRuleBody } from '@/testHelpers/cssRuleUtils';

describe('External API access styles', () => {
  it('does not separate the introductory content from the preceding modal content', () => {
    const css = readFileSync(join(cwd(), 'src/styles/components.css'), 'utf8');
    const accessRule = getRuleBody(css, '.userspace-external-api-access');

    expect(accessRule).not.toMatch(/border-top\s*:/);
    expect(accessRule).not.toMatch(/padding-top\s*:/);
  });
});
