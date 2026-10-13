import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import { ContentProtectionAccessLevelCard } from './ContentProtectionAccessLevelCard';

it('grants and clears only editable categories', () => {
  const onUpdate = vi.fn();
  render(
    <ContentProtectionAccessLevelCard
      level={{ id: 'standard', name: 'Standard', granted_category_ids: [], guidance: '' }}
      categories={[
        {
          id: 'internal',
          name: 'Internal',
          description: '',
          includes: [],
          excludes: [],
          examples: [],
          denial_message: '',
          threshold_override: null,
          system: false,
        },
        {
          id: 'confidential',
          name: 'Confidential',
          description: '',
          includes: [],
          excludes: [],
          examples: [],
          denial_message: '',
          threshold_override: null,
          system: false,
        },
        {
          id: 'rule_override',
          name: 'Rule override',
          description: '',
          includes: [],
          excludes: [],
          examples: [],
          denial_message: '',
          threshold_override: null,
          system: true,
        },
      ]}
      onUpdate={onUpdate}
    />,
  );

  fireEvent.click(screen.getByRole('button', { name: 'Grant all categories' }));
  fireEvent.click(screen.getByRole('button', { name: 'Revoke all categories' }));

  expect(onUpdate).toHaveBeenNthCalledWith(1, 'standard', {
    granted_category_ids: ['internal', 'confidential'],
  });
  expect(onUpdate).toHaveBeenNthCalledWith(2, 'standard', { granted_category_ids: [] });
  expect(screen.queryByRole('checkbox', { name: 'Rule override' })).toBeNull();
});
