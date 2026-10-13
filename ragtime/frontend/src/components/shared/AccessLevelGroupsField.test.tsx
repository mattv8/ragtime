import { cleanup, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';

import { AccessLevelGroupsField } from './AccessLevelGroupsField';

afterEach(cleanup);

it('filters searchable groups and updates only the changed mapping while preserving unknown chips', async () => {
  const user = userEvent.setup();
  const onAdd = vi.fn();
  const onRemove = vi.fn();

  render(
    <AccessLevelGroupsField
      idPrefix="level-finance-"
      groups={[
        {
          id: 'finance',
          label: 'Finance',
          badge: <span className="provider-badge">Internal</span>,
        },
        { id: 'engineering', label: 'Engineering' },
      ]}
      mappedIds={['finance', 'retired-group']}
      onAdd={onAdd}
      onRemove={onRemove}
    />,
  );

  expect(screen.getByText('Internal').className).toBe('provider-badge');
  expect(screen.getByText('Unknown (retired-)')).toBeTruthy();
  await user.click(screen.getByRole('button', { name: 'Groups for this level' }));
  await user.type(screen.getByRole('textbox', { name: 'Filter groups' }), 'Engineering');

  await user.click(screen.getByRole('checkbox', { name: 'Engineering' }));
  await user.click(screen.getByRole('button', { name: 'Remove Finance from this level' }));
  await user.click(screen.getByRole('button', { name: 'Remove unknown group retired-group' }));

  expect(onAdd).toHaveBeenCalledOnce();
  expect(onAdd).toHaveBeenCalledWith('engineering');
  expect(onRemove).toHaveBeenNthCalledWith(1, 'finance');
  expect(onRemove).toHaveBeenNthCalledWith(2, 'retired-group');
});

it('shows its disabled hint instead of the add control while keeping mapped groups removable only when enabled', () => {
  render(
    <AccessLevelGroupsField
      idPrefix="level-finance-"
      groups={[{ id: 'finance', label: 'Finance' }]}
      mappedIds={['finance']}
      onAdd={vi.fn()}
      onRemove={vi.fn()}
      disabled
      disabledHint="Save this level before mapping groups."
    />,
  );

  expect(screen.queryByLabelText('Groups for this level')).toBeNull();
  expect(screen.getByText('Save this level before mapping groups.')).toBeTruthy();
  expect(
    (screen.getByRole('button', { name: 'Remove Finance from this level' }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
});
