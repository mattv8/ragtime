import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';

import { RowDeleteButton } from './RowDeleteButton';

afterEach(cleanup);

function renderInRail(props: Partial<Parameters<typeof RowDeleteButton>[0]> = {}) {
  const onDelete = props.onDelete ?? vi.fn(() => true);
  render(
    <div className="access-md-list">
      <ol>
        <li>
          <button type="button" className="access-md-rail-row">
            Finance
          </button>
          <RowDeleteButton label="Delete Finance" disabled={false} {...props} onDelete={onDelete} />
        </li>
        <li>
          <button type="button" className="access-md-rail-row">
            Standard
          </button>
        </li>
      </ol>
    </div>,
  );
  return onDelete;
}

it('deletes immediately without confirm and focuses the neighboring row', async () => {
  const onDelete = renderInRail();
  await userEvent.click(screen.getByRole('button', { name: 'Delete Finance' }));

  expect(onDelete).toHaveBeenCalledOnce();
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Standard' })),
  );
});

it('exposes a blocked reason and skips the confirm step', async () => {
  const onDelete = renderInRail({
    confirm: true,
    blockedReason: 'Cannot delete the default.',
    onDelete: vi.fn(() => false),
  });
  const trash = screen.getByRole('button', { name: 'Delete Finance' });

  expect(trash.getAttribute('aria-disabled')).toBe('true');
  expect(trash.getAttribute('aria-describedby')).toBeTruthy();
  expect(document.getElementById(trash.getAttribute('aria-describedby')!)?.textContent).toBe(
    'Cannot delete the default.',
  );
  await userEvent.click(trash);

  expect(onDelete).toHaveBeenCalledOnce();
  expect(screen.queryByRole('group', { name: 'Confirm Delete Finance' })).toBeNull();
});

it('focuses Cancel when confirming and returns focus to the trash on Escape', async () => {
  const onDelete = renderInRail({ confirm: true });
  await userEvent.click(screen.getByRole('button', { name: 'Delete Finance' }));

  expect(screen.getByRole('group', { name: 'Confirm Delete Finance' })).toBeTruthy();
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' })),
  );
  await userEvent.keyboard('{Escape}');

  expect(onDelete).not.toHaveBeenCalled();
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Delete Finance' })),
  );
});

it('returns focus to the trash when a confirmed delete fails', async () => {
  const onDelete = renderInRail({ confirm: true, onDelete: vi.fn(async () => false) });
  await userEvent.click(screen.getByRole('button', { name: 'Delete Finance' }));
  await userEvent.click(screen.getByRole('button', { name: 'Confirm delete' }));

  expect(onDelete).toHaveBeenCalledOnce();
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Delete Finance' })),
  );
});
