import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { ToolGroupDeleteDialog } from './ToolGroupDeleteDialog';

afterEach(cleanup);

it('ignores Escape while pending and restores opener on unmount', async () => {
  const user = userEvent.setup();
  const opener = document.createElement('button');
  document.body.appendChild(opener);
  opener.focus();
  const close = vi.fn();
  const { unmount } = render(
    <ToolGroupDeleteDialog
      groupName="A"
      toolCount={0}
      onConfirm={() => new Promise<void>(() => {})}
      onClose={close}
    />,
  );
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' })),
  );
  await user.click(screen.getByRole('button', { name: 'Delete group' }));
  await user.keyboard('{Escape}');
  expect(close).not.toHaveBeenCalled();
  unmount();
  expect(document.activeElement).toBe(opener);
  opener.remove();
});

it('describes the consequences and reports failures', async () => {
  const onConfirm = vi.fn().mockRejectedValue(new Error('Delete failed'));
  render(
    <ToolGroupDeleteDialog
      groupName="Alpha"
      toolCount={1}
      onConfirm={onConfirm}
      onClose={vi.fn()}
    />,
  );
  expect(
    screen.getByText('The 1 tool in this group move to Ungrouped. No tools are deleted.'),
  ).toBeTruthy();
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' })),
  );
  fireEvent.click(screen.getByRole('button', { name: 'Delete group' }));
  expect((await screen.findByRole('alert')).textContent).toBe('Delete failed');
});

it.each([
  [0, 'This group has no tools.'],
  [1, 'The 1 tool in this group move to Ungrouped. No tools are deleted.'],
  [4, 'The 4 tools in this group move to Ungrouped. No tools are deleted.'],
])('shows consequences for %i tools', (count, copy) => {
  render(
    <ToolGroupDeleteDialog
      groupName="Alpha"
      toolCount={count}
      onConfirm={vi.fn()}
      onClose={vi.fn()}
    />,
  );
  expect(screen.getByRole('heading', { name: 'Delete group “Alpha”?' })).toBeTruthy();
  expect(screen.getByText(copy)).toBeTruthy();
  expect(
    screen.getByText('Chats and workspaces that selected this group lose that selection.'),
  ).toBeTruthy();
});

it('traps focus and invokes Cancel and Escape', async () => {
  const user = userEvent.setup();
  const close = vi.fn();
  render(
    <ToolGroupDeleteDialog groupName="Alpha" toolCount={0} onConfirm={vi.fn()} onClose={close} />,
  );
  const cancel = screen.getByRole('button', { name: 'Cancel' });
  await waitFor(() => expect(document.activeElement).toBe(cancel));
  await user.tab({ shift: true });
  expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Delete group' }));
  await user.tab();
  expect(document.activeElement).toBe(cancel);
  await user.keyboard('{Escape}');
  await user.click(cancel);
  expect(close).toHaveBeenCalledTimes(2);
});

it('calls confirm and shows busy state until rejection', async () => {
  const user = userEvent.setup();
  let reject!: (error: Error) => void;
  const onConfirm = vi.fn(
    () =>
      new Promise<void>((_, rejectPromise) => {
        reject = rejectPromise;
      }),
  );
  render(
    <ToolGroupDeleteDialog
      groupName="Alpha"
      toolCount={2}
      onConfirm={onConfirm}
      onClose={vi.fn()}
    />,
  );
  await user.click(screen.getByRole('button', { name: 'Delete group' }));
  expect(onConfirm).toHaveBeenCalledOnce();
  expect(screen.getByRole('dialog').getAttribute('aria-busy')).toBe('true');
  expect((screen.getByRole('button', { name: 'Deleting…' }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  reject(new Error('Delete failed'));
  expect((await screen.findByRole('alert')).textContent).toBe('Delete failed');
});
