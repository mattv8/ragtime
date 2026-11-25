import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { MoveToGroupDialog } from './MoveToGroupDialog';

afterEach(cleanup);
const groups = [
  { id: 'g1', name: 'Alpha', description: 'First', toolCount: 1 },
  { id: 'g2', name: 'Beta', description: 'Second', toolCount: 2 },
];

it('treats a stale current group as Ungrouped and focuses it', async () => {
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="missing"
      groups={groups}
      onMove={vi.fn()}
      onClose={vi.fn()}
    />,
  );
  const radio = screen.getByRole('radio', { name: /No group/ }) as HTMLInputElement;
  expect(radio.checked).toBe(true);
  expect(within(radio.closest('label')!).getByText('Current')).toBeTruthy();
  expect((screen.getByRole('button', { name: 'Move' }) as HTMLButtonElement).disabled).toBe(true);
  await waitFor(() => expect(document.activeElement).toBe(radio));
});

it('submits new name on Enter and focuses invalid input', async () => {
  const user = userEvent.setup();
  const onMove = vi.fn().mockResolvedValue(undefined);
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={onMove}
      onClose={vi.fn()}
    />,
  );
  await user.click(screen.getByRole('radio', { name: 'New group…' }));
  const name = screen.getByLabelText('Group name');
  await user.type(name, 'alpha');
  await user.click(screen.getByRole('button', { name: 'Move' }));
  expect(document.activeElement).toBe(name);
  await user.clear(name);
  await user.type(name, 'New{Enter}');
  expect(onMove).toHaveBeenCalledWith({ kind: 'new', name: 'New' });
});

it('ignores Escape while pending and restores opener on unmount', async () => {
  const user = userEvent.setup();
  const opener = document.createElement('button');
  document.body.appendChild(opener);
  opener.focus();
  const close = vi.fn();
  const { unmount } = render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={() => new Promise<void>(() => {})}
      onClose={close}
    />,
  );
  await waitFor(() =>
    expect(document.activeElement).toBe(screen.getByRole('radio', { name: /Alpha/ })),
  );
  await user.click(screen.getByRole('radio', { name: /Beta/ }));
  await user.click(screen.getByRole('button', { name: 'Move' }));
  await user.keyboard('{Escape}');
  expect(close).not.toHaveBeenCalled();
  unmount();
  expect(document.activeElement).toBe(opener);
  opener.remove();
});

it('preselects current group and submits a changed target', async () => {
  const onMove = vi.fn().mockResolvedValue(undefined);
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="a"
      groups={[{ id: 'a', name: 'Alpha', description: '', toolCount: 1 }]}
      onMove={onMove}
      onClose={vi.fn()}
    />,
  );
  expect((screen.getByRole('radio', { name: /Alpha/ }) as HTMLInputElement).checked).toBe(true);
  expect((screen.getByRole('button', { name: 'Move' }) as HTMLButtonElement).disabled).toBe(true);
  fireEvent.click(screen.getByRole('radio', { name: /No group/ }));
  fireEvent.click(screen.getByRole('button', { name: 'Move' }));
  await waitFor(() => expect(onMove).toHaveBeenCalledWith({ kind: 'group', groupId: null }));
});

it('orders radios and preselects the current group with focus and pill', async () => {
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={vi.fn()}
      onClose={vi.fn()}
    />,
  );
  expect(screen.getByRole('dialog').getAttribute('aria-labelledby')).toBe(
    'tools-move-to-group-title',
  );
  expect(screen.getByRole('heading', { name: 'Move “Tool” to group' })).toBeTruthy();
  const radios = screen.getAllByRole('radio') as HTMLInputElement[];
  expect(radios.map((radio) => radio.value)).toEqual(['g1', 'g2', '', 'new']);
  expect(radios[0].checked).toBe(true);
  expect(within(radios[0].closest('label')!).getByText('Current')).toBeTruthy();
  await waitFor(() => expect(document.activeElement).toBe(radios[0]));
});

it('preselects Ungrouped and marks it current', async () => {
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId={null}
      groups={groups}
      onMove={vi.fn()}
      onClose={vi.fn()}
    />,
  );
  const radio = screen.getByRole('radio', { name: /No group/ }) as HTMLInputElement;
  expect(radio.checked).toBe(true);
  expect(within(radio.closest('label')!).getByText('Current')).toBeTruthy();
  await waitFor(() => expect(document.activeElement).toBe(radio));
});

it.each([
  ['Beta', { kind: 'group', groupId: 'g2' }],
  ['No group', { kind: 'group', groupId: null }],
])('moves to %s', async (label, target) => {
  const user = userEvent.setup();
  const onMove = vi.fn().mockResolvedValue(undefined);
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={onMove}
      onClose={vi.fn()}
    />,
  );
  expect((screen.getByRole('button', { name: 'Move' }) as HTMLButtonElement).disabled).toBe(true);
  await user.click(screen.getByRole('radio', { name: new RegExp(label) }));
  await user.click(screen.getByRole('button', { name: 'Move' }));
  expect(onMove).toHaveBeenCalledWith(target);
});

it('validates and trims a new group name', async () => {
  const user = userEvent.setup();
  const onMove = vi.fn().mockResolvedValue(undefined);
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={onMove}
      onClose={vi.fn()}
    />,
  );
  await user.click(screen.getByRole('radio', { name: 'New group…' }));
  expect((screen.getByRole('button', { name: 'Move' }) as HTMLButtonElement).disabled).toBe(true);
  const name = screen.getByLabelText('Group name');
  await user.type(name, ' alpha ');
  await user.click(screen.getByRole('button', { name: 'Move' }));
  expect(screen.getByRole('alert').textContent).toBe('A group named “alpha” already exists.');
  expect(onMove).not.toHaveBeenCalled();
  await user.clear(name);
  await user.type(name, ' Trimmed ');
  await user.click(screen.getByRole('button', { name: 'Move' }));
  expect(onMove).toHaveBeenCalledWith({ kind: 'new', name: 'Trimmed' });
});

it('shows pending and error state without losing selection', async () => {
  const user = userEvent.setup();
  let reject!: (error: Error) => void;
  const onMove = vi.fn(
    () =>
      new Promise<void>((_, rejectPromise) => {
        reject = rejectPromise;
      }),
  );
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={onMove}
      onClose={vi.fn()}
    />,
  );
  const target = screen.getByRole('radio', { name: /Beta/ }) as HTMLInputElement;
  await user.click(target);
  await user.click(screen.getByRole('button', { name: 'Move' }));
  expect(screen.getByRole('dialog').getAttribute('aria-busy')).toBe('true');
  expect((screen.getByRole('button', { name: 'Moving…' }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  reject(new Error('Move failed'));
  expect((await screen.findByRole('alert')).textContent).toBe('Move failed');
  expect(target.checked).toBe(true);
});

it('traps focus and closes on Escape and Cancel', async () => {
  const user = userEvent.setup();
  const onClose = vi.fn();
  render(
    <MoveToGroupDialog
      toolName="Tool"
      currentGroupId="g1"
      groups={groups}
      onMove={vi.fn()}
      onClose={onClose}
    />,
  );
  const first = screen.getByRole('radio', { name: /Alpha/ });
  await waitFor(() => expect(document.activeElement).toBe(first));
  await user.tab({ shift: true });
  expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Cancel' }));
  await user.tab();
  expect(document.activeElement).toBe(first);
  await user.keyboard('{Escape}');
  await user.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(onClose).toHaveBeenCalledTimes(2);
});
