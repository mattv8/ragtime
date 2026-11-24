import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, expect, it, vi } from 'vitest';
import { ToolGroupDialog, validateToolGroupName } from './ToolGroupDialog';

afterEach(cleanup);

it('restores opener focus after unmount', async () => {
  const opener = document.createElement('button');
  document.body.appendChild(opener);
  opener.focus();
  const { unmount } = render(
    <ToolGroupDialog mode="create" existingNames={[]} onSubmit={vi.fn()} onClose={vi.fn()} />,
  );
  await waitFor(() => expect(document.activeElement).toBe(screen.getByLabelText('Name')));
  unmount();
  expect(document.activeElement).toBe(opener);
  opener.remove();
});

it('ignores Escape and duplicate submits while saving', async () => {
  const user = userEvent.setup();
  const close = vi.fn();
  const submit = vi.fn(() => new Promise<void>(() => {}));
  render(
    <ToolGroupDialog
      mode="create"
      initialName="Group"
      existingNames={[]}
      onSubmit={submit}
      onClose={close}
    />,
  );
  const name = screen.getByLabelText('Name');
  await waitFor(() => expect(document.activeElement).toBe(name));
  await user.keyboard('{Enter}{Enter}{Escape}');
  expect(submit).toHaveBeenCalledTimes(1);
  expect(close).not.toHaveBeenCalled();
});

it('does not submit twice before the busy render or close on success', async () => {
  const onClose = vi.fn();
  let resolve!: () => void;
  const onSubmit = vi.fn(
    () =>
      new Promise<void>((done) => {
        resolve = done;
      }),
  );
  render(
    <ToolGroupDialog
      mode="create"
      initialName="Group"
      existingNames={[]}
      onSubmit={onSubmit}
      onClose={onClose}
    />,
  );
  const form = screen.getByRole('button', { name: 'Create group' }).closest('form')!;
  fireEvent.submit(form);
  fireEvent.submit(form);
  expect(onSubmit).toHaveBeenCalledTimes(1);
  resolve();
  await waitFor(() => expect(onClose).not.toHaveBeenCalled());
});

it('focuses the invalid name after validation', async () => {
  const user = userEvent.setup();
  render(<ToolGroupDialog mode="create" existingNames={[]} onSubmit={vi.fn()} onClose={vi.fn()} />);
  await user.click(screen.getByRole('button', { name: 'Create group' }));
  expect(document.activeElement).toBe(screen.getByLabelText('Name'));
});

it('validates names with reserved and duplicate handling', () => {
  expect(validateToolGroupName(' ', [])).toBe('Enter a group name.');
  expect(validateToolGroupName('x'.repeat(121), [])).toBe('Use 120 characters or fewer.');
  expect(validateToolGroupName(' unGROUPED ', [])).toBe('“Ungrouped” is reserved.');
  expect(validateToolGroupName(' Foo ', ['foo'])).toBe('A group named “Foo” already exists.');
  expect(validateToolGroupName('FOO', ['foo'], 'foo')).toBe(null);
  expect(validateToolGroupName('x'.repeat(120), [])).toBe(null);
});

it('shows and refreshes associated validation errors', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  render(
    <ToolGroupDialog
      mode="create"
      existingNames={['Existing']}
      onSubmit={onSubmit}
      onClose={vi.fn()}
    />,
  );
  await user.click(screen.getByRole('button', { name: 'Create group' }));
  const name = screen.getByLabelText('Name');
  expect(name.getAttribute('aria-invalid')).toBe('true');
  expect(name.getAttribute('aria-describedby')).toBe('tools-group-name-error');
  expect(screen.getByRole('alert').textContent).toBe('Enter a group name.');
  await user.type(name, 'Existing');
  expect(screen.getByRole('alert').textContent).toBe('A group named “Existing” already exists.');
  expect(onSubmit).not.toHaveBeenCalled();
});

it('focuses and selects edit name, and submits with Enter', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn().mockResolvedValue(undefined);
  render(
    <ToolGroupDialog
      mode="edit"
      initialName="Original"
      existingNames={['original']}
      onSubmit={onSubmit}
      onClose={vi.fn()}
    />,
  );
  expect(screen.getByRole('dialog').getAttribute('aria-label')).toBeNull();
  expect(screen.getByRole('heading', { name: 'Edit group' })).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Save' })).toBeTruthy();
  const name = screen.getByLabelText('Name') as HTMLInputElement;
  await waitFor(() => expect(document.activeElement).toBe(name));
  expect(name.selectionStart).toBe(0);
  expect(name.selectionEnd).toBe(8);
  await user.keyboard('{Enter}');
  await waitFor(() => expect(onSubmit).toHaveBeenCalledWith({ name: 'Original', description: '' }));
});

it.each(['Control', 'Meta'])(
  'submits description with %s+Enter, but not plain Enter',
  async (modifier) => {
    const user = userEvent.setup();
    const onSubmit = vi.fn().mockResolvedValue(undefined);
    render(
      <ToolGroupDialog
        mode="create"
        initialName="Group"
        existingNames={[]}
        onSubmit={onSubmit}
        onClose={vi.fn()}
      />,
    );
    await user.click(screen.getByLabelText('Description'));
    await user.keyboard('{Enter}');
    expect(onSubmit).not.toHaveBeenCalled();
    await user.keyboard(`{${modifier}>}{Enter}{/${modifier}}`);
    await waitFor(() => expect(onSubmit).toHaveBeenCalledOnce());
  },
);

it('traps focus and closes on Escape or Cancel', async () => {
  const user = userEvent.setup();
  const close = vi.fn();
  render(<ToolGroupDialog mode="create" existingNames={[]} onSubmit={vi.fn()} onClose={close} />);
  const name = screen.getByLabelText('Name');
  await waitFor(() => expect(document.activeElement).toBe(name));
  await user.tab({ shift: true });
  expect(document.activeElement).toBe(screen.getByRole('button', { name: 'Create group' }));
  await user.tab();
  expect(document.activeElement).toBe(name);
  await user.keyboard('{Escape}');
  await user.click(screen.getByRole('button', { name: 'Cancel' }));
  expect(close).toHaveBeenCalledTimes(2);
});

it('disables buttons while saving and keeps the dialog open on rejection', async () => {
  const user = userEvent.setup();
  let reject!: (error: Error) => void;
  const onSubmit = vi.fn(
    () =>
      new Promise<void>((_, rejectPromise) => {
        reject = rejectPromise;
      }),
  );
  render(
    <ToolGroupDialog
      mode="create"
      initialName="New"
      existingNames={[]}
      onSubmit={onSubmit}
      onClose={vi.fn()}
    />,
  );
  await user.click(screen.getByRole('button', { name: 'Create group' }));
  expect(screen.getByRole('dialog').getAttribute('aria-busy')).toBe('true');
  expect((screen.getByRole('button', { name: 'Saving…' }) as HTMLButtonElement).disabled).toBe(
    true,
  );
  expect((screen.getByRole('button', { name: 'Cancel' }) as HTMLButtonElement).disabled).toBe(true);
  reject(new Error('Failed'));
  expect((await screen.findByRole('alert')).textContent).toBe('Failed');
  expect(screen.getByRole('dialog')).toBeTruthy();
});

it('submits trimmed values and displays rejected saves', async () => {
  const onSubmit = vi.fn().mockRejectedValue(new Error('Save failed'));
  render(
    <ToolGroupDialog mode="create" existingNames={[]} onSubmit={onSubmit} onClose={vi.fn()} />,
  );
  fireEvent.change(screen.getByLabelText('Name'), { target: { value: ' Team ' } });
  fireEvent.change(screen.getByLabelText('Description'), { target: { value: ' notes ' } });
  fireEvent.submit(screen.getByRole('button', { name: 'Create group' }).closest('form')!);
  await waitFor(() =>
    expect(onSubmit).toHaveBeenCalledWith({ name: 'Team', description: 'notes' }),
  );
  expect((await screen.findByRole('alert')).textContent).toBe('Save failed');
});
