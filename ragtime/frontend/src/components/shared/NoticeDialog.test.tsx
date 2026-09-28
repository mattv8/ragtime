import { useState } from 'react';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it } from 'vitest';

import { NoticeDialog } from './NoticeDialog';

describe('NoticeDialog', () => {
  afterEach(cleanup);

  function Harness() {
    const [isOpen, setIsOpen] = useState(false);
    return (
      <>
        <button type="button" onClick={() => setIsOpen(true)}>
          Open notice
        </button>
        {isOpen ? (
          <NoticeDialog
            title="Upload failed"
            message="The attachment could not be uploaded."
            dialogKey="upload-failure"
            onClose={() => setIsOpen(false)}
          />
        ) : null}
      </>
    );
  }

  it('uses alert dialog semantics and focuses the confirmation action', async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: 'Open notice' }));

    const dialog = screen.getByRole('alertdialog', { name: 'Upload failed' });
    expect(dialog.getAttribute('aria-modal')).toBe('true');
    expect(dialog.getAttribute('aria-describedby')).toBeTruthy();
    expect(screen.getByText('The attachment could not be uploaded.')).toBeTruthy();
    expect(document.querySelector('[data-notice-dialog="upload-failure"]')).toBeTruthy();
    await waitFor(() =>
      expect(document.activeElement).toBe(screen.getByRole('button', { name: 'OK' })),
    );
  });

  it.each([
    ['Escape', async (user: ReturnType<typeof userEvent.setup>) => user.keyboard('{Escape}')],
    [
      'backdrop',
      async (user: ReturnType<typeof userEvent.setup>) =>
        user.click(document.querySelector('[data-notice-dialog="upload-failure"]') as HTMLElement),
    ],
    [
      'confirmation',
      async (user: ReturnType<typeof userEvent.setup>) =>
        user.click(screen.getByRole('button', { name: 'OK' })),
    ],
  ])('closes through %s and restores opener focus', async (_method, close) => {
    const user = userEvent.setup();
    render(<Harness />);

    const trigger = screen.getByRole('button', { name: 'Open notice' });
    await user.click(trigger);
    await close(user);

    expect(screen.queryByRole('alertdialog')).toBeNull();
    expect(document.activeElement).toBe(trigger);
  });

  it('closes through the close button', async () => {
    const user = userEvent.setup();
    render(<Harness />);

    await user.click(screen.getByRole('button', { name: 'Open notice' }));
    fireEvent.click(screen.getByRole('button', { name: 'Close' }));

    expect(screen.queryByRole('alertdialog')).toBeNull();
  });
});
