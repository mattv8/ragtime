import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { UserSpaceStatusOverlay, type UserSpaceStatusOverlayItem } from './UserSpaceStatusOverlay';

describe('UserSpaceStatusOverlay', () => {
  it('keeps progress rows visible while only notice rows are dismissible', async () => {
    const user = userEvent.setup();
    const onDismiss = vi.fn();
    const items: UserSpaceStatusOverlayItem[] = [
      {
        id: 'progress',
        tone: 'status',
        content: 'Bootstrapping workspace...',
      },
      {
        id: 'preview',
        tone: 'success',
        content: 'Preview is ready.',
        dismissLabel: 'Dismiss preview notice',
      },
      {
        id: 'warning',
        tone: 'warning',
        content: 'Possible live data query issue: timed out',
        dismissLabel: 'Dismiss live data warning',
      },
      {
        id: 'error',
        tone: 'error',
        content: 'Runtime failed.',
        dismissLabel: 'Dismiss error message',
      },
    ];

    render(<UserSpaceStatusOverlay items={items} onDismiss={onDismiss} />);

    const progressRow = screen.getByText('Bootstrapping workspace...').closest('p');
    expect(progressRow).not.toBeNull();
    expect(within(progressRow as HTMLElement).queryByRole('button')).toBeNull();
    expect(screen.getAllByRole('button')).toHaveLength(3);

    await user.click(screen.getByRole('button', { name: 'Dismiss preview notice' }));
    expect(onDismiss).toHaveBeenNthCalledWith(1, 'preview');

    await user.click(screen.getByRole('button', { name: 'Dismiss live data warning' }));
    expect(onDismiss).toHaveBeenNthCalledWith(2, 'warning');

    await user.click(screen.getByRole('button', { name: 'Dismiss error message' }));
    expect(onDismiss).toHaveBeenNthCalledWith(3, 'error');
  });
});
