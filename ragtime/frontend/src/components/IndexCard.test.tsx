import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { IndexCard } from './IndexCard';

describe('IndexCard', () => {
  it('disables the toggle checkbox when toggleDisabled is true', async () => {
    const user = userEvent.setup();
    const onToggle = vi.fn();

    render(
      <IndexCard title="Git index" enabled={true} onToggle={onToggle} toggleDisabled={true} />,
    );

    const checkbox = screen.getByRole('checkbox');
    expect((checkbox as HTMLInputElement).disabled).toBe(true);

    await user.click(checkbox);
    expect(onToggle).not.toHaveBeenCalled();
  });
});
