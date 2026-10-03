import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { WarningsBanner } from './WarningsBanner';

describe('WarningsBanner', () => {
  it('disables an unavailable action', () => {
    const onClick = vi.fn();
    render(
      <WarningsBanner
        warnings={['A warning']}
        action={{ label: 'Checking…', onClick, disabled: true }}
      />,
    );

    const action = screen.getByRole('button', { name: 'Checking…' });
    expect((action as HTMLButtonElement).disabled).toBe(true);
    expect(action.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(action);
    expect(onClick).not.toHaveBeenCalled();
  });
});
