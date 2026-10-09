import { fireEvent, render, screen } from '@testing-library/react';
import { expect, it, vi } from 'vitest';

import { AccessDetailHeader } from './AccessDetailHeader';

it('labels its detail heading and invokes its single collapse action', () => {
  const onClose = vi.fn();
  render(
    <AccessDetailHeader
      headingId="detail-title"
      title="Finance"
      meta={<span>Internal</span>}
      onClose={onClose}
    />,
  );
  expect(screen.getByRole('heading', { name: 'Finance' }).getAttribute('id')).toBe('detail-title');
  const close = screen.getByRole('button', { name: 'Close access editor' });
  expect(close.hasAttribute('data-access-detail-close')).toBe(true);
  fireEvent.click(close);
  expect(onClose).toHaveBeenCalledOnce();
});
