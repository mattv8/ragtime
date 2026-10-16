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

it('uses the supplied close label', () => {
  render(
    <AccessDetailHeader
      headingId="detail-title"
      title="Finance"
      closeLabel="Return to access levels"
      onClose={vi.fn()}
    />,
  );

  expect(screen.getByRole('button', { name: 'Return to access levels' })).toBeTruthy();
});

it('keeps a right-aligned action separate from a long title', () => {
  render(
    <AccessDetailHeader
      headingId="detail-title"
      title="A very long access level name that should retain its action"
      actions={<button type="button">Make default</button>}
      onClose={vi.fn()}
    />,
  );

  expect(document.querySelector('[data-access-detail-actions]')).toBeTruthy();
  expect(screen.getByRole('button', { name: 'Make default' })).toBeTruthy();
});
