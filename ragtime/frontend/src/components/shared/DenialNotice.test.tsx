import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { DenialNotice } from './DenialNotice';

describe('DenialNotice', () => {
  it('renders safe supplied reason as a labelled note without actions', () => {
    render(
      <DenialNotice
        requestId="request-1"
        reason="This request cannot be completed."
        timestamp="2026-10-10T12:00:00.000Z"
      />,
    );

    expect(
      screen.getByRole('note', { name: 'Request blocked' }).getAttribute('data-denial-request-id'),
    ).toBe('request-1');
    expect(screen.getByText('This request cannot be completed.')).not.toBeNull();
    expect(screen.queryByRole('button')).toBeNull();
  });
});
