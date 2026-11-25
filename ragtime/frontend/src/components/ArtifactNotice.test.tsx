import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ArtifactNotice } from './ArtifactNotice';

afterEach(cleanup);

describe('ArtifactNotice', () => {
  it('offers retry for errors and calls the supplied handler', () => {
    const retry = vi.fn();
    render(<ArtifactNotice state={{ reason: 'error' }} onRetry={retry} />);
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(retry).toHaveBeenCalledOnce();
    expect(screen.getByRole('alert')).toBeDefined();
  });

  it('does not offer retry for unavailable exports', () => {
    render(<ArtifactNotice state={{ reason: 'notfound' }} onRetry={vi.fn()} />);
    expect(screen.getByRole('status')).toBeDefined();
    expect(screen.queryByRole('button')).toBeNull();
    expect(screen.queryByRole('heading')).toBeNull();
    expect(screen.getByText('Export not available').tagName).toBe('P');
  });
});
