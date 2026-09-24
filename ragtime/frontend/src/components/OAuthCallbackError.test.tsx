import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { OAuthCallbackError } from './OAuthCallbackError';

vi.mock('./WebGLGradient', () => ({
  default: () => <div data-testid="webgl-gradient" />,
}));

describe('OAuthCallbackError gradient shell', () => {
  it('renders the callback error card inside the shared auth gradient surface', () => {
    render(
      <OAuthCallbackError
        title="Bad redirect"
        summary="The redirect_uri is not allowed."
        nextSteps={['Check the client configuration.']}
      />,
    );

    const surface = document.querySelector('[data-auth-surface="gradient"]');
    expect(surface).toBeTruthy();
    expect(screen.getByTestId('webgl-gradient')).toBeTruthy();
    expect(screen.getByText('OAuth Callback Error')).toBeTruthy();
    expect(screen.getByText('Bad redirect')).toBeTruthy();
  });

  it('renders optional authorization recovery actions', () => {
    const onRetry = vi.fn();
    const onBack = vi.fn();

    render(
      <OAuthCallbackError title="Failed" summary="Try again." onRetry={onRetry} onBack={onBack} />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Retry authorization' }));
    fireEvent.click(screen.getByRole('button', { name: 'Back to workspace' }));
    expect(onRetry).toHaveBeenCalledOnce();
    expect(onBack).toHaveBeenCalledOnce();
    expect(document.querySelector('#oauth-authorization-error')).toBeTruthy();
  });
});
