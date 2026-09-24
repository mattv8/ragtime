import { cleanup, render, screen, within } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AuthRecoveryState, type AuthRecoveryStateProps } from './AuthRecoveryState';

function renderRecovery(props: Partial<AuthRecoveryStateProps> = {}) {
  const defaultProps: AuthRecoveryStateProps = {
    action: 'retry-bootstrap',
    busy: false,
    message: 'Unable to connect',
    onRetry: vi.fn(),
    ...props,
  };
  return render(<AuthRecoveryState {...defaultProps} />);
}

function getButton() {
  const region = screen.getByRole('region');
  return within(region).getByRole('button');
}

afterEach(() => {
  cleanup();
});

describe('AuthRecoveryState', () => {
  describe('action labels', () => {
    it('displays "Try connecting again" for retry-bootstrap action', () => {
      renderRecovery({ action: 'retry-bootstrap' });
      expect(getButton().textContent).toContain('Try connecting again');
    });

    it('displays "Retry sign out" for retry-logout action', () => {
      renderRecovery({ action: 'retry-logout' });
      expect(getButton().textContent).toContain('Retry sign out');
    });

    it('displays "Check session again" for check-session action', () => {
      renderRecovery({ action: 'check-session' });
      expect(getButton().textContent).toContain('Check session again');
    });
  });

  describe('stable action names', () => {
    it('maintains stable button label while busy', () => {
      const { rerender } = render(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="Failed"
          onRetry={() => {}}
        />,
      );

      const initialLabel = getButton().textContent;

      rerender(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={true}
          message="Retrying..."
          onRetry={() => {}}
        />,
      );

      expect(getButton().textContent).toBe(initialLabel);
    });
  });

  describe('disabled busy behavior', () => {
    it('disables button when busy', () => {
      renderRecovery({ busy: true });
      const button = getButton();
      expect((button as HTMLButtonElement).disabled).toBe(true);
    });

    it('enables button when not busy', () => {
      renderRecovery({ busy: false });
      const button = getButton();
      expect((button as HTMLButtonElement).disabled).toBe(false);
    });

    it('prevents double-submit when button is disabled', async () => {
      const onRetry = vi.fn();
      renderRecovery({ busy: true, onRetry });
      const button = getButton();

      await userEvent.click(button);
      expect(onRetry).not.toHaveBeenCalled();
    });
  });

  describe('callback behavior', () => {
    it('invokes onRetry when button is clicked', async () => {
      const onRetry = vi.fn();
      renderRecovery({ onRetry, busy: false });

      await userEvent.click(getButton());
      expect(onRetry).toHaveBeenCalledTimes(1);
    });

    it('does not invoke callback when disabled', async () => {
      const onRetry = vi.fn();
      renderRecovery({ onRetry, busy: true });

      await userEvent.click(getButton());
      expect(onRetry).not.toHaveBeenCalled();
    });
  });

  describe('error and progress semantics', () => {
    it('renders message as alert role when present', () => {
      renderRecovery({ message: 'Network error occurred' });
      const alert = screen.getByRole('alert');
      expect(alert).toBeTruthy();
      expect(alert.textContent).toContain('Network error occurred');
    });

    it('does not render alert when message is empty', () => {
      renderRecovery({ message: '' });
      expect(screen.queryByRole('alert')).toBeNull();
    });

    it('announces progress with polite live region', () => {
      renderRecovery({ busy: true });
      const region = screen.getByRole('region');
      expect(region.getAttribute('aria-live')).toBeNull();
      expect(screen.getByText(/Attempting recovery/i).getAttribute('aria-live')).toBe('polite');
    });

    it('keeps one polite progress region mounted across busy transitions', () => {
      const { rerender } = render(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="Failed"
          onRetry={() => {}}
        />,
      );
      const liveRegion = document.querySelector('[aria-live="polite"]');
      expect(liveRegion).toBeTruthy();

      rerender(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={true}
          message="Failed"
          onRetry={() => {}}
        />,
      );

      expect(document.querySelector('[aria-live="polite"]')).toBe(liveRegion);
      expect(liveRegion?.textContent).toContain('Attempting recovery');
    });

    it('shows progress indicator when busy', () => {
      renderRecovery({ busy: true });
      expect(screen.getByText(/Attempting recovery/i)).toBeTruthy();
    });

    it('hides progress indicator when not busy', () => {
      renderRecovery({ busy: false });
      expect(screen.queryByText(/Attempting recovery/i)).toBeNull();
    });
  });

  describe('aria-busy region', () => {
    it('marks region as busy when busy=true', () => {
      renderRecovery({ busy: true });
      const region = screen.getByRole('region');
      expect(region.getAttribute('aria-busy')).toBe('true');
    });

    it('marks region as not busy when busy=false', () => {
      renderRecovery({ busy: false });
      const region = screen.getByRole('region');
      expect(region.getAttribute('aria-busy')).toBe('false');
    });

    it('applies aria-busy to button', () => {
      renderRecovery({ busy: true });
      expect(getButton().getAttribute('aria-busy')).toBe('true');
    });
  });

  describe('stable boundary identity', () => {
    it('has stable auth-recovery-state id', () => {
      renderRecovery();
      expect(screen.getByRole('region').getAttribute('id')).toBe('auth-recovery-state');
    });
  });

  describe('initial and failure focus', () => {
    it('focuses button on initial mount', () => {
      renderRecovery({ busy: false });
      expect(document.activeElement).toBe(getButton());
    });

    it('focuses button when transitioning from busy to not busy (failed retry)', () => {
      const { rerender } = render(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={true}
          message="Retrying..."
          onRetry={() => {}}
        />,
      );

      const button = getButton();
      expect(document.activeElement).not.toBe(button);

      rerender(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="Failed to connect"
          onRetry={() => {}}
        />,
      );

      expect(document.activeElement).toBe(button);
    });

    it('does not focus button while busy (active retry)', () => {
      renderRecovery({ busy: true });
      expect(document.activeElement).not.toBe(getButton());
    });
  });

  describe('styling and accessibility', () => {
    it('applies login-card class for consistent auth styling', () => {
      renderRecovery();
      const region = screen.getByRole('region');
      expect(region.classList.contains('login-card')).toBe(true);
    });

    it('applies btn and btn-primary classes to button', () => {
      renderRecovery();
      const button = getButton();
      expect(button.classList.contains('btn')).toBe(true);
      expect(button.classList.contains('btn-primary')).toBe(true);
      expect(button.classList.contains('login-submit')).toBe(true);
    });

    it('includes h1 with login-title class for header', () => {
      renderRecovery();
      const heading = screen.getByRole('heading', { level: 1 });
      expect(heading.classList.contains('login-title')).toBe(true);
      expect(heading.textContent).toContain('Connection Issue');
    });
  });

  describe('message display', () => {
    it('displays custom message text', () => {
      const message = 'Server temporarily unavailable';
      renderRecovery({ message });
      const alert = screen.getByRole('alert');
      expect(alert.textContent).toContain(message);
    });

    it('updates message when prop changes', () => {
      const { rerender } = render(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="First error"
          onRetry={() => {}}
        />,
      );

      let alert = screen.getByRole('alert');
      expect(alert.textContent).toContain('First error');

      rerender(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="Second error"
          onRetry={() => {}}
        />,
      );

      alert = screen.getByRole('alert');
      expect(alert.textContent).toContain('Second error');
    });
  });

  describe('action changes', () => {
    it('updates action label when action prop changes', () => {
      const { rerender } = render(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="Error"
          onRetry={() => {}}
        />,
      );

      expect(getButton().textContent).toContain('Try connecting again');

      rerender(
        <AuthRecoveryState action="retry-logout" busy={false} message="Error" onRetry={() => {}} />,
      );

      expect(getButton().textContent).toContain('Retry sign out');
    });
  });

  describe('region aria-label', () => {
    it('includes action name in region aria-label', () => {
      renderRecovery({ action: 'retry-bootstrap' });
      const region = screen.getByRole('region');
      const ariaLabel = region.getAttribute('aria-label');
      expect(ariaLabel).toBeTruthy();
      expect(ariaLabel).toContain('Try connecting again');
    });

    it('updates aria-label when action changes', () => {
      const { rerender } = render(
        <AuthRecoveryState
          action="retry-bootstrap"
          busy={false}
          message="Error"
          onRetry={() => {}}
        />,
      );

      let region = screen.getByRole('region');
      let ariaLabel = region.getAttribute('aria-label');
      expect(ariaLabel).toContain('Try connecting again');

      rerender(
        <AuthRecoveryState
          action="check-session"
          busy={false}
          message="Error"
          onRetry={() => {}}
        />,
      );

      region = screen.getByRole('region');
      ariaLabel = region.getAttribute('aria-label');
      expect(ariaLabel).toContain('Check session again');
    });
  });
});
