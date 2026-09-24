import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { sessionLifecycle } from '@/auth/sessionLifecycle';
import { useAuthSession } from '@/auth/useAuthSession';
import { Manage2FAModal } from './Manage2FAModal';

const authenticatedStatus = {
  authenticated: true,
  ldap_configured: false,
  local_admin_enabled: true,
  debug_mode: false,
  api_key_configured: true,
  session_cookie_secure: true,
  allowed_origins_open: false,
  chat_enabled: false,
  userspace_generation_enabled: false,
};

const currentUser = {
  id: 'account-under-test',
  username: 'account-under-test',
  display_name: 'Account Under Test',
  email: null,
  role: 'user' as const,
  auth_provider: 'local' as const,
  chat_enabled_effective: false,
  userspace_generation_enabled_effective: false,
};

const renewedUser = { ...currentUser, username: 'account-renewed' };

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

function AuthenticatedMfaHarness() {
  const { currentUser: user, phase, refresh } = useAuthSession();

  if (phase !== 'authenticated' || !user) {
    return <p>Private account controls are unavailable.</p>;
  }

  return (
    <section aria-label="Authenticated account controls">
      <p>Signed in as {user.username}</p>
      <button type="button" onClick={() => void refresh('return')}>
        Refresh session
      </button>
      <Manage2FAModal isOpen onClose={() => undefined} />
    </section>
  );
}

function installAuthenticatedFetch({
  deferRenewalPair = false,
  deferMfaStatus = false,
  deferEnrollment = false,
  anonymousDuringRenewal = false,
}: {
  deferRenewalPair?: boolean;
  deferMfaStatus?: boolean;
  deferEnrollment?: boolean;
  anonymousDuringRenewal?: boolean;
} = {}) {
  let releaseStatus: ((response: Response) => void) | undefined;
  let releaseUser: ((response: Response) => void) | undefined;
  let releaseMfaStatus: ((response: Response) => void) | undefined;
  let releaseEnrollment: ((response: Response) => void) | undefined;
  let statusCalls = 0;

  vi.stubGlobal(
    'fetch',
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith('/auth/mfa/status')) {
        if (deferMfaStatus) {
          return new Promise<Response>((resolve) => {
            releaseMfaStatus = resolve;
          });
        }
        return Promise.resolve(
          json({
            enabled: false,
            required: false,
            recovery_codes_remaining: 0,
            methods_enrolled: [],
            allowed_methods: ['totp'],
            webauthn_credential_count: 0,
          }),
        );
      }
      if (url.endsWith('/auth/mfa/webauthn/credentials')) {
        return Promise.resolve(json({ credentials: [] }));
      }
      if (url.endsWith('/auth/mfa/enroll/start')) {
        return Promise.resolve(
          json({
            secret: 'SYNTHETIC-TOTP-SECRET',
            otpauth_uri: 'otpauth://totp/Synthetic?secret=SYNTHETIC-TOTP-SECRET',
            enrollment_token: 'synthetic-enrollment-token',
          }),
        );
      }
      if (url.endsWith('/auth/mfa/enroll/complete')) {
        if (deferEnrollment) {
          return new Promise<Response>((resolve) => {
            releaseEnrollment = resolve;
          });
        }
        return Promise.resolve(
          json({
            success: true,
            user: currentUser,
            recovery_codes: ['synthetic-recovery-one', 'synthetic-recovery-two'],
          }),
        );
      }
      if (url.endsWith('/auth/status')) {
        statusCalls += 1;
        if (anonymousDuringRenewal && statusCalls === 2) {
          return Promise.resolve(json({ ...authenticatedStatus, authenticated: false }));
        }
        if (deferRenewalPair && statusCalls === 2) {
          return new Promise<Response>((resolve) => {
            releaseStatus = resolve;
          });
        }
        return Promise.resolve(json(authenticatedStatus));
      }
      if (url.endsWith('/auth/me')) {
        if (deferRenewalPair && statusCalls === 2) {
          return new Promise<Response>((resolve) => {
            releaseUser = resolve;
          });
        }
        return Promise.resolve(json(currentUser));
      }
      return Promise.resolve(json({ detail: `Unexpected request: ${url}` }, 404));
    }),
  );

  return {
    releaseRenewalStatus: () => releaseStatus?.(json(authenticatedStatus)),
    releaseRenewalUser: () => releaseUser?.(json(renewedUser)),
    releaseMfaStatus: () =>
      releaseMfaStatus?.(
        json({
          enabled: false,
          required: false,
          recovery_codes_remaining: 0,
          methods_enrolled: [],
          allowed_methods: ['totp'],
          webauthn_credential_count: 0,
        }),
      ),
    expireMfaStatus: () => releaseMfaStatus?.(json({ detail: 'expired session' }, 401)),
    releaseEnrollment: () =>
      releaseEnrollment?.(
        json({
          success: true,
          user: currentUser,
          recovery_codes: ['synthetic-recovery-one', 'synthetic-recovery-two'],
        }),
      ),
  };
}

beforeEach(() => {
  if (sessionLifecycle.signedOutIntent) sessionLifecycle.retrySignedOutSession();
});

afterEach(() => {
  cleanup();
  if (sessionLifecycle.phase === 'authenticated') {
    sessionLifecycle.expire(sessionLifecycle.capture('session'));
  }
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('Manage2FAModal account enrollment session lifecycle', () => {
  it('waits for MFA status before mounting enrollment controls', async () => {
    const { releaseMfaStatus } = installAuthenticatedFetch({ deferMfaStatus: true });
    const { rerender } = render(<Manage2FAModal isOpen={false} onClose={() => undefined} />);

    rerender(<Manage2FAModal isOpen onClose={() => undefined} />);

    expect((await screen.findByRole('status')).textContent).toBe('Loading status...');
    expect(
      vi
        .mocked(fetch)
        .mock.calls.filter(([request]) => String(request).endsWith('/auth/mfa/enroll/start')),
    ).toHaveLength(0);

    releaseMfaStatus();
    const verificationCode = await screen.findByLabelText('Verification code');
    expect(verificationCode).toBeTruthy();
  });

  it('keeps recovery codes mounted through a tokenless TOTP renewal until Done', async () => {
    const { releaseRenewalStatus, releaseRenewalUser } = installAuthenticatedFetch({
      deferRenewalPair: true,
    });
    render(<AuthenticatedMfaHarness />);

    await screen.findByText('Signed in as account-under-test');
    const verificationCode = await screen.findByLabelText('Verification code');
    const generationBeforeEnrollment = sessionLifecycle.generation;

    fireEvent.change(verificationCode, { target: { value: '111111' } });
    fireEvent.click(screen.getByRole('button', { name: 'Finish setup' }));

    await screen.findByText('Save these recovery codes now. They will not be shown again.');
    expect(screen.getByText('synthetic-recovery-one')).toBeTruthy();
    expect(sessionLifecycle.generation).toBeGreaterThan(generationBeforeEnrollment);

    releaseRenewalStatus();
    await waitFor(() =>
      expect(
        vi.mocked(fetch).mock.calls.filter(([request]) => String(request).endsWith('/auth/me')),
      ).toHaveLength(2),
    );
    expect(screen.getByText('synthetic-recovery-one')).toBeTruthy();

    releaseRenewalUser();
    await screen.findByText('Signed in as account-renewed');
    expect(screen.getByText('synthetic-recovery-one')).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    expect(screen.getByText('synthetic-recovery-one')).toBeTruthy();
    fireEvent.click(screen.getByRole('button', { name: 'Done' }));
    await waitFor(() => expect(screen.queryByText('synthetic-recovery-one')).toBeNull());
    expect(screen.getByText('Authenticator:')).toBeTruthy();
  });

  it('removes the private modal when a current account request expires the session', async () => {
    const { expireMfaStatus } = installAuthenticatedFetch({ deferMfaStatus: true });
    render(<AuthenticatedMfaHarness />);

    await screen.findByText('Signed in as account-under-test');
    expect(screen.getByText('Manage 2FA')).toBeTruthy();
    expireMfaStatus();
    await waitFor(() =>
      expect(screen.getByText('Private account controls are unavailable.')).toBeTruthy(),
    );
    expect(screen.queryByText('Manage 2FA')).toBeNull();
  });

  it('keeps the real wizard mounted when an old cookie reports anonymous before enrollment returns', async () => {
    const { releaseEnrollment } = installAuthenticatedFetch({
      deferEnrollment: true,
      anonymousDuringRenewal: true,
    });
    render(<AuthenticatedMfaHarness />);

    await screen.findByText('Signed in as account-under-test');
    const verificationCode = await screen.findByLabelText('Verification code');
    fireEvent.change(verificationCode, { target: { value: '111111' } });
    fireEvent.click(screen.getByRole('button', { name: 'Finish setup' }));
    fireEvent.click(screen.getByRole('button', { name: 'Refresh session' }));

    await waitFor(() => expect(screen.getByText('Manage 2FA')).toBeTruthy());
    expect(screen.getByText('Signed in as account-under-test')).toBeTruthy();
    releaseEnrollment();
    await screen.findByText('Save these recovery codes now. They will not be shown again.');
  });
});
