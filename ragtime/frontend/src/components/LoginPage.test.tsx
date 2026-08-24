import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { AuthStatus, LoginResponse } from '@/types';

import { LoginCard } from './LoginPage';

const apiMock = vi.hoisted(() => ({
  login: vi.fn(),
  getCurrentUser: vi.fn(),
  getDebugTotpCode: vi.fn(),
}));

vi.mock('@/api', () => ({
  api: apiMock,
}));

const mfaRequiredResponse: LoginResponse = {
  success: false,
  role: 'user',
  mfa_required: true,
  mfa_challenge_token: 'challenge-token',
  mfa_methods: ['totp'],
  mfa_preferred_method: null,
};

function debugAuthStatus(overrides: Partial<AuthStatus> = {}): AuthStatus {
  return {
    authenticated: false,
    ldap_configured: false,
    local_admin_enabled: true,
    debug_mode: true,
    debug_username: 'local:debugadmin',
    debug_password: 'debug-password',
    debug_totp_code: '111111',
    api_key_configured: false,
    session_cookie_secure: false,
    allowed_origins_open: false,
    ...overrides,
  };
}

async function settleAsyncWork() {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(0);
  });
}

async function openMfaVerifyStep() {
  fireEvent.click(screen.getByRole('button', { name: 'Sign In' }));
  await settleAsyncWork();
  return screen.getByLabelText('Authenticator or recovery code');
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe('LoginCard debug TOTP pre-fill rotation', () => {
  it('keeps the pre-filled MFA code rotating while the verify step is open', async () => {
    apiMock.login.mockResolvedValue(mfaRequiredResponse);
    apiMock.getDebugTotpCode
      .mockResolvedValueOnce({ code: '222222' })
      .mockResolvedValueOnce({ code: '333333' });

    render(<LoginCard authStatus={debugAuthStatus()} onLoginSuccess={vi.fn()} />);

    const mfaInput = await openMfaVerifyStep();

    await settleAsyncWork();
    expect(mfaInput).toHaveProperty('value', '222222');

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    expect(mfaInput).toHaveProperty('value', '333333');
    expect(apiMock.getDebugTotpCode).toHaveBeenCalledTimes(2);
  });

  it('stops auto-rotation once the user edits the MFA code', async () => {
    apiMock.login.mockResolvedValue(mfaRequiredResponse);
    apiMock.getDebugTotpCode.mockResolvedValue({ code: '999999' });

    render(<LoginCard authStatus={debugAuthStatus()} onLoginSuccess={vi.fn()} />);

    const mfaInput = await openMfaVerifyStep();
    fireEvent.change(mfaInput, { target: { value: '444455' } });

    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });
    await act(async () => {
      await vi.advanceTimersByTimeAsync(15_000);
    });

    expect(mfaInput).toHaveProperty('value', '444455');
  });

  it('does not poll for debug codes on the credentials step', async () => {
    apiMock.login.mockResolvedValue(mfaRequiredResponse);

    render(<LoginCard authStatus={debugAuthStatus()} onLoginSuccess={vi.fn()} />);

    await vi.advanceTimersByTimeAsync(30_000);

    expect(apiMock.getDebugTotpCode).not.toHaveBeenCalled();
  });

  it('does not poll for debug codes when no debug code was prefilled', async () => {
    render(
      <LoginCard
        authStatus={debugAuthStatus({
          debug_totp_code: null,
          debug_username: undefined,
          debug_password: undefined,
        })}
        onLoginSuccess={vi.fn()}
      />,
    );

    await vi.advanceTimersByTimeAsync(15_000);

    expect(apiMock.getDebugTotpCode).not.toHaveBeenCalled();
  });
});
