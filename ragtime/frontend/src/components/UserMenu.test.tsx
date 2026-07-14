import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { UserMenu } from './UserMenu';
import type { User, WebauthnCredentialSummary } from '@/types';

const apiMock = vi.hoisted(() => ({
  getMfaStatus: vi.fn(),
  listWebauthnCredentials: vi.fn(),
  updateMyThemePack: vi.fn(),
  deleteWebauthnCredential: vi.fn(),
  renameWebauthnCredential: vi.fn(),
  startWebauthnRegistration: vi.fn(),
  completeWebauthnRegistration: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));
vi.mock('@/utils/webauthn', () => ({
  createPasskeyCredential: vi.fn(),
  isWebAuthnSupported: () => true,
}));
vi.mock('./DeleteConfirmButton', () => ({
  DeleteConfirmButton: ({
    onDelete,
    disabled,
    buttonText,
  }: {
    onDelete: () => void;
    disabled?: boolean;
    buttonText?: string;
  }) => (
    <button type="button" disabled={disabled} onClick={onDelete}>
      {buttonText ?? 'Delete'}
    </button>
  ),
}));

function createStorageStub(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear: vi.fn(() => store.clear()),
    getItem: vi.fn((key: string) => store.get(key) ?? null),
    key: vi.fn((index: number) => Array.from(store.keys())[index] ?? null),
    removeItem: vi.fn((key: string) => {
      store.delete(key);
    }),
    setItem: vi.fn((key: string, value: string) => {
      store.set(key, value);
    }),
  };
}

function installLocalStorageStub() {
  const storage = createStorageStub();
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: storage,
  });
}

const user: User = {
  id: 'user-1',
  username: 'alice',
  display_name: 'Alice',
  email: 'alice@example.com',
  role: 'user',
  auth_provider: 'local',
  theme_pack: null,
};

const oldCredential: WebauthnCredentialSummary = {
  id: 'old-row-id',
  name: 'Old passkey',
  created_at: '2026-01-01T00:00:00Z',
  last_used_at: null,
  transports: [],
};

async function openPasskeysTab(userInteraction: ReturnType<typeof userEvent.setup>) {
  await userInteraction.click(screen.getByRole('button', { name: /alice/i }));
  await userInteraction.click(screen.getByRole('button', { name: /manage 2fa/i }));
  await userInteraction.click(await screen.findByRole('tab', { name: /passkeys/i }));
}

async function deleteOldPasskey(userInteraction: ReturnType<typeof userEvent.setup>) {
  await screen.findByText('Old passkey');
  await userInteraction.click(screen.getByRole('button', { name: 'Delete' }));
}

describe('UserMenu passkey management', () => {
  beforeEach(() => {
    installLocalStorageStub();
    apiMock.getMfaStatus.mockResolvedValue({
      enabled: true,
      required: false,
      recovery_codes_remaining: 5,
      methods_enrolled: ['webauthn'],
      allowed_methods: ['totp', 'webauthn'],
      webauthn_credential_count: 1,
    });
    apiMock.updateMyThemePack.mockResolvedValue(undefined);
    apiMock.deleteWebauthnCredential.mockResolvedValue(undefined);
    apiMock.renameWebauthnCredential.mockResolvedValue(oldCredential);
    apiMock.startWebauthnRegistration.mockResolvedValue({
      options: {},
      registration_token: 'token',
    });
    apiMock.completeWebauthnRegistration.mockResolvedValue({
      success: true,
      credential_id: 'new-credential-id',
      name: 'Passkey',
    });
  });

  afterEach(() => {
    cleanup();
    // resetAllMocks (not clearAllMocks) also clears queued mock*Once implementations,
    // preventing a never-resolving `mockReturnValueOnce` in one test from leaking
    // into the next when the modal reloads its passkey list asynchronously.
    vi.resetAllMocks();
  });

  it('does not show stale passkeys while refreshing the modal list', async () => {
    const userInteraction = userEvent.setup();
    apiMock.listWebauthnCredentials.mockResolvedValueOnce({ credentials: [oldCredential] });
    render(<UserMenu user={user} onLogout={vi.fn()} />);

    await openPasskeysTab(userInteraction);
    await screen.findByText('Old passkey');

    const overlay = document.querySelector('.modal-overlay');
    if (!overlay) {
      throw new Error('Expected passkeys modal overlay to be rendered');
    }
    fireEvent.click(overlay);
    await waitFor(() => {
      expect(document.querySelector('.modal-overlay')).toBeNull();
    });

    apiMock.listWebauthnCredentials.mockReturnValueOnce(new Promise(() => {}));
    // Clicking a tab (a userEvent mousedown) closes the user-menu dropdown, so
    // reopen it before reopening the modal.
    await openPasskeysTab(userInteraction);

    await waitFor(() => {
      expect(screen.queryByText('Loading passkeys...')).not.toBeNull();
    });
    expect(screen.queryByText('Old passkey')).toBeNull();
  });

  it('removes a passkey row when delete reports it is already missing on the server', async () => {
    const userInteraction = userEvent.setup();
    apiMock.listWebauthnCredentials
      .mockResolvedValueOnce({ credentials: [oldCredential] })
      .mockResolvedValueOnce({ credentials: [] });
    apiMock.deleteWebauthnCredential.mockRejectedValueOnce(
      Object.assign(new Error('Credential not found'), { status: 404 }),
    );
    render(<UserMenu user={user} onLogout={vi.fn()} />);

    await openPasskeysTab(userInteraction);
    await deleteOldPasskey(userInteraction);

    await waitFor(() => {
      expect(screen.queryByText('Old passkey')).toBeNull();
    });
    expect(screen.queryByText('Credential not found')).toBeNull();
    expect(apiMock.deleteWebauthnCredential).toHaveBeenCalledWith('old-row-id');
  });

  it('restores the passkey row when delete fails and refresh also fails', async () => {
    const userInteraction = userEvent.setup();
    apiMock.listWebauthnCredentials
      .mockResolvedValueOnce({ credentials: [oldCredential] })
      .mockRejectedValueOnce(new Error('Failed to refresh passkeys'));
    apiMock.deleteWebauthnCredential.mockRejectedValueOnce(new Error('Delete failed'));
    render(<UserMenu user={user} onLogout={vi.fn()} />);

    await openPasskeysTab(userInteraction);
    await deleteOldPasskey(userInteraction);

    await screen.findByText('Delete failed');
    expect(screen.queryByText('Old passkey')).not.toBeNull();
  });

  it('keeps the row and shows the error when delete fails for an existing passkey', async () => {
    const userInteraction = userEvent.setup();
    apiMock.listWebauthnCredentials
      .mockResolvedValueOnce({ credentials: [oldCredential] })
      .mockResolvedValueOnce({ credentials: [oldCredential] });
    apiMock.deleteWebauthnCredential.mockRejectedValueOnce(
      Object.assign(new Error('Delete failed'), { status: 500 }),
    );
    render(<UserMenu user={user} onLogout={vi.fn()} />);

    await openPasskeysTab(userInteraction);
    await deleteOldPasskey(userInteraction);

    await screen.findByText('Delete failed');
    expect(screen.queryByText('Old passkey')).not.toBeNull();
  });
});
