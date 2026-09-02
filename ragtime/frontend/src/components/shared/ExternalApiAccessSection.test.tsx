import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ExternalApiAccessSection } from './ExternalApiAccessSection';

const apiMock = vi.hoisted(() => ({
  getWorkspaceExternalApiManifest: vi.fn(),
  listWorkspaceExternalApiEndpoints: vi.fn(),
  createWorkspaceExternalApiEndpoint: vi.fn(),
  deleteWorkspaceExternalApiEndpoint: vi.fn(),
  listWorkspaceExternalApiCredentials: vi.fn(),
  createWorkspaceExternalApiCredential: vi.fn(),
  rotateWorkspaceExternalApiCredential: vi.fn(),
  revokeWorkspaceExternalApiCredential: vi.fn(),
  listWorkspaceExternalApiRequests: vi.fn(),
}));

vi.mock('@/api', () => ({ api: apiMock }));

function manifestResponse() {
  return {
    preview_origin: 'https://ws-preview.example.com',
    version: 1,
    valid: true,
    errors: [],
    candidates: [
      {
        key: 'accounting-periods',
        label: 'Accounting periods',
        description: 'Lists periods.',
        method: 'GET',
        path: '/backend/periods',
        valid: true,
        errors: [],
      },
      {
        key: 'trial-balance',
        label: 'Trial balance',
        description: 'Returns trial balance rows.',
        method: 'HEAD',
        path: '/backend/trial-balance',
        valid: true,
        errors: [],
      },
    ],
  };
}

function publishedEndpoint(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: 'endpoint-1',
    key: 'accounting-periods',
    label: 'Accounting periods',
    description: 'Lists periods.',
    method: 'GET',
    path: '/backend/periods',
    enabled: true,
    stale: false,
    definition_hash: 'hash-1',
    approved_at: '2026-09-01T12:00:00Z',
    ...overrides,
  };
}

function credentialItem(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: 'cred-1',
    label: 'August workpapers',
    token_prefix: 'rtws_abcd1234',
    enabled: true,
    expires_at: '2026-09-30T12:00:00Z',
    last_used_at: '2026-09-01T12:30:00Z',
    request_count: 14,
    revoked_at: null,
    endpoint_keys: ['accounting-periods'],
    ...overrides,
  };
}

function requestHistoryItem(overrides: Partial<Record<string, unknown>> = {}) {
  return {
    id: 'req-1',
    credential_id: 'cred-1',
    credential_label: 'August workpapers',
    endpoint_key: 'accounting-periods',
    endpoint_label: 'Accounting periods',
    method: 'GET',
    path_template: '/backend/periods',
    status_code: 200,
    duration_ms: 82,
    created_at: '2026-09-01T12:31:00Z',
    ...overrides,
  };
}

beforeEach(() => {
  vi.spyOn(window, 'confirm').mockReturnValue(true);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe('ExternalApiAccessSection', () => {
  it('loads candidates, shows stale publication state, creates a credential, and reveals token examples once', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceExternalApiManifest.mockResolvedValue(manifestResponse());
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValue({
      preview_origin: 'https://ws-preview.example.com',
      items: [publishedEndpoint({ stale: true, definition_hash: 'hash-stale' })],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValue({ items: [] });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValue({
      cursor: null,
      limit: 20,
      items: [],
    });
    apiMock.createWorkspaceExternalApiEndpoint.mockResolvedValue(
      publishedEndpoint({ stale: false }),
    );
    apiMock.createWorkspaceExternalApiCredential.mockResolvedValue({
      id: 'cred-1',
      label: 'August workpapers',
      token: 'rtws_public_selector_secret',
      token_prefix: 'rtws_public_selector',
      enabled: true,
      expires_at: null,
      endpoint_keys: ['accounting-periods'],
    });

    render(
      <ExternalApiAccessSection
        workspaceId="ws-1"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    expect(await screen.findByText('Accounting periods')).toBeTruthy();
    expect(screen.getByText(/stale/i)).toBeTruthy();
    expect(screen.getByRole('button', { name: /reapprove/i })).toBeTruthy();

    await user.click(screen.getByRole('button', { name: /reapprove/i }));
    expect(apiMock.createWorkspaceExternalApiEndpoint).toHaveBeenCalledWith(
      'ws-1',
      'accounting-periods',
    );

    await user.click(screen.getByRole('checkbox', { name: /accounting periods/i }));
    await user.type(screen.getByLabelText(/credential label/i), 'August workpapers');
    await user.click(screen.getByRole('button', { name: /create credential/i }));

    expect(apiMock.createWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', {
      label: 'August workpapers',
      endpoint_keys: ['accounting-periods'],
      expires_at: null,
    });

    expect(await screen.findByDisplayValue('rtws_public_selector_secret')).toBeTruthy();
    expect(screen.getByText(/Authorization: Bearer rtws_public_selector_secret/)).toBeTruthy();
    expect(
      screen.getByText(/Web\.Contents\("https:\/\/ws-preview\.example\.com\/backend\/periods"/),
    ).toBeTruthy();
    expect(screen.queryByText('rtws_abcd1234')).toBeNull();
  });

  it('never renders raw secrets in the credential list, supports rotate and revoke confirmations, and loads request history', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceExternalApiManifest.mockResolvedValue(manifestResponse());
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValue({
      preview_origin: 'https://ws-preview.example.com',
      items: [publishedEndpoint()],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValue({
      items: [credentialItem()],
    });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValue({
      cursor: null,
      limit: 20,
      items: [requestHistoryItem()],
    });
    apiMock.rotateWorkspaceExternalApiCredential.mockResolvedValue({
      id: 'cred-1',
      label: 'August workpapers',
      token: 'rtws_rotated_selector_secret',
      token_prefix: 'rtws_rotated_selector',
      enabled: true,
      expires_at: '2026-09-30T12:00:00Z',
      endpoint_keys: ['accounting-periods'],
    });
    apiMock.revokeWorkspaceExternalApiCredential.mockResolvedValue(
      credentialItem({ enabled: false, revoked_at: '2026-09-02T00:00:00Z' }),
    );

    render(
      <ExternalApiAccessSection
        workspaceId="ws-1"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    const credentialsRegion = await screen.findByRole('region', { name: /service credentials/i });
    expect(within(credentialsRegion).getByText('August workpapers')).toBeTruthy();
    expect(within(credentialsRegion).getByText('rtws_abcd1234')).toBeTruthy();
    expect(within(credentialsRegion).queryByText('rtws_public_selector_secret')).toBeNull();

    await user.click(within(credentialsRegion).getByRole('button', { name: /rotate credential/i }));
    expect(window.confirm).toHaveBeenCalled();
    expect(apiMock.rotateWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', 'cred-1');
    expect(await screen.findByDisplayValue('rtws_rotated_selector_secret')).toBeTruthy();

    await user.click(within(credentialsRegion).getByRole('button', { name: /revoke credential/i }));
    expect(apiMock.revokeWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', 'cred-1');
    expect(await within(credentialsRegion).findByText(/revoked/i)).toBeTruthy();

    const historyRegion = screen.getByRole('region', { name: /request history/i });
    expect(within(historyRegion).getByText('Accounting periods')).toBeTruthy();
    expect(within(historyRegion).getByText('/backend/periods')).toBeTruthy();
    expect(within(historyRegion).getByText('200')).toBeTruthy();
  });

  it('shows loading, empty, and unauthorized states', async () => {
    const manifestDeferred = new Promise(() => undefined);
    apiMock.getWorkspaceExternalApiManifest.mockImplementation(() => manifestDeferred);
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValue({
      preview_origin: 'https://ws-preview.example.com',
      items: [],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValue({ items: [] });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValue({
      cursor: null,
      limit: 20,
      items: [],
    });

    const { rerender } = render(
      <ExternalApiAccessSection
        workspaceId="ws-1"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    expect(screen.getByText(/loading external api access/i)).toBeTruthy();

    apiMock.getWorkspaceExternalApiManifest.mockRejectedValueOnce(new Error('Forbidden'));
    rerender(
      <ExternalApiAccessSection
        workspaceId="ws-2"
        previewOrigin="https://ws-preview.example.com"
      />,
    );
    expect((await screen.findByRole('alert')).textContent).toContain('Forbidden');

    apiMock.getWorkspaceExternalApiManifest.mockResolvedValueOnce({
      preview_origin: 'https://ws-preview.example.com',
      version: 1,
      valid: true,
      errors: [],
      candidates: [],
    });
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValueOnce({
      preview_origin: 'https://ws-preview.example.com',
      items: [],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValueOnce({ items: [] });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValueOnce({
      cursor: null,
      limit: 20,
      items: [],
    });

    rerender(
      <ExternalApiAccessSection
        workspaceId="ws-3"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/no published endpoints yet/i)).toBeTruthy();
      expect(screen.getByText(/no service credentials yet/i)).toBeTruthy();
      expect(screen.getByText(/no requests recorded yet/i)).toBeTruthy();
    });
  });
});
