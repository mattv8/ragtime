import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

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
  deleteWorkspaceExternalApiCredential: vi.fn(),
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

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
  vi.restoreAllMocks();
});

describe('ExternalApiAccessSection', () => {
  it('renders the combined endpoints pane with endpoint-level credential selection and preserves values across hide/show', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceExternalApiManifest.mockResolvedValue(manifestResponse());
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValue({
      preview_origin: 'https://ws-preview.example.com',
      items: [publishedEndpoint()],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValue({ items: [] });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValue({
      cursor: null,
      limit: 20,
      items: [],
    });

    render(
      <ExternalApiAccessSection
        workspaceId="ws-1"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    await screen.findByText('Accounting periods');
    const endpointsSection = document.querySelector('#workspace-external-api-endpoints');
    expect(endpointsSection).toBeTruthy();
    expect(document.querySelectorAll('#workspace-external-api-endpoints')).toHaveLength(1);
    expect(document.querySelector('#workspace-external-api-create-credential')).toBeNull();

    const endpointArticle = endpointsSection?.querySelector(
      '[data-endpoint-key="accounting-periods"]',
    ) as HTMLElement | null;
    expect(endpointArticle).toBeTruthy();
    expect(endpointArticle?.classList.contains('is-credential-selected')).toBe(false);

    const endpointCheckbox = await screen.findByRole('checkbox', {
      name: 'Use Accounting periods for credential',
    });
    expect(document.querySelector('#workspace-external-api-credential-details')).toBeNull();
    expect(screen.queryByLabelText(/credential label/i)).toBeNull();
    expect(screen.queryByLabelText(/expiry \(optional\)/i)).toBeNull();
    expect(screen.queryByRole('button', { name: /create credential/i })).toBeNull();

    await user.click(endpointCheckbox);
    expect(document.querySelector('#workspace-external-api-credential-details')).toBeTruthy();
    expect(endpointArticle?.classList.contains('is-credential-selected')).toBe(true);
    expect(
      endpointCheckbox.closest('label')?.classList.contains('userspace-external-api-checkbox-row'),
    ).toBe(true);
    expect(
      endpointCheckbox
        .closest('label')
        ?.classList.contains('userspace-external-api-endpoint-selection-control'),
    ).toBe(true);
    expect(
      within(endpointsSection as HTMLElement).getByRole('heading', { name: 'Create credential' }),
    ).toBeTruthy();

    const credentialLabelInput = screen.getByLabelText(/credential label/i);
    const expiryInput = screen.getByLabelText(/expiry \(optional\)/i);
    expect(screen.getByRole('button', { name: /create credential/i })).toBeTruthy();

    await user.type(credentialLabelInput, 'Quarter close');
    await user.type(expiryInput, '2026-09-30T12:00');

    await user.click(endpointCheckbox);
    expect(document.querySelector('#workspace-external-api-credential-details')).toBeNull();
    expect(endpointArticle?.classList.contains('is-credential-selected')).toBe(false);
    expect(screen.queryByLabelText(/credential label/i)).toBeNull();
    expect(screen.queryByLabelText(/expiry \(optional\)/i)).toBeNull();
    expect(screen.queryByRole('button', { name: /create credential/i })).toBeNull();

    await user.click(endpointCheckbox);
    expect(document.querySelector('#workspace-external-api-credential-details')).toBeTruthy();
    expect((screen.getByLabelText(/credential label/i) as HTMLInputElement).value).toBe(
      'Quarter close',
    );
    expect((screen.getByLabelText(/expiry \(optional\)/i) as HTMLInputElement).value).toBe(
      '2026-09-30T12:00',
    );
  });

  it('loads candidates, shows stale publication state, creates a credential, and opens the token dialog in a portal', async () => {
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
    expect(
      screen.queryByRole('checkbox', { name: 'Use Accounting periods for credential' }),
    ).toBeNull();

    await user.click(screen.getByRole('button', { name: /reapprove/i }));
    expect(apiMock.createWorkspaceExternalApiEndpoint).toHaveBeenCalledWith(
      'ws-1',
      'accounting-periods',
    );
    expect(
      await screen.findByRole('checkbox', { name: 'Use Accounting periods for credential' }),
    ).toBeTruthy();

    await user.click(
      screen.getByRole('checkbox', { name: 'Use Accounting periods for credential' }),
    );
    await user.type(screen.getByLabelText(/credential label/i), 'August workpapers');
    await user.click(screen.getByRole('button', { name: /create credential/i }));

    expect(apiMock.createWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', {
      label: 'August workpapers',
      endpoint_keys: ['accounting-periods'],
      expires_at: null,
    });

    const dialog = await screen.findByRole('dialog', { name: /created credential/i });
    expect(document.body.querySelector('#userspace-external-api-token-dialog')).toBe(dialog);
    expect(screen.getByText('Copy this token now. It cannot be shown again.')).toBeTruthy();
    expect(screen.getByText(/Authorization: Bearer rtws_public_selector_secret/)).toBeTruthy();
    await user.click(screen.getByRole('tab', { name: 'Power Query' }));
    expect(screen.getByRole('tabpanel', { name: /power query/i }).textContent).toContain(
      'Web.Contents',
    );
    await user.click(screen.getByRole('button', { name: 'I saved this token' }));
    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: /created credential/i })).toBeNull();
    });
    expect(screen.queryByText('rtws_abcd1234')).toBeNull();
  });

  it('opens rotate and revoke dialogs without window.confirm, reveals rotated secrets, and loads request history', async () => {
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
    expect(screen.getByRole('dialog', { name: /rotate credential/i })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Rotate token' }));
    expect(apiMock.rotateWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', 'cred-1');
    expect(await screen.findByRole('dialog', { name: /rotated credential/i })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'I saved this token' }));

    await user.click(within(credentialsRegion).getByRole('button', { name: /revoke credential/i }));
    expect(screen.getByRole('dialog', { name: /revoke credential/i })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Revoke credential' }));
    expect(apiMock.revokeWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', 'cred-1');
    expect(await within(credentialsRegion).findByText(/revoked/i)).toBeTruthy();
    expect(
      within(credentialsRegion).queryByRole('button', { name: /rotate credential/i }),
    ).toBeNull();
    expect(
      within(credentialsRegion).queryByRole('button', { name: /revoke credential/i }),
    ).toBeNull();
    expect(
      within(credentialsRegion).getByRole('button', { name: /delete credential/i }),
    ).toBeTruthy();

    const historyRegion = screen.getByRole('region', { name: /request history/i });
    expect(within(historyRegion).getByText('Accounting periods')).toBeTruthy();
    expect(within(historyRegion).getByText('/backend/periods')).toBeTruthy();
    expect(within(historyRegion).getByText('200')).toBeTruthy();
  });

  it('renders service credential cards with status in the header and actions in a separate bottom container', async () => {
    apiMock.getWorkspaceExternalApiManifest.mockResolvedValue(manifestResponse());
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValue({
      preview_origin: 'https://ws-preview.example.com',
      items: [publishedEndpoint()],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValue({
      items: [
        credentialItem(),
        credentialItem({
          id: 'cred-2',
          label: 'Revoked credential',
          token_prefix: 'rtws_revoked',
          enabled: false,
          revoked_at: '2026-09-02T00:00:00Z',
        }),
      ],
    });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValue({
      cursor: null,
      limit: 20,
      items: [requestHistoryItem()],
    });

    const { container } = render(
      <ExternalApiAccessSection
        workspaceId="ws-1"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    const credentialsRegion = await screen.findByRole('region', { name: /service credentials/i });
    const credentialList = credentialsRegion.querySelector(
      '.userspace-external-api-credential-list',
    );
    expect(credentialList).toBeTruthy();
    expect(credentialList?.classList.contains('userspace-external-api-list')).toBe(true);

    const activeCredential = credentialsRegion.querySelector('[data-credential-id="cred-1"]');
    expect(activeCredential).toBeTruthy();
    expect(activeCredential?.classList.contains('userspace-external-api-credential-row')).toBe(
      true,
    );

    const activeHeader = activeCredential?.querySelector('.userspace-external-api-row-header');
    const activeStatus = activeHeader?.querySelector('.userspace-external-api-status');
    expect(activeStatus?.textContent).toContain('Enabled');
    expect(activeHeader?.lastElementChild).toBe(activeStatus);

    const activeActions = activeCredential?.querySelector(
      '.userspace-external-api-row-actions.userspace-external-api-credential-actions',
    );
    expect(activeActions).toBeTruthy();
    expect(activeActions?.querySelector('.userspace-external-api-status')).toBeNull();
    expect(
      within(activeActions as HTMLElement).getByRole('button', { name: /rotate credential/i }),
    ).toBeTruthy();
    expect(
      within(activeActions as HTMLElement).getByRole('button', { name: /revoke credential/i }),
    ).toBeTruthy();
    expect(
      within(activeActions as HTMLElement).queryByRole('button', { name: /delete credential/i }),
    ).toBeNull();

    const revokedCredential = credentialsRegion.querySelector('[data-credential-id="cred-2"]');
    expect(revokedCredential).toBeTruthy();
    const revokedActions = revokedCredential?.querySelector(
      '.userspace-external-api-row-actions.userspace-external-api-credential-actions',
    );
    expect(revokedActions).toBeTruthy();
    const revokedStatus = revokedCredential
      ?.querySelector('.userspace-external-api-row-header')
      ?.querySelector('.userspace-external-api-status');
    expect(revokedStatus?.textContent).toContain('Revoked');
    expect(revokedActions?.querySelector('.userspace-external-api-status')).toBeNull();
    expect(
      within(revokedActions as HTMLElement).getByRole('button', { name: /delete credential/i }),
    ).toBeTruthy();
    expect(
      within(revokedActions as HTMLElement).queryByRole('button', { name: /rotate credential/i }),
    ).toBeNull();
    expect(
      within(revokedActions as HTMLElement).queryByRole('button', { name: /revoke credential/i }),
    ).toBeNull();

    const endpointArticle = container.querySelector(
      '#workspace-external-api-endpoints .userspace-external-api-row',
    );
    expect(endpointArticle?.classList.contains('userspace-external-api-credential-row')).toBe(
      false,
    );
    const endpointHeader = endpointArticle?.querySelector('.userspace-external-api-row-header');
    expect(endpointHeader?.querySelector('.userspace-external-api-status')?.textContent).toContain(
      'Published',
    );
    expect(
      endpointArticle?.querySelector(
        '.userspace-external-api-row-actions .userspace-external-api-status',
      ),
    ).toBeNull();
  });

  it('deletes revoked credentials, refreshes history, and surfaces mutation failures after closing the confirm dialog', async () => {
    const user = userEvent.setup();
    apiMock.getWorkspaceExternalApiManifest.mockResolvedValue(manifestResponse());
    apiMock.listWorkspaceExternalApiEndpoints.mockResolvedValue({
      preview_origin: 'https://ws-preview.example.com',
      items: [publishedEndpoint()],
    });
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValueOnce({
      items: [credentialItem({ enabled: false, revoked_at: '2026-09-02T00:00:00Z' })],
    });
    apiMock.listWorkspaceExternalApiRequests
      .mockResolvedValueOnce({
        cursor: null,
        limit: 20,
        items: [requestHistoryItem()],
      })
      .mockResolvedValueOnce({
        cursor: null,
        limit: 20,
        items: [requestHistoryItem({ id: 'req-2', credential_label: null })],
      });
    apiMock.deleteWorkspaceExternalApiCredential.mockResolvedValue(undefined);

    render(
      <ExternalApiAccessSection
        workspaceId="ws-1"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    const credentialsRegion = await screen.findByRole('region', { name: /service credentials/i });
    await user.click(within(credentialsRegion).getByRole('button', { name: /delete credential/i }));
    expect(screen.getByRole('dialog', { name: /delete credential/i })).toBeTruthy();
    await user.click(screen.getByRole('button', { name: 'Delete permanently' }));

    await waitFor(() => {
      expect(apiMock.deleteWorkspaceExternalApiCredential).toHaveBeenCalledWith('ws-1', 'cred-1');
    });
    await waitFor(() => {
      expect(within(credentialsRegion).queryByText('August workpapers')).toBeNull();
    });
    const historyRegion = screen.getByRole('region', { name: /request history/i });
    expect(await within(historyRegion).findByText('Unknown credential')).toBeTruthy();

    cleanup();
    apiMock.listWorkspaceExternalApiCredentials.mockResolvedValue({ items: [credentialItem()] });
    apiMock.listWorkspaceExternalApiRequests.mockResolvedValue({
      cursor: null,
      limit: 20,
      items: [requestHistoryItem()],
    });
    apiMock.deleteWorkspaceExternalApiCredential.mockRejectedValueOnce(new Error('Delete failed'));

    render(
      <ExternalApiAccessSection
        workspaceId="ws-2"
        previewOrigin="https://ws-preview.example.com"
      />,
    );

    const secondCredentialsRegion = await screen.findByRole('region', {
      name: /service credentials/i,
    });
    apiMock.revokeWorkspaceExternalApiCredential.mockRejectedValueOnce(new Error('Revoke failed'));
    await user.click(
      within(secondCredentialsRegion).getByRole('button', { name: /revoke credential/i }),
    );
    await user.click(screen.getByRole('button', { name: 'Revoke credential' }));
    await waitFor(() => {
      expect(screen.queryByRole('dialog', { name: /revoke credential/i })).toBeNull();
    });
    expect((await screen.findByRole('alert')).textContent).toContain('Revoke failed');
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
