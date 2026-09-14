import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

const apiMock = vi.hoisted(() => ({
  getObjectStorageAdminSettings: vi.fn(),
  updateObjectStorageAdminSettings: vi.fn(),
  testObjectStorageAdminSettings: vi.fn(),
  startObjectStorageMigration: vi.fn(),
  retryObjectStorageMigration: vi.fn(),
}));
vi.mock('@/api/client', () => ({ api: apiMock }));
import { ObjectStorageSettings } from './ObjectStorageSettings';

const settings = {
  mode: 'external' as const,
  default_backend_id: 'external-1',
  endpoint: 'http://s3',
  region: 'us-east-1',
  bucket: 'ragtime',
  access_key_configured: true,
  secret_key_configured: true,
  existing_local_workspaces: 2,
  migrations: [],
};

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});
describe('ObjectStorageSettings', () => {
  it('keeps configured credentials empty and saves migration only after settings save', async () => {
    apiMock.getObjectStorageAdminSettings.mockResolvedValue(settings);
    apiMock.updateObjectStorageAdminSettings.mockResolvedValue(settings);
    apiMock.startObjectStorageMigration.mockResolvedValue({ jobs: [] });
    render(<ObjectStorageSettings />);
    await screen.findByLabelText('Access key (configured)');
    expect((screen.getByLabelText('Access key (configured)') as HTMLInputElement).value).toBe('');
    await userEvent.click(screen.getByLabelText('Migrate existing workspaces after saving'));
    await userEvent.click(screen.getByRole('button', { name: 'Save storage settings' }));
    await waitFor(() => expect(apiMock.updateObjectStorageAdminSettings).toHaveBeenCalledOnce());
    expect(apiMock.startObjectStorageMigration).toHaveBeenCalledOnce();
  });
});
