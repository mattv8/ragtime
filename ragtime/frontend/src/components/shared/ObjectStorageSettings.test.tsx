import { act, cleanup, render, screen } from '@testing-library/react';
import { createRef } from 'react';
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
import { ObjectStorageSettings, type ObjectStorageSettingsHandle } from './ObjectStorageSettings';

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
  it('renders only the secondary action for the selected storage mode', async () => {
    apiMock.getObjectStorageAdminSettings.mockResolvedValue(settings);
    render(<ObjectStorageSettings />);

    await screen.findByLabelText('Access key (configured)');
    expect(screen.getByRole('button', { name: 'Test connection' })).toBeTruthy();
    expect(screen.queryByRole('button', { name: 'Refresh status' })).toBeNull();
    expect(screen.queryByRole('button', { name: 'Save storage settings' })).toBeNull();

    await userEvent.click(screen.getByLabelText('Local (default)'));
    expect(screen.queryByRole('button', { name: 'Test connection' })).toBeNull();
    expect(screen.getByRole('button', { name: 'Refresh status' })).toBeTruthy();
  });

  it('keeps configured credentials empty and saves migration through its handle', async () => {
    apiMock.getObjectStorageAdminSettings.mockResolvedValue(settings);
    apiMock.updateObjectStorageAdminSettings.mockResolvedValue(settings);
    apiMock.startObjectStorageMigration.mockResolvedValue({ jobs: [] });
    const ref = createRef<ObjectStorageSettingsHandle>();
    render(<ObjectStorageSettings ref={ref} />);

    await screen.findByLabelText('Access key (configured)');
    expect((screen.getByLabelText('Access key (configured)') as HTMLInputElement).value).toBe('');
    await userEvent.click(screen.getByLabelText('Migrate existing workspaces after saving'));
    await act(async () => ref.current?.save());

    expect(apiMock.updateObjectStorageAdminSettings).toHaveBeenCalledOnce();
    expect(apiMock.startObjectStorageMigration).toHaveBeenCalledOnce();
  });

  it('shows and propagates storage save failures', async () => {
    apiMock.getObjectStorageAdminSettings.mockResolvedValue(settings);
    apiMock.updateObjectStorageAdminSettings.mockRejectedValue(new Error('Storage save failed'));
    const ref = createRef<ObjectStorageSettingsHandle>();
    render(<ObjectStorageSettings ref={ref} />);

    await screen.findByLabelText('Access key (configured)');
    await expect(ref.current?.save()).rejects.toThrow('Storage save failed');
    expect((await screen.findByRole('status')).textContent).toContain('Storage save failed');
  });

  it('rejects saving while initial storage settings are loading', async () => {
    let resolveSettings: (value: typeof settings) => void;
    const settingsPromise = new Promise<typeof settings>((resolve) => {
      resolveSettings = resolve;
    });
    apiMock.getObjectStorageAdminSettings.mockReturnValue(settingsPromise);
    const ref = createRef<ObjectStorageSettingsHandle>();
    render(<ObjectStorageSettings ref={ref} />);

    await expect(ref.current?.save()).rejects.toThrow('Object storage settings are still loading');
    await act(async () => resolveSettings(settings));
  });

  it('rejects saving while another object storage operation is active', async () => {
    let resolveTest: () => void;
    const testPromise = new Promise<void>((resolve) => {
      resolveTest = resolve;
    });
    apiMock.getObjectStorageAdminSettings.mockResolvedValue(settings);
    apiMock.testObjectStorageAdminSettings.mockReturnValue(testPromise);
    const ref = createRef<ObjectStorageSettingsHandle>();
    render(<ObjectStorageSettings ref={ref} />);

    await screen.findByLabelText('Access key (configured)');
    screen.getByRole('button', { name: 'Test connection' }).click();
    await expect(ref.current?.save()).rejects.toThrow(
      'Another object storage operation is in progress',
    );
    expect(apiMock.updateObjectStorageAdminSettings).not.toHaveBeenCalled();
    await act(async () => resolveTest());
  });
});
