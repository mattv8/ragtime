import { describe, expect, it } from 'vitest';
import type { CloudOAuthProviderStatus } from '@/types';
import { getUnconfiguredCloudOAuthProviders } from './cloudOAuthSetupHelp';

function providerStatus(
  provider: CloudOAuthProviderStatus['provider'],
  configured: boolean,
): CloudOAuthProviderStatus {
  return {
    provider,
    configured,
    auth_url_available: configured,
  };
}

describe('getUnconfiguredCloudOAuthProviders', () => {
  it('returns no setup guidance when all cloud OAuth providers are configured', () => {
    const missingProviders = getUnconfiguredCloudOAuthProviders([
      providerStatus('microsoft_drive', true),
      providerStatus('google_drive', true),
    ]);

    expect(missingProviders).toEqual([]);
  });

  it('returns provider-specific env var guidance for unconfigured providers', () => {
    const missingProviders = getUnconfiguredCloudOAuthProviders([
      providerStatus('microsoft_drive', false),
      providerStatus('google_drive', false),
    ]);

    expect(missingProviders).toEqual([
      {
        provider: 'microsoft_drive',
        label: 'OneDrive/SharePoint',
        envVars: [
          'CLOUD_MOUNT_MICROSOFT_CLIENT_ID',
          'CLOUD_MOUNT_MICROSOFT_CLIENT_SECRET',
          'CLOUD_MOUNT_MICROSOFT_TENANT_ID',
        ],
      },
      {
        provider: 'google_drive',
        label: 'Google Drive',
        envVars: ['CLOUD_MOUNT_GOOGLE_CLIENT_ID', 'CLOUD_MOUNT_GOOGLE_CLIENT_SECRET'],
      },
    ]);
  });
});
