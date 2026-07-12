import type { CloudOAuthProviderStatus } from '@/types';

export interface CloudOAuthSetupProviderHelp {
  provider: CloudOAuthProviderStatus['provider'];
  label: string;
  envVars: string[];
}

const CLOUD_OAUTH_SETUP_HELP: CloudOAuthSetupProviderHelp[] = [
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
];

export function getUnconfiguredCloudOAuthProviders(
  providerStatuses: CloudOAuthProviderStatus[],
): CloudOAuthSetupProviderHelp[] {
  return CLOUD_OAUTH_SETUP_HELP.filter((providerHelp) => {
    const providerStatus = providerStatuses.find(
      (status) => status.provider === providerHelp.provider,
    );
    return providerStatus?.configured === false;
  });
}
