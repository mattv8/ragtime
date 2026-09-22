import type { GitWebhookConfig, GitWebhookEnableResponse } from '@/types';

/** Creates a fresh Git webhook configuration for component tests. */
export function createGitWebhookConfig(
  overrides: Partial<GitWebhookConfig> = {},
): GitWebhookConfig {
  return {
    enabled: false,
    paused: false,
    webhook_url: null,
    provider: 'github',
    branch: 'main',
    created_at: null,
    ...overrides,
  };
}

/** Creates a fresh successful Git webhook enable response for component tests. */
export function createGitWebhookEnableResponse(
  overrides: Partial<GitWebhookEnableResponse> = {},
): GitWebhookEnableResponse {
  const { secret = 'secret-once', ...configOverrides } = overrides;

  return {
    ...createGitWebhookConfig({ enabled: true, ...configOverrides }),
    secret,
  };
}
