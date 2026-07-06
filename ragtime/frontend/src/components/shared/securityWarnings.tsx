import type { ReactNode } from 'react';

export const API_KEY_INFO_HIGHLIGHT = 'api_key_info';

export function renderApiKeySecurityWarning(): ReactNode {
  return (
    <>
      The API endpoint accepts an API Key for authentication (set via <code>API_KEY</code>{' '}
      environment variable). Without an API key, anyone with network access can use your LLM and
      tools.
    </>
  );
}

export function renderRuntimeAuthSecurityWarning(): ReactNode {
  return (
    <>
      Runtime auth is missing or using a legacy/default token. Set{' '}
      <code>RUNTIME_AUTH_TOKEN</code> to a strong random value (
      <code>openssl rand -base64 32</code>) in your <code>.env</code>, update your compose file to
      the latest template, and remove the deprecated <code>RUNTIME_MANAGER_AUTH_TOKEN</code> /{' '}
      <code>RUNTIME_WORKER_AUTH_TOKEN</code> variables.
    </>
  );
}

export function renderHttpSecurityWarning(includeAdditionallyPrefix = false): ReactNode {
  return (
    <>
      {includeAdditionallyPrefix ? 'Additionally, y' : 'Y'}ou are currently accessing over HTTP -
      API keys and credentials will be transmitted in plaintext. Consider using HTTPS via a reverse
      proxy or setting <code>ENABLE_HTTPS=true</code>.
    </>
  );
}
