import type { ReactNode } from 'react';

export const API_KEY_INFO_HIGHLIGHT = 'api_key_info';

export function renderApiKeySecurityWarning(): ReactNode {
  return (
    <>
      The API endpoint accepts an API Key for authentication (set via <code>API_KEY</code> environment variable).
      Without an API key, anyone with network access can use your LLM and tools.
    </>
  );
}

export function renderHttpSecurityWarning(includeAdditionallyPrefix = false): ReactNode {
  return (
    <>
      {includeAdditionallyPrefix ? 'Additionally, y' : 'Y'}ou are currently accessing over HTTP - API keys and credentials will be transmitted in plaintext.
      Consider using HTTPS via a reverse proxy or setting <code>ENABLE_HTTPS=true</code>.
    </>
  );
}
