export interface OAuthParams {
  client_id: string;
  redirect_uri: string;
  response_type: string;
  code_challenge: string;
  code_challenge_method: string;
  state: string;
  resource?: string;
  scope?: string;
}

export interface AuthorizeError {
  summary: string;
  nextSteps: string[];
}

/** Parse the safe error envelopes returned by both OAuth authorization routes. */
export function parseAuthorizeError(data: unknown): AuthorizeError {
  const fallback = 'The authorization request could not be completed.';
  if (!data || typeof data !== 'object') return { summary: fallback, nextSteps: [] };

  const envelope = data as Record<string, unknown>;
  const summary = [
    envelope.detail,
    envelope.message,
    envelope.error_description,
    envelope.error,
  ].find((value): value is string => typeof value === 'string' && value.trim().length > 0);
  const nextSteps = Array.isArray(envelope.next_steps)
    ? envelope.next_steps.filter((step): step is string => typeof step === 'string')
    : [];
  return { summary: summary ?? fallback, nextSteps };
}

export function buildAuthorizeForm(params: OAuthParams): URLSearchParams {
  const formData = new URLSearchParams();
  formData.append('client_id', params.client_id);
  formData.append('redirect_uri', params.redirect_uri);
  formData.append('response_type', params.response_type);
  formData.append('code_challenge', params.code_challenge);
  formData.append('code_challenge_method', params.code_challenge_method);
  formData.append('state', params.state);
  if (params.resource) formData.append('resource', params.resource);
  if (params.scope) formData.append('scope', params.scope);
  return formData;
}
