import { describe, expect, it } from 'vitest';

import { buildAuthorizeForm, parseAuthorizeError, type OAuthParams } from './oauthAuthorization';

describe('buildAuthorizeForm', () => {
  it('serializes OAuth and optional PKCE request parameters without defaults', () => {
    const params: OAuthParams = {
      client_id: 'client',
      redirect_uri: 'https://example.test/callback',
      response_type: 'code',
      code_challenge: 'challenge',
      code_challenge_method: 'S256',
      state: 'state',
      resource: 'https://example.test/mcp',
      scope: 'tools.read tools.search',
    };

    expect(buildAuthorizeForm(params).toString()).toBe(
      'client_id=client&redirect_uri=https%3A%2F%2Fexample.test%2Fcallback&response_type=code&code_challenge=challenge&code_challenge_method=S256&state=state&resource=https%3A%2F%2Fexample.test%2Fmcp&scope=tools.read+tools.search',
    );
  });

  it('omits absent optional fields', () => {
    const params: OAuthParams = {
      client_id: 'client',
      redirect_uri: 'https://example.test/callback',
      response_type: 'code',
      code_challenge: 'challenge',
      code_challenge_method: 'S256',
      state: 'state',
    };

    expect(buildAuthorizeForm(params).has('resource')).toBe(false);
    expect(buildAuthorizeForm(params).has('scope')).toBe(false);
  });
});

describe('parseAuthorizeError', () => {
  it('preserves the backend error and next_steps envelope', () => {
    expect(
      parseAuthorizeError({
        error: 'access_denied',
        next_steps: ['Sign in again', 'Ask an administrator for access'],
      }),
    ).toEqual({
      summary: 'access_denied',
      nextSteps: ['Sign in again', 'Ask an administrator for access'],
    });
  });

  it('accepts error_description and ignores non-string error content', () => {
    expect(
      parseAuthorizeError({
        error_description: 'The client is not permitted.',
        next_steps: [1, 'Retry'],
      }),
    ).toEqual({ summary: 'The client is not permitted.', nextSteps: ['Retry'] });
    expect(parseAuthorizeError({ error: { message: 'unsafe' }, next_steps: 'Retry' })).toEqual({
      summary: 'The authorization request could not be completed.',
      nextSteps: [],
    });
  });
});
