import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentProtectionApiError, contentProtectionApi } from './contentProtection';

afterEach(() => vi.unstubAllGlobals());

describe('contentProtectionApi', () => {
  it('uses the canonical config URL and exposes safe structured error messages', async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          detail: {
            code: 'content_denied',
            message: 'Content is unavailable.',
            request_id: 'req-1',
          },
        }),
        {
          status: 403,
          headers: { 'Content-Type': 'application/json' },
        },
      ),
    );
    vi.stubGlobal('fetch', fetch);

    await expect(contentProtectionApi.getConfig()).rejects.toEqual(
      expect.objectContaining<Partial<ContentProtectionApiError>>({
        message: 'Content is unavailable.',
        status: 403,
      }),
    );
    expect(fetch).toHaveBeenCalledWith(
      '/indexes/content-protection/config',
      expect.objectContaining({ credentials: 'include' }),
    );
  });

  it('falls back to a stable status message when an error detail has no safe text', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: {} }), { status: 503 })),
    );
    await expect(contentProtectionApi.getCatalog()).rejects.toThrow(
      'Request failed with status 503',
    );
  });
});
