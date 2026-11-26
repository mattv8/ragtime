import { afterEach, describe, expect, it, vi } from 'vitest';
import { SolidworksApiError, solidworksApi } from './solidworks';

const REQUEST = { pdm_tool_id: 'pdm-a', parent: '\\13_Ragtime', name: 'Sandbox' };

function respondWith(status: number, body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify(body), {
          status,
          headers: { 'Content-Type': 'application/json' },
        }),
    ),
  );
}

async function sandboxError(): Promise<SolidworksApiError> {
  const error = await solidworksApi.createPdmSandbox('sw-a', REQUEST).then(
    () => null,
    (caught: unknown) => caught,
  );
  expect(error).toBeInstanceOf(SolidworksApiError);
  return error as SolidworksApiError;
}

describe('solidworksApi.createPdmSandbox errors', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('posts to the sandbox route of the tool', async () => {
    respondWith(200, {
      folder: '\\13_Ragtime\\Sandbox',
      covered: false,
      approved_folders: [],
      allow_write: true,
      scripts_enabled: false,
    });
    await solidworksApi.createPdmSandbox('sw-a', REQUEST);
    expect(fetch).toHaveBeenCalledWith(
      '/indexes/solidworks/tools/sw-a/pdm/sandbox',
      expect.objectContaining({ method: 'POST', body: JSON.stringify(REQUEST) }),
    );
  });

  it('keeps the code and the message of a 409', async () => {
    respondWith(409, {
      detail: {
        code: 'sandbox_exists',
        message: 'A folder named “Sandbox” already exists in \\13_Ragtime. Choose another name.',
      },
    });
    const error = await sandboxError();
    expect(error.status).toBe(409);
    expect(error.code).toBe('sandbox_exists');
    expect(error.message).toBe(
      'A folder named “Sandbox” already exists in \\13_Ragtime. Choose another name.',
    );
  });

  it('keeps the missing rights from a rights refusal', async () => {
    respondWith(409, {
      detail: { code: 'pdm_rights_missing', message: 'denied', missing: ['read', 4] },
    });
    const error = await sandboxError();
    expect(error.code).toBe('pdm_rights_missing');
    expect(error.missing).toEqual(['read']);
  });

  it('uses the first validation message of a 422', async () => {
    respondWith(422, {
      detail: [{ loc: ['body', 'name'], msg: 'String should match pattern', type: 'x' }],
    });
    const error = await sandboxError();
    expect(error.status).toBe(422);
    expect(error.message).toBe('String should match pattern');
  });

  it('falls back to the status when a 422 has no readable message', async () => {
    respondWith(422, { detail: [] });
    const error = await sandboxError();
    expect(error.message).toBe('Request failed with status 422');
  });
});
