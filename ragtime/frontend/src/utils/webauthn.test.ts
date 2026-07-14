import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  base64UrlToBuffer,
  bufferToBase64Url,
  createPasskeyCredential,
  getPasskeyAssertion,
  isWebAuthnSupported,
  WebAuthnCancelledError,
} from './webauthn';

const createPasskeyOptions = {
  challenge: 'Y2hhbGxlbmdl',
  rp: { name: 'Ragtime' },
  user: { id: 'dXNlcg', name: 'u', displayName: 'u' },
  pubKeyCredParams: [{ alg: -7, type: 'public-key' as const }],
};

const getPasskeyOptions = {
  challenge: 'Y2hhbGxlbmdl',
  allowCredentials: [{ id: 'Y3JlZA', type: 'public-key' as const }],
};

const webAuthnOriginals = {
  navigator: globalThis.navigator,
  PublicKeyCredential: globalThis.PublicKeyCredential,
};

function installWebAuthnMocks() {
  const credentials = {
    create: vi.fn(),
    get: vi.fn(),
  };

  Object.defineProperty(globalThis, 'navigator', {
    value: { credentials },
    configurable: true,
  });
  Object.defineProperty(globalThis, 'PublicKeyCredential', {
    value: function () {
      // no-op constructor
    },
    configurable: true,
  });

  return credentials;
}

function restoreWebAuthnMocks() {
  Object.defineProperty(globalThis, 'navigator', {
    value: webAuthnOriginals.navigator,
    configurable: true,
  });
  Object.defineProperty(globalThis, 'PublicKeyCredential', {
    value: webAuthnOriginals.PublicKeyCredential,
    configurable: true,
  });
  vi.restoreAllMocks();
}

describe('isWebAuthnSupported', () => {
  it('returns false in the jsdom test environment', () => {
    expect(isWebAuthnSupported()).toBe(false);
  });
});

describe('base64url encoding helpers', () => {
  it('round-trips arbitrary buffers to base64url', () => {
    const inputs = [
      new Uint8Array([]).buffer,
      new Uint8Array([0]).buffer,
      new Uint8Array([255, 254, 253, 252]).buffer,
      crypto.getRandomValues(new Uint8Array(64)).buffer,
    ];

    for (const buffer of inputs) {
      expect(base64UrlToBuffer(bufferToBase64Url(buffer))).toEqual(buffer);
    }
  });

  it('round-trips base64url values with padding stripped', () => {
    const values = [
      'dGVzdA', // "test" without padding
      'Zg', // "f" without padding
      'Zm9vYmFy', // "foobar" without padding
    ];

    for (const value of values) {
      const buffer = base64UrlToBuffer(value);
      expect(bufferToBase64Url(buffer)).toBe(value);
    }
  });

  it('correctly round-trips base64url-specific characters (- and _)', () => {
    const base64url = '-_8';
    const buffer = base64UrlToBuffer(base64url);
    expect(bufferToBase64Url(buffer)).toBe(base64url);
  });

  it('handles base64url values that already include padding', () => {
    const value = 'dGVzdA==';
    const buffer = base64UrlToBuffer(value);
    expect(new TextDecoder().decode(buffer)).toBe('test');
  });

  it('round-trips a value with the full byte range', () => {
    const original = new Uint8Array(Array.from({ length: 256 }, (_, i) => i)).buffer;
    const encoded = bufferToBase64Url(original);
    expect(encoded).not.toContain('+');
    expect(encoded).not.toContain('/');
    expect(encoded).not.toContain('=');
    expect(base64UrlToBuffer(encoded)).toEqual(original);
  });
});

describe('createPasskeyCredential cancellation mapping', () => {
  let credentials: ReturnType<typeof installWebAuthnMocks>;

  beforeEach(() => {
    // jsdom does not expose navigator.credentials.create; install a mock.
    credentials = installWebAuthnMocks();
  });

  afterEach(() => {
    restoreWebAuthnMocks();
  });

  it('throws WebAuthnCancelledError when create() rejects with NotAllowedError', async () => {
    credentials.create.mockRejectedValue(new DOMException('User cancelled', 'NotAllowedError'));

    await expect(createPasskeyCredential(createPasskeyOptions)).rejects.toBeInstanceOf(
      WebAuthnCancelledError,
    );
  });

  it('throws WebAuthnCancelledError when native create() rejects with NotAllowedError', async () => {
    Object.defineProperty(globalThis.PublicKeyCredential, 'parseCreationOptionsFromJSON', {
      value: vi.fn(() => ({
        challenge: new TextEncoder().encode('challenge').buffer,
        rp: { name: 'Ragtime' },
        user: { id: new TextEncoder().encode('user').buffer, name: 'u', displayName: 'u' },
        pubKeyCredParams: [{ alg: -7, type: 'public-key' }],
      })),
      configurable: true,
    });
    credentials.create.mockRejectedValue(new DOMException('User cancelled', 'NotAllowedError'));

    await expect(createPasskeyCredential(createPasskeyOptions)).rejects.toBeInstanceOf(
      WebAuthnCancelledError,
    );
  });

  it('throws WebAuthnCancelledError when create() rejects with AbortError', async () => {
    credentials.create.mockRejectedValue(new DOMException('Operation aborted', 'AbortError'));

    await expect(createPasskeyCredential(createPasskeyOptions)).rejects.toBeInstanceOf(
      WebAuthnCancelledError,
    );
  });

  it('re-throws non-cancellation errors unchanged', async () => {
    const otherError = new Error('Some other failure');
    credentials.create.mockRejectedValue(otherError);

    await expect(createPasskeyCredential(createPasskeyOptions)).rejects.toBe(otherError);
  });

  it('returns a JSON credential from create() when native parse helpers are unavailable', async () => {
    const rawId = new Uint8Array([1, 2, 3, 4]).buffer;
    const clientDataJSON = new TextEncoder().encode('{"a":1}').buffer;
    const attestationObject = new Uint8Array([5, 6, 7, 8]).buffer;

    credentials.create.mockResolvedValue({
      id: 'AQIDBA',
      rawId,
      type: 'public-key',
      authenticatorAttachment: 'cross-platform',
      response: {
        clientDataJSON,
        attestationObject,
        getTransports: () => ['usb', 'nfc'],
      },
      getClientExtensionResults: () => ({ credProps: { rk: true } }),
    } as unknown as Credential);

    const result = await createPasskeyCredential({
      ...createPasskeyOptions,
      excludeCredentials: [{ id: 'aWQ', type: 'public-key' }],
    });

    expect(result).toEqual({
      id: 'AQIDBA',
      rawId: 'AQIDBA',
      response: {
        clientDataJSON: 'eyJhIjoxfQ',
        attestationObject: 'BQYHCA',
        transports: ['usb', 'nfc'],
      },
      type: 'public-key',
      clientExtensionResults: { credProps: { rk: true } },
      authenticatorAttachment: 'cross-platform',
    });

    const createCall = credentials.create.mock.calls[0][0] as {
      publicKey: PublicKeyCredentialCreationOptions;
    };
    expect(Array.from(new Uint8Array(createCall.publicKey.challenge as ArrayBuffer))).toEqual(
      Array.from(new TextEncoder().encode('challenge')),
    );
    expect(Array.from(new Uint8Array(createCall.publicKey.user.id as ArrayBuffer))).toEqual(
      Array.from(new TextEncoder().encode('user')),
    );
    expect(createCall.publicKey.excludeCredentials).toHaveLength(1);
    expect(
      Array.from(new Uint8Array(createCall.publicKey.excludeCredentials![0].id as ArrayBuffer)),
    ).toEqual(Array.from(new TextEncoder().encode('id')));
  });
});

describe('getPasskeyAssertion cancellation mapping', () => {
  let credentials: ReturnType<typeof installWebAuthnMocks>;

  beforeEach(() => {
    credentials = installWebAuthnMocks();
  });

  afterEach(() => {
    restoreWebAuthnMocks();
  });

  it('throws WebAuthnCancelledError when get() rejects with NotAllowedError', async () => {
    credentials.get.mockRejectedValue(new DOMException('User cancelled', 'NotAllowedError'));

    await expect(getPasskeyAssertion(getPasskeyOptions)).rejects.toBeInstanceOf(
      WebAuthnCancelledError,
    );
  });

  it('throws WebAuthnCancelledError when native get() rejects with NotAllowedError', async () => {
    Object.defineProperty(globalThis.PublicKeyCredential, 'parseRequestOptionsFromJSON', {
      value: vi.fn(() => ({
        challenge: new TextEncoder().encode('challenge').buffer,
        allowCredentials: [{ id: new TextEncoder().encode('cred').buffer, type: 'public-key' }],
      })),
      configurable: true,
    });
    credentials.get.mockRejectedValue(new DOMException('User cancelled', 'NotAllowedError'));

    await expect(getPasskeyAssertion(getPasskeyOptions)).rejects.toBeInstanceOf(
      WebAuthnCancelledError,
    );
  });

  it('returns a JSON assertion from get() when native parse helpers are unavailable', async () => {
    const rawId = new Uint8Array([9, 10, 11, 12]).buffer;
    const clientDataJSON = new TextEncoder().encode('{"type":"get"}').buffer;
    const authenticatorData = new Uint8Array([13, 14, 15, 16]).buffer;
    const signature = new Uint8Array([17, 18, 19, 20]).buffer;
    const userHandle = new TextEncoder().encode('user').buffer;

    credentials.get.mockResolvedValue({
      id: 'CQoLDA',
      rawId,
      type: 'public-key',
      authenticatorAttachment: 'platform',
      response: {
        clientDataJSON,
        authenticatorData,
        signature,
        userHandle,
      },
      getClientExtensionResults: () => ({}),
    } as unknown as Credential);

    const result = await getPasskeyAssertion(getPasskeyOptions);

    expect(result).toEqual({
      id: 'CQoLDA',
      rawId: 'CQoLDA',
      response: {
        clientDataJSON: 'eyJ0eXBlIjoiZ2V0In0',
        authenticatorData: 'DQ4PEA',
        signature: 'ERITFA',
        userHandle: 'dXNlcg',
      },
      type: 'public-key',
      clientExtensionResults: {},
      authenticatorAttachment: 'platform',
    });

    const getCall = credentials.get.mock.calls[0][0] as {
      publicKey: PublicKeyCredentialRequestOptions;
    };
    expect(Array.from(new Uint8Array(getCall.publicKey.challenge as ArrayBuffer))).toEqual(
      Array.from(new TextEncoder().encode('challenge')),
    );
    expect(getCall.publicKey.allowCredentials).toHaveLength(1);
    expect(
      Array.from(new Uint8Array(getCall.publicKey.allowCredentials![0].id as ArrayBuffer)),
    ).toEqual(Array.from(new TextEncoder().encode('cred')));
  });
});
