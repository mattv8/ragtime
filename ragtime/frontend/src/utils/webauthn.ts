/**
 * WebAuthn helpers for passkey 2FA.
 *
 * Converts between server-side WebAuthn JSON (base64url fields as used by
 * py_webauthn) and the browser Credential Management API binary fields.
 */

export class WebAuthnCancelledError extends Error {
  constructor(message = 'Passkey ceremony was cancelled or timed out') {
    super(message);
    this.name = 'WebAuthnCancelledError';
  }
}

/**
 * Returns true when the runtime supports WebAuthn credential creation.
 */
export function isWebAuthnSupported(): boolean {
  return (
    typeof window !== 'undefined' &&
    typeof window.PublicKeyCredential !== 'undefined' &&
    typeof navigator !== 'undefined' &&
    typeof navigator.credentials !== 'undefined' &&
    typeof navigator.credentials.create === 'function' &&
    typeof navigator.credentials.get === 'function'
  );
}

/**
 * Decode a base64url string to an ArrayBuffer. Handles missing padding and the
 * base64url character substitutions (`-` -> `+`, `_` -> `/`).
 */
export function base64UrlToBuffer(value: string): ArrayBuffer {
  const padded = value.padEnd(value.length + ((4 - (value.length % 4)) % 4), '=');
  const base64 = padded.replace(/-/g, '+').replace(/_/g, '/');
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

/**
 * Encode an ArrayBuffer to a base64url string without padding.
 */
export function bufferToBase64Url(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=/g, '');
}

function normalizeBase64UrlField(value: unknown): string {
  if (typeof value !== 'string') {
    return '';
  }
  return value;
}

function mapBase64UrlList(
  items: Array<Record<string, unknown>> | undefined,
  key: string,
): PublicKeyCredentialDescriptor[] | undefined {
  if (!Array.isArray(items)) {
    return undefined;
  }
  return items
    .map((item) => {
      const id = normalizeBase64UrlField(item[key]);
      if (!id) {
        return null;
      }
      return {
        id: base64UrlToBuffer(id),
        type: typeof item.type === 'string' ? item.type : 'public-key',
      } as PublicKeyCredentialDescriptor;
    })
    .filter((item): item is PublicKeyCredentialDescriptor => item !== null);
}

/**
 * Convert a server-supplied PublicKeyCredentialCreationOptionsJSON object into
 * the binary form required by navigator.credentials.create().
 */
function decodeCreationOptions(
  optionsJson: Record<string, unknown>,
): PublicKeyCredentialCreationOptions {
  const challenge = normalizeBase64UrlField(optionsJson.challenge);
  const user = optionsJson.user as Record<string, unknown> | undefined;
  const userId = user ? normalizeBase64UrlField(user.id) : '';
  const excludeCredentials = mapBase64UrlList(
    optionsJson.excludeCredentials as Array<Record<string, unknown>> | undefined,
    'id',
  );

  const decoded: PublicKeyCredentialCreationOptions = {
    ...optionsJson,
    challenge: base64UrlToBuffer(challenge),
    user: {
      ...(user ?? {}),
      id: base64UrlToBuffer(userId),
    } as PublicKeyCredentialUserEntity,
  } as PublicKeyCredentialCreationOptions;

  if (excludeCredentials) {
    decoded.excludeCredentials = excludeCredentials;
  }

  return decoded;
}

/**
 * Convert a server-supplied PublicKeyCredentialRequestOptionsJSON object into
 * the binary form required by navigator.credentials.get().
 */
function decodeRequestOptions(
  optionsJson: Record<string, unknown>,
): PublicKeyCredentialRequestOptions {
  const challenge = normalizeBase64UrlField(optionsJson.challenge);
  const allowCredentials = mapBase64UrlList(
    optionsJson.allowCredentials as Array<Record<string, unknown>> | undefined,
    'id',
  );

  const decoded: PublicKeyCredentialRequestOptions = {
    ...optionsJson,
    challenge: base64UrlToBuffer(challenge),
  } as PublicKeyCredentialRequestOptions;

  if (allowCredentials) {
    decoded.allowCredentials = allowCredentials;
  }

  return decoded;
}

/**
 * Build a JSON-serializable registration response from a browser
 * PublicKeyCredential. Returned shape matches what py_webauthn expects.
 */
function encodeRegistrationCredential(credential: PublicKeyCredential): Record<string, unknown> {
  const response = credential.response as AuthenticatorAttestationResponse;
  const transports = typeof response.getTransports === 'function' ? response.getTransports() : [];

  return {
    id: credential.id,
    rawId: bufferToBase64Url(credential.rawId),
    response: {
      clientDataJSON: bufferToBase64Url(response.clientDataJSON),
      attestationObject: bufferToBase64Url(response.attestationObject),
      transports,
    },
    type: credential.type,
    clientExtensionResults: credential.getClientExtensionResults?.() ?? {},
    authenticatorAttachment: credential.authenticatorAttachment,
  };
}

/**
 * Build a JSON-serializable authentication assertion from a browser
 * PublicKeyCredential. Returned shape matches what py_webauthn expects.
 */
function encodeAuthenticationCredential(credential: PublicKeyCredential): Record<string, unknown> {
  const response = credential.response as AuthenticatorAssertionResponse;

  const encodedResponse: Record<string, unknown> = {
    clientDataJSON: bufferToBase64Url(response.clientDataJSON),
    authenticatorData: bufferToBase64Url(response.authenticatorData),
    signature: bufferToBase64Url(response.signature),
  };

  if (response.userHandle) {
    encodedResponse.userHandle = bufferToBase64Url(response.userHandle);
  }

  return {
    id: credential.id,
    rawId: bufferToBase64Url(credential.rawId),
    response: encodedResponse,
    type: credential.type,
    clientExtensionResults: credential.getClientExtensionResults?.() ?? {},
    authenticatorAttachment: credential.authenticatorAttachment,
  };
}

function isWebAuthnCancellationError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === 'NotAllowedError' || error.name === 'AbortError')
  );
}

function wrapWebAuthnError(error: unknown): never {
  if (isWebAuthnCancellationError(error)) {
    throw new WebAuthnCancelledError(error instanceof Error ? error.message : undefined);
  }
  throw error;
}

interface CredentialWithToJSON {
  toJSON(): Record<string, unknown>;
}

interface PublicKeyCredentialWithJSON extends PublicKeyCredential, CredentialWithToJSON {}

/**
 * Browser-provided Level 3 WebAuthn JSON helpers. We access them via a cast
 * because the DOM types shipped with TypeScript 5.6 do not yet expose them.
 */
function getPublicKeyCredentialHelpers() {
  return window.PublicKeyCredential as unknown as {
    parseCreationOptionsFromJSON?(
      options: Record<string, unknown>,
    ): PublicKeyCredentialCreationOptions;
    parseRequestOptionsFromJSON?(
      options: Record<string, unknown>,
    ): PublicKeyCredentialRequestOptions;
  };
}

/**
 * Create a new passkey credential from server-provided registration options.
 *
 * Uses the native JSON parsing/conversion helpers when available, with a
 * manual base64url fallback for older browsers.
 */
export async function createPasskeyCredential(
  optionsJson: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const helpers = getPublicKeyCredentialHelpers();
  if (typeof helpers.parseCreationOptionsFromJSON === 'function') {
    const options = helpers.parseCreationOptionsFromJSON(optionsJson);
    const credential = (await navigator.credentials
      .create({ publicKey: options })
      .catch(wrapWebAuthnError)) as PublicKeyCredentialWithJSON | null;
    if (!credential) {
      throw new WebAuthnCancelledError('No credential returned from create()');
    }
    return credential.toJSON();
  }

  const options = decodeCreationOptions(optionsJson);
  const credential = (await navigator.credentials
    .create({ publicKey: options })
    .catch(wrapWebAuthnError)) as PublicKeyCredential | null;
  if (!credential) {
    throw new WebAuthnCancelledError('No credential returned from create()');
  }
  return encodeRegistrationCredential(credential);
}

/**
 * Get an existing passkey assertion from server-provided authentication options.
 *
 * Uses native JSON helpers when available, with a manual base64url fallback for
 * older browsers.
 */
export async function getPasskeyAssertion(
  optionsJson: Record<string, unknown>,
): Promise<Record<string, unknown>> {
  const helpers = getPublicKeyCredentialHelpers();
  if (typeof helpers.parseRequestOptionsFromJSON === 'function') {
    const options = helpers.parseRequestOptionsFromJSON(optionsJson);
    const credential = (await navigator.credentials
      .get({ publicKey: options })
      .catch(wrapWebAuthnError)) as PublicKeyCredentialWithJSON | null;
    if (!credential) {
      throw new WebAuthnCancelledError('No credential returned from get()');
    }
    return credential.toJSON();
  }

  const options = decodeRequestOptions(optionsJson);
  const credential = (await navigator.credentials
    .get({ publicKey: options })
    .catch(wrapWebAuthnError)) as PublicKeyCredential | null;
  if (!credential) {
    throw new WebAuthnCancelledError('No credential returned from get()');
  }
  return encodeAuthenticationCredential(credential);
}
