/**
 * Generate a secure random credential value.
 * Uses crypto.getRandomValues() if available, falls back to Math.random().
 */
export function generateCredentialValue(length: number): string {
  const alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789';
  const randomValues = new Uint8Array(length);

  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(randomValues);
  } else {
    for (let index = 0; index < randomValues.length; index += 1) {
      randomValues[index] = Math.floor(Math.random() * alphabet.length);
    }
  }

  return Array.from(randomValues, (value) => alphabet[value % alphabet.length]).join('');
}
